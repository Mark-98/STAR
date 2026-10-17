"""STAR Stage 1: DrQ with forward-dynamics auxiliary learning only."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from star.rl import utils
from star.rl.algos.utils.networks import SharedProjectionCNN, Encoder, Actor, RLProjection, Critic, ForwardDynamics, ImageDecoder


def _consume_legacy_decoder_rng(repr_dim):
    """Match the audited constructor's CPU RNG without retaining a decoder.

    The historical detached decoder is not part of STAR's training losses, but
    simply removing its initialization changes subsequent replay/augmentation
    randomness. The four-seed single-update study retained this consumption.
    """
    ImageDecoder(repr_dim)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device
        ).to(x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class STARStage1:
    def __init__(self, args):

        print("STAR Stage 1: forward dynamics")
        self.device = "cuda"
        self.critic_target_tau = args.critic_target_tau

        self.obs_shape = args.obs_shape
        self.action_shape = args.action_shape
        self.repr_dim = args.repr_dim
        self.hidden_dim = args.hidden_dim


        self.actor_log_std_min = args.actor_log_std_min
        self.actor_log_std_max = args.actor_log_std_max
        self.actor_update_freq = args.actor_update_freq

        self.init_temperature = args.init_temperature

        self.critic_target_update_freq = args.critic_target_update_freq
        self.random_stack = bool(args.random_stack)

        self.args = args
        if args.use_ib or args.encoder_type != "default" or args.dyna != "fwd":
            raise ValueError("STAR Stage 1 supports default encoder, use_ib=false and dyna=fwd only.")
        encoder = SharedProjectionCNN(args.obs_shape, out_dim=args.repr_dim, ln=args.ln).cuda()

        print("repr_dim", self.repr_dim)
        print("lr", args.lr)

        # models
        actor_encoder = Encoder(
            encoder,
            nn.Identity(),
            RLProjection(encoder.out_shape, args.repr_dim),
        )
        critic_encoder = Encoder(
            encoder,
            nn.Identity(),
            RLProjection(encoder.out_shape, args.repr_dim),
        )

        self.actor = Actor(
            actor_encoder,
            args.action_shape,
            self.hidden_dim,
            self.actor_log_std_min,
            self.actor_log_std_max,
        ).cuda()

        self.critic = Critic(critic_encoder, args.action_shape, self.hidden_dim).cuda()
        self.critic_target = deepcopy(self.critic)
        if self.args.dyna_alpha > 0:
            self.dyna = ForwardDynamics(encoder.out_shape, args.action_shape, self.hidden_dim).cuda()
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(args.action_shape)

        # optimizer
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr, betas=(args.actor_beta, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.lr, betas=(args.critic_beta, 0.999))
        if self.args.dyna_alpha > 0:
            self.dyna_opt = torch.optim.Adam(self.dyna.parameters(), lr=args.lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        self.train()
        _consume_legacy_decoder_rng(args.repr_dim)
        args.stage1_legacy_decoder_rng_restored = True
        args.stage1_initial_update_count = 1
        args.stage1_gradient_clipping_enabled = False

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        if self.args.dyna_alpha > 0:
            self.dyna.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode=False, decoding=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            if eval_mode:
                action = mu
            else:
                action = pi
            return action.cpu().numpy()[0]

    def update_actor_and_alpha(self, obs_z, step=None, update_alpha=True):
        metrics = dict()
        _, pi, log_pi, log_std = self.actor(obs_z, detach=True, encoded=True)
        actor_Q1, actor_Q2 = self.critic(obs_z, pi, detach=True, encoded=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        metrics['actor_loss'] = actor_loss.item()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if update_alpha:
            self.log_alpha_opt.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha_value'] = self.alpha

            alpha_loss.backward()
            self.log_alpha_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        org_obs, action, reward, discount, org_next_obs = utils.to_torch(batch, self.device)

        reward = reward.unsqueeze(1) if reward.dim() == 1 else reward
        discount = discount.unsqueeze(1) if discount.dim() == 1 else discount

        metrics['batch_reward'] = reward.mean().item()

        # augment
        obs = self.aug(org_obs.float())
        next_obs = self.aug(org_next_obs.float())
        obs_z = self.critic.encoder.shared_cnn(obs)
        metrics.update(self.update_critic_fw(obs_z, action, reward, discount, next_obs, step))

        if step % self.args.actor_update_freq == 0:
            metrics.update(self.update_actor_and_alpha(obs_z.detach(), step))


        # update critic target
        if step % self.args.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.args.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.args.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.args.encoder_tau
            )

        return metrics


    def update_critic_fw(self, obs_z, action, reward, discount, next_obs, step=None):
        metrics = dict()
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (discount * target_V)

        current_Q1, current_Q2 = self.critic(obs_z, action, encoded=True)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target_Q'] = target_Q.mean().item()
        metrics['critic_current_Q1'] = current_Q1.mean().item()
        metrics['critic_current_Q2'] = current_Q2.mean().item()

        if self.args.dyna_alpha > 0:
            pred_next = self.dyna(obs_z, action)
            target_encoder = self.critic_target.encoder if self.args.use_target_enc else self.critic.encoder
            with torch.no_grad():
                next_obs_z = target_encoder.shared_cnn(next_obs)
            dyna_loss = self.dyna.loss(pred_next, next_obs_z)
            metrics['fwd_loss'] = dyna_loss.item()
        else:
            dyna_loss = torch.zeros((), device=critic_loss.device)
        total_loss = critic_loss + dyna_loss * self.args.dyna_alpha

        self.critic_opt.zero_grad()

        if self.args.dyna_alpha > 0:
            self.dyna_opt.zero_grad(set_to_none=True)

        total_loss.backward()

        self.critic_opt.step()
        if self.args.dyna_alpha > 0:
            self.dyna_opt.step()

        return metrics
