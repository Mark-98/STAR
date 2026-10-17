"""STAR Stage 2: robust encoder and MLP feature alignment; frozen controller."""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as TF
from star.rl import utils, augmentations
from star.rl.checkpoint import load_checkpoint
from star.rl.algos.utils.networks import SharedProjectionCNN


class MLP(nn.Module):
    """
    Flexible MLP for 512-dimension image latent
    """
    def __init__(self,
                 state_dim,
                 device,
                 hidden_dim=1024,
                 num_layers=6,
                 ln=False,
                last_ln=False
                 ):

        super(MLP, self).__init__()
        self.device = device

        input_dim = state_dim

        # Construct middle layers dynamically
        layers = [nn.Linear(input_dim, hidden_dim), nn.Mish()]
        for _ in range(num_layers - 1):  # Add additional hidden layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if ln:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Mish())
        self.mid_layer = nn.Sequential(*layers)

        # Output layer
        if last_ln:
            self.final_layer = nn.Sequential(
                nn.Linear(hidden_dim, state_dim),
                nn.LayerNorm(state_dim)
            )
        else:
            self.final_layer = nn.Linear(hidden_dim, state_dim)
        print("number of MLP parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, x):
        x = self.mid_layer(x)
        return self.final_layer(x)

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

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device)
        shift = shift.to(x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RandomRotateAug(nn.Module):
    def __init__(self, degrees=180, pad=17):
        super().__init__()
        self.degrees = (-float(degrees), float(degrees)) if isinstance(degrees, (int, float)) else tuple(degrees)
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        x = F.pad(x, tuple([self.pad] * 4), mode='replicate')
        angle = torch.empty(1, device=x.device).uniform_(*self.degrees).item()
        x = TF.functional.rotate(x, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
        crop_h, crop_w = (x.size(-2) - h) // 2, (x.size(-1) - w) // 2
        return x[..., crop_h:crop_h + h, crop_w:crop_w + w]


class RandomRotateShiftAug(nn.Module):
    def __init__(self, degrees=180, pad=16):
        super().__init__()
        self.rotate = RandomRotateAug(degrees=degrees)
        self.shift = RandomShiftsAug(pad=pad)

    def forward(self, x):
        return self.shift(self.rotate(x))


class STARStage2:
    def __init__(self, args):
        if args.gen_type != 'mlp' or args.rc or args.use_ib or args.encoder_type != 'default':
            raise ValueError('STAR Stage 2 requires mlp, rc=false, use_ib=false, encoder_type=default.')
        if args.overlay_type not in ('random', 'random_rotate_shift') or args.q_act_sample or args.multi_act:
            raise ValueError('STAR supports random overlay or random_rotate_shift and a single actor prediction.')
        self.args = args
        self.device = 'cuda'
        self.obs_shape, self.action_shape = args.obs_shape, args.action_shape
        self.repr_dim, self.hidden_dim = args.repr_dim, args.hidden_dim
        self.teacher_agent = load_checkpoint(args.teacher_work_dir, map_location=self.device)['agent']
        teacher_args = self.teacher_agent.args
        expected_dynamics = getattr(args, 'expected_teacher_dyna_alpha', None)
        if expected_dynamics is not None and getattr(teacher_args, 'dyna_alpha', None) != expected_dynamics:
            raise ValueError('Teacher dynamics coefficient does not match the requested recipe.')
        for field in ('obs_shape', 'action_shape', 'repr_dim', 'task_name', 'action_repeat'):
            actual, expected = getattr(teacher_args, field), getattr(args, field)
            if actual != expected:
                raise ValueError(f'Teacher {field}={actual!r} does not match {expected!r}.')
        if teacher_args.use_ib or teacher_args.encoder_type != 'default':
            raise ValueError('Stage 2 needs a default, non-IB teacher encoder.')
        # Preserve the original initialization order and MLP state-dict layout.
        self.aug_encoder = SharedProjectionCNN(args.obs_shape, out_dim=args.repr_dim).cuda()
        if args.load_enc:
            self.aug_encoder.load_state_dict(self.teacher_agent.critic.encoder.shared_cnn.state_dict())
        self.adapter = MLP(self.aug_encoder.out_shape, device=self.device,
                           hidden_dim=args.hidden_dim, num_layers=args.gen_layer, ln=args.ad_ln).cuda()
        self.aug_encoder_opt = torch.optim.Adam(self.aug_encoder.parameters(), lr=args.lr)
        self.adapter_opt = torch.optim.Adam(self.adapter.parameters(), lr=args.lr)
        for name in ('actor', 'critic', 'critic_target', 'dyna', 'image_decoder'):
            module = getattr(self.teacher_agent, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()
                for parameter in module.parameters():
                    parameter.grad = None
        self.aug = RandomShiftsAug(pad=4)
        self.random_rotate_shift_aug = RandomRotateShiftAug(degrees=180, pad=16)
        self.train()

    def __setstate__(self, state):
        # Legacy snapshots used this name even when the model was an ordinary MLP.
        state = dict(state)
        for old, new in (('generator_diffusion', 'adapter'), ('generator_diffusion_opt', 'adapter_opt')):
            if old in state:
                state[new] = state.pop(old)
        for unused in ('discriminator_diffusion', 'cos', 'previous_emb', 'student_actor', 'student_actor_opt'):
            state.pop(unused, None)
        self.__dict__.update(state)
        if self.args.gen_type != 'mlp':
            raise ValueError('This release loads STAR MLP feature-alignment snapshots only.')
        if not hasattr(self, 'random_rotate_shift_aug'):
            self.random_rotate_shift_aug = RandomRotateShiftAug(degrees=180, pad=16)
        self.device = str(next(self.aug_encoder.parameters()).device)

    def train(self, training=True):
        self.training = training
        self.aug_encoder.train(training)
        self.adapter.train(training)

    def _should_update_aug_encoder(self):
        return not (self.args.gen_type == 'mlp' and
                    getattr(self, 'freeze_aug_encoder_mlp', getattr(self.args, 'freeze_aug_encoder_mlp', False)))

    @torch.no_grad()
    def act(self, obs, step=0, eval_mode=False, decoding=False, get_latent=False, **kwargs):
        if decoding or kwargs.get('input_control_noise', False):
            raise ValueError('Image reconstruction/control-noise experiments are outside STAR Stage 2.')
        device = next(self.aug_encoder.parameters()).device
        obs = torch.as_tensor(obs, device=device).unsqueeze(0)
        latent = self.adapter(self.aug_encoder(obs))
        actor = self.teacher_agent.actor
        mu, pi, _, _ = actor(latent, compute_log_pi=False, encoded=True)
        action = (mu if eval_mode else pi).cpu().numpy()[0]
        return (action, latent) if get_latent else action

    def update(self, replay_iter, step):
        device = next(self.aug_encoder.parameters()).device
        org_obs, _, _, _, org_next_obs, _ = utils.to_torch(next(replay_iter), device)
        obs = self.aug(org_obs.float())
        # Retain this unused draw to preserve the source augmentation RNG sequence.
        self.aug(org_next_obs.float())
        if self.args.overlay_type == 'random_rotate_shift':
            aug_obs = self.random_rotate_shift_aug(obs)
        else:
            aug_obs = augmentations.random_ratio_overlay(obs, max_alpha=self.args.max_alpha)
        with torch.no_grad():
            teacher_obs = obs if self.args.weak_aug else org_obs.float()
            teacher_latent = self.teacher_agent.critic.encoder.shared_cnn(teacher_obs)
        encoded = self.aug_encoder(aug_obs)
        if not self._should_update_aug_encoder():
            encoded = encoded.detach()
        latent = self.adapter(encoded)
        loss = F.mse_loss(latent, teacher_latent.detach())
        if getattr(self.args, 'clean_loss', False):
            clean_latent = self.adapter(self.aug_encoder(obs))
            loss = loss + self.args.clean_loss_weight * F.mse_loss(clean_latent, teacher_latent.detach())
        self.adapter_opt.zero_grad(set_to_none=True)
        if self._should_update_aug_encoder():
            self.aug_encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.adapter_opt.step()
        if self._should_update_aug_encoder():
            self.aug_encoder_opt.step()
        return {'G/recon_loss': loss.item()}
