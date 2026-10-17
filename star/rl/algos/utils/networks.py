import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape
def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)
def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi
def  weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        assert size in {84, 100}, f"unexpected size: {size}"
        self.size = size

    def forward(self, x):
        assert x.ndim == 4, "input must be a 4D tensor"
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        assert x.size(3) == 100, f"unexpected size: {x.size(3)}"
        if self.size == 84:
            p = 8
        return x[:, :, p:-p, p:-p]
class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.0
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape, out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        y = self.projection(x)
        return y
class SharedProjectionCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32, out_dim=256, ln=True):
        super().__init__()
        assert len(obs_shape) == 3

        layers = [
            CenterCrop(size=84),
            NormalizeImg(),
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
        ]
        for _ in range(1, num_layers):
            layers += [nn.ReLU(), nn.Conv2d(num_filters, num_filters, 3, stride=1)]

        # conv trunk  shape
        trunk = nn.Sequential(*layers)
        conv_out_shape = _get_out_shape(obs_shape, trunk)      # (C,H,W)
        conv_out_dim = int(np.prod(conv_out_shape))            # C*H*W

        # projection head
        self.layers = nn.Sequential(
            trunk,
            Flatten(),
            nn.Linear(conv_out_dim, out_dim),
            nn.LayerNorm(out_dim) if ln else nn.Identity(),
        )

        self.out_shape = out_dim
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)
class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)
class ForwardDynamics(nn.Module):
    def __init__(self, emb_shape, action_shape, hidden_dim): # SMG hidden_dim=1024
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_shape + action_shape, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_shape),
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        joint_h = torch.cat([obs, action], dim=1)
        return self.mlp(joint_h)

    def loss(self, pre_next_obs, next_obs_z):
        fwd_loss = F.mse_loss(pre_next_obs, next_obs_z)
        return fwd_loss
class ImageDecoder(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=emb_dim, out_features=14112)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x).view(-1, 32, 21, 21)
        x = self.relu(x)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(x)
        x = self.sigmoid(self.conv3(x))
        return x

    def loss(self, obs, obs_recon):
        obs_normalized = obs / 255.0
        obs_recon = obs_recon
        recon_loss = nn.L1Loss()(obs_normalized, obs_recon)
        return recon_loss
class Actor(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape),
        )
        self.mlp.apply(weight_init)

    def forward(
        self,
        x,
        compute_pi=True,
        compute_log_pi=True,
        detach=False,
        compute_attrib=False,
        encoded =False,
    ):
        if encoded:
            x=self.encoder.head_cnn(x)
            if detach:
                x=x.detach()
            x=self.encoder.projection(x)
        else:
            x = self.encoder(x, detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std
class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk =nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  #  mid_emb
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),           #  Q head
        )
        self.apply(weight_init)


    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))

    def mid_emb(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        # net  3  : (Linear, ReLU, Linear)
        for layer in list(self.trunk)[:3]:
            x = layer(x)
        return x
class Critic(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(self.encoder.out_dim, action_shape, hidden_dim)
        self.Q2 = QFunction(self.encoder.out_dim, action_shape, hidden_dim)

    def forward(self, x, action, detach=False, encoded=False):
        if encoded:
            x=self.encoder.head_cnn(x)
            if detach:
                x=x.detach()
            x=self.encoder.projection(x)
        else:
            x = self.encoder(x, detach)
        return self.Q1(x, action), self.Q2(x, action)

    def get_mid_emb(self, x, action, detach=False, return_mean=True, encoded=False):
        if encoded:
            x=self.encoder.head_cnn(x)
            if detach:
                x=x.detach()
            x=self.encoder.projection(x)
        else:
            x = self.encoder(x, detach)
        mid_emb_1 = self.Q1.mid_emb(x, action)
        mid_emb_2 = self.Q2.mid_emb(x, action)
        if return_mean:
            return (mid_emb_1 + mid_emb_2) / 2
        else:
            return mid_emb_1
