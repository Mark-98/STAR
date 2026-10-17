from typing import Dict, Optional
import copy
import os
import pathlib
from contextlib import contextmanager

import dill
import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.stage2_alignment import Stage2AlignmentObjective
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)


def ensure_huggingface_hub_compat():
    if hasattr(huggingface_hub, 'cached_download'):
        return

    def _cached_download_compat(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = _cached_download_compat


def weight_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


_IMAGE_RANGE_NORMALIZER = get_image_range_normalizer()
for param in _IMAGE_RANGE_NORMALIZER.parameters():
    param.requires_grad_(False)


_DEFAULT_PLACES365_ROOTS = ()  # Use overlay_root or PLACES365_ROOT explicitly.


def _make_projector_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == 'mish':
        return nn.Mish()
    if name in ('silu', 'swish'):
        return nn.SiLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f'Unsupported projector activation: {name}')


class GatedProjectorActivation(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        name = str(name).lower()
        if name not in ('swiglu', 'geglu'):
            raise ValueError(f'Unsupported gated projector activation: {name}')
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        if self.name == 'swiglu':
            return x * F.silu(gate)
        return x * F.gelu(gate)


class LatentProjectorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        residual: bool = False,
        residual_scale: float = 1.0,
        zero_init_output: bool = False,
        activation: str = 'mish',
        input_layer_norm: bool = True,
        layer_norm_every_layer: bool = False,
        output_layer_norm: bool = False,
        architecture: str = 'legacy',
    ):
        super().__init__()
        assert num_layers >= 1
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        activation = str(activation).lower()
        gated_activation = activation in ('swiglu', 'geglu')
        self.activation = activation
        self.input_layer_norm = bool(input_layer_norm)
        self.layer_norm_every_layer = bool(layer_norm_every_layer)
        self.output_layer_norm = bool(output_layer_norm)
        self.architecture = str(architecture).lower()
        if self.architecture not in ('legacy', 'standard'):
            raise ValueError(
                f'projector architecture must be legacy or standard, got {architecture}'
            )

        if self.architecture == 'legacy':
            layers = [nn.Linear(input_dim, hidden_dim)]
            if self.input_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
            for _ in range(num_layers):
                if gated_activation:
                    layers.append(nn.Linear(in_dim, hidden_dim * 2))
                    if self.layer_norm_every_layer:
                        layers.append(nn.LayerNorm(hidden_dim * 2))
                    layers.append(GatedProjectorActivation(activation))
                else:
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    if self.layer_norm_every_layer:
                        layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(_make_projector_activation(activation))
                in_dim = hidden_dim
            output_layer = nn.Linear(in_dim, input_dim)
            layers.append(output_layer)
            if self.output_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
        else:
            layers = []
            in_dim = input_dim
            for layer_idx in range(num_layers - 1):
                linear_dim = hidden_dim * 2 if gated_activation else hidden_dim
                layers.append(nn.Linear(in_dim, linear_dim))
                use_layer_norm = (
                    self.input_layer_norm
                    if layer_idx == 0
                    else self.layer_norm_every_layer
                )
                if use_layer_norm:
                    layers.append(nn.LayerNorm(linear_dim))
                if gated_activation:
                    layers.append(GatedProjectorActivation(activation))
                else:
                    layers.append(_make_projector_activation(activation))
                in_dim = hidden_dim
            output_layer = nn.Linear(in_dim, input_dim)
            layers.append(output_layer)
            if self.output_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
        self.net = nn.Sequential(*layers)
        self.net.apply(weight_init)
        if zero_init_output:
            nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.residual:
            y = x + self.residual_scale * y
        return y


class SpatialFeatureAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 1,
        residual: bool = True,
        residual_scale: float = 0.03,
        zero_init_output: bool = True,
        activation: str = 'mish',
        norm: str = 'group',
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        if kernel_size not in (1, 3):
            raise ValueError(f'pre_spatial_adapter_kernel_size must be 1 or 3, got {kernel_size}')

        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, padding=padding),
        ]
        norm = str(norm).lower()
        if norm in ('none', 'identity', ''):
            pass
        elif norm == 'group':
            num_groups = min(32, hidden_channels)
            while hidden_channels % num_groups != 0:
                num_groups -= 1
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels))
        else:
            raise ValueError(f'Unsupported pre-spatial adapter norm: {norm}')
        layers.extend([
            _make_projector_activation(activation),
            nn.Conv2d(hidden_channels, channels, kernel_size=kernel_size, padding=padding),
        ])
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if zero_init_output:
            output_layer = self.net[-1]
            nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.residual:
            y = x + self.residual_scale * y
        return y


class FlattenSpatialFeatureAdapter(nn.Module):
    def __init__(
        self,
        feature_shape,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        residual: bool = True,
        residual_scale: float = 0.03,
        zero_init_output: bool = True,
        activation: str = 'mish',
        layer_norm: bool = True,
    ):
        super().__init__()
        if len(feature_shape) != 3:
            raise ValueError(f'feature_shape must be [C, H, W], got {feature_shape}')
        if num_layers < 1:
            raise ValueError(f'pre_spatial_adapter_mlp_num_layers must be >= 1, got {num_layers}')

        self.feature_shape = tuple(int(v) for v in feature_shape)
        self.input_dim = int(self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2])
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        if hidden_dim is None:
            hidden_dim = self.input_dim
        hidden_dim = int(hidden_dim)

        layers = []
        in_dim = self.input_dim
        output_layer = None
        if num_layers == 1:
            output_layer = nn.Linear(in_dim, self.input_dim)
            layers.append(output_layer)
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(_make_projector_activation(activation))
                in_dim = hidden_dim
            output_layer = nn.Linear(in_dim, self.input_dim)
            layers.append(output_layer)

        self.net = nn.Sequential(*layers)
        self.net.apply(weight_init)
        if zero_init_output:
            nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.reshape(x.shape[0], -1))
        y = y.reshape(x.shape[0], *self.feature_shape)
        if self.residual:
            y = x + self.residual_scale * y
        return y


class RandomShiftsAug(nn.Module):
    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad <= 0:
            return x
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
        ).to(x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(
            x,
            grid,
            padding_mode='zeros',
            align_corners=False,
        )


class Places365Overlay:
    def __init__(
        self,
        root: Optional[str] = None,
        dataset: str = 'places365_standard',
        image_size: int = 84,
        num_workers: int = 4,
        use_val: bool = False,
    ):
        self.root = root
        self.dataset = dataset
        self.image_size = image_size
        self.num_workers = num_workers
        self.use_val = use_val
        self.dataloader = None
        self.iterator = None
        self._warned_missing = False

    def _resolve_partition_dir(self) -> Optional[pathlib.Path]:
        partition = 'val' if self.use_val else 'train'
        candidates = []

        if self.root is not None:
            candidates.append(pathlib.Path(self.root).expanduser())
        env_root = os.environ.get('PLACES365_ROOT')
        if env_root:
            candidates.append(pathlib.Path(env_root).expanduser())
        candidates.extend(_DEFAULT_PLACES365_ROOTS)

        for root in candidates:
            direct = root / partition
            nested = root / self.dataset / partition
            if direct.exists():
                return direct
            if nested.exists():
                return nested
        return None

    def available(self) -> bool:
        return self._resolve_partition_dir() is not None

    def _ensure_loader(self, batch_size: int, device: torch.device):
        if self.dataloader is not None:
            return True

        partition_dir = self._resolve_partition_dir()
        if partition_dir is None:
            if not self._warned_missing:
                print(
                    'Places365 overlay requested but dataset was not found. '
                    'Set overlay_root or PLACES365_ROOT. '
                    'Falling back to batch overlay.'
                )
                self._warned_missing = True
            return False

        transform = tv_transforms.Compose([
            tv_transforms.RandomResizedCrop(self.image_size),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
        ])
        self.dataloader = torch.utils.data.DataLoader(
            tv_datasets.ImageFolder(str(partition_dir), transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(device.type == 'cuda'),
            drop_last=True,
        )
        self.iterator = iter(self.dataloader)
        print(f'Loaded Places365 overlay dataset from {partition_dir}')
        return True

    def get_batch(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self._ensure_loader(batch_size=batch_size, device=device):
            return None

        while True:
            try:
                imgs, _ = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                imgs, _ = next(self.iterator)

            if imgs.shape[0] < batch_size:
                self.iterator = iter(self.dataloader)
                continue
            imgs = imgs[:batch_size]
            return imgs.to(device=device, dtype=torch.float32, non_blocking=True)


def _normalize_places_overlay(imgs: torch.Tensor) -> torch.Tensor:
    # Keep overlay images on the exact same [0, 1] -> [-1, 1] mapping as dataset RGB inputs.
    normalizer = _IMAGE_RANGE_NORMALIZER.to(device=imgs.device, dtype=imgs.dtype)
    return normalizer.normalize(imgs)


def random_overlay(
    x: torch.Tensor,
    alpha: float = 0.5,
    mode: str = 'places365',
    places_overlay: Optional[Places365Overlay] = None,
) -> torch.Tensor:
    if alpha <= 0:
        return x

    if mode == 'places365' and places_overlay is not None:
        imgs = places_overlay.get_batch(batch_size=x.size(0), device=x.device)
        if imgs is not None:
            imgs = _normalize_places_overlay(imgs)
            repeat_factor = max(x.size(1) // 3, 1)
            imgs = imgs.repeat(1, repeat_factor, 1, 1)
            return ((1.0 - alpha) * x) + (alpha * imgs)

    if x.size(0) <= 1:
        return x

    overlay_source = x[torch.randperm(x.shape[0], device=x.device)]
    return ((1.0 - alpha) * x) + (alpha * overlay_source)


def random_overlay_sequence(
    x: torch.Tensor,
    alpha: float = 0.5,
    mode: str = 'places365',
    places_overlay: Optional[Places365Overlay] = None,
) -> torch.Tensor:
    """
    Apply the same overlay to every frame within a sample.
    x: (B, T, C, H, W)
    """
    if alpha <= 0:
        return x

    assert x.ndim == 5
    batch_size, n_steps = x.shape[:2]

    if mode == 'places365' and places_overlay is not None:
        imgs = places_overlay.get_batch(batch_size=batch_size, device=x.device)
        if imgs is not None:
            imgs = _normalize_places_overlay(imgs)
            overlay = imgs[:, None, ...].expand(batch_size, n_steps, *imgs.shape[1:])
            return ((1.0 - alpha) * x) + (alpha * overlay)

    if batch_size <= 1:
        return x

    overlay_source = x[torch.randperm(batch_size, device=x.device), 0]
    overlay = overlay_source[:, None, ...].expand(batch_size, n_steps, *overlay_source.shape[1:])
    return ((1.0 - alpha) * x) + (alpha * overlay)


def _random_crop_shift(
    x: torch.Tensor,
    size: int,
    n_stack: int,
    shift_max: int,
) -> torch.Tensor:
    """
    Crop each image with a linearly-shifted offset per stack position.

    x: (B * n_stack, C, H', W')  where H' > size
    returns: (n_stack, B, C, size, size)

    Items 0..B-1 belong to stack 0, items B..2B-1 to stack 1, etc.
    A random shift (dh, dw) is drawn per batch item and scaled by the
    stack index so that consecutive frames appear to move smoothly.
    """
    n = x.shape[0]
    c = x.shape[1]
    b = n // n_stack
    img_size = x.shape[-1]
    pad = (img_size - size) // 2

    h0 = torch.full((b,), pad, dtype=torch.long)
    w0 = torch.full((b,), pad, dtype=torch.long)
    dh = torch.randint(-shift_max, shift_max + 1, (b,))
    dw = torch.randint(-shift_max, shift_max + 1, (b,))

    h1 = torch.zeros(n, dtype=torch.long)
    w1 = torch.zeros(n, dtype=torch.long)
    for i in range(n_stack):
        h1[b * i:b * (i + 1)] = (h0 + dh * i).clamp(0, img_size - size)
        w1[b * i:b * (i + 1)] = (w0 + dw * i).clamp(0, img_size - size)

    cropped = torch.stack([
        x[i, :, h1[i]:h1[i] + size, w1[i]:w1[i] + size]
        for i in range(n)
    ])  # (n, C, size, size)

    return cropped.reshape(n_stack, b, c, size, size)


def random_ratio_overlay_sequence(
    x: torch.Tensor,
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    sro_shift: int = 6,
    places_overlay: Optional[Places365Overlay] = None,
    return_alpha: bool = False,
):
    """
    Random-ratio overlay with per-sample alpha and temporally-shifted places images.

    Adapted from DAD_CHAN/augmentations.py::random_ratio_overlay for the policy's
    (B, T, C, H, W) layout and the policy's [-1, 1]-normalised RGB inputs.

    x: (B, T, C, H, W)  values in [-1, 1]
    min_alpha: lower bound of the uniform alpha drawn per sample
    max_alpha: upper bound of the uniform alpha drawn per sample
    sro_shift: maximum pixel shift between consecutive time steps
    places_overlay: Places365Overlay instance; falls back to batch-shuffle if None
    """
    assert x.ndim == 5
    batch_size, n_steps, c, h, w = x.shape

    # Per-sample random blending ratio
    alpha = min_alpha + (
        torch.rand(batch_size, 1, 1, 1, 1, device=x.device)
        * (max_alpha - min_alpha)
    )

    if places_overlay is not None:
        # Load a larger image so we have room to shift across time steps
        large_size = h + 2 * n_steps * sro_shift
        imgs_base = places_overlay.get_batch(batch_size=batch_size, device=x.device)
        if imgs_base is not None:
            imgs_base = _normalize_places_overlay(imgs_base)
            # Upscale to the required large size
            imgs_large = F.interpolate(
                imgs_base, size=(large_size, large_size),
                mode='bilinear', align_corners=False,
            )
            # Repeat along the batch dim: (B*n_steps, 3, large_size, large_size)
            # Ordering: stack 0 → indices 0..B-1, stack 1 → B..2B-1, …
            imgs_rep = imgs_large.repeat(n_steps, 1, 1, 1)
            # Apply per-step shift crops → (n_steps, B, 3, h, w)
            imgs_shifted = _random_crop_shift(imgs_rep, size=h, n_stack=n_steps, shift_max=sro_shift)
            # → (B, n_steps, 3, h, w)
            overlay = imgs_shifted.permute(1, 0, 2, 3, 4)
            # Handle stacked-channel case (C != 3) by tiling channel dim
            if c != overlay.shape[2]:
                repeat_factor = max(c // overlay.shape[2], 1)
                overlay = overlay.repeat(1, 1, repeat_factor, 1, 1)
            overlay = overlay[:, :, :c, :, :]
            result = (1.0 - alpha) * x + alpha * overlay
            return (result, alpha) if return_alpha else result

    # Fallback: shuffle the batch and use other samples as overlay
    if batch_size <= 1:
        return (x, torch.zeros_like(alpha)) if return_alpha else x
    overlay_source = x[torch.randperm(batch_size, device=x.device), 0]
    overlay = overlay_source[:, None, ...].expand(batch_size, n_steps, c, h, w)
    result = (1.0 - alpha) * x + alpha * overlay
    return (result, alpha) if return_alpha else result


class DiffusionUnetHybridImageDADPolicy(DiffusionUnetHybridImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        pretrained_student_checkpoint_path: Optional[str] = None,
        teacher_checkpoint_path: Optional[str] = None,
        load_full_student_policy: bool = False,
        latent_loss_weight: float = 1.0,
        diffusion_loss_weight: float = 0.0,
        diffusion_loss_input_mode: str = 'clean',
        pair_loss_type: str = 'mse',
        pair_huber_delta: float = 1.0,
        normalize_latent_loss_features: bool = False,
        latent_loss_feature_norm_eps: float = 1.0e-6,
        lambda_pair: float = 1.0,
        lambda_mean: float = 0.0,
        lambda_coral: float = 0.0,
        coral: Optional[Dict] = None,
        projector_hidden_dim: int = 1024,
        projector_num_layers: int = 4,
        projector_residual: bool = False,
        projector_residual_scale: float = 1.0,
        projector_zero_init_output: bool = False,
        projector_activation: str = 'mish',
        projector_input_layer_norm: bool = True,
        projector_layer_norm_every_layer: bool = False,
        projector_output_layer_norm: bool = False,
        projector_architecture: str = 'legacy',
        rgb_adapter_mode: str = 'per_view',
        joint_projector_hidden_dim: Optional[int] = None,
        joint_projector_num_layers: Optional[int] = None,
        joint_projector_residual: Optional[bool] = None,
        joint_projector_residual_scale: Optional[float] = None,
        joint_projector_zero_init_output: Optional[bool] = None,
        joint_projector_activation: Optional[str] = None,
        joint_projector_input_layer_norm: Optional[bool] = None,
        joint_projector_layer_norm_every_layer: Optional[bool] = None,
        joint_projector_output_layer_norm: Optional[bool] = None,
        joint_projector_architecture: Optional[str] = None,
        use_pre_spatial_adapter: bool = False,
        pre_spatial_adapter_type: str = 'conv',
        pre_spatial_adapter_hidden_channels: Optional[int] = None,
        pre_spatial_adapter_kernel_size: int = 1,
        pre_spatial_adapter_mlp_hidden_dim: Optional[int] = None,
        pre_spatial_adapter_mlp_num_layers: int = 2,
        pre_spatial_adapter_mlp_layer_norm: bool = True,
        pre_spatial_adapter_residual: bool = True,
        pre_spatial_adapter_residual_scale: float = 0.03,
        pre_spatial_adapter_zero_init_output: bool = True,
        pre_spatial_adapter_activation: Optional[str] = None,
        pre_spatial_adapter_norm: str = 'group',
        use_fusion_projector: bool = False,
        fusion_projector_hidden_dim: Optional[int] = None,
        fusion_projector_num_layers: int = 2,
        use_residual_fusion_projector: bool = True,
        local_latent_loss_weight: float = 1.0,
        global_latent_loss_weight: float = 1.0,
        robust_latent_loss_weight: float = 1.0,
        clean_latent_loss_weight: float = 0.0,
        robust_latent_input_mode: str = 'shift_overlay',
        input_grad_regularization_weight: float = 0.0,
        input_grad_regularization_loss_weighted: bool = False,
        input_grad_regularization_auto_calibrate: bool = False,
        input_grad_regularization_target_ratio: float = 0.05,
        input_grad_regularization_calibration_batches: int = 5,
        input_grad_regularization_lambda_scale: float = 1.0,
        input_grad_regularization_log_batches: int = 5,
        robust_shift_pad: int = 4,
        overlay_mode: str = 'places365',
        overlay_root: Optional[str] = None,
        overlay_dataset: str = 'places365_standard',
        overlay_num_workers: int = 4,
        overlay_use_val_split: bool = False,
        overlay_prob: float = 1.0,
        overlay_alpha: float = 0.5,
        min_overlay_alpha: float = 0.0,
        max_overlay_alpha: float = 1.0,
        use_random_ratio_overlay: bool = False,
        overlay_shift: int = 6,
        share_overlay_across_views: bool = True,
        deterministic_latent_crop: bool = True,
        freeze_teacher: bool = True,
        freeze_student_encoder: bool = False,
        freeze_diffusion_model: bool = False,
        **kwargs,
    ):
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            crop_shape=crop_shape,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            **kwargs,
        )

        self.rgb_keys = [
            key for key, attr in shape_meta['obs'].items()
            if attr.get('type', 'low_dim') == 'rgb'
        ]
        self.shape_meta = shape_meta

        self.pretrained_student_checkpoint_path = pretrained_student_checkpoint_path
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.load_full_student_policy = load_full_student_policy
        self.latent_loss_weight = float(latent_loss_weight)
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        self.diffusion_loss_input_mode = str(diffusion_loss_input_mode).lower()
        if self.diffusion_loss_input_mode not in ('clean', 'robust', 'both'):
            raise ValueError(
                'diffusion_loss_input_mode must be one of clean, robust, or both, '
                f'got {self.diffusion_loss_input_mode}'
            )
        coral_cfg = {} if coral is None else coral
        self.stage2_alignment_objective = Stage2AlignmentObjective(
            pair_loss_type=pair_loss_type,
            pair_huber_delta=pair_huber_delta,
            lambda_pair=lambda_pair,
            lambda_mean=lambda_mean,
            lambda_coral=lambda_coral,
            coral_eps=coral_cfg.get('eps', 1.0e-5),
            coral_unbiased=coral_cfg.get('unbiased', True),
        )
        self.projector_residual = bool(projector_residual)
        self.projector_residual_scale = float(projector_residual_scale)
        self.projector_zero_init_output = bool(projector_zero_init_output)
        self.projector_activation = str(projector_activation).lower()
        self.projector_input_layer_norm = bool(projector_input_layer_norm)
        self.projector_layer_norm_every_layer = bool(projector_layer_norm_every_layer)
        self.projector_output_layer_norm = bool(projector_output_layer_norm)
        self.projector_architecture = str(projector_architecture).lower()
        self.rgb_adapter_mode = str(rgb_adapter_mode).lower()
        if self.rgb_adapter_mode not in ('per_view', 'joint'):
            raise ValueError(
                f'rgb_adapter_mode must be one of per_view or joint, got {self.rgb_adapter_mode}'
            )
        if joint_projector_hidden_dim is None:
            joint_projector_hidden_dim = projector_hidden_dim
        if joint_projector_num_layers is None:
            joint_projector_num_layers = projector_num_layers
        if joint_projector_residual is None:
            joint_projector_residual = self.projector_residual
        if joint_projector_residual_scale is None:
            joint_projector_residual_scale = self.projector_residual_scale
        if joint_projector_zero_init_output is None:
            joint_projector_zero_init_output = self.projector_zero_init_output
        if joint_projector_activation is None:
            joint_projector_activation = self.projector_activation
        if joint_projector_input_layer_norm is None:
            joint_projector_input_layer_norm = self.projector_input_layer_norm
        if joint_projector_layer_norm_every_layer is None:
            joint_projector_layer_norm_every_layer = self.projector_layer_norm_every_layer
        if joint_projector_output_layer_norm is None:
            joint_projector_output_layer_norm = self.projector_output_layer_norm
        if joint_projector_architecture is None:
            joint_projector_architecture = self.projector_architecture
        self.joint_projector_hidden_dim = int(joint_projector_hidden_dim)
        self.joint_projector_num_layers = int(joint_projector_num_layers)
        self.joint_projector_residual = bool(joint_projector_residual)
        self.joint_projector_residual_scale = float(joint_projector_residual_scale)
        self.joint_projector_zero_init_output = bool(joint_projector_zero_init_output)
        self.joint_projector_activation = str(joint_projector_activation).lower()
        self.joint_projector_input_layer_norm = bool(joint_projector_input_layer_norm)
        self.joint_projector_layer_norm_every_layer = bool(joint_projector_layer_norm_every_layer)
        self.joint_projector_output_layer_norm = bool(joint_projector_output_layer_norm)
        self.joint_projector_architecture = str(joint_projector_architecture).lower()
        self.use_pre_spatial_adapter = bool(use_pre_spatial_adapter)
        self.pre_spatial_adapter_type = str(pre_spatial_adapter_type).lower()
        self.pre_spatial_adapter_hidden_channels = (
            None if pre_spatial_adapter_hidden_channels is None
            else int(pre_spatial_adapter_hidden_channels)
        )
        self.pre_spatial_adapter_kernel_size = int(pre_spatial_adapter_kernel_size)
        self.pre_spatial_adapter_mlp_hidden_dim = (
            None if pre_spatial_adapter_mlp_hidden_dim is None
            else int(pre_spatial_adapter_mlp_hidden_dim)
        )
        self.pre_spatial_adapter_mlp_num_layers = int(pre_spatial_adapter_mlp_num_layers)
        self.pre_spatial_adapter_mlp_layer_norm = bool(pre_spatial_adapter_mlp_layer_norm)
        self.pre_spatial_adapter_residual = bool(pre_spatial_adapter_residual)
        self.pre_spatial_adapter_residual_scale = float(pre_spatial_adapter_residual_scale)
        self.pre_spatial_adapter_zero_init_output = bool(pre_spatial_adapter_zero_init_output)
        if pre_spatial_adapter_activation is None:
            pre_spatial_adapter_activation = self.projector_activation
        self.pre_spatial_adapter_activation = str(pre_spatial_adapter_activation).lower()
        self.pre_spatial_adapter_norm = str(pre_spatial_adapter_norm).lower()
        self.pre_spatial_adapter_keys = []
        self.use_fusion_projector = bool(use_fusion_projector)
        if fusion_projector_hidden_dim is None:
            fusion_projector_hidden_dim = projector_hidden_dim
        self.fusion_projector_hidden_dim = int(fusion_projector_hidden_dim)
        self.fusion_projector_num_layers = int(fusion_projector_num_layers)
        self.use_residual_fusion_projector = bool(use_residual_fusion_projector)
        self.local_latent_loss_weight = float(local_latent_loss_weight)
        self.global_latent_loss_weight = float(global_latent_loss_weight)
        self.robust_latent_loss_weight = float(robust_latent_loss_weight)
        self.clean_latent_loss_weight = float(clean_latent_loss_weight)
        self.normalize_latent_loss_features = bool(normalize_latent_loss_features)
        self.latent_loss_feature_norm_eps = float(latent_loss_feature_norm_eps)
        if self.latent_loss_feature_norm_eps <= 0.0:
            raise ValueError(
                'latent_loss_feature_norm_eps must be positive, '
                f'got {self.latent_loss_feature_norm_eps}'
            )
        self.robust_latent_input_mode = str(robust_latent_input_mode).lower()
        if self.robust_latent_input_mode not in ('shift_overlay', 'shift_only', 'clean'):
            raise ValueError(
                'robust_latent_input_mode must be one of shift_overlay, shift_only, or clean, '
                f'got {self.robust_latent_input_mode}'
            )
        self.input_grad_regularization_weight = float(input_grad_regularization_weight)
        self.input_grad_regularization_loss_weighted = bool(input_grad_regularization_loss_weighted)
        self.input_grad_regularization_auto_calibrate = bool(input_grad_regularization_auto_calibrate)
        self.input_grad_regularization_target_ratio = float(input_grad_regularization_target_ratio)
        self.input_grad_regularization_calibration_batches = int(input_grad_regularization_calibration_batches)
        self.input_grad_regularization_lambda_scale = float(input_grad_regularization_lambda_scale)
        self.input_grad_regularization_log_batches = int(input_grad_regularization_log_batches)
        if self.input_grad_regularization_weight < 0:
            raise ValueError(
                'input_grad_regularization_weight must be non-negative, '
                f'got {self.input_grad_regularization_weight}'
            )
        if self.input_grad_regularization_target_ratio < 0:
            raise ValueError(
                'input_grad_regularization_target_ratio must be non-negative, '
                f'got {self.input_grad_regularization_target_ratio}'
            )
        if self.input_grad_regularization_lambda_scale < 0:
            raise ValueError(
                'input_grad_regularization_lambda_scale must be non-negative, '
                f'got {self.input_grad_regularization_lambda_scale}'
            )
        self.input_grad_regularization_calibration_batches = max(
            self.input_grad_regularization_calibration_batches,
            1,
        )
        self.input_grad_regularization_log_batches = max(
            self.input_grad_regularization_log_batches,
            0,
        )
        self._input_grad_reg_eps = 1e-12
        self._input_grad_reg_num_logged = 0
        self._input_grad_reg_calibration_count = 0
        self._input_grad_reg_distill_sum = 0.0
        self._input_grad_reg_penalty_sum = 0.0
        self._input_grad_reg_calibrated_lambda = None
        if self.robust_latent_loss_weight <= 0 and self.clean_latent_loss_weight <= 0:
            raise ValueError(
                'At least one of robust_latent_loss_weight or clean_latent_loss_weight must be positive.'
            )
        self.overlay_mode = str(overlay_mode)
        self.overlay_root = overlay_root
        self.overlay_dataset = str(overlay_dataset)
        self.overlay_num_workers = int(overlay_num_workers)
        self.overlay_use_val_split = bool(overlay_use_val_split)
        self.overlay_prob = float(overlay_prob)
        self.overlay_alpha = float(overlay_alpha)
        self.min_overlay_alpha = float(min_overlay_alpha)
        self.max_overlay_alpha = float(max_overlay_alpha)
        self.use_random_ratio_overlay = bool(use_random_ratio_overlay)
        self.overlay_shift = int(overlay_shift)
        self.share_overlay_across_views = bool(share_overlay_across_views)
        self.deterministic_latent_crop = bool(deterministic_latent_crop)
        if not freeze_teacher:
            raise ValueError('Stage-2 teacher encoder must remain frozen.')
        if not 0.0 <= self.overlay_prob <= 1.0:
            raise ValueError(f'overlay_prob must be in [0, 1], got {self.overlay_prob}')
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError(f'overlay_alpha must be in [0, 1], got {self.overlay_alpha}')
        if not 0.0 <= self.min_overlay_alpha <= self.max_overlay_alpha <= 1.0:
            raise ValueError(
                'Random overlay alpha bounds must satisfy '
                f'0 <= min <= max <= 1, got {self.min_overlay_alpha}, '
                f'{self.max_overlay_alpha}'
            )

        self.teacher_obs_encoder = copy.deepcopy(self.obs_encoder)
        self.rgb_feature_dims, self.rgb_feature_dim, self.low_dim_feature_dim = self._infer_obs_feature_dims()
        self.student_encoder_mlps = nn.ModuleDict()
        if self.rgb_adapter_mode == 'per_view':
            self.student_encoder_mlps = nn.ModuleDict({
                key: LatentProjectorMLP(
                    input_dim=feature_dim,
                    hidden_dim=projector_hidden_dim,
                    num_layers=projector_num_layers,
                    residual=self.projector_residual,
                    residual_scale=self.projector_residual_scale,
                    zero_init_output=self.projector_zero_init_output,
                    activation=self.projector_activation,
                    input_layer_norm=self.projector_input_layer_norm,
                    layer_norm_every_layer=self.projector_layer_norm_every_layer,
                    output_layer_norm=self.projector_output_layer_norm,
                    architecture=self.projector_architecture,
                )
                for key, feature_dim in self.rgb_feature_dims.items()
            })
        self.student_joint_rgb_mlp = None
        if self.rgb_adapter_mode == 'joint':
            if self.rgb_feature_dim <= 0:
                raise RuntimeError('rgb_adapter_mode=joint requires at least one rgb observation key.')
            self.student_joint_rgb_mlp = LatentProjectorMLP(
                input_dim=self.rgb_feature_dim,
                hidden_dim=self.joint_projector_hidden_dim,
                num_layers=self.joint_projector_num_layers,
                residual=self.joint_projector_residual,
                residual_scale=self.joint_projector_residual_scale,
                zero_init_output=self.joint_projector_zero_init_output,
                activation=self.joint_projector_activation,
                input_layer_norm=self.joint_projector_input_layer_norm,
                layer_norm_every_layer=self.joint_projector_layer_norm_every_layer,
                output_layer_norm=self.joint_projector_output_layer_norm,
                architecture=self.joint_projector_architecture,
            )
        self.student_fusion_mlp = None
        if self.use_fusion_projector and self.rgb_feature_dim > 0:
            self.student_fusion_mlp = LatentProjectorMLP(
                input_dim=self.rgb_feature_dim,
                hidden_dim=self.fusion_projector_hidden_dim,
                num_layers=self.fusion_projector_num_layers,
                residual=self.projector_residual,
                residual_scale=self.projector_residual_scale,
                zero_init_output=self.projector_zero_init_output,
                activation=self.projector_activation,
                input_layer_norm=self.projector_input_layer_norm,
                layer_norm_every_layer=self.projector_layer_norm_every_layer,
                output_layer_norm=self.projector_output_layer_norm,
                architecture=self.projector_architecture,
            )
        self.robust_shift_aug = RandomShiftsAug(robust_shift_pad)
        self.places_overlay = Places365Overlay(
            root=overlay_root,
            dataset=overlay_dataset,
            image_size=shape_meta['obs'][self.rgb_keys[0]]['shape'][-1] if len(self.rgb_keys) > 0 else 84,
            num_workers=overlay_num_workers,
            use_val=overlay_use_val_split,
        )
        self._last_loss_components = dict()
        self._last_augmentation_metrics = dict()

        if pretrained_student_checkpoint_path is not None:
            self._load_student_from_checkpoint(
                pretrained_student_checkpoint_path,
                load_full_policy=load_full_student_policy,
            )

        teacher_path = teacher_checkpoint_path or pretrained_student_checkpoint_path
        if teacher_path is not None:
            self._load_teacher_from_checkpoint(teacher_path)
        else:
            self.teacher_obs_encoder.load_state_dict(
                copy.deepcopy(self.obs_encoder.state_dict())
            )

        if self.use_pre_spatial_adapter:
            self._install_pre_spatial_adapters()

        self.teacher_obs_encoder.eval()
        for param in self.teacher_obs_encoder.parameters():
            param.requires_grad = False

        if freeze_student_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False
            self._set_pre_spatial_adapters_requires_grad(True)

        if freeze_diffusion_model:
            for param in self.model.parameters():
                param.requires_grad = False

        print(
            "Student encoder MLP params: %e"
            % sum(p.numel() for p in self.student_encoder_mlps.parameters())
        )
        if self.student_joint_rgb_mlp is not None:
            print(
                "Student joint RGB MLP params: %e"
                % sum(p.numel() for p in self.student_joint_rgb_mlp.parameters())
            )
        if self.student_fusion_mlp is not None:
            print(
                "Student fusion MLP params: %e"
                % sum(p.numel() for p in self.student_fusion_mlp.parameters())
            )
        if self.use_pre_spatial_adapter:
            print(
                "Student pre-spatial adapter params: %e"
                % sum(
                    p.numel()
                    for adapter in self._iter_pre_spatial_adapters()
                    for p in adapter.parameters()
                )
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'teacher_obs_encoder'):
            self.teacher_obs_encoder.eval()
        return self

    def get_stage2_optimizer_param_groups(
        self,
        base_lr: float,
        student_encoder_lr: Optional[float] = None,
        adapter_lr: Optional[float] = None,
    ):
        adapter_modules = [self.student_encoder_mlps]
        for module in (self.student_joint_rgb_mlp, self.student_fusion_mlp):
            if module is not None:
                adapter_modules.append(module)
        adapter_modules.extend(self._iter_pre_spatial_adapters())

        adapter_param_ids = {
            id(param)
            for module in adapter_modules
            for param in module.parameters()
            if param.requires_grad
        }
        student_encoder_params = [
            param
            for param in self.obs_encoder.parameters()
            if param.requires_grad and id(param) not in adapter_param_ids
        ]
        student_encoder_param_ids = {id(param) for param in student_encoder_params}
        adapter_params = [
            param
            for module in adapter_modules
            for param in module.parameters()
            if param.requires_grad
        ]
        other_params = [
            param
            for param in self.parameters()
            if param.requires_grad
            and id(param) not in adapter_param_ids
            and id(param) not in student_encoder_param_ids
        ]

        groups = []
        if student_encoder_params:
            groups.append({
                'params': student_encoder_params,
                'lr': base_lr if student_encoder_lr is None else student_encoder_lr,
                'name': 'student_encoder',
            })
        if adapter_params:
            groups.append({
                'params': adapter_params,
                'lr': base_lr if adapter_lr is None else adapter_lr,
                'name': 'adapter',
            })
        if other_params:
            groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other',
            })

        grouped_param_ids = [id(param) for group in groups for param in group['params']]
        if len(grouped_param_ids) != len(set(grouped_param_ids)):
            raise RuntimeError('A trainable parameter was assigned to multiple optimizer groups.')
        expected_param_ids = {id(param) for param in self.parameters() if param.requires_grad}
        if set(grouped_param_ids) != expected_param_ids:
            raise RuntimeError('Stage-2 optimizer groups do not cover all trainable parameters.')
        return groups

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> pathlib.Path:
        path = pathlib.Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = pathlib.Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path

    def _load_checkpoint_payload(self, checkpoint_path: str):
        ensure_huggingface_hub_compat()
        path = self._resolve_checkpoint_path(checkpoint_path)
        return torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')

    def _extract_model_state_dict(self, payload: dict) -> dict:
        if 'state_dicts' in payload and 'model' in payload['state_dicts']:
            return payload['state_dicts']['model']
        raise KeyError('Could not find model state_dict in checkpoint payload.')

    def _load_matching_state_dict(self, module: nn.Module, source_state: dict, prefix: str = '') -> int:
        target_state = module.state_dict()
        filtered_state = dict()
        for key, value in source_state.items():
            target_key = key
            if prefix:
                if not key.startswith(prefix):
                    continue
                target_key = key[len(prefix):]
            if target_key in target_state and target_state[target_key].shape == value.shape:
                filtered_state[target_key] = value
        module.load_state_dict(filtered_state, strict=False)
        return len(filtered_state)

    def _load_student_from_checkpoint(self, checkpoint_path: str, load_full_policy: bool):
        payload = self._load_checkpoint_payload(checkpoint_path)
        model_state = self._extract_model_state_dict(payload)

        if load_full_policy:
            current_state = self.state_dict()
            filtered_state = dict()
            for key, value in model_state.items():
                if (
                    key.startswith('teacher_obs_encoder.')
                    or key.startswith('student_encoder_mlp.')
                    or key.startswith('student_encoder_mlps.')
                    or key.startswith('student_joint_rgb_mlp.')
                    or key.startswith('student_fusion_mlp.')
                    or key.startswith('latent_projector.')
                ):
                    continue
                if key in current_state and current_state[key].shape == value.shape:
                    filtered_state[key] = value
            self.load_state_dict(filtered_state, strict=False)
            print(f'Loaded {len(filtered_state)} student policy tensors from {checkpoint_path}')
        else:
            loaded = self._load_matching_state_dict(
                self.obs_encoder,
                model_state,
                prefix='obs_encoder.',
            )
            print(f'Loaded {loaded} student encoder tensors from {checkpoint_path}')

    def _load_teacher_from_checkpoint(self, checkpoint_path: str):
        payload = self._load_checkpoint_payload(checkpoint_path)
        model_state = self._extract_model_state_dict(payload)
        loaded = self._load_matching_state_dict(
            self.teacher_obs_encoder,
            model_state,
            prefix='obs_encoder.',
        )
        print(f'Loaded {loaded} teacher encoder tensors from {checkpoint_path}')

    def _install_pre_spatial_adapters(self):
        self.pre_spatial_adapter_keys = []
        for key in self.rgb_keys:
            if key not in self.obs_encoder.obs_nets:
                continue
            visual_core = self.obs_encoder.obs_nets[key]
            if not hasattr(visual_core, 'backbone') or not hasattr(visual_core, 'pool'):
                raise RuntimeError(
                    f'pre-spatial adapter requires a VisualCore-style encoder for {key}.'
                )
            if visual_core.pool is None:
                raise RuntimeError(
                    f'pre-spatial adapter requires an image pool module for {key}.'
                )
            if not isinstance(visual_core.nets, nn.Sequential):
                raise RuntimeError(
                    f'pre-spatial adapter expected nn.Sequential visual_core.nets for {key}.'
                )

            modules = list(visual_core.nets.children())
            backbone_idx = next(
                (idx for idx, module in enumerate(modules) if module is visual_core.backbone),
                None,
            )
            pool_idx = next(
                (idx for idx, module in enumerate(modules) if module is visual_core.pool),
                None,
            )
            if backbone_idx is None or pool_idx is None or pool_idx <= backbone_idx:
                raise RuntimeError(
                    f'Could not locate backbone -> pool ordering for rgb key {key}.'
                )

            feat_shape = visual_core.backbone.output_shape(visual_core.input_shape)
            if self.pre_spatial_adapter_type == 'conv':
                channels = int(feat_shape[0])
                adapter = SpatialFeatureAdapter(
                    channels=channels,
                    hidden_channels=self.pre_spatial_adapter_hidden_channels,
                    kernel_size=self.pre_spatial_adapter_kernel_size,
                    residual=self.pre_spatial_adapter_residual,
                    residual_scale=self.pre_spatial_adapter_residual_scale,
                    zero_init_output=self.pre_spatial_adapter_zero_init_output,
                    activation=self.pre_spatial_adapter_activation,
                    norm=self.pre_spatial_adapter_norm,
                )
            elif self.pre_spatial_adapter_type in ('flatten_mlp', 'flat_mlp', 'mlp'):
                adapter = FlattenSpatialFeatureAdapter(
                    feature_shape=feat_shape,
                    hidden_dim=self.pre_spatial_adapter_mlp_hidden_dim,
                    num_layers=self.pre_spatial_adapter_mlp_num_layers,
                    residual=self.pre_spatial_adapter_residual,
                    residual_scale=self.pre_spatial_adapter_residual_scale,
                    zero_init_output=self.pre_spatial_adapter_zero_init_output,
                    activation=self.pre_spatial_adapter_activation,
                    layer_norm=self.pre_spatial_adapter_mlp_layer_norm,
                )
            else:
                raise ValueError(
                    f'Unsupported pre_spatial_adapter_type: {self.pre_spatial_adapter_type}'
                )
            visual_core.nets = nn.Sequential(
                *modules[:backbone_idx + 1],
                adapter,
                *modules[backbone_idx + 1:],
            )
            self.pre_spatial_adapter_keys.append(key)

        if len(self.pre_spatial_adapter_keys) == 0:
            raise RuntimeError('use_pre_spatial_adapter=True but no rgb encoders were found.')

    def _iter_pre_spatial_adapters(self):
        for key in getattr(self, 'pre_spatial_adapter_keys', []):
            if key not in self.obs_encoder.obs_nets:
                continue
            for module in self.obs_encoder.obs_nets[key].modules():
                if isinstance(module, (SpatialFeatureAdapter, FlattenSpatialFeatureAdapter)):
                    yield module

    def _set_pre_spatial_adapters_requires_grad(self, enabled: bool):
        for adapter in self._iter_pre_spatial_adapters():
            for param in adapter.parameters():
                param.requires_grad = enabled

    @contextmanager
    def _module_eval_mode(self, module: nn.Module, enabled: bool):
        if not enabled:
            yield
            return
        previous_training = module.training
        module.eval()
        try:
            yield
        finally:
            module.train(previous_training)

    def _augment_rgb_tensor(self, x: torch.Tensor) -> torch.Tensor:
        y = self.robust_shift_aug(x)
        return y

    def _overlay_enabled(self) -> bool:
        alpha = self.max_overlay_alpha if self.use_random_ratio_overlay else self.overlay_alpha
        return self.overlay_prob > 0.0 and alpha > 0.0

    def _sample_overlay_alpha(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (batch_size, 1, 1, 1, 1)
        if not self.use_random_ratio_overlay:
            return torch.full(shape, self.overlay_alpha, device=device, dtype=dtype)
        return self.min_overlay_alpha + (
            torch.rand(shape, device=device, dtype=dtype)
            * (self.max_overlay_alpha - self.min_overlay_alpha)
        )

    def _build_places_overlay_sequence(
        self,
        batch_size: int,
        n_steps: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        imgs_base = self.places_overlay.get_batch(batch_size=batch_size, device=device)
        if imgs_base is None:
            return None
        imgs_base = _normalize_places_overlay(imgs_base)

        if self.use_random_ratio_overlay:
            large_size = max(height, width) + 2 * n_steps * self.overlay_shift
            imgs_large = F.interpolate(
                imgs_base,
                size=(large_size, large_size),
                mode='bilinear',
                align_corners=False,
            )
            imgs_rep = imgs_large.repeat(n_steps, 1, 1, 1)
            imgs_shifted = _random_crop_shift(
                imgs_rep,
                size=max(height, width),
                n_stack=n_steps,
                shift_max=self.overlay_shift,
            )
            overlay = imgs_shifted.permute(1, 0, 2, 3, 4)
        else:
            if imgs_base.shape[-2:] != (height, width):
                imgs_base = F.interpolate(
                    imgs_base,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                )
            overlay = imgs_base[:, None, ...].expand(batch_size, n_steps, *imgs_base.shape[1:])

        if overlay.shape[-2:] != (height, width):
            flat = overlay.reshape(-1, overlay.shape[2], overlay.shape[3], overlay.shape[4])
            flat = F.interpolate(
                flat,
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            )
            overlay = flat.reshape(batch_size, n_steps, flat.shape[1], height, width)
        return overlay

    def _build_batch_overlay_sequence(self, reference: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps = reference.shape[:2]
        overlay_source = reference[torch.randperm(batch_size, device=reference.device), 0]
        return overlay_source[:, None, ...].expand(
            batch_size, n_steps, *overlay_source.shape[1:]
        )

    def _match_overlay_to_target(
        self,
        overlay: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_steps, target_channels, height, width = target.shape
        overlay_channels = overlay.shape[2]

        if overlay_channels != target_channels:
            repeat_factor = max(
                (target_channels + overlay_channels - 1) // overlay_channels,
                1,
            )
            overlay = overlay.repeat(1, 1, repeat_factor, 1, 1)
            overlay = overlay[:, :, :target_channels, :, :]

        if overlay.shape[-2:] != (height, width):
            flat = overlay.reshape(-1, overlay.shape[2], overlay.shape[3], overlay.shape[4])
            flat = F.interpolate(
                flat,
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            )
            overlay = flat.reshape(batch_size, n_steps, flat.shape[1], height, width)

        return overlay.to(dtype=target.dtype)

    def _make_robust_obs(self, nobs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        robust_nobs = dict()
        shifted_rgb = dict()
        self._last_augmentation_metrics = dict()
        for key, value in nobs.items():
            if key not in self.rgb_keys:
                robust_nobs[key] = value
                continue

            assert value.ndim == 5
            batch_size, n_steps = value.shape[:2]
            flat = value.reshape(-1, *value.shape[-3:])
            shifted = self._augment_rgb_tensor(flat).reshape(
                batch_size, n_steps, *value.shape[-3:]
            )
            shifted_rgb[key] = shifted

        if len(shifted_rgb) == 0:
            return robust_nobs

        reference_key = next(key for key in self.rgb_keys if key in shifted_rgb)
        reference = shifted_rgb[reference_key]
        batch_size, n_steps = reference.shape[:2]

        shared_overlay = None
        shared_overlay_mask = None
        shared_alpha = None
        effective_alphas = []
        use_shared_overlay = (
            self.share_overlay_across_views
            and self._overlay_enabled()
        )

        if use_shared_overlay:
            height, width = reference.shape[-2:]
            if self.overlay_mode == 'places365':
                shared_overlay = self._build_places_overlay_sequence(
                    batch_size=batch_size,
                    n_steps=n_steps,
                    height=height,
                    width=width,
                    device=reference.device,
                )
            if shared_overlay is None:
                shared_overlay = self._build_batch_overlay_sequence(reference)

            shared_overlay_mask = (
                torch.rand((batch_size, 1, 1, 1, 1), device=reference.device) < self.overlay_prob
            ).to(dtype=reference.dtype)
            shared_alpha = self._sample_overlay_alpha(
                batch_size=batch_size,
                device=reference.device,
                dtype=reference.dtype,
            )
            effective_alphas.append(shared_alpha * shared_overlay_mask)

        for key, shifted in shifted_rgb.items():
            if use_shared_overlay:
                overlay = self._match_overlay_to_target(shared_overlay, shifted)
                mixed = ((1.0 - shared_alpha) * shifted) + (shared_alpha * overlay)
                shifted = mixed * shared_overlay_mask + shifted * (1.0 - shared_overlay_mask)
            elif self._overlay_enabled():
                if self.use_random_ratio_overlay:
                    mixed, alpha = random_ratio_overlay_sequence(
                        shifted,
                        min_alpha=self.min_overlay_alpha,
                        max_alpha=self.max_overlay_alpha,
                        sro_shift=self.overlay_shift,
                        places_overlay=self.places_overlay,
                        return_alpha=True,
                    )
                else:
                    alpha = self._sample_overlay_alpha(
                        batch_size=batch_size,
                        device=shifted.device,
                        dtype=shifted.dtype,
                    )
                    mixed = random_overlay_sequence(
                        shifted,
                        alpha=self.overlay_alpha,
                        mode=self.overlay_mode,
                        places_overlay=self.places_overlay,
                    )
                overlay_mask = (
                    torch.rand((batch_size, 1, 1, 1, 1), device=shifted.device) < self.overlay_prob
                ).to(dtype=shifted.dtype)
                shifted = mixed * overlay_mask + shifted * (1.0 - overlay_mask)
                effective_alphas.append(alpha * overlay_mask)
            robust_nobs[key] = shifted

        pixel_delta_mse = torch.stack([
            (robust_nobs[key].float() - nobs[key].float()).pow(2).mean()
            for key in shifted_rgb
        ]).mean()
        if len(effective_alphas) > 0:
            effective_alpha = torch.cat([
                alpha.reshape(alpha.shape[0], -1)
                for alpha in effective_alphas
            ], dim=1)
        else:
            effective_alpha = torch.zeros(
                (batch_size, 1),
                device=reference.device,
                dtype=reference.dtype,
            )
        self._last_augmentation_metrics = {
            'augmentation/overlay_applied_fraction': (effective_alpha > 0).float().mean(),
            'augmentation/overlay_alpha_mean': effective_alpha.float().mean(),
            'augmentation/overlay_alpha_max': effective_alpha.float().amax(),
            'augmentation/robust_rgb_delta_rms': pixel_delta_mse.sqrt(),
        }
        return robust_nobs

    def _make_shift_only_obs(self, nobs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        shifted_nobs = dict()
        for key, value in nobs.items():
            if key not in self.rgb_keys:
                shifted_nobs[key] = value
                continue

            assert value.ndim == 5
            batch_size, n_steps = value.shape[:2]
            flat = value.reshape(-1, *value.shape[-3:])
            shifted_nobs[key] = self._augment_rgb_tensor(flat).reshape(
                batch_size, n_steps, *value.shape[-3:]
            )
        rgb_deltas = [
            (shifted_nobs[key].float() - nobs[key].float()).pow(2).mean()
            for key in self.rgb_keys
            if key in shifted_nobs
        ]
        if len(rgb_deltas) > 0:
            zero = rgb_deltas[0] * 0.0
            self._last_augmentation_metrics = {
                'augmentation/overlay_applied_fraction': zero,
                'augmentation/overlay_alpha_mean': zero,
                'augmentation/overlay_alpha_max': zero,
                'augmentation/robust_rgb_delta_rms': torch.stack(rgb_deltas).mean().sqrt(),
            }
        return shifted_nobs

    def _make_robust_latent_input_obs(self, nobs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._last_augmentation_metrics = dict()
        if self.robust_latent_input_mode == 'shift_overlay':
            return self._make_robust_obs(nobs)
        if self.robust_latent_input_mode == 'shift_only':
            return self._make_shift_only_obs(nobs)
        rgb_reference = next((nobs[key] for key in self.rgb_keys if key in nobs), None)
        if rgb_reference is not None:
            zero = rgb_reference.sum() * 0.0
            self._last_augmentation_metrics = {
                'augmentation/overlay_applied_fraction': zero,
                'augmentation/overlay_alpha_mean': zero,
                'augmentation/overlay_alpha_max': zero,
                'augmentation/robust_rgb_delta_rms': zero,
            }
        return nobs

    def _slice_and_flatten_obs(self, nobs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )

    def _flatten_all_obs(self, nobs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return dict_apply(
            nobs,
            lambda x: x.reshape(-1, *x.shape[2:]),
        )

    def _encode_obs_features(
        self,
        encoder: nn.Module,
        nobs: Dict[str, torch.Tensor],
        deterministic_crop: bool,
        use_obs_steps_only: bool = True,
    ) -> torch.Tensor:
        if use_obs_steps_only:
            this_nobs = self._slice_and_flatten_obs(nobs)
        else:
            this_nobs = self._flatten_all_obs(nobs)
        with self._module_eval_mode(encoder, deterministic_crop):
            return encoder(this_nobs)

    def _infer_obs_feature_dims(self):
        example_nobs = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            example_nobs[key] = torch.zeros(
                (1,) + shape,
                dtype=self.dtype,
                device=self.device,
            )
        rgb_feature_dict, low_dim_feature = self._split_encoder_features_by_key(
            self.obs_encoder,
            example_nobs,
        )
        rgb_feature_dims = {
            key: int(value.shape[-1])
            for key, value in rgb_feature_dict.items()
        }
        rgb_feature_dim = int(sum(rgb_feature_dims.values()))
        low_dim_feature_dim = 0 if low_dim_feature is None else low_dim_feature.shape[-1]
        return rgb_feature_dims, rgb_feature_dim, low_dim_feature_dim

    def _sample_shared_random_crop_index_overrides(
        self,
        encoder: nn.Module,
        nobs: Dict[str, torch.Tensor],
        use_obs_steps_only: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if use_obs_steps_only:
            this_nobs = self._slice_and_flatten_obs(nobs)
        else:
            this_nobs = self._flatten_all_obs(nobs)

        crop_index_overrides = dict()
        for key in encoder.obs_shapes:
            randomizer = encoder.obs_randomizers[key]
            if randomizer is None or not hasattr(randomizer, 'crop_height'):
                continue
            crop_index_overrides[key] = dmvc.sample_random_crop_indices(
                images=this_nobs[key],
                crop_height=randomizer.crop_height,
                crop_width=randomizer.crop_width,
                num_crops=randomizer.num_crops,
            )
        return crop_index_overrides

    def _apply_obs_crop_index_overrides(
        self,
        encoder: nn.Module,
        obs_dict: Dict[str, torch.Tensor],
        crop_index_overrides: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cropped_obs = dict()
        for key in encoder.obs_shapes:
            x = obs_dict[key]
            crop_indices = crop_index_overrides.get(key)
            if crop_indices is None:
                cropped_obs[key] = x
                continue

            randomizer = encoder.obs_randomizers[key]
            if randomizer is None or not hasattr(randomizer, 'crop_height'):
                cropped_obs[key] = x
                continue

            if hasattr(randomizer, 'forward_in_with_crop_indices'):
                cropped_obs[key] = randomizer.forward_in_with_crop_indices(x, crop_indices)
            else:
                cropped_obs[key] = dmvc.take_image_crops(
                    images=x,
                    crop_indices=crop_indices,
                    crop_height=randomizer.crop_height,
                    crop_width=randomizer.crop_width,
                    pos_enc=randomizer.pos_enc,
                )
        return cropped_obs

    def _split_encoder_features_by_key(
        self,
        encoder: nn.Module,
        obs_dict: Dict[str, torch.Tensor],
        skip_randomizer_forward_in: bool = False,
    ):
        rgb_features = dict()
        low_dim_features = []

        for key in encoder.obs_shapes:
            x = obs_dict[key]
            if (not skip_randomizer_forward_in) and encoder.obs_randomizers[key] is not None:
                x = encoder.obs_randomizers[key].forward_in(x)
            if encoder.obs_nets[key] is not None:
                x = encoder.obs_nets[key](x)
                if encoder.activation is not None:
                    x = encoder.activation(x)
            if encoder.obs_randomizers[key] is not None:
                x = encoder.obs_randomizers[key].forward_out(x)
            x = x.reshape(x.shape[0], -1)
            if key in self.rgb_keys:
                rgb_features[key] = x
            else:
                low_dim_features.append(x)

        low_dim_result = None
        if len(low_dim_features) > 0:
            low_dim_result = torch.cat(low_dim_features, dim=-1)

        return rgb_features, low_dim_result

    def _concat_rgb_feature_dict(self, rgb_feature_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        ordered_features = [
            rgb_feature_dict[key]
            for key in self.rgb_keys
            if key in rgb_feature_dict
        ]
        if len(ordered_features) == 0:
            return None
        return torch.cat(ordered_features, dim=-1)

    def _split_concat_rgb_feature(
        self,
        rgb_feature: torch.Tensor,
        template_rgb_feature_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        split_features = dict()
        start = 0
        for key in self.rgb_keys:
            if template_rgb_feature_dict is not None and key not in template_rgb_feature_dict:
                continue
            feature_dim = self.rgb_feature_dims.get(key)
            if feature_dim is None:
                continue
            end = start + feature_dim
            split_features[key] = rgb_feature[..., start:end]
            start = end
        if start != rgb_feature.shape[-1]:
            raise RuntimeError(
                f'Could not split RGB feature of dim {rgb_feature.shape[-1]}; consumed {start}.'
            )
        return split_features

    def _split_encoder_features(
        self,
        encoder: nn.Module,
        obs_dict: Dict[str, torch.Tensor],
        skip_randomizer_forward_in: bool = False,
    ):
        rgb_feature_dict, low_dim_result = self._split_encoder_features_by_key(
            encoder,
            obs_dict,
            skip_randomizer_forward_in=skip_randomizer_forward_in,
        )
        rgb_result = self._concat_rgb_feature_dict(rgb_feature_dict)
        return rgb_result, low_dim_result

    def _encode_obs_split_features(
        self,
        encoder: nn.Module,
        nobs: Dict[str, torch.Tensor],
        deterministic_crop: bool,
        use_obs_steps_only: bool = True,
        crop_index_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if use_obs_steps_only:
            this_nobs = self._slice_and_flatten_obs(nobs)
        else:
            this_nobs = self._flatten_all_obs(nobs)
        skip_randomizer_forward_in = False
        if crop_index_overrides is not None:
            this_nobs = self._apply_obs_crop_index_overrides(
                encoder,
                this_nobs,
                crop_index_overrides,
            )
            skip_randomizer_forward_in = True
        with self._module_eval_mode(encoder, deterministic_crop):
            return self._split_encoder_features(
                encoder,
                this_nobs,
                skip_randomizer_forward_in=skip_randomizer_forward_in,
            )

    def _encode_obs_split_feature_dicts(
        self,
        encoder: nn.Module,
        nobs: Dict[str, torch.Tensor],
        deterministic_crop: bool,
        use_obs_steps_only: bool = True,
        crop_index_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if use_obs_steps_only:
            this_nobs = self._slice_and_flatten_obs(nobs)
        else:
            this_nobs = self._flatten_all_obs(nobs)
        skip_randomizer_forward_in = False
        if crop_index_overrides is not None:
            this_nobs = self._apply_obs_crop_index_overrides(
                encoder,
                this_nobs,
                crop_index_overrides,
            )
            skip_randomizer_forward_in = True
        with self._module_eval_mode(encoder, deterministic_crop):
            return self._split_encoder_features_by_key(
                encoder,
                this_nobs,
                skip_randomizer_forward_in=skip_randomizer_forward_in,
            )

    def _project_student_rgb_feature_dict(
        self,
        rgb_feature_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.rgb_adapter_mode == 'joint':
            rgb_feature = self._concat_rgb_feature_dict(rgb_feature_dict)
            if rgb_feature is None:
                raise RuntimeError('Joint RGB adapter requires at least one rgb observation key.')
            if self.student_joint_rgb_mlp is None:
                raise RuntimeError('rgb_adapter_mode=joint but student_joint_rgb_mlp is missing.')
            projected_rgb_feature = self.student_joint_rgb_mlp(rgb_feature)
            return self._split_concat_rgb_feature(
                projected_rgb_feature,
                template_rgb_feature_dict=rgb_feature_dict,
            )

        projected = dict()
        for key in self.rgb_keys:
            if key not in rgb_feature_dict:
                continue
            if key not in self.student_encoder_mlps.keys():
                raise KeyError(f'Missing student projector for rgb key: {key}')
            projected[key] = self.student_encoder_mlps[key](rgb_feature_dict[key])
        return projected

    def _fuse_projected_student_rgb_feature_dict(
        self,
        projected_rgb_feature_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        projected_rgb_feature = self._concat_rgb_feature_dict(projected_rgb_feature_dict)
        if projected_rgb_feature is None:
            raise RuntimeError('RGB-only latent distillation requires at least one rgb observation key.')

        if self.student_fusion_mlp is None:
            return projected_rgb_feature

        fused_rgb_feature = self.student_fusion_mlp(projected_rgb_feature)
        if self.use_residual_fusion_projector and not self.projector_residual:
            fused_rgb_feature = projected_rgb_feature + fused_rgb_feature
        return fused_rgb_feature

    def _compute_rgb_latent_loss(
        self,
        student_rgb_feature_dict: Dict[str, torch.Tensor],
        teacher_rgb_feature_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for key in self.rgb_keys:
            if key not in student_rgb_feature_dict or key not in teacher_rgb_feature_dict:
                continue
            student_feature, teacher_feature = self._normalize_latent_loss_feature_pair(
                student_rgb_feature_dict[key],
                teacher_rgb_feature_dict[key],
            )
            losses.append(
                self.stage2_alignment_objective.pair_alignment_loss(
                    student_feature,
                    teacher_feature,
                )
            )
        if len(losses) == 0:
            raise RuntimeError('RGB-only latent distillation requires at least one rgb observation key.')
        return torch.stack(losses).mean()

    def _normalize_latent_loss_feature_pair(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ):
        if not self.normalize_latent_loss_features:
            return student_feature, teacher_feature
        return (
            F.normalize(
                student_feature,
                p=2,
                dim=-1,
                eps=self.latent_loss_feature_norm_eps,
            ),
            F.normalize(
                teacher_feature,
                p=2,
                dim=-1,
                eps=self.latent_loss_feature_norm_eps,
            ),
        )

    def _per_sample_feature_pair_loss(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        student_feature, teacher_feature = self._normalize_latent_loss_feature_pair(
            student_feature,
            teacher_feature,
        )
        diff = (student_feature - teacher_feature.detach()).reshape(
            student_feature.shape[0],
            -1,
        )
        pair_loss_type = self.stage2_alignment_objective.pair_loss_type
        if pair_loss_type == 'mse':
            frame_loss = diff.pow(2).mean(dim=1)
        elif pair_loss_type == 'l1':
            frame_loss = diff.abs().mean(dim=1)
        elif pair_loss_type == 'huber':
            abs_diff = diff.abs()
            delta = self.stage2_alignment_objective.pair_huber_delta
            frame_loss = torch.where(
                abs_diff <= delta,
                0.5 * diff.pow(2),
                delta * (abs_diff - 0.5 * delta),
            ).mean(dim=1)
        elif pair_loss_type == 'smooth_l1':
            abs_diff = diff.abs()
            beta = self.stage2_alignment_objective.pair_huber_delta
            frame_loss = torch.where(
                abs_diff < beta,
                0.5 * diff.pow(2) / beta,
                abs_diff - 0.5 * beta,
            ).mean(dim=1)
        else:
            raise RuntimeError(f'Unhandled pair_loss_type: {pair_loss_type}')
        if frame_loss.shape[0] % batch_size != 0:
            raise RuntimeError(
                f'Cannot reshape {frame_loss.shape[0]} frame losses into batch size {batch_size}.'
            )
        return frame_loss.reshape(batch_size, -1).mean(dim=1)

    def _compute_rgb_latent_loss_per_sample(
        self,
        student_rgb_feature_dict: Dict[str, torch.Tensor],
        teacher_rgb_feature_dict: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        losses = []
        for key in self.rgb_keys:
            if key not in student_rgb_feature_dict or key not in teacher_rgb_feature_dict:
                continue
            losses.append(
                self._per_sample_feature_pair_loss(
                    student_rgb_feature_dict[key],
                    teacher_rgb_feature_dict[key],
                    batch_size=batch_size,
                )
            )
        if len(losses) == 0:
            raise RuntimeError('RGB-only latent distillation requires at least one rgb observation key.')
        return torch.stack(losses, dim=0).mean(dim=0)

    def _prepare_input_grad_obs(self, nobs: Dict[str, torch.Tensor]):
        input_grad_nobs = dict()
        input_tensors = []
        for key, value in nobs.items():
            if key in self.rgb_keys:
                value = value.detach().clone().requires_grad_(True)
                input_tensors.append(value)
            input_grad_nobs[key] = value
        return input_grad_nobs, input_tensors

    def _compute_input_grad_penalty(
        self,
        per_sample_distill_loss: torch.Tensor,
        input_tensors,
    ) -> torch.Tensor:
        if len(input_tensors) == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        grads = torch.autograd.grad(
            per_sample_distill_loss.sum(),
            input_tensors,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        per_sample_penalties = []
        for grad in grads:
            if grad is None:
                continue
            per_sample_penalties.append(grad.flatten(1).pow(2).sum(dim=1))
        if len(per_sample_penalties) == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return torch.stack(per_sample_penalties, dim=0).sum(dim=0).mean()

    def _should_compute_input_grad_regularization(self) -> bool:
        if not torch.is_grad_enabled():
            return False
        return (
            self.input_grad_regularization_weight > 0
            or self.input_grad_regularization_auto_calibrate
        )

    def _compute_input_grad_effective_lambda_and_metrics(
        self,
        distill_loss_for_ratio: torch.Tensor,
        input_grad_penalty: torch.Tensor,
    ):
        distill_value = float(distill_loss_for_ratio.detach().item())
        penalty_value = float(input_grad_penalty.detach().item())
        base_lambda = self.input_grad_regularization_weight
        calibrated_lambda = (
            0.0 if self._input_grad_reg_calibrated_lambda is None
            else float(self._input_grad_reg_calibrated_lambda)
        )
        calibrating = False

        if self.input_grad_regularization_auto_calibrate:
            if self._input_grad_reg_calibrated_lambda is None:
                calibrating = True
                if distill_value > self._input_grad_reg_eps and penalty_value > self._input_grad_reg_eps:
                    self._input_grad_reg_distill_sum += distill_value
                    self._input_grad_reg_penalty_sum += penalty_value
                    self._input_grad_reg_calibration_count += 1

                if self._input_grad_reg_calibration_count >= self.input_grad_regularization_calibration_batches:
                    mean_distill = (
                        self._input_grad_reg_distill_sum
                        / self._input_grad_reg_calibration_count
                    )
                    mean_penalty = (
                        self._input_grad_reg_penalty_sum
                        / self._input_grad_reg_calibration_count
                    )
                    if mean_penalty > self._input_grad_reg_eps:
                        self._input_grad_reg_calibrated_lambda = (
                            self.input_grad_regularization_target_ratio
                            * mean_distill
                            / mean_penalty
                        )
                    else:
                        self._input_grad_reg_calibrated_lambda = 0.0
                    calibrated_lambda = float(self._input_grad_reg_calibrated_lambda)
                    calibrating = False
                    print(
                        '[input-grad-calibration] '
                        f'batches={self._input_grad_reg_calibration_count} '
                        f'mean_distill={mean_distill:.6e} '
                        f'mean_grad_penalty={mean_penalty:.6e} '
                        f'target_ratio={self.input_grad_regularization_target_ratio:.6e} '
                        f'calibrated_lambda={calibrated_lambda:.6e} '
                        f'lambda_scale={self.input_grad_regularization_lambda_scale:.6e}',
                        flush=True,
                    )

            base_lambda = (
                0.0 if self._input_grad_reg_calibrated_lambda is None
                else float(self._input_grad_reg_calibrated_lambda)
            )

        effective_lambda = base_lambda * self.input_grad_regularization_lambda_scale
        ratio = (
            effective_lambda * penalty_value / max(distill_value, self._input_grad_reg_eps)
        )
        lambda_for_ratio_001 = (
            0.01 * distill_value / penalty_value
            if penalty_value > self._input_grad_reg_eps else 0.0
        )
        lambda_for_ratio_005 = (
            0.05 * distill_value / penalty_value
            if penalty_value > self._input_grad_reg_eps else 0.0
        )
        lambda_for_ratio_01 = (
            0.1 * distill_value / penalty_value
            if penalty_value > self._input_grad_reg_eps else 0.0
        )

        should_print = self._input_grad_reg_num_logged < self.input_grad_regularization_log_batches
        if should_print:
            print(
                '[input-grad-scale] '
                f'batch={self._input_grad_reg_num_logged} '
                f'distill={distill_value:.6e} '
                f'grad_penalty={penalty_value:.6e} '
                f'effective_lambda={effective_lambda:.6e} '
                f'ratio={ratio:.6e} '
                f'lambda@0.01={lambda_for_ratio_001:.6e} '
                f'lambda@0.05={lambda_for_ratio_005:.6e} '
                f'lambda@0.1={lambda_for_ratio_01:.6e} '
                f'calibrating={calibrating}',
                flush=True,
            )
        self._input_grad_reg_num_logged += 1

        return effective_lambda, {
            'input_grad_distill_loss_for_ratio': distill_value,
            'input_grad_effective_lambda': float(effective_lambda),
            'input_grad_ratio': float(ratio),
            'input_grad_calibrated_lambda': float(calibrated_lambda),
            'input_grad_calibration_batches_seen': int(self._input_grad_reg_calibration_count),
            'input_grad_calibrating': float(calibrating),
            'input_grad_lambda_for_ratio_0p01': float(lambda_for_ratio_001),
            'input_grad_lambda_for_ratio_0p05': float(lambda_for_ratio_005),
            'input_grad_lambda_for_ratio_0p1': float(lambda_for_ratio_01),
        }

    def _compute_student_teacher_rgb_latent_losses(
        self,
        nobs: Dict[str, torch.Tensor],
        teacher_rgb_feature_dict: Dict[str, torch.Tensor],
        teacher_rgb_feature: torch.Tensor,
        crop_index_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ):
        batch_size = next(iter(nobs.values())).shape[0]
        student_rgb_feature_dict, _ = self._encode_obs_split_feature_dicts(
            self.obs_encoder,
            nobs,
            deterministic_crop=self.deterministic_latent_crop,
            use_obs_steps_only=True,
            crop_index_overrides=crop_index_overrides,
        )
        projected_student_rgb_feature_dict = self._project_student_rgb_feature_dict(
            student_rgb_feature_dict,
        )
        local_latent_loss_per_sample = self._compute_rgb_latent_loss_per_sample(
            projected_student_rgb_feature_dict,
            teacher_rgb_feature_dict,
            batch_size=batch_size,
        )
        local_latent_loss = local_latent_loss_per_sample.mean()
        projected_student_rgb_feature = self._fuse_projected_student_rgb_feature_dict(
            projected_student_rgb_feature_dict,
        )
        global_latent_loss_per_sample = self._per_sample_feature_pair_loss(
            projected_student_rgb_feature,
            teacher_rgb_feature,
            batch_size=batch_size,
        )
        global_latent_loss = global_latent_loss_per_sample.mean()
        latent_loss_per_sample = (
            self.local_latent_loss_weight * local_latent_loss_per_sample
            + self.global_latent_loss_weight * global_latent_loss_per_sample
        )
        latent_loss = (
            self.local_latent_loss_weight * local_latent_loss
            + self.global_latent_loss_weight * global_latent_loss
        )
        return (
            latent_loss,
            local_latent_loss,
            global_latent_loss,
            latent_loss_per_sample,
            projected_student_rgb_feature,
        )

    def _compute_stage2_stats_alignment_terms(
        self,
        teacher_rgb_feature: torch.Tensor,
        branch_student_features,
    ):
        zero = teacher_rgb_feature.sum() * 0.0
        mean_loss = zero
        coral_loss = zero
        metric_sums = dict()
        metric_weight_sum = 0.0

        for branch_weight, student_rgb_feature in branch_student_features:
            branch_weight = float(branch_weight)
            if branch_weight <= 0.0 or student_rgb_feature is None:
                continue
            loss_student_rgb_feature, loss_teacher_rgb_feature = (
                self._normalize_latent_loss_feature_pair(
                    student_rgb_feature,
                    teacher_rgb_feature,
                )
            )

            branch_mean_loss = self.stage2_alignment_objective.mean_alignment_loss(
                loss_student_rgb_feature,
                loss_teacher_rgb_feature,
            )
            branch_coral_loss, _ = (
                self.stage2_alignment_objective.coral_covariance_loss(
                    loss_student_rgb_feature,
                    loss_teacher_rgb_feature,
                )
            )
            _, branch_metrics = self.stage2_alignment_objective.coral_covariance_loss(
                student_rgb_feature,
                teacher_rgb_feature,
            )
            mean_loss = mean_loss + branch_weight * branch_mean_loss
            coral_loss = coral_loss + branch_weight * branch_coral_loss
            metric_weight_sum += branch_weight
            for key, value in branch_metrics.items():
                if key not in metric_sums:
                    metric_sums[key] = branch_weight * value
                else:
                    metric_sums[key] = metric_sums[key] + branch_weight * value

        if metric_weight_sum <= 0.0:
            _, metrics = self.stage2_alignment_objective.coral_covariance_loss(
                teacher_rgb_feature,
                teacher_rgb_feature,
            )
        else:
            metrics = {
                key: value / metric_weight_sum
                for key, value in metric_sums.items()
            }
        metrics['stats/latent_loss_features_normalized'] = (
            zero + (1.0 if self.normalize_latent_loss_features else 0.0)
        )
        return mean_loss, coral_loss, metrics

    def _encode_student_features(
        self,
        nobs: Dict[str, torch.Tensor],
        deterministic_crop: bool,
        use_obs_steps_only: bool = True,
        crop_index_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        student_rgb_feature_dict, student_low_dim_feature = self._encode_obs_split_feature_dicts(
            self.obs_encoder,
            nobs,
            deterministic_crop=deterministic_crop,
            use_obs_steps_only=use_obs_steps_only,
            crop_index_overrides=crop_index_overrides,
        )
        projected_rgb_feature_dict = self._project_student_rgb_feature_dict(
            student_rgb_feature_dict,
        )
        projected_rgb_feature = self._fuse_projected_student_rgb_feature_dict(
            projected_rgb_feature_dict,
        )

        features = [projected_rgb_feature]
        if student_low_dim_feature is not None:
            features.append(student_low_dim_feature)
        return torch.cat(features, dim=-1)

    def get_last_loss_components(self) -> Dict[str, float]:
        return dict(self._last_loss_components)

    def _compute_student_diffusion_loss(
        self,
        nobs: Dict[str, torch.Tensor],
        nactions: torch.Tensor,
        crop_index_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            nobs_features = self._encode_student_features(
                nobs,
                deterministic_crop=self.deterministic_latent_crop,
                use_obs_steps_only=True,
                crop_index_overrides=crop_index_overrides,
            )
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            nobs_features = self._encode_student_features(
                nobs,
                deterministic_crop=self.deterministic_latent_crop,
                use_obs_steps_only=False,
                crop_index_overrides=crop_index_overrides,
            )
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory,
            noise,
            timesteps,
        )
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f'Unsupported prediction type {pred_type}')

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.reshape(batch_size, -1).mean(dim=1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Stage-2 diffusion task loss is not finite.')
        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size, obs_steps = value.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_steps = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            nobs_features = self._encode_student_features(
                nobs,
                deterministic_crop=True,
                use_obs_steps_only=True,
            )
            global_cond = nobs_features.reshape(batch_size, -1)
            cond_data = torch.zeros(size=(batch_size, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            nobs_features = self._encode_student_features(
                nobs,
                deterministic_crop=True,
                use_obs_steps_only=True,
            )
            nobs_features = nobs_features.reshape(batch_size, obs_steps, -1)
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim + self.obs_feature_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :obs_steps, action_dim:] = nobs_features
            cond_mask[:, :obs_steps, action_dim:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
        }

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs_clean = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        diffusion_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        diffusion_clean_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        diffusion_robust_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        crop_index_overrides = None
        if not self.deterministic_latent_crop:
            crop_index_overrides = self._sample_shared_random_crop_index_overrides(
                self.obs_encoder,
                nobs_clean,
                use_obs_steps_only=True,
            )

        with torch.no_grad():
            teacher_rgb_feature_dict, _ = self._encode_obs_split_feature_dicts(
                self.teacher_obs_encoder,
                nobs_clean,
                deterministic_crop=self.deterministic_latent_crop,
                use_obs_steps_only=True,
                crop_index_overrides=crop_index_overrides,
            )
        teacher_rgb_feature = self._concat_rgb_feature_dict(teacher_rgb_feature_dict)
        if teacher_rgb_feature is None:
            raise RuntimeError('RGB-only latent distillation requires at least one rgb observation key.')

        robust_latent_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        robust_local_latent_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        robust_global_latent_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        robust_projected_rgb_feature = None
        nobs_robust = None
        input_grad_penalty_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        input_grad_penalty_count = 0
        input_grad_distill_loss_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        input_grad_distill_loss_count = 0
        compute_input_grad_regularization = self._should_compute_input_grad_regularization()
        if self.robust_latent_loss_weight > 0:
            nobs_robust = self._make_robust_latent_input_obs(nobs_clean)
            robust_input_tensors = []
            if compute_input_grad_regularization:
                nobs_robust, robust_input_tensors = self._prepare_input_grad_obs(nobs_robust)
            (
                robust_latent_loss,
                robust_local_latent_loss,
                robust_global_latent_loss,
                robust_latent_loss_per_sample,
                robust_projected_rgb_feature,
            ) = self._compute_student_teacher_rgb_latent_losses(
                nobs_robust,
                teacher_rgb_feature_dict,
                teacher_rgb_feature,
                crop_index_overrides=crop_index_overrides,
            )
            if compute_input_grad_regularization:
                grad_loss = robust_latent_loss_per_sample
                if self.input_grad_regularization_loss_weighted:
                    grad_loss = self.robust_latent_loss_weight * grad_loss
                input_grad_distill_loss_sum = input_grad_distill_loss_sum + grad_loss.mean()
                input_grad_distill_loss_count += 1
                input_grad_penalty_sum = input_grad_penalty_sum + self._compute_input_grad_penalty(
                    grad_loss,
                    robust_input_tensors,
                )
                input_grad_penalty_count += 1

        if self.diffusion_loss_weight > 0.0:
            diffusion_losses = []
            if self.diffusion_loss_input_mode in ('clean', 'both'):
                diffusion_clean_loss = self._compute_student_diffusion_loss(
                    nobs_clean,
                    nactions,
                    crop_index_overrides=crop_index_overrides,
                )
                diffusion_losses.append(diffusion_clean_loss)
            if self.diffusion_loss_input_mode in ('robust', 'both'):
                if nobs_robust is None:
                    nobs_robust = self._make_robust_latent_input_obs(nobs_clean)
                diffusion_robust_loss = self._compute_student_diffusion_loss(
                    nobs_robust,
                    nactions,
                    crop_index_overrides=crop_index_overrides,
                )
                diffusion_losses.append(diffusion_robust_loss)
            diffusion_loss = torch.stack(diffusion_losses).mean()

        clean_latent_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        clean_local_latent_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        clean_global_latent_loss = torch.zeros((), device=self.device, dtype=self.dtype)
        clean_projected_rgb_feature = None
        if self.clean_latent_loss_weight > 0:
            nobs_clean_for_student = nobs_clean
            clean_input_tensors = []
            if compute_input_grad_regularization:
                nobs_clean_for_student, clean_input_tensors = self._prepare_input_grad_obs(nobs_clean)
            (
                clean_latent_loss,
                clean_local_latent_loss,
                clean_global_latent_loss,
                clean_latent_loss_per_sample,
                clean_projected_rgb_feature,
            ) = self._compute_student_teacher_rgb_latent_losses(
                nobs_clean_for_student,
                teacher_rgb_feature_dict,
                teacher_rgb_feature,
                crop_index_overrides=crop_index_overrides,
            )
            if compute_input_grad_regularization:
                grad_loss = clean_latent_loss_per_sample
                if self.input_grad_regularization_loss_weighted:
                    grad_loss = self.clean_latent_loss_weight * grad_loss
                input_grad_distill_loss_sum = input_grad_distill_loss_sum + grad_loss.mean()
                input_grad_distill_loss_count += 1
                input_grad_penalty_sum = input_grad_penalty_sum + self._compute_input_grad_penalty(
                    grad_loss,
                    clean_input_tensors,
                )
                input_grad_penalty_count += 1

        local_latent_loss = (
            self.robust_latent_loss_weight * robust_local_latent_loss
            + self.clean_latent_loss_weight * clean_local_latent_loss
        )
        global_latent_loss = (
            self.robust_latent_loss_weight * robust_global_latent_loss
            + self.clean_latent_loss_weight * clean_global_latent_loss
        )
        latent_loss = (
            self.robust_latent_loss_weight * robust_latent_loss
            + self.clean_latent_loss_weight * clean_latent_loss
        )
        mean_alignment_loss, coral_alignment_loss, stage2_stats_metrics = (
            self._compute_stage2_stats_alignment_terms(
                teacher_rgb_feature=teacher_rgb_feature,
                branch_student_features=(
                    (self.robust_latent_loss_weight, robust_projected_rgb_feature),
                    (self.clean_latent_loss_weight, clean_projected_rgb_feature),
                ),
            )
        )
        stage2_alignment_loss, stage2_alignment_metrics = (
            self.stage2_alignment_objective.combine_loss_terms(
                pair_raw=latent_loss,
                mean_raw=mean_alignment_loss,
                coral_raw=coral_alignment_loss,
                stats_metrics=stage2_stats_metrics,
            )
        )
        input_grad_penalty = torch.zeros((), device=self.device, dtype=self.dtype)
        if input_grad_penalty_count > 0:
            input_grad_penalty = input_grad_penalty_sum / input_grad_penalty_count
        input_grad_distill_loss_for_ratio = latent_loss.detach()
        if input_grad_distill_loss_count > 0:
            input_grad_distill_loss_for_ratio = (
                input_grad_distill_loss_sum / input_grad_distill_loss_count
            )
        input_grad_effective_lambda = (
            self.input_grad_regularization_weight
            * self.input_grad_regularization_lambda_scale
        )
        input_grad_metrics = {
            'input_grad_distill_loss_for_ratio': float(input_grad_distill_loss_for_ratio.detach().item()),
            'input_grad_effective_lambda': float(input_grad_effective_lambda),
            'input_grad_ratio': 0.0,
            'input_grad_calibrated_lambda': (
                0.0 if self._input_grad_reg_calibrated_lambda is None
                else float(self._input_grad_reg_calibrated_lambda)
            ),
            'input_grad_calibration_batches_seen': int(self._input_grad_reg_calibration_count),
            'input_grad_calibrating': 0.0,
            'input_grad_lambda_for_ratio_0p01': 0.0,
            'input_grad_lambda_for_ratio_0p05': 0.0,
            'input_grad_lambda_for_ratio_0p1': 0.0,
        }
        if compute_input_grad_regularization and input_grad_penalty_count > 0:
            input_grad_effective_lambda, input_grad_metrics = (
                self._compute_input_grad_effective_lambda_and_metrics(
                    input_grad_distill_loss_for_ratio,
                    input_grad_penalty,
                )
            )
        input_grad_regularization_loss = (
            input_grad_effective_lambda * input_grad_penalty
        )
        regularized_latent_loss = stage2_alignment_loss + input_grad_regularization_loss

        total_loss = (
            self.diffusion_loss_weight * diffusion_loss
            + self.latent_loss_weight * regularized_latent_loss
        )
        stage2_alignment_metrics = {
            key: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for key, value in stage2_alignment_metrics.items()
        }
        augmentation_metrics = {
            key: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for key, value in self._last_augmentation_metrics.items()
        }
        self._last_loss_components = {
            'total_loss': float(total_loss.detach().item()),
            'diffusion_loss': float(diffusion_loss.detach().item()),
            'diffusion_clean_loss': float(diffusion_clean_loss.detach().item()),
            'diffusion_robust_loss': float(diffusion_robust_loss.detach().item()),
            'latent_loss': float(latent_loss.detach().item()),
            'stage2_alignment_loss': float(stage2_alignment_loss.detach().item()),
            'regularized_latent_loss': float(regularized_latent_loss.detach().item()),
            'input_grad_penalty': float(input_grad_penalty.detach().item()),
            'input_grad_regularization_loss': float(input_grad_regularization_loss.detach().item()),
            **input_grad_metrics,
            **stage2_alignment_metrics,
            **augmentation_metrics,
            'local_latent_loss': float(local_latent_loss.detach().item()),
            'global_latent_loss': float(global_latent_loss.detach().item()),
            'robust_latent_loss': float(robust_latent_loss.detach().item()),
            'robust_local_latent_loss': float(robust_local_latent_loss.detach().item()),
            'robust_global_latent_loss': float(robust_global_latent_loss.detach().item()),
            'clean_latent_loss': float(clean_latent_loss.detach().item()),
            'clean_local_latent_loss': float(clean_local_latent_loss.detach().item()),
            'clean_global_latent_loss': float(clean_global_latent_loss.detach().item()),
        }
        return total_loss
