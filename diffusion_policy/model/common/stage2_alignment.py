from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2AlignmentObjective(nn.Module):
    def __init__(
        self,
        pair_loss_type: str = 'mse',
        pair_huber_delta: float = 1.0,
        lambda_pair: float = 1.0,
        lambda_mean: float = 0.0,
        lambda_coral: float = 0.0,
        coral_eps: float = 1.0e-5,
        coral_unbiased: bool = True,
    ):
        super().__init__()
        pair_loss_type = str(pair_loss_type).lower()
        pair_loss_aliases = {
            'l2': 'mse',
            'mae': 'l1',
        }
        pair_loss_type = pair_loss_aliases.get(pair_loss_type, pair_loss_type)
        supported_pair_losses = ('mse', 'l1', 'huber', 'smooth_l1')
        if pair_loss_type not in supported_pair_losses:
            raise ValueError(f'Unsupported Stage-2 pair_loss_type: {pair_loss_type}')
        pair_huber_delta = float(pair_huber_delta)
        if pair_huber_delta <= 0.0:
            raise ValueError(
                f'pair_huber_delta must be positive, got {pair_huber_delta}'
            )
        self.pair_loss_type = pair_loss_type
        self.pair_huber_delta = pair_huber_delta
        self.lambda_pair = float(lambda_pair)
        self.lambda_mean = float(lambda_mean)
        self.lambda_coral = float(lambda_coral)
        self.coral_eps = float(coral_eps)
        self.coral_unbiased = bool(coral_unbiased)

    def _check_finite(self, name: str, value: torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Stage-2 alignment {name} is not finite.')

    def _validate_stats_view(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if student.shape != teacher.shape:
            raise ValueError(
                'Stage-2 student/teacher stats views must have the same shape, '
                f'got student={tuple(student.shape)} teacher={tuple(teacher.shape)}.'
            )
        if student.ndim != 2:
            raise ValueError(
                'Stage-2 mean/CORAL alignment expects an explicit [B, D] stats view, '
                f'got shape {tuple(student.shape)}.'
            )
        return student.float(), teacher.detach().float()

    def pair_alignment_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        teacher = teacher.detach()
        if self.pair_loss_type == 'mse':
            return F.mse_loss(student, teacher)
        if self.pair_loss_type == 'l1':
            return F.l1_loss(student, teacher)
        diff = (student - teacher).abs()
        if self.pair_loss_type == 'huber':
            delta = self.pair_huber_delta
            return torch.where(
                diff <= delta,
                0.5 * diff.pow(2),
                delta * (diff - 0.5 * delta),
            ).mean()
        if self.pair_loss_type == 'smooth_l1':
            beta = self.pair_huber_delta
            return torch.where(
                diff < beta,
                0.5 * diff.pow(2) / beta,
                diff - 0.5 * beta,
            ).mean()
        raise RuntimeError(f'Unhandled pair_loss_type: {self.pair_loss_type}')

    def mean_alignment_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        student_stats, teacher_stats = self._validate_stats_view(student, teacher)
        student_mean = student_stats.mean(dim=0)
        teacher_mean = teacher_stats.mean(dim=0)
        loss = (student_mean - teacher_mean).pow(2).mean()
        self._check_finite('mean loss', loss)
        return loss

    def _covariance(
        self,
        stats_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, feature_dim = stats_view.shape
        if batch_size < 2:
            cov = stats_view.new_zeros((feature_dim, feature_dim))
            skipped = stats_view.new_ones(())
            return cov, skipped

        centered = stats_view - stats_view.mean(dim=0, keepdim=True)
        denom = batch_size - 1 if self.coral_unbiased else batch_size
        cov = centered.transpose(0, 1).matmul(centered) / float(denom)
        skipped = stats_view.new_zeros(())
        return cov, skipped

    def coral_covariance_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        student_stats, teacher_stats = self._validate_stats_view(student, teacher)
        batch_size, feature_dim = student_stats.shape

        teacher_mean = teacher_stats.mean(dim=0)
        student_mean = student_stats.mean(dim=0)
        teacher_latent_mean = teacher_stats.mean()
        student_latent_mean = student_stats.mean()
        teacher_output_norm = teacher_stats.norm(p=2, dim=1)
        student_output_norm = student_stats.norm(p=2, dim=1)
        teacher_output_norm_mean = teacher_output_norm.mean()
        student_output_norm_mean = student_output_norm.mean()
        teacher_output_rms = teacher_stats.pow(2).mean().sqrt()
        student_output_rms = student_stats.pow(2).mean().sqrt()
        normalized_teacher = F.normalize(teacher_stats, p=2, dim=1, eps=self.coral_eps)
        normalized_student = F.normalize(student_stats, p=2, dim=1, eps=self.coral_eps)
        normalized_pair_mse = F.mse_loss(normalized_student, normalized_teacher)
        normalized_cosine_similarity = (normalized_student * normalized_teacher).sum(dim=1).mean()
        teacher_cov, teacher_skipped = self._covariance(teacher_stats)
        student_cov, student_skipped = self._covariance(student_stats)
        coral_skipped = torch.maximum(teacher_skipped, student_skipped)

        if batch_size < 2:
            coral_loss = student_stats.sum() * 0.0
        else:
            cov_gap = student_cov - teacher_cov
            coral_loss = cov_gap.pow(2).sum() / float(4 * feature_dim * feature_dim)

        cov_gap = student_cov - teacher_cov
        teacher_std = torch.sqrt(torch.diagonal(teacher_cov).clamp_min(0.0))
        student_std = torch.sqrt(torch.diagonal(student_cov).clamp_min(0.0))
        metrics = {
            'stats/teacher_latent_mean': teacher_latent_mean,
            'stats/student_latent_mean': student_latent_mean,
            'stats/latent_mean_delta': student_latent_mean - teacher_latent_mean,
            'stats/teacher_output_norm_mean': teacher_output_norm_mean,
            'stats/student_output_norm_mean': student_output_norm_mean,
            'stats/output_norm_ratio': (
                student_output_norm_mean
                / teacher_output_norm_mean.clamp_min(self.coral_eps)
            ),
            'stats/output_norm_gap': student_output_norm_mean - teacher_output_norm_mean,
            'stats/teacher_output_rms': teacher_output_rms,
            'stats/student_output_rms': student_output_rms,
            'stats/output_rms_ratio': (
                student_output_rms
                / teacher_output_rms.clamp_min(self.coral_eps)
            ),
            'stats/normalized_pair_mse': normalized_pair_mse,
            'stats/normalized_cosine_similarity': normalized_cosine_similarity,
            'stats/teacher_mean_norm': teacher_mean.norm(p=2),
            'stats/student_mean_norm': student_mean.norm(p=2),
            'stats/mean_gap': (student_mean - teacher_mean).norm(p=2),
            'stats/teacher_cov_trace': torch.trace(teacher_cov),
            'stats/student_cov_trace': torch.trace(student_cov),
            'stats/covariance_gap': torch.linalg.matrix_norm(cov_gap, ord='fro'),
            'stats/teacher_std_mean': teacher_std.mean(),
            'stats/student_std_mean': student_std.mean(),
            'stats/student_std_min': student_std.min(),
            'stats/coral_skipped': coral_skipped,
        }
        self._check_finite('CORAL loss', coral_loss)
        for key, value in metrics.items():
            self._check_finite(key, value)
        return coral_loss, metrics

    def combine_loss_terms(
        self,
        pair_raw: torch.Tensor,
        mean_raw: torch.Tensor,
        coral_raw: torch.Tensor,
        stats_metrics: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._check_finite('pair loss', pair_raw)
        self._check_finite('mean loss', mean_raw)
        self._check_finite('CORAL loss', coral_raw)

        pair_weighted = pair_raw * self.lambda_pair
        mean_weighted = mean_raw * self.lambda_mean
        coral_weighted = coral_raw * self.lambda_coral

        total = pair_weighted
        if self.lambda_mean != 0.0:
            total = total + mean_weighted
        if self.lambda_coral != 0.0:
            total = total + coral_weighted
        self._check_finite('total loss', total)

        metrics = {
            'loss/total': total.detach(),
            'loss/pair_raw': pair_raw.detach(),
            'loss/pair_weighted': pair_weighted.detach(),
            'loss/mean_raw': mean_raw.detach(),
            'loss/mean_weighted': mean_weighted.detach(),
            'loss/coral_raw': coral_raw.detach(),
            'loss/coral_weighted': coral_weighted.detach(),
        }
        metrics.update({key: value.detach() for key, value in stats_metrics.items()})
        for key, value in metrics.items():
            self._check_finite(key, value)
        return total, metrics

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        pair_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if pair_loss is None:
            pair_raw = self.pair_alignment_loss(student, teacher)
        else:
            pair_raw = pair_loss
        self._check_finite('pair loss', pair_raw)

        mean_raw = self.mean_alignment_loss(student, teacher)
        coral_raw, stats_metrics = self.coral_covariance_loss(student, teacher)
        return self.combine_loss_terms(
            pair_raw=pair_raw,
            mean_raw=mean_raw,
            coral_raw=coral_raw,
            stats_metrics=stats_metrics,
        )
