from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_matching_prior(
    feature_t1: torch.Tensor,
    feature_t2: torch.Tensor,
    local_radius: int = 2,
) -> torch.Tensor:
    """Estimate patch-level change evidence from local DINO feature matching."""
    if feature_t1.shape != feature_t2.shape:
        raise ValueError("feature_t1 and feature_t2 must have identical shapes")
    if feature_t1.ndim != 4:
        raise ValueError("features must have shape [batch, channels, height, width]")
    if local_radius < 0:
        raise ValueError("local_radius must be non-negative")

    source = F.normalize(feature_t1.float(), dim=1)
    target = F.normalize(feature_t2.float(), dim=1)
    similarity_12 = _best_local_similarity(source, target, local_radius)
    similarity_21 = _best_local_similarity(target, source, local_radius)
    residual = 0.25 * ((1.0 - similarity_12) + (1.0 - similarity_21))
    return residual.clamp(0.0, 1.0)


def cam_teacher_consistency_loss(
    student_cam: torch.Tensor,
    teacher_cam: torch.Tensor,
    labels: torch.Tensor,
    positive_only: bool = True,
) -> torch.Tensor:
    """Keep student CAM close to a detached teacher CAM."""
    if student_cam.shape != teacher_cam.shape:
        raise ValueError("student_cam and teacher_cam must have identical shapes")
    if student_cam.ndim != 3:
        raise ValueError("CAM tensors must have shape [batch, height, width]")
    if labels.ndim != 1 or labels.shape[0] != student_cam.shape[0]:
        raise ValueError("labels must have shape [batch]")

    if positive_only:
        selected = labels > 0.5
    else:
        selected = torch.ones_like(labels, dtype=torch.bool)
    if not selected.any():
        return student_cam.new_zeros(())
    return F.smooth_l1_loss(
        student_cam[selected].float(),
        teacher_cam[selected].detach().float(),
    )


class StrongCAMMatchingPriorLoss(nn.Module):
    """Image-level MIL with a frozen local-matching localization prior."""

    expects_model_output = True

    def __init__(
        self,
        positive_weight: float,
        prior_consistency_weight: float = 0.1,
        prior_coverage_weight: float = 0.05,
        prior_separation_weight: float = 0.05,
        low_prior_suppression_weight: float = 0.05,
        negative_cam_weight: float = 0.05,
        negative_topk_weight: float = 0.0,
        negative_topk_ratio: float = 0.05,
        negative_margin: float = 0.05,
        negative_prior_interference_weight: float = 0.0,
        negative_prior_quantile: float = 0.8,
        local_radius: int = 2,
        robust_low_quantile: float = 0.05,
        robust_high_quantile: float = 0.95,
        high_prior_quantile: float = 0.8,
        low_prior_quantile: float = 0.5,
        coverage_target: float = 0.6,
        separation_margin: float = 0.2,
    ) -> None:
        super().__init__()
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        for name, value in (
            ("prior_consistency_weight", prior_consistency_weight),
            ("prior_coverage_weight", prior_coverage_weight),
            ("prior_separation_weight", prior_separation_weight),
            ("low_prior_suppression_weight", low_prior_suppression_weight),
            ("negative_cam_weight", negative_cam_weight),
            ("negative_topk_weight", negative_topk_weight),
            (
                "negative_prior_interference_weight",
                negative_prior_interference_weight,
            ),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 < negative_topk_ratio <= 1:
            raise ValueError("negative_topk_ratio must be in (0, 1]")
        if negative_margin < 0:
            raise ValueError("negative_margin must be non-negative")
        if not 0 < negative_prior_quantile < 1:
            raise ValueError("negative_prior_quantile must be in (0, 1)")
        if local_radius < 0:
            raise ValueError("local_radius must be non-negative")
        if not 0 <= robust_low_quantile < robust_high_quantile <= 1:
            raise ValueError(
                "robust quantiles must satisfy 0 <= low < high <= 1"
            )
        if not 0 < low_prior_quantile < high_prior_quantile < 1:
            raise ValueError(
                "prior quantiles must satisfy 0 < low < high < 1"
            )
        if not 0 <= coverage_target <= 1:
            raise ValueError("coverage_target must be in [0, 1]")
        if separation_margin < 0:
            raise ValueError("separation_margin must be non-negative")

        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))
        self.prior_consistency_weight = float(prior_consistency_weight)
        self.prior_coverage_weight = float(prior_coverage_weight)
        self.prior_separation_weight = float(prior_separation_weight)
        self.low_prior_suppression_weight = float(low_prior_suppression_weight)
        self.negative_cam_weight = float(negative_cam_weight)
        self.negative_topk_weight = float(negative_topk_weight)
        self.negative_topk_ratio = float(negative_topk_ratio)
        self.negative_margin = float(negative_margin)
        self.negative_prior_interference_weight = float(
            negative_prior_interference_weight
        )
        self.negative_prior_quantile = float(negative_prior_quantile)
        self.local_radius = int(local_radius)
        self.robust_low_quantile = float(robust_low_quantile)
        self.robust_high_quantile = float(robust_high_quantile)
        self.high_prior_quantile = float(high_prior_quantile)
        self.low_prior_quantile = float(low_prior_quantile)
        self.coverage_target = float(coverage_target)
        self.separation_margin = float(separation_margin)

    def forward(
        self, output: dict[str, Any], labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = output["logits"].float()
        labels = labels.float()
        bag = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.positive_weight
        )

        cam = output["cam_score"].float().clamp(1e-6, 1.0 - 1e-6)
        negative = labels < 0.5
        positive = labels > 0.5
        if negative.any():
            negative_cam_scores = cam[negative]
            negative_cam = negative_cam_scores.mean()
            negative_topk = _topk_margin_mean(
                negative_cam_scores,
                self.negative_topk_ratio,
                self.negative_margin,
            )
            if self.negative_prior_interference_weight > 0:
                negative_prior = local_matching_prior(
                    output["feature_t1"][negative].detach(),
                    output["feature_t2"][negative].detach(),
                    self.local_radius,
                )
                negative_prior = _robust_normalize(
                    negative_prior,
                    self.robust_low_quantile,
                    self.robust_high_quantile,
                ).detach()
                negative_prior_interference = _negative_prior_interference(
                    negative_cam_scores,
                    negative_prior,
                    self.negative_prior_quantile,
                )
            else:
                negative_prior_interference = bag.new_zeros(())
        else:
            negative_cam = bag.new_zeros(())
            negative_topk = bag.new_zeros(())
            negative_prior_interference = bag.new_zeros(())

        if positive.any() and self._uses_prior():
            prior = local_matching_prior(
                output["feature_t1"][positive].detach(),
                output["feature_t2"][positive].detach(),
                self.local_radius,
            )
            target = _robust_normalize(
                prior,
                self.robust_low_quantile,
                self.robust_high_quantile,
            ).detach()
            positive_cam = cam[positive]
            prior_consistency = F.smooth_l1_loss(positive_cam, target)
            high_mean, low_mean = _prior_band_means(
                positive_cam,
                target,
                self.high_prior_quantile,
                self.low_prior_quantile,
            )
            prior_coverage = F.relu(self.coverage_target - high_mean).mean()
            prior_separation = F.relu(
                self.separation_margin - (high_mean - low_mean)
            ).mean()
            low_prior_suppression = low_mean.mean()
        else:
            prior_consistency = bag.new_zeros(())
            prior_coverage = bag.new_zeros(())
            prior_separation = bag.new_zeros(())
            low_prior_suppression = bag.new_zeros(())

        total = (
            bag
            + self.prior_consistency_weight * prior_consistency
            + self.prior_coverage_weight * prior_coverage
            + self.prior_separation_weight * prior_separation
            + self.low_prior_suppression_weight * low_prior_suppression
            + self.negative_cam_weight * negative_cam
            + self.negative_topk_weight * negative_topk
            + self.negative_prior_interference_weight * negative_prior_interference
        )
        return {
            "total": total,
            "bag": bag,
            "prior_consistency": prior_consistency,
            "prior_coverage": prior_coverage,
            "prior_separation": prior_separation,
            "low_prior_suppression": low_prior_suppression,
            "negative_cam": negative_cam,
            "negative_topk": negative_topk,
            "negative_prior_interference": negative_prior_interference,
        }

    def _uses_prior(self) -> bool:
        return (
            self.prior_consistency_weight > 0
            or self.prior_coverage_weight > 0
            or self.prior_separation_weight > 0
            or self.low_prior_suppression_weight > 0
        )


def _best_local_similarity(
    source: torch.Tensor, target: torch.Tensor, local_radius: int
) -> torch.Tensor:
    batch_size, channels, height, width = source.shape
    source_flat = source.flatten(2)
    if local_radius == 0:
        return (source_flat * target.flatten(2)).sum(dim=1).reshape(
            batch_size, height, width
        )

    kernel_size = 2 * local_radius + 1
    padded = F.pad(
        target,
        (local_radius, local_radius, local_radius, local_radius),
        mode="replicate",
    )
    windows = F.unfold(padded, kernel_size=kernel_size)
    windows = windows.reshape(
        batch_size, channels, kernel_size * kernel_size, height * width
    )
    similarities = (source_flat.unsqueeze(2) * windows).sum(dim=1)
    return similarities.max(dim=1).values.reshape(batch_size, height, width)


def _robust_normalize(
    score: torch.Tensor,
    low_quantile: float,
    high_quantile: float,
) -> torch.Tensor:
    flat = score.flatten(1)
    low = torch.quantile(flat, low_quantile, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat, high_quantile, dim=1).view(-1, 1, 1)
    return ((score - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def _prior_band_means(
    cam: torch.Tensor,
    prior: torch.Tensor,
    high_quantile: float,
    low_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_prior = prior.flatten(1)
    high_threshold = torch.quantile(flat_prior, high_quantile, dim=1).view(-1, 1, 1)
    low_threshold = torch.quantile(flat_prior, low_quantile, dim=1).view(-1, 1, 1)
    high_mask = prior >= high_threshold
    low_mask = prior <= low_threshold
    return _masked_image_mean(cam, high_mask), _masked_image_mean(cam, low_mask)


def _masked_image_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted_sum = (values * mask.to(values.dtype)).flatten(1).sum(dim=1)
    count = mask.flatten(1).sum(dim=1).clamp_min(1).to(values.dtype)
    return weighted_sum / count


def _topk_margin_mean(
    scores: torch.Tensor,
    topk_ratio: float,
    margin: float,
) -> torch.Tensor:
    flat = scores.flatten(1)
    count = max(1, round(flat.shape[1] * topk_ratio))
    topk = flat.topk(count, dim=1).values
    return F.relu(topk - margin).mean()


def _negative_prior_interference(
    cam: torch.Tensor,
    prior: torch.Tensor,
    negative_prior_quantile: float,
) -> torch.Tensor:
    flat_prior = prior.flatten(1)
    threshold = torch.quantile(
        flat_prior,
        negative_prior_quantile,
        dim=1,
    ).view(-1, 1, 1)
    high_negative_prior = prior >= threshold
    return _masked_image_mean(cam, high_negative_prior).mean()
