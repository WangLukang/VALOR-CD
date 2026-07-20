from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


def apply_pseudo_label_refinement(
    scores: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Refine CAM scores before pseudo-label conversion."""
    settings = dict(config or {})
    name = str(settings.get("name", "none"))
    if name in {"none", ""}:
        return scores
    if name == "image_difference":
        return image_difference_guided_refinement(scores, t1, t2, settings)
    raise ValueError(f"Unsupported pseudo-label refinement: {name}")


def image_difference_guided_refinement(
    scores: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Suppress CAM regions unsupported by high-resolution T1/T2 differences."""
    settings = dict(config or {})
    if scores.ndim != 3:
        raise ValueError("scores must have shape [B, H, W]")
    if t1.shape != t2.shape or t1.ndim != 4:
        raise ValueError("t1 and t2 must have identical shape [B, C, H, W]")
    if t1.shape[0] != scores.shape[0]:
        raise ValueError("scores and images must have the same batch size")

    strength = float(settings.get("strength", 0.85))
    if not 0 <= strength <= 1:
        raise ValueError("refinement strength must be in [0, 1]")
    gamma = float(settings.get("gamma", 1.0))
    if gamma <= 0:
        raise ValueError("refinement gamma must be positive")
    min_gate = float(settings.get("min_gate", 0.05))
    if not 0 <= min_gate <= 1:
        raise ValueError("min_gate must be in [0, 1]")

    base_prior = image_difference_prior(
        t1,
        t2,
        low_quantile=float(settings.get("low_quantile", 0.05)),
        high_quantile=float(settings.get("high_quantile", 0.95)),
        blur_kernel=int(settings.get("blur_kernel", 3)),
    )
    if base_prior.shape[-2:] != scores.shape[-2:]:
        base_prior = F.interpolate(
            base_prior.unsqueeze(1),
            size=scores.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    base_prior = base_prior.to(device=scores.device, dtype=scores.dtype)
    prior = base_prior.pow(gamma)
    gate = min_gate + (1.0 - min_gate) * prior
    if bool(settings.get("preserve_high_scores", False)):
        score_threshold = float(settings.get("score_threshold", 0.5))
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        score_temperature = float(settings.get("score_temperature", 0.08))
        if score_temperature <= 0:
            raise ValueError("score_temperature must be positive")
        score_gate = torch.sigmoid((scores.float() - score_threshold) / score_temperature)
        gate = torch.maximum(gate, score_gate.to(dtype=gate.dtype))
    gate = (1.0 - strength) + strength * gate
    gate = apply_overexpansion_gate(scores, gate, base_prior, settings)
    return (scores * gate).clamp(0.0, 1.0)


def apply_overexpansion_gate(
    scores: torch.Tensor,
    gate: torch.Tensor,
    base_prior: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    """Apply a stricter boundary gate only for over-expanded CAM maps."""
    settings = dict(config.get("overexpansion") or {})
    if not settings or not bool(settings.get("enabled", True)):
        return gate
    score_threshold = float(
        settings.get("score_threshold", config.get("score_threshold", 0.5))
    )
    if not 0 <= score_threshold <= 1:
        raise ValueError("overexpansion score_threshold must be in [0, 1]")
    trigger_ratio = float(settings.get("trigger_foreground_ratio", 0.85))
    if not 0 <= trigger_ratio <= 1:
        raise ValueError("trigger_foreground_ratio must be in [0, 1]")
    overexpanded = (
        (scores >= score_threshold).float().flatten(1).mean(dim=1)
        >= trigger_ratio
    ).view(-1, 1, 1)
    if not overexpanded.any():
        return gate

    strength = float(settings.get("strength", 0.65))
    if not 0 <= strength <= 1:
        raise ValueError("overexpansion strength must be in [0, 1]")
    min_gate = float(settings.get("min_gate", 0.25))
    if not 0 <= min_gate <= 1:
        raise ValueError("overexpansion min_gate must be in [0, 1]")
    gamma = float(settings.get("gamma", 1.0))
    if gamma <= 0:
        raise ValueError("overexpansion gamma must be positive")

    strict_gate = min_gate + (1.0 - min_gate) * base_prior.pow(gamma)
    strict_gate = (1.0 - strength) + strength * strict_gate
    stricter_gate = torch.minimum(gate, strict_gate)
    return torch.where(overexpanded, stricter_gate, gate)


def image_difference_prior(
    t1: torch.Tensor,
    t2: torch.Tensor,
    low_quantile: float = 0.05,
    high_quantile: float = 0.95,
    blur_kernel: int = 3,
) -> torch.Tensor:
    """Build a full-resolution normalized image-difference prior."""
    if t1.shape != t2.shape or t1.ndim != 4:
        raise ValueError("t1 and t2 must have identical shape [B, C, H, W]")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("difference prior quantiles must satisfy 0 <= low < high <= 1")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer")

    diff = (t2.float() - t1.float()).abs().mean(dim=1)
    if blur_kernel > 1:
        padding = blur_kernel // 2
        diff = F.avg_pool2d(
            diff.unsqueeze(1),
            kernel_size=blur_kernel,
            stride=1,
            padding=padding,
        ).squeeze(1)
    flat = diff.flatten(1)
    low = torch.quantile(flat, low_quantile, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat, high_quantile, dim=1).view(-1, 1, 1)
    return ((diff - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def build_soft_pseudo_labels(
    scores: torch.Tensor,
    image_labels: torch.Tensor,
    image_probabilities: torch.Tensor,
    threshold: float,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert CAM scores into soft pseudo labels and confidence weights."""
    settings = dict(config or {})
    if scores.ndim != 3:
        raise ValueError("scores must have shape [B, H, W]")
    if image_labels.ndim != 1 or image_labels.shape[0] != scores.shape[0]:
        raise ValueError("image_labels must have shape [B]")
    if (
        image_probabilities.ndim != 1
        or image_probabilities.shape[0] != scores.shape[0]
    ):
        raise ValueError("image_probabilities must have shape [B]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    temperature = float(settings.get("temperature", 0.08))
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gamma = float(settings.get("gamma", 1.0))
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    min_confidence = float(settings.get("min_confidence", 0.2))
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    negative_confidence = float(settings.get("negative_confidence", 1.0))
    if not 0 <= negative_confidence <= 1:
        raise ValueError("negative_confidence must be in [0, 1]")
    probability_power = float(settings.get("probability_power", 1.0))
    if probability_power < 0:
        raise ValueError("probability_power must be non-negative")

    scores = scores.float().clamp(0.0, 1.0)
    image_labels = image_labels.float()
    image_probabilities = image_probabilities.float().clamp(0.0, 1.0)

    pseudo = torch.sigmoid((scores - threshold) / temperature).pow(gamma)
    confidence = (pseudo - 0.5).abs().mul(2.0).clamp(0.0, 1.0)
    confidence = min_confidence + (1.0 - min_confidence) * confidence

    positive = (image_labels > 0.5).view(-1, 1, 1)
    probability_gate = image_probabilities.view(-1, 1, 1).pow(probability_power)
    pseudo = torch.where(positive, pseudo * probability_gate, torch.zeros_like(pseudo))
    confidence = torch.where(
        positive,
        confidence * probability_gate,
        torch.full_like(confidence, negative_confidence),
    )
    return pseudo.clamp(0.0, 1.0), confidence.clamp(0.0, 1.0)


def pseudo_label_quality(
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Summarize one generated pseudo-label map."""
    if pseudo.shape != confidence.shape:
        raise ValueError("pseudo and confidence must have identical shapes")
    foreground = pseudo >= threshold
    return {
        "pseudo_mean": float(pseudo.mean()),
        "pseudo_max": float(pseudo.max()),
        "confidence_mean": float(confidence.mean()),
        "foreground_ratio": float(foreground.float().mean()),
    }


def save_grayscale_float_png(path: str | Path, values: torch.Tensor) -> None:
    """Save a [0, 1] tensor as an 8-bit grayscale PNG."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = values.detach().float().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    Image.fromarray(array, mode="L").save(destination)
