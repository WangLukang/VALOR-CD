from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from valor_cd.losses import local_matching_prior


def apply_score_calibration(
    scores: torch.Tensor,
    output: dict[str, Any],
    output_size: tuple[int, int],
    config: dict[str, Any] | None,
) -> torch.Tensor:
    """Apply optional CAM score calibration at inference/validation time."""
    if not config:
        return scores
    name = str(config.get("name", "none"))
    if name == "none":
        return scores
    if name == "negative_interference":
        return negative_interference_calibration(scores, output, output_size, config)
    raise ValueError(f"Unsupported score calibration: {name}")


def negative_interference_calibration(
    scores: torch.Tensor,
    output: dict[str, Any],
    output_size: tuple[int, int],
    config: dict[str, Any],
) -> torch.Tensor:
    """Suppress high-residual nuisance regions when image evidence is negative.

    The gate depends on the image-level probability rather than the ground-truth
    label, so this remains valid for test-time use.
    """
    for key in ("feature_t1", "feature_t2", "logits"):
        if key not in output:
            raise KeyError(f"negative_interference calibration requires '{key}'")
    strength = float(config.get("strength", 0.5))
    if strength < 0:
        raise ValueError("score calibration strength must be non-negative")
    probability_threshold = float(config.get("probability_threshold", 0.5))
    if not 0 < probability_threshold <= 1:
        raise ValueError("probability_threshold must be in (0, 1]")
    probability_power = float(config.get("probability_power", 1.0))
    if probability_power < 0:
        raise ValueError("probability_power must be non-negative")
    prior_quantile = float(config.get("prior_quantile", 0.8))
    if not 0 < prior_quantile < 1:
        raise ValueError("prior_quantile must be in (0, 1)")
    mode = str(config.get("mode", "multiply"))

    prior = local_matching_prior(
        output["feature_t1"].detach(),
        output["feature_t2"].detach(),
        local_radius=int(config.get("local_radius", 2)),
    )
    prior = _robust_normalize(
        prior,
        low_quantile=float(config.get("robust_low_quantile", 0.05)),
        high_quantile=float(config.get("robust_high_quantile", 0.95)),
    )
    interference = _high_prior_interference(prior, prior_quantile)
    interference = F.interpolate(
        interference.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    probability = output["logits"].sigmoid().detach().float().view(-1, 1, 1)
    gate = ((probability_threshold - probability) / probability_threshold).clamp(0, 1)
    if probability_power != 1:
        gate = gate.pow(probability_power)

    if mode == "multiply":
        calibrated = scores * (1.0 - strength * gate * interference).clamp_min(0.0)
    elif mode == "subtract":
        calibrated = scores - strength * gate * interference
    else:
        raise ValueError("score calibration mode must be 'multiply' or 'subtract'")
    return calibrated.clamp(0.0, 1.0)


def _robust_normalize(
    score: torch.Tensor,
    low_quantile: float,
    high_quantile: float,
) -> torch.Tensor:
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("robust quantiles must satisfy 0 <= low < high <= 1")
    flat = score.flatten(1)
    low = torch.quantile(flat, low_quantile, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat, high_quantile, dim=1).view(-1, 1, 1)
    return ((score - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def _high_prior_interference(
    prior: torch.Tensor,
    prior_quantile: float,
) -> torch.Tensor:
    flat = prior.flatten(1)
    threshold = torch.quantile(flat, prior_quantile, dim=1).view(-1, 1, 1)
    return ((prior - threshold) / (1.0 - threshold).clamp_min(1e-6)).clamp(0.0, 1.0)
