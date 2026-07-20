from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def crf_like_refine(
    probabilities: torch.Tensor,
    guide: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Refine binary probabilities with a local bilateral mean-field update.

    This is a lightweight DenseCRF-style post-process implemented only with
    PyTorch. The unary term comes from the pseudo-label probability map, while
    pairwise messages are edge-aware through the temporal image-pair guide.
    """
    settings = dict(config or {})
    squeeze = probabilities.ndim == 2
    if squeeze:
        probabilities = probabilities.unsqueeze(0)
    if guide.ndim == 3:
        guide = guide.unsqueeze(0)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [B, H, W] or [H, W]")
    if guide.ndim != 4:
        raise ValueError("guide must have shape [B, C, H, W] or [C, H, W]")
    if probabilities.shape[0] != guide.shape[0] or probabilities.shape[-2:] != guide.shape[-2:]:
        raise ValueError("probabilities and guide must share batch and spatial dimensions")

    iterations = int(settings.get("iterations", 5))
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    radius = int(settings.get("radius", 3))
    if radius < 0:
        raise ValueError("radius must be non-negative")
    unary_weight = float(settings.get("unary_weight", 1.0))
    if unary_weight < 0:
        raise ValueError("unary_weight must be non-negative")
    bilateral_weight = float(settings.get("bilateral_weight", 1.0))
    if bilateral_weight < 0:
        raise ValueError("bilateral_weight must be non-negative")
    spatial_weight = float(settings.get("spatial_weight", 0.15))
    if spatial_weight < 0:
        raise ValueError("spatial_weight must be non-negative")
    spatial_sigma = float(settings.get("spatial_sigma", max(radius, 1)))
    if spatial_sigma <= 0:
        raise ValueError("spatial_sigma must be positive")
    color_sigma = float(settings.get("color_sigma", 0.12))
    if color_sigma <= 0:
        raise ValueError("color_sigma must be positive")

    probabilities = probabilities.float().clamp(1e-4, 1.0 - 1e-4)
    guide = guide.float().clamp(0.0, 1.0)
    unary = torch.logit(probabilities) * unary_weight
    q = probabilities
    for _ in range(iterations):
        message = q.new_zeros(q.shape)
        if bilateral_weight > 0 and radius > 0:
            message = message + bilateral_weight * (
                local_bilateral_filter(q, guide, radius, spatial_sigma, color_sigma) - 0.5
            )
        if spatial_weight > 0 and radius > 0:
            kernel_size = 2 * radius + 1
            spatial = F.avg_pool2d(
                q.unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=radius,
            ).squeeze(1)
            message = message + spatial_weight * (spatial - 0.5)
        q = torch.sigmoid(unary + 2.0 * message)
    return q.squeeze(0) if squeeze else q


def scribble_random_walk_refine(
    probabilities: torch.Tensor,
    guide: torch.Tensor,
    foreground_seed: torch.Tensor,
    background_seed: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Propagate foreground/background scribbles with edge-aware random walks.

    The high-confidence scribble pixels are clamped at every iteration. Unknown
    pixels are updated by local bilateral diffusion in the temporal guide space,
    with an optional weak anchor to the original soft pseudo label.
    """
    settings = dict(config or {})
    squeeze = probabilities.ndim == 2
    if squeeze:
        probabilities = probabilities.unsqueeze(0)
        foreground_seed = foreground_seed.unsqueeze(0)
        background_seed = background_seed.unsqueeze(0)
    if guide.ndim == 3:
        guide = guide.unsqueeze(0)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [B, H, W] or [H, W]")
    if foreground_seed.shape != probabilities.shape or background_seed.shape != probabilities.shape:
        raise ValueError("scribble seeds must have the same shape as probabilities")
    if guide.ndim != 4:
        raise ValueError("guide must have shape [B, C, H, W] or [C, H, W]")
    if probabilities.shape[0] != guide.shape[0] or probabilities.shape[-2:] != guide.shape[-2:]:
        raise ValueError("probabilities and guide must share batch and spatial dimensions")

    iterations = int(settings.get("iterations", 32))
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    radius = int(settings.get("radius", 3))
    if radius < 0:
        raise ValueError("radius must be non-negative")
    spatial_sigma = float(settings.get("spatial_sigma", max(radius, 1)))
    if spatial_sigma <= 0:
        raise ValueError("spatial_sigma must be positive")
    color_sigma = float(settings.get("color_sigma", 0.10))
    if color_sigma <= 0:
        raise ValueError("color_sigma must be positive")
    soft_anchor_weight = float(settings.get("soft_anchor_weight", 0.05))
    if not 0 <= soft_anchor_weight <= 1:
        raise ValueError("soft_anchor_weight must be in [0, 1]")
    initial_unknown = settings.get("initial_unknown", 0.5)
    if initial_unknown == "soft":
        initial_unknown_value: float | None = None
    else:
        initial_unknown_value = float(initial_unknown)
        if not 0 <= initial_unknown_value <= 1:
            raise ValueError("initial_unknown must be 'soft' or a value in [0, 1]")
    spatial_weight = float(settings.get("spatial_weight", 0.05))
    if not 0 <= spatial_weight <= 1:
        raise ValueError("spatial_weight must be in [0, 1]")
    sharpen_temperature = float(settings.get("sharpen_temperature", 0.0))
    if sharpen_temperature < 0:
        raise ValueError("sharpen_temperature must be non-negative")

    probabilities = probabilities.float().clamp(0.0, 1.0)
    guide = guide.float().clamp(0.0, 1.0)
    foreground_seed = foreground_seed.bool()
    background_seed = background_seed.bool() & ~foreground_seed
    unknown = ~(foreground_seed | background_seed)

    if initial_unknown_value is None:
        q = probabilities.clone()
    else:
        q = torch.full_like(probabilities, initial_unknown_value)
    q = torch.where(foreground_seed, torch.ones_like(q), q)
    q = torch.where(background_seed, torch.zeros_like(q), q)
    anchor = probabilities

    for _ in range(iterations):
        if radius > 0:
            bilateral = local_bilateral_filter(q, guide, radius, spatial_sigma, color_sigma)
        else:
            bilateral = q
        if spatial_weight > 0 and radius > 0:
            kernel_size = 2 * radius + 1
            spatial = F.avg_pool2d(
                q.unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=radius,
            ).squeeze(1)
            propagated = (1.0 - spatial_weight) * bilateral + spatial_weight * spatial
        else:
            propagated = bilateral
        propagated = (1.0 - soft_anchor_weight) * propagated + soft_anchor_weight * anchor
        q = torch.where(unknown, propagated, q)
        q = torch.where(foreground_seed, torch.ones_like(q), q)
        q = torch.where(background_seed, torch.zeros_like(q), q)

    if sharpen_temperature > 0:
        q = torch.sigmoid((q - 0.5) / sharpen_temperature)
        q = torch.where(foreground_seed, torch.ones_like(q), q)
        q = torch.where(background_seed, torch.zeros_like(q), q)
    q = q.clamp(0.0, 1.0)
    return q.squeeze(0) if squeeze else q


def local_bilateral_filter(
    values: torch.Tensor,
    guide: torch.Tensor,
    radius: int,
    spatial_sigma: float,
    color_sigma: float,
) -> torch.Tensor:
    """Apply a local bilateral filter to [B, H, W] values using [B, C, H, W] guide."""
    if values.ndim != 3:
        raise ValueError("values must have shape [B, H, W]")
    if guide.ndim != 4:
        raise ValueError("guide must have shape [B, C, H, W]")
    if values.shape[0] != guide.shape[0] or values.shape[-2:] != guide.shape[-2:]:
        raise ValueError("values and guide must share batch and spatial dimensions")
    if radius <= 0:
        return values

    batch_size, height, width = values.shape
    kernel_size = 2 * radius + 1
    neighbors = kernel_size * kernel_size
    sample_count = height * width

    value_patches = F.unfold(
        F.pad(values.unsqueeze(1), (radius, radius, radius, radius), mode="reflect"),
        kernel_size=kernel_size,
    ).view(batch_size, neighbors, sample_count)
    guide_patches = F.unfold(
        F.pad(guide, (radius, radius, radius, radius), mode="reflect"),
        kernel_size=kernel_size,
    ).view(batch_size, guide.shape[1], neighbors, sample_count)
    guide_center = guide.flatten(2).unsqueeze(2)

    color_distance = (guide_patches - guide_center).pow(2).sum(dim=1)
    color_weight = torch.exp(-color_distance / (2.0 * color_sigma * color_sigma))
    spatial_weight = _spatial_kernel(radius, spatial_sigma, values.device, values.dtype)
    weights = color_weight * spatial_weight.view(1, neighbors, 1)
    filtered = (weights * value_patches).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return filtered.view(batch_size, height, width)


def _spatial_kernel(
    radius: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    distance = xx.pow(2) + yy.pow(2)
    return torch.exp(-distance.flatten() / (2.0 * sigma * sigma))
