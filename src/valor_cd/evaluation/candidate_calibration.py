from __future__ import annotations

from typing import Any

import torch


def apply_candidate_calibration(
    scores: torch.Tensor,
    threshold: float,
    config: dict[str, Any] | None,
) -> torch.Tensor:
    """Filter negative-like connected components after CAM thresholding."""
    if not config or not bool(config.get("enabled", True)):
        return scores
    name = str(config.get("name", "none"))
    if name == "none":
        return scores
    if name == "negative_component_filter":
        return negative_component_filter(scores, threshold, config)
    if name == "fixed_component_filter":
        return fixed_component_filter(scores, threshold, config)
    raise ValueError(f"Unsupported candidate calibration: {name}")


def negative_component_filter(
    scores: torch.Tensor,
    threshold: float,
    config: dict[str, Any],
) -> torch.Tensor:
    """Remove connected components that look like calibrated negative clutter."""
    reject_area_ratio = config.get("reject_area_ratio")
    reject_mean_score = config.get("reject_mean_score")
    reject_peak_score = config.get("reject_peak_score")
    if (
        reject_area_ratio is None
        or reject_mean_score is None
        or reject_peak_score is None
    ):
        return scores

    original_shape_was_2d = scores.ndim == 2
    batched_scores = scores.unsqueeze(0) if original_shape_was_2d else scores
    if batched_scores.ndim != 3:
        raise ValueError("candidate calibration expects scores with shape [B, H, W]")
    if not 0 <= threshold <= 1:
        raise ValueError("candidate calibration threshold must be in [0, 1]")

    min_component_pixels = int(config.get("min_component_pixels", 4))
    if min_component_pixels <= 0:
        raise ValueError("min_component_pixels must be positive")
    connectivity = int(config.get("connectivity", 8))
    if connectivity not in {4, 8}:
        raise ValueError("candidate calibration connectivity must be 4 or 8")

    scores_cpu = batched_scores.detach().float().cpu()
    filtered = scores_cpu.clone()
    for sample_index, sample_scores in enumerate(scores_cpu):
        height, width = sample_scores.shape
        binary = sample_scores >= threshold
        for ys, xs in _iter_components(binary, connectivity):
            area_pixels = int(ys.numel())
            if area_pixels < min_component_pixels:
                continue
            values = sample_scores[ys, xs]
            area_ratio = area_pixels / float(height * width)
            if (
                area_ratio <= float(reject_area_ratio)
                and float(values.mean()) <= float(reject_mean_score)
                and float(values.max()) <= float(reject_peak_score)
            ):
                filtered[sample_index, ys, xs] = 0.0

    filtered = filtered.to(device=batched_scores.device, dtype=batched_scores.dtype)
    return filtered.squeeze(0) if original_shape_was_2d else filtered




def fixed_component_filter(
    scores: torch.Tensor,
    threshold: float,
    config: dict[str, Any],
) -> torch.Tensor:
    """Apply mask-free, prior-fixed component filtering.

    This filter does not estimate parameters from validation masks. It only
    removes connected components that violate fixed geometric/score priors.
    """
    original_shape_was_2d = scores.ndim == 2
    batched_scores = scores.unsqueeze(0) if original_shape_was_2d else scores
    if batched_scores.ndim != 3:
        raise ValueError("fixed component filter expects scores with shape [B, H, W]")
    if not 0 <= threshold <= 1:
        raise ValueError("fixed component filter threshold must be in [0, 1]")

    min_component_pixels = int(config.get("min_component_pixels", 1))
    if min_component_pixels < 1:
        raise ValueError("min_component_pixels must be positive")
    connectivity = int(config.get("connectivity", 8))
    if connectivity not in {4, 8}:
        raise ValueError("fixed component filter connectivity must be 4 or 8")
    max_area_ratio = config.get("max_area_ratio")
    min_mean_score = config.get("min_mean_score")
    min_peak_score = config.get("min_peak_score")

    scores_cpu = batched_scores.detach().float().cpu()
    filtered = scores_cpu.clone()
    for sample_index, sample_scores in enumerate(scores_cpu):
        height, width = sample_scores.shape
        binary = sample_scores >= threshold
        for ys, xs in _iter_components(binary, connectivity):
            area_pixels = int(ys.numel())
            values = sample_scores[ys, xs]
            area_ratio = area_pixels / float(height * width)
            remove = area_pixels < min_component_pixels
            if max_area_ratio is not None and area_ratio > float(max_area_ratio):
                remove = True
            if min_mean_score is not None and float(values.mean()) < float(min_mean_score):
                remove = True
            if min_peak_score is not None and float(values.max()) < float(min_peak_score):
                remove = True
            if remove:
                filtered[sample_index, ys, xs] = 0.0

    filtered = filtered.to(device=batched_scores.device, dtype=batched_scores.dtype)
    return filtered.squeeze(0) if original_shape_was_2d else filtered

def extract_candidate_statistics(
    scores: torch.Tensor,
    threshold: float,
    config: dict[str, Any] | None = None,
) -> list[dict[str, float | int]]:
    """Collect connected-component statistics without using pixel labels."""
    return [
        {
            key: value
            for key, value in component.items()
            if key in {"area_pixels", "area_ratio", "mean_score", "peak_score"}
        }
        for component in extract_candidate_components(
            scores,
            threshold,
            config=config,
        )
    ]


def extract_candidate_components(
    scores: torch.Tensor,
    threshold: float,
    config: dict[str, Any] | None = None,
    targets: torch.Tensor | None = None,
) -> list[dict[str, float | int]]:
    """Collect connected-component stats and optional target overlaps."""
    settings = config or {}
    original_shape_was_2d = scores.ndim == 2
    batched_scores = scores.unsqueeze(0) if original_shape_was_2d else scores
    if batched_scores.ndim != 3:
        raise ValueError("candidate components expect scores with shape [B, H, W]")
    if not 0 <= threshold <= 1:
        raise ValueError("candidate components threshold must be in [0, 1]")
    if targets is not None:
        batched_targets = targets.unsqueeze(0) if targets.ndim == 2 else targets
        if batched_targets.shape != batched_scores.shape:
            raise ValueError("targets must have the same shape as scores")
        batched_targets = batched_targets.detach().cpu() > 0.5
    else:
        batched_targets = None

    min_component_pixels = int(settings.get("min_component_pixels", 4))
    if min_component_pixels <= 0:
        raise ValueError("min_component_pixels must be positive")
    connectivity = int(settings.get("connectivity", 8))
    if connectivity not in {4, 8}:
        raise ValueError("candidate calibration connectivity must be 4 or 8")

    components: list[dict[str, float | int]] = []
    scores_cpu = batched_scores.detach().float().cpu()
    for sample_index, sample_scores in enumerate(scores_cpu):
        height, width = sample_scores.shape
        binary = sample_scores >= threshold
        for ys, xs in _iter_components(binary, connectivity):
            area_pixels = int(ys.numel())
            if area_pixels < min_component_pixels:
                continue
            values = sample_scores[ys, xs]
            component: dict[str, float | int] = {
                "sample_index": sample_index,
                "y0": int(ys.min()),
                "x0": int(xs.min()),
                "y1": int(ys.max()) + 1,
                "x1": int(xs.max()) + 1,
                "area_pixels": area_pixels,
                "area_ratio": area_pixels / float(height * width),
                "mean_score": float(values.mean()),
                "peak_score": float(values.max()),
            }
            if batched_targets is not None:
                target_values = batched_targets[sample_index, ys, xs]
                true_positive_pixels = int(target_values.sum())
                component["tp_pixels"] = true_positive_pixels
                component["fp_pixels"] = area_pixels - true_positive_pixels
            components.append(component)
    return components


def build_negative_component_filter_config(
    statistics: list[dict[str, float | int]],
    base_config: dict[str, Any],
    pixel_threshold: float,
    negative_images: int,
) -> dict[str, Any]:
    """Estimate component rejection thresholds from image-level negatives."""
    config = dict(base_config)
    config["name"] = "negative_component_filter"
    config["pixel_threshold"] = float(pixel_threshold)
    config["negative_images"] = int(negative_images)
    config["negative_components"] = len(statistics)
    if not statistics:
        config["enabled"] = False
        return config

    area_quantile = _checked_quantile(config, "area_quantile", 0.95)
    mean_quantile = _checked_quantile(config, "mean_score_quantile", 0.75)
    peak_quantile = _checked_quantile(config, "peak_score_quantile", 0.75)

    reject_area_ratio = _stat_quantile(statistics, "area_ratio", area_quantile)
    max_reject_area_ratio = config.get("max_reject_area_ratio")
    if max_reject_area_ratio is not None:
        reject_area_ratio = min(reject_area_ratio, float(max_reject_area_ratio))

    config["enabled"] = True
    config["reject_area_ratio"] = reject_area_ratio
    config["reject_mean_score"] = _stat_quantile(
        statistics,
        "mean_score",
        mean_quantile,
    )
    config["reject_peak_score"] = _stat_quantile(
        statistics,
        "peak_score",
        peak_quantile,
    )
    return config


def _checked_quantile(
    config: dict[str, Any],
    name: str,
    default: float,
) -> float:
    value = float(config.get(name, default))
    if not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1)")
    return value


def _stat_quantile(
    statistics: list[dict[str, float | int]],
    key: str,
    quantile: float,
) -> float:
    values = torch.tensor([float(item[key]) for item in statistics], dtype=torch.float32)
    return float(torch.quantile(values, quantile))


def _iter_components(
    mask: torch.Tensor,
    connectivity: int,
):
    if mask.ndim != 2:
        raise ValueError("connected components expect a 2D mask")
    height, width = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    offsets = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 4
        else [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    )
    for y0, x0 in torch.nonzero(mask, as_tuple=False).tolist():
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        ys: list[int] = []
        xs: list[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and bool(mask[ny, nx])
                    and not bool(visited[ny, nx])
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        yield torch.tensor(ys, dtype=torch.long), torch.tensor(xs, dtype=torch.long)
