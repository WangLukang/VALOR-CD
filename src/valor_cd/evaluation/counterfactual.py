from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .candidate_calibration import extract_candidate_components


@dataclass(frozen=True)
class CounterfactualCandidate:
    sample_index: int
    y0: int
    x0: int
    y1: int
    x1: int
    area_pixels: int
    area_ratio: float
    mean_score: float
    peak_score: float


def counterfactual_replace_t2_with_t1(
    t1: torch.Tensor,
    t2: torch.Tensor,
    boxes: list[CounterfactualCandidate],
    padding: int = 0,
) -> torch.Tensor:
    """Create counterfactual pairs by restoring candidate T2 regions from T1."""
    if t1.shape != t2.shape:
        raise ValueError("T1 and T2 must have identical shapes")
    if t1.ndim != 4:
        raise ValueError("T1 and T2 must have shape [B, C, H, W]")
    _, _, height, width = t1.shape
    counterfactual = t2[[box.sample_index for box in boxes]].clone()
    for index, box in enumerate(boxes):
        y0 = max(0, int(box.y0) - padding)
        x0 = max(0, int(box.x0) - padding)
        y1 = min(height, int(box.y1) + padding)
        x1 = min(width, int(box.x1) + padding)
        counterfactual[index, :, y0:y1, x0:x1] = t1[
            box.sample_index,
            :,
            y0:y1,
            x0:x1,
        ]
    return counterfactual


def extract_counterfactual_candidates(
    scores: torch.Tensor,
    threshold: float,
    config: dict[str, Any] | None = None,
) -> list[CounterfactualCandidate]:
    """Extract connected CAM candidates with bounding boxes."""
    settings = dict(config or {})
    components = extract_candidate_components(scores, threshold, settings)
    min_area_ratio = float(settings.get("min_area_ratio", 0.0))
    max_candidates_per_image = int(settings.get("max_candidates_per_image", 3))
    if max_candidates_per_image <= 0:
        raise ValueError("max_candidates_per_image must be positive")

    candidates_by_sample: dict[int, list[CounterfactualCandidate]] = {}
    for component in components:
        area_ratio = float(component["area_ratio"])
        if area_ratio < min_area_ratio:
            continue
        box = _component_to_box(component)
        candidates_by_sample.setdefault(box.sample_index, []).append(box)

    selected: list[CounterfactualCandidate] = []
    for sample_candidates in candidates_by_sample.values():
        sample_candidates.sort(
            key=lambda item: (item.peak_score, item.mean_score, item.area_pixels),
            reverse=True,
        )
        selected.extend(sample_candidates[:max_candidates_per_image])
    selected.sort(key=lambda item: (item.sample_index, item.y0, item.x0))
    return selected


@torch.no_grad()
def verify_counterfactual_candidates(
    model: torch.nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    scores: torch.Tensor,
    base_probabilities: torch.Tensor,
    threshold: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep candidates whose T2->T1 restoration reduces image probability."""
    settings = dict(config or {})
    candidates = extract_counterfactual_candidates(scores, threshold, settings)
    if not candidates:
        return {
            "scores": scores,
            "verified_mask": torch.zeros_like(scores, dtype=torch.bool),
            "candidates": [],
        }

    drop_threshold = float(settings.get("drop_threshold", 0.03))
    relative_drop_threshold = float(settings.get("relative_drop_threshold", 0.05))
    padding = int(settings.get("padding", 4))
    mode = str(settings.get("mode", "verified_only"))
    unverified_scale = float(settings.get("unverified_scale", 0.9))
    if drop_threshold < 0:
        raise ValueError("drop_threshold must be non-negative")
    if relative_drop_threshold < 0:
        raise ValueError("relative_drop_threshold must be non-negative")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if mode not in {"verified_only", "soft_suppress"}:
        raise ValueError("counterfactual mode must be 'verified_only' or 'soft_suppress'")
    if not 0 <= unverified_scale <= 1:
        raise ValueError("unverified_scale must be in [0, 1]")

    counterfactual_t1 = t1[[box.sample_index for box in candidates]]
    counterfactual_t2 = counterfactual_replace_t2_with_t1(
        t1,
        t2,
        candidates,
        padding=padding,
    )
    output = model(counterfactual_t1, counterfactual_t2)
    counterfactual_probabilities = output["logits"].sigmoid().detach().float()

    verified_mask = torch.zeros_like(scores, dtype=torch.bool)
    active_mask = scores >= threshold
    verified_scores = scores.clone() if mode == "soft_suppress" else torch.zeros_like(scores)
    candidate_records: list[dict[str, float | int | bool]] = []
    for index, box in enumerate(candidates):
        base_probability = float(base_probabilities[box.sample_index])
        counterfactual_probability = float(counterfactual_probabilities[index])
        drop = base_probability - counterfactual_probability
        relative_drop = drop / max(base_probability, 1e-6)
        verified = (
            drop >= drop_threshold
            or relative_drop >= relative_drop_threshold
        )
        if verified:
            verified_mask[
                box.sample_index,
                box.y0 : box.y1,
                box.x0 : box.x1,
            ] |= active_mask[box.sample_index, box.y0 : box.y1, box.x0 : box.x1]
        elif mode == "soft_suppress":
            region = active_mask[box.sample_index, box.y0 : box.y1, box.x0 : box.x1]
            verified_scores[
                box.sample_index,
                box.y0 : box.y1,
                box.x0 : box.x1,
            ] = torch.where(
                region,
                verified_scores[
                    box.sample_index,
                    box.y0 : box.y1,
                    box.x0 : box.x1,
                ]
                * unverified_scale,
                verified_scores[
                    box.sample_index,
                    box.y0 : box.y1,
                    box.x0 : box.x1,
                ],
            )
        candidate_records.append(
            {
                "sample_index": box.sample_index,
                "y0": box.y0,
                "x0": box.x0,
                "y1": box.y1,
                "x1": box.x1,
                "area_pixels": box.area_pixels,
                "area_ratio": box.area_ratio,
                "mean_score": box.mean_score,
                "peak_score": box.peak_score,
                "base_probability": base_probability,
                "counterfactual_probability": counterfactual_probability,
                "probability_drop": drop,
                "relative_probability_drop": relative_drop,
                "verified": verified,
            }
        )

    if mode == "verified_only":
        verified_scores = scores * verified_mask.to(scores.dtype)
    else:
        verified_scores = torch.where(verified_mask, scores, verified_scores)
    return {
        "scores": verified_scores,
        "verified_mask": verified_mask,
        "candidates": candidate_records,
    }


def _component_to_box(component: dict[str, float | int]) -> CounterfactualCandidate:
    return CounterfactualCandidate(
        sample_index=int(component["sample_index"]),
        y0=int(component["y0"]),
        x0=int(component["x0"]),
        y1=int(component["y1"]),
        x1=int(component["x1"]),
        area_pixels=int(component["area_pixels"]),
        area_ratio=float(component["area_ratio"]),
        mean_score=float(component["mean_score"]),
        peak_score=float(component["peak_score"]),
    )
