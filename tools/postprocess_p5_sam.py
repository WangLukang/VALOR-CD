from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.experiment import choose_device, resolve_from, write_json  # noqa: E402


@dataclass(frozen=True)
class CandidateRegion:
    component_id: int
    mask: np.ndarray
    box_xyxy: np.ndarray
    area: int
    mean_probability: float
    peak_probability: float


@dataclass(frozen=True)
class SamSelection:
    mask: np.ndarray
    sam_score: float
    combined_score: float
    mean_probability: float
    pseudo_precision: float
    component_coverage: float
    area_expansion: float


@dataclass(frozen=True)
class SeedGrowPolicy:
    mode: str
    candidate_threshold: float
    fg_point_threshold: float
    min_component_coverage: float
    allow_topk_seed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process P5 pseudo labels with P5-prompted SAM."
    )
    parser.add_argument("--source-dir", default=str(ROOT / "outputs" / "p5_soft_pseudo_labels"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "p5_sam_pseudo_labels_t050"))
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "sam" / "sam_vit_h_4b8939.pth"))
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_h")
    parser.add_argument(
        "--sam-family",
        choices=("sam",),
        default="sam",
        help="Promptable mask backend used by the public pipeline.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("t2", "t2_diff_t1", "best_t1_t2"),
        default="t2",
        help=(
            "Image used by SAM. best_t1_t2 runs SAM on both original T1/T2 images "
            "and chooses the better mask per candidate region."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--seed-threshold",
        type=float,
        help="Core seed threshold saved for visualization. Defaults to --fg-point-threshold.",
    )
    parser.add_argument("--fg-point-threshold", type=float, default=0.85)
    parser.add_argument("--bg-point-threshold", type=float, default=0.2)
    parser.add_argument("--require-core-seed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--positive-points", type=int, default=3)
    parser.add_argument("--negative-points", type=int, default=4)
    parser.add_argument("--min-region-pixels", type=int, default=8)
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--box-pad-pixels", type=int, default=6)
    parser.add_argument("--box-pad-ratio", type=float, default=0.10)
    parser.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-sam-score", type=float, default=0.0)
    parser.add_argument("--min-component-coverage", type=float, default=0.20)
    parser.add_argument("--min-pseudo-precision", type=float, default=0.12)
    parser.add_argument("--max-area-expansion", type=float, default=8.0)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.75)
    parser.add_argument("--adaptive-seed-grow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-large-area-ratio", type=float, default=0.05)
    parser.add_argument("--adaptive-huge-area-ratio", type=float, default=0.20)
    parser.add_argument("--adaptive-large-candidate-threshold", type=float, default=0.40)
    parser.add_argument("--adaptive-large-seed-threshold", type=float, default=0.55)
    parser.add_argument("--adaptive-huge-candidate-threshold", type=float, default=0.35)
    parser.add_argument("--adaptive-huge-seed-threshold", type=float, default=0.50)
    parser.add_argument("--adaptive-huge-min-mean-score", type=float, default=0.45)
    parser.add_argument("--adaptive-huge-min-peak-score", type=float, default=0.55)
    parser.add_argument("--adaptive-huge-min-component-coverage", type=float, default=0.30)
    parser.add_argument("--adaptive-huge-allow-topk-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-kernel", type=int, default=0)
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_rows(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]], Path]:
    summary_path = source_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"P5 summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = str(summary.get("split", "train"))
    manifest_path = source_dir / f"{split}_pseudo_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"P5 manifest does not exist: {manifest_path}")
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8-sig", newline="")))
    return summary, rows, manifest_path


def load_probability(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_rgb_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)



def load_gray_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def robust_uint8(values: np.ndarray, low_quantile: float = 0.02, high_quantile: float = 0.98) -> np.ndarray:
    values = values.astype(np.float32)
    low = float(np.quantile(values, low_quantile))
    high = float(np.quantile(values, high_quantile))
    if high <= low + 1e-6:
        scaled = np.zeros_like(values, dtype=np.float32)
    else:
        scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).round().astype(np.uint8)


def build_sam_input_image(data_root: Path, row: dict[str, str], image_mode: str) -> np.ndarray:
    if image_mode == "t2":
        return load_rgb_uint8(data_root / row["t2"])
    if image_mode == "t2_diff_t1":
        t1 = load_gray_float(data_root / row["t1"])
        t2 = load_gray_float(data_root / row["t2"])
        if t1.shape != t2.shape:
            raise ValueError(f"T1/T2 shape mismatch for {row['id']}: {t1.shape} vs {t2.shape}")
        diff = np.abs(t2 - t1)
        return np.stack(
            (
                robust_uint8(t2, 0.01, 0.99),
                robust_uint8(diff, 0.02, 0.98),
                robust_uint8(t1, 0.01, 0.99),
            ),
            axis=-1,
        )
    raise ValueError(f"Unsupported SAM image mode: {image_mode}")

def load_sam_backend(sam_family: str):
    if sam_family != "sam":
        raise ValueError(f"Unsupported SAM family: {sam_family}")
    try:
        module = importlib.import_module("segment_anything")
    except ModuleNotFoundError as exc:
        detail = str(exc)
    else:
        if hasattr(module, "SamPredictor") and hasattr(module, "sam_model_registry"):
            return module.SamPredictor, module.sam_model_registry, "segment_anything"
        detail = "module is missing SamPredictor or sam_model_registry"
    raise RuntimeError(
        "SAM post-processing requires the official segment-anything package. "
        "Install it with: python -m pip install "
        "git+https://github.com/facebookresearch/segment-anything.git. "
        f"Import detail: {detail}"
    )


def load_sam_predictor(
    checkpoint: Path,
    model_type: str,
    sam_family: str,
    device: torch.device,
):
    SamPredictor, sam_model_registry, backend_name = load_sam_backend(sam_family)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "SAM checkpoint not found. Place the matching checkpoint here: "
            f"{checkpoint}"
        )
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")
    try:
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load the SAM checkpoint with backend {backend_name}. "
            "Check that the model type matches the checkpoint."
        ) from exc
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam), backend_name


def expand_box(
    x: int,
    y: int,
    width: int,
    height: int,
    shape: tuple[int, int],
    *,
    pad_pixels: int,
    pad_ratio: float,
) -> np.ndarray:
    image_height, image_width = shape
    pad = max(int(pad_pixels), int(round(max(width, height) * pad_ratio)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_width - 1, x + width - 1 + pad)
    y1 = min(image_height - 1, y + height - 1 + pad)
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def extract_candidate_regions(
    probabilities: np.ndarray,
    *,
    threshold: float,
    min_region_pixels: int,
    max_regions: int,
    box_pad_pixels: int,
    box_pad_ratio: float,
) -> list[CandidateRegion]:
    binary = (probabilities >= threshold).astype(np.uint8)
    if not binary.any():
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: list[CandidateRegion] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_region_pixels:
            continue
        mask = labels == component_id
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        box = expand_box(
            x,
            y,
            width,
            height,
            probabilities.shape,
            pad_pixels=box_pad_pixels,
            pad_ratio=box_pad_ratio,
        )
        region_scores = probabilities[mask]
        regions.append(
            CandidateRegion(
                component_id=component_id,
                mask=mask,
                box_xyxy=box,
                area=area,
                mean_probability=float(region_scores.mean()),
                peak_probability=float(region_scores.max()),
            )
        )
    regions.sort(key=lambda region: (region.mean_probability, region.area), reverse=True)
    if max_regions > 0:
        regions = regions[:max_regions]
    return regions


def select_spread_points(
    ys: np.ndarray,
    xs: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    prefer_high: bool,
    min_distance: float = 8.0,
) -> list[tuple[float, float]]:
    if count <= 0 or len(xs) == 0:
        return []
    order = np.argsort(scores)
    if prefer_high:
        order = order[::-1]
    selected: list[tuple[float, float]] = []
    for index in order:
        x = float(xs[index])
        y = float(ys[index])
        if all((x - px) ** 2 + (y - py) ** 2 >= min_distance**2 for px, py in selected):
            selected.append((x, y))
        if len(selected) >= count:
            break
    if not selected:
        best = int(order[0])
        selected.append((float(xs[best]), float(ys[best])))
    return selected


def region_has_core_seed(
    probabilities: np.ndarray,
    region: CandidateRegion,
    *,
    fg_point_threshold: float,
) -> bool:
    return bool(np.any(region.mask & (probabilities >= fg_point_threshold)))


def candidate_extraction_threshold(args: argparse.Namespace) -> float:
    if not args.adaptive_seed_grow:
        return float(args.threshold)
    return min(
        float(args.threshold),
        float(args.adaptive_large_candidate_threshold),
        float(args.adaptive_huge_candidate_threshold),
    )


def seed_grow_policy(args: argparse.Namespace, region: CandidateRegion, image_area: int) -> SeedGrowPolicy:
    area_ratio = region.area / max(int(image_area), 1)
    if args.adaptive_seed_grow and area_ratio >= args.adaptive_huge_area_ratio:
        allow_topk = bool(args.adaptive_huge_allow_topk_seed)
        return SeedGrowPolicy(
            mode="huge",
            candidate_threshold=float(args.adaptive_huge_candidate_threshold),
            fg_point_threshold=float(args.adaptive_huge_seed_threshold),
            min_component_coverage=float(args.adaptive_huge_min_component_coverage),
            allow_topk_seed=allow_topk,
        )
    if args.adaptive_seed_grow and area_ratio >= args.adaptive_large_area_ratio:
        return SeedGrowPolicy(
            mode="large",
            candidate_threshold=float(args.adaptive_large_candidate_threshold),
            fg_point_threshold=float(args.adaptive_large_seed_threshold),
            min_component_coverage=float(args.min_component_coverage),
            allow_topk_seed=False,
        )
    return SeedGrowPolicy(
        mode="base",
        candidate_threshold=float(args.threshold),
        fg_point_threshold=float(args.fg_point_threshold),
        min_component_coverage=float(args.min_component_coverage),
        allow_topk_seed=False,
    )


def seed_gate(
    args: argparse.Namespace,
    probabilities: np.ndarray,
    region: CandidateRegion,
    policy: SeedGrowPolicy,
) -> tuple[bool, bool, bool]:
    has_core = region_has_core_seed(
        probabilities,
        region,
        fg_point_threshold=policy.fg_point_threshold,
    )
    uses_topk_seed = bool(args.require_core_seed and not has_core and policy.allow_topk_seed)
    allowed = bool((not args.require_core_seed) or has_core or uses_topk_seed)
    return allowed, has_core, uses_topk_seed


def empty_sam_stats(region_count: int = 0) -> dict[str, float | int]:
    return {
        "sam_regions": region_count,
        "sam_accepted_regions": 0,
        "sam_seed_fallback_regions": 0,
        "sam_mean_selection_score": 0.0,
        "sam_t1_selected_regions": 0,
        "sam_t2_selected_regions": 0,
        "sam_adaptive_large_regions": 0,
        "sam_adaptive_huge_regions": 0,
        "sam_adaptive_topk_seed_regions": 0,
        "sam_adaptive_skipped_no_core_regions": 0,
    }


def update_policy_stats(
    stats: dict[str, float | int],
    policy: SeedGrowPolicy,
    *,
    allowed: bool,
    uses_topk_seed: bool,
) -> None:
    if policy.mode == "large":
        stats["sam_adaptive_large_regions"] = int(stats["sam_adaptive_large_regions"]) + 1
    elif policy.mode == "huge":
        stats["sam_adaptive_huge_regions"] = int(stats["sam_adaptive_huge_regions"]) + 1
    if uses_topk_seed:
        stats["sam_adaptive_topk_seed_regions"] = int(stats["sam_adaptive_topk_seed_regions"]) + 1
    if not allowed:
        stats["sam_adaptive_skipped_no_core_regions"] = int(stats["sam_adaptive_skipped_no_core_regions"]) + 1


def build_seed_mask(
    probabilities: np.ndarray,
    regions: list[CandidateRegion],
    args: argparse.Namespace,
    *,
    default_seed_threshold: float,
) -> np.ndarray:
    if not args.adaptive_seed_grow:
        return (probabilities >= default_seed_threshold).astype(np.uint8)
    seed_mask = np.zeros(probabilities.shape, dtype=np.uint8)
    image_area = int(probabilities.size)
    for region in regions:
        policy = seed_grow_policy(args, region, image_area)
        seed_mask[region.mask & (probabilities >= policy.fg_point_threshold)] = 1
    return seed_mask


def sample_positive_points(
    probabilities: np.ndarray,
    region: CandidateRegion,
    *,
    count: int,
    fg_point_threshold: float,
) -> list[tuple[float, float]]:
    candidate = region.mask & (probabilities >= fg_point_threshold)
    if not candidate.any():
        candidate = region.mask
    ys, xs = np.where(candidate)
    scores = probabilities[ys, xs]
    return select_spread_points(ys, xs, scores, count=count, prefer_high=True)


def sample_negative_points(
    probabilities: np.ndarray,
    region: CandidateRegion,
    *,
    count: int,
    bg_point_threshold: float,
) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    x0, y0, x1, y1 = region.box_xyxy.astype(int).tolist()
    candidate = np.zeros_like(probabilities, dtype=bool)
    candidate[y0 : y1 + 1, x0 : x1 + 1] = True
    dilated_region = cv2.dilate(region.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    candidate &= ~dilated_region
    candidate &= probabilities <= bg_point_threshold
    if not candidate.any():
        candidate = probabilities <= float(np.quantile(probabilities, 0.10))
        candidate &= ~region.mask
    ys, xs = np.where(candidate)
    scores = probabilities[ys, xs]
    return select_spread_points(ys, xs, scores, count=count, prefer_high=False)


def build_prompt(
    probabilities: np.ndarray,
    region: CandidateRegion,
    *,
    positive_points: int,
    negative_points: int,
    fg_point_threshold: float,
    bg_point_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    positives = sample_positive_points(
        probabilities,
        region,
        count=positive_points,
        fg_point_threshold=fg_point_threshold,
    )
    negatives = sample_negative_points(
        probabilities,
        region,
        count=negative_points,
        bg_point_threshold=bg_point_threshold,
    )
    points = positives + negatives
    if not points:
        return None, None, region.box_xyxy
    labels = [1] * len(positives) + [0] * len(negatives)
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        region.box_xyxy,
    )


def score_candidate_mask(
    mask: np.ndarray,
    probabilities: np.ndarray,
    region: CandidateRegion,
    *,
    threshold: float,
    sam_score: float,
) -> SamSelection | None:
    mask = mask.astype(bool)
    area = int(mask.sum())
    if area == 0:
        return None
    image_area = int(mask.size)
    area_ratio = area / max(image_area, 1)
    component_overlap = int(np.logical_and(mask, region.mask).sum())
    component_coverage = component_overlap / max(region.area, 1)
    pseudo_binary = probabilities >= threshold
    pseudo_precision = int(np.logical_and(mask, pseudo_binary).sum()) / max(area, 1)
    mean_probability = float(probabilities[mask].mean())
    area_expansion = area / max(region.area, 1)
    expansion_penalty = max(0.0, area_expansion - 1.0) / max(area_expansion, 1.0)
    combined_score = (
        0.35 * float(sam_score)
        + 0.30 * mean_probability
        + 0.20 * pseudo_precision
        + 0.15 * component_coverage
        - 0.10 * expansion_penalty
    )
    return SamSelection(
        mask=mask,
        sam_score=float(sam_score),
        combined_score=float(combined_score),
        mean_probability=mean_probability,
        pseudo_precision=float(pseudo_precision),
        component_coverage=float(component_coverage),
        area_expansion=float(area_expansion),
    )


def choose_sam_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    region: CandidateRegion,
    *,
    threshold: float,
    min_sam_score: float,
    min_component_coverage: float,
    min_pseudo_precision: float,
    max_area_expansion: float,
    max_mask_area_ratio: float,
) -> SamSelection | None:
    selections: list[SamSelection] = []
    for mask, score in zip(masks, scores):
        selection = score_candidate_mask(
            mask,
            probabilities,
            region,
            threshold=threshold,
            sam_score=float(score),
        )
        if selection is None:
            continue
        if selection.sam_score < min_sam_score:
            continue
        if selection.component_coverage < min_component_coverage:
            continue
        if selection.pseudo_precision < min_pseudo_precision:
            continue
        if selection.area_expansion > max_area_expansion:
            continue
        if float(selection.mask.mean()) > max_mask_area_ratio:
            continue
        selections.append(selection)
    if not selections:
        return None
    return max(selections, key=lambda item: item.combined_score)



def predict_region_selection(
    predictor: Any,
    probabilities: np.ndarray,
    region: CandidateRegion,
    args: argparse.Namespace,
    policy: SeedGrowPolicy,
) -> SamSelection | None:
    point_coords, point_labels, box = build_prompt(
        probabilities,
        region,
        positive_points=args.positive_points,
        negative_points=args.negative_points,
        fg_point_threshold=policy.fg_point_threshold,
        bg_point_threshold=args.bg_point_threshold,
    )
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=args.multimask_output,
    )
    return choose_sam_mask(
        masks,
        scores,
        probabilities,
        region,
        threshold=policy.candidate_threshold,
        min_sam_score=args.min_sam_score,
        min_component_coverage=policy.min_component_coverage,
        min_pseudo_precision=args.min_pseudo_precision,
        max_area_expansion=args.max_area_expansion,
        max_mask_area_ratio=args.max_mask_area_ratio,
    )

def clean_binary_mask(
    mask: np.ndarray,
    *,
    open_kernel: int,
    close_kernel: int,
    min_component_pixels: int,
) -> np.ndarray:
    output = mask.astype(np.uint8)
    if open_kernel > 1:
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    if close_kernel > 1:
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
    if min_component_pixels > 1 and output.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(output, connectivity=8)
        cleaned = np.zeros_like(output)
        for component in range(1, count):
            if int(stats[component, cv2.CC_STAT_AREA]) >= min_component_pixels:
                cleaned[labels == component] = 1
        output = cleaned
    return output


def run_sam_on_image(
    predictor: Any,
    image_rgb: np.ndarray,
    probabilities: np.ndarray,
    regions: list[CandidateRegion],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int]]:
    predictor.set_image(image_rgb)
    output = np.zeros(probabilities.shape, dtype=np.uint8)
    accepted = 0
    fallback = 0
    score_sum = 0.0
    image_area = int(probabilities.size)
    stats = empty_sam_stats(len(regions))
    for region in regions:
        policy = seed_grow_policy(args, region, image_area)
        allowed, _has_core, uses_topk_seed = seed_gate(args, probabilities, region, policy)
        update_policy_stats(stats, policy, allowed=allowed, uses_topk_seed=uses_topk_seed)
        if not allowed:
            continue
        selection = predict_region_selection(predictor, probabilities, region, args, policy)
        if selection is None:
            if args.seed_fallback:
                output[region.mask] = 1
                fallback += 1
            continue
        output[selection.mask] = 1
        accepted += 1
        score_sum += selection.combined_score
    output = clean_binary_mask(
        output,
        open_kernel=args.open_kernel,
        close_kernel=args.close_kernel,
        min_component_pixels=args.min_component_pixels,
    )
    stats.update(
        {
            "sam_accepted_regions": accepted,
            "sam_seed_fallback_regions": fallback,
            "sam_mean_selection_score": score_sum / max(accepted, 1),
        }
    )
    return output, stats


def run_sam_on_t1_t2_best(
    predictor: Any,
    t1_rgb: np.ndarray,
    t2_rgb: np.ndarray,
    probabilities: np.ndarray,
    regions: list[CandidateRegion],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int]]:
    image_area = int(probabilities.size)
    policies = [seed_grow_policy(args, region, image_area) for region in regions]
    gates = [seed_gate(args, probabilities, region, policy) for region, policy in zip(regions, policies)]
    selections_by_region: list[list[tuple[str, SamSelection]]] = [[] for _ in regions]
    for source_name, image_rgb in (("t1", t1_rgb), ("t2", t2_rgb)):
        predictor.set_image(image_rgb)
        for index, (region, policy, gate) in enumerate(zip(regions, policies, gates)):
            allowed, _has_core, _uses_topk_seed = gate
            if not allowed:
                continue
            selection = predict_region_selection(predictor, probabilities, region, args, policy)
            if selection is not None:
                selections_by_region[index].append((source_name, selection))

    output = np.zeros(probabilities.shape, dtype=np.uint8)
    accepted = 0
    fallback = 0
    score_sum = 0.0
    t1_selected = 0
    t2_selected = 0
    stats = empty_sam_stats(len(regions))
    for region, policy, gate, selections in zip(regions, policies, gates, selections_by_region):
        allowed, _has_core, uses_topk_seed = gate
        update_policy_stats(stats, policy, allowed=allowed, uses_topk_seed=uses_topk_seed)
        if not allowed:
            continue
        if not selections:
            if args.seed_fallback:
                output[region.mask] = 1
                fallback += 1
            continue
        source_name, selection = max(selections, key=lambda item: item[1].combined_score)
        output[selection.mask] = 1
        accepted += 1
        score_sum += selection.combined_score
        if source_name == "t1":
            t1_selected += 1
        elif source_name == "t2":
            t2_selected += 1

    output = clean_binary_mask(
        output,
        open_kernel=args.open_kernel,
        close_kernel=args.close_kernel,
        min_component_pixels=args.min_component_pixels,
    )
    stats.update(
        {
            "sam_accepted_regions": accepted,
            "sam_seed_fallback_regions": fallback,
            "sam_mean_selection_score": score_sum / max(accepted, 1),
            "sam_t1_selected_regions": t1_selected,
            "sam_t2_selected_regions": t2_selected,
        }
    )
    return output, stats

def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8).clip(0, 1) * 255, mode="L").save(path)


def relative_to_output(path: Path, output_dir: Path) -> str:
    return str(path.relative_to(output_dir)).replace("\\", "/")


def overlay_mask(image: Image.Image, mask_path: Path, size: int) -> Image.Image:
    base = np.asarray(image.resize((size, size))).astype(np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L").resize((size, size)), dtype=np.float32) / 255.0
    alpha = 0.55 * mask[..., None]
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    output = base * (1.0 - alpha) + red * alpha
    return Image.fromarray((output.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def write_preview(
    rows: list[dict[str, str]],
    *,
    source_dir: Path,
    output_dir: Path,
    data_root: Path,
    count: int,
) -> Path | None:
    positives = [row for row in rows if int(float(row["label"])) == 1]
    if not positives or count <= 0:
        return None
    positives.sort(key=lambda row: float(row["sam_foreground_ratio"]), reverse=True)
    preview_rows = positives[:count]
    cell = 224
    label_height = 28
    columns = ("T1", "T2", "soft P5", "seed", "SAM binary", "T2 overlay")
    canvas = Image.new("RGB", (cell * len(columns), (cell + label_height) * len(preview_rows)), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, row in enumerate(preview_rows):
        y = row_index * (cell + label_height)
        t1 = Image.open(data_root / row["t1"]).convert("RGB").resize((cell, cell))
        t2 = Image.open(data_root / row["t2"]).convert("RGB").resize((cell, cell))
        soft = Image.open(source_dir / row["source_soft_pseudo_label"]).convert("L").resize((cell, cell)).convert("RGB")
        seed = Image.open(output_dir / row["sam_seed_label"]).convert("L").resize((cell, cell)).convert("RGB")
        binary_path = output_dir / row["pseudo_label"]
        binary = Image.open(binary_path).convert("L").resize((cell, cell)).convert("RGB")
        overlay = overlay_mask(t2, binary_path, cell)
        for column_index, image in enumerate((t1, t2, soft, seed, binary, overlay)):
            x = column_index * cell
            canvas.paste(image, (x, y + label_height))
            title = columns[column_index]
            if column_index == 0:
                title = f"{row['id']} | fg={float(row['sam_foreground_ratio']):.3f}"
            draw.text((x + 4, y + 6), title, fill=(0, 0, 0))

    preview_dir = output_dir / "preview_montages"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "train_sam_top_foreground_preview.png"
    canvas.save(preview_path)
    return preview_path


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return choose_device()
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SAM, but torch.cuda.is_available() is false.")
    return torch.device(value)


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    if args.seed_threshold is not None and not 0 <= args.seed_threshold <= 1:
        raise ValueError("--seed-threshold must be in [0, 1]")
    if args.positive_points < 0 or args.negative_points < 0:
        raise ValueError("point counts must be non-negative")
    if args.max_regions < 0:
        raise ValueError("--max-regions must be non-negative")
    if args.min_region_pixels <= 0:
        raise ValueError("--min-region-pixels must be positive")
    if args.max_area_expansion <= 0:
        raise ValueError("--max-area-expansion must be positive")
    if args.adaptive_seed_grow:
        adaptive_thresholds = (
            args.adaptive_large_candidate_threshold,
            args.adaptive_large_seed_threshold,
            args.adaptive_huge_candidate_threshold,
            args.adaptive_huge_seed_threshold,
            args.adaptive_huge_min_mean_score,
            args.adaptive_huge_min_peak_score,
            args.adaptive_huge_min_component_coverage,
        )
        if any(not 0 <= float(value) <= 1 for value in adaptive_thresholds):
            raise ValueError("adaptive Seed-Grow thresholds must be in [0, 1]")
        if not 0 <= args.adaptive_large_area_ratio <= args.adaptive_huge_area_ratio <= 1:
            raise ValueError("adaptive area ratios must satisfy 0 <= large <= huge <= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    source_dir = resolve_from(ROOT, args.source_dir)
    output_dir = resolve_from(ROOT, args.output_dir)
    checkpoint = resolve_from(ROOT, args.checkpoint)
    device = resolve_device(args.device)
    predictor, sam_backend = load_sam_predictor(checkpoint, args.model_type, args.sam_family, device)

    summary, rows, source_manifest = load_rows(source_dir)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    data_root = Path(summary["data_root"])
    label_dir = output_dir / "pseudo_labels"
    seed_dir = output_dir / "sam_seed_labels"
    confidence_dir = output_dir / "pseudo_confidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_threshold = float(args.seed_threshold if args.seed_threshold is not None else args.fg_point_threshold)
    extract_threshold = candidate_extraction_threshold(args)

    output_rows: list[dict[str, str]] = []
    for row in tqdm(rows, desc="P5 SAM postprocess", dynamic_ncols=True):
        sample_id = row["id"]
        probabilities = load_probability(source_dir / row["pseudo_label"])
        regions = extract_candidate_regions(
            probabilities,
            threshold=extract_threshold,
            min_region_pixels=args.min_region_pixels,
            max_regions=args.max_regions,
            box_pad_pixels=args.box_pad_pixels,
            box_pad_ratio=args.box_pad_ratio,
        )
        seed_mask = build_seed_mask(
            probabilities,
            regions,
            args,
            default_seed_threshold=seed_threshold,
        )
        if not regions:
            mask = np.zeros(probabilities.shape, dtype=np.uint8)
            stats: dict[str, float | int] = {
                "sam_regions": 0,
                "sam_accepted_regions": 0,
                "sam_seed_fallback_regions": 0,
                "sam_mean_selection_score": 0.0,
                "sam_t1_selected_regions": 0,
                "sam_t2_selected_regions": 0,
                "sam_adaptive_large_regions": 0,
                "sam_adaptive_huge_regions": 0,
                "sam_adaptive_topk_seed_regions": 0,
                "sam_adaptive_skipped_no_core_regions": 0,
            }
        else:
            if args.image_mode == "best_t1_t2":
                t1 = load_rgb_uint8(data_root / row["t1"])
                t2 = load_rgb_uint8(data_root / row["t2"])
                mask, stats = run_sam_on_t1_t2_best(predictor, t1, t2, probabilities, regions, args)
            else:
                sam_image = build_sam_input_image(data_root, row, args.image_mode)
                mask, stats = run_sam_on_image(predictor, sam_image, probabilities, regions, args)

        output_path = label_dir / f"{sample_id}.png"
        seed_path = seed_dir / f"{sample_id}.png"
        confidence_path = confidence_dir / f"{sample_id}.png"
        save_mask(output_path, mask)
        save_mask(seed_path, seed_mask)
        save_mask(confidence_path, np.ones_like(mask, dtype=np.uint8))

        output_row = dict(row)
        output_row["source_soft_pseudo_label"] = row["pseudo_label"]
        output_row["sam_seed_label"] = relative_to_output(seed_path, output_dir)
        output_row["pseudo_label"] = relative_to_output(output_path, output_dir)
        output_row["pseudo_confidence"] = relative_to_output(confidence_path, output_dir)
        output_row["sam_foreground_ratio"] = f"{float(mask.mean()):.8f}"
        output_row["sam_regions"] = str(int(stats["sam_regions"]))
        output_row["sam_accepted_regions"] = str(int(stats["sam_accepted_regions"]))
        output_row["sam_seed_fallback_regions"] = str(int(stats["sam_seed_fallback_regions"]))
        output_row["sam_t1_selected_regions"] = str(int(stats["sam_t1_selected_regions"]))
        output_row["sam_t2_selected_regions"] = str(int(stats["sam_t2_selected_regions"]))
        output_row["sam_adaptive_large_regions"] = str(int(stats["sam_adaptive_large_regions"]))
        output_row["sam_adaptive_huge_regions"] = str(int(stats["sam_adaptive_huge_regions"]))
        output_row["sam_adaptive_topk_seed_regions"] = str(int(stats["sam_adaptive_topk_seed_regions"]))
        output_row["sam_adaptive_skipped_no_core_regions"] = str(int(stats["sam_adaptive_skipped_no_core_regions"]))
        output_row["sam_mean_selection_score"] = f"{float(stats['sam_mean_selection_score']):.8f}"
        output_rows.append(output_row)

    manifest_path = output_dir / f"{summary.get('split', 'train')}_sam_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    preview_path = write_preview(
        output_rows,
        source_dir=source_dir,
        output_dir=output_dir,
        data_root=data_root,
        count=args.preview_count,
    )
    foreground = [float(row["sam_foreground_ratio"]) for row in output_rows]
    accepted = [int(row["sam_accepted_regions"]) for row in output_rows]
    fallbacks = [int(row["sam_seed_fallback_regions"]) for row in output_rows]
    t1_selected = [int(row["sam_t1_selected_regions"]) for row in output_rows]
    t2_selected = [int(row["sam_t2_selected_regions"]) for row in output_rows]
    adaptive_large = [int(row["sam_adaptive_large_regions"]) for row in output_rows]
    adaptive_huge = [int(row["sam_adaptive_huge_regions"]) for row in output_rows]
    adaptive_topk = [int(row["sam_adaptive_topk_seed_regions"]) for row in output_rows]
    adaptive_skipped = [int(row["sam_adaptive_skipped_no_core_regions"]) for row in output_rows]
    result = {
        "source_dir": str(source_dir),
        "source_manifest": str(source_manifest),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "split": summary.get("split", "train"),
        "checkpoint": str(checkpoint),
        "model_type": args.model_type,
        "sam_family": args.sam_family,
        "sam_backend": sam_backend,
        "image_mode": args.image_mode,
        "device": str(device),
        "threshold": args.threshold,
        "candidate_extraction_threshold": extract_threshold,
        "seed_threshold": seed_threshold,
        "require_core_seed": args.require_core_seed,
        "fg_point_threshold": args.fg_point_threshold,
        "adaptive_seed_grow": args.adaptive_seed_grow,
        "adaptive_large_area_ratio": args.adaptive_large_area_ratio,
        "adaptive_huge_area_ratio": args.adaptive_huge_area_ratio,
        "adaptive_large_candidate_threshold": args.adaptive_large_candidate_threshold,
        "adaptive_large_seed_threshold": args.adaptive_large_seed_threshold,
        "adaptive_huge_candidate_threshold": args.adaptive_huge_candidate_threshold,
        "adaptive_huge_seed_threshold": args.adaptive_huge_seed_threshold,
        "adaptive_huge_min_mean_score": args.adaptive_huge_min_mean_score,
        "adaptive_huge_min_peak_score": args.adaptive_huge_min_peak_score,
        "adaptive_huge_min_component_coverage": args.adaptive_huge_min_component_coverage,
        "adaptive_huge_allow_topk_seed": args.adaptive_huge_allow_topk_seed,
        "bg_point_threshold": args.bg_point_threshold,
        "positive_points": args.positive_points,
        "negative_points": args.negative_points,
        "min_region_pixels": args.min_region_pixels,
        "max_regions": args.max_regions,
        "box_pad_pixels": args.box_pad_pixels,
        "box_pad_ratio": args.box_pad_ratio,
        "min_sam_score": args.min_sam_score,
        "min_component_coverage": args.min_component_coverage,
        "min_pseudo_precision": args.min_pseudo_precision,
        "max_area_expansion": args.max_area_expansion,
        "max_mask_area_ratio": args.max_mask_area_ratio,
        "seed_fallback": args.seed_fallback,
        "open_kernel": args.open_kernel,
        "close_kernel": args.close_kernel,
        "min_component_pixels": args.min_component_pixels,
        "samples": len(output_rows),
        "foreground_ratio_mean": sum(foreground) / max(len(foreground), 1),
        "accepted_regions_mean": sum(accepted) / max(len(accepted), 1),
        "seed_fallback_regions_mean": sum(fallbacks) / max(len(fallbacks), 1),
        "t1_selected_regions_mean": sum(t1_selected) / max(len(t1_selected), 1),
        "t2_selected_regions_mean": sum(t2_selected) / max(len(t2_selected), 1),
        "adaptive_large_regions_mean": sum(adaptive_large) / max(len(adaptive_large), 1),
        "adaptive_huge_regions_mean": sum(adaptive_huge) / max(len(adaptive_huge), 1),
        "adaptive_topk_seed_regions_mean": sum(adaptive_topk) / max(len(adaptive_topk), 1),
        "adaptive_skipped_no_core_regions_mean": sum(adaptive_skipped) / max(len(adaptive_skipped), 1),
        "preview": str(preview_path) if preview_path else None,
    }
    write_json(output_dir / "summary.json", result)
    print(f"Generated P5-prompted SAM pseudo labels: {len(output_rows)} samples")
    print(f"Manifest: {manifest_path}")
    if preview_path:
        print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
