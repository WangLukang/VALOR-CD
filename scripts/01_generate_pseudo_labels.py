from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def as_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def run_step(name: str, command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n[{name}] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def add_optional(command: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def append_sam_options(command: list[str], sam_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--model-type",
            str(sam_cfg.get("model_type", "vit_h")),
            "--sam-family",
            str(sam_cfg.get("sam_family", "sam")),
            "--image-mode",
            str(sam_cfg.get("image_mode", "t2")),
            "--threshold",
            str(sam_cfg.get("threshold", 0.5)),
            "--seed-threshold",
            str(sam_cfg.get("seed_threshold", sam_cfg.get("fg_point_threshold", 0.85))),
            "--fg-point-threshold",
            str(sam_cfg.get("fg_point_threshold", 0.85)),
            "--bg-point-threshold",
            str(sam_cfg.get("bg_point_threshold", 0.2)),
            "--positive-points",
            str(sam_cfg.get("positive_points", 3)),
            "--negative-points",
            str(sam_cfg.get("negative_points", 4)),
            "--min-region-pixels",
            str(sam_cfg.get("min_region_pixels", 8)),
            "--max-regions",
            str(sam_cfg.get("max_regions", 12)),
            "--box-pad-pixels",
            str(sam_cfg.get("box_pad_pixels", 6)),
            "--box-pad-ratio",
            str(sam_cfg.get("box_pad_ratio", 0.10)),
            "--min-sam-score",
            str(sam_cfg.get("min_sam_score", 0.0)),
            "--min-component-coverage",
            str(sam_cfg.get("min_component_coverage", 0.20)),
            "--min-pseudo-precision",
            str(sam_cfg.get("min_pseudo_precision", 0.12)),
            "--max-area-expansion",
            str(sam_cfg.get("max_area_expansion", 8.0)),
            "--max-mask-area-ratio",
            str(sam_cfg.get("max_mask_area_ratio", 0.75)),
            "--open-kernel",
            str(sam_cfg.get("open_kernel", 0)),
            "--close-kernel",
            str(sam_cfg.get("close_kernel", 3)),
            "--min-component-pixels",
            str(sam_cfg.get("min_component_pixels", 8)),
            "--preview-count",
            str(sam_cfg.get("preview_count", 8)),
            "--device",
            str(sam_cfg.get("device", "auto")),
        ]
    )

    adaptive_cfg = dict(sam_cfg.get("adaptive_seed_grow", {}))
    command.append("--adaptive-seed-grow" if bool(adaptive_cfg.get("enabled", False)) else "--no-adaptive-seed-grow")
    adaptive_option_map = {
        "large_area_ratio": "--adaptive-large-area-ratio",
        "huge_area_ratio": "--adaptive-huge-area-ratio",
        "large_candidate_threshold": "--adaptive-large-candidate-threshold",
        "large_seed_threshold": "--adaptive-large-seed-threshold",
        "huge_candidate_threshold": "--adaptive-huge-candidate-threshold",
        "huge_seed_threshold": "--adaptive-huge-seed-threshold",
        "huge_min_mean_score": "--adaptive-huge-min-mean-score",
        "huge_min_peak_score": "--adaptive-huge-min-peak-score",
        "huge_min_component_coverage": "--adaptive-huge-min-component-coverage",
    }
    for key, flag in adaptive_option_map.items():
        add_optional(command, flag, adaptive_cfg.get(key))
    command.append(
        "--adaptive-huge-allow-topk-seed"
        if bool(adaptive_cfg.get("huge_allow_topk_seed", True))
        else "--no-adaptive-huge-allow-topk-seed"
    )
    command.append("--seed-fallback" if bool(sam_cfg.get("seed_fallback", True)) else "--no-seed-fallback")
    command.append("--require-core-seed" if bool(sam_cfg.get("require_core_seed", False)) else "--no-require-core-seed")
    add_optional(command, "--max-samples", args.max_sam_samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: train the weak teacher and generate soft/SAM pseudo labels."
    )
    parser.add_argument("--config", default="configs/whu/stage1.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sam-only", action="store_true", help="Regenerate SAM hard labels from existing soft pseudo labels.")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-soft-pseudo", action="store_true")
    parser.add_argument("--skip-sam", action="store_true")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers for generated configs.")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-sam-samples", type=int)
    args = parser.parse_args()
    cfg = load_yaml(root_path(args.config))

    run_mode = str(cfg.get("run_mode", "")).strip().lower().replace("-", "_")
    if args.sam_only or run_mode == "sam_only":
        args.sam_only = True
        args.skip_teacher = True
        args.skip_calibration = True
        args.skip_soft_pseudo = True

    output_root = root_path(cfg.get("output_root", "outputs/stage1_pseudo"))
    runtime_dir = output_root / "runtime_configs"
    if not args.dry_run:
        runtime_dir.mkdir(parents=True, exist_ok=True)

    sam_cfg = dict(cfg["sam_pseudo"])
    soft_cfg = dict(cfg.get("soft_pseudo", {}))

    if args.sam_only:
        soft_output = root_path(sam_cfg.get("source_dir", soft_cfg.get("output_dir", output_root / "soft_pseudo_labels")))
    else:
        teacher_cfg = dict(cfg["weak_teacher"])
        teacher_cfg["output_dir"] = as_rel(root_path(teacher_cfg.get("output_dir", output_root / "weak_teacher")))
        if args.num_workers is not None:
            teacher_cfg.setdefault("data", {})["num_workers"] = args.num_workers
        teacher_config_path = runtime_dir / "weak_teacher.yaml"
        if not args.dry_run:
            write_yaml(teacher_config_path, teacher_cfg)

        calibration_cfg = dict(cfg.get("calibration", {}))
        calibration_output = root_path(calibration_cfg.get("output_dir", output_root / "calibrated_teacher"))
        calibrated_checkpoint = calibration_output / "best_candidate_calibrated.pt"
        fixed_calibration_cfg = {
            "checkpoint": as_rel(root_path(teacher_cfg["output_dir"]) / "best.pt"),
            "output_dir": as_rel(calibration_output),
            "mode": calibration_cfg.get("mode", "preserve_checkpoint"),
        }
        if "pixel_threshold" in calibration_cfg:
            fixed_calibration_cfg["pixel_threshold"] = float(calibration_cfg["pixel_threshold"])
        if "candidate_filter" in calibration_cfg:
            fixed_calibration_cfg["candidate_filter"] = calibration_cfg.get("candidate_filter")
        fixed_calibration_config_path = runtime_dir / "fixed_calibration.yaml"
        if not args.dry_run:
            write_yaml(fixed_calibration_config_path, fixed_calibration_cfg)

        soft_cfg = dict(cfg["soft_pseudo"])
        soft_cfg["checkpoint"] = as_rel(calibrated_checkpoint)
        soft_cfg["output_dir"] = as_rel(root_path(soft_cfg.get("output_dir", output_root / "soft_pseudo_labels")))
        soft_config_path = runtime_dir / "soft_pseudo.yaml"
        if not args.dry_run:
            write_yaml(soft_config_path, soft_cfg)
        soft_output = root_path(soft_cfg["output_dir"])

    sam_source = root_path(sam_cfg.get("source_dir", soft_output))
    sam_output = root_path(sam_cfg.get("output_dir", output_root / "sam_pseudo_labels"))
    sam_checkpoint = root_path(sam_cfg.get("checkpoint", "models/sam/sam_vit_h_4b8939.pth"))

    if not args.skip_teacher:
        command = [args.python, "tools/train_p1_cam.py", "--config", as_rel(teacher_config_path)]
        add_optional(command, "--max-train-batches", args.max_train_batches)
        add_optional(command, "--max-val-batches", args.max_val_batches)
        run_step("1/4 Train DINOv2 StrongCAM weak teacher", command, dry_run=args.dry_run)

    if not args.skip_calibration:
        command = [args.python, "tools/create_fixed_calibrated_checkpoint.py", "--config", as_rel(fixed_calibration_config_path)]
        run_step("2/4 Write fixed mask-free teacher checkpoint", command, dry_run=args.dry_run)

    if not args.skip_soft_pseudo:
        command = [args.python, "tools/generate_p5_soft_pseudo_labels.py", "--config", as_rel(soft_config_path)]
        add_optional(command, "--max-batches", args.max_train_batches)
        run_step("3/4 Generate counterfactual-suppressed soft pseudo labels", command, dry_run=args.dry_run)

    if not args.skip_sam:
        command = [
            args.python,
            "tools/postprocess_p5_sam.py",
            "--source-dir",
            as_rel(sam_source),
            "--output-dir",
            as_rel(sam_output),
            "--checkpoint",
            as_rel(sam_checkpoint),
        ]
        append_sam_options(command, sam_cfg, args)
        run_step("4/4 Generate SAM hard pseudo labels", command, dry_run=args.dry_run)

    print("\nStage 1 finished. Pseudo labels are under:")
    print(f"  soft: {as_rel(soft_output)}")
    print(f"  hard: {as_rel(sam_output)}")


if __name__ == "__main__":
    main()
