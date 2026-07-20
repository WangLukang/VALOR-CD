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


def root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def as_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def add_optional(command: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: evaluate the final detector on val/test.")
    parser.add_argument("--config", default="configs/whu/stage3.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pixel-threshold", type=float)
    parser.add_argument("--image-threshold", type=float)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--no-export-masks", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(root_path(args.config))
    checkpoint = root_path(args.checkpoint or cfg["checkpoint"])
    split = args.split or str(cfg.get("split", "test"))
    output_dir = root_path(args.output_dir or cfg.get("output_dir", f"outputs/stage3_{split}"))

    command = [
        args.python,
        "tools/evaluate_dense_detector.py",
        "--checkpoint",
        as_rel(checkpoint),
        "--split",
        split,
        "--output-dir",
        as_rel(output_dir),
    ]
    add_optional(command, "--batch-size", args.batch_size if args.batch_size is not None else cfg.get("batch_size"))
    add_optional(command, "--num-workers", args.num_workers if args.num_workers is not None else cfg.get("num_workers"))
    add_optional(command, "--pixel-threshold", args.pixel_threshold if args.pixel_threshold is not None else cfg.get("pixel_threshold"))
    add_optional(command, "--image-threshold", args.image_threshold if args.image_threshold is not None else cfg.get("image_threshold"))
    add_optional(command, "--max-batches", args.max_batches)
    if bool(cfg.get("export_masks", True)) and not args.no_export_masks:
        command.append("--export-masks")

    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"[Stage 3 Test] {printable}", flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
