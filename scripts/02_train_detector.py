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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: train one detector with joint soft/SAM hard-mask distillation."
    )
    parser.add_argument("--config", default="configs/whu/stage2.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--tiny-random-model", action="store_true", help="Quick smoke test only.")
    args = parser.parse_args()

    cfg = load_yaml(root_path(args.config))
    output_root = root_path(cfg.get("output_root", "outputs/stage2_detector"))
    runtime_dir = output_root / "runtime_configs"
    if not args.dry_run:
        runtime_dir.mkdir(parents=True, exist_ok=True)

    detector_cfg = dict(cfg.get("detector", {}))
    if not detector_cfg:
        raise ValueError("Stage2 config must contain a detector block")
    detector_cfg["output_dir"] = as_rel(
        root_path(detector_cfg.get("output_dir", output_root / "joint_seed_grow_sam_vith_detector"))
    )
    detector_cfg.setdefault("train", {}).pop("warm_start", None)
    if args.num_workers is not None:
        detector_cfg.setdefault("data", {})["num_workers"] = args.num_workers
    if args.batch_size is not None:
        detector_cfg.setdefault("data", {})["batch_size"] = args.batch_size
        detector_cfg.setdefault("data", {})["val_batch_size"] = args.batch_size
    if args.epochs is not None:
        detector_cfg.setdefault("train", {})["epochs"] = args.epochs
    detector_config_path = runtime_dir / "detector.yaml"
    if not args.dry_run:
        write_yaml(detector_config_path, detector_cfg)

    if not args.skip_detector:
        command = [args.python, "tools/train_dense_detector.py", "--config", as_rel(detector_config_path)]
        add_optional(command, "--max-train-batches", args.max_train_batches)
        add_optional(command, "--max-val-batches", args.max_val_batches)
        if args.tiny_random_model:
            command.append("--tiny-random-model")
        run_step("1/1 Train joint soft/SAM hard-mask distillation detector", command, dry_run=args.dry_run)

    print("\nStage 2 finished. Final detector checkpoint:")
    print(f"  {as_rel(root_path(detector_cfg['output_dir']) / 'best.pt')}")


if __name__ == "__main__":
    main()
