from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.experiment import load_yaml, resolve_from, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a calibrated P1 checkpoint with mask-free post-processing parameters."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _selection(mode: str, note: str) -> dict[str, Any]:
    return {
        "metric": mode,
        "uses_validation_pixel_masks": False,
        "fixed_after_sweep": True,
        "note": note,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_from(ROOT, args.config))
    checkpoint_path = resolve_from(ROOT, config["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"P1 checkpoint does not exist: {checkpoint_path}")

    output_dir = resolve_from(ROOT, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_candidate_calibrated.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode = str(config.get("mode", "fixed"))

    if mode in {"preserve_checkpoint", "checkpoint", "preserve"}:
        pixel_threshold = float(checkpoint.get("pixel_threshold", 0.5))
        candidate_filter = checkpoint.get("candidate_calibration")
        pixel_calibration = checkpoint.get("pixel_calibration") or {
            "name": "checkpoint_threshold",
            "pixel_threshold": pixel_threshold,
            "uses_validation_pixel_masks": False,
        }
        selection = _selection(
            "preserve_mask_free_checkpoint_calibration",
            "Preserve the teacher checkpoint calibration estimated from image-level negative samples; no validation pixel masks are used for this calibration step.",
        )
    else:
        pixel_threshold = float(config.get("pixel_threshold", checkpoint.get("pixel_threshold", 0.5)))
        candidate_filter: dict[str, Any] | None = config.get("candidate_filter")
        if candidate_filter:
            candidate_filter = dict(candidate_filter)
            candidate_filter.setdefault("name", "fixed_component_filter")
            candidate_filter.setdefault("enabled", True)
            candidate_filter["pixel_threshold"] = pixel_threshold
            selection = _selection(
                "fixed_mask_free_prior",
                "Candidate filtering uses fixed priors and does not tune parameters with validation masks.",
            )
        else:
            selection = _selection(
                "none",
                "No candidate post-processing filter is applied.",
            )
        pixel_calibration = {
            "name": "fixed_threshold",
            "pixel_threshold": pixel_threshold,
            "uses_validation_pixel_masks": False,
        }

    calibrated = dict(checkpoint)
    calibrated["pixel_threshold"] = pixel_threshold
    calibrated["pixel_calibration"] = pixel_calibration
    calibrated["candidate_calibration"] = candidate_filter
    calibrated["candidate_calibration_selection"] = selection
    calibrated["config"] = dict(checkpoint["config"])
    calibrated["config"]["evaluation"] = dict(checkpoint["config"].get("evaluation", {}))
    calibrated["config"]["evaluation"]["cam_threshold"] = pixel_threshold
    if candidate_filter:
        calibrated["config"]["evaluation"]["candidate_calibration"] = candidate_filter
    else:
        calibrated["config"]["evaluation"].pop("candidate_calibration", None)

    torch.save(calibrated, output_path)
    summary = {
        "source_checkpoint": str(checkpoint_path),
        "output_checkpoint": str(output_path),
        "mode": mode,
        "pixel_threshold": pixel_threshold,
        "pixel_calibration": pixel_calibration,
        "candidate_filter": candidate_filter,
        "selection": selection,
    }
    write_json(output_dir / "fixed_calibration.json", summary)
    print("Calibrated checkpoint written:")
    print(f"  {output_path}")
    print("uses_validation_pixel_masks=False")


if __name__ == "__main__":
    main()
