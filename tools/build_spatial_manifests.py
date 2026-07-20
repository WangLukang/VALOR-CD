from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.data.spatial_split import (  # noqa: E402
    make_image_label_rows,
    read_rows,
    spatial_split,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spatially isolated manifests.")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "whu" / "data.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = (ROOT / config["root"]).resolve()
    split_config = config["split"]
    rows = make_image_label_rows(
        read_rows(root / split_config["source"]),
        min_change_pixels=int(split_config["min_change_pixels"]),
    )
    splits = spatial_split(
        rows,
        axis=split_config["axis"],
        train_ratio=float(split_config["train_ratio"]),
        val_ratio=float(split_config["val_ratio"]),
        gap_tiles=int(split_config["gap_tiles"]),
    )
    splits["all"] = rows

    output_dir = root / split_config["output_dir"]
    for name, split_rows in splits.items():
        write_manifest(output_dir / f"{name}.csv", split_rows)

    print(f"Wrote spatial manifests to {output_dir}")
    for name in ("train", "val", "test", "ignored"):
        positives = sum(int(row["label"]) for row in splits[name])
        print(
            f"{name:>7}: {len(splits[name]):4d} samples, "
            f"{positives:4d} positive, {len(splits[name]) - positives:4d} negative"
        )


if __name__ == "__main__":
    main()
