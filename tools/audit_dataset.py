from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.data.spatial_split import tile_coordinate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paired change data manifests.")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "whu" / "data.yaml")
    )
    parser.add_argument("--check-all-images", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = (ROOT / config["root"]).resolve()
    axis = config["split"]["axis"]
    manifests = {
        name: read_manifest(root / path)
        for name, path in config["manifests"].items()
    }

    ids = {name: {row["id"] for row in rows} for name, rows in manifests.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise AssertionError(f"{left}/{right} share {len(overlap)} sample IDs")

    bands = {
        name: {tile_coordinate(row["id"], axis) for row in rows}
        for name, rows in manifests.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = bands[left] & bands[right]
        if overlap:
            raise AssertionError(f"{left}/{right} share {len(overlap)} spatial bands")

    for name, rows in manifests.items():
        samples = rows if args.check_all_images else rows[:10]
        for row in samples:
            paths = [root / row[key] for key in ("t1", "t2", "mask")]
            if any(not path.is_file() for path in paths):
                raise FileNotFoundError(f"Missing triplet for sample {row['id']}")
            sizes = [Image.open(path).size for path in paths]
            if len(set(sizes)) != 1:
                raise ValueError(f"Size mismatch for sample {row['id']}: {sizes}")

        positives = sum(int(row["label"]) for row in rows)
        coordinates = sorted(coord for _, coord in bands[name])
        print(
            f"{name:>5}: {len(rows):4d} samples, {positives:4d} positive, "
            f"{len(rows) - positives:4d} negative, "
            f"{axis}s={min(coordinates)}..{max(coordinates)}"
        )
    print("Dataset audit passed.")


if __name__ == "__main__":
    main()
