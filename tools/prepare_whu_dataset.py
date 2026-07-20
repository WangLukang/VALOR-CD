from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.data.spatial_split import (  # noqa: E402
    make_image_label_rows,
    spatial_split,
    write_manifest,
)

TILE_FIELDS = [
    "id",
    "scene",
    "row",
    "col",
    "x",
    "y",
    "tile_size",
    "change_pixels",
    "change_ratio",
    "t1",
    "t2",
    "mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the WHU-CD large TIFF pair as non-overlapping 224x224 tiles."
    )
    parser.add_argument("--before", default="data/WHU/before/before.tif")
    parser.add_argument("--after", default="data/WHU/after/after.tif")
    parser.add_argument("--mask", default="data/WHU/change label/change_label.tif")
    parser.add_argument("--output-root", default="data/WHU_224")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--gap-tiles", type=int, default=1)
    parser.add_argument("--min-change-pixels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def write_tiles(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TILE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def rgb_from_window(dataset: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    if dataset.count < 3:
        raise ValueError(f"Expected at least three image bands, found {dataset.count}: {dataset.name}")
    array = dataset.read((1, 2, 3), window=window)
    array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0:
        raise ValueError("tile-size must be positive")

    before_path = resolve(args.before)
    after_path = resolve(args.after)
    mask_path = resolve(args.mask)
    output_root = resolve(args.output_root)
    for source in (before_path, after_path, mask_path):
        if not source.is_file():
            raise FileNotFoundError(source)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_root)

    with (
        rasterio.open(before_path) as before,
        rasterio.open(after_path) as after,
        rasterio.open(mask_path) as mask,
    ):
        shapes = {(before.width, before.height), (after.width, after.height), (mask.width, mask.height)}
        if len(shapes) != 1:
            raise ValueError(
                "WHU source dimensions differ: "
                f"before={(before.width, before.height)}, "
                f"after={(after.width, after.height)}, mask={(mask.width, mask.height)}"
            )

        width, height = before.width, before.height
        cols = width // args.tile_size
        rows = height // args.tile_size
        if rows == 0 or cols == 0:
            raise ValueError(f"Source image {(width, height)} is smaller than tile size {args.tile_size}")

        tile_rows: list[dict[str, Any]] = []
        progress = tqdm(total=rows * cols, desc="Preparing WHU-CD tiles")
        for row in range(rows):
            y = row * args.tile_size
            for col in range(cols):
                x = col * args.tile_size
                window = Window(x, y, args.tile_size, args.tile_size)
                sample_id = f"BCD_r{row:04d}_c{col:04d}"
                t1_rel = f"T1/{sample_id}.png"
                t2_rel = f"T2/{sample_id}.png"
                mask_rel = f"masks/{sample_id}.png"

                t1_array = rgb_from_window(before, window)
                t2_array = rgb_from_window(after, window)
                mask_array = (mask.read(1, window=window) > 0).astype(np.uint8) * 255

                for relative in (t1_rel, t2_rel, mask_rel):
                    (output_root / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(t1_array).save(output_root / t1_rel)
                Image.fromarray(t2_array).save(output_root / t2_rel)
                Image.fromarray(mask_array).save(output_root / mask_rel)

                change_pixels = int((mask_array > 0).sum())
                change_ratio = change_pixels / float(args.tile_size * args.tile_size)
                tile_rows.append(
                    {
                        "id": sample_id,
                        "scene": "BCD",
                        "row": row,
                        "col": col,
                        "x": x,
                        "y": y,
                        "tile_size": args.tile_size,
                        "change_pixels": change_pixels,
                        "change_ratio": f"{change_ratio:.8f}",
                        "t1": t1_rel,
                        "t2": t2_rel,
                        "mask": mask_rel,
                    }
                )
                progress.update(1)
        progress.close()

    write_tiles(output_root / "tiles.csv", tile_rows)
    image_label_rows = make_image_label_rows(
        tile_rows, min_change_pixels=args.min_change_pixels
    )
    splits = spatial_split(
        image_label_rows,
        axis="row",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        gap_tiles=args.gap_tiles,
    )
    splits["all"] = image_label_rows
    manifest_root = output_root / "manifests_spatial"
    for split, split_rows in splits.items():
        write_manifest(manifest_root / f"{split}.csv", split_rows)

    print(f"Prepared WHU-CD tiles: {output_root}")
    print(
        f"Source={(width, height)}, tile={args.tile_size}, grid={rows}x{cols}, "
        f"discarded_border={(width - cols * args.tile_size, height - rows * args.tile_size)}"
    )
    for split in ("train", "val", "test", "ignored"):
        split_rows = splits[split]
        positives = sum(int(row["label"]) for row in split_rows)
        print(
            f"{split:>7}: {len(split_rows):5d} samples, {positives:5d} positive, "
            f"{len(split_rows) - positives:5d} negative"
        )


if __name__ == "__main__":
    main()
