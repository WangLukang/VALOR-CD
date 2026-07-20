from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TILE_FIELDS = [
    "id",
    "scene",
    "source_split",
    "source_id",
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
MANIFEST_FIELDS = ["id", "t1", "t2", "label", "mask", "change_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LEVIR-CD as 224x224 paired tiles with official train/val/test splits.")
    parser.add_argument("--source-root", default="data/LEVIR")
    parser.add_argument("--output-root", default="data/LEVIR_224")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--min-change-pixels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_row(tile_row: dict[str, Any], min_change_pixels: int) -> dict[str, Any]:
    return {
        "id": tile_row["id"],
        "t1": tile_row["t1"],
        "t2": tile_row["t2"],
        "label": int(int(tile_row["change_pixels"]) >= min_change_pixels),
        "mask": tile_row["mask"],
        "change_ratio": tile_row["change_ratio"],
    }


def axis_positions(length: int, tile_size: int) -> list[int]:
    if length < tile_size:
        raise ValueError(f"Image side {length} is smaller than tile size {tile_size}")
    positions = list(range(0, max(length - tile_size + 1, 1), tile_size))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def save_tile(image: Image.Image, box: tuple[int, int, int, int], path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).convert(mode).save(path)


def numeric_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    try:
        return stem.rsplit("_", 1)[0], int(stem.rsplit("_", 1)[1])
    except Exception:
        return stem, 0


def prepare_split(source_root: Path, output_root: Path, split: str, tile_size: int, min_change_pixels: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_root = source_root / split
    for sub in ["A", "B", "label"]:
        if not (split_root / sub).is_dir():
            raise FileNotFoundError(split_root / sub)

    t1_files = sorted((split_root / "A").glob("*.png"), key=numeric_key)
    tile_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for t1_path in t1_files:
        t2_path = split_root / "B" / t1_path.name
        mask_path = split_root / "label" / t1_path.name
        if not t2_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair for {t1_path.name}: {t2_path}, {mask_path}")

        t1 = Image.open(t1_path).convert("RGB")
        t2 = Image.open(t2_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if t1.size != t2.size or t1.size != mask.size:
            raise ValueError(f"Size mismatch for {t1_path.name}: {t1.size}, {t2.size}, {mask.size}")

        width, height = t1.size
        xs = axis_positions(width, tile_size)
        ys = axis_positions(height, tile_size)
        source_id = t1_path.stem
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                box = (x, y, x + tile_size, y + tile_size)
                sample_id = f"LEVIR_{split}_{source_id}_r{row:04d}_c{col:04d}"
                t1_rel = f"T1/{sample_id}.png"
                t2_rel = f"T2/{sample_id}.png"
                mask_rel = f"masks/{sample_id}.png"
                save_tile(t1, box, output_root / t1_rel, "RGB")
                save_tile(t2, box, output_root / t2_rel, "RGB")

                mask_tile = mask.crop(box).convert("L")
                mask_array = (np.array(mask_tile) > 0).astype(np.uint8) * 255
                (output_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask_array, mode="L").save(output_root / mask_rel)
                change_pixels = int((mask_array > 0).sum())
                change_ratio = change_pixels / float(tile_size * tile_size)

                tile_row = {
                    "id": sample_id,
                    "scene": f"LEVIR_{split}_{source_id}",
                    "source_split": split,
                    "source_id": source_id,
                    "row": row,
                    "col": col,
                    "x": x,
                    "y": y,
                    "tile_size": tile_size,
                    "change_pixels": change_pixels,
                    "change_ratio": f"{change_ratio:.8f}",
                    "t1": t1_rel,
                    "t2": t2_rel,
                    "mask": mask_rel,
                }
                tile_rows.append(tile_row)
                manifest_rows.append(manifest_row(tile_row, min_change_pixels))
    return tile_rows, manifest_rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = (repo_root / args.source_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_root)

    all_tiles: list[dict[str, Any]] = []
    all_manifest: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        tile_rows, manifest_rows = prepare_split(source_root, output_root, split, args.tile_size, args.min_change_pixels)
        all_tiles.extend(tile_rows)
        all_manifest.extend(manifest_rows)
        manifests[split] = manifest_rows

    write_csv(output_root / "tiles.csv", all_tiles, TILE_FIELDS)
    split_dir = output_root / "manifests_official"
    for split, rows in manifests.items():
        write_csv(split_dir / f"{split}.csv", rows, MANIFEST_FIELDS)
    write_csv(split_dir / "all.csv", all_manifest, MANIFEST_FIELDS)
    write_csv(split_dir / "ignored.csv", [], MANIFEST_FIELDS)

    print(f"Prepared LEVIR tiles: {output_root}")
    print(f"Total: {len(all_manifest)} samples")
    for split in ["train", "val", "test"]:
        rows = manifests[split]
        positives = sum(int(row["label"]) for row in rows)
        print(f"{split:>7}: {len(rows):5d} samples, {positives:5d} positive, {len(rows) - positives:5d} negative")


if __name__ == "__main__":
    main()
