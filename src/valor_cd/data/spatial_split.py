from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TILE_PATTERN = re.compile(r"^(?P<scene>.+)_r(?P<row>\d+)_c(?P<col>\d+)$")
MANIFEST_FIELDS = ["id", "t1", "t2", "label", "mask", "change_ratio"]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Source CSV is empty: {path}")
    return rows


def make_image_label_rows(
    rows: Iterable[dict[str, str]], *, min_change_pixels: int = 1
) -> list[dict[str, Any]]:
    if min_change_pixels < 1:
        raise ValueError("min_change_pixels must be at least 1")
    output = []
    for row in rows:
        missing = {"id", "t1", "t2", "mask", "change_pixels", "change_ratio"} - set(row)
        if missing:
            raise ValueError(f"Source row is missing columns: {sorted(missing)}")
        output.append(
            {
                "id": row["id"],
                "t1": row["t1"],
                "t2": row["t2"],
                "label": int(int(row["change_pixels"]) >= min_change_pixels),
                "mask": row["mask"],
                "change_ratio": row["change_ratio"],
            }
        )
    return output


def spatial_split(
    rows: Iterable[dict[str, Any]],
    *,
    axis: str = "row",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    gap_tiles: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    if axis not in {"row", "col"}:
        raise ValueError("axis must be 'row' or 'col'")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    if gap_tiles < 0:
        raise ValueError("gap_tiles must be non-negative")

    indexed: list[tuple[str, int, dict[str, Any]]] = []
    for row in rows:
        match = TILE_PATTERN.match(str(row["id"]))
        if not match:
            raise ValueError(f"Cannot parse spatial coordinates from ID: {row['id']}")
        indexed.append((match.group("scene"), int(match.group(axis)), row))

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
        "ignored": [],
    }
    for scene in sorted({item[0] for item in indexed}):
        scene_rows = [(coord, row) for name, coord, row in indexed if name == scene]
        coordinates = sorted({coord for coord, _ in scene_rows})
        train_cut = round(len(coordinates) * train_ratio)
        val_cut = train_cut + round(len(coordinates) * val_ratio)

        coordinate_splits = {
            "train": set(coordinates[: max(train_cut - gap_tiles, 0)]),
            "val": set(
                coordinates[
                    min(train_cut + gap_tiles, len(coordinates)) : max(
                        val_cut - gap_tiles, 0
                    )
                ]
            ),
            "test": set(coordinates[min(val_cut + gap_tiles, len(coordinates)) :]),
        }
        for coordinate, row in scene_rows:
            destination = "ignored"
            for name, selected in coordinate_splits.items():
                if coordinate in selected:
                    destination = name
                    break
            splits[destination].append(row)

    for name in ("train", "val", "test"):
        if not splits[name]:
            raise ValueError(f"Spatial split is empty: {name}")
    return splits


def write_manifest(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def tile_coordinate(sample_id: str, axis: str) -> tuple[str, int]:
    match = TILE_PATTERN.match(sample_id)
    if not match:
        raise ValueError(f"Cannot parse spatial coordinates from ID: {sample_id}")
    return match.group("scene"), int(match.group(axis))
