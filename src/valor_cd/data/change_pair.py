from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ChangePairDataset(Dataset[dict[str, Any]]):
    """CSV-driven paired-image dataset with image-level training labels.

    Images are returned in RGB ``[0, 1]``. Pixel masks are opt-in so that a
    training dataset cannot consume dense labels accidentally.
    """

    REQUIRED_COLUMNS = {"id", "t1", "t2", "label"}

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        image_size: int = 224,
        augment: bool = False,
        return_mask: bool = False,
    ) -> None:
        self.root = Path(root)
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.root / self.manifest
        self.image_size = image_size
        self.augment = augment
        self.return_mask = return_mask

        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        missing = self.REQUIRED_COLUMNS - set(self.rows[0])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row in self.rows:
            if row["label"] not in {"0", "1", "0.0", "1.0"}:
                raise ValueError(
                    f"Sample {row['id']} has a non-binary image label: {row['label']}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        t1 = self._load(row["t1"], "RGB")
        t2 = self._load(row["t2"], "RGB")
        if t1.size != t2.size:
            raise ValueError(f"T1/T2 sizes differ for sample {row['id']}")

        mask = None
        if self.return_mask:
            mask_value = row.get("mask", "").strip()
            if not mask_value:
                raise ValueError(
                    f"Mask was requested but sample {row['id']} has no mask path"
                )
            mask = self._load(mask_value, "L")
            if mask.size != t1.size:
                raise ValueError(f"Image/mask sizes differ for sample {row['id']}")

        if self.augment:
            t1, t2, mask = self._paired_geometry(t1, t2, mask)

        sample: dict[str, Any] = {
            "id": row["id"],
            "t1": TF.to_tensor(t1),
            "t2": TF.to_tensor(t2),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "change_ratio": torch.tensor(
                float(row.get("change_ratio", 0.0)), dtype=torch.float32
            ),
        }
        if mask is not None:
            sample["mask"] = (TF.pil_to_tensor(mask).squeeze(0) > 0).float()
        return sample

    def positive_class_weight(self) -> float:
        positives = sum(float(row["label"]) > 0.5 for row in self.rows)
        negatives = len(self.rows) - positives
        if positives == 0:
            raise ValueError("Manifest contains no positive samples")
        return negatives / positives

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load(self, value: str, mode: str) -> Image.Image:
        path = self._resolve(value)
        image = Image.open(path).convert(mode)
        if image.size != (self.image_size, self.image_size):
            interpolation = (
                TF.InterpolationMode.BILINEAR
                if mode == "RGB"
                else TF.InterpolationMode.NEAREST
            )
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=interpolation,
            )
        return image

    @staticmethod
    def _paired_geometry(
        t1: Image.Image, t2: Image.Image, mask: Image.Image | None
    ) -> tuple[Image.Image, Image.Image, Image.Image | None]:
        if random.random() < 0.5:
            t1, t2 = TF.hflip(t1), TF.hflip(t2)
            mask = TF.hflip(mask) if mask is not None else None
        if random.random() < 0.5:
            t1, t2 = TF.vflip(t1), TF.vflip(t2)
            mask = TF.vflip(mask) if mask is not None else None
        return t1, t2, mask
