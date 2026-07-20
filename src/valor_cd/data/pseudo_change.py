from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch.nn.functional as F
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class BoundaryMaskSource:
    def __init__(
        self,
        *,
        name: str,
        root: str | Path,
        manifest: str | Path,
        column: str = "pseudo_label",
    ) -> None:
        self.name = str(name)
        self.root = Path(root)
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.root / self.manifest
        self.column = str(column)
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Boundary manifest is empty: {self.manifest}")
        missing = {"id", self.column} - set(rows[0])
        if missing:
            raise ValueError(
                f"Boundary manifest {self.manifest} is missing columns: {sorted(missing)}"
            )
        self.paths = {row["id"]: row[self.column] for row in rows}

    def resolve(self, sample_id: str) -> Path | None:
        value = self.paths.get(sample_id)
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.root / path


class HardLabelSource:
    def __init__(
        self,
        *,
        root: str | Path,
        manifest: str | Path,
        label_column: str = "edge_hard_label",
        confidence_column: str = "edge_hard_confidence",
    ) -> None:
        self.root = Path(root)
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.root / self.manifest
        self.label_column = str(label_column)
        self.confidence_column = str(confidence_column)
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Hard-label manifest is empty: {self.manifest}")
        missing = {"id", self.label_column, self.confidence_column} - set(rows[0])
        if missing:
            raise ValueError(
                f"Hard-label manifest {self.manifest} is missing columns: "
                f"{sorted(missing)}"
            )
        self.rows = {row["id"]: row for row in rows}

    def resolve(self, sample_id: str) -> tuple[Path, Path]:
        row = self.rows.get(sample_id)
        if row is None:
            raise KeyError(f"Hard-label source has no sample id: {sample_id}")
        label_path = Path(row[self.label_column])
        confidence_path = Path(row[self.confidence_column])
        if not label_path.is_absolute():
            label_path = self.root / label_path
        if not confidence_path.is_absolute():
            confidence_path = self.root / confidence_path
        return label_path, confidence_path


class PseudoLabelChangeDataset(Dataset[dict[str, Any]]):
    """P5 pseudo-label dataset for the P6 dense student.

    T1/T2 are resolved from the original data root. Pseudo labels and confidence
    maps are resolved from the P5 output directory.
    """

    REQUIRED_COLUMNS = {"id", "t1", "t2", "label", "pseudo_label"}

    def __init__(
        self,
        data_root: str | Path,
        pseudo_root: str | Path,
        manifest: str | Path,
        *,
        image_size: int = 224,
        augment: bool = False,
        pseudo_confidence_mode: str = "manifest",
        pseudo_confidence_value: float = 1.0,
        boundary_sources: list[dict[str, Any]] | None = None,
        boundary_radius: int = 2,
        hard_label_source: dict[str, Any] | None = None,
        hard_boundary_radius: int = 2,
        time_swap_probability: float = 0.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.pseudo_root = Path(pseudo_root)
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.pseudo_root / self.manifest
        self.image_size = int(image_size)
        self.augment = augment
        self.time_swap_probability = float(time_swap_probability)
        if not 0.0 <= self.time_swap_probability <= 1.0:
            raise ValueError("time_swap_probability must be in [0, 1]")
        self.pseudo_confidence_mode = str(pseudo_confidence_mode).lower()
        if self.pseudo_confidence_mode not in {"manifest", "constant"}:
            raise ValueError(
                "pseudo_confidence_mode must be either 'manifest' or 'constant'"
            )
        self.pseudo_confidence_value = float(pseudo_confidence_value)
        if not 0.0 <= self.pseudo_confidence_value <= 1.0:
            raise ValueError("pseudo_confidence_value must be in [0, 1]")
        self.boundary_radius = int(boundary_radius)
        if self.boundary_radius < 0:
            raise ValueError("boundary_radius must be non-negative")
        self.hard_boundary_radius = int(hard_boundary_radius)
        if self.hard_boundary_radius < 0:
            raise ValueError("hard_boundary_radius must be non-negative")
        self.boundary_sources = [
            BoundaryMaskSource(**source) for source in (boundary_sources or [])
        ]
        self.hard_label_source = (
            HardLabelSource(**hard_label_source) if hard_label_source else None
        )

        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        missing = self.REQUIRED_COLUMNS - set(self.rows[0])
        if self.pseudo_confidence_mode == "manifest":
            missing = missing | ({"pseudo_confidence"} - set(self.rows[0]))
        if missing:
            raise ValueError(f"Pseudo manifest is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        t1 = self._load_data_image(row["t1"], "RGB")
        t2 = self._load_data_image(row["t2"], "RGB")
        pseudo = self._load_pseudo_image(row["pseudo_label"], "L")
        confidence = self._load_confidence_image(row, pseudo.size)
        boundary_masks = self._load_boundary_masks(row["id"])
        hard_label = None
        hard_confidence = None
        if self.hard_label_source is not None:
            hard_label_path, hard_confidence_path = self.hard_label_source.resolve(row["id"])
            hard_label = self._load(hard_label_path.parent, hard_label_path.name, "L")
            hard_confidence = self._load(
                hard_confidence_path.parent,
                hard_confidence_path.name,
                "L",
            )
        if not (t1.size == t2.size == pseudo.size == confidence.size):
            raise ValueError(f"Image/pseudo sizes differ for sample {row['id']}")
        for mask in boundary_masks:
            if mask.size != t1.size:
                raise ValueError(f"Boundary mask size differs for sample {row['id']}")
        if hard_label is not None and hard_label.size != t1.size:
            raise ValueError(f"Hard label size differs for sample {row['id']}")
        if hard_confidence is not None and hard_confidence.size != t1.size:
            raise ValueError(f"Hard confidence size differs for sample {row['id']}")

        if self.augment:
            (
                t1,
                t2,
                pseudo,
                confidence,
                boundary_masks,
                hard_label,
                hard_confidence,
            ) = self._paired_geometry(
                t1,
                t2,
                pseudo,
                confidence,
                boundary_masks,
                hard_label,
                hard_confidence,
            )
            if random.random() < self.time_swap_probability:
                t1, t2 = t2, t1
        sample = {
            "id": row["id"],
            "t1": TF.to_tensor(t1),
            "t2": TF.to_tensor(t2),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "pseudo_label": TF.pil_to_tensor(pseudo).squeeze(0).float() / 255.0,
            "pseudo_confidence": TF.pil_to_tensor(confidence).squeeze(0).float() / 255.0,
            "change_ratio": torch.tensor(
                float(row.get("change_ratio", 0.0)), dtype=torch.float32
            ),
        }
        if self.boundary_sources:
            hard_mask = self._merge_binary_masks(boundary_masks)
            sample["boundary_target"] = self._boundary_from_mask(
                hard_mask,
                radius=self.boundary_radius,
            )
            sample["boundary_hard_mask"] = hard_mask
        if hard_label is not None and hard_confidence is not None:
            hard_mask = (TF.pil_to_tensor(hard_label).squeeze(0).float() / 255.0) > 0.5
            hard_mask = hard_mask.float()
            sample["hard_label"] = hard_mask
            sample["hard_confidence"] = (
                TF.pil_to_tensor(hard_confidence).squeeze(0).float() / 255.0
            )
            sample["hard_boundary_target"] = self._boundary_from_mask(
                hard_mask,
                radius=self.hard_boundary_radius,
            )
        return sample

    def positive_class_weight(self) -> float:
        positives = sum(float(row["label"]) > 0.5 for row in self.rows)
        negatives = len(self.rows) - positives
        if positives == 0:
            raise ValueError("Pseudo manifest contains no positive samples")
        return negatives / positives

    def _load_data_image(self, value: str, mode: str) -> Image.Image:
        return self._load(self.data_root, value, mode)

    def _load_pseudo_image(self, value: str, mode: str) -> Image.Image:
        return self._load(self.pseudo_root, value, mode)

    def _load_confidence_image(
        self,
        row: dict[str, str],
        size: tuple[int, int],
    ) -> Image.Image:
        if self.pseudo_confidence_mode == "constant":
            value = int(round(self.pseudo_confidence_value * 255.0))
            return Image.new("L", size, value)
        return self._load_pseudo_image(row["pseudo_confidence"], "L")

    def _load_boundary_masks(self, sample_id: str) -> list[Image.Image]:
        masks = []
        for source in self.boundary_sources:
            path = source.resolve(sample_id)
            if path is None:
                masks.append(Image.new("L", (self.image_size, self.image_size), 0))
            else:
                masks.append(self._load(path.parent, path.name, "L"))
        return masks

    def _load(self, root: Path, value: str, mode: str) -> Image.Image:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        image = Image.open(path).convert(mode)
        if image.size != (self.image_size, self.image_size):
            interpolation = (
                TF.InterpolationMode.BILINEAR
                if mode == "RGB"
                else TF.InterpolationMode.BILINEAR
            )
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=interpolation,
            )
        return image

    @staticmethod
    def _paired_geometry(
        t1: Image.Image,
        t2: Image.Image,
        pseudo: Image.Image,
        confidence: Image.Image,
        boundary_masks: list[Image.Image],
        hard_label: Image.Image | None,
        hard_confidence: Image.Image | None,
    ) -> tuple[
        Image.Image,
        Image.Image,
        Image.Image,
        Image.Image,
        list[Image.Image],
        Image.Image | None,
        Image.Image | None,
    ]:
        if random.random() < 0.5:
            t1, t2 = TF.hflip(t1), TF.hflip(t2)
            pseudo, confidence = TF.hflip(pseudo), TF.hflip(confidence)
            boundary_masks = [TF.hflip(mask) for mask in boundary_masks]
            hard_label = TF.hflip(hard_label) if hard_label is not None else None
            hard_confidence = (
                TF.hflip(hard_confidence) if hard_confidence is not None else None
            )
        if random.random() < 0.5:
            t1, t2 = TF.vflip(t1), TF.vflip(t2)
            pseudo, confidence = TF.vflip(pseudo), TF.vflip(confidence)
            boundary_masks = [TF.vflip(mask) for mask in boundary_masks]
            hard_label = TF.vflip(hard_label) if hard_label is not None else None
            hard_confidence = (
                TF.vflip(hard_confidence) if hard_confidence is not None else None
            )
        return t1, t2, pseudo, confidence, boundary_masks, hard_label, hard_confidence

    @staticmethod
    def _merge_binary_masks(masks: list[Image.Image]) -> torch.Tensor:
        if not masks:
            raise ValueError("No boundary masks were provided")
        tensors = [
            (TF.pil_to_tensor(mask).squeeze(0).float() / 255.0) > 0.5
            for mask in masks
        ]
        return torch.stack(tensors, dim=0).any(dim=0).float()

    @staticmethod
    def _boundary_from_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.float()
        kernel = radius * 2 + 1
        values = mask.unsqueeze(0).unsqueeze(0).float()
        dilated = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=radius)
        eroded = 1.0 - F.max_pool2d(1.0 - values, kernel_size=kernel, stride=1, padding=radius)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        return boundary.squeeze(0).squeeze(0)
