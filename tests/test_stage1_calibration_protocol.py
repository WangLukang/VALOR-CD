from __future__ import annotations

import ast
import importlib.util
import inspect
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "valor_cd_train_p1_cam",
    ROOT / "tools" / "train_p1_cam.py",
)
assert SPEC is not None and SPEC.loader is not None
TRAIN_P1_CAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_P1_CAM)


class MaskGuardBatch(dict):
    def __getitem__(self, key):
        if key == "mask":
            raise AssertionError("Calibration must not access pixel-level masks")
        return super().__getitem__(key)


class ConstantCam(torch.nn.Module):
    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = t1.shape[0]
        return {
            "cam_score": torch.full(
                (batch_size, 2, 2),
                0.8,
                device=t1.device,
            )
        }


class Stage1CalibrationProtocolTest(unittest.TestCase):
    def make_mask_free_loader(self) -> list[MaskGuardBatch]:
        return [
            MaskGuardBatch(
                t1=torch.zeros(1, 3, 8, 10),
                t2=torch.zeros(1, 3, 8, 10),
                label=torch.zeros(1),
            )
        ]

    def test_calibration_loader_uses_train_manifest_without_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            train_manifest = data_root / "train.csv"
            val_manifest = data_root / "val.csv"
            header = "id,t1,t2,label\n"
            train_manifest.write_text(
                header + "train-pair,t1.png,t2.png,0\n",
                encoding="utf-8",
            )
            val_manifest.write_text(
                header + "val-pair,t1.png,t2.png,0\n",
                encoding="utf-8",
            )

            loader = TRAIN_P1_CAM.build_train_calibration_loader(
                data_root=data_root,
                data_config={
                    "manifests": {
                        "train": train_manifest.name,
                        "val": val_manifest.name,
                    }
                },
                image_size=224,
                batch_size=1,
                num_workers=0,
            )

            self.assertEqual(loader.dataset.manifest, train_manifest)
            self.assertFalse(loader.dataset.augment)
            self.assertFalse(loader.dataset.return_mask)
            self.assertIsInstance(loader.sampler, SequentialSampler)

    def test_main_wires_both_calibrations_to_train_loader(self) -> None:
        main_tree = ast.parse(textwrap.dedent(inspect.getsource(TRAIN_P1_CAM.main)))
        calibration_calls = {
            node.func.id: node
            for node in ast.walk(main_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {"calibrate_negative_threshold", "calibrate_negative_candidates"}
        }

        self.assertEqual(
            set(calibration_calls),
            {"calibrate_negative_threshold", "calibrate_negative_candidates"},
        )
        for call in calibration_calls.values():
            self.assertGreaterEqual(len(call.args), 2)
            self.assertIsInstance(call.args[1], ast.Name)
            self.assertEqual(call.args[1].id, "train_calibration_loader")

    def test_negative_threshold_calibration_does_not_access_masks(self) -> None:
        calibration = TRAIN_P1_CAM.calibrate_negative_threshold(
            ConstantCam(),
            self.make_mask_free_loader(),
            torch.device("cpu"),
            False,
            0.995,
            100,
            [196, 224, 252],
            None,
            None,
        )

        self.assertEqual(calibration["negative_images"], 1)
        self.assertEqual(calibration["negative_pixels"], 80)

    def test_negative_candidate_calibration_does_not_access_masks(self) -> None:
        calibration = TRAIN_P1_CAM.calibrate_negative_candidates(
            ConstantCam(),
            self.make_mask_free_loader(),
            torch.device("cpu"),
            False,
            0.5,
            [196, 224, 252],
            None,
            {
                "name": "negative_component_filter",
                "min_component_pixels": 4,
                "connectivity": 8,
                "area_quantile": 0.9,
                "mean_score_quantile": 0.9,
                "peak_score_quantile": 0.9,
                "max_reject_area_ratio": 0.05,
            },
            None,
        )

        self.assertEqual(calibration["negative_images"], 1)
        self.assertEqual(calibration["negative_components"], 1)


if __name__ == "__main__":
    unittest.main()
