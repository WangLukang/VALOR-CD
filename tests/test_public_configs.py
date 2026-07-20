from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class PublicConfigurationTest(unittest.TestCase):
    def load(self, dataset: str, stage: str) -> dict:
        path = ROOT / "configs" / dataset / f"{stage}.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_frozen_main_configuration(self) -> None:
        for dataset in ("whu", "levir"):
            with self.subTest(dataset=dataset):
                stage1 = self.load(dataset, "stage1")
                stage2 = self.load(dataset, "stage2")
                stage3 = self.load(dataset, "stage3")

                self.assertEqual(stage1["weak_teacher"]["model"]["model_id"], "facebook/dinov2-base")
                self.assertEqual(stage1["weak_teacher"]["model"]["topk_ratio"], 0.05)
                self.assertEqual(stage1["weak_teacher"]["train"]["epochs"], 10)
                self.assertEqual(stage1["sam_pseudo"]["model_type"], "vit_h")
                self.assertEqual(stage1["sam_pseudo"]["image_mode"], "t2")
                self.assertEqual(stage2["detector"]["model"]["model_id"], "facebook/dinov2-base")
                self.assertEqual(stage2["detector"]["train"]["epochs"], 50)
                self.assertEqual(stage3["pixel_threshold"], 0.7)

    def test_public_configs_do_not_contain_absolute_windows_paths(self) -> None:
        for path in (ROOT / "configs").rglob("*.yaml"):
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                self.assertNotRegex(text, r"[A-Za-z]:\\")


if __name__ == "__main__":
    unittest.main()
