from __future__ import annotations

from dataclasses import dataclass

import torch


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-12
    return {
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "false_positive_rate": fp / (fp + tn + eps),
    }


@dataclass
class BinaryMetricAccumulator:
    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = scores.detach().reshape(-1).cpu() >= self.threshold
        targets_bool = targets.detach().reshape(-1).cpu() > 0.5
        self.tp += int((predictions & targets_bool).sum())
        self.fp += int((predictions & ~targets_bool).sum())
        self.fn += int((~predictions & targets_bool).sum())
        self.tn += int((~predictions & ~targets_bool).sum())

    def compute(self) -> dict[str, float]:
        return metrics_from_counts(self.tp, self.fp, self.fn, self.tn)

    def counts(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}
