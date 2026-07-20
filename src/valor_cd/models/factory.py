from __future__ import annotations

from typing import Any

import torch.nn as nn

from .dinov2_dense_change import DINOv2DenseChangeDetector
from .dinov2_strong_cam import DINOv2StrongCAM


def build_cam_model(config: dict[str, Any], *, pretrained_override: bool | None = None) -> nn.Module:
    parameters = dict(config)
    architecture = parameters.pop("architecture", None)
    if pretrained_override is not None:
        parameters["pretrained"] = pretrained_override
    if architecture == "dinov2_strong_cam":
        return DINOv2StrongCAM(**parameters)
    raise ValueError(f"Unsupported CAM architecture in the clean pipeline: {architecture}")


def build_dense_change_model(config: dict[str, Any], *, pretrained_override: bool | None = None) -> nn.Module:
    parameters = dict(config)
    architecture = parameters.pop("architecture", None)
    if pretrained_override is not None:
        parameters["pretrained"] = pretrained_override
    if architecture == "dinov2_edge_guided_dense_change":
        parameters.setdefault("use_edge_guidance", True)
        return DINOv2DenseChangeDetector(**parameters)
    raise ValueError(f"Unsupported dense change architecture in the clean pipeline: {architecture}")
