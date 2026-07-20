from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from .normalization import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_DINOV2 = "facebook/dinov2-base"


def _load_transformers_classes() -> tuple[type[Any], type[Any]]:
    try:
        from transformers import Dinov2Config, Dinov2Model
    except ImportError as error:
        raise ImportError(
            "DINOv2 requires Transformers. Install project dependencies inside "
            "the wscd Conda environment."
        ) from error
    return Dinov2Config, Dinov2Model


class DINOv2MultiLayerCAM(nn.Module):
    """CAM baseline built from multiple DINOv2 transformer layers."""

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV2,
        pretrained: bool = True,
        local_files_only: bool = False,
        selected_layers: list[int] | tuple[int, ...] = (3, 6, 9, 12),
        fusion_channels: int = 256,
        backbone_image_size: int = 518,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        config_class, model_class = _load_transformers_classes()
        self.model_id = model_id
        self.selected_layers = tuple(selected_layers)
        if not self.selected_layers:
            raise ValueError("selected_layers must not be empty")

        if pretrained:
            try:
                self.backbone = model_class.from_pretrained(
                    model_id, local_files_only=local_files_only
                )
            except OSError as error:
                raise RuntimeError(
                    f"Could not load DINOv2 weights from {model_id}. Check the "
                    "network connection or provide a local Transformers model directory."
                ) from error
        else:
            config = config_class(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                patch_size=patch_size,
                image_size=backbone_image_size,
            )
            self.backbone = model_class(config)

        depth = self.backbone.config.num_hidden_layers
        invalid = [index for index in self.selected_layers if index < 1 or index > depth]
        if invalid:
            raise ValueError(f"selected_layers must be between 1 and {depth}: {invalid}")

        hidden_size = self.backbone.config.hidden_size
        self.projections = nn.ModuleList(
            nn.Conv2d(hidden_size, fusion_channels, kernel_size=1, bias=False)
            for _ in self.selected_layers
        )
        self.classifier = nn.Conv2d(fusion_channels, 1, kernel_size=1, bias=False)
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        self._initialize_head()

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, Any]:
        if t1.shape != t2.shape:
            raise ValueError("T1 and T2 must have identical shapes")
        batch_size, _, height, width = t1.shape
        patch_size = int(self.backbone.config.patch_size)
        grid_height = height // patch_size
        grid_width = width // patch_size

        paired = torch.cat((t1, t2), dim=0)
        paired = (paired - self.image_mean) / self.image_std
        output = self.backbone(pixel_values=paired, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("DINOv2 did not return hidden states")

        layer_differences = []
        projected_differences = []
        for layer_index, projection in zip(self.selected_layers, self.projections):
            hidden = self.backbone.layernorm(output.hidden_states[layer_index])
            patch_tokens = hidden[:, 1:, :]
            feature = patch_tokens.transpose(1, 2).reshape(
                batch_size * 2, -1, grid_height, grid_width
            )
            feature_t1, feature_t2 = feature.split(batch_size, dim=0)
            difference = torch.abs(feature_t2 - feature_t1)
            layer_differences.append(difference)
            projected_differences.append(projection(difference))

        fused_difference = torch.stack(projected_differences, dim=0).mean(dim=0)
        raw_cam = self.classifier(fused_difference)
        logits = raw_cam.mean(dim=(-2, -1)).flatten()
        normalized_cam = self._normalize_cam(torch.relu(raw_cam))
        probability = logits.sigmoid().view(-1, 1, 1, 1)
        return {
            "logits": logits,
            "raw_cam": raw_cam.squeeze(1),
            "cam": normalized_cam.squeeze(1),
            "cam_score": (normalized_cam * probability).squeeze(1),
            "difference": fused_difference,
            "layer_differences": tuple(layer_differences),
        }

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.backbone.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.projections.parameters()
        yield from self.classifier.parameters()

    def _initialize_head(self) -> None:
        for projection in self.projections:
            nn.init.kaiming_normal_(projection.weight, mode="fan_out", nonlinearity="linear")
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)

    @staticmethod
    def _normalize_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        minimum = cam.amin(dim=(-2, -1), keepdim=True)
        maximum = cam.amax(dim=(-2, -1), keepdim=True)
        return (cam - minimum) / (maximum - minimum + eps)
