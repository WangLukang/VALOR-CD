from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from .dinov2_cam import DEFAULT_DINOV2, _load_transformers_classes
from .normalization import IMAGENET_MEAN, IMAGENET_STD


class DINOv2StrongCAM(nn.Module):
    """Frozen DINOv2 with rich temporal fusion and Top-K MIL."""

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV2,
        pretrained: bool = True,
        local_files_only: bool = False,
        freeze_backbone: bool = True,
        embedding_dim: int = 256,
        topk_ratio: float = 0.05,
        backbone_image_size: int = 518,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must be in (0, 1]")
        config_class, model_class = _load_transformers_classes()
        if pretrained:
            self.backbone = model_class.from_pretrained(
                model_id, local_files_only=local_files_only
            )
        else:
            config = config_class(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                patch_size=patch_size,
                image_size=backbone_image_size,
            )
            self.backbone = model_class(config)

        self.freeze_backbone = freeze_backbone
        self.topk_ratio = float(topk_ratio)
        channels = int(self.backbone.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Conv2d(channels * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "DINOv2StrongCAM":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, Any]:
        if t1.shape != t2.shape:
            raise ValueError("T1 and T2 must have identical shapes")
        batch_size, _, height, width = t1.shape
        patch_size = int(self.backbone.config.patch_size)
        grid_height = height // patch_size
        grid_width = width // patch_size

        paired = torch.cat((t1, t2), dim=0)
        paired = (paired - self.image_mean) / self.image_std
        features = self._extract(paired)
        feature_t1, feature_t2 = features.split(batch_size, dim=0)
        fused = torch.cat(
            (
                feature_t1,
                feature_t2,
                torch.abs(feature_t2 - feature_t1),
                feature_t1 * feature_t2,
            ),
            dim=1,
        )
        embedding = self.projection(fused)
        raw_cam = self.classifier(embedding)
        logits = self._topk_logit(raw_cam)
        probability_cam = raw_cam.sigmoid()
        return {
            "logits": logits,
            "raw_cam": raw_cam.squeeze(1),
            "cam": probability_cam.squeeze(1),
            "cam_score": probability_cam.squeeze(1),
            "difference": embedding,
            "feature_t1": feature_t1,
            "feature_t2": feature_t2,
            "fused_features": fused,
        }

    def _extract(self, image: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            output = self.backbone(pixel_values=image)
            tokens = output.last_hidden_state[:, 1:, :]
        batch_size = image.shape[0]
        patch_size = int(self.backbone.config.patch_size)
        height = image.shape[-2] // patch_size
        width = image.shape[-1] // patch_size
        return tokens.transpose(1, 2).reshape(batch_size, -1, height, width)

    def _topk_logit(self, raw_cam: torch.Tensor) -> torch.Tensor:
        flat = raw_cam.flatten(1)
        count = max(1, round(flat.shape[1] * self.topk_ratio))
        return flat.topk(count, dim=1).values.mean(dim=1)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for parameter in self.backbone.parameters() if parameter.requires_grad)

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.projection.parameters()
        yield from self.classifier.parameters()
