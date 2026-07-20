from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2_cam import DEFAULT_DINOV2, _load_transformers_classes
from .normalization import IMAGENET_MEAN, IMAGENET_STD

class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DINOv2DenseChangeDetector(nn.Module):
    """P6 dense change detector with an optional edge-guidance branch."""

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV2,
        pretrained: bool = True,
        local_files_only: bool = False,
        freeze_backbone: bool = True,
        selected_layers: list[int] | tuple[int, ...] = (6, 9, 12),
        fusion_channels: int = 128,
        decoder_channels: int = 128,
        high_res_channels: int = 32,
        topk_ratio: float = 0.05,
        use_boundary_head: bool = False,
        use_edge_guidance: bool = False,
        edge_channels: int = 32,
        edge_residual_init: float = 0.05,
        edge_attention_init: float = 0.05,
        backbone_image_size: int = 518,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        if not selected_layers:
            raise ValueError("selected_layers must not be empty")
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

        depth = int(self.backbone.config.num_hidden_layers)
        invalid = [index for index in selected_layers if index < 1 or index > depth]
        if invalid:
            raise ValueError(f"selected_layers must be between 1 and {depth}: {invalid}")

        self.selected_layers = tuple(int(index) for index in selected_layers)
        self.freeze_backbone = bool(freeze_backbone)
        self.topk_ratio = float(topk_ratio)
        self.use_edge_guidance = bool(use_edge_guidance)
        hidden = int(self.backbone.config.hidden_size)
        self.layer_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden * 4, fusion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.GELU(),
            )
            for _ in self.selected_layers
        )
        self.decoder = nn.Sequential(
            ConvNormAct(fusion_channels, decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
        )
        self.high_res_stem = nn.Sequential(
            ConvNormAct(9, high_res_channels),
            ConvNormAct(high_res_channels, high_res_channels),
        )
        self.refiner = nn.Sequential(
            ConvNormAct(decoder_channels + high_res_channels, decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels // 2),
        )
        refined_channels = decoder_channels // 2
        if self.use_edge_guidance:
            self.edge_encoder = nn.Sequential(
                ConvNormAct(3, edge_channels),
                ConvNormAct(edge_channels, edge_channels),
                ConvNormAct(edge_channels, edge_channels),
            )
            self.edge_to_refined = nn.Sequential(
                ConvNormAct(edge_channels, refined_channels),
                nn.Conv2d(refined_channels, refined_channels, kernel_size=1),
            )
            self.edge_gate = nn.Sequential(
                nn.Conv2d(edge_channels, refined_channels, kernel_size=1),
                nn.Sigmoid(),
            )
            self.edge_attention_head = nn.Conv2d(edge_channels, 1, kernel_size=1)
            self.edge_residual_scale = nn.Parameter(
                torch.tensor(float(edge_residual_init), dtype=torch.float32)
            )
            self.edge_attention_scale = nn.Parameter(
                torch.tensor(float(edge_attention_init), dtype=torch.float32)
            )
            self.register_buffer(
                "edge_gray_weights",
                torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
            )
        else:
            self.edge_encoder = None
            self.edge_to_refined = None
            self.edge_gate = None
            self.edge_attention_head = None
        self.mask_head = nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        self.boundary_head = (
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
            if use_boundary_head
            else None
        )
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "DINOv2DenseChangeDetector":
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
        output = self._extract_hidden_states(paired)
        if output.hidden_states is None:
            raise RuntimeError("DINOv2 did not return hidden states")

        projected = []
        fused_features = []
        for layer_index, projection in zip(self.selected_layers, self.layer_projections):
            hidden = self.backbone.layernorm(output.hidden_states[layer_index])
            tokens = hidden[:, 1:, :]
            feature = tokens.transpose(1, 2).reshape(
                batch_size * 2, -1, grid_height, grid_width
            )
            feature_t1, feature_t2 = feature.split(batch_size, dim=0)
            fused = torch.cat(
                (
                    feature_t1,
                    feature_t2,
                    torch.abs(feature_t2 - feature_t1),
                    feature_t1 * feature_t2,
                ),
                dim=1,
            )
            fused_features.append(fused)
            projected.append(projection(fused))

        low_res = torch.stack(projected, dim=0).mean(dim=0)
        decoded = self.decoder(low_res)
        decoded = F.interpolate(
            decoded,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        high_res = self.high_res_stem(torch.cat((t1, t2, torch.abs(t2 - t1)), dim=1))
        refined = self.refiner(torch.cat((decoded, high_res), dim=1))
        edge_bank = None
        edge_gate = None
        edge_attention = None
        edge_strength = None
        if self.use_edge_guidance:
            if (
                self.edge_encoder is None
                or self.edge_to_refined is None
                or self.edge_gate is None
                or self.edge_attention_head is None
            ):
                raise RuntimeError("Edge guidance modules were not initialized")
            edge_bank = self._edge_guidance_bank(t1, t2)
            edge_strength = self._edge_strength_from_bank(edge_bank)
            edge_features = self.edge_encoder(edge_bank)
            edge_delta = self.edge_to_refined(edge_features)
            edge_gate = self.edge_gate(edge_features)
            edge_attention = self.edge_attention_head(edge_features).sigmoid()
            refined = refined + self.edge_residual_scale * edge_gate * edge_delta
            refined = refined * (1.0 + self.edge_attention_scale * edge_attention)
        logits = self.mask_head(refined).squeeze(1)
        probabilities = logits.sigmoid()
        image_logits = self._topk_logit(logits)
        result = {
            "logits": logits,
            "probabilities": probabilities,
            "image_logits": image_logits,
            "image_probability": image_logits.sigmoid(),
            "low_res_difference": low_res,
            "fused_features": tuple(fused_features),
        }
        if edge_bank is not None and edge_gate is not None:
            result.update(
                {
                    "edge_bank": edge_bank.detach(),
                    "edge_strength": edge_strength.detach() if edge_strength is not None else None,
                    "edge_gate": edge_gate.detach(),
                    "edge_attention": edge_attention.detach()
                    if edge_attention is not None
                    else None,
                    "edge_residual_scale": self.edge_residual_scale.detach(),
                    "edge_attention_scale": self.edge_attention_scale.detach(),
                }
            )
        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(refined).squeeze(1)
            result["boundary_logits"] = boundary_logits
            result["boundary_probabilities"] = boundary_logits.sigmoid()
        return result

    def _extract_hidden_states(self, image: torch.Tensor) -> Any:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            return self.backbone(pixel_values=image, output_hidden_states=True)

    def _topk_logit(self, logits: torch.Tensor) -> torch.Tensor:
        flat = logits.flatten(1)
        count = max(1, round(flat.shape[1] * self.topk_ratio))
        return flat.topk(count, dim=1).values.mean(dim=1)

    def _rgb_to_gray(self, image: torch.Tensor) -> torch.Tensor:
        weights = self.edge_gray_weights.to(device=image.device, dtype=image.dtype)
        return (image * weights).sum(dim=1, keepdim=True)

    def _edge_guidance_bank(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        gray_t1 = self._rgb_to_gray(t1)
        gray_t2 = self._rgb_to_gray(t2)
        gray_diff = torch.abs(gray_t2 - gray_t1)
        return torch.cat((gray_t1, gray_t2, gray_diff), dim=1)

    @staticmethod
    def _edge_strength_from_bank(edge_bank: torch.Tensor) -> torch.Tensor:
        if edge_bank.shape[1] >= 3:
            return edge_bank[:, 2, :, :].clamp(0.0, 1.0)
        return edge_bank.amax(dim=1).clamp(0.0, 1.0)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return (
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.layer_projections.parameters()
        yield from self.decoder.parameters()
        yield from self.high_res_stem.parameters()
        yield from self.refiner.parameters()
        if self.edge_encoder is not None:
            yield from self.edge_encoder.parameters()
        if self.edge_to_refined is not None:
            yield from self.edge_to_refined.parameters()
        if self.edge_gate is not None:
            yield from self.edge_gate.parameters()
            yield self.edge_residual_scale
        if self.edge_attention_head is not None:
            yield from self.edge_attention_head.parameters()
            yield self.edge_attention_scale
        yield from self.mask_head.parameters()
        if self.boundary_head is not None:
            yield from self.boundary_head.parameters()
