from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointSoftHardDistillationLoss(nn.Module):
    """Distill soft probabilities and SAM hard masks into one dense logit map."""

    expects_hard_target = True

    def __init__(
        self,
        *,
        soft_distill_weight: float = 1.0,
        hard_dice_weight: float = 0.7,
        hard_bce_weight: float = 0.2,
        hard_boundary_weight: float = 0.2,
        pairwise_affinity_weight: float = 0.05,
        soft_loss_type: str = "sce",
        hard_loss_type: str = "gce",
        generalized_ce_q: float = 0.7,
        sce_alpha: float = 0.5,
        sce_beta: float = 0.5,
        reverse_ce_eps: float = 1e-4,
        image_weight: float = 0.2,
        negative_weight: float = 0.08,
        confidence_floor: float = 0.05,
        dice_eps: float = 1e-6,
        image_pos_weight: float | None = None,
        edge_power: float = 1.0,
    ) -> None:
        super().__init__()
        self.soft_distill_weight = float(soft_distill_weight)
        self.hard_dice_weight = float(hard_dice_weight)
        self.hard_bce_weight = float(hard_bce_weight)
        self.hard_boundary_weight = float(hard_boundary_weight)
        self.pairwise_affinity_weight = float(pairwise_affinity_weight)
        self.soft_loss_type = str(soft_loss_type).lower()
        self.hard_loss_type = str(hard_loss_type).lower()
        self.generalized_ce_q = float(generalized_ce_q)
        if not 0.0 < self.generalized_ce_q <= 1.0:
            raise ValueError("generalized_ce_q must be in (0, 1]")
        self.sce_alpha = float(sce_alpha)
        self.sce_beta = float(sce_beta)
        self.reverse_ce_eps = float(reverse_ce_eps)
        self.image_weight = float(image_weight)
        self.negative_weight = float(negative_weight)
        self.confidence_floor = float(confidence_floor)
        self.dice_eps = float(dice_eps)
        self.edge_power = float(edge_power)
        if image_pos_weight is None:
            self.register_buffer("image_pos_weight", None)
        else:
            self.register_buffer(
                "image_pos_weight",
                torch.tensor(float(image_pos_weight), dtype=torch.float32),
            )

    def forward(
        self,
        output: dict[str, torch.Tensor],
        pseudo_label: torch.Tensor,
        confidence: torch.Tensor,
        image_label: torch.Tensor,
        hard_label: torch.Tensor | None = None,
        hard_confidence: torch.Tensor | None = None,
        hard_boundary_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = output["logits"].float()
        probabilities = logits.sigmoid()
        pseudo_label = pseudo_label.float().clamp(0.0, 1.0)
        confidence = confidence.float().clamp_min(self.confidence_floor)
        image_label = image_label.float()

        soft_distill = self._robust_binary_loss(
            logits, pseudo_label, confidence, self.soft_loss_type
        )
        zero = logits.sum() * 0.0
        hard_robust = zero
        hard_dice = zero
        hard_boundary = zero
        if hard_label is not None:
            hard_label = hard_label.float().clamp(0.0, 1.0)
            if hard_confidence is None:
                hard_confidence = torch.ones_like(hard_label)
            hard_confidence = hard_confidence.float().clamp_min(self.confidence_floor)
            hard_robust = self._robust_binary_loss(
                logits, hard_label, hard_confidence, self.hard_loss_type
            )
            hard_dice = self._weighted_soft_dice_loss(
                probabilities, hard_label, hard_confidence
            )
            if hard_boundary_target is not None and self.hard_boundary_weight > 0:
                boundary_weight = hard_boundary_target.float().clamp(0.0, 1.0) * hard_confidence
                if torch.any(boundary_weight > 0):
                    boundary_map = F.binary_cross_entropy_with_logits(
                        logits, hard_label, reduction="none"
                    )
                    hard_boundary = (
                        boundary_map * boundary_weight
                    ).sum() / boundary_weight.sum().clamp_min(1.0)

        image_loss = F.binary_cross_entropy_with_logits(
            output["image_logits"].float(), image_label, pos_weight=self.image_pos_weight
        )
        negative = image_label < 0.5
        negative_loss = probabilities[negative].mean() if negative.any() else zero

        pairwise_affinity = zero
        edge_strength = self._resolve_edge_strength(output, probabilities)
        if edge_strength is not None and self.pairwise_affinity_weight > 0:
            homogeneous_weight = (1.0 - edge_strength).clamp(0.0, 1.0).pow(self.edge_power)
            pairwise_affinity = self._edge_gated_pairwise_loss(
                probabilities, homogeneous_weight
            )

        total = (
            self.soft_distill_weight * soft_distill
            + self.hard_bce_weight * hard_robust
            + self.hard_dice_weight * hard_dice
            + self.hard_boundary_weight * hard_boundary
            + self.image_weight * image_loss
            + self.negative_weight * negative_loss
            + self.pairwise_affinity_weight * pairwise_affinity
        )
        return {
            "total": total,
            "soft_distill": soft_distill.detach(),
            "hard_robust": hard_robust.detach(),
            "hard_dice": hard_dice.detach(),
            "hard_boundary": hard_boundary.detach(),
            "image": image_loss.detach(),
            "negative": negative_loss.detach(),
            "pairwise_affinity": pairwise_affinity.detach(),
        }

    def _robust_binary_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        if loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        elif loss_type == "gce":
            loss = self._generalized_cross_entropy(logits.sigmoid(), target)
        elif loss_type == "sce":
            ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            rce = self._reverse_cross_entropy(logits.sigmoid(), target)
            loss = self.sce_alpha * ce + self.sce_beta * rce
        else:
            raise ValueError(f"Unsupported robust binary loss: {loss_type}")
        weight = weight.float().clamp_min(self.confidence_floor)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _generalized_cross_entropy(
        self, probability: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        agreement = (
            target * probability + (1.0 - target) * (1.0 - probability)
        ).clamp_min(1e-6)
        return (1.0 - agreement.pow(self.generalized_ce_q)) / self.generalized_ce_q

    def _reverse_cross_entropy(
        self, probability: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        target = target.clamp(self.reverse_ce_eps, 1.0 - self.reverse_ce_eps)
        return -(
            probability * target.log()
            + (1.0 - probability) * (1.0 - target).log()
        )

    def _weighted_soft_dice_loss(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        dims = (1, 2)
        intersection = (confidence * probabilities * target).sum(dim=dims)
        denominator = (confidence * probabilities).sum(dim=dims) + (
            confidence * target
        ).sum(dim=dims)
        dice = (2.0 * intersection + self.dice_eps) / (denominator + self.dice_eps)
        return 1.0 - dice.mean()

    @staticmethod
    def _resolve_edge_strength(
        output: dict[str, torch.Tensor], probabilities: torch.Tensor
    ) -> torch.Tensor | None:
        edge_strength = output.get("edge_strength")
        if edge_strength is None and "edge_bank" in output:
            edge_bank = output["edge_bank"]
            if isinstance(edge_bank, torch.Tensor) and edge_bank.ndim == 4:
                edge_strength = (
                    edge_bank[:, 3:, :, :].amax(dim=1)
                    if edge_bank.shape[1] > 3
                    else edge_bank[:, -1, :, :]
                )
        if edge_strength is None or not isinstance(edge_strength, torch.Tensor):
            return None
        edge_strength = edge_strength.float().clamp(0.0, 1.0)
        if edge_strength.shape[-2:] != probabilities.shape[-2:]:
            edge_strength = F.interpolate(
                edge_strength.unsqueeze(1),
                size=probabilities.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return edge_strength

    @staticmethod
    def _edge_gated_pairwise_loss(
        probabilities: torch.Tensor, homogeneous_weight: torch.Tensor
    ) -> torch.Tensor:
        p_left = probabilities[:, :, :-1]
        p_right = probabilities[:, :, 1:]
        w_x = torch.minimum(homogeneous_weight[:, :, :-1], homogeneous_weight[:, :, 1:])
        disagreement_x = p_left * (1.0 - p_right) + p_right * (1.0 - p_left)

        p_top = probabilities[:, :-1, :]
        p_bottom = probabilities[:, 1:, :]
        w_y = torch.minimum(homogeneous_weight[:, :-1, :], homogeneous_weight[:, 1:, :])
        disagreement_y = p_top * (1.0 - p_bottom) + p_bottom * (1.0 - p_top)

        loss_x = (disagreement_x * w_x).sum() / w_x.sum().clamp_min(1.0)
        loss_y = (disagreement_y * w_y).sum() / w_y.sum().clamp_min(1.0)
        return 0.5 * (loss_x + loss_y)
