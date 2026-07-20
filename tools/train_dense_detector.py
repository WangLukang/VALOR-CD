from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.data import ChangePairDataset, PseudoLabelChangeDataset  # noqa: E402
from valor_cd.evaluation import BinaryMetricAccumulator  # noqa: E402
from valor_cd.experiment import (  # noqa: E402
    choose_device,
    load_yaml,
    resolve_from,
    set_seed,
    write_json,
)
from valor_cd.losses import JointSoftHardDistillationLoss  # noqa: E402
from valor_cd.models import build_dense_change_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a P6 dense student from P5 pseudo labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--hard-label-dir")
    parser.add_argument("--hard-label-manifest")
    parser.add_argument("--no-train-shuffle", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument(
        "--tiny-random-model",
        action="store_true",
        help="Use a tiny random DINOv2 config for quick implementation smoke tests.",
    )
    return parser.parse_args()


def load_p6_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(resolve_from(ROOT, args.config))
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config.setdefault("data", {})["num_workers"] = args.num_workers
    if args.hard_label_dir:
        hard_source = config.setdefault("data", {}).setdefault("hard_label_source", {})
        hard_source["root"] = args.hard_label_dir
    if args.hard_label_manifest:
        hard_source = config.setdefault("data", {}).setdefault("hard_label_source", {})
        hard_source["manifest"] = args.hard_label_manifest
    if args.tiny_random_model:
        config["model"].update(
            {
                "pretrained": False,
                "freeze_backbone": True,
                "selected_layers": [1, 2],
                "fusion_channels": 32,
                "decoder_channels": 32,
                "high_res_channels": 16,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "backbone_image_size": int(config.get("data", {}).get("image_size", 224)),
            }
        )
        if (
            config["model"].get("architecture") == "dinov2_edge_guided_dense_change"
            or bool(config["model"].get("use_edge_guidance", False))
        ):
            config["model"]["edge_channels"] = 8
    if args.no_warm_start:
        config.setdefault("train", {}).pop("warm_start", None)
    return config


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def load_p5_summary(pseudo_dir: Path) -> dict[str, Any]:
    summary_path = pseudo_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"P5 summary does not exist: {summary_path}\n"
            "Run P5 Soft Pseudo Labels first."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_datasets(config: dict[str, Any]) -> tuple[PseudoLabelChangeDataset, ChangePairDataset]:
    data_config = dict(config["data"])
    pseudo_dir = resolve_from(ROOT, data_config["pseudo_dir"])
    p5_summary = load_p5_summary(pseudo_dir)
    supervised_data_config = load_yaml(resolve_from(ROOT, data_config["supervised_config"]))
    data_root_value = data_config.get("data_root") or p5_summary.get("data_root")
    if data_root_value is None and p5_summary.get("source_dir"):
        source_summary_path = Path(p5_summary["source_dir"]) / "summary.json"
        if source_summary_path.is_file():
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            data_root_value = source_summary.get("data_root")
    data_root = (
        resolve_from(ROOT, data_root_value)
        if data_root_value is not None
        else resolve_from(ROOT, supervised_data_config["root"])
    )
    pseudo_manifest = data_config.get(
        "pseudo_manifest",
        f"{p5_summary.get('split', 'train')}_pseudo_manifest.csv",
    )
    image_size = int(data_config.get("image_size", 224))
    train_dataset = PseudoLabelChangeDataset(
        data_root,
        pseudo_dir,
        pseudo_manifest,
        image_size=image_size,
        augment=bool(data_config.get("augment", True)),
        pseudo_confidence_mode=str(data_config.get("pseudo_confidence_mode", "manifest")),
        pseudo_confidence_value=float(data_config.get("pseudo_confidence_value", 1.0)),
        boundary_sources=data_config.get("boundary_sources"),
        boundary_radius=int(data_config.get("boundary_radius", 2)),
        hard_label_source=data_config.get("hard_label_source"),
        hard_boundary_radius=int(data_config.get("hard_boundary_radius", 2)),
    )

    val_split = str(data_config.get("val_split", "val"))
    val_dataset = ChangePairDataset(
        resolve_from(ROOT, supervised_data_config["root"]),
        supervised_data_config["manifests"][val_split],
        image_size=int(supervised_data_config.get("image_size", image_size)),
        augment=False,
        return_mask=True,
    )
    return train_dataset, val_dataset


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> AdamW:
    train_config = dict(config["train"])
    head_parameters = list(model.head_parameters())
    encoder_parameters = list(model.encoder_parameters())
    groups: list[dict[str, Any]] = [
        {
            "params": head_parameters,
            "lr": float(train_config["lr"]),
            "weight_decay": float(train_config.get("weight_decay", 0.01)),
        }
    ]
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": float(train_config.get("encoder_lr", train_config["lr"])),
                "weight_decay": float(train_config.get("encoder_weight_decay", 0.01)),
            }
        )
    return AdamW(groups)


def build_loss(config: dict[str, Any], train_dataset: PseudoLabelChangeDataset, device: torch.device):
    loss_config = dict(config["train"].get("loss", {}))
    loss_name = str(loss_config.pop("name", "joint_soft_hard_distillation"))
    if bool(loss_config.pop("use_image_pos_weight", True)):
        loss_config["image_pos_weight"] = train_dataset.positive_class_weight()
    if loss_name != "joint_soft_hard_distillation":
        raise ValueError(
            "The public pipeline supports only joint_soft_hard_distillation; "
            f"received {loss_name!r}."
        )
    return JointSoftHardDistillationLoss(**loss_config).to(device)


def build_ema_teacher_if_configured(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.nn.Module | None:
    ema_config = dict(config.get("train", {}).get("ema_consistency", {}))
    if not bool(ema_config.get("enabled", False)):
        return None
    teacher = copy.deepcopy(model)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    decay: float,
) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)
    teacher.eval()


def ema_consistency_weight(ema_config: dict[str, Any], epoch: int) -> float:
    base_weight = float(ema_config.get("weight", 0.0))
    warmup_epochs = int(ema_config.get("warmup_epochs", 0))
    ramp_epochs = int(ema_config.get("ramp_epochs", 0))
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return base_weight
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / ramp_epochs))
    return base_weight * progress


def compute_ema_consistency_loss(
    student_output: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
    pseudo_label: torch.Tensor,
    confidence: torch.Tensor,
    image_label: torch.Tensor,
    ema_config: dict[str, Any],
    epoch: int,
) -> dict[str, torch.Tensor]:
    weight = ema_consistency_weight(ema_config, epoch)
    logits = student_output["logits"].float()
    zero = logits.sum() * 0.0
    if weight <= 0:
        return {"total": zero, "ema_consistency": zero.detach()}

    student_prob = logits.sigmoid()
    teacher_prob = teacher_output["probabilities"].detach().float()
    teacher_confidence = (teacher_prob - 0.5).abs().mul(2.0).clamp(0.0, 1.0)
    confidence_threshold = float(ema_config.get("teacher_confidence_threshold", 0.65))
    mask = teacher_confidence >= confidence_threshold
    if bool(ema_config.get("positive_only", False)):
        mask = mask & (image_label.view(-1, 1, 1) > 0.5)
    if not bool(ema_config.get("include_negative_images", True)):
        mask = mask & (image_label.view(-1, 1, 1) > 0.5)

    pixel_weight = teacher_confidence
    if bool(ema_config.get("use_pseudo_confidence", True)):
        pixel_weight = pixel_weight * confidence.float().clamp(0.0, 1.0)
    pixel_weight = pixel_weight * mask.float()
    if torch.count_nonzero(pixel_weight) == 0:
        return {"total": zero, "ema_consistency": zero.detach()}

    loss_type = str(ema_config.get("loss", "mse")).lower()
    if loss_type == "bce":
        consistency = F.binary_cross_entropy_with_logits(
            logits,
            teacher_prob,
            reduction="none",
        )
    elif loss_type == "kl":
        eps = 1e-6
        student_prob = student_prob.clamp(eps, 1.0 - eps)
        teacher_prob = teacher_prob.clamp(eps, 1.0 - eps)
        consistency = (
            teacher_prob * (teacher_prob.log() - student_prob.log())
            + (1.0 - teacher_prob)
            * ((1.0 - teacher_prob).log() - (1.0 - student_prob).log())
        )
    elif loss_type == "mse":
        consistency = (student_prob - teacher_prob).square()
    else:
        raise ValueError(f"Unsupported EMA consistency loss: {loss_type}")

    consistency = (consistency * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    total = float(weight) * consistency
    return {
        "total": total,
        "ema_consistency": consistency.detach(),
        "ema_weighted": total.detach(),
        "ema_pixel_ratio": mask.float().mean().detach(),
    }


class LossMeter:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, losses: dict[str, torch.Tensor], batch_size: int) -> None:
        self.count += batch_size
        for name, value in losses.items():
            self.totals[name] += float(value.detach().cpu()) * batch_size

    def compute(self) -> dict[str, float]:
        return {
            name: value / max(self.count, 1)
            for name, value in sorted(self.totals.items())
        }


def train_one_epoch(
    model: torch.nn.Module,
    ema_teacher: torch.nn.Module | None,
    loader: DataLoader,
    criterion: ConfidenceWeightedDensePseudoLoss,
    optimizer: AdamW,
    scaler: Any,
    device: torch.device,
    *,
    mixed_precision: bool,
    image_threshold: float,
    max_batches: int | None,
    stage_name: str,
    epoch: int,
    ema_config: dict[str, Any],
) -> dict[str, Any]:
    model.train()
    if ema_teacher is not None:
        ema_teacher.eval()
    losses = LossMeter()
    image_metrics = BinaryMetricAccumulator(image_threshold)
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"train {stage_name}",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        t1 = batch["t1"].to(device, non_blocking=True)
        t2 = batch["t2"].to(device, non_blocking=True)
        pseudo = batch["pseudo_label"].to(device, non_blocking=True)
        confidence = batch["pseudo_confidence"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        boundary = (
            batch["boundary_target"].to(device, non_blocking=True)
            if "boundary_target" in batch
            else None
        )
        hard_label = (
            batch["hard_label"].to(device, non_blocking=True)
            if "hard_label" in batch
            else None
        )
        hard_confidence = (
            batch["hard_confidence"].to(device, non_blocking=True)
            if "hard_confidence" in batch
            else None
        )
        hard_boundary = (
            batch["hard_boundary_target"].to(device, non_blocking=True)
            if "hard_boundary_target" in batch
            else boundary
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        ):
            output = model(t1, t2)
            teacher_output = None
            if ema_teacher is not None:
                with torch.no_grad():
                    teacher_output = ema_teacher(t1, t2)
            if bool(getattr(criterion, "expects_hard_target", False)):
                batch_losses = criterion(
                    output,
                    pseudo,
                    confidence,
                    labels,
                    hard_label,
                    hard_confidence,
                    hard_boundary,
                )
            elif bool(getattr(criterion, "expects_boundary_target", False)):
                batch_losses = criterion(output, pseudo, confidence, labels, boundary)
            else:
                batch_losses = criterion(output, pseudo, confidence, labels)
            if teacher_output is not None:
                ema_losses = compute_ema_consistency_loss(
                    output,
                    teacher_output,
                    pseudo,
                    confidence,
                    labels,
                    ema_config,
                    epoch,
                )
                batch_losses = dict(batch_losses)
                batch_losses["total"] = batch_losses["total"] + ema_losses["total"]
                for name, value in ema_losses.items():
                    if name != "total":
                        batch_losses[name] = value
        scaler.scale(batch_losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema_teacher is not None:
            update_ema_teacher(
                ema_teacher,
                model,
                decay=float(ema_config.get("decay", 0.99)),
            )

        losses.update(batch_losses, t1.shape[0])
        image_metrics.update(output["image_probability"].detach().cpu(), labels.detach().cpu())
        progress.set_postfix(
            loss=f"{losses.compute().get('total', 0.0):.4f}",
            img_f1=f"{image_metrics.compute()['f1']:.4f}",
        )

    return {
        "loss": losses.compute(),
        "image": {**image_metrics.counts(), **image_metrics.compute()},
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool,
    pixel_threshold: float,
    image_threshold: float,
    max_batches: int | None,
    stage_name: str,
) -> dict[str, Any]:
    model.eval()
    pixel_metrics = BinaryMetricAccumulator(pixel_threshold)
    image_metrics = BinaryMetricAccumulator(image_threshold)
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"validate {stage_name}",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        t1 = batch["t1"].to(device, non_blocking=True)
        t2 = batch["t2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        ):
            output = model(t1, t2)
        probabilities = output["probabilities"].detach().cpu()
        pixel_metrics.update(probabilities, batch["mask"])
        image_metrics.update(output["image_probability"].detach().cpu(), labels.detach().cpu())
        progress.set_postfix(pixel_f1=f"{pixel_metrics.compute()['f1']:.4f}")
    return {
        "pixel": {**pixel_metrics.counts(), **pixel_metrics.compute()},
        "image": {**image_metrics.counts(), **image_metrics.compute()},
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_teacher: torch.nn.Module | None = None,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    config: dict[str, Any],
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )
    if ema_teacher is not None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["ema_model"] = {
            key: value.detach().cpu()
            for key, value in ema_teacher.state_dict().items()
        }
        torch.save(checkpoint, path)


def load_warm_start_if_configured(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    warm_start = config.get("train", {}).get("warm_start")
    if not warm_start:
        return None
    checkpoint_path = resolve_from(ROOT, warm_start)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "source_epoch": checkpoint.get("epoch"),
    }


def main() -> None:
    args = parse_args()
    config = load_p6_config(args)
    set_seed(int(config.get("seed", 42)))
    output_dir = resolve_from(ROOT, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_datasets(config)
    data_config = config["data"]
    train_loader = make_loader(
        train_dataset,
        int(data_config["batch_size"]),
        int(data_config.get("num_workers", 0)),
        shuffle=not args.no_train_shuffle,
    )
    val_loader = make_loader(
        val_dataset,
        int(data_config.get("val_batch_size", data_config["batch_size"])),
        int(data_config.get("num_workers", 0)),
        shuffle=False,
    )

    device = choose_device()
    model = build_dense_change_model(
        config["model"],
        pretrained_override=False if args.no_pretrained else None,
    ).to(device)
    warm_start_info = load_warm_start_if_configured(model, config)
    if warm_start_info is not None:
        print(
            "Warm-started from "
            f"{warm_start_info['checkpoint']} "
            f"(missing={len(warm_start_info['missing_keys'])}, "
            f"unexpected={len(warm_start_info['unexpected_keys'])})"
        )
    ema_config = dict(config.get("train", {}).get("ema_consistency", {}))
    ema_teacher = build_ema_teacher_if_configured(model, config)
    optimizer = build_optimizer(model, config)
    epochs = int(config["train"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = build_loss(config, train_dataset, device)
    mixed_precision = bool(config["train"].get("mixed_precision", True))
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=mixed_precision and device.type == "cuda",
    )
    pixel_threshold = float(config["evaluation"].get("pixel_threshold", 0.5))
    image_threshold = float(config["evaluation"].get("image_threshold", 0.5))

    print(
        f"Training {config.get('stage_name', 'soft-hard-distillation')} dense detector on "
        f"{device}: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={data_config['batch_size']}"
    )
    if ema_teacher is not None:
        print(
            "EMA consistency enabled: "
            f"decay={float(ema_config.get('decay', 0.99)):.4f}, "
            f"weight={float(ema_config.get('weight', 0.0)):.4f}"
        )

    best_f1 = -1.0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            ema_teacher,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            mixed_precision=mixed_precision,
            image_threshold=image_threshold,
            max_batches=args.max_train_batches,
            stage_name=str(config.get("stage_name", "soft-hard-distillation")),
            epoch=epoch,
            ema_config=ema_config,
        )
        val_model = (
            ema_teacher
            if ema_teacher is not None
            and bool(ema_config.get("evaluate_teacher", True))
            else model
        )
        val_metrics = evaluate(
            val_model,
            val_loader,
            device,
            mixed_precision=mixed_precision,
            pixel_threshold=pixel_threshold,
            image_threshold=image_threshold,
            max_batches=args.max_val_batches,
            stage_name=str(config.get("stage_name", "soft-hard-distillation")),
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr(),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        val_pixel = val_metrics["pixel"]
        train_image = train_metrics["image"]
        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"loss={train_metrics['loss']['total']:.4f}, "
            f"train_img_P/R/F1={train_image['precision']:.4f}/"
            f"{train_image['recall']:.4f}/{train_image['f1']:.4f}, "
            f"val_pixel_P/R/F1/IoU={val_pixel['precision']:.4f}/"
            f"{val_pixel['recall']:.4f}/{val_pixel['f1']:.4f}/"
            f"{val_pixel['iou']:.4f}"
        )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            ema_teacher=ema_teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        if val_pixel["f1"] > best_f1:
            best_f1 = float(val_pixel["f1"])
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                ema_teacher=ema_teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

    summary = {
        "config": config,
        "device": str(device),
        "epochs": epochs,
        "seconds": time.time() - started,
        "best_val_f1": best_f1,
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "warm_start": warm_start_info,
        "ema_consistency": ema_config if ema_teacher is not None else None,
        "history": history,
    }
    write_json(output_dir / "summary.json", summary)
    if history:
        with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "epoch",
                "loss",
                "train_image_precision",
                "train_image_recall",
                "train_image_f1",
                "val_pixel_precision",
                "val_pixel_recall",
                "val_pixel_f1",
                "val_pixel_iou",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in history:
                writer.writerow(
                    {
                        "epoch": item["epoch"],
                        "loss": item["train"]["loss"]["total"],
                        "train_image_precision": item["train"]["image"]["precision"],
                        "train_image_recall": item["train"]["image"]["recall"],
                        "train_image_f1": item["train"]["image"]["f1"],
                        "val_pixel_precision": item["val"]["pixel"]["precision"],
                        "val_pixel_recall": item["val"]["pixel"]["recall"],
                        "val_pixel_f1": item["val"]["pixel"]["f1"],
                        "val_pixel_iou": item["val"]["pixel"]["iou"],
                    }
                )
    print(f"Best val pixel F1: {best_f1:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
