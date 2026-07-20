from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.data import ChangePairDataset  # noqa: E402
from valor_cd.evaluation import (  # noqa: E402
    BinaryMetricAccumulator,
    apply_candidate_calibration,
    apply_score_calibration,
    build_negative_component_filter_config,
    extract_candidate_statistics,
)
from valor_cd.experiment import (  # noqa: E402
    choose_device,
    load_yaml,
    resolve_from,
    set_seed,
    write_json,
)
from valor_cd.losses import (  # noqa: E402
    StrongCAMMatchingPriorLoss,
    cam_teacher_consistency_loss,
)
from valor_cd.models import build_cam_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the P1 temporal CAM baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use random encoder weights for an implementation smoke test only.",
    )
    return parser.parse_args()


def make_loader(
    dataset: ChangePairDataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_training_criterion(
    train_config: dict, positive_weight: torch.Tensor
) -> nn.Module:
    loss_config = dict(train_config.get("loss", {"name": "bce"}))
    loss_name = str(loss_config.pop("name", "bce"))
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    if loss_name == "strong_cam_matching_prior":
        return StrongCAMMatchingPriorLoss(
            positive_weight=float(positive_weight.detach().cpu()),
            **loss_config,
        ).to(positive_weight.device)
    raise ValueError(f"Unsupported training loss: {loss_name}")


def compute_losses(
    criterion: nn.Module,
    output: dict,
    labels: torch.Tensor,
    teacher_output: dict | None = None,
    teacher_weight: float = 0.0,
    teacher_positive_only: bool = True,
) -> dict[str, torch.Tensor]:
    if bool(getattr(criterion, "expects_model_output", False)):
        losses = criterion(output, labels)
    else:
        loss = criterion(output["logits"], labels)
        losses = {"total": loss, "bag": loss}
    if teacher_output is not None and teacher_weight > 0:
        teacher_consistency = cam_teacher_consistency_loss(
            output["cam_score"].float(),
            teacher_output["cam_score"].float(),
            labels.float(),
            positive_only=teacher_positive_only,
        )
        losses = dict(losses)
        losses["teacher_consistency"] = teacher_consistency
        losses["total"] = losses["total"] + teacher_weight * teacher_consistency
    return losses


def build_ema_teacher(model: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(model)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    if not 0 <= momentum <= 1:
        raise ValueError("EMA momentum must be in [0, 1]")
    if momentum == 1:
        return
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters(), strict=True
    ):
        teacher_parameter.data.mul_(momentum).add_(
            student_parameter.detach().data,
            alpha=1.0 - momentum,
        )
    for teacher_buffer, student_buffer in zip(
        teacher.buffers(), student.buffers(), strict=True
    ):
        if teacher_buffer.is_floating_point():
            teacher_buffer.data.mul_(momentum).add_(
                student_buffer.detach().data,
                alpha=1.0 - momentum,
            )
        else:
            teacher_buffer.data.copy_(student_buffer.detach().data)


def set_optional_loss_weight(
    criterion: nn.Module,
    attribute: str,
    value: float | None,
) -> None:
    if value is not None and hasattr(criterion, attribute):
        setattr(criterion, attribute, float(value))


@torch.no_grad()
def predict_cam_scores(
    model: nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    output_size: tuple[int, int],
    mixed_precision: bool,
    prediction_scales: list[int] | tuple[int, ...] | None,
    score_calibration: dict | None = None,
    candidate_calibration: dict | None = None,
    pixel_threshold: float | None = None,
) -> torch.Tensor:
    if not prediction_scales:
        with torch.autocast(
            device_type=t1.device.type, dtype=torch.float16, enabled=mixed_precision
        ):
            output = model(t1, t2)
        scores = F.interpolate(
            output["cam_score"].unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        scores = apply_score_calibration(scores, output, output_size, score_calibration)
        return _apply_candidate_calibration_if_enabled(
            scores,
            candidate_calibration,
            pixel_threshold,
        )

    predictions = []
    for scale in prediction_scales:
        size = (int(scale), int(scale))
        scaled_t1 = F.interpolate(t1, size=size, mode="bilinear", align_corners=False)
        scaled_t2 = F.interpolate(t2, size=size, mode="bilinear", align_corners=False)
        with torch.autocast(
            device_type=t1.device.type, dtype=torch.float16, enabled=mixed_precision
        ):
            output = model(scaled_t1, scaled_t2)
        scores = F.interpolate(
            output["cam_score"].unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        scores = apply_score_calibration(scores, output, output_size, score_calibration)
        predictions.append(scores)
    scores = torch.stack(predictions).mean(dim=0)
    return _apply_candidate_calibration_if_enabled(
        scores,
        candidate_calibration,
        pixel_threshold,
    )


def _apply_candidate_calibration_if_enabled(
    scores: torch.Tensor,
    candidate_calibration: dict | None,
    pixel_threshold: float | None,
) -> torch.Tensor:
    if candidate_calibration is None or pixel_threshold is None:
        return scores
    return apply_candidate_calibration(
        scores,
        pixel_threshold,
        candidate_calibration,
    )


@torch.no_grad()
def calibrate_negative_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    background_quantile: float,
    histogram_bins: int,
    prediction_scales: list[int] | tuple[int, ...] | None,
    score_calibration: dict | None,
    max_batches: int | None,
) -> dict[str, float | int]:
    if not 0 < background_quantile < 1:
        raise ValueError("background_quantile must be in (0, 1)")
    model.eval()
    histogram = torch.zeros(histogram_bins, dtype=torch.float64)
    negative_images = 0
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc="calibrate negative CAM",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        negative = batch["label"] < 0.5
        if not negative.any():
            continue
        t1 = batch["t1"][negative].to(device, non_blocking=True)
        t2 = batch["t2"][negative].to(device, non_blocking=True)
        scores = predict_cam_scores(
            model,
            t1,
            t2,
            tuple(batch["mask"].shape[-2:]),
            mixed_precision,
            prediction_scales,
            score_calibration,
        )
        histogram += torch.histc(
            scores.detach().float().cpu(),
            bins=histogram_bins,
            min=0.0,
            max=1.0,
        ).double()
        negative_images += int(negative.sum())
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("Validation subset contains no image-level negative samples")
    index = int(
        torch.searchsorted(histogram.cumsum(0), background_quantile * total).clamp(
            max=histogram_bins - 1
        )
    )
    threshold = min((index + 1) / histogram_bins, 1.0)
    return {
        "pixel_threshold": threshold,
        "background_quantile": background_quantile,
        "estimated_background_fpr": float(histogram[index + 1 :].sum() / total),
        "negative_images": negative_images,
        "negative_pixels": total,
    }


@torch.no_grad()
def calibrate_negative_candidates(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    pixel_threshold: float,
    prediction_scales: list[int] | tuple[int, ...] | None,
    score_calibration: dict | None,
    candidate_config: dict,
    max_batches: int | None,
) -> dict:
    model.eval()
    statistics: list[dict[str, float | int]] = []
    negative_images = 0
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc="calibrate negative candidates",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        negative = batch["label"] < 0.5
        if not negative.any():
            continue
        t1 = batch["t1"][negative].to(device, non_blocking=True)
        t2 = batch["t2"][negative].to(device, non_blocking=True)
        scores = predict_cam_scores(
            model,
            t1,
            t2,
            tuple(batch["mask"].shape[-2:]),
            mixed_precision,
            prediction_scales,
            score_calibration,
        )
        statistics.extend(
            extract_candidate_statistics(
                scores.detach().cpu(),
                pixel_threshold,
                candidate_config,
            )
        )
        negative_images += int(negative.sum())

    return build_negative_component_filter_config(
        statistics,
        candidate_config,
        pixel_threshold,
        negative_images,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    mixed_precision: bool,
    max_batches: int | None,
    phase: str,
    pixel_threshold: float | None = None,
    prediction_scales: list[int] | tuple[int, ...] | None = None,
    score_calibration: dict | None = None,
    candidate_calibration: dict | None = None,
    teacher_model: nn.Module | None = None,
    teacher_weight: float = 0.0,
    teacher_positive_only: bool = True,
    teacher_ema_momentum: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if teacher_model is not None:
        teacher_model.eval()
    image_metrics = BinaryMetricAccumulator(threshold)
    pixel_metrics = (
        BinaryMetricAccumulator(pixel_threshold)
        if pixel_threshold is not None
        else None
    )
    loss_totals: dict[str, float] = {}
    total_samples = 0
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=phase,
        dynamic_ncols=True,
        leave=False,
    )

    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision,
        ):
            teacher_output = None
            if training and teacher_model is not None and teacher_weight > 0:
                with torch.no_grad():
                    teacher_output = teacher_model(t1, t2)
            output = model(t1, t2)
            losses = compute_losses(
                criterion,
                output,
                labels,
                teacher_output=teacher_output,
                teacher_weight=teacher_weight,
                teacher_positive_only=teacher_positive_only,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    losses["total"].backward()
                    optimizer.step()
                else:
                    scaler.scale(losses["total"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                if teacher_model is not None and teacher_ema_momentum is not None:
                    update_ema_teacher(teacher_model, model, teacher_ema_momentum)

        count = labels.numel()
        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach()) * count
        total_samples += count
        image_metrics.update(output["logits"].sigmoid(), labels)
        image_result = image_metrics.compute()
        running_loss = loss_totals.get("total", 0.0) / max(total_samples, 1)

        if pixel_metrics is not None:
            masks = batch["mask"]
            cam_scores = predict_cam_scores(
                model,
                t1,
                t2,
                tuple(masks.shape[-2:]),
                mixed_precision,
                prediction_scales,
                score_calibration,
                candidate_calibration,
                pixel_threshold,
            ).cpu()
            pixel_metrics.update(cam_scores, masks)
            pixel_result = pixel_metrics.compute()
            progress.set_postfix(
                loss=f"{running_loss:.4f}",
                pixel_P=f"{pixel_result['precision']:.3f}",
                pixel_R=f"{pixel_result['recall']:.3f}",
                pixel_F1=f"{pixel_result['f1']:.3f}",
                pixel_IoU=f"{pixel_result['iou']:.3f}",
            )
        else:
            progress.set_postfix(
                loss=f"{running_loss:.4f}",
                image_P=f"{image_result['precision']:.3f}",
                image_R=f"{image_result['recall']:.3f}",
                image_F1=f"{image_result['f1']:.3f}",
            )

    result = {
        f"image_{name}": value
        for name, value in image_metrics.compute().items()
    }
    for name, value in loss_totals.items():
        result[f"loss_{name}"] = value / max(total_samples, 1)
    result["loss"] = result.get("loss_total", 0.0)
    if pixel_metrics is not None:
        result.update(
            {
                f"pixel_{name}": value
                for name, value in pixel_metrics.compute().items()
            }
        )
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    if args.output_dir:
        config["output_dir"] = args.output_dir
    set_seed(int(config["seed"]))
    device = choose_device()

    data_config = load_yaml(resolve_from(ROOT, config["data"]["config"]))
    data_root = resolve_from(ROOT, data_config["root"])
    image_size = int(data_config["image_size"])
    train_dataset = ChangePairDataset(
        data_root,
        data_config["manifests"]["train"],
        image_size=image_size,
        augment=True,
        return_mask=False,
    )
    val_dataset = ChangePairDataset(
        data_root,
        data_config["manifests"]["val"],
        image_size=image_size,
        augment=False,
        return_mask=True,
    )
    batch_size = args.batch_size or int(config["data"]["batch_size"])
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config["data"]["num_workers"])
    )
    train_loader = make_loader(train_dataset, batch_size, workers, True)
    val_loader = make_loader(val_dataset, batch_size, workers, False)

    model = build_cam_model(config["model"]).to(device)
    positive_weight = torch.tensor(train_dataset.positive_class_weight(), device=device)
    criterion = build_training_criterion(config["train"], positive_weight)
    base_low_prior_weight = getattr(criterion, "low_prior_suppression_weight", None)
    self_teacher_config = dict(config["train"].get("self_teacher", {}))
    self_teacher_enabled = bool(self_teacher_config.get("enabled", False))
    teacher_warmup_epochs = int(self_teacher_config.get("warmup_epochs", 1))
    if teacher_warmup_epochs < 0:
        raise ValueError("self_teacher.warmup_epochs must be non-negative")
    teacher_weight = float(self_teacher_config.get("consistency_weight", 0.0))
    teacher_ramp_epochs = int(self_teacher_config.get("ramp_epochs", 0))
    teacher_ema_momentum = float(self_teacher_config.get("ema_momentum", 0.99))
    teacher_positive_only = bool(self_teacher_config.get("positive_only", True))
    prior_weight_after_warmup = self_teacher_config.get("prior_weight_after_warmup")
    teacher_model = (
        build_ema_teacher(model).to(device)
        if self_teacher_enabled and teacher_warmup_epochs == 0
        else None
    )
    learning_rate = float(config["train"]["learning_rate"])
    encoder_multiplier = float(
        config["train"].get("encoder_learning_rate_multiplier", 1.0)
    )
    encoder_parameters = list(model.encoder_parameters())
    head_parameters = list(model.head_parameters())
    parameter_groups = [{"params": head_parameters, "lr": learning_rate}]
    if encoder_parameters:
        parameter_groups.insert(
            0,
            {
                "params": encoder_parameters,
                "lr": learning_rate * encoder_multiplier,
            },
        )
    optimizer = AdamW(
        parameter_groups,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["train"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    threshold = float(config["train"]["image_threshold"])
    evaluation_config = config["evaluation"]
    pixel_threshold = float(evaluation_config["cam_threshold"])
    background_quantile = evaluation_config.get("background_quantile")
    histogram_bins = int(evaluation_config.get("histogram_bins", 10000))
    prediction_scales = evaluation_config.get("prediction_scales")
    score_calibration = evaluation_config.get("score_calibration")
    candidate_config = evaluation_config.get("candidate_calibration")
    checkpoint_metric = str(config["evaluation"].get("checkpoint_metric", "pixel_f1"))
    mixed_precision = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if mixed_precision else None
    output_dir = resolve_from(ROOT, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0
    print(
        f"Training P1 CAM on {device}: architecture={config['model']['architecture']}, "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"positive_weight={float(positive_weight):.4f}, "
        f"best_checkpoint=val_{checkpoint_metric}, "
        f"self_teacher={self_teacher_enabled}"
    )
    for epoch in range(1, epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        if teacher_model is None:
            set_optional_loss_weight(
                criterion,
                "low_prior_suppression_weight",
                base_low_prior_weight,
            )
            current_teacher_weight = 0.0
        else:
            set_optional_loss_weight(
                criterion,
                "low_prior_suppression_weight",
                prior_weight_after_warmup,
            )
            if teacher_ramp_epochs > 0:
                ramp_progress = min(
                    1.0,
                    max(0, epoch - teacher_warmup_epochs) / teacher_ramp_epochs,
                )
                current_teacher_weight = teacher_weight * ramp_progress
            else:
                current_teacher_weight = teacher_weight
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            threshold,
            optimizer,
            scaler,
            mixed_precision,
            args.max_train_batches,
            phase="train image-level",
            teacher_model=teacher_model,
            teacher_weight=current_teacher_weight,
            teacher_positive_only=teacher_positive_only,
            teacher_ema_momentum=teacher_ema_momentum if teacher_model is not None else None,
        )
        calibration = None
        candidate_calibration = None
        if background_quantile is not None:
            calibration = calibrate_negative_threshold(
                model,
                val_loader,
                device,
                mixed_precision,
                float(background_quantile),
                histogram_bins,
                prediction_scales,
                score_calibration,
                args.max_val_batches,
            )
            pixel_threshold = float(calibration["pixel_threshold"])
            print(
                f"negative-label pixel calibration: threshold={pixel_threshold:.4f} "
                f"estimated_FPR={float(calibration['estimated_background_fpr']):.4f}"
            )
        if candidate_config:
            candidate_name = str(candidate_config.get("name", "none"))
            if candidate_name == "fixed_component_filter":
                candidate_calibration = dict(candidate_config)
                print(
                    "fixed candidate filter: "
                    f"min_pixels={candidate_calibration.get('min_component_pixels', 1)} "
                    f"max_area_ratio={candidate_calibration.get('max_area_ratio', 'none')} "
                    f"min_mean={candidate_calibration.get('min_mean_score', 'none')} "
                    f"min_peak={candidate_calibration.get('min_peak_score', 'none')}"
                )
            elif candidate_name == "none" or not bool(candidate_config.get("enabled", True)):
                candidate_calibration = None
            else:
                candidate_calibration = calibrate_negative_candidates(
                    model,
                    val_loader,
                    device,
                    mixed_precision,
                    pixel_threshold,
                    prediction_scales,
                    score_calibration,
                    dict(candidate_config),
                    args.max_val_batches,
                )
                if candidate_calibration.get("enabled", True):
                    print(
                        "negative candidate calibration: "
                        f"components={candidate_calibration['negative_components']} "
                        f"area<={candidate_calibration['reject_area_ratio']:.5f} "
                        f"mean<={candidate_calibration['reject_mean_score']:.4f} "
                        f"peak<={candidate_calibration['reject_peak_score']:.4f}"
                    )
                else:
                    print("negative candidate calibration: no components to filter")
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold,
            None,
            None,
            mixed_precision,
            args.max_val_batches,
            phase="val pixel-level",
            pixel_threshold=pixel_threshold,
            prediction_scales=prediction_scales,
            score_calibration=score_calibration,
            candidate_calibration=candidate_calibration,
        )
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "teacher_model": teacher_model.state_dict()
            if teacher_model is not None
            else None,
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
            "pixel_threshold": pixel_threshold,
            "pixel_calibration": calibration,
            "candidate_calibration": candidate_calibration,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if checkpoint_metric not in val_metrics:
            raise KeyError(
                f"Checkpoint metric '{checkpoint_metric}' is not available in "
                f"validation metrics: {sorted(val_metrics)}"
            )
        if val_metrics[checkpoint_metric] > best_score:
            best_score = val_metrics[checkpoint_metric]
            torch.save(checkpoint, output_dir / "best.pt")
        write_json(output_dir / "history.json", history)
        if (
            self_teacher_enabled
            and teacher_model is None
            and epoch >= teacher_warmup_epochs
        ):
            teacher_model = build_ema_teacher(model).to(device)
            print(
                "initialized EMA teacher from current model: "
                f"warmup_epoch={epoch}, momentum={teacher_ema_momentum:.4f}"
            )
        auxiliary = " ".join(
            f"{name[5:]}={train_metrics[name]:.4f}"
            for name in sorted(train_metrics)
            if name.startswith("loss_") and name not in {"loss_total", "loss_bag"}
        )
        auxiliary = f" {auxiliary}" if auxiliary else ""
        print(
            f"train image-level: loss={train_metrics['loss']:.4f} "
            f"P={train_metrics['image_precision']:.4f} "
            f"R={train_metrics['image_recall']:.4f} "
            f"F1={train_metrics['image_f1']:.4f}{auxiliary}\n"
            f"val pixel-level:   P={val_metrics['pixel_precision']:.4f} "
            f"R={val_metrics['pixel_recall']:.4f} "
            f"F1={val_metrics['pixel_f1']:.4f} "
            f"IoU={val_metrics['pixel_iou']:.4f} "
            f"best_{checkpoint_metric}={best_score:.4f} "
            f"time={time.time() - start:.1f}s"
        )


if __name__ == "__main__":
    main()
