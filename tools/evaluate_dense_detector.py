from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.data import ChangePairDataset  # noqa: E402
from valor_cd.evaluation import BinaryMetricAccumulator  # noqa: E402
from valor_cd.experiment import choose_device, load_yaml, resolve_from, write_json  # noqa: E402
from valor_cd.models import build_dense_change_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained dense change detector.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pixel-threshold", type=float)
    parser.add_argument("--image-threshold", type=float)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--export-masks", action="store_true")
    return parser.parse_args()


def make_loader(dataset: ChangePairDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def save_probability(path: Path, values: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (values.detach().cpu().float().clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def save_binary(path: Path, values: torch.Tensor, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (values.detach().cpu().float() >= threshold).byte().mul(255).numpy()
    Image.fromarray(array, mode="L").save(path)


def relative_to_output(path: Path, output_dir: Path) -> str:
    return str(path.relative_to(output_dir)).replace("\\", "/")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool,
    pixel_threshold: float,
    image_threshold: float,
    output_dir: Path,
    export_masks: bool,
    max_batches: int | None,
    stage_name: str,
) -> dict[str, Any]:
    model.eval()
    pixel_metrics = BinaryMetricAccumulator(pixel_threshold)
    image_metrics = BinaryMetricAccumulator(image_threshold)
    rows: list[dict[str, Any]] = []
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"evaluate {stage_name}",
        dynamic_ncols=True,
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
        image_probability = output["image_probability"].detach().cpu()
        pixel_metrics.update(probabilities, batch["mask"])
        image_metrics.update(image_probability, labels.detach().cpu())
        batch_ids = list(batch["id"])
        for index, sample_id in enumerate(batch_ids):
            row = {
                "id": sample_id,
                "label": int(labels[index].item()),
                "image_probability": float(image_probability[index].item()),
                "change_ratio": float(batch["change_ratio"][index]),
                "foreground_ratio": float((probabilities[index] >= pixel_threshold).float().mean()),
            }
            if export_masks:
                probability_path = output_dir / "probabilities" / f"{sample_id}.png"
                mask_path = output_dir / "masks" / f"{sample_id}.png"
                save_probability(probability_path, probabilities[index])
                save_binary(mask_path, probabilities[index], pixel_threshold)
                row["probability"] = relative_to_output(probability_path, output_dir)
                row["mask"] = relative_to_output(mask_path, output_dir)
            rows.append(row)
        progress.set_postfix(pixel_f1=f"{pixel_metrics.compute()['f1']:.4f}")

    if rows:
        with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "pixel": {**pixel_metrics.counts(), **pixel_metrics.compute()},
        "image": {**image_metrics.counts(), **image_metrics.compute()},
        "num_predictions": len(rows),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_from(ROOT, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Dense detector checkpoint does not exist: {checkpoint_path}\n"
            "Run Stage 2 training first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_config = dict(config["data"])
    supervised_config = load_yaml(resolve_from(ROOT, data_config["supervised_config"]))
    image_size = int(data_config.get("image_size", supervised_config.get("image_size", 224)))
    dataset = ChangePairDataset(
        resolve_from(ROOT, supervised_config["root"]),
        supervised_config["manifests"][args.split],
        image_size=image_size,
        augment=False,
        return_mask=True,
    )
    batch_size = int(args.batch_size or data_config.get("val_batch_size", data_config["batch_size"]))
    num_workers = int(args.num_workers if args.num_workers is not None else data_config.get("num_workers", 0))
    loader = make_loader(dataset, batch_size, num_workers)
    device = choose_device()
    model = build_dense_change_model(config["model"], pretrained_override=False).to(device)
    use_ema_model = bool(config.get("evaluation", {}).get("use_ema_model", False))
    state_key = "ema_model" if use_ema_model and "ema_model" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key])

    evaluation_config = dict(config.get("evaluation", {}))
    pixel_threshold = float(
        args.pixel_threshold
        if args.pixel_threshold is not None
        else evaluation_config.get("pixel_threshold", 0.5)
    )
    image_threshold = float(
        args.image_threshold
        if args.image_threshold is not None
        else evaluation_config.get("image_threshold", 0.5)
    )
    mixed_precision = bool(config["train"].get("mixed_precision", True))
    output_dir = resolve_from(
        ROOT,
        args.output_dir or Path(config["output_dir"]) / f"eval_{args.split}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_name = str(config.get("stage_name", "soft-hard-distillation"))
    results = evaluate(
        model,
        loader,
        device,
        mixed_precision=mixed_precision,
        pixel_threshold=pixel_threshold,
        image_threshold=image_threshold,
        output_dir=output_dir,
        export_masks=args.export_masks,
        max_batches=args.max_batches,
        stage_name=stage_name,
    )
    results.update(
        {
            "checkpoint": str(checkpoint_path),
            "state_key": state_key,
            "split": args.split,
            "pixel_threshold": pixel_threshold,
            "image_threshold": image_threshold,
            "device": str(device),
        }
    )
    write_json(output_dir / "metrics.json", results)
    pixel = results["pixel"]
    image = results["image"]
    print(
        f"{stage_name} Pixel P/R/F1/IoU: "
        f"{pixel['precision']:.4f}/"
        f"{pixel['recall']:.4f}/"
        f"{pixel['f1']:.4f}/"
        f"{pixel['iou']:.4f}"
    )
    print(
        f"{stage_name} Image P/R/F1: "
        f"{image['precision']:.4f}/"
        f"{image['recall']:.4f}/"
        f"{image['f1']:.4f}"
    )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
