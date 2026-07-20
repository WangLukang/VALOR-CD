from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from valor_cd.data import ChangePairDataset  # noqa: E402
from valor_cd.evaluation import (  # noqa: E402
    apply_candidate_calibration,
    apply_score_calibration,
    verify_counterfactual_candidates,
)
from valor_cd.experiment import choose_device, load_yaml, resolve_from, write_json  # noqa: E402
from valor_cd.models import build_cam_model  # noqa: E402
from valor_cd.pseudo_label import (  # noqa: E402
    apply_pseudo_label_refinement,
    build_soft_pseudo_labels,
    pseudo_label_quality,
    save_grayscale_float_png,
)




def save_score(path: Path, score: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = score.clamp(0, 1).mul(255).byte().cpu().numpy()
    Image.fromarray(array, mode="L").save(path)


@torch.no_grad()
def predict_scores(
    model: torch.nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    output_size: tuple[int, int],
    scales: list[int] | tuple[int, ...] | None,
    score_calibration: dict | None = None,
    candidate_calibration: dict | None = None,
    pixel_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_output = model(t1, t2)
    if not scales:
        score = F.interpolate(
            base_output["cam_score"].unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        score = apply_score_calibration(score, base_output, output_size, score_calibration)
        score = _apply_candidate_calibration_if_enabled(score, candidate_calibration, pixel_threshold)
        return base_output["logits"].sigmoid(), score

    predictions = []
    for scale in scales:
        size = (int(scale), int(scale))
        output = model(
            F.interpolate(t1, size=size, mode="bilinear", align_corners=False),
            F.interpolate(t2, size=size, mode="bilinear", align_corners=False),
        )
        score = F.interpolate(
            output["cam_score"].unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        score = apply_score_calibration(score, output, output_size, score_calibration)
        predictions.append(score)
    score = torch.stack(predictions).mean(dim=0)
    score = _apply_candidate_calibration_if_enabled(score, candidate_calibration, pixel_threshold)
    return base_output["logits"].sigmoid(), score


def _apply_candidate_calibration_if_enabled(
    score: torch.Tensor,
    candidate_calibration: dict | None,
    pixel_threshold: float | None,
) -> torch.Tensor:
    if candidate_calibration is None or pixel_threshold is None:
        return score
    return apply_candidate_calibration(score, pixel_threshold, candidate_calibration)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P5 soft pseudo labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--export-cams", action="store_true")
    return parser.parse_args()


def load_p5_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(resolve_from(ROOT, args.config))
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.split:
        config["split"] = args.split
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.export_cams:
        config["export_cams"] = True
    return config


def relative_to_output(path: Path, output_dir: Path) -> str:
    return str(path.relative_to(output_dir)).replace("\\", "/")


def main() -> None:
    args = parse_args()
    p5_config = load_p5_config(args)
    checkpoint_path = resolve_from(ROOT, p5_config["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}\n"
            "Run P4/P3 first, or pass --checkpoint."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint["config"]
    data_config = load_yaml(resolve_from(ROOT, checkpoint_config["data"]["config"]))
    data_root = resolve_from(ROOT, data_config["root"])
    split = str(p5_config.get("split", "train"))
    dataset = ChangePairDataset(
        data_root,
        data_config["manifests"][split],
        image_size=int(data_config["image_size"]),
        augment=False,
        return_mask=False,
    )
    row_by_id = {row["id"]: row for row in dataset.rows}
    loader = DataLoader(
        dataset,
        batch_size=int(checkpoint_config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(
            p5_config.get(
                "num_workers",
                checkpoint_config["data"]["num_workers"],
            )
        ),
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device()
    model = build_cam_model(checkpoint_config["model"], pretrained_override=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cam_threshold = float(
        checkpoint.get(
            "pixel_threshold",
            checkpoint_config["evaluation"]["cam_threshold"],
        )
    )
    prediction_scales = checkpoint_config["evaluation"].get("prediction_scales")
    score_calibration = checkpoint_config["evaluation"].get("score_calibration")
    candidate_calibration = checkpoint.get("candidate_calibration")
    counterfactual_config = dict(p5_config.get("counterfactual", {}))
    pseudo_config = dict(p5_config.get("pseudo_label", {}))
    refinement_config = dict(pseudo_config.get("refinement") or {})
    if (
        refinement_config
        and str(refinement_config.get("name", "none")) != "none"
        and bool(refinement_config.get("preserve_high_scores", False))
    ):
        refinement_config.setdefault("score_threshold", cam_threshold)
        pseudo_config["refinement"] = refinement_config

    output_dir = resolve_from(ROOT, p5_config["output_dir"])
    pseudo_dir = output_dir / "pseudo_labels"
    confidence_dir = output_dir / "pseudo_confidence"
    cam_dir = output_dir / "p4_cam_scores"
    refined_cam_dir = output_dir / "p5_refined_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    candidate_total = 0
    candidate_verified = 0
    positive_images = 0
    negative_images = 0
    total_batches = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"generate P5 {split}",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            t1 = batch["t1"].to(device, non_blocking=True)
            t2 = batch["t2"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            probabilities, p3_scores = predict_scores(
                model,
                t1,
                t2,
                tuple(t1.shape[-2:]),
                prediction_scales,
                score_calibration,
                candidate_calibration,
                cam_threshold,
            )
            p4_scores = p3_scores.clone()
            positive = labels > 0.5
            positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
            positive_images += int(positive.sum())
            negative_images += int((~positive).sum())
            candidate_counts = {int(index): {"total": 0, "verified": 0} for index in range(labels.numel())}
            if positive_indices.numel() > 0:
                verification = verify_counterfactual_candidates(
                    model,
                    t1[positive],
                    t2[positive],
                    p3_scores[positive],
                    probabilities[positive],
                    cam_threshold,
                    counterfactual_config,
                )
                p4_scores[positive] = verification["scores"]
                for record in verification["candidates"]:
                    local_index = int(record["sample_index"])
                    batch_sample_index = int(positive_indices[local_index])
                    candidate_counts[batch_sample_index]["total"] += 1
                    candidate_total += 1
                    if bool(record["verified"]):
                        candidate_counts[batch_sample_index]["verified"] += 1
                        candidate_verified += 1

            refined_scores = apply_pseudo_label_refinement(
                p4_scores,
                t1,
                t2,
                refinement_config,
            )
            pseudo, confidence = build_soft_pseudo_labels(
                refined_scores.detach().cpu(),
                labels.detach().cpu(),
                probabilities.detach().cpu(),
                cam_threshold,
                pseudo_config,
            )
            p4_scores_cpu = p4_scores.detach().cpu()
            refined_scores_cpu = refined_scores.detach().cpu()
            for index, sample_id in enumerate(batch["id"]):
                pseudo_path = pseudo_dir / f"{sample_id}.png"
                confidence_path = confidence_dir / f"{sample_id}.png"
                save_grayscale_float_png(pseudo_path, pseudo[index])
                save_grayscale_float_png(confidence_path, confidence[index])
                if bool(p5_config.get("export_cams", False)):
                    save_score(cam_dir / f"{sample_id}.png", p4_scores_cpu[index])
                    save_score(refined_cam_dir / f"{sample_id}.png", refined_scores_cpu[index])

                quality = pseudo_label_quality(
                    pseudo[index],
                    confidence[index],
                    float(pseudo_config.get("foreground_threshold", 0.5)),
                )
                source_row = row_by_id[sample_id]
                counts = candidate_counts[index]
                manifest_rows.append(
                    {
                        "id": sample_id,
                        "t1": source_row["t1"],
                        "t2": source_row["t2"],
                        "label": int(labels[index].item()),
                        "pseudo_label": relative_to_output(pseudo_path, output_dir),
                        "pseudo_confidence": relative_to_output(confidence_path, output_dir),
                        "image_probability": f"{float(probabilities[index]):.8f}",
                        "p4_candidate_total": counts["total"],
                        "p4_candidate_verified": counts["verified"],
                        "pseudo_mean": f"{quality['pseudo_mean']:.8f}",
                        "pseudo_max": f"{quality['pseudo_max']:.8f}",
                        "confidence_mean": f"{quality['confidence_mean']:.8f}",
                        "foreground_ratio": f"{quality['foreground_ratio']:.8f}",
                    }
                )

    manifest_path = output_dir / f"{split}_pseudo_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "pixel_threshold": cam_threshold,
        "candidate_calibration": candidate_calibration,
        "counterfactual": counterfactual_config,
        "pseudo_label": pseudo_config,
        "samples": len(manifest_rows),
        "positive_images": positive_images,
        "negative_images": negative_images,
        "candidate_total": candidate_total,
        "candidate_verified": candidate_verified,
        "candidate_verification_rate": candidate_verified / max(candidate_total, 1),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Generated P5 pseudo labels: {len(manifest_rows)} samples")
    print(f"Positive/negative: {positive_images}/{negative_images}")
    print(f"Candidates: {candidate_verified}/{candidate_total} verified")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
