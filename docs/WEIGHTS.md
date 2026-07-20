# Model Weights

## DINOv2-Base

Both the weak teacher and dense detector use the official `facebook/dinov2-base` model through Transformers. With `pretrained: true` and `local_files_only: false`, the files are downloaded into the normal Hugging Face cache on first use.

Official model page: <https://huggingface.co/facebook/dinov2-base>

For an offline machine, pre-download the model into the Hugging Face cache and then set `local_files_only: true` in both Stage 1 and Stage 2 configurations.

## SAM ViT-H

Download `sam_vit_h_4b8939.pth` from the checkpoint links in Meta's official [Segment Anything repository](https://github.com/facebookresearch/segment-anything), then place it at:

```text
models/sam/sam_vit_h_4b8939.pth
```

The file is approximately 2.4 GB and is ignored by Git. Do not commit or redistribute it from this repository.

## Trained VALOR-CD Checkpoints

Stage 2 writes:

```text
outputs/<dataset>/stage2_detector/final_detector/best.pt
outputs/<dataset>/stage2_detector/final_detector/last.pt
```

The public Stage 3 configurations use `best.pt`. If pretrained detector checkpoints are released, host them as GitHub Release assets or in an external model repository and publish SHA-256 checksums; do not add them to Git history.
