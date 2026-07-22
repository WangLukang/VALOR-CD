# VALOR-CD

[![CI](https://github.com/WangLukang/VALOR-CD/actions/workflows/ci.yml/badge.svg)](https://github.com/WangLukang/VALOR-CD/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

VALOR-CD: Candidate Reliability Validation and Robust Soft–Hard Distillation for Image-Level Weakly Supervised Building Change Detection

> The public configuration is frozen to **DINOv2-Base + Top-5% MIL + SAM ViT-H + a fixed inference threshold of 0.7** for both WHU-CD and LEVIR-CD.

## Method

```text
Stage 1 (training only)
T1, T2, image label
  -> frozen DINOv2-Base StrongCAM + Top-5% MIL
  -> negative-response calibration + counterfactual candidate validation
  -> difference-refined soft pseudo label
  -> candidate-constrained SAM ViT-H boundary reconstruction

Stage 2 (training only)
T1, T2, soft target, SAM hard target
  -> edge-guided DINOv2-Base dense detector
  -> robust soft-hard offline distillation

Stage 3 (inference)
T1, T2 -> trained dense detector -> probability map -> threshold 0.7
```

The weak teacher, candidate validation, pseudo-label construction, and SAM are removed at inference. Only the trained dense detector is used.

## Repository Layout

```text
configs/                 Final WHU-CD and LEVIR-CD configurations
scripts/                 Three public entry points
src/valor_cd/             Models, losses, data loading, and evaluation
tools/                   Training, inference, data preparation, and release checks
models/sam/README.md      SAM ViT-H checkpoint instructions
docs/                    Dataset, weight, and reproducibility details
```

Datasets, pretrained weights, pseudo labels, checkpoints, and experiment outputs are intentionally excluded from Git.

## Installation

The reference environment is Python 3.10, PyTorch 2.5.1, CUDA 12.4, and Windows/Linux. Create it with:

```bash
conda env create -f environment.yml
conda activate wscd
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

Alternatively, install a PyTorch build matching your CUDA driver and then run:

```bash
python -m pip install -e .
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

DINOv2-Base is downloaded automatically from the official `facebook/dinov2-base` Hugging Face repository. See [docs/WEIGHTS.md](docs/WEIGHTS.md) for offline use and the SAM ViT-H checkpoint.

## Released Checkpoints

The released `best.pt` checkpoints for WHU-CD and LEVIR-CD are available from [Baidu Netdisk](https://pan.baidu.com/s/1XSdV0ZKYTZJe5r85iNg5Mw?pwd=93p4) (extraction code: `93p4`).

After downloading, place a dense-detector checkpoint at the path specified by the corresponding Stage 3 configuration, or provide its location with `--checkpoint`.

## Data Preparation

Download the datasets from their official pages:

- [WHU Building Change Detection Dataset](https://gpcv.whu.edu.cn/data/building_dataset.html)
- [LEVIR-CD](https://justchenhao.github.io/LEVIR/)

### WHU-CD

Arrange the original large TIFF files as:

```text
data/WHU/
  before/before.tif
  after/after.tif
  change label/change_label.tif
```

Create the non-overlapping 224x224 tiles and spatially isolated manifests:

```bash
python tools/prepare_whu_dataset.py
```

### LEVIR-CD

Arrange the official split as:

```text
data/LEVIR/
  train/A/  train/B/  train/label/
  val/A/    val/B/    val/label/
  test/A/   test/B/   test/label/
```

Create 224x224 covering tiles while preserving the official split:

```bash
python tools/prepare_levir_dataset.py
```

The complete preprocessing protocol and split statistics are documented in [docs/DATASETS.md](docs/DATASETS.md).

## SAM ViT-H

Download the official ViT-H checkpoint and place it at:

```text
models/sam/sam_vit_h_4b8939.pth
```

The checkpoint is not redistributed by this repository. Stage 1 checks the path before SAM processing.

## Training and Evaluation

### WHU-CD

```bash
python scripts/01_generate_pseudo_labels.py --config configs/whu/stage1.yaml
python scripts/02_train_detector.py --config configs/whu/stage2.yaml
python scripts/03_test.py --config configs/whu/stage3.yaml
```

### LEVIR-CD

```bash
python scripts/01_generate_pseudo_labels.py --config configs/levir/stage1.yaml
python scripts/02_train_detector.py --config configs/levir/stage2.yaml
python scripts/03_test.py --config configs/levir/stage3.yaml
```

## Reference Results

Pixel-level results from the frozen public configuration (`best.pt`, threshold 0.7):

| Dataset | Split | Precision | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|
| WHU-CD | validation | 87.41 | 84.98 | 86.17 | 75.71 |
| WHU-CD | test | 88.98 | 86.26 | 87.60 | 77.93 |
| LEVIR-CD | validation | 68.77 | 75.94 | 72.18 | 56.46 |
| LEVIR-CD | test | 69.97 | 72.89 | 71.40 | 55.52 |

Small numerical differences can arise from CUDA kernels, package versions, and checkpoint selection ties.

## Quick Checks

Print the complete command chain without training:

```bash
python scripts/01_generate_pseudo_labels.py --config configs/whu/stage1.yaml --dry-run
python scripts/02_train_detector.py --config configs/whu/stage2.yaml --dry-run
python scripts/03_test.py --config configs/whu/stage3.yaml --dry-run
```

Audit the repository before a public push:

```bash
python tools/check_release.py
```

## Citation

The repository includes a software citation in `CITATION.cff`. The associated paper title and BibTeX entry will be added when the manuscript metadata is finalized. Please also cite DINOv2, Segment Anything, WHU-CD, and LEVIR-CD when using their corresponding assets.

## License

The code is released under the [MIT License](LICENSE). Third-party models and datasets remain governed by their own licenses and terms of use.
