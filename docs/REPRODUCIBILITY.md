# Reproducibility Protocol

## Frozen Main Configuration

The same principal setup is used for WHU-CD and LEVIR-CD:

- DINOv2-Base weak teacher with a frozen backbone.
- Top-5% MIL pooling.
- 10 Stage 1 epochs, batch size 8, random seed 42.
- Mask-free negative-response calibration from image-level negative training samples.
- Counterfactual candidate attenuation and difference-supported soft pseudo labels.
- SAM ViT-H on the T2 RGB image, constrained by soft-label candidates.
- Edge-guided DINOv2-Base dense detector.
- Joint confidence-weighted soft SCE and hard GCE/Dice/boundary supervision.
- 50 Stage 2 epochs, batch size 8, random seed 42.
- Fixed pixel threshold 0.7 for final validation/test reporting.

## Supervision Boundary

Training masks are stored locally because they are part of the public datasets, but Stage 1 does not load them into the weak-supervision loss. It reads only image-level labels derived from whether a mask contains at least one changed pixel.

Negative-response and candidate calibration are estimated only from image-level negative training pairs, without loading pixel-level masks.

The Stage 1 weak-teacher checkpoint is selected by validation image-level F1. Within Stage 1, validation masks are used only for pixel-level metric monitoring and never for calibration.

In Stage 1, validation masks may be used to:

1. monitor pixel-level pseudo-label quality;
2. report validation pixel-level metrics.

Validation masks are not used to tune candidate-filter, counterfactual, difference-refinement, SAM, or final inference-threshold parameters. The released values are fixed across both datasets.

## Thresholds

Three thresholds have different roles:

- Stage 1 `evaluation.cam_threshold`: internal CAM monitoring and candidate extraction.
- Stage 2 `evaluation.pixel_threshold: 0.5`: training-time validation monitor used by the saved experiment.
- Stage 3 `pixel_threshold: 0.7`: frozen final binary prediction threshold used for the reported public results.

Keeping these roles explicit avoids silently sweeping test thresholds.

## Determinism

The code seeds Python and PyTorch. Exact bitwise reproducibility is not guaranteed across GPU models, CUDA/cuDNN versions, or nondeterministic kernels. Use `environment.yml`, retain the generated runtime YAML files, and record the selected checkpoint hash for archival experiments.
