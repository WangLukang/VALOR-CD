from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cff"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
FORBIDDEN_TOP_LEVEL = {"data", "outputs", "comparison", "Fig", "runs", "checkpoints"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def repository_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(ROOT).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        if parts and parts[0] in FORBIDDEN_TOP_LEVEL:
            continue
        files.append(path)
    return files


def check_public_configuration(errors: list[str]) -> None:
    for dataset in ("whu", "levir"):
        stage1 = yaml.safe_load((ROOT / "configs" / dataset / "stage1.yaml").read_text(encoding="utf-8"))
        stage2 = yaml.safe_load((ROOT / "configs" / dataset / "stage2.yaml").read_text(encoding="utf-8"))
        stage3 = yaml.safe_load((ROOT / "configs" / dataset / "stage3.yaml").read_text(encoding="utf-8"))

        teacher = stage1["weak_teacher"]["model"]
        sam = stage1["sam_pseudo"]
        detector = stage2["detector"]["model"]
        expected = {
            "teacher model": (teacher["model_id"], "facebook/dinov2-base"),
            "teacher Top-k": (float(teacher["topk_ratio"]), 0.05),
            "SAM model": (sam["model_type"], "vit_h"),
            "SAM image": (sam["image_mode"], "t2"),
            "detector model": (detector["model_id"], "facebook/dinov2-base"),
            "inference threshold": (float(stage3["pixel_threshold"]), 0.7),
        }
        for label, (actual, wanted) in expected.items():
            if actual != wanted:
                errors.append(f"{dataset}: {label} is {actual!r}, expected {wanted!r}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    required = [
        "README.md",
        "pyproject.toml",
        "environment.yml",
        ".gitignore",
        "scripts/01_generate_pseudo_labels.py",
        "scripts/02_train_detector.py",
        "scripts/03_test.py",
        "src/valor_cd/data/__init__.py",
        "src/valor_cd/data/change_pair.py",
        "src/valor_cd/data/pseudo_change.py",
        "src/valor_cd/data/spatial_split.py",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    if (ROOT / ".git").is_dir():
        tracked_files = set(
            subprocess.run(
                ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.splitlines()
        )
        for relative in required:
            if relative not in tracked_files:
                errors.append(f"required public file is not tracked: {relative}")
        for name in FORBIDDEN_TOP_LEVEL:
            tracked = subprocess.run(
                ["git", "ls-files", "--", name],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if tracked:
                errors.append(f"forbidden public content is tracked under {name}/")
    else:
        for name in FORBIDDEN_TOP_LEVEL:
            if (ROOT / name).exists():
                warnings.append(f"local ignored directory exists: {name}/")

    local_user_path = re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE)
    conda_path = re.compile(r"[A-Za-z]:\\[^\n\r]*conda", re.IGNORECASE)
    hf_token = re.compile("hf_" + r"[A-Za-z0-9]{20,}")
    for path in repository_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"large file ({path.stat().st_size / 1024**2:.1f} MB): {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if local_user_path.search(text):
            errors.append(f"machine-specific user path: {relative}")
        if conda_path.search(text):
            errors.append(f"machine-specific Conda path: {relative}")
        if hf_token.search(text):
            errors.append(f"possible Hugging Face token: {relative}")

    try:
        check_public_configuration(errors)
    except Exception as error:
        errors.append(f"could not validate public configurations: {error}")

    if not (ROOT / "LICENSE").is_file():
        warnings.append("LICENSE has not been selected yet; add one before the public push")
    if not (ROOT / "CITATION.cff").is_file():
        warnings.append("CITATION.cff is pending final paper metadata")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Release check failed with {len(errors)} error(s).")
        return 1
    print(f"Release check passed: {len(repository_files())} files scanned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
