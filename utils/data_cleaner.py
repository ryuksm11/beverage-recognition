"""
Clean raw downloaded images before splitting.
Removes corrupt files and exact duplicates (by MD5 hash).

Usage:
    python utils/data_cleaner.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _md5(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_corrupt_images(directory: Path) -> int:
    removed = 0
    for path in directory.rglob("*"):
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, Exception):
            path.unlink()
            removed += 1
    return removed


def deduplicate_images(directory: Path) -> int:
    seen: set[str] = set()
    removed = 0
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        digest = _md5(path)
        if digest in seen:
            path.unlink()
            removed += 1
        else:
            seen.add(digest)
    return removed


def validate_dataset_balance(raw_dir: Path, min_per_class: int = 50) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(
            1 for p in class_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        )
        counts[class_dir.name] = n
        if n < min_per_class:
            logger.warning(f"  '{class_dir.name}' has only {n} images (min={min_per_class})")
    return counts


if __name__ == "__main__":
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["data_raw"])

    logger.info("Removing corrupt images...")
    total_corrupt = 0
    total_dupes = 0

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        corrupt = remove_corrupt_images(class_dir)
        dupes = deduplicate_images(class_dir)
        total_corrupt += corrupt
        total_dupes += dupes
        logger.info(f"  {class_dir.name}: removed {corrupt} corrupt, {dupes} duplicates")

    logger.info(f"Total removed — corrupt: {total_corrupt}, duplicates: {total_dupes}")

    logger.info("\nClass balance after cleaning:")
    counts = validate_dataset_balance(raw_dir)
    for cls, n in counts.items():
        logger.info(f"  {cls}: {n} images")
