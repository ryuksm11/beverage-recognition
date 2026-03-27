"""
Remove backgrounds from training and validation images using rembg.

Output mirrors the processed/ directory structure under data/segmented/:
    data/segmented/train/{class_name}/{image}.png   (RGBA — transparency = removed background)
    data/segmented/val/{class_name}/{image}.png

Only train and val are segmented — test set stays as-is for honest evaluation.
Already-processed files are skipped (safe to re-run).

Run once after splitting the dataset and before retraining:
    conda activate beverage-cnn
    python scripts/segment_training_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

_SPLITS = ("train", "val")


def _segment_split(
    processed_dir: Path,
    segmented_dir: Path,
    split: str,
    remove_bg,
) -> None:
    src_root = processed_dir / split
    dst_root = segmented_dir / split

    image_paths = list(src_root.glob("**/*.jpg")) + list(src_root.glob("**/*.jpeg")) + list(src_root.glob("**/*.png"))

    logger.info(f"  [{split}] {len(image_paths)} images to segment")

    for src_path in tqdm(image_paths, desc=f"Segmenting {split}", unit="img"):
        class_dir = src_path.parent.name
        dst_path = dst_root / class_dir / (src_path.stem + ".png")

        if dst_path.exists():
            continue  # already done — skip

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img_bytes = src_path.read_bytes()
            result_bytes = remove_bg(img_bytes)
            dst_path.write_bytes(result_bytes)
        except Exception as exc:
            logger.warning(f"    Skipping {src_path.name}: {exc}")


def main() -> None:
    try:
        from rembg import remove as remove_bg
    except ImportError:
        logger.error("rembg not installed. Run: pip install rembg")
        sys.exit(1)

    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["data_processed"])
    segmented_dir = resolve_path(cfg["paths"]["data_segmented"])

    if not processed_dir.exists():
        logger.error(f"Processed data not found at {processed_dir}. Run training/dataset.py first.")
        sys.exit(1)

    logger.info(f"Segmenting images: {processed_dir} → {segmented_dir}")

    for split in _SPLITS:
        _segment_split(processed_dir, segmented_dir, split, remove_bg)

    total = sum(1 for _ in segmented_dir.glob("**/*.png"))
    logger.info(f"Done. {total} segmented images saved to {segmented_dir}")


if __name__ == "__main__":
    main()
