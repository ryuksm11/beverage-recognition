"""
Download indoor background images for background-paste augmentation.

Uses icrawler (already a project dependency) with varied indoor scene queries
so the model learns to separate the bottle from any background.

Run once before segmenting + retraining:
    conda activate beverage-cnn
    python scripts/download_backgrounds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icrawler.builtin import BingImageCrawler

from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

# Varied queries — want diverse backgrounds, not product shots
_QUERIES = [
    "indoor room background plain",
    "kitchen counter surface",
    "wooden table top",
    "office desk clean",
    "concrete wall background",
    "supermarket shelf background",
    "cafe table background",
    "home interior background",
]

_IMAGES_PER_QUERY = 30   # 8 queries × 30 = ~240 backgrounds total


def main() -> None:
    cfg = load_config()
    out_dir = resolve_path(cfg["paths"]["data_backgrounds"])
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    if existing >= len(_QUERIES) * _IMAGES_PER_QUERY * 0.8:
        logger.info(f"Backgrounds already downloaded ({existing} files in {out_dir}). Skipping.")
        return

    logger.info(f"Downloading backgrounds to {out_dir}")

    for query in _QUERIES:
        logger.info(f"  Query: '{query}'")
        crawler = BingImageCrawler(storage={"root_dir": str(out_dir)})
        crawler.crawl(keyword=query, max_num=_IMAGES_PER_QUERY, file_idx_offset="auto")

    total = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    logger.info(f"Done. {total} background images in {out_dir}")


if __name__ == "__main__":
    main()
