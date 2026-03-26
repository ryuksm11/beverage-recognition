"""
Download training images from Bing for each beverage class.
Uses icrawler's BingImageCrawler under the hood.

Usage:
    python training/downloader.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icrawler.builtin import BingImageCrawler

from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)


def download_images_for_class(
    class_name: str,
    output_dir: Path,
    num_images: int = 200,
    delay: float = 2.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {num_images} images for '{class_name}' → {output_dir}")

    crawler = BingImageCrawler(
        storage={"root_dir": str(output_dir)},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
    )
    crawler.crawl(
        keyword=f"{class_name} beverage drink packaged",
        max_num=num_images,
        filters={"type": "photo"},
    )

    downloaded = len(list(output_dir.iterdir()))
    logger.info(f"  Saved {downloaded} files for '{class_name}'")

    time.sleep(delay)


def download_all_classes(
    classes: list[str],
    raw_dir: Path,
    num_per_class: int = 200,
    delay: float = 2.0,
) -> None:
    logger.info(f"Starting download for {len(classes)} classes")
    for cls in classes:
        # Sanitize class name for use as a directory name
        safe_name = cls.replace(" ", "_").replace("/", "-")
        output_dir = raw_dir / safe_name
        if output_dir.exists() and len(list(output_dir.iterdir())) >= num_per_class * 0.5:
            logger.info(f"Skipping '{cls}' — directory already populated")
            continue
        download_images_for_class(cls, output_dir, num_per_class, delay)
    logger.info("All downloads complete.")


if __name__ == "__main__":
    cfg = load_config()
    download_all_classes(
        classes=cfg["classes"],
        raw_dir=resolve_path(cfg["paths"]["data_raw"]),
        num_per_class=cfg["download"]["images_per_class"],
        delay=cfg["download"]["request_delay_sec"],
    )
