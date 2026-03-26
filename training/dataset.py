"""
Build train/val/test splits and PyTorch DataLoaders.

Usage (split + sanity check):
    python training/dataset.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.augmentation import get_eval_transforms, get_train_transforms
from utils.config_loader import load_config, resolve_path
from utils.data_cleaner import VALID_EXTENSIONS
from utils.logger import get_logger

logger = get_logger(__name__)


class BeverageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Any = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _collect_images(raw_dir: Path, classes: list[str]) -> tuple[list[Path], list[int]]:
    """Walk raw_dir and return (paths, integer labels) aligned with `classes`."""
    paths, labels = [], []
    class_to_idx = {cls.replace(" ", "_").replace("/", "-"): i for i, cls in enumerate(classes)}

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        idx = class_to_idx.get(class_dir.name)
        if idx is None:
            logger.warning(f"Directory '{class_dir.name}' not in class list — skipping")
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(img_path)
                labels.append(idx)

    return paths, labels


def split_dataset(
    raw_dir: Path,
    processed_dir: Path,
    classes: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Copy images from raw_dir into processed_dir/{train,val,test}/{class_name}/."""
    paths, labels = _collect_images(raw_dir, classes)

    if not paths:
        raise RuntimeError(f"No images found in {raw_dir}. Run downloader.py first.")

    # First split: train vs (val + test)
    test_ratio = 1.0 - train_ratio - val_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(val_ratio + test_ratio), stratify=labels, random_state=seed
    )

    # Second split: val vs test (from the temp set)
    # val_ratio relative to the temp set size
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1.0 - val_fraction), stratify=temp_labels, random_state=seed
    )

    split_map = {
        "train": (train_paths, train_labels),
        "val":   (val_paths,   val_labels),
        "test":  (test_paths,  test_labels),
    }

    idx_to_safe_name = {i: cls.replace(" ", "_").replace("/", "-") for i, cls in enumerate(classes)}

    for split_name, (split_paths, split_labels) in split_map.items():
        for src, label in zip(split_paths, split_labels):
            dst_dir = processed_dir / split_name / idx_to_safe_name[label]
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)
        logger.info(f"  {split_name}: {len(split_paths)} images")

    return {k: v[0] for k, v in split_map.items()}


def get_dataloaders(
    processed_dir: Path,
    cfg: dict,
) -> dict[str, DataLoader]:
    from torchvision.datasets import ImageFolder

    train_dataset = ImageFolder(
        root=str(processed_dir / "train"),
        transform=get_train_transforms(cfg),
    )
    val_dataset = ImageFolder(
        root=str(processed_dir / "val"),
        transform=get_eval_transforms(cfg),
    )
    test_dataset = ImageFolder(
        root=str(processed_dir / "test"),
        transform=get_eval_transforms(cfg),
    )

    loader_kwargs = {
        "batch_size":  cfg["training"]["batch_size"],
        "num_workers": cfg["training"]["num_workers"],
        "pin_memory":  cfg["training"]["pin_memory"],
    }

    return {
        "train": DataLoader(train_dataset, shuffle=True,  **loader_kwargs),
        "val":   DataLoader(val_dataset,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_dataset,  shuffle=False, **loader_kwargs),
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    cfg = load_config()
    raw_dir       = resolve_path(cfg["paths"]["data_raw"])
    processed_dir = resolve_path(cfg["paths"]["data_processed"])

    logger.info("Splitting dataset...")
    split_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        classes=cfg["classes"],
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        seed=cfg["split"]["seed"],
    )

    logger.info("Building DataLoaders...")
    loaders = get_dataloaders(processed_dir, cfg)

    # Sanity check: print shapes and save an augmentation grid
    batch, labels = next(iter(loaders["train"]))
    logger.info(f"Batch shape : {batch.shape}")   # [B, 3, 224, 224]
    logger.info(f"Labels      : {labels.tolist()}")

    grid = make_grid(batch[:8], nrow=4, normalize=True)
    img = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Training batch sample (augmented)")
    out_path = resolve_path(cfg["paths"]["models"]) / "augmentation_grid.png"
    plt.savefig(out_path, bbox_inches="tight")
    logger.info(f"Augmentation grid saved → {out_path}")
