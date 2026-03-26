"""
Tests for training/dataset.py and training/augmentation.py.
Run: pytest tests/test_dataset.py -v

These tests use synthetic data (no real images required) so they can run
before the download step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.augmentation import get_eval_transforms, get_train_transforms
from training.dataset import BeverageDataset
from utils.config_loader import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def synthetic_dataset(tmp_path, cfg):
    """Create a tiny in-memory dataset using blank PIL images."""
    images, labels = [], []
    for label in range(3):
        for _ in range(5):
            img_path = tmp_path / f"class{label}_{_}.jpg"
            Image.new("RGB", (300, 300), color=(label * 80, 100, 200)).save(img_path)
            images.append(img_path)
            labels.append(label)
    transform = get_eval_transforms(cfg)
    return BeverageDataset(images, labels, transform=transform)


def test_dataset_length(synthetic_dataset):
    assert len(synthetic_dataset) == 15


def test_dataset_output_shape(synthetic_dataset, cfg):
    tensor, label = synthetic_dataset[0]
    image_size = cfg["augmentation"]["image_size"]
    assert tensor.shape == (3, image_size, image_size)
    assert isinstance(label, int)


def test_dataset_label_range(synthetic_dataset):
    labels = [synthetic_dataset[i][1] for i in range(len(synthetic_dataset))]
    assert min(labels) == 0
    assert max(labels) == 2


def test_train_transforms_output_shape(cfg):
    transform = get_train_transforms(cfg)
    img = Image.new("RGB", (300, 300))
    tensor = transform(img)
    image_size = cfg["augmentation"]["image_size"]
    assert tensor.shape == (3, image_size, image_size)


def test_eval_transforms_output_shape(cfg):
    transform = get_eval_transforms(cfg)
    img = Image.new("RGB", (300, 300))
    tensor = transform(img)
    image_size = cfg["augmentation"]["image_size"]
    assert tensor.shape == (3, image_size, image_size)


def test_dataloader_batch_shape(synthetic_dataset, cfg):
    loader = torch.utils.data.DataLoader(
        synthetic_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    batch, labels = next(iter(loader))
    image_size = cfg["augmentation"]["image_size"]
    assert batch.shape == (4, 3, image_size, image_size)
    assert labels.shape == (4,)


def test_tensor_normalized(synthetic_dataset):
    # After ImageNet normalization, pixel values are not constrained to [0, 1]
    tensor, _ = synthetic_dataset[0]
    assert tensor.dtype == torch.float32
    # Values should be roughly in [-3, 3] range after normalization
    assert tensor.min() > -5.0
    assert tensor.max() < 5.0


def test_train_transforms_new_augmentations_output_shape(cfg):
    transform = get_train_transforms(cfg)
    img = Image.new("RGB", (300, 300), color=(200, 50, 50))
    tensor = transform(img)
    image_size = cfg["augmentation"]["image_size"]
    assert tensor.shape == (3, image_size, image_size)
    assert tensor.dtype == torch.float32


def test_train_transforms_tensor_bounds_with_erasing(cfg):
    # RandomErasing fills with value=0 in tensor space.
    # After Normalize: (0 - mean) / std ≈ -2.12 for the red channel — must stay within (-5, 5).
    transform = get_train_transforms(cfg)
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    for _ in range(30):
        tensor = transform(img)
        assert tensor.min() > -5.0
        assert tensor.max() < 5.0


def test_augmentation_config_keys_present(cfg):
    t = cfg["augmentation"]["train"]
    assert 0.0 < t["random_affine_translate_x"] <= 0.15
    assert 0.0 < t["random_affine_translate_y"] <= 0.15
    assert 0.0 < t["random_perspective_distortion"] <= 0.25
    assert 0.0 < t["random_perspective_prob"] <= 1.0
    assert 0.0 < t["random_grayscale_prob"] <= 0.3
    assert t["gaussian_blur_kernel_size"] % 2 == 1, "kernel_size must be odd"
    assert t["gaussian_blur_sigma_min"] < t["gaussian_blur_sigma_max"]
    assert t["gaussian_blur_sigma_max"] <= 1.5, "sigma above 1.5 risks OCR text degradation"
    assert 0.0 < t["random_erasing_prob"] <= 1.0
    assert t["random_erasing_scale_max"] <= 0.15, "erase area above 15% risks covering brand name"
    assert t["random_erasing_ratio_min"] < t["random_erasing_ratio_max"]
