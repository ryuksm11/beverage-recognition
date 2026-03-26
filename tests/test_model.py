"""
Unit tests for training/model.py.
Run: pytest tests/test_model.py -v

Uses pretrained=False so tests run offline and fast (no weight download needed).
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pytest
import timm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.model import build_model, unfreeze_last_n_blocks
from utils.config_loader import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def model(cfg):
    """EfficientNet-B0 without pretrained weights — fast, offline-safe."""
    c = copy.deepcopy(cfg)
    c["model"]["pretrained"] = False
    return build_model(c)


def test_model_output_shape(model, cfg):
    dummy = torch.zeros(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, len(cfg["classes"]))


def test_head_is_unfrozen_after_build(model):
    # Every classifier parameter must be trainable
    for param in model.classifier.parameters():
        assert param.requires_grad, "Classifier params must be trainable after build_model"

    # At least one base parameter must be frozen
    base_frozen = any(
        not p.requires_grad
        for name, p in model.named_parameters()
        if "classifier" not in name
    )
    assert base_frozen, "Base network should be frozen after build_model"


def test_unfreeze_last_n_blocks(model, cfg):
    n = cfg["training"]["phase2"]["unfreeze_last_n_blocks"]
    unfreeze_last_n_blocks(model, n)

    for block_group in model.blocks[-n:]:
        for param in block_group.parameters():
            assert param.requires_grad, f"Last {n} block groups must be trainable after unfreeze"


def test_checkpoint_save_load(model, cfg):
    dummy = torch.zeros(1, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        out_before = model(dummy)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": cfg["classes"],
                "architecture": cfg["model"]["architecture"],
            },
            ckpt_path,
        )

        loaded = timm.create_model(
            cfg["model"]["architecture"],
            pretrained=False,
            num_classes=len(cfg["classes"]),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        loaded.load_state_dict(ckpt["model_state_dict"])
        loaded.eval()

        with torch.no_grad():
            out_after = loaded(dummy)

    assert torch.allclose(out_before, out_after, atol=1e-6), \
        "Predictions must be identical after checkpoint save/load"


def test_class_probabilities_sum(model, cfg):
    dummy = torch.zeros(4, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        logits = model(dummy)

    probs = F.softmax(logits, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
        "Softmax probabilities must sum to 1.0 for each sample"
