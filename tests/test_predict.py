"""
Unit tests for inference/predict.py.
Run: pytest tests/test_predict.py -v

All tests use synthetic data and a freshly initialised (random-weight) model —
no real checkpoint or real images required.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import timm
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predict import Predictor
from utils.config_loader import load_config
from utils.ocr_helper import extract_volume_from_text


@pytest.fixture
def cfg():
    return load_config()


def _make_predictor(cfg: dict) -> Predictor:
    """Return a Predictor backed by a random-weight model — no checkpoint file needed."""
    c = copy.deepcopy(cfg)
    num_classes = len(c["classes"])
    arch = c["model"]["architecture"]

    # Build a real (random-weight) model so load_state_dict works correctly.
    real_model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    real_model.eval()

    fake_ckpt = {
        "model_state_dict": real_model.state_dict(),
        "classes": c["classes"],
        "architecture": arch,
    }

    with patch("torch.load", return_value=fake_ckpt), \
         patch("timm.create_model", return_value=real_model):
        predictor = Predictor("fake/checkpoint.pth", c, device=torch.device("cpu"))

    return predictor


@pytest.fixture
def predictor(cfg):
    return _make_predictor(cfg)


def _dummy_image(width: int = 224, height: int = 224, mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (width, height), color=(128, 64, 32))


# ── Output schema tests ───────────────────────────────────────────────────────

def test_output_schema_keys(predictor):
    result = predictor.predict(_dummy_image())
    assert set(result.keys()) == {"class", "confidence", "flavor", "volume_ml", "top_k"}


def test_output_class_is_known(predictor, cfg):
    result = predictor.predict(_dummy_image())
    assert result["class"] in cfg["classes"], \
        f"Predicted class '{result['class']}' not in known classes"


def test_output_confidence_in_range(predictor):
    result = predictor.predict(_dummy_image())
    assert 0.0 <= result["confidence"] <= 1.0


def test_top_k_length(predictor, cfg):
    result = predictor.predict(_dummy_image())
    expected_k = cfg["inference"]["top_k"]
    assert len(result["top_k"]) == expected_k, \
        f"Expected {expected_k} top_k entries, got {len(result['top_k'])}"


def test_top_k_confidences_sum_lte_one(predictor):
    result = predictor.predict(_dummy_image())
    total = sum(e["confidence"] for e in result["top_k"])
    assert total <= 1.0 + 1e-4, f"top_k confidences sum {total:.6f} exceeds 1.0"


def test_top_k_entries_have_required_keys(predictor):
    result = predictor.predict(_dummy_image())
    for entry in result["top_k"]:
        assert "class" in entry and "confidence" in entry


def test_flavor_is_none_or_string(predictor):
    result = predictor.predict(_dummy_image())
    assert result["flavor"] is None or isinstance(result["flavor"], str)


# ── Image handling tests ──────────────────────────────────────────────────────

def test_rgba_image_converted_to_rgb(predictor):
    """RGBA input must be accepted — converted to RGB internally."""
    result = predictor.predict(_dummy_image(mode="RGBA"))
    assert "class" in result


def test_non_square_image_accepted(predictor):
    """Eval transform (resize + center-crop) must handle non-square inputs."""
    result = predictor.predict(_dummy_image(width=480, height=640))
    assert "class" in result


# ── OCR integration tests ─────────────────────────────────────────────────────

def test_flavor_detected_when_ocr_returns_flavor(predictor):
    """When OCR text contains a known flavor, flavor field must be non-None."""
    with patch("inference.predict.extract_text_from_image", return_value="Tropicana Orange Juice"), \
         patch("inference.predict.extract_flavor_from_text", return_value="Orange"):
        result = predictor.predict(_dummy_image())
    assert result["flavor"] == "Orange"


def test_flavor_none_when_ocr_returns_empty(predictor):
    """When OCR finds no text, flavor must be None."""
    with patch("inference.predict.extract_text_from_image", return_value=""):
        result = predictor.predict(_dummy_image())
    assert result["flavor"] is None


# ── Volume extraction unit tests ──────────────────────────────────────────────

def test_volume_ml_in_schema(predictor):
    """volume_ml must be present in prediction output (None or int)."""
    result = predictor.predict(_dummy_image())
    assert "volume_ml" in result
    assert result["volume_ml"] is None or isinstance(result["volume_ml"], int)


def test_volume_detected_from_ocr(predictor):
    """When OCR text contains a volume string, volume_ml must match."""
    with patch("inference.predict.extract_text_from_image", return_value="Sprite 500ml Chilled"):
        result = predictor.predict(_dummy_image())
    assert result["volume_ml"] == 500
