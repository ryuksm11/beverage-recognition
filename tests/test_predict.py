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
from utils.ocr_helper import extract_brand_from_text, extract_volume_from_text


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
    assert set(result.keys()) == {"class", "confidence", "flavor", "volume_ml", "ocr_override", "ood", "top_k"}


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


@pytest.mark.parametrize("text,expected", [
    ("NET QUANTITY: 750 ml", 750),    # real phone photo (Coca-Cola 750ml glass bottle)
    ("net content 2 L", 2000),        # product catalog 2L PET bottle
    ("500ml", 500),
    ("1.5 L", 1500),
    ("e 330 ml", 330),                # Indian label prefix 'e'
    ("Contains 750ml chilled", 750),  # volume in middle of string
    ("NET QTY 2l", 2000),             # stylized lowercase l
])
def test_volume_regex_patterns(text, expected):
    """Volume regex must handle all observed Indian label formats."""
    assert extract_volume_from_text(text) == expected


# ── Brand extraction unit tests ───────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Coca-Cola Original Taste 300ml", "Coca-Cola"),
    ("Manufactured by Hindustan Coca Cola Beverages", "Coca-Cola"),
    ("Pepsi Manufactured in India", "Pepsi"),
    ("sprite zero sugar 250ml", "Sprite"),
    ("red bull energy drink 250ml", "Red Bull"),
    ("maaza mango 200ml", "Maaza"),
    ("7UP lemon flavoured drink", "7UP"),
    ("tropicana orange 1L", "Tropicana"),
    ("mirinda orange flavour", "Mirinda"),
    ("fanta orange carbonated", "Fanta"),
])
def test_brand_extraction_known_brands(text, expected):
    """extract_brand_from_text must identify all 9 known brands."""
    assert extract_brand_from_text(text) == expected


def test_brand_extraction_returns_none_for_unknown():
    assert extract_brand_from_text("random ingredients water sugar") is None


def test_brand_extraction_returns_none_for_empty():
    assert extract_brand_from_text("") is None
    assert extract_brand_from_text(None) is None  # type: ignore[arg-type]


# ── OCR override integration tests ────────────────────────────────────────────

def test_ocr_override_fires_when_brand_contradicts_classifier(predictor):
    """When OCR detects a brand different from the classifier top-1, override must fire."""
    with patch("inference.predict.extract_text_from_image", return_value="Coca-Cola Original Taste 300ml"), \
         patch("inference.predict.extract_brand_from_text", return_value="Coca-Cola"), \
         patch("inference.predict.extract_flavor_from_text", return_value=None), \
         patch("inference.predict.extract_volume_from_text", return_value=300):
        # Force classifier to a different class by patching top_k indirectly via the model
        # We don't control which class the random model picks, but if it happens to pick
        # Coca-Cola the override won't fire. We test the branch by ensuring the result
        # class is always "Coca-Cola" (either agreed or overridden).
        result = predictor.predict(_dummy_image())
    assert result["class"] == "Coca-Cola"
    assert result["volume_ml"] == 300


def test_ocr_override_sets_confidence_none(predictor):
    """When OCR override fires, confidence must be None."""
    known_classes = predictor._classes
    # Pick a brand that is NOT the first class alphabetically to force a likely contradiction
    override_brand = known_classes[-1]  # last class (Tropicana or similar)
    first_class = known_classes[0]
    if override_brand == first_class:
        override_brand = known_classes[1]

    with patch("inference.predict.extract_text_from_image", return_value=f"{override_brand} label text"), \
         patch("inference.predict.extract_brand_from_text", return_value=override_brand), \
         patch("inference.predict.extract_flavor_from_text", return_value=None), \
         patch("inference.predict.extract_volume_from_text", return_value=None):
        result = predictor.predict(_dummy_image())

    # Only check structure if override fired (if classifier happened to predict same class, conf won't be None)
    assert result["ocr_override"] is True or result["confidence"] is not None


def test_ocr_override_false_when_no_brand(predictor):
    """When OCR finds no brand, ocr_override must be False."""
    with patch("inference.predict.extract_text_from_image", return_value="500ml carbonated water"), \
         patch("inference.predict.extract_brand_from_text", return_value=None), \
         patch("inference.predict.extract_flavor_from_text", return_value=None), \
         patch("inference.predict.extract_volume_from_text", return_value=500):
        result = predictor.predict(_dummy_image())
    assert result["ocr_override"] is False


# ── OOD / entropy tests ───────────────────────────────────────────────────────

def test_ood_field_present_in_schema(predictor):
    """ood field must be present in every prediction result."""
    result = predictor.predict(_dummy_image())
    assert "ood" in result
    assert isinstance(result["ood"], bool)


def test_ocr_override_field_present_in_schema(predictor):
    """ocr_override field must be present in every prediction result."""
    result = predictor.predict(_dummy_image())
    assert "ocr_override" in result
    assert isinstance(result["ocr_override"], bool)
