"""
Unit tests for inference/retriever.py.
Run: pytest tests/test_retriever.py -v

All tests use a synthetic in-memory product DB via tmp_path — no real
data/product_db/ files required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.retriever import ProductRetriever
from utils.config_loader import load_config


# ── Fixtures ──────────────────────────────────────────────────────────────────

_COCA_COLA = {
    "class_name": "Coca-Cola",
    "brand": "Coca-Cola",
    "brand_origin": "United States",
    "manufacturer": "The Coca-Cola Company",
    "local_manufacturer": "Hindustan Coca-Cola Beverages Pvt Ltd",
    "product_type": "Carbonated Soft Drink",
    "flavors": ["Original", "Zero Sugar"],
    "ingredients": ["Carbonated Water", "Sugar", "Caffeine"],
    "packaging": [
        {"type": "can",        "volume_ml": 330},
        {"type": "PET bottle", "volume_ml": 500},
        {"type": "PET bottle", "volume_ml": 1500},
    ],
    "website": "https://www.coca-cola.com",
    "image_url": None,
    "last_scraped": "2026-03-01T00:00:00+00:00",
}

_RED_BULL = {
    "class_name": "Red Bull",
    "brand": "Red Bull",
    "brand_origin": "Austria",
    "manufacturer": "Red Bull GmbH",
    "local_manufacturer": None,
    "product_type": "Energy Drink",
    "flavors": ["Original", "Sugar Free"],
    "ingredients": ["Carbonated Water", "Sucrose", "Taurine", "Caffeine"],
    "packaging": [
        {"type": "can", "volume_ml": 250},
        {"type": "can", "volume_ml": 355},
    ],
    "website": "https://www.redbull.com",
    "image_url": None,
    "last_scraped": "2026-03-01T00:00:00+00:00",
}


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    """Write synthetic product JSON files to a temp directory."""
    for record in [_COCA_COLA, _RED_BULL]:
        filename = record["class_name"].replace(" ", "_") + ".json"
        (tmp_path / filename).write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def retriever(cfg, db_dir):
    """ProductRetriever pointed at the synthetic db_dir."""
    with patch("inference.retriever.resolve_path", return_value=db_dir):
        return ProductRetriever(cfg)


# ── Loading tests ─────────────────────────────────────────────────────────────

def test_loads_all_records(retriever):
    assert len(retriever.known_classes) == 2


def test_known_classes_contains_expected(retriever):
    assert "Coca-Cola" in retriever.known_classes
    assert "Red Bull" in retriever.known_classes


# ── get() — exact match ───────────────────────────────────────────────────────

def test_get_exact_match_returns_record(retriever):
    product = retriever.get("Coca-Cola")
    assert product is not None
    assert product["class_name"] == "Coca-Cola"
    assert product["brand_origin"] == "United States"


def test_get_exact_match_red_bull(retriever):
    product = retriever.get("Red Bull")
    assert product is not None
    assert product["local_manufacturer"] is None  # imported, not bottled locally


# ── get() — fuzzy match ───────────────────────────────────────────────────────

def test_get_fuzzy_match_typo(retriever):
    """'Coca Cola' (missing hyphen) should still resolve to 'Coca-Cola'."""
    product = retriever.get("Coca Cola")
    assert product is not None
    assert product["class_name"] == "Coca-Cola"


def test_get_fuzzy_match_case(retriever):
    """'coca-cola' (lowercase) should still resolve."""
    product = retriever.get("coca-cola")
    assert product is not None
    assert product["class_name"] == "Coca-Cola"


# ── get() — no match ─────────────────────────────────────────────────────────

def test_get_completely_unknown_returns_none(retriever):
    result = retriever.get("XYZ Unknown Brand 12345")
    assert result is None


# ── get_container_type() ──────────────────────────────────────────────────────

def test_get_container_type_can(retriever):
    assert retriever.get_container_type("Coca-Cola", 330) == "can"


def test_get_container_type_pet_bottle(retriever):
    assert retriever.get_container_type("Coca-Cola", 500) == "PET bottle"


def test_get_container_type_large_bottle(retriever):
    assert retriever.get_container_type("Coca-Cola", 1500) == "PET bottle"


def test_get_container_type_red_bull_can(retriever):
    assert retriever.get_container_type("Red Bull", 250) == "can"


def test_get_container_type_volume_not_in_db(retriever):
    """Volume not listed in packaging variants → None."""
    assert retriever.get_container_type("Coca-Cola", 999) is None


def test_get_container_type_unknown_class(retriever):
    """Unknown class → None (not an error)."""
    assert retriever.get_container_type("XYZ Brand", 330) is None


def test_get_container_type_none_volume_graceful(retriever):
    """If volume_ml is None (OCR miss), caller should guard — but passing 0 returns None."""
    assert retriever.get_container_type("Coca-Cola", 0) is None
