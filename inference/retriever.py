"""
Product retriever for the beverage knowledge base.

Loads all JSON files from data/product_db/ on init and exposes:
  - get(class_name)                    -> ProductInfo | None
  - get_container_type(class_name, ml) -> str | None

Usage:
    from inference.retriever import ProductRetriever
    retriever = ProductRetriever(cfg)
    product = retriever.get("Sprite")
    container = retriever.get_container_type("Sprite", 500)  # → "PET bottle"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thefuzz import process as fuzz_process

from scraper.product_schema import ProductInfo
from utils.config_loader import resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

_FUZZY_THRESHOLD = 80  # minimum score (0–100) to accept a fuzzy match


class ProductRetriever:
    """Loads the product database and provides lookup by class name."""

    def __init__(self, cfg: dict) -> None:
        db_dir = resolve_path(cfg["paths"]["data_product_db"])
        self._db: dict[str, ProductInfo] = {}

        for json_path in sorted(db_dir.glob("*.json")):
            record: ProductInfo = json.loads(json_path.read_text(encoding="utf-8"))
            self._db[record["class_name"]] = record

        logger.info(f"ProductRetriever: loaded {len(self._db)} records from {db_dir}")

    @property
    def known_classes(self) -> list[str]:
        """Return the list of class names present in the product DB."""
        return list(self._db.keys())

    def get(self, class_name: str) -> ProductInfo | None:
        """
        Return the ProductInfo record for class_name.

        Tries exact match first, then falls back to fuzzy matching via thefuzz.
        Returns None when no match meets _FUZZY_THRESHOLD.
        """
        if class_name in self._db:
            return self._db[class_name]

        if not self._db:
            return None

        best_match, score = fuzz_process.extractOne(class_name, self._db.keys())
        if score >= _FUZZY_THRESHOLD:
            logger.warning(
                f"ProductRetriever: fuzzy match '{class_name}' → '{best_match}' (score={score})"
            )
            return self._db[best_match]

        logger.warning(
            f"ProductRetriever: no match for '{class_name}' "
            f"(best fuzzy score={score} < threshold {_FUZZY_THRESHOLD})"
        )
        return None

    def get_container_type(self, class_name: str, volume_ml: int) -> str | None:
        """
        Look up the container type (e.g. 'can', 'PET bottle') for a given
        class + volume_ml by scanning the product's packaging variants.

        Returns None if the class is not found or no variant matches volume_ml.
        """
        product = self.get(class_name)
        if product is None:
            return None

        for variant in product["packaging"]:
            if variant["volume_ml"] == volume_ml:
                return variant["type"]

        return None
