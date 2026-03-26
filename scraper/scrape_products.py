"""
Scraper for the beverage product knowledge base.

Strategy:
  1. Query OpenFoodFacts API for ingredients and flavor data.
  2. Merge with MANUAL_OVERRIDES (brand origin, local manufacturer,
     packaging variants, website) — factual metadata that OpenFoodFacts
     doesn't reliably supply for Indian SKUs.
  3. Write one JSON file per class to data/product_db/.

Usage:
  python scraper/scrape_products.py               # all 9 classes
  python scraper/scrape_products.py --class "Red Bull"  # single class
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Manual overrides ──────────────────────────────────────────────────────────
# Fields that OpenFoodFacts cannot reliably supply for the Indian market.
# Packaging variants listed largest → smallest (common convention).

MANUAL_OVERRIDES: dict[str, dict] = {
    "Coca-Cola": {
        "brand_origin": "United States",
        "manufacturer": "The Coca-Cola Company",
        "local_manufacturer": "Hindustan Coca-Cola Beverages Pvt Ltd",
        "product_type": "Carbonated Soft Drink",
        "flavors": ["Original", "Zero Sugar", "Diet"],
        "default_ingredients": [
            "Carbonated Water", "Sugar", "Caramel Color (INS 150d)",
            "Acidity Regulator (INS 338)", "Caffeine", "Natural Flavours",
        ],
        "packaging": [
            {"type": "PET bottle", "volume_ml": 2000},
            {"type": "PET bottle", "volume_ml": 1500},
            {"type": "PET bottle", "volume_ml": 750},
            {"type": "PET bottle", "volume_ml": 500},
            {"type": "can", "volume_ml": 330},
            {"type": "glass bottle", "volume_ml": 300},
            {"type": "PET bottle", "volume_ml": 250},
        ],
        "website": "https://www.coca-cola.com",
    },
    "Sprite": {
        "brand_origin": "United States",
        "manufacturer": "The Coca-Cola Company",
        "local_manufacturer": "Hindustan Coca-Cola Beverages Pvt Ltd",
        "product_type": "Carbonated Soft Drink",
        "flavors": ["Original", "Zero Sugar"],
        "default_ingredients": [
            "Carbonated Water", "Sugar", "Acidity Regulator (INS 330)",
            "Natural Lemon and Lime Flavours",
        ],
        "packaging": [
            {"type": "PET bottle", "volume_ml": 2000},
            {"type": "PET bottle", "volume_ml": 1500},
            {"type": "PET bottle", "volume_ml": 600},
            {"type": "PET bottle", "volume_ml": 500},
            {"type": "can", "volume_ml": 330},
            {"type": "PET bottle", "volume_ml": 250},
        ],
        "website": "https://www.sprite.com",
    },
    "Pepsi": {
        "brand_origin": "United States",
        "manufacturer": "PepsiCo Inc.",
        "local_manufacturer": "PepsiCo India Holdings Pvt Ltd",
        "product_type": "Carbonated Soft Drink",
        "flavors": ["Original", "Black (Zero Sugar)", "Diet"],
        "default_ingredients": [
            "Carbonated Water", "Sugar", "Caramel Color (INS 150d)",
            "Acidity Regulators (INS 338, INS 330)", "Caffeine", "Natural Flavours",
        ],
        "packaging": [
            {"type": "PET bottle", "volume_ml": 2000},
            {"type": "PET bottle", "volume_ml": 1500},
            {"type": "PET bottle", "volume_ml": 750},
            {"type": "PET bottle", "volume_ml": 500},
            {"type": "can", "volume_ml": 330},
            {"type": "PET bottle", "volume_ml": 250},
        ],
        "website": "https://www.pepsi.com",
    },
    "Maaza": {
        "brand_origin": "India",
        "manufacturer": "The Coca-Cola Company",
        "local_manufacturer": "Hindustan Coca-Cola Beverages Pvt Ltd",
        "product_type": "Fruit Drink",
        "flavors": ["Mango", "Apple"],
        "default_ingredients": [
            "Water", "Mango Pulp", "Sugar", "Acidity Regulator (INS 330)",
            "Natural Mango Flavouring Substance", "Preservative (INS 211)",
        ],
        "packaging": [
            {"type": "PET bottle", "volume_ml": 1200},
            {"type": "PET bottle", "volume_ml": 600},
            {"type": "tetra pack", "volume_ml": 200},
        ],
        "website": "https://www.maazaworld.com",
    },
    "Fanta": {
        "brand_origin": "United States",
        "manufacturer": "The Coca-Cola Company",
        "local_manufacturer": "Hindustan Coca-Cola Beverages Pvt Ltd",
        "product_type": "Carbonated Soft Drink",
        "flavors": ["Orange", "Lemon", "Green Apple", "Watermelon", "Grape"],
        "default_ingredients": [
            "Carbonated Water", "Sugar", "Acidity Regulator (INS 330)",
            "Stabilizer (INS 445)", "Natural and Artificial Orange Flavours",
            "Color (INS 110)",
        ],
        "packaging": [
            {"type": "PET bottle", "volume_ml": 2000},
            {"type": "PET bottle", "volume_ml": 500},
            {"type": "can", "volume_ml": 330},
            {"type": "PET bottle", "volume_ml": 250},
        ],
        "website": "https://www.fanta.com",
    },
    "Red Bull": {
        "brand_origin": "Austria",
        "manufacturer": "Red Bull GmbH",
        "local_manufacturer": None,  # imported into India, not bottled locally
        "product_type": "Energy Drink",
        "flavors": ["Original", "Sugar Free", "Watermelon", "Tropical", "Blueberry"],
        "default_ingredients": [
            "Carbonated Water", "Sucrose", "Glucose", "Citric Acid",
            "Taurine", "Sodium Bicarbonate", "Magnesium Carbonate",
            "Caffeine", "Niacinamide", "Calcium Pantothenate",
            "Pyridoxine HCl", "Vitamin B12", "Natural and Artificial Flavours",
        ],
        "packaging": [
            {"type": "can", "volume_ml": 473},
            {"type": "can", "volume_ml": 355},
            {"type": "can", "volume_ml": 250},
        ],
        "website": "https://www.redbull.com",
    },
    "Tropicana": {
        "brand_origin": "United States",
        "manufacturer": "PepsiCo Inc.",
        "local_manufacturer": "PepsiCo India Holdings Pvt Ltd",
        "product_type": "Fruit Juice",
        "flavors": ["Orange", "Apple", "Mixed Fruit", "Guava", "Mango", "Litchi"],
        "default_ingredients": [
            "Reconstituted Fruit Juice", "Sugar", "Acidity Regulator (INS 330)",
            "Ascorbic Acid (Vitamin C)", "Natural Fruit Flavouring Substance",
        ],
        "packaging": [
            {"type": "tetra pack", "volume_ml": 1000},
            {"type": "tetra pack", "volume_ml": 200},
            {"type": "PET bottle", "volume_ml": 1000},
            {"type": "PET bottle", "volume_ml": 500},
        ],
        "website": "https://www.tropicana.com",
    },
    "7UP": {
        "brand_origin": "United States",
        "manufacturer": "PepsiCo Inc.",
        "local_manufacturer": "PepsiCo India Holdings Pvt Ltd",
        "product_type": "Carbonated Soft Drink",
        "flavors": ["Original", "Nimbooz (Lime)"],
        "default_ingredients": [
            "Carbonated Water", "Sugar", "Acidity Regulator (INS 330)",
            "Natural Lemon and Lime Flavours", "Preservative (INS 211)",
        ],
        "packaging": [
            {"type": "PET bottle", "volume_ml": 2000},
            {"type": "PET bottle", "volume_ml": 1500},
            {"type": "PET bottle", "volume_ml": 500},
            {"type": "can", "volume_ml": 330},
            {"type": "PET bottle", "volume_ml": 250},
        ],
        "website": "https://www.7up.com",
    },
    "Mirinda": {
        "brand_origin": "United States",
        "manufacturer": "PepsiCo Inc.",
        "local_manufacturer": "PepsiCo India Holdings Pvt Ltd",
        "product_type": "Carbonated Soft Drink",
        "flavors": ["Orange", "Lime", "Green Mango", "Kala Khatta"],
        "default_ingredients": [
            "Carbonated Water", "Sugar", "Acidity Regulator (INS 330)",
            "Artificial Orange Flavour", "Stabilizer (INS 445)",
            "Color (INS 110)",
        ],
        "packaging": [
            {"type": "PET bottle", "volume_ml": 2000},
            {"type": "PET bottle", "volume_ml": 500},
            {"type": "can", "volume_ml": 330},
            {"type": "PET bottle", "volume_ml": 250},
        ],
        "website": "https://www.pepsi.com/mirinda",
    },
}


# ── OpenFoodFacts helpers ─────────────────────────────────────────────────────

def _query_openfoodfacts(brand: str, cfg: dict) -> dict | None:
    """
    Query OpenFoodFacts search API and return the best-matching product dict,
    or None if no usable result is found.
    """
    params = {
        "search_terms": brand,
        "json": 1,
        "page_size": 5,
        "fields": "product_name,brands,ingredients_text,categories_tags,completeness",
    }
    headers = {"User-Agent": cfg["scraper"]["user_agent"]}
    url = cfg["scraper"]["openfoodfacts_api"]

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"OpenFoodFacts request failed for '{brand}': {exc}")
        return None

    products = data.get("products", [])
    if not products:
        return None

    # Pick the product with highest completeness whose name/brand contains the search term
    brand_lower = brand.lower().replace(" ", "")
    candidates = [
        p for p in products
        if brand_lower in p.get("product_name", "").lower().replace(" ", "")
        or brand_lower in p.get("brands", "").lower().replace(" ", "")
    ]
    if not candidates:
        candidates = products  # fall back to all results

    return max(candidates, key=lambda p: p.get("completeness", 0))


def _parse_ingredients(product: dict) -> list[str]:
    raw = product.get("ingredients_text", "") or ""
    # Split on commas/semicolons, strip whitespace and parenthetical notes
    import re
    parts = re.split(r"[,;]", raw)
    cleaned = []
    for part in parts:
        part = re.sub(r"\(.*?\)", "", part).strip(" .*-_")
        if part and len(part) > 1:
            cleaned.append(part.title())
    return cleaned[:15]  # cap at 15 to avoid noise


# ── Core scrape function ──────────────────────────────────────────────────────

def scrape_class(class_name: str, cfg: dict, output_dir: Path) -> None:
    overrides = MANUAL_OVERRIDES[class_name]
    logger.info(f"Scraping: {class_name}")

    # Query OpenFoodFacts for ingredients
    of_product = _query_openfoodfacts(class_name, cfg)
    ingredients = _parse_ingredients(of_product) if of_product else []

    # Fall back to curated ingredients if API returned nothing useful
    if not ingredients:
        logger.warning(f"  No ingredients from OpenFoodFacts for {class_name} — using curated fallback")
        ingredients = overrides["default_ingredients"]

    record = {
        "class_name": class_name,
        "brand": class_name,
        "brand_origin": overrides["brand_origin"],
        "manufacturer": overrides["manufacturer"],
        "local_manufacturer": overrides["local_manufacturer"],
        "product_type": overrides["product_type"],
        "flavors": overrides["flavors"],
        "ingredients": ingredients,
        "packaging": overrides["packaging"],
        "website": overrides["website"],
        "image_url": None,
        "last_scraped": datetime.now(timezone.utc).isoformat(),
    }

    # Write to data/product_db/{class_name_with_underscores}.json
    filename = class_name.replace(" ", "_") + ".json"
    out_path = output_dir / filename
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    logger.info(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_scraper(target_class: str | None = None) -> None:
    cfg = load_config()
    output_dir = resolve_path(cfg["paths"]["data_product_db"])
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [target_class] if target_class else cfg["classes"]
    delay = cfg["scraper"]["request_delay_sec"]

    for i, class_name in enumerate(classes):
        if class_name not in MANUAL_OVERRIDES:
            logger.error(f"No manual overrides defined for '{class_name}' — skipping")
            continue
        scrape_class(class_name, cfg, output_dir)
        if i < len(classes) - 1:
            time.sleep(delay)

    logger.info(f"Done. {len(list(output_dir.glob('*.json')))} JSON files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape beverage product info")
    parser.add_argument("--class", dest="target_class", default=None,
                        help="Scrape a single class by name (default: all classes)")
    args = parser.parse_args()
    run_scraper(args.target_class)
