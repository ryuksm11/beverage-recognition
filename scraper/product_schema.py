"""
TypedDict definitions for the product database schema.
Used by scrape_products.py (writing) and retriever.py (reading).
"""

from __future__ import annotations

from typing import Optional
from typing import TypedDict


class PackagingVariant(TypedDict):
    type: str        # "can" | "PET bottle" | "glass bottle" | "tetra pack"
    volume_ml: int   # e.g. 330, 500, 1000, 1500


class ProductInfo(TypedDict):
    class_name: str
    brand: str
    brand_origin: str                    # home country of the brand
    manufacturer: str                    # global parent company
    local_manufacturer: Optional[str]    # bottler/manufacturer in India; null if imported
    product_type: str
    flavors: list[str]
    ingredients: list[str]
    packaging: list[PackagingVariant]    # all known packaging variants
    website: str
    image_url: Optional[str]
    last_scraped: str                    # ISO 8601 datetime string
