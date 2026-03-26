"""
Headless schema verification for the full inference + retrieval pipeline.
Run: python scripts/verify_ui.py

Checks:
  1. Predictor loads and outputs the correct schema keys.
  2. ProductRetriever loads all 9 records.
  3. Every known class returns a non-None ProductInfo record.
  4. get_container_type() returns a string for at least one packaging variant per class.

Prints PASS / FAIL for each check and exits with code 1 if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from inference.predict import Predictor
from inference.retriever import ProductRetriever
from utils.config_loader import load_config, resolve_path

EXPECTED_SCHEMA_KEYS = {"class", "confidence", "flavor", "volume_ml", "top_k"}
EXPECTED_PRODUCT_KEYS = {
    "class_name", "brand", "brand_origin", "manufacturer", "local_manufacturer",
    "product_type", "flavors", "ingredients", "packaging", "website",
    "image_url", "last_scraped",
}


def _check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> None:
    cfg = load_config()
    all_pass = True

    print("\n=== verify_ui: loading pipeline ===")
    ckpt_path = resolve_path(cfg["paths"]["models"]) / cfg["training"]["checkpoint_filename"]
    predictor = Predictor(ckpt_path, cfg)
    retriever = ProductRetriever(cfg)

    # ── 1. Prediction schema ──────────────────────────────────────────────────
    print("\n--- Prediction schema ---")
    dummy = Image.new("RGB", (224, 224), color=(128, 64, 32))
    result = predictor.predict(dummy)

    all_pass &= _check("output has all required keys", set(result.keys()) == EXPECTED_SCHEMA_KEYS)
    all_pass &= _check("confidence in [0, 1]", 0.0 <= result["confidence"] <= 1.0)
    all_pass &= _check("class is a string", isinstance(result["class"], str))
    all_pass &= _check(
        "top_k is non-empty list of dicts",
        isinstance(result["top_k"], list)
        and len(result["top_k"]) > 0
        and all("class" in e and "confidence" in e for e in result["top_k"]),
    )
    all_pass &= _check(
        "volume_ml is None or int",
        result["volume_ml"] is None or isinstance(result["volume_ml"], int),
    )
    all_pass &= _check(
        "flavor is None or str",
        result["flavor"] is None or isinstance(result["flavor"], str),
    )

    # ── 2. Retriever — record count ───────────────────────────────────────────
    print("\n--- Retriever record count ---")
    all_pass &= _check(
        f"loaded {len(cfg['classes'])} records (one per class)",
        len(retriever.known_classes) == len(cfg["classes"]),
    )

    # ── 3. Every class resolves to a ProductInfo ──────────────────────────────
    print("\n--- Per-class product record ---")
    for class_name in cfg["classes"]:
        product = retriever.get(class_name)
        ok = product is not None and set(EXPECTED_PRODUCT_KEYS).issubset(product.keys())
        all_pass &= _check(f"{class_name}: record present and schema valid", ok)

    # ── 4. Container type lookup for first packaging variant ──────────────────
    print("\n--- Container type lookup ---")
    for class_name in cfg["classes"]:
        product = retriever.get(class_name)
        if product and product["packaging"]:
            first = product["packaging"][0]
            ct = retriever.get_container_type(class_name, first["volume_ml"])
            all_pass &= _check(
                f"{class_name} {first['volume_ml']}ml → '{ct}'",
                isinstance(ct, str) and len(ct) > 0,
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'All checks passed.' if all_pass else 'Some checks FAILED.'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
