"""
OCR diagnostic script — shows raw EasyOCR fragments, confidence scores,
and which regex branch fires (or why volume extraction fails).

Usage:
    python scripts/debug_ocr.py path/to/image.jpg
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from utils.ocr_helper import (
    _EASYOCR_MIN_CONF,
    _VOLUME_PREFIXES,
    _get_easyocr_reader,
    _preprocess,
    extract_volume_from_text,
)


def run(image_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    print(f"\nImage size: {img.size}")

    reader = _get_easyocr_reader()

    # Pass 1: full image
    arr_full = np.array(_preprocess(img, min_side=1600).convert("RGB"))
    results_full = reader.readtext(arr_full)
    print(f"\n--- Pass 1: full image (upscaled to ≥1600px) ---")
    for _, text, conf in results_full:
        flag = "  KEPT" if conf >= _EASYOCR_MIN_CONF else "DROPPED"
        print(f"  [{flag}] conf={conf:.2f}  text={repr(text)}")

    # Pass 2: bottom-third crop
    w, h = img.size
    bottom_crop = img.crop((0, int(h * 0.60), w, h))
    arr_crop = np.array(_preprocess(bottom_crop, min_side=2400).convert("RGB"))
    results_crop = reader.readtext(arr_crop)
    print(f"\n--- Pass 2: bottom-third crop (upscaled to ≥2400px) ---")
    for _, text, conf in results_crop:
        flag = "  KEPT" if conf >= _EASYOCR_MIN_CONF else "DROPPED"
        print(f"  [{flag}] conf={conf:.2f}  text={repr(text)}")

    fragments = (
        [t for (_, t, c) in results_full if c >= _EASYOCR_MIN_CONF]
        + [t for (_, t, c) in results_crop if c >= _EASYOCR_MIN_CONF]
    )
    ocr_text = " ".join(fragments)
    print(f"\nJoined OCR text: {repr(ocr_text)}")

    # Regex trace
    print("\n--- Regex trace ---")
    ml_match = re.search(
        _VOLUME_PREFIXES + r"(\d+(?:\.\d+)?)\s*(?:ml|mL|ML)\b",
        ocr_text, re.IGNORECASE,
    )
    if ml_match:
        print(f"  ml regex matched: {repr(ml_match.group())} → {int(float(ml_match.group(1)))} ml")
    else:
        print("  ml regex: NO MATCH")

    l_match = re.search(
        _VOLUME_PREFIXES + r"(\d+(?:\.\d+)?)\s*(?:litre|liter|ltr|l)\b",
        ocr_text, re.IGNORECASE,
    )
    if l_match:
        print(f"  litre regex matched: {repr(l_match.group())} → {int(float(l_match.group(1)) * 1000)} ml")
    else:
        print("  litre regex: NO MATCH")

    misread_match = re.search(
        _VOLUME_PREFIXES + r"(\d+(?:\.\d+)?)\s*[012Ii]\b",
        ocr_text, re.IGNORECASE,
    )
    if misread_match:
        val = float(misread_match.group(1))
        in_range = 0.2 <= val <= 3.0
        print(f"  misread fallback matched: {repr(misread_match.group())} → val={val}, in_range={in_range}")
    else:
        print("  misread fallback: NO MATCH")

    final = extract_volume_from_text(ocr_text)
    print(f"\n==> extract_volume_from_text result: {final} ml")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_ocr.py path/to/image.jpg")
        sys.exit(1)
    run(" ".join(sys.argv[1:]))
