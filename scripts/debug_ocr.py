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
    _CONF_CROP,
    _CONF_PASS1,
    _VOLUME_PREFIXES,
    _get_easyocr_reader,
    _upscale,
    extract_volume_from_text,
)


def _run_pass(reader, image: Image.Image, min_side: int, conf_threshold: float, label: str) -> list[str]:
    arr = np.array(_upscale(image, min_side=min_side))
    results = reader.readtext(arr)
    print(f"\n--- {label} ---")
    kept = []
    for _, text, conf in results:
        flag = "  KEPT" if conf >= conf_threshold else "DROPPED"
        print(f"  [{flag}] conf={conf:.2f}  text={repr(text)}")
        if conf >= conf_threshold:
            kept.append(text)
    return kept


def run(image_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"\nImage size: {img.size}")

    reader = _get_easyocr_reader()
    all_fragments: list[str] = []

    # Pass 1 — full image
    all_fragments += _run_pass(
        reader, img, min_side=1600, conf_threshold=_CONF_PASS1,
        label=f"Pass 1: full image (≥1600px, conf≥{_CONF_PASS1})"
    )

    # Pass 2 — middle band (real photos: label occupies centre of frame)
    crop2 = img.crop((0, int(h * 0.30), w, int(h * 0.70)))
    all_fragments += _run_pass(
        reader, crop2, min_side=2400, conf_threshold=_CONF_CROP,
        label=f"Pass 2: middle band 30–70% (≥2400px, conf≥{_CONF_CROP})"
    )

    # Pass 3 — bottom 30% (catalog shots: label in lower half)
    crop3 = img.crop((0, int(h * 0.70), w, h))
    all_fragments += _run_pass(
        reader, crop3, min_side=2400, conf_threshold=_CONF_CROP,
        label=f"Pass 3: bottom 30% crop (≥2400px, conf≥{_CONF_CROP})"
    )

    # Pass 4 — bottom 15%, highest resolution
    crop4 = img.crop((0, int(h * 0.85), w, h))
    all_fragments += _run_pass(
        reader, crop4, min_side=3200, conf_threshold=_CONF_CROP,
        label=f"Pass 4: bottom 15% crop (≥3200px, conf≥{_CONF_CROP})"
    )

    ocr_text = " ".join(all_fragments)
    print(f"\nJoined OCR text:\n  {repr(ocr_text)}")

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
    print(f"\n==> extract_volume_from_text result: {final}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_ocr.py path/to/image.jpg")
        sys.exit(1)
    run(" ".join(sys.argv[1:]))
