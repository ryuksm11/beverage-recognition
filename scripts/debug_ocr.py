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
    _EASYOCR_AVAILABLE,
    _EASYOCR_MIN_CONF,
    _TESSERACT_AVAILABLE,
    _VOLUME_PREFIXES,
    _get_easyocr_reader,
    _preprocess,
    extract_volume_from_text,
)


def run(image_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    print(f"\nImage size: {img.size}")

    # ── EasyOCR raw dump ────────────────────────────────────────────────────
    if not _EASYOCR_AVAILABLE:
        print("EasyOCR not installed — falling back to Tesseract")
    else:
        print(f"\n--- EasyOCR (min_conf={_EASYOCR_MIN_CONF}) ---")
        reader = _get_easyocr_reader()

        # Raw (no preprocessing)
        arr_raw = np.array(img)
        results_raw = reader.readtext(arr_raw)
        print(f"\nRaw image ({img.size}):")
        for bbox, text, conf in results_raw:
            flag = "  KEPT" if conf >= _EASYOCR_MIN_CONF else "DROPPED"
            print(f"  [{flag}] conf={conf:.2f}  text={repr(text)}")

        # With preprocessing
        preprocessed = _preprocess(img)
        arr_pre = np.array(preprocessed.convert("RGB"))
        results_pre = reader.readtext(arr_pre)
        print(f"\nPreprocessed image ({preprocessed.size}):")
        for bbox, text, conf in results_pre:
            flag = "  KEPT" if conf >= _EASYOCR_MIN_CONF else "DROPPED"
            print(f"  [{flag}] conf={conf:.2f}  text={repr(text)}")

        kept_fragments = [t for (_, t, c) in results_pre if c >= _EASYOCR_MIN_CONF]
        ocr_text = " ".join(kept_fragments)
        print(f"\nJoined OCR text: {repr(ocr_text)}")

    # ── Regex trace ─────────────────────────────────────────────────────────
    if _EASYOCR_AVAILABLE:
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
