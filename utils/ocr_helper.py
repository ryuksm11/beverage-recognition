# Requires: pip install easyocr

from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import easyocr
    _easyocr_reader: easyocr.Reader | None = None  # lazy-initialised on first use
except ImportError as e:
    raise ImportError("EasyOCR is required: pip install easyocr==1.7.2") from e

_KNOWN_FLAVORS = [
    "Orange", "Apple", "Mango", "Lemon", "Lime", "Grape", "Strawberry",
    "Pineapple", "Watermelon", "Guava", "Litchi", "Mixed Fruit",
    "Cola", "Ginger", "Mint", "Peach", "Berry", "Cherry",
    "Zero Sugar", "Diet", "Original",
]

# Brand name fragments that should NOT be treated as flavor matches.
_BRAND_FRAGMENTS = ["coca-cola", "coca cola", "cocacola", "pepsi-cola"]

# Volume label prefixes common on Indian packaged beverages
_VOLUME_PREFIXES = r"(?:net\s+(?:content|quantity|qty|wt\.?)|net\.?\s*e|e\s+)?\s*"

# Minimum EasyOCR confidence to include a text fragment
_EASYOCR_MIN_CONF = 0.2


def _get_easyocr_reader() -> "easyocr.Reader":
    """Initialise EasyOCR reader once and reuse across calls."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _preprocess(image: Image.Image, min_side: int = 1600) -> Image.Image:
    """Upscale small images, boost contrast and sharpen for better OCR."""
    w, h = image.size
    if min(w, h) < min_side:
        scale = min_side / min(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract raw text from a beverage label image using EasyOCR.
    Runs two passes: full image + bottom-third crop for small volume text.
    Returns empty string if EasyOCR finds nothing.
    """
    reader = _get_easyocr_reader()

    # Pass 1: full image upscaled to ≥1600px — captures brand, flavor, general text
    arr_full = np.array(_preprocess(image, min_side=1600).convert("RGB"))
    results_full = reader.readtext(arr_full)
    fragments = [t for (_, t, c) in results_full if c >= _EASYOCR_MIN_CONF]

    # Pass 2: bottom-third crop upscaled to ≥2400px — targets small volume text
    # (e.g. "2l", "500ml") typically printed near the base of the label
    w, h = image.size
    bottom_crop = image.crop((0, int(h * 0.60), w, h))
    arr_crop = np.array(_preprocess(bottom_crop, min_side=2400).convert("RGB"))
    results_crop = reader.readtext(arr_crop)
    fragments += [t for (_, t, c) in results_crop if c >= _EASYOCR_MIN_CONF]

    return " ".join(fragments)


def extract_volume_from_text(ocr_text: str) -> int | None:
    """
    Extract volume in ml from OCR text.
    Handles common Indian label formats:
      '330ml', '500 mL', '500ML', 'NET CONTENT 500 ml',
      'Net Quantity: 500ml', '1.5 L', '1L', 'e 500 ml'
    Returns integer ml value, or None if not found.
    """
    if not ocr_text:
        return None

    ml_match = re.search(
        _VOLUME_PREFIXES + r"(\d+(?:\.\d+)?)\s*(?:ml|mL|ML)\b",
        ocr_text,
        re.IGNORECASE,
    )
    if ml_match:
        return int(float(ml_match.group(1)))

    l_match = re.search(
        _VOLUME_PREFIXES + r"(\d+(?:\.\d+)?)\s*(?:litre|liter|ltr|l)\b",
        ocr_text,
        re.IGNORECASE,
    )
    if l_match:
        return int(float(l_match.group(1)) * 1000)

    # Fallback: OCR misreads lowercase 'l' (litre) as '0', '1', '2', 'I', or 'i'
    # depending on the bottle font (observed: Coca-Cola 2L → "22", "21", "20").
    # Only fires for plausible litre quantities (0.2–3 L) to limit false positives.
    misread_match = re.search(
        _VOLUME_PREFIXES + r"(\d+(?:\.\d+)?)\s*[012Ii]\b",
        ocr_text,
        re.IGNORECASE,
    )
    if misread_match:
        val = float(misread_match.group(1))
        if 0.2 <= val <= 3.0:
            return int(val * 1000)

    return None


def extract_flavor_from_text(ocr_text: str, extra_flavors: list[str] | None = None) -> str | None:
    if not ocr_text:
        return None
    candidates = _KNOWN_FLAVORS + (extra_flavors or [])
    text_lower = ocr_text.lower()

    # Strip known brand fragments before flavor scanning to avoid false positives
    cleaned = text_lower
    for fragment in _BRAND_FRAGMENTS:
        cleaned = cleaned.replace(fragment, " ")

    for flavor in candidates:
        if re.search(r"\b" + re.escape(flavor.lower()) + r"\b", cleaned):
            return flavor
    return None
