# Requires: pip install easyocr  (primary)
# Fallback:  brew install tesseract && pip install pytesseract

from __future__ import annotations

import re

import numpy as np

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
    _easyocr_reader: easyocr.Reader | None = None  # lazy-initialised on first use
except ImportError:
    _EASYOCR_AVAILABLE = False
    _easyocr_reader = None

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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


def _extract_with_easyocr(image: Image.Image) -> str:
    reader = _get_easyocr_reader()

    # Pass 1: full image, upscaled to ≥1600px for general text
    arr_full = np.array(_preprocess(image, min_side=1600).convert("RGB"))
    results_full = reader.readtext(arr_full)
    fragments = [t for (_, t, c) in results_full if c >= _EASYOCR_MIN_CONF]

    # Pass 2: bottom-third crop, upscaled to ≥2400px — volume text (e.g. "2l")
    # is typically printed small near the bottom of beverage labels
    w, h = image.size
    bottom_crop = image.crop((0, int(h * 0.60), w, h))
    arr_crop = np.array(_preprocess(bottom_crop, min_side=2400).convert("RGB"))
    results_crop = reader.readtext(arr_crop)
    fragments += [t for (_, t, c) in results_crop if c >= _EASYOCR_MIN_CONF]

    return " ".join(fragments)


def _alphanum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalnum() for c in text) / len(text)


def _extract_with_tesseract(image: Image.Image, lang: str = "eng") -> str:
    preprocessed = _preprocess(image)
    best = ""
    for psm in ["--psm 6", "--psm 3"]:
        result = pytesseract.image_to_string(
            preprocessed, lang=lang, config=psm
        ).strip()
        if _alphanum_ratio(result) > _alphanum_ratio(best):
            best = result
    return best


def extract_text_from_image(image: Image.Image, lang: str = "eng") -> str:
    """
    Extract raw text from a beverage label image.
    Uses EasyOCR (primary) or Tesseract (fallback).
    Returns empty string if neither engine is available.
    """
    if _EASYOCR_AVAILABLE:
        return _extract_with_easyocr(image)
    if _TESSERACT_AVAILABLE:
        return _extract_with_tesseract(image, lang=lang)
    return ""


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
