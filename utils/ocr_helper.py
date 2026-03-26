# Requires: brew install tesseract && pip install pytesseract

from __future__ import annotations

import re

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

from PIL import Image, ImageFilter, ImageOps

_KNOWN_FLAVORS = [
    "Orange", "Apple", "Mango", "Lemon", "Lime", "Grape", "Strawberry",
    "Pineapple", "Watermelon", "Guava", "Litchi", "Mixed Fruit",
    "Cola", "Ginger", "Mint", "Peach", "Berry", "Cherry",
    "Zero Sugar", "Diet", "Original",
]


def extract_text_from_image(image: Image.Image, lang: str = "eng") -> str:
    if not _TESSERACT_AVAILABLE:
        return ""
    # Upscale small images so Tesseract has enough resolution to read label text,
    # then boost contrast and sharpen before passing to OCR.
    MIN_SIDE = 640
    w, h = image.size
    if min(w, h) < MIN_SIDE:
        scale = MIN_SIDE / min(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(gray, lang=lang, config="--psm 11").strip()


def extract_volume_from_text(ocr_text: str) -> int | None:
    """
    Extract volume in ml from OCR text.
    Handles: '330ml', '500 mL', '1.5 L', '1L', etc.
    Returns integer ml value, or None if not found.
    """
    if not ocr_text:
        return None
    # Direct ml/mL/ML match
    ml_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ml|mL|ML)\b", ocr_text)
    if ml_match:
        return int(float(ml_match.group(1)))
    # Litre match — convert to ml
    l_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:l|L)\b", ocr_text)
    if l_match:
        return int(float(l_match.group(1)) * 1000)
    return None


def extract_flavor_from_text(ocr_text: str, extra_flavors: list[str] | None = None) -> str | None:
    if not ocr_text:
        return None
    candidates = _KNOWN_FLAVORS + (extra_flavors or [])
    text_lower = ocr_text.lower()
    for flavor in candidates:
        # whole-word match to avoid "Cola" matching inside "Coca-Cola" body text
        if re.search(r"\b" + re.escape(flavor.lower()) + r"\b", text_lower):
            return flavor
    return None
