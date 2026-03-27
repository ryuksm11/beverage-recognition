# Requires: pip install easyocr

from __future__ import annotations

import re

import numpy as np
from PIL import Image

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

# Pass 1 (full image): keep noise low
_CONF_PASS1 = 0.2
# Passes 2 + 3 (volume-targeted crops): lower threshold to catch small/stylized text
_CONF_CROP = 0.1


def _get_easyocr_reader() -> "easyocr.Reader":
    """Initialise EasyOCR reader once and reuse across calls."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _upscale(image: Image.Image, min_side: int) -> Image.Image:
    """Upscale image so the shorter side is at least min_side pixels."""
    w, h = image.size
    if min(w, h) < min_side:
        scale = min_side / min(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract raw text from a beverage label image using EasyOCR.

    Four passes (all on color — no grayscale conversion):
      Pass 1: full image upscaled to ≥1600px — brand, flavor, general text
      Pass 2: middle band (30–70% height) at ≥2400px — label body on real photos
              where the bottle fills the frame and the label is vertically centred
      Pass 3: bottom 30% (70–100% height) at ≥2400px — volume text for product
              catalog shots where the label sits in the lower half of the image
      Pass 4: bottom 15% (85–100% height) at ≥3200px — very small NET QTY line

    Passes 2-4 use a lower confidence threshold (0.1 vs 0.2) to capture
    small, stylized text that Pass 1 misses at standard confidence.
    """
    reader = _get_easyocr_reader()
    image = image.convert("RGB")
    fragments: list[str] = []

    # Pass 1 — full image
    arr = np.array(_upscale(image, min_side=1600))
    for (_, t, c) in reader.readtext(arr):
        if c >= _CONF_PASS1:
            fragments.append(t)

    w, h = image.size

    # Pass 2 — middle band (real phone photos: label occupies centre of frame)
    crop2 = image.crop((0, int(h * 0.30), w, int(h * 0.70)))
    arr2 = np.array(_upscale(crop2, min_side=2400))
    for (_, t, c) in reader.readtext(arr2):
        if c >= _CONF_CROP:
            fragments.append(t)

    # Pass 3 — bottom 30% (product catalog shots: label in lower half)
    crop3 = image.crop((0, int(h * 0.70), w, h))
    arr3 = np.array(_upscale(crop3, min_side=2400))
    for (_, t, c) in reader.readtext(arr3):
        if c >= _CONF_CROP:
            fragments.append(t)

    # Pass 4 — bottom 15%, highest resolution
    crop4 = image.crop((0, int(h * 0.85), w, h))
    arr4 = np.array(_upscale(crop4, min_side=3200))
    for (_, t, c) in reader.readtext(arr4):
        if c >= _CONF_CROP:
            fragments.append(t)

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

    # Cross-fragment fallback: EasyOCR sometimes returns '750' and 'ml' as separate
    # non-adjacent bounding boxes (different visual baselines on the label).
    # Collect all standalone numeric tokens; if exactly one is a plausible ml volume
    # (100–3000) and 'ml' appears anywhere as a word token, pair them.
    standalone_nums = [
        float(tok) for tok in ocr_text.split()
        if re.fullmatch(r"\d+(?:\.\d+)?", tok)
    ]
    has_ml_token = bool(re.search(r"\bml\b", ocr_text, re.IGNORECASE))
    plausible_ml = [n for n in standalone_nums if 100 <= n <= 3000]
    if has_ml_token and len(set(plausible_ml)) == 1:
        return int(plausible_ml[0])

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
