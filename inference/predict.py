"""
Inference pipeline for beverage image classification.

Usage (module):
    from inference.predict import Predictor
    predictor = Predictor("models/best_checkpoint.pth", cfg)
    result = predictor.predict(pil_image)

Usage (CLI):
    python inference/predict.py --image path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import timm
import torch
import torch.nn.functional as F
from PIL import Image

from training.augmentation import get_eval_transforms
from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger
from utils.ocr_helper import (
    extract_brand_from_text,
    extract_flavor_from_text,
    extract_text_from_image,
    extract_volume_from_text,
)

logger = get_logger(__name__)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Predictor:
    """Wraps the trained EfficientNet-B0 checkpoint for single-image inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: dict,
        device: torch.device | None = None,
    ) -> None:
        self._cfg = cfg
        self._device = device or _get_device()
        self._transform = get_eval_transforms(cfg)
        self._top_k = cfg["inference"]["top_k"]
        self._tta_n = cfg["inference"].get("tta_n", 5)
        self._entropy_ratio = cfg["inference"].get("entropy_rejection_ratio", 0.75)
        self._max_entropy = math.log(9)  # log(num_classes)

        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self._classes: list[str] = ckpt["classes"]
        architecture = ckpt.get("architecture", cfg["model"]["architecture"])

        self._model = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=len(self._classes),
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"({len(self._classes)} classes, device={self._device})"
        )

    def predict(self, image: Image.Image) -> dict:
        """
        Run classification + OCR on a PIL image.

        Output schema:
            {
                "class":        str,
                "confidence":   float | None,   # None when OCR override fires
                "flavor":       str | None,
                "volume_ml":    int | None,
                "ocr_override": bool,           # True → class came from OCR brand text, not classifier
                "ood":          bool,           # True → entropy too high, likely not a beverage
                "top_k":        [{"class": str, "confidence": float}, ...]
            }
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ── TTA: average softmax over N augmented passes ─────────────────────
        tta_probs_list = []
        with torch.no_grad():
            for _ in range(self._tta_n):
                tensor = self._transform(image).unsqueeze(0).to(self._device)
                p = F.softmax(self._model(tensor), dim=1).squeeze(0)
                tta_probs_list.append(p)
        probs = torch.stack(tta_probs_list).mean(0)  # [num_classes]

        # ── Entropy-based OOD check ──────────────────────────────────────────
        entropy = -sum(
            p.item() * math.log(p.item() + 1e-10) for p in probs
        )
        ood = (entropy / self._max_entropy) > self._entropy_ratio

        k = min(self._top_k, len(self._classes))
        top_probs, top_indices = torch.topk(probs, k)
        top_k = [
            {"class": self._classes[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]
        classifier_class = top_k[0]["class"]
        classifier_conf = top_k[0]["confidence"]

        # ── OCR pipeline ─────────────────────────────────────────────────────
        ocr_text = extract_text_from_image(image)
        flavor = extract_flavor_from_text(ocr_text) if ocr_text else None
        volume_ml = extract_volume_from_text(ocr_text) if ocr_text else None
        ocr_brand = extract_brand_from_text(ocr_text) if ocr_text else None

        # ── OCR brand override ────────────────────────────────────────────────
        # If OCR positively identifies a brand that contradicts the classifier,
        # trust the label text (handles back-view images where the front logo
        # is not visible but brand name appears in the ingredients / fine print).
        if ocr_brand and ocr_brand != classifier_class:
            return {
                "class":        ocr_brand,
                "confidence":   None,       # no softmax score for OCR-derived class
                "flavor":       flavor,
                "volume_ml":    volume_ml,
                "ocr_override": True,
                "ood":          False,
                "top_k":        top_k,
            }

        return {
            "class":        classifier_class,
            "confidence":   classifier_conf,
            "flavor":       flavor,
            "volume_ml":    volume_ml,
            "ocr_override": False,
            "ood":          ood,
            "top_k":        top_k,
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Beverage image classifier")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pth checkpoint (default: models/best_checkpoint.pth)"
    )
    args = parser.parse_args()

    cfg = load_config()
    ckpt_path = (
        args.checkpoint
        or str(resolve_path(cfg["paths"]["models"]) / cfg["training"]["checkpoint_filename"])
    )

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    predictor = Predictor(ckpt_path, cfg)
    img = Image.open(args.image)
    result = predictor.predict(img)
    print(json.dumps(result, indent=2))
