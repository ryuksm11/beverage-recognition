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

import timm
import torch
import torch.nn.functional as F
from PIL import Image

from training.augmentation import get_eval_transforms
from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger
from utils.ocr_helper import extract_flavor_from_text, extract_text_from_image, extract_volume_from_text

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
        Run classification + OCR flavor detection on a PIL image.

        Returns dict matching the confirmed output schema:
            {
                "class": str,
                "confidence": float,
                "flavor": str | None,
                "top_k": [{"class": str, "confidence": float}, ...]
            }
        """
        # Ensure RGB (handles RGBA, palette images, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self._transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        k = min(self._top_k, len(self._classes))
        top_probs, top_indices = torch.topk(probs, k)

        top_k = [
            {"class": self._classes[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

        # OCR — flavor and volume detection (gracefully skipped if tesseract unavailable)
        ocr_text = extract_text_from_image(image)
        flavor = extract_flavor_from_text(ocr_text) if ocr_text else None
        volume_ml = extract_volume_from_text(ocr_text) if ocr_text else None

        return {
            "class": top_k[0]["class"],
            "confidence": top_k[0]["confidence"],
            "flavor": flavor,
            "volume_ml": volume_ml,
            "top_k": top_k,
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

    predictor = Predictor(ckpt_path, cfg)
    img = Image.open(args.image)
    result = predictor.predict(img)
    print(json.dumps(result, indent=2))
