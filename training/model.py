from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


def build_model(cfg: dict) -> nn.Module:
    """
    Create EfficientNet-B0 with a new classification head.
    All base parameters are frozen; only the classifier head is trainable.
    """
    num_classes = len(cfg["classes"])
    arch = cfg["model"]["architecture"]
    pretrained = cfg["model"]["pretrained"]

    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)

    # Freeze entire network
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier head only (phase 1)
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Built {arch} | classes={num_classes} | trainable={trainable:,}/{total:,} params")
    return model


def unfreeze_last_n_blocks(model: nn.Module, n: int) -> None:
    """
    Unfreeze the last n EfficientNet block groups plus conv_head, bn2,
    and the classifier head for phase-2 fine-tuning.
    """
    for block_group in model.blocks[-n:]:
        for param in block_group.parameters():
            param.requires_grad = True

    for component in (model.conv_head, model.bn2, model.classifier):
        for param in component.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Unfroze last {n} blocks | trainable={trainable:,}/{total:,} params")
