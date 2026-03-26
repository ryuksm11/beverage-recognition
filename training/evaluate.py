from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
    return_cm: bool = False,
) -> tuple[float, float, np.ndarray, np.ndarray | None]:
    """
    Evaluate model on a DataLoader.
    Returns (accuracy, avg_loss, per_class_f1, confusion_matrix_or_None).
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += images.size(0)

    avg_loss = running_loss / total
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / total

    per_class_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    cm = (
        confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        if return_cm
        else None
    )
    return accuracy, avg_loss, per_class_f1, cm


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], save_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()

    out = save_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {out}")


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    phase1_end: int,
    save_dir: Path,
) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax in (ax1, ax2):
        ax.axvline(
            x=phase1_end + 0.5, color="gray", linestyle="--", alpha=0.6, label="Phase 2 start"
        )

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train acc")
    ax2.plot(epochs, val_accs, label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    fig.tight_layout()
    out = save_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Training curves saved → {out}")
