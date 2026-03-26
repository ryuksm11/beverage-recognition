"""
Two-phase training script for the beverage CNN.

Phase 1 — classifier head only (base frozen)
Phase 2 — unfreeze last N blocks + fine-tune

Usage:
    python training/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import get_dataloaders
from training.evaluate import evaluate, save_confusion_matrix, save_training_curves
from training.model import build_model, unfreeze_last_n_blocks
from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger
from utils.seed import get_device, set_global_seed

logger = get_logger(__name__)


class _EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def _run_phase(
    phase_name: str,
    model: nn.Module,
    loaders: dict,
    cfg: dict,
    device: torch.device,
    models_dir: Path,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Run one training phase.
    Returns (train_losses, val_losses, train_accs, val_accs).
    Saves best checkpoint (by val_loss) to models_dir during this phase.
    """
    phase_cfg = cfg["training"][phase_name]
    num_epochs = phase_cfg["num_epochs"]
    lr = phase_cfg["lr"]
    weight_decay = phase_cfg["weight_decay"]

    es_cfg = cfg["training"]["early_stopping"]
    early_stopping = _EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"])

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    best_val_loss = float("inf")
    checkpoint_path = models_dir / cfg["training"]["checkpoint_filename"]

    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        val_acc, val_loss, _, _ = evaluate(
            model, loaders["val"], device, criterion, len(cfg["classes"])
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        logger.info(
            f"[{phase_name}] Epoch {epoch:02d}/{num_epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "phase": phase_name,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "classes": dataset_classes,
                    "architecture": cfg["model"]["architecture"],
                },
                checkpoint_path,
            )
            logger.info(f"  Checkpoint saved (val_loss={val_loss:.4f})")

        if early_stopping.step(val_loss):
            logger.info(f"  Early stopping triggered at epoch {epoch}")
            break

    return train_losses, val_losses, train_accs, val_accs


def run_training() -> None:
    cfg = load_config()
    set_global_seed(cfg["training"]["seed"])

    device = get_device()
    logger.info(f"Device: {device}")

    processed_dir = resolve_path(cfg["paths"]["data_processed"])
    models_dir = resolve_path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_dataloaders(processed_dir, cfg)
    # Derive class list from ImageFolder (alphabetical by directory name) — this is the
    # ground-truth mapping the model learns. cfg["classes"] uses a different order.
    dataset_classes = [c.replace("_", " ") for c in loaders["train"].dataset.classes]
    model = build_model(cfg).to(device)

    # ── Phase 1: classifier head only ─────────────────────────────────────
    logger.info("=== Phase 1: Training classifier head (base frozen) ===")
    tr_l1, vl_l1, tr_a1, vl_a1 = _run_phase("phase1", model, loaders, cfg, device, models_dir)

    # Restore best phase-1 weights before evaluating the gate
    checkpoint_path = models_dir / cfg["training"]["checkpoint_filename"]
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    gate1 = cfg["evaluation_gates"]["min_val_accuracy_phase1"]
    best_p1_acc = max(vl_a1)
    if best_p1_acc < gate1:
        logger.warning(f"Phase 1 gate FAILED: best val acc={best_p1_acc:.3f} < required {gate1}")
    else:
        logger.info(f"Phase 1 gate PASSED: best val acc={best_p1_acc:.3f} >= {gate1}")

    # ── Phase 2: fine-tune last N blocks ──────────────────────────────────
    logger.info("=== Phase 2: Fine-tuning last N blocks ===")
    n = cfg["training"]["phase2"]["unfreeze_last_n_blocks"]
    unfreeze_last_n_blocks(model, n)

    tr_l2, vl_l2, tr_a2, vl_a2 = _run_phase("phase2", model, loaders, cfg, device, models_dir)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    gate2 = cfg["evaluation_gates"]["min_val_accuracy_phase2"]
    best_p2_acc = max(vl_a2)
    if best_p2_acc < gate2:
        logger.warning(f"Phase 2 gate FAILED: best val acc={best_p2_acc:.3f} < required {gate2}")
    else:
        logger.info(f"Phase 2 gate PASSED: best val acc={best_p2_acc:.3f} >= {gate2}")

    # ── Test-set evaluation ────────────────────────────────────────────────
    logger.info("=== Evaluating on test set ===")
    num_classes = len(cfg["classes"])
    criterion = nn.CrossEntropyLoss()
    test_acc, _, per_class_f1, cm = evaluate(
        model, loaders["test"], device, criterion, num_classes, return_cm=True
    )
    logger.info(f"Test accuracy: {test_acc:.4f}")

    min_f1 = cfg["evaluation_gates"]["min_per_class_f1"]
    for cls, f1 in zip(dataset_classes, per_class_f1):
        status = "PASS" if f1 >= min_f1 else "FAIL"
        logger.info(f"  [{status}] {cls}: F1={f1:.3f}")

    save_confusion_matrix(cm, dataset_classes, models_dir)
    save_training_curves(
        tr_l1 + tr_l2,
        vl_l1 + vl_l2,
        tr_a1 + tr_a2,
        vl_a1 + vl_a2,
        phase1_end=len(tr_l1),
        save_dir=models_dir,
    )
    logger.info("Training complete.")


if __name__ == "__main__":
    run_training()
