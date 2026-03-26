"""
2-epoch smoke test for the training pipeline.
Usage: python scripts/smoke_test_training.py

Loads up to 10 images per class from data/processed/train/, trains for 2 epochs
using a tiny DataLoader, then runs PASS/FAIL assertions on key invariants.
Exits with code 1 if any assertion fails.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import ImageFolder

from training.augmentation import get_train_transforms
from training.model import build_model
from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)

_RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    _RESULTS.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    logger.info(f"  [{status}] {name}{suffix}")


def run_smoke_test() -> None:
    cfg = load_config()
    set_global_seed(cfg["training"]["seed"])

    device = torch.device("cpu")  # always CPU for portability
    num_classes = len(cfg["classes"])

    # ── Tiny dataset: up to 10 images per class ───────────────────────────
    processed_dir = resolve_path(cfg["paths"]["data_processed"])
    train_root = processed_dir / "train"

    if not train_root.exists():
        logger.error(f"Processed train dir not found: {train_root}")
        logger.error("Run `python training/dataset.py` first.")
        sys.exit(1)

    full_dataset = ImageFolder(str(train_root), transform=get_train_transforms(cfg))

    per_class: dict[int, list[int]] = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(full_dataset.samples):
        per_class[label].append(idx)

    selected: list[int] = []
    for indices in per_class.values():
        selected.extend(indices[:10])

    subset = torch.utils.data.Subset(full_dataset, selected)
    loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=True, num_workers=0)

    # ── Model (no pretrained weights for speed) ───────────────────────────
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["model"]["pretrained"] = False
    model = build_model(smoke_cfg).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # ── 2-epoch training loop ─────────────────────────────────────────────
    epoch_losses: list[float] = []
    first_batch_output: torch.Tensor | None = None

    for epoch in range(1, 3):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if first_batch_output is None:
                first_batch_output = outputs.detach()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        epoch_loss = running_loss / total
        epoch_losses.append(epoch_loss)
        logger.info(f"  Epoch {epoch}: loss={epoch_loss:.4f}")

    # ── Assertions ────────────────────────────────────────────────────────
    logger.info("\nSmoke test assertions:")

    _check(
        "output shape",
        first_batch_output is not None and first_batch_output.shape[1] == num_classes,
        f"expected [..., {num_classes}], got shape {first_batch_output.shape if first_batch_output is not None else 'N/A'}",
    )

    _check(
        "loss decreases epoch 1 → 2",
        epoch_losses[1] < epoch_losses[0],
        f"epoch1={epoch_losses[0]:.4f}  epoch2={epoch_losses[1]:.4f}",
    )

    model.eval()
    dummy = torch.zeros(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy)
        probs = F.softmax(logits, dim=1)
        sums = probs.sum(dim=1)

    _check(
        "softmax sums to 1",
        torch.allclose(sums, torch.ones(4), atol=1e-5),
        f"sums={[round(s, 6) for s in sums.tolist()]}",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke_test.pth"
        torch.save({"model_state_dict": model.state_dict(), "classes": cfg["classes"]}, ckpt_path)

        loaded = timm.create_model(
            cfg["model"]["architecture"], pretrained=False, num_classes=num_classes
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        loaded.load_state_dict(ckpt["model_state_dict"])
        loaded.eval()

        with torch.no_grad():
            out_orig = model(dummy)
            out_loaded = loaded(dummy)

        _check(
            "checkpoint round-trip",
            torch.allclose(out_orig, out_loaded, atol=1e-6),
            "predictions must match after save/load",
        )

    # ── Summary ───────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total_checks = len(_RESULTS)
    logger.info(f"\nSmoke test: {passed}/{total_checks} checks passed")

    if passed < total_checks:
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
