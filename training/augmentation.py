from __future__ import annotations

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class _PILToAlbumentations:
    """
    Wraps an Albumentations Compose so it accepts PIL Images (as ImageFolder delivers)
    and returns a CHW float tensor (as the training loop expects).
    """

    def __init__(self, transform: A.Compose) -> None:
        self._transform = transform

    def __call__(self, img: Image.Image):
        arr = np.array(img.convert("RGB"))
        return self._transform(image=arr)["image"]


def get_train_transforms(cfg: dict) -> _PILToAlbumentations:
    aug = cfg["augmentation"]
    t = aug["train"]
    size = aug["image_size"]

    transform = A.Compose([
        # ── Geometry ───────────────────────────────────────────────────────────
        A.RandomResizedCrop(height=size, width=size, scale=(0.6, 1.0)),
        A.HorizontalFlip(p=t["horizontal_flip_prob"]),
        A.Rotate(limit=t["random_rotation_deg"], p=0.5),
        A.Perspective(scale=(0.05, t["random_perspective_distortion"]),
                      p=t["random_perspective_prob"]),
        # Shift — simulate off-centre framing
        A.ShiftScaleRotate(
            shift_limit=t["random_shift_limit"],
            scale_limit=0,
            rotate_limit=0,
            p=0.5,
        ),

        # ── Colour ─────────────────────────────────────────────────────────────
        A.ColorJitter(
            brightness=t["color_jitter_brightness"],
            contrast=t["color_jitter_contrast"],
            saturation=t["color_jitter_saturation"],
            hue=t["color_jitter_hue"],
            p=0.8,
        ),
        A.ToGray(p=t["random_grayscale_prob"]),

        # ── Blur / noise ───────────────────────────────────────────────────────
        A.GaussianBlur(blur_limit=(3, 7), p=t["gaussian_blur_prob"]),

        # ── Real-world simulation (v2) ─────────────────────────────────────────
        # Lighting variation — shadows cast by hands, shelves, etc.
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=t["random_shadow_prob"]),
        # Camera motion — handheld phone shake
        A.MotionBlur(blur_limit=t["motion_blur_limit"], p=t["motion_blur_prob"]),
        # Partial occlusion — thumb, sticker, price tag, glare patch
        A.CoarseDropout(
            max_holes=t["coarse_dropout_holes"],
            max_height=t["coarse_dropout_size"],
            max_width=t["coarse_dropout_size"],
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=t["coarse_dropout_prob"],
        ),

        # ── Normalise + convert ────────────────────────────────────────────────
        A.Normalize(mean=aug["mean"], std=aug["std"]),
        ToTensorV2(),
    ])

    return _PILToAlbumentations(transform)


def get_eval_transforms(cfg: dict) -> _PILToAlbumentations:
    aug = cfg["augmentation"]
    size = aug["image_size"]
    resize_to = aug["resize_to"]

    transform = A.Compose([
        A.Resize(height=resize_to, width=resize_to),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=aug["mean"], std=aug["std"]),
        ToTensorV2(),
    ])

    return _PILToAlbumentations(transform)
