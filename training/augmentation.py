from __future__ import annotations

from torchvision import transforms


def get_train_transforms(cfg: dict) -> transforms.Compose:
    aug = cfg["augmentation"]
    t = aug["train"]
    return transforms.Compose([
        transforms.RandomResizedCrop(aug["image_size"]),
        transforms.RandomAffine(degrees=0, translate=(t["random_affine_translate_x"], t["random_affine_translate_y"]), fill=0),
        transforms.RandomHorizontalFlip(p=t["horizontal_flip_prob"]),
        transforms.RandomRotation(degrees=t["random_rotation_deg"]),
        transforms.RandomPerspective(distortion_scale=t["random_perspective_distortion"], p=t["random_perspective_prob"]),
        transforms.ColorJitter(
            brightness=t["color_jitter_brightness"],
            contrast=t["color_jitter_contrast"],
            saturation=t["color_jitter_saturation"],
            hue=t["color_jitter_hue"],
        ),
        transforms.RandomGrayscale(p=t["random_grayscale_prob"]),
        transforms.GaussianBlur(kernel_size=t["gaussian_blur_kernel_size"], sigma=(t["gaussian_blur_sigma_min"], t["gaussian_blur_sigma_max"])),
        transforms.ToTensor(),
        # RandomErasing must be after ToTensor and before Normalize.
        # value=0 fills in tensor space → after Normalize maps to ~-2.1, within (-5, 5) bounds.
        transforms.RandomErasing(
            p=t["random_erasing_prob"],
            scale=(t["random_erasing_scale_min"], t["random_erasing_scale_max"]),
            ratio=(t["random_erasing_ratio_min"], t["random_erasing_ratio_max"]),
            value=0,
        ),
        transforms.Normalize(mean=aug["mean"], std=aug["std"]),
    ])


def get_eval_transforms(cfg: dict) -> transforms.Compose:
    aug = cfg["augmentation"]
    return transforms.Compose([
        transforms.Resize(aug["resize_to"]),
        transforms.CenterCrop(aug["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=aug["mean"], std=aug["std"]),
    ])
