"""
dataset.py
----------
Data loading, augmentation, and splitting for the resistor dataset.

Expected folder layout:
    data_root/
        4 Band/   <- all 4-band resistor subfolders
        5 Band/   <- all 5-band resistor subfolders

Augmentation strategy:
    Train    - heavy (rotation, flips, color jitter, perspective)
    Val/Test - resize + normalize only
    TTA      - light random transforms averaged at inference time
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# ── Image statistics ──────────────────────────────────────────────────────────
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
IMG_SIZE = 128

# Only these extensions are treated as images (excludes __pycache__ etc.)
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in VALID_EXTS


# ── Transforms ────────────────────────────────────────────────────────────────

def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def tta_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ── Dataset splitting ─────────────────────────────────────────────────────────

def load_datasets(
    data_root: str,
    val_split:  float = 0.15,
    test_split: float = 0.10,
    seed:       int   = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Stratified train/val/test split of an ImageFolder dataset."""

    # is_valid_file filters out __pycache__ and any non-image files
    full = datasets.ImageFolder(data_root, is_valid_file=is_image)
    n    = len(full)

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n)
    labels  = np.array([full.targets[i] for i in indices])

    test_idx  = []
    val_idx   = []
    train_idx = []

    for cls in range(len(full.classes)):
        cls_idx = indices[labels == cls].tolist()
        t = int(len(cls_idx) * test_split)
        v = int(len(cls_idx) * val_split)
        test_idx  += cls_idx[:t]
        val_idx   += cls_idx[t:t+v]
        train_idx += cls_idx[t+v:]

    train_ds = datasets.ImageFolder(data_root, transform=train_transform(),
                                    is_valid_file=is_image)
    val_ds   = datasets.ImageFolder(data_root, transform=val_transform(),
                                    is_valid_file=is_image)
    test_ds  = datasets.ImageFolder(data_root, transform=val_transform(),
                                    is_valid_file=is_image)

    return (
        Subset(train_ds, train_idx),
        Subset(val_ds,   val_idx),
        Subset(test_ds,  test_idx),
    )


def make_loaders(
    data_root:   str,
    batch_size:  int   = 32,
    num_workers: int   = 0,   # 0 = no multiprocessing (required on Windows)
    val_split:   float = 0.15,
    test_split:  float = 0.10,
    seed:        int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """Build train/val/test DataLoaders with weighted sampling."""

    train_ds, val_ds, test_ds = load_datasets(
        data_root, val_split, test_split, seed
    )

    class_names  = train_ds.dataset.classes
    train_labels = [train_ds.dataset.targets[i] for i in train_ds.indices]
    counts       = np.bincount(train_labels)
    weights      = 1.0 / counts
    sample_wts   = torch.tensor([weights[l] for l in train_labels],
                                 dtype=torch.float)
    sampler = WeightedRandomSampler(sample_wts, num_samples=len(sample_wts),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=False)

    print(f'Classes     : {class_names}')
    print(f'Train       : {len(train_ds)} images')
    print(f'Val         : {len(val_ds)} images')
    print(f'Test        : {len(test_ds)} images')
    print(f'Class counts: {dict(zip(class_names, counts))}')

    return train_loader, val_loader, test_loader, class_names