"""
data_prep.py
------------
Organise the two Kaggle resistor datasets into the folder
structure expected by the training pipeline.

Run once after unzipping your downloads.

Output:
    data/resistors/
        4band/   ← all 4-band images
        5band/   ← all 5-band images

Usage:
    python data_prep.py --barrett downloads/barrettotte_resistors \
                        --eralp   downloads/eralpozcan_resistors \
                        --out     data/resistors
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import random
from PIL import Image
import matplotlib.pyplot as plt

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(root: str) -> list[Path]:
    """Recursively collect all image files under root."""
    return [p for p in Path(root).rglob('*')
            if p.suffix.lower() in IMG_EXTENSIONS]


def classify_by_folder(path: Path) -> str | None:
    """
    Infer band count from the folder name.
    Returns '4band', '5band', or None if unrecognised.
    """
    parts = [p.lower() for p in path.parts]
    for part in parts:
        if '4' in part and ('band' in part or 'colour' in part or 'color' in part):
            return '4band'
        if '5' in part and ('band' in part or 'colour' in part or 'color' in part):
            return '5band'
        # Fallback: folder literally named '4' or '5'
        if part in ('4', '4band', 'four', 'fourband'):
            return '4band'
        if part in ('5', '5band', 'five', 'fiveband'):
            return '5band'
    return None


def prep_dataset(barrett_root: str | None, eralp_root: str | None,
                 out_root: str, dry_run: bool = False):

    out = Path(out_root)
    dest = {
        '4band': out / '4band',
        '5band': out / '5band',
    }

    if not dry_run:
        for d in dest.values():
            d.mkdir(parents=True, exist_ok=True)

    counts   = defaultdict(int)
    skipped  = 0
    all_imgs = []

    for source_name, source_root in [('barrett', barrett_root),
                                     ('eralp',   eralp_root)]:
        if not source_root:
            continue
        if not Path(source_root).exists():
            print(f'Warning: {source_root} not found, skipping.')
            continue

        imgs = collect_images(source_root)
        print(f'{source_name}: found {len(imgs)} images in {source_root}')

        for img_path in imgs:
            label = classify_by_folder(img_path)
            if label is None:
                skipped += 1
                continue

            # Unique filename: source_originalname_counter
            new_name = f'{source_name}_{img_path.stem}_{counts[label]:05d}{img_path.suffix}'
            dest_path = dest[label] / new_name

            if not dry_run:
                shutil.copy2(img_path, dest_path)

            counts[label] += 1
            all_imgs.append((dest_path, label))

    print(f'\n{"─"*40}')
    print(f'4-band images : {counts["4band"]}')
    print(f'5-band images : {counts["5band"]}')
    print(f'Skipped       : {skipped}  (folder name unrecognised)')
    print(f'Output        : {out_root}')

    if skipped > 0:
        print('\nTip: rename unrecognised folders to contain "4band" or "5band".')

    if not dry_run and all_imgs:
        _show_sample(all_imgs, out_root)


def _show_sample(all_imgs, out_root, n: int = 16):
    """Display a random sample grid and save it."""
    sample = random.sample(all_imgs, min(n, len(all_imgs)))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for ax, (path, label) in zip(axes.flat, sample):
        try:
            img = Image.open(path).convert('RGB')
            ax.imshow(img)
        except Exception:
            ax.set_facecolor('lightgray')
        ax.set_title(label, fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for ax in axes.flat[len(sample):]:
        ax.axis('off')

    fig.suptitle('Dataset Sample', fontsize=14)
    plt.tight_layout()
    out = Path(out_root) / 'sample_grid.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'Sample grid saved: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare resistor dataset')
    parser.add_argument('--barrett',  default=None,
                        help='Root folder of barrettotte/resistors dataset')
    parser.add_argument('--eralp',    default=None,
                        help='Root folder of eralpozcan/resistor-dataset')
    parser.add_argument('--out',      default='data/resistors',
                        help='Output root folder')
    parser.add_argument('--dry_run',  action='store_true',
                        help='Print counts without copying files')
    args = parser.parse_args()

    if not args.barrett and not args.eralp:
        print('Provide at least one dataset via --barrett or --eralp.')
    else:
        prep_dataset(args.barrett, args.eralp, args.out, args.dry_run)
