"""
evaluate.py
-----------
Full evaluation of a trained resistor CNN on the held-out test set.

Produces:
  1. Baseline accuracy  (single forward pass)
  2. TTA accuracy       (N augmented passes, averaged scores)
  3. Confusion matrix   (saved as PNG)
  4. Per-class Precision / Recall / F1 / Support
  5. Grad-CAM saliency maps (saved as PNG)
  6. Learning curves    (saved as PNG, loaded from history.json)

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth \
                       --data data/resistors
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from model   import ResistorCNN
from dataset import make_loaders, val_transform, tta_transform, IMG_SIZE, MEAN, STD


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():   return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


# ── Load checkpoint ───────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device):
    ckpt        = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt['class_names']
    model       = ResistorCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt["epoch"]}, '
          f'val acc {ckpt["val_acc"]:.2f}%)')
    return model, class_names


# ── Standard inference ────────────────────────────────────────────────────────
@torch.inference_mode()
def predict_loader(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())
    return (torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())


# ── Test-Time Augmentation ────────────────────────────────────────────────────
def predict_tta(model, test_dataset, device, n_passes: int = 8,
                batch_size: int = 32):
    """
    Run N augmented forward passes over the test set.
    Average softmax scores, then take argmax.
    Adds ~1-3% accuracy over single-pass inference at zero training cost.
    """
    from torch.utils.data import DataLoader
    from copy import deepcopy

    n = len(test_dataset)
    num_classes = len(test_dataset.dataset.classes)

    score_accum = np.zeros((n, num_classes), dtype=np.float32)

    # One clean pass first
    clean_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2)
    _, true_labels, clean_probs = predict_loader(model, clean_loader, device)
    score_accum += clean_probs

    # N augmented passes
    tta_ds = deepcopy(test_dataset)
    tta_ds.dataset.transform = tta_transform()

    for t in range(n_passes):
        loader = DataLoader(tta_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2)
        _, _, probs = predict_loader(model, loader, device)
        score_accum += probs
        print(f'  TTA pass {t+1}/{n_passes}')

    tta_preds = score_accum.argmax(axis=1)
    return tta_preds, true_labels, score_accum


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Resistor CNN – Confusion Matrix (TTA)', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Confusion matrix saved: {save_path}')


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
    Highlights which image regions the network uses to make its prediction.
    Hooks into the final conv block's output.
    """

    def __init__(self, model: ResistorCNN):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Target: output of the last conv block (features[-1])
        target_layer = self.model.features[-1][-2]  # last BN before pool

        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        img_tensor : (1, 3, H, W)
        Returns    : (H, W) heatmap in [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(img_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return cam, class_idx


def plot_gradcam(model, test_dataset, class_names, device,
                 n_samples: int = 6, save_path: str = 'gradcam.png'):
    gradcam    = GradCAM(model)
    inv_norm   = transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD])

    indices = np.random.choice(len(test_dataset), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 2,
                             figsize=(6, 2.5 * n_samples))

    for row, idx in enumerate(indices):
        img_t, true_label = test_dataset[idx]
        img_t  = img_t.unsqueeze(0).to(device)
        img_t.requires_grad_(False)

        # Grad-CAM needs gradients through the model
        model.train()    # enable grad computation
        cam, pred_idx = gradcam(img_t.clone().requires_grad_(True))
        model.eval()

        # Denormalize for display
        img_disp = inv_norm(img_t.squeeze().cpu()).permute(1, 2, 0).numpy()
        img_disp = np.clip(img_disp, 0, 1)

        # Overlay heatmap
        heatmap = cm.jet(cam)[..., :3]
        overlay = 0.55 * img_disp + 0.45 * heatmap

        axes[row, 0].imshow(img_disp)
        axes[row, 0].set_title(f'True: {class_names[true_label]}',
                                fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f'Pred: {class_names[pred_idx]}',
                                fontsize=9)
        axes[row, 1].axis('off')

    fig.suptitle('Grad-CAM Saliency – Final Conv Block', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Grad-CAM saved: {save_path}')


# ── Learning curves ───────────────────────────────────────────────────────────
def plot_learning_curves(history_path: str, test_acc: float,
                         save_path: str = 'learning_curves.png'):
    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(epochs, h['train_loss'], label='Train')
    ax1.semilogy(epochs, h['val_loss'],   label='Val', linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, h['train_acc'], label='Train')
    ax2.plot(epochs, h['val_acc'],   label='Val', linestyle='--')
    ax2.axhline(test_acc, color='k', linestyle=':', linewidth=1.5,
                label=f'Test TTA ({test_acc:.1f}%)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True)

    fig.suptitle('Training History', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Learning curves saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(checkpoint: str, data_root: str, save_dir: str = 'results',
             tta_n: int = 8, batch_size: int = 32):

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()

    model, class_names = load_model(checkpoint, device)

    # Load test set
    from dataset import load_datasets
    _, _, test_ds = load_datasets(data_root)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Baseline
    print('\n--- Baseline (single pass) ---')
    base_preds, true_labels, _ = predict_loader(model, test_loader, device)
    base_acc = (base_preds == true_labels).mean() * 100
    print(f'Baseline accuracy: {base_acc:.2f}%')

    # TTA
    print(f'\n--- TTA ({tta_n} passes) ---')
    tta_preds, true_labels, _ = predict_tta(model, test_ds, device,
                                             n_passes=tta_n,
                                             batch_size=batch_size)
    tta_acc = (tta_preds == true_labels).mean() * 100
    print(f'TTA accuracy: {tta_acc:.2f}%')

    # Classification report
    print('\n--- Per-class Metrics ---')
    print(classification_report(true_labels, tta_preds,
                                 target_names=class_names, digits=3))

    # Plots
    plot_confusion_matrix(true_labels, tta_preds, class_names,
                          f'{save_dir}/confusion_matrix.png')

    plot_gradcam(model, test_ds, class_names, device,
                 n_samples=6, save_path=f'{save_dir}/gradcam.png')

    history_path = Path(checkpoint).parent / 'history.json'
    if history_path.exists():
        plot_learning_curves(str(history_path), tta_acc,
                             f'{save_dir}/learning_curves.png')

    print(f'\nAll results saved to: {save_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--data',       default='data/resistors')
    parser.add_argument('--save',       default='results')
    parser.add_argument('--tta_n',      type=int, default=8)
    parser.add_argument('--batch',      type=int, default=32)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data, args.save, args.tta_n, args.batch)