"""
train.py
--------
Full training loop for the resistor band classifier.

Features:
  - Label-smoothed cross-entropy  (one line in PyTorch)
  - Cosine annealing LR schedule  (one line in PyTorch)
  - Per-epoch train/val metrics
  - Checkpoint saving (best val accuracy + final)
  - Early stopping
  - GPU / MPS / CPU auto-detection
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model   import ResistorCNN, count_parameters
from dataset import make_loaders


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device('mps')
    return torch.device('cpu')


# ── One epoch of training ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total * 100


# ── Validation pass ───────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total * 100


# ── Main training function ─────────────────────────────────────────────────────
def train(
    data_root:    str   = 'data/resistors',
    save_dir:     str   = 'checkpoints',
    epochs:       int   = 60,
    batch_size:   int   = 32,
    lr:           float = 3e-4,
    label_smooth: float = 0.1,
    dropout1:     float = 0.50,
    dropout2:     float = 0.25,
    patience:     int   = 15,
    num_workers:  int   = 4,
    seed:         int   = 42,
):
    torch.manual_seed(seed)
    device = get_device()
    print(f'\nDevice: {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, class_names = make_loaders(
        data_root, batch_size, num_workers, seed=seed
    )
    num_classes = len(class_names)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ResistorCNN(num_classes=num_classes,
                        dropout1=dropout1, dropout2=dropout2).to(device)
    print(f'Parameters  : {count_parameters(model):,}')

    # ── Loss: label-smoothed cross-entropy ────────────────────────────────────
    # label_smoothing=0.1 converts one-hot targets to:
    #   [ε/K, ..., 1-ε+ε/K, ...]  where K = num_classes
    # This prevents overconfidence on noisy/ambiguous training images.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # ── Optimizer + cosine LR schedule ───────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Checkpointing + early stopping setup ─────────────────────────────────
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc  = 0.0
    epochs_no_imp = 0
    history       = {'train_loss': [], 'train_acc': [],
                     'val_loss':   [], 'val_acc':   [], 'lr': []}

    print(f'\n{"Epoch":>5}  {"LR":>8}  {"T-Loss":>8}  {"T-Acc":>7}  '
          f'{"V-Loss":>8}  {"V-Acc":>7}  {"":>6}')
    print('-' * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        improved = val_acc > best_val_acc
        tag      = '  ← best' if improved else ''

        print(f'{epoch:>5}  {current_lr:>8.2e}  {train_loss:>8.4f}  '
              f'{train_acc:>6.2f}%  {val_loss:>8.4f}  {val_acc:>6.2f}%'
              f'{tag}  ({time.time()-t0:.1f}s)')

        if improved:
            best_val_acc  = val_acc
            epochs_no_imp = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_acc':     val_acc,
                'class_names': class_names,
            }, f'{save_dir}/best_model.pth')
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f'\nEarly stopping at epoch {epoch} '
                      f'(no improvement for {patience} epochs)')
                break

    # Save final model + history
    torch.save({
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'val_acc':     val_acc,
        'class_names': class_names,
    }, f'{save_dir}/final_model.pth')

    import json
    with open(f'{save_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nBest val accuracy : {best_val_acc:.2f}%')
    print(f'Checkpoints saved : {save_dir}/')
    return history


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train resistor CNN')
    parser.add_argument('--data',         default='data/resistors')
    parser.add_argument('--save',         default='checkpoints')
    parser.add_argument('--epochs',       type=int,   default=60)
    parser.add_argument('--batch',        type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--workers',      type=int,   default=4)
    args = parser.parse_args()

    train(
        data_root    = args.data,
        save_dir     = args.save,
        epochs       = args.epochs,
        batch_size   = args.batch,
        lr           = args.lr,
        label_smooth = args.label_smooth,
        patience     = args.patience,
        num_workers  = args.workers,
    )
