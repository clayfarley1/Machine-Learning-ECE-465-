# Resistor Band Classifier — PyTorch CNN from Scratch

Classifies **4-band vs 5-band** resistors from images.  
Built for ECE465 ML course deliverable.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Workflow

### 1. Prepare the dataset

Download both Kaggle datasets, unzip them, then run:

```bash
python data_prep.py \
    --barrett downloads/barrettotte_resistors \
    --eralp   downloads/eralpozcan_resistors \
    --out     data/resistors
```

This builds:
```
data/resistors/
    4band/
    5band/
```

---

### 2. Train

```bash
python train.py --data data/resistors --epochs 60 --batch 32
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/resistors` | Dataset root |
| `--epochs` | `60` | Max training epochs |
| `--batch` | `32` | Batch size |
| `--lr` | `3e-4` | Initial learning rate |
| `--label_smooth` | `0.1` | Label smoothing epsilon |
| `--patience` | `15` | Early stopping patience |

Checkpoints saved to `checkpoints/best_model.pth`.

---

### 3. Evaluate

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data data/resistors \
    --tta_n 8
```

Outputs saved to `results/`:
- `confusion_matrix.png`
- `gradcam.png`
- `learning_curves.png`
- Per-class Precision / Recall / F1 printed to console

---

### 4. Predict on new images

```bash
# Single image
python predict.py --checkpoint checkpoints/best_model.pth \
                  --input path/to/image.jpg --tta

# Folder of images
python predict.py --checkpoint checkpoints/best_model.pth \
                  --input path/to/folder/
```

---

## Architecture

```
Input (3 × 128 × 128)  →  zero-center normalize
  Block 1: Conv(3×3,64)  → BN → ReLU → Conv(3×3,64)  → BN → ReLU → MaxPool
  Block 2: Conv(3×3,128) → BN → ReLU → Conv(3×3,128) → BN → ReLU → MaxPool
  Block 3: Conv(3×3,256) → BN → ReLU → Conv(3×3,256) → BN → ReLU → MaxPool
  Block 4: Conv(3×3,512) → BN → ReLU → Conv(3×3,512) → BN → ReLU
Global Average Pooling
Dropout(0.5) → FC(256) → ReLU → Dropout(0.25) → FC(2)
```

**~2.4M trainable parameters**

---

## Design Decisions

| Choice | Rationale |
|---|---|
| He initialization | Optimal variance for ReLU activations; faster convergence |
| Batch Norm after every conv | Two datasets with different cameras/lighting → different activation distributions; BN stabilizes this |
| Global Average Pooling | Resistor can appear anywhere in frame; GAP is position-invariant and kills ~8M params vs Flatten |
| Label smoothing ε=0.1 | Lighting shifts create soft label noise across datasets; prevents overconfident predictions |
| Cosine LR annealing | Smoother than step decay; no accuracy drops at step boundaries |
| WeightedRandomSampler | Handles class imbalance without duplicating images |
| TTA (8 passes) | Free accuracy gain at inference — no retraining required |

---

## File Structure

```
resistor_cnn/
├── model.py          # CNN architecture
├── dataset.py        # Data loading, augmentation, splitting
├── train.py          # Training loop
├── evaluate.py       # TTA, confusion matrix, Grad-CAM, metrics
├── predict.py        # Inference on new images
├── data_prep.py      # Kaggle dataset → folder structure
└── requirements.txt
```
