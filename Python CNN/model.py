"""
model.py
--------
4-band vs 5-band resistor classifier built from scratch in PyTorch.

Architecture: 4-block VGG-style CNN
  - Filters double each block: 64 → 128 → 256 → 512
  - Two conv layers per block before MaxPool
  - Batch Norm after every conv (stabilizes training across datasets
    with different lighting/camera setups)
  - Global Average Pooling instead of Flatten (position-invariant,
    massively fewer parameters, less overfitting)
  - He initialization on all conv and linear layers
  - Dropout before each FC layer
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two conv layers with BN + ReLU, then MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),

        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),

        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class ResistorCNN(nn.Module):
    """
    4-block convolutional network for 4-band vs 5-band resistor classification.

    Input:  (N, 3, 128, 128)   RGB image batch
    Output: (N, num_classes)   raw logits (apply softmax for probabilities)
    """

    def __init__(self, num_classes: int = 2, dropout1: float = 0.5,
                 dropout2: float = 0.25):
        super().__init__()

        # ── Feature extractor ───────────────────────────────────────────
        self.features = nn.Sequential(
            _conv_block(3,   64),    # 128×128 → 64×64
            _conv_block(64,  128),   #  64×64  → 32×32
            _conv_block(128, 256),   #  32×32  → 16×16
            _conv_block(256, 512),   #  16×16  →  8×8
        )

        # Global Average Pool: (N, 512, 8, 8) → (N, 512)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout2),
            nn.Linear(256, num_classes),
        )

        # He initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)          # (N, 512)
        x = self.classifier(x)    # (N, num_classes)
        return x

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final conv block output (used for Grad-CAM)."""
        return self.features(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = ResistorCNN(num_classes=2)
    dummy = torch.randn(4, 3, 128, 128)
    out   = model(dummy)
    print(f'Output shape : {out.shape}')
    print(f'Parameters   : {count_parameters(model):,}')