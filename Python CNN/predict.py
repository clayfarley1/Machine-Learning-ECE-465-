"""
predict.py
----------
Run inference with a trained resistor band classifier.

Usage:
    # Single image
    python predict.py --checkpoint checkpoints/best_model.pth \
                      --input path/to/image.jpg

    # Folder of images
    python predict.py --checkpoint checkpoints/best_model.pth \
                      --input path/to/folder/

    # With TTA for higher confidence
    python predict.py --checkpoint checkpoints/best_model.pth \
                      --input path/to/image.jpg --tta
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from model   import ResistorCNN
from dataset import val_transform, tta_transform, IMG_SIZE


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def load_model(checkpoint_path: str, device):
    ckpt        = torch.load(checkpoint_path, map_location=device,
                             weights_only=False)
    class_names = ckpt['class_names']
    model       = ResistorCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, class_names


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert('RGB')
    return img


@torch.inference_mode()
def predict_single(model, img: Image.Image, class_names: list,
                   device, use_tta: bool = False, tta_n: int = 8):
    """
    Predict the class of a single PIL image.
    Returns (predicted_class, confidence, all_probs_dict)
    """
    tf_clean = val_transform()
    tf_tta   = tta_transform()

    # Baseline pass
    x = tf_clean(img).unsqueeze(0).to(device)
    logits = model(x)
    probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    if use_tta:
        # Enable grad temporarily for augmentation-based passes
        score_accum = probs.copy()
        for _ in range(tta_n):
            xt = tf_tta(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                lgt = model(xt)
                score_accum += F.softmax(lgt, dim=1).squeeze().cpu().numpy()
        probs = score_accum / (tta_n + 1)

    pred_idx    = int(probs.argmax())
    pred_class  = class_names[pred_idx]
    confidence  = float(probs[pred_idx]) * 100
    probs_dict  = {cn: float(p)*100 for cn, p in zip(class_names, probs)}

    return pred_class, confidence, probs_dict


def predict_folder(model, folder: str, class_names: list,
                   device, use_tta: bool = False):
    """
    Run inference on all images in a folder.
    Prints a summary table and returns a list of result dicts.
    """
    paths   = sorted(Path(folder).glob('**/*'))
    paths   = [p for p in paths if p.suffix.lower() in IMG_EXTENSIONS]

    if not paths:
        print(f'No images found in {folder}')
        return []

    results = []
    print(f'\n{"File":<45}  {"Prediction":<10}  {"Confidence":>10}')
    print('-' * 70)

    for p in paths:
        img = load_image(str(p))
        pred, conf, probs = predict_single(model, img, class_names,
                                           device, use_tta)
        results.append({'file': str(p), 'prediction': pred,
                        'confidence': conf, 'probs': probs})
        print(f'{p.name:<45}  {pred:<10}  {conf:>9.1f}%')

    # Summary
    from collections import Counter
    counts = Counter(r['prediction'] for r in results)
    print(f'\nTotal: {len(results)} images')
    for cls, n in counts.items():
        print(f'  {cls}: {n} ({n/len(results)*100:.1f}%)')

    return results


def main():
    parser = argparse.ArgumentParser(description='Resistor CNN inference')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--input',      required=True,
                        help='Path to image file or folder')
    parser.add_argument('--tta',        action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--tta_n',      type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    print(f'Device: {device}')

    model, class_names = load_model(args.checkpoint, device)
    print(f'Classes: {class_names}')

    p = Path(args.input)

    if p.is_dir():
        predict_folder(model, str(p), class_names, device, args.tta)
    elif p.is_file():
        img = load_image(str(p))
        pred, conf, probs = predict_single(model, img, class_names,
                                           device, args.tta, args.tta_n)
        print(f'\nFile       : {p.name}')
        print(f'Prediction : {pred}')
        print(f'Confidence : {conf:.1f}%')
        print('All scores :')
        for cls, score in probs.items():
            bar = '█' * int(score / 5)
            print(f'  {cls:<10} {score:>5.1f}%  {bar}')
    else:
        print(f'Error: {args.input} is not a valid file or directory.')


if __name__ == '__main__':
    main()
