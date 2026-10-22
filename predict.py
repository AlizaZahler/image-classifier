#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch
from image_classifier.checkpoint import load_model
from image_classifier.data import CLASSES, load_rgb, transforms_for
from image_classifier.model import resolve_device
from image_classifier.reports import write_json

@torch.inference_mode()
def predict(model, paths, device):
    model.eval()
    transform = transforms_for(False)
    results = []
    for path in paths:
        scores = model(transform(load_rgb(path)).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
        results.append({'file': str(path), 'label': list(CLASSES)[max(range(3), key=scores.__getitem__)],
                        'scores': dict(zip(CLASSES, scores)), 'score_type': 'uncalibrated model scores'})
    return results

def main():
    p = argparse.ArgumentParser(description='Predict 2d, 3d or live and uncalibrated model scores.')
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--device', default='auto')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    paths = sorted(p for p in a.input.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}) if a.input.is_dir() else [a.input]
    if not paths:
        p.error('Image not supported')
    device = resolve_device(a.device)
    model, _ = load_model(a.checkpoint, device)
    results = predict(model, paths, device)
    print(json.dumps(results, indent=2))
    if a.output:
        write_json(a.output, results)

if __name__ == '__main__':
    main()
