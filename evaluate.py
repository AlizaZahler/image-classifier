#!/usr/bin/env python3
import argparse
from pathlib import Path
from image_classifier.checkpoint import load_model, seed_all
from image_classifier.data import fingerprint, verify_dataset
from image_classifier.engine import evaluate
from image_classifier.model import resolve_device
from image_classifier.reports import evaluation_report
from image_classifier.workflow import make_loader

ROOT = Path(__file__).resolve().parent

def evaluate_checkpoint(root, checkpoint, output, device='auto', batch_size=4):
    device = resolve_device(device)
    data = verify_dataset(root)
    model, state = load_model(checkpoint, device)
    if state.get('status') != 'refitted':
        raise ValueError('Test evaluation requires a final refitted checkpoint; tuning checkpoints use validation only')
    if state['manifest_fingerprint'] != fingerprint(root):
        raise ValueError('Checkpoint was trained with different manifests')
    seed_all(state['config']['seed'])
    loader = make_loader(root, data['test'], {'batch_size': batch_size, 'workers': 0}, False)
    metrics, predictions = evaluate(model, loader, device)
    if metrics['count'] != len(data['test']):
        raise RuntimeError('Evaluation did not cover every test image exactly once')
    evaluation_report(root, data['test'], metrics, predictions, output, checkpoint)
    print(f'Test n={metrics["count"]}, accuracy={metrics["accuracy"]:.4f}, macro-F1={metrics["macro_f1"]:.4f}')
    return metrics

def main():
    p = argparse.ArgumentParser(description='Evaluate a final refitted model on the outer test set.')
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    evaluate_checkpoint(a.root, a.checkpoint, a.output or a.checkpoint.parent / 'evaluation', a.device, a.batch_size)

if __name__ == '__main__':
    main()
