#!/usr/bin/env python3
"""Real pretrained ConvNeXt acceptance checks, using fitting images only."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from image_classifier.checkpoint import atomic_save, load_model, model_metadata, seed_all
from image_classifier.data import ImageDataset, counts, validation_split, verify_dataset
from image_classifier.engine import evaluate, train_epoch
from image_classifier.model import configure_training, create_model, make_optimizer
from image_classifier.reports import write_json
from image_classifier.workflow import class_weights

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'runs'/'smoke')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'config.json').read_text())
    seed_all(config['seed'])
    data = verify_dataset(ROOT)
    fit, val = validation_split(data['train'])
    subset = [next(r for r in fit if r['label'] == label) for label in ('2d','3d','live')]
    loader = DataLoader(ImageDataset(ROOT, subset, training=False), batch_size=1)
    report = {'data': {k: counts(v) for k,v in {**data,'fit':fit,'validation':val}.items()}, 'devices': {},
              'note': 'Smoke checks only; this is not a trained final model. No test metrics computed.'}
    devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
    torch.set_num_threads(2)
    for name in devices:
        print(f'Checking real ConvNeXt on {name}', flush=True)
        device = torch.device(name)
        model = create_model().to(device)
        configure_training(model, 'head')
        # Compare frozen weights exactly without keeping a second GPU model.
        frozen = {k: v.detach().cpu().clone() for k,v in model.features.state_dict().items()}
        optimizer = make_optimizer(model, 0., .003, .01)
        weights = class_weights(subset, device)
        losses = []
        for _ in range(3):
            losses.append(train_epoch(model, loader, optimizer, device, weights, 'head', 3)['loss'])
        assert losses[-1] < losses[0], losses
        assert all(torch.equal(v, model.features.state_dict()[k].cpu()) for k,v in frozen.items())
        assert not model.features.training
        before = evaluate(model, loader, device)[1]['scores']
        path = args.output/f'{name}_smoke.pt'
        atomic_save(path, {**model_metadata(config), 'model': model.state_dict(), 'status': 'smoke_only'})
        del optimizer, frozen, model
        gc.collect()
        model, state = load_model(path, device)
        after = evaluate(model, loader, device)[1]['scores']
        torch.testing.assert_close(torch.tensor(before), torch.tensor(after), atol=1e-6, rtol=1e-5)
        # Full-network backward exercises the much larger fine-tuning graph.
        optimizer = make_optimizer_full(model)
        full = train_epoch(model, loader, optimizer, device, weights, 'full', 3)
        report['devices'][name] = {'warmup_losses': losses, 'frozen_features_unchanged': True,
                                   'reload_predictions_match': True, 'full_backward_gradient_norm': full['max_gradient_norm']}
        del model, optimizer
        gc.collect()
        if name == 'mps':
            torch.mps.empty_cache()
        print(f'{name}: passed', flush=True)
    report['mps_available'] = 'mps' in devices
    write_json(args.output/'smoke_results.json', report)
    print(json.dumps(report, indent=2))

def make_optimizer_full(model):
    configure_training(model, 'full')
    return make_optimizer(model, 1e-5, 1e-4, .01)

if __name__ == '__main__':
    main()
