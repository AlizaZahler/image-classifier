import gc
import importlib.metadata
import json
import math
import platform
import shutil
import time
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .checkpoint import atomic_save, load_checkpoint, model_metadata, restore_rng, rng_state, seed_all
from .data import CLASSES, ImageDataset, counts, fingerprint, validation_split, verify_dataset
from .engine import evaluate, train_epoch
from .model import configure_training, create_model, make_optimizer, resolve_device
from .reports import plots, write_json


def make_loader(root, rows, config, training, batch_size=None):
    return DataLoader(ImageDataset(root, rows, training), batch_size=batch_size or config['batch_size'],
                      shuffle=training, num_workers=config['workers'], drop_last=False)

def class_weights(rows, device):
    counter = Counter(r['label'] for r in rows)
    if any(counter[label] == 0 for label in CLASSES):
        raise ValueError('Training data must include all three classes')
    return torch.tensor([len(rows)/(3*counter[label]) for label in CLASSES], device=device)

def better(metrics, previous):
    return previous is None or (metrics['macro_f1'], -metrics['loss']) > (previous['macro_f1'], -previous['loss'])

def is_oom(error):
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or 'out of memory' in message

def run_stage(root, run, name, config, fit, val, mode, feature_lr, head_lr, epochs,
              initial=None, early_stop=False):
    """Resume at epoch boundaries; a failed epoch is replayed from last.pt."""
    stage = run / name
    stage.mkdir(exist_ok=True)
    last, best_path = stage / 'last.pt', stage / 'best.pt'
    done = stage / 'completed.json'
    if done.exists():
        return json.loads(done.read_text())
    device = resolve_device(config['device'])
    seed_all(config['seed'])
    model = create_model(config['dropout'], pretrained=not (last.exists() or initial))
    if initial and not last.exists():
        model.load_state_dict(load_checkpoint(initial)['model'])
    model.to(device)
    configure_training(model, mode)
    optimizer = make_optimizer(model, feature_lr, head_lr, config['weight_decay'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
                 if mode != 'head' else None)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    weights = class_weights(fit, device)
    best, best_epoch, significant, stale, epoch = None, 0, -1., 0, 0
    batch, accumulation = config['batch_size'], config['accumulation']
    effective_batch = batch * accumulation
    history, adjustments = [], []

    def snapshot():
        return {**model_metadata(config), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict() if scaler else None, 'rng': rng_state(),
                'stage': name, 'mode': mode, 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
                'significant': significant, 'stale': stale, 'batch_size': batch,
                'accumulation': accumulation, 'history': history, 'adjustments': adjustments,
                'manifest_fingerprint': fingerprint(root), 'config': config}

    def restore(state):
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        if scaler:
            scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])

    if last.exists():
        state = load_checkpoint(last)
        if state['manifest_fingerprint'] != fingerprint(root) or state['config'] != config:
            raise ValueError('Cannot resume with changed data or configuration')
        restore(state)
        epoch, best, best_epoch = state['epoch'], state['best'], state['best_epoch']
        significant, stale = state['significant'], state['stale']
        batch, accumulation = state['batch_size'], state['accumulation']
        history, adjustments = state['history'], state['adjustments']
        del state
        if best_epoch == epoch and best is not None:
            shutil.copy2(last, best_path)
    else:
        atomic_save(last, snapshot())
    while epoch < epochs and not (early_stop and stale >= config['patience']):
        try:
            train_loader = make_loader(root, fit, config, True, batch)
            lrs = [group['lr'] for group in optimizer.param_groups]
            train_metrics = train_epoch(model, train_loader, optimizer, device, weights, mode,
                                        accumulation, config['label_smoothing'], scaler)
            val_metrics = evaluate(model, make_loader(root, val, config, False, batch), device)[0] if val else None
        except RuntimeError as error:
            if not is_oom(error) or batch <= 1:
                raise
            # Release failed forward/backward tensors before reloading epoch state.
            error.__traceback__ = None
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            state = load_checkpoint(last)
            restore(state)
            del state
            previous = batch
            batch = max(1, batch // 2)
            accumulation = math.ceil(effective_batch / batch)
            adjustments.append({'epoch': epoch+1, 'old_batch': previous, 'batch_size': batch,
                                'accumulation': accumulation, 'reason': 'out_of_memory'})
            atomic_save(last, snapshot())
            print(f'{name}: OOM; replaying epoch {epoch+1}, batch={batch}, accumulation={accumulation}', flush=True)
            continue
        epoch += 1
        if scheduler:
            scheduler.step()
        history.append({'stage': name, 'epoch': epoch, 'train': train_metrics,
                        'validation': val_metrics, 'learning_rates': lrs})
        new_best = val_metrics is not None and better(val_metrics, best)
        if new_best:
            best, best_epoch = val_metrics, epoch
        if val_metrics:
            if val_metrics['macro_f1'] > significant + config['min_delta']:
                significant, stale = val_metrics['macro_f1'], 0
            else:
                stale += 1
        atomic_save(last, snapshot())
        if new_best:
            # last.pt is the transaction commit; recover a missing copy on resume.
            shutil.copy2(last, best_path)
        write_json(stage / 'history.json', history)
        print(f'{name} epoch {epoch}/{epochs}: loss={train_metrics["loss"]:.4f}' +
              (f' val_macro_f1={val_metrics["macro_f1"]:.4f}' if val_metrics else ''), flush=True)
    write_json(stage / 'history.json', history)
    if best_epoch == epoch and val:
        shutil.copy2(last, best_path)
    # Baseline is the same terminal warm-up checkpoint used by all trials.
    if mode == 'head' or not val:
        shutil.copy2(last, best_path)
        best_epoch = epoch
        best = history[-1]['validation'] if val else None
    result = {'stage': name, 'mode': mode, 'best_epoch': best_epoch, 'metrics': best,
              'checkpoint': str(best_path), 'terminal_checkpoint': str(last),
              'feature_lr': feature_lr, 'head_lr': head_lr}
    write_json(done, result)
    del model, optimizer, scheduler, weights, scaler
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return result

def tune_and_refit(root, run, config, resume=False):
    root, run = Path(root).resolve(), Path(run).resolve()
    if run.exists() and not resume:
        raise ValueError(f'Run directory already exists: {run}; use --resume or another --output')
    run.mkdir(parents=True, exist_ok=True)
    if resume:
        saved = json.loads((run / 'config.json').read_text())
        if saved != config:
            raise ValueError('Resume uses the original saved configuration')
    else:
        write_json(run / 'config.json', config)
    data = verify_dataset(root)
    fit, val = validation_split(data['train'], config['validation_fraction'], config['seed'])
    split_info = {'manifest_fingerprint': fingerprint(root),
                  'fit_ids': [r['image_id'] for r in fit], 'validation_ids': [r['image_id'] for r in val],
                  'fit': counts(fit), 'validation': counts(val), 'test': counts(data['test'])}
    if resume and json.loads((run / 'internal_split.json').read_text()) != split_info:
        raise ValueError('Saved split differs from current data')
    write_json(run / 'internal_split.json', split_info)
    if not resume:
        write_json(run / 'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
             'device': str(resolve_device(config['device'])),
             'dependencies': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'Pillow', 'numpy', 'matplotlib')}})
    baseline = run_stage(root, run, 'warmup', config, fit, val, 'head', 0., config['warmup_lr'], config['warmup_epochs'])
    candidates = [baseline]
    for name, trial in config['trials'].items():
        candidates.append(run_stage(root, run, f'trial_{name}', config, fit, val,
            trial['mode'], trial['feature_lr'], trial['head_lr'], config['epochs'],
            initial=baseline['terminal_checkpoint'], early_stop=True))
    rank = {'head': 0, 'last': 1, 'full': 2}
    winner = min(candidates, key=lambda r: (-r['metrics']['macro_f1'], r['metrics']['loss'], rank[r['mode']], r['stage']))
    write_json(run / 'trial_comparison.json', {'candidates': candidates, 'winner': winner})
    shutil.copy2(winner['checkpoint'], run / 'best.pt')
    refit_warmup = run_stage(root, run, 'refit_warmup', config, data['train'], None, 'head', 0.,
                             config['warmup_lr'], config['warmup_epochs'])
    final = refit_warmup
    if winner['mode'] != 'head':
        final = run_stage(root, run, 'refit', config, data['train'], None, winner['mode'],
                          winner['feature_lr'], winner['head_lr'], winner['best_epoch'],
                          initial=refit_warmup['terminal_checkpoint'])
    state = load_checkpoint(final['terminal_checkpoint'])
    artifact = {key: state[key] for key in ('architecture', 'class_to_idx', 'preprocessing', 'dropout', 'model', 'manifest_fingerprint', 'config')}
    artifact.update(selection=winner, training_count=len(data['train']), status='refitted')
    atomic_save(run / 'final.pt', artifact)
    history = []
    for path in sorted(run.glob('*/history.json')):
        history.extend(json.loads(path.read_text()))
    write_json(run / 'history.json', history)
    plots(history, run)
    print(f'Final model: {run / "final.pt"}', flush=True)
    return run / 'final.pt'
