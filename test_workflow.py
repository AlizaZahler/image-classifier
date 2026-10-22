"""Fast integration: exercise the orchestration with a tiny ConvNeXt-shaped net."""
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from image_classifier import workflow as w
from image_classifier.checkpoint import load_checkpoint

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*[nn.Linear(3, 3) for _ in range(8)])
        self.classifier = nn.Sequential(nn.Linear(3, 3))
    def forward(self, x):
        return self.classifier(self.features(x))


def setup(monkeypatch, tmp_path):
    config = json.loads((Path(__file__).parents[1] / 'config.json').read_text())
    config.update(device='cpu', epochs=2, warmup_epochs=1, batch_size=4, accumulation=2)
    rows = [{'image_id': f'{label}{i}', 'label': label, 'subtype': label, 'split_group': f'{label}{i}'}
            for label in ('2d','3d','live') for i in range(10)]
    monkeypatch.setattr(w, 'verify_dataset', lambda root: {'train': rows, 'test': rows[:3]})
    monkeypatch.setattr(w, 'fingerprint', lambda root: 'fixture')
    monkeypatch.setattr(w, 'create_model', lambda *a, **k: Tiny())
    def loader(root, rows, config, training, batch_size=None):
        labels = torch.tensor([{'2d':0, '3d':1, 'live':2}[r['label']] for r in rows])
        return DataLoader(TensorDataset(torch.nn.functional.one_hot(labels,3).float(), labels),
                          batch_size=batch_size or 4, shuffle=training)
    monkeypatch.setattr(w, 'make_loader', loader)
    return config, rows


def test_complete_workflow_and_resume(monkeypatch, tmp_path):
    config, _ = setup(monkeypatch, tmp_path)
    run = tmp_path / 'run'
    final = w.tune_and_refit(tmp_path, run, config)
    saved = load_checkpoint(final)
    assert saved['status'] == 'refitted' and saved['training_count'] == 30
    assert (run/'trial_comparison.json').exists() and (run/'training_curves.png').exists()
    before = {k:v.clone() for k,v in saved['model'].items()}
    w.tune_and_refit(tmp_path, run, config, resume=True)
    after = load_checkpoint(final)['model']
    for k in before:
        assert torch.equal(before[k], after[k])


def test_oom_replays_epoch_and_reduces_batch(monkeypatch, tmp_path):
    config, rows = setup(monkeypatch, tmp_path)
    original = w.train_epoch
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(args[1].batch_size)
        if len(calls) == 1:
            # Simulate an OOM after some parameters and RNG have changed.
            original(*args, **kwargs)
            raise RuntimeError('MPS backend out of memory')
        return original(*args, **kwargs)
    monkeypatch.setattr(w, 'train_epoch', fail_once)
    run = tmp_path / 'oom'
    run.mkdir()
    w.run_stage(tmp_path, run, 'warmup', config, rows, None, 'head', 0., .001, 1)
    state = load_checkpoint(run/'warmup/last.pt')
    assert calls == [4, 2] and state['accumulation'] == 4 and state['epoch'] == 1
    assert state['adjustments'][0]['reason'] == 'out_of_memory'


def test_interrupted_stage_resume_matches_uninterrupted(monkeypatch, tmp_path):
    config, rows = setup(monkeypatch, tmp_path)
    run1, run2 = tmp_path/'reference', tmp_path/'interrupted'
    run1.mkdir(); run2.mkdir()
    w.run_stage(tmp_path, run1, 'trial', config, rows, rows, 'full', 1e-4, 1e-3, 2)
    original = w.train_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    monkeypatch.setattr(w, 'train_epoch', interrupt)
    with pytest.raises(KeyboardInterrupt):
        w.run_stage(tmp_path, run2, 'trial', config, rows, rows, 'full', 1e-4, 1e-3, 2)
    monkeypatch.setattr(w, 'train_epoch', original)
    w.run_stage(tmp_path, run2, 'trial', config, rows, rows, 'full', 1e-4, 1e-3, 2)
    a, b = [load_checkpoint(r/'trial/last.pt') for r in (run1, run2)]
    assert a['epoch'] == b['epoch'] == 2
    assert a['scheduler'] == b['scheduler']
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k])
