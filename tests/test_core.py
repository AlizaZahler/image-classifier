import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from image_classifier.checkpoint import atomic_save, load_checkpoint, restore_rng, rng_state, seed_all
from image_classifier.data import CLASSES, validation_split
from image_classifier.engine import train_epoch, evaluate
from image_classifier.metrics import classification_metrics


def test_accumulation_incomplete_window():
    seed_all(42)
    x = torch.randn(11, 5)
    y = torch.tensor([0, 0, 1, 2, 2, 1, 0, 1, 2, 1, 0])
    model = nn.Linear(5, 3)
    other = copy.deepcopy(model)
    # Windows of six and five images; last microbatch contains one image.
    weights = torch.tensor([.8, 1.1, 1.4])
    for m, batch, accumulation in ((model, 2, 3), (other, 6, 1)):
        opt = torch.optim.SGD(m.parameters(), lr=.05)
        train_epoch(m, DataLoader(TensorDataset(x, y), batch_size=batch), opt,
                    torch.device('cpu'), weights, accumulation=accumulation, configure=False)
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


def test_small_training_reduces_loss():
    seed_all(4)
    x = torch.eye(3).repeat(8, 1)
    y = torch.arange(3).repeat(8)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    m = nn.Linear(3, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=.1)
    losses = [train_epoch(m, loader, opt, torch.device('cpu'), torch.ones(3), accumulation=2,
                          configure=False)['loss'] for _ in range(6)]
    assert np.isfinite(losses).all() and losses[-1] < losses[0]


def test_checkpoint_restores_optimizer_scheduler_rng(tmp_path):
    seed_all(9)
    model = nn.Linear(3, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
    loader = DataLoader(TensorDataset(torch.eye(3), torch.arange(3)), batch_size=2, shuffle=True)
    train_epoch(model, loader, opt, torch.device('cpu'), torch.ones(3), configure=False)
    scheduler.step()
    path = tmp_path / 'last.pt'
    atomic_save(path, {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                      'scheduler': scheduler.state_dict(), 'epoch': 1, 'rng': rng_state()})
    expected_random = (random.random(), np.random.rand(), torch.rand(3))
    train_epoch(model, loader, opt, torch.device('cpu'), torch.ones(3), configure=False)
    scheduler.step()
    expected = copy.deepcopy(model.state_dict())
    saved = load_checkpoint(path)
    model.load_state_dict(saved['model'])
    opt.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'])
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(3), expected_random[2])
    train_epoch(model, loader, opt, torch.device('cpu'), torch.ones(3), configure=False)
    scheduler.step()
    assert saved['epoch'] == 1 and scheduler.last_epoch == 2
    for k in expected:
        torch.testing.assert_close(expected[k], model.state_dict()[k], atol=0, rtol=0)


def test_group_split():
    rows = [{'image_id': f'{c}-{i}-{j}', 'label': c, 'subtype': c+'style', 'split_group': f'{c}-{i}'}
            for c in CLASSES for i in range(10) for j in range(2)]
    fit, val = validation_split(rows)
    assert len(val) == 12
    assert not {r['split_group'] for r in fit} & {r['split_group'] for r in val}
    assert validation_split(rows) == (fit, val)


def test_metrics_and_eval_exact_count():
    m = nn.Identity()
    labels = torch.arange(274) % 3
    x = torch.nn.functional.one_hot(labels, 3).float()*5
    metrics, first = evaluate(m, DataLoader(TensorDataset(x, labels), batch_size=4), torch.device('cpu'))
    _, second = evaluate(m, DataLoader(TensorDataset(x, labels), batch_size=7), torch.device('cpu'))
    assert metrics['count'] == 274 and metrics['macro_f1'] == 1.
    assert first == second
    assert classification_metrics([0, 1, 2], [0, 0, 0])['balanced_accuracy'] == 1/3
