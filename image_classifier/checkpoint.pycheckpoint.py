import os
import random
from pathlib import Path
import numpy as np
import torch
from .data import CLASSES, PREPROCESS
from .model import ARCHITECTURE, create_model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def rng_state():
    state = np.random.get_state()
    result = {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
              'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result['mps'] = torch.mps.get_rng_state()
    return result

def restore_rng(state):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])

def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp)
    os.replace(temp, path)

def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def model_metadata(config):
    return {'architecture': ARCHITECTURE, 'class_to_idx': CLASSES, 'preprocessing': PREPROCESS,
            'dropout': config['dropout']}

def load_model(path, device):
    state = load_checkpoint(path)
    if state['architecture'] != ARCHITECTURE or state['class_to_idx'] != CLASSES or state['preprocessing'] != PREPROCESS:
        raise ValueError('Checkpoint architecture, class mapping or preprocessing is incompatible')
    model = create_model(state['dropout'], pretrained=False)
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state
