import contextlib
import time
import torch
from torch.nn import functional as F
from .metrics import classification_metrics
from .model import configure_training

def autocast(device):
    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()

def train_epoch(model, loader, optimizer, device, class_weights, mode='full', accumulation=8,
                smoothing=.05, scaler=None, configure=True):
    """Mean of class-weighted per-image losses; normalize each actual window.

    reduction='sum' / actual image count is intentional: this makes uneven
    microbatches equivalent to one full batch, including label smoothing.
    """
    started = time.monotonic()
    if configure:
        configure_training(model, mode)
    else:
        model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, samples, window = 0., 0, 0
    targets, predictions = [], []
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norms = []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with autocast(device):
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=smoothing, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite training loss at batch {i}')
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        n = y.numel()
        window += n
        samples += n
        total_loss += float(loss.detach())
        targets.extend(y.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
        if (i+1) % accumulation == 0 or i+1 == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            for p in params:
                if p.grad is not None:
                    p.grad.div_(window)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            grad_norms.append(float(norm))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window = 0
    if not samples:
        raise ValueError('Empty training loader')
    result = classification_metrics(targets, predictions)
    result.update(loss=total_loss/samples, elapsed_seconds=time.monotonic()-started,
                  max_gradient_norm=max(grad_norms))
    return result

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    targets, predictions, scores = [], [], []
    total_loss = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction='sum'))
        targets.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        scores.extend(logits.softmax(1).cpu().tolist())
    if not targets:
        raise ValueError('Empty evaluation loader')
    result = classification_metrics(targets, predictions)
    result['loss'] = total_loss / len(targets)
    return result, {'targets': targets, 'predictions': predictions, 'scores': scores}
