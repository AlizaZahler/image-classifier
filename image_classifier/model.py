import torch
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

ARCHITECTURE = 'convnext_tiny'

def create_model(dropout=.2, pretrained=True):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    n = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, 3))
    return model

def configure_training(model, mode):
    if mode not in ('head', 'last', 'full'):
        raise ValueError(f'Unknown mode: {mode}')
    model.requires_grad_(mode == 'full')
    model.classifier.requires_grad_(True)
    if mode == 'last':
        # Include the downsampling transition into the final stage.
        model.features[6:].requires_grad_(True)
    model.train()
    if mode != 'full':
        model.features.eval()
        if mode == 'last':
            for module in list(model.features.children())[6:]:
                module.train()

def make_optimizer(model, feature_lr, head_lr, weight_decay):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = head_lr if name.startswith('classifier.') else feature_lr
        decay = 0. if p.ndim <= 1 or name.endswith('.bias') else weight_decay
        groups.setdefault((lr, decay), []).append(p)
    return torch.optim.AdamW([{'params': ps, 'lr': lr, 'weight_decay': wd}
                             for (lr, wd), ps in groups.items()])

def resolve_device(request='auto'):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(request)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device.type not in ('cpu', 'mps', 'cuda'):
        raise ValueError(f'Unsupported device: {device}')
    return device
