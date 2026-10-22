import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights

CLASSES = {'2d': 0, '3d': 1, 'live': 2}
PREPROCESS = {'weights': 'IMAGENET1K_V1', 'crop_size': 224, 'resize_size': 236,
              'mean': [.485, .456, .406], 'std': [.229, .224, .225],
              'interpolation': 'bilinear', 'alpha_background': 'white', 'exif': True}

def load_rgb(path):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGBA')
            bg = Image.new('RGBA', im.size, 'white')
            bg.alpha_composite(im)
            return bg.convert('RGB')
    except Exception as e:
        raise ValueError(f'Cannot decode image {path}: {e}') from e

def read_manifest(root, split):
    with (Path(root) / 'splits' / f'{split}.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty {split} manifest')
    for row in rows:
        row['file'] = str(Path(split) / row['label'] / Path(row['path']).name)
    return rows

def fingerprint(root):
    h = hashlib.sha256()
    for name in ('train', 'test'):
        h.update((Path(root) / 'splits' / f'{name}.csv').read_bytes())
    return h.hexdigest()

def verify_dataset(root):
    import json
    root = Path(root)
    if json.loads((root / 'class_to_idx.json').read_text()) != CLASSES:
        raise ValueError('class_to_idx.json must equal 2d=0, 3d=1, live=2')
    splits = {s: read_manifest(root, s) for s in ('train', 'test')}
    seen_ids, seen_hashes, groups, hash_groups = {}, {}, {}, {}
    for split, rows in splits.items():
        expected = set()
        for r in rows:
            p = root / r['file']
            if r['label'] not in CLASSES or r['split'] != split or Path(r['path']).parent.name != r['label']:
                raise ValueError(f'Inconsistent folder/manifest label: {p}')
            if not r['split_group']:
                raise ValueError(f'Missing split_group: {p}')
            if r['image_id'] in seen_ids:
                raise ValueError(f'Duplicate image ID: {p}; {seen_ids[r["image_id"]]}')
            seen_ids[r['image_id']] = str(p)
            try:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                original = hashlib.sha256((root / r['path']).read_bytes()).hexdigest()
            except OSError as e:
                raise ValueError(f'Missing/unreadable image {p}: {e}') from e
            if digest != r['sha256'] or original != digest:
                raise ValueError(f'Image checksum mismatch: {p}')
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(f'Exact duplicate crosses outer split: {p}')
            seen_hashes[digest] = split
            g = r['split_group']
            if digest in hash_groups and hash_groups[digest] != g:
                raise ValueError(f'Exact duplicate is assigned inconsistent split_group units: {p}')
            hash_groups[digest] = g
            if g in groups and groups[g] != split:
                raise ValueError(f'Group {g} crosses outer split: {p}')
            groups[g] = split
            load_rgb(p)
            expected.add(p.resolve())
        actual = {p.resolve() for label in CLASSES for p in (root / split / label).iterdir()
                  if p.is_file() and not p.name.startswith('.')}
        if actual != expected:
            raise ValueError(f'{split} folder differs from manifest: {actual ^ expected}')
    return splits

def validation_split(rows, fraction=.2, seed=20261021):
    """Subset-sum allocation within homogeneous label/subtype groups.

    Mixed groups are kept intact in fitting; never split duplicate components.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r['split_group']].append(r)
    strata = defaultdict(list)
    for g, members in groups.items():
        keys = {(r['label'], r['subtype']) for r in members}
        if len(keys) == 1:
            strata[next(iter(keys))].append(g)
    rng = random.Random(seed)
    selected = set()
    for key, gs in sorted(strata.items()):
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        target = round(total * fraction)
        reachable = {0: ()}
        for g in gs:
            n = len(groups[g])
            for count, chosen in list(reachable.items()):
                if count + n < total and count + n not in reachable:
                    reachable[count + n] = chosen + (g,)
        viable = [n for n in reachable if 0 < n < total]
        if not viable:
            raise ValueError(f'Cannot reserve validation for subtype {key}; needs independent groups')
        best = min(viable, key=lambda n: (abs(n-target), n))
        selected.update(reachable[best])
    fit = [r for r in rows if r['split_group'] not in selected]
    val = [r for r in rows if r['split_group'] in selected]
    for subset in (fit, val):
        if set(r['label'] for r in subset) != set(CLASSES):
            raise ValueError('Internal split does not contain every class')
    assert not {r['split_group'] for r in fit} & {r['split_group'] for r in val}
    return fit, val

def transforms_for(training=False):
    if not training:
        return ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.8, 1.)),
        transforms.RandomHorizontalFlip(.5),
        transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1),
        transforms.ToTensor(), transforms.Normalize(PREPROCESS['mean'], PREPROCESS['std'])])

class ImageDataset(Dataset):
    def __init__(self, root, rows, training=False):
        self.root, self.rows = Path(root), rows
        self.transform = transforms_for(training)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(load_rgb(self.root / row['file'])), CLASSES[row['label']]

def counts(rows):
    return {'total': len(rows), 'classes': dict(Counter(r['label'] for r in rows)),
            'subtypes': dict(Counter(r['subtype'] for r in rows)),
            'groups': len({r['split_group'] for r in rows})}
