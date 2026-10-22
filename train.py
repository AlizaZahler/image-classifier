#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from image_classifier.data import counts, validation_split, verify_dataset
from image_classifier.reports import write_json
from image_classifier.workflow import tune_and_refit

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description='Tune on internal validation, then refit on all outer training images.')
    p.add_argument('--tune', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--resume', type=Path, help='Run directory; resumes its original configuration')
    p.add_argument('--output', type=Path)
    p.add_argument('--device')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--accumulation', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--warmup-epochs', type=int)
    p.add_argument('--warmup-lr', type=float)
    p.add_argument('--feature-lr', type=float, help='Override feature LR for all three trials')
    p.add_argument('--head-lr', type=float, help='Override classifier LR for all three trials')
    p.add_argument('--seed', type=int)
    p.add_argument('--evaluate', action='store_true', help='Evaluate the final refitted model after selection is complete')
    args = p.parse_args()
    config = json.loads((args.resume / 'config.json' if args.resume else args.config).read_text())
    override_keys = ('device', 'batch_size', 'accumulation', 'epochs', 'warmup_epochs', 'warmup_lr', 'seed')
    if args.resume and any(getattr(args, k) is not None for k in (*override_keys, 'feature_lr', 'head_lr')):
        p.error('Resume cannot change configuration; start a new run for overrides')
    for key in override_keys:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.batch_size and args.accumulation is None:
        if 32 % args.batch_size:
            p.error('--batch-size must divide 32, or set --accumulation explicitly')
        config['accumulation'] = 32 // args.batch_size
    for trial in config['trials'].values():
        for key in ('feature_lr', 'head_lr'):
            if getattr(args, key) is not None:
                trial[key] = getattr(args, key)
    for key in ('batch_size', 'accumulation', 'epochs', 'warmup_epochs', 'patience'):
        if config[key] < 1:
            p.error(f'{key} must be positive')
    if not 0 < config['validation_fraction'] < 1:
        p.error('validation_fraction must be between zero and one')
    if args.verify_only:
        data = verify_dataset(args.root)
        fit, val = validation_split(data['train'], config['validation_fraction'], config['seed'])
        report = {s: counts(rows) for s, rows in {**data, 'fit': fit, 'validation': val}.items()}
        print(json.dumps(report, indent=2))
        if args.output:
            write_json(args.output / 'verification.json', report)
        return
    if not args.tune and not args.resume:
        p.error('Choose --tune, --resume RUN, or --verify-only')
    run = args.resume or args.output or args.root / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')
    final = tune_and_refit(args.root, run, config, bool(args.resume))
    if args.evaluate:
        from evaluate import evaluate_checkpoint
        evaluate_checkpoint(args.root, final, run / 'evaluation', config['device'], config['batch_size'])

if __name__ == '__main__':
    main()
