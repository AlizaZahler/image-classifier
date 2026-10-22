# Image classifier

This implementation uses pretrained TorchVision ConvNeXt Tiny to classify production style as **2d**, **3d**, or **live**. The source-based labels include provisional labels; model quality depends on their accuracy. No accuracy target is assumed.

The final command runs the complete experiment and can take a substantial time on an M2 Air. It prints progress after each epoch. It first validates the data, tunes using training data only, refits on all training images, and only then evaluates the test set. Omit `--evaluate` to run evaluation separately:

```bash
python evaluate.py --checkpoint runs/<run>/final.pt
python predict.py --checkpoint runs/<run>/final.pt --input /path/to/image-or-folder
python train.py --resume runs/<run> --evaluate
```

`--resume` restores the run's original configuration. An interrupted epoch restarts from its beginning; completed stages are reused. Each stage's `last.pt` stores model, optimizer, scheduler, scaler where applicable, epoch, Python/NumPy/Torch/device random states, metrics, early-stopping state and memory adjustments. Do not move or change the dataset while resuming. Completed run stage pointers currently contain absolute paths; keep the run at its original location when resuming. `final.pt` is portable for prediction.

To install on another machine with Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` records the exact tested local environment. CUDA systems should use the appropriate official PyTorch installation command for their CUDA version. Local defaults use MPS float32, with CPU fallback if MPS is unavailable. CUDA alone uses autocast and a gradient scaler.

## Configuration

Edit `config.json`, or use command-line overrides:

```bash
python train.py --tune --device mps --batch-size 2 --epochs 20 --seed 20261021
python train.py --tune --device cpu --output runs/cpu-experiment
```

Overrides include `--accumulation`, `--warmup-epochs`, `--warmup-lr`, `--feature-lr`, and `--head-lr`. Feature/head LR overrides affect all fine-tuning trials; edit the configuration to change individual trials. Specifying a batch size of 1, 2, 4, 8, 16 or 32 automatically adjusts accumulation to effective batch size 32, unless explicitly overridden.

Defaults: 3 classifier warm-up epochs, maximum 20 epochs per fine-tuning trial, AdamW, class-weighted cross-entropy with 0.05 smoothing, physical batch 4 / accumulation 8, weight decay 0.01 excluding biases and normalization vectors, gradient norm clipping at 1, patience 5 and meaningful macro-F1 improvement threshold 0.001. No balancing sampler is used.

A trains the final feature stage **and its preceding downsampling layer** at 3e-5, with classifier at 3e-4. B trains the full network at 1e-5 / 1e-4. C trains it at 3e-5 / 3e-4. The classifier includes its existing normalization plus dropout 0.2 and a new three-output linear layer. Frozen features remain in evaluation mode.

Every trial starts from the same terminal warm-up checkpoint. That checkpoint is also the classifier-only baseline. The best validation macro-F1 wins; exact ties use lower unweighted cross-entropy, then classifier-only / partial / full complexity. The 0.001 threshold governs early stopping, not checkpoint ranking. Refit repeats warm-up from pretrained weights and runs the selected fine-tuning epoch count. Its cosine schedule retains the original 20-epoch horizon (or the configured trial limit), even when refit ends earlier.

The weighted training objective is the mean of class-weighted **per-image** smoothed losses. Accumulation sums these losses, divides gradients by the actual images in the window and then clips. This definition intentionally differs from PyTorch's default weighted mean denominator, and makes incomplete microbatches equivalent to a single larger batch. Weights are recomputed from fitting data during tuning and all training data during refit.

On a recognized training/validation OOM, batch size halves, accumulation rises to preserve the effective batch size, and the failed epoch is replayed from its saved state. Failure at batch size 1 is reported. This recovery cannot catch an operating-system process kill or an allocation failure before the first epoch checkpoint; resume with the existing run for an interrupted epoch, or start a new run at a smaller batch size for startup allocation failures.

## Data and split protection

The original outer train/test split is unchanged. Verification decodes every active file, verifies SHA256 against the manifest and original, checks folder labels, rejects extra files, repeated IDs, cross-split exact duplicates and overlapping `split_group` units. No `approved_only` filter is applied: current source-labeled rows are explicitly included.

The internal validation split uses deterministic subset-sum selection within label/subtype strata, preserving entire `split_group` units. Mixed-stratum components stay in fitting; if a subtype cannot supply independent groups, splitting fails clearly. Saved IDs and manifest fingerprints prevent accidental changed-data resume. Count differences from precisely 20% are recorded. Existing source works are allowed on both sides, as in the outer dataset design.

Images receive EXIF correction, white alpha compositing and RGB conversion. Training uses a mild crop, horizontal flip and color jitter; evaluation and prediction use `IMAGENET1K_V1.transforms()` (resize 236, center crop 224 and ImageNet normalization). Preprocessing and class mapping are checked when reloading a checkpoint.

## Outputs

Each `runs/<run>/` contains configuration, dependency/device details, manifest fingerprint and internal IDs, per-stage epoch histories and resumable checkpoints, `trial_comparison.json`, selected tuning `best.pt`, the refitted `final.pt`, combined history and training curves. `best.pt` is a validation-selected model; use `final.pt` for test evaluation.

Evaluation saves accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, grouped source/subtype/work metrics with counts, per-image predictions and an HTML mistake gallery. Group macro-F1 always includes all three classes, including absent labels; inspect counts and per-class recall when interpreting single-class sources. Softmax scores are uncalibrated model scores, not measured probabilities of correctness.

The test set represents styles from sources/productions also seen in training. Its metrics do **not** establish generalization to entirely new productions. Do not use test results to select trials or training duration. The evaluator requires a final refitted checkpoint.

## Verification

```bash
python -m pytest -q
python smoke_test.py
```

## Official references

- [ConvNeXt Tiny and weight preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)
- [Transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)
- [AMP and gradient accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
