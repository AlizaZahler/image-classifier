# Image Dataset — 80/20 train/test split

Version 0.3, prepared 2026-10-21. Location: `/Users/alizazahler/Images Dataset`.

| Label | Train | Test | Total |
|---|---:|---:|---:|
| 2d | 460 | 115 | 575 |
| 3d | 395 | 99 | 494 |
| live | 240 | 60 | 300 |
| **Total** | **1095** | **274** | **1369** |


Training is 79.985% and testing is 20.015%; integer rounding accounts for the small difference from 80/20. There is no active validation split.

## Ready-to-use folders

```
train/
  2d/    # 460 images
  3d/    # 395 images
  live/  # 240 images
test/
  2d/    # 115 images
  3d/    # 99 images
  live/  # 60 images
```

These are hard links to the corresponding source images, so the split folders add negligible image storage. Editing the contents of a linked file also edits its source; treat image files as immutable and save transformed images separately. Excluded candidates are absent from these folders. The `images/` collection retains downloaded sources and excluded candidates, so use `train/` and `test/` for folder-based training rather than scanning all of `images/`.

`metadata.csv` and `splits/train.csv` / `splits/test.csv` contain source paths, labels and split assignments. `splits/train.txt` / `splits/test.txt` provide path lists. `class_to_idx.json` fixes the mapping 2d=0, 3d=1, live=2.

## Coverage and grouping

Every one of the 14 hand-drawn animation works and all three rendered films has at least two images in both train and test. The 30 doodle subjects also have at least two per split. Photorealistic indoor renders and real photographic candidates appear in both. Full counts: `splits/style_coverage.csv` and `splits/split_report.json`.

Detected exact/perceptual duplicate components stay in one split. Hypersim scenes are kept whole. Rendered movie frames are grouped in 30-second intervals. Consecutive singleton hand-drawn samples are paired within their source clip; detected duplicate components are kept whole. Seed 20261021 makes the result reproducible. The splitter balances class counts at 80/20 while providing style coverage.

This intentionally includes the same works/styles in both sets, as requested. Frame grouping reduces leakage but does not prove complete shot independence, and perceptual hashing cannot find every similar frame. This test measures performance on represented styles, rather than performance on wholly unseen productions. Keep the test set out of training and parameter selection.

## Browse and review

Open `review/train.html` or `review/test.html` to browse each split, `review/index.html` for the full active collection, and `review/new_additions.html` for the color/stylized expansion. Labels follow production method: hand-drawn 2d, 3D-model-rendered 3d even when it looks drawn, and physical camera capture live. Source labels and thumbnail screening remain provisional; original photos still require individual provenance/attribution review.

Review decisions are in `review/label_overrides.csv`, with rejected samples in `review/excluded.csv`. After changes run `python scripts/finalize_pilot.py` to audit and regenerate the current 80/20 split. To resplit only the active manifest, use `python scripts/split_train_test.py`. Changes to samples/review decisions can change membership; freeze a version before a training run.

## PyTorch

Install `scripts/requirements.txt` and install PyTorch separately for your hardware. Use `scripts/pytorch_dataset.py` with split `train` or `test`. Its default approved_only=True filters to individually approved samples; explicitly use approved_only=False for this source-labeled research dataset. The CSV loader composites PNG transparency onto white consistently with the gallery. Folder-based ImageFolder is also possible using the ready-to-use directories; ensure equivalent alpha handling in preprocessing.

## Sources, rights and previous versions

Sources: QuickDraw (human doodles), Anita (275 color/composited hand-drawn frames), Hypersim (300 indoor renders), Blender open movies (194 stylized animation frames), and COCO (300 photo candidates). Research and per-image terms remain in `sources/` and the manifest. There are 575 2d, 494 3d and 300 live images.

No material has been published. Image terms are separate from a future code license. Anita contains per-work CC BY-NC-SA 4.0 and CC BY terms; verify exact CC BY versions before redistribution. Big Buck Bunny and Sintel project terms are CC BY 3.0. The exact downloaded Caminandes archive has a redistribution hold pending license verification. COCO attribution checks also remain pending. See `sources/expansion_research.md` and per-image redistribution fields.

The previous train/validation/test manifests and scripts are backed up in `versions/v0.2/`; the original pilot is in `versions/v0.1/`. Old split-policy descriptions in those versions and research history are historical. Current assignments are exclusively train/test.

## PyTorch classifier

Training, evaluation and prediction are implemented in `image_classifier/`, with entry points `train.py`, `evaluate.py`, and `predict.py`. See [TRAINING.md](TRAINING.md) for setup, commands, experiment details and verification. The new loader explicitly includes current source-labeled images; it does not use the legacy approved-only filter.
