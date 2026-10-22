# Image Dataset — 80/20 train/test split

| Label | Train | Test | Total |
|---|---:|---:|---:|
| 2d | 460 | 115 | 575 |
| 3d | 395 | 99 | 494 |
| live | 240 | 60 | 300 |
| **Total** | **1095** | **274** | **1369** |


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
## Improvements
-find a better dataset with a larger range of 2D, 3D, and live images. 
-improve model architecture

Final goal is to see how the model reacts to AI-generated images and hopefully train it to detect the difference between hyper-realistic generated images and pictures taken on a camera. 

## Sources

Sources: QuickDraw (human doodles), Anita (275 color/composited hand-drawn frames), Hypersim (300 indoor renders), Blender open movies (194 stylized animation frames), and COCO (300 photo candidates). Research and per-image terms remain in `sources/` and the manifest. There are 575 2d, 494 3d and 300 live images.

No material has been published. Image terms are separate from a future code license. Anita contains per-work CC BY-NC-SA 4.0 and CC BY terms; verify exact CC BY versions before redistribution. Big Buck Bunny and Sintel project terms are CC BY 3.0. The exact downloaded Caminandes archive has a redistribution hold pending license verification. COCO attribution checks also remain pending. See `sources/expansion_research.md` and per-image redistribution fields.

The previous train/validation/test manifests and scripts are backed up in `versions/v0.2/`; the original pilot is in `versions/v0.1/`. Old split-policy descriptions in those versions and research history are historical. Current assignments are exclusively train/test.

## PyTorch classifier

Training, evaluation and prediction are implemented in `image_classifier/`, with entry points `train.py`, `evaluate.py`, and `predict.py`. See [TRAINING.md](TRAINING.md) for setup, commands, experiment details and verification. The new loader explicitly includes current source-labeled images; it does not use the legacy approved-only filter.
