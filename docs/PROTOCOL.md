# AnyViewBench evaluation protocol

This page gives the details behind `scripts/eval_avb.py`. The README shows how to run it.

## Resolution

Frames are resized once, directly from the native frame, to the model's 576 grid, and the
intrinsics are scaled in memory to match. Nothing on disk changes. The grid follows the rule
the checkpoint was trained and evaluated with. The training data was stored at long side 384
with both sides snapped to a multiple of 16 (ties round down) and then resized to long side
576 with the same snapping, so the grid is computed in those two stages (`snap_shape` in
`scripts/infer.py`). For 16:9 inputs this gives 576x304 (a single nearest-16 snap would give
576x320). The difference matters: on DROID the 576x320 grid scores about
0.5 dB PSNR lower with this checkpoint. 4:3 inputs give 576x432 either way. Metrics are
computed at that grid: the ground truth is resized to it with the same Lanczos filter.

## Paper pixel chain

The evaluation used for the paper read frames stored at long side 384 (JPEG) and upsampled
them to 576 with the same rule, for both the input and the ground truth. This blurs them
slightly. `--legacy-bake 384` reproduces that chain from the native frames (Lanczos to the
384 grid, JPEG quality 100, Lanczos to the 576 grid). It reproduces the paper within 0.04 dB
PSNR on every zero-shot dataset. The benchmark protocol resizes the native frames directly.

## Noise and seeds

Sampling noise is deterministic: every episode uses the same noise draw (`--seed`, default 0),
the convention used for the paper and for the reference table. The documented command on
`droid_OOD_LR` (32 episodes) gives 11.15 / 0.333 / 0.642, and 11.27 / 0.336 / 0.633 with
`--legacy-bake 384`.

## Metrics

Metrics are computed per frame at the 576 grid on float images in [0, 1], then averaged over
the frames of an episode, over the episodes of a split, over the splits of a dataset group,
and over the groups (all unweighted means):

- PSNR with data range 1.
- SSIM as in scikit-image's defaults: a 7x7 uniform window (radius 3), K1 = 0.01, K2 = 0.03,
  sample covariance, computed per channel and averaged (the 11x11 Gaussian variant gives
  higher values).
- LPIPS with the VGG backbone of the `lpips` package, inputs mapped to [-1, 1].

Frames whose ground truth is entirely black are excluded from the averages.

## Restricted splits

The Ego-Exo4D and Waymo splits do not include pixels. `scripts/eval_avb.py` skips them with a
notice unless they have been rebuilt from the official downloads (see
[RESTRICTED_DATA.md](RESTRICTED_DATA.md)). When nothing is left to evaluate the script exits
with status 1 and writes no outputs.
