# Restricted AnyViewBench splits (Ego-Exo4D, Waymo)

Two source datasets do not allow us to redistribute their pixels: Ego-Exo4D (its license says
the data may not appear in another dataset or product) and the Waymo Open Dataset (only
users registered at waymo.com/open may receive it). The AnyViewBench splits built from them
come without image frames. Each affected episode contains:

- `metadata.json` and `lowdim/`: intrinsics, per-frame poses, input/target camera roles (as
  in every other episode);
- `fetch_manifest.json` (version 2): the source take or segment, the source camera of each
  view, the original frame index and timestamp of every frame, the exact processing chain, and
  the SHA-256 of every reference frame.

`scripts/prepare_restricted.py` rebuilds `rgb/cam0` (target view) and `rgb/cam1` (input view)
from your own official download and checks every frame against the manifest. We tested both
procedures against the official data: the rebuilt frames are byte-identical to our reference
frames, so evaluation results are identical by construction.

## Ego-Exo4D

Which data: only the `downscaled_takes/448` part of the referenced takes (the 796x448
frame-aligned videos, about 3 GB for all 128 episodes; between 2 MB and 140 MB per episode).
Every view is an exo GoPro camera; no Aria views are used, so no rotation is involved.

1. Register at https://ego4d-data.org/ and install the `egoexo` downloader. Download the
   `downscaled_takes/448` part of the takes listed in the manifests (the take uids are in
   each `fetch_manifest.json` under `identity.source`), for example
   `egoexo -o /path/to/EgoExo4D --parts downscaled_takes/448 --uids <uid> ...` (check the
   flag names against the current Ego-Exo4D documentation).
2. Run the extractor. It needs an `ffmpeg` binary; the `imageio-ffmpeg` wheel from the
   requirements provides one:

       python scripts/prepare_restricted.py --avb-root data/AnyViewBench_zeroshot \
           --dataset egoexo4d --source-root /path/to/EgoExo4D

   Processing chain: decode `takes/<take>/frame_aligned_videos/downscaled/448/<cam>.mp4`
   with ffmpeg, write frame n of the take as MJPEG at ffmpeg quality 2 (this is how the
   reference frames were produced). Only the JPEG comment marker depends on the ffmpeg
   version; the extractor rewrites it to the reference value, which makes the files
   byte-identical (verified on four episodes from four takes and splits, 82 of 82 frames
   each). Runtime is 0.3 to 35 s per episode on a CPU.

The same procedure applies to the in-distribution Ego-Exo4D splits in `AnyViewBench_indist`.

## Waymo Open Dataset

Which data: the Perception dataset v1 validation tfrecords named in the manifests (one
segment per episode, roughly 1 GB each; about 55 to 85 GB for all 64 episodes).

1. Register at https://waymo.com/open and download the referenced validation segments
   (`identity.source.tfrecord` in each `fetch_manifest.json`) into one directory.
2. Run the extractor (no extra packages: it reads the tfrecords with its own minimal
   TFRecord and protobuf parser):

       python scripts/prepare_restricted.py --avb-root data/AnyViewBench_indist \
           --dataset waymo --source-root /path/to/waymo/validation

   Processing chain: take the JPEG bytes of the named camera in record n of the segment,
   decode them with libjpeg defaults, and re-encode with Pillow at quality 95 (4:2:0, baseline,
   standard Huffman tables); no resize or crop. This reproduces the reference frames
   byte-identically (verified on every front-camera frame we could compare, 47 frames over four
   episodes and splits; the tfrecord reading path was verified on Waymo's tutorial segment).
   The extractor also checks the official calibration and vehicle poses against the released
   `lowdim/` files.

## If a checksum differs

The extractor stops with the first differing frame. A different ffmpeg or Pillow build is the
usual cause; the manifests are the ground truth. Open an issue with the tool versions and the
episode name.
