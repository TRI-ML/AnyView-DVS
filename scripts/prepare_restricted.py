# Rebuild the pixels of restricted AnyViewBench splits from the official dataset downloads.
'''
The Ego-Exo4D and Waymo splits come without rgb/ because their licenses do not allow
redistribution. Each of their episodes carries a fetch_manifest.json that names the source
take or segment, the source camera of each view, the original frame indices, the processing
chain, and the SHA-256 of every reference frame. This script rebuilds rgb/cam0 (target view)
and rgb/cam1 (input view) from your own official download and verifies the checksums.

Usage:
    python scripts/prepare_restricted.py --avb-root data/AnyViewBench_zeroshot \\
        --dataset egoexo4d --source-root /path/to/official/EgoExo4D
    python scripts/prepare_restricted.py --avb-root data/AnyViewBench_indist \\
        --dataset waymo --source-root /path/to/waymo_open_dataset_v1/validation

Ego-Exo4D: source-root holds the official layout with takes/<take>/frame_aligned_videos/
downscaled/448/<cam>.mp4 (download only the "downscaled_takes/448" part of the referenced
takes; ffmpeg is needed, the imageio-ffmpeg wheel provides one). Waymo: source-root holds the
Perception v1 validation tfrecords named in the manifests. See docs/RESTRICTED_DATA.md.
'''

import argparse
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)

from anyview.restricted import compare_sha256  # noqa: E402

DATASET_SPLIT_PREFIX = {'egoexo4d': 'egoexo4d_', 'waymo': 'waymo_'}


def parse_args():
    parser = argparse.ArgumentParser(description='Rebuild restricted AnyViewBench pixels from official downloads',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--avb-root', type=str, required=True, help='Benchmark tree (contains splits.json)')
    parser.add_argument('--dataset', type=str, required=True, choices=sorted(DATASET_SPLIT_PREFIX))
    parser.add_argument('--source-root', type=str, required=True, help='Official download of the source dataset')
    parser.add_argument('--splits', type=str, default='all', help='Comma-separated split names or "all"')
    parser.add_argument('--overwrite', type=int, default=0, help='1 = rebuild episodes that already have rgb/')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.dataset == 'egoexo4d':
        from anyview.restricted_egoexo4d import rebuild_episode
    else:
        from anyview.restricted_waymo import rebuild_episode

    with open(os.path.join(args.avb_root, 'splits.json'), 'r') as f:
        index = json.load(f)
    prefix = DATASET_SPLIT_PREFIX[args.dataset]
    splits = [s for s in index['splits'] if s.startswith(prefix)]
    if args.splits.strip().lower() != 'all':
        wanted = [s.strip() for s in args.splits.split(',') if s.strip()]
        splits = [s for s in splits if s in wanted]
    if not splits:
        raise SystemExit(f'No {args.dataset} splits in {args.avb_root}')

    (done, kept, failed) = (0, 0, [])
    for split in splits:
        info = index['splits'][split]
        for episode in info['episodes']:
            ep_dir = os.path.join(args.avb_root, info.get('dir', split), episode)
            if not args.overwrite and os.path.isdir(os.path.join(ep_dir, 'rgb', 'cam1')):
                kept += 1
                continue
            manifest_fp = os.path.join(ep_dir, 'fetch_manifest.json')
            with open(manifest_fp, 'r') as f:
                manifest = json.load(f)
            identity = manifest.get('identity')
            if identity is None:
                raise SystemExit(f'{manifest_fp} has no identity block (manifest version 1); '
                                 f'download a current benchmark archive')
            try:
                produced = rebuild_episode(ep_dir, args.source_root, identity)
            except Exception as e:
                failed.append(episode)
                print(f'FAILED {episode}: {e}', file=sys.stderr)
                continue
            expected = manifest['sha256']
            bad = compare_sha256(produced, expected)
            if bad:
                failed.append(episode)
                print(f'FAILED {episode}: {len(bad)} frames differ from the reference checksums '
                      f'(first: {bad[0]})', file=sys.stderr)
                continue
            done += 1
            print(f'  {episode}: {sum(len(v) for v in expected.values())} frames rebuilt and verified')
    print(f'{done} episodes rebuilt, {kept} already present, {len(failed)} failed')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
