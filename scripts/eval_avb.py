# AVB benchmark evaluation: loop episodes, run DVS inference, score vs target GT.
'''
Evaluate an AnyView-DVS checkpoint on the AVB benchmark.

Loops over the requested AVB splits, runs the same inference path as infer.py on
every episode, computes PSNR / SSIM / LPIPS of the generated target view against
target_gt, and prints + saves the per-dataset and group-average table.

Official protocol: native frames, inference and metrics at the 576 grid (long side 576,
nearest multiple of 16), intrinsics untouched. --legacy-bake 384 only reproduces the
numbers printed in the paper, whose evaluation pipeline used for the paper read frames stored at long side 384
and upsampled them to 576 (input and ground truth alike).

Example:
    python scripts/eval_avb.py --avb-root /data/avb --ckpt model.pt \
        --tokenizer tokenizer.pth --out out/avb_eval
'''

import argparse
import json
import os
import sys
import traceback

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_RELEASE_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import infer as infer_lib  # shared helpers + generate_target_view (import-light)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate AnyView-DVS on the AVB benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--avb-root', type=str, required=True,
                        help='AVB root directory (contains splits.json, name_mapping.json and '
                             'one folder per dataset group holding the episode directories)')
    parser.add_argument('--splits', type=str, default='all',
                        help='Comma-separated split names, or "all" (default)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='DiT checkpoint path (.pt)')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Video tokenizer (VAE) checkpoint path (.pth)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for tables, metrics, and videos')
    parser.add_argument('--num-steps', type=int, default=35,
                        help='Diffusion sampling steps')
    parser.add_argument('--guidance', type=float, default=0.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=0,
                        help='Sampling seed (the same noise draw for every episode)')
    parser.add_argument('--save-videos', type=int, default=1,
                        help='Write per-episode input | pred | gt comparison mp4s (1 = on)')
    parser.add_argument('--fps', type=int, default=12,
                        help='Comparison video frame rate')
    parser.add_argument('--stop-after', type=int, default=-1,
                        help='If > 0, only evaluate this many episodes per split (smoke runs)')
    parser.add_argument('--legacy-bake', type=int, default=0,
                        help='Reproduce the numbers printed in the paper: emulate the original '
                             'evaluation pipeline used for the paper, whose frames were stored at long side N '
                             '(use 384) and upsampled to 576. 0 = official protocol (native '
                             'frames)')
    parser.add_argument('--save-pred', type=int, default=0,
                        help='Write per-episode predicted frames as PNGs (1 = on)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device')
    args = parser.parse_args()
    return args


def load_splits(avb_root):
    '''
    Read splits.json: split -> [episodes] plus the group grouping used for the
    paper table. Falls back to a single "all" group when no grouping is stored.
    :return (split_map, groups): both dicts; groups maps group name to [split names].
    '''
    path = os.path.join(avb_root, 'splits.json')
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data.get('splits'), dict):
        split_map = data['splits']
    else:
        split_map = {k: v for (k, v) in data.items() if isinstance(v, list)}
    if not split_map:
        raise ValueError(f'No splits found in {path}')

    groups = None
    for key in ('groups', 'panels'):
        if isinstance(data.get(key), dict):
            groups = {group: list(members) for (group, members) in data[key].items()}
            break
    if groups is None:
        groups = {'all': sorted(split_map.keys())}

    return (split_map, groups)


def as_uint8_video(video):
    '''
    Normalize a video to (T, H, W, 3) uint8 numpy, from either a (T, 3, H, W)
    float tensor in [0, 1] or an already-uint8 (T, H, W, 3) array.
    '''
    import numpy as np
    import torch

    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    video = np.asarray(video)
    # Layout by dtype: uint8 videos are channels-last already; float ones are (T, 3, H, W).
    if video.dtype != np.uint8:
        video = (np.clip(video, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if video.ndim == 4 and video.shape[1] == 3 and video.shape[-1] != 3:
            video = video.transpose(0, 2, 3, 1)
    assert video.ndim == 4 and video.shape[-1] == 3, f'bad video shape {video.shape}'
    return video


vidar_resize_shape = infer_lib.resize_spec_shape  # one stage of the resize rule used when the training data was prepared


def legacy_bake_video(frames, bake_res, target_res=576):
    '''
    Emulates the input path of the evaluation used for the paper: frames were stored at long side
    bake_res (Lanczos, JPEG quality 100) and read back through a [-1, target_res, s16]
    Lanczos resize. Applies to input and GT alike.
    :param frames: (T, H, W, 3) uint8 numpy array (native).
    :return video: (T, H', W', 3) uint8 numpy array at the paper-evaluation resolution.
    '''
    import io
    import numpy as np
    from PIL import Image

    hw_bake = vidar_resize_shape(frames.shape[1:3], bake_res)
    hw_eval = vidar_resize_shape(hw_bake, target_res)
    out = []
    for frame in frames:
        img = Image.fromarray(frame).resize((hw_bake[1], hw_bake[0]), resample=Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=100)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = img.resize((hw_eval[1], hw_eval[0]), resample=Image.LANCZOS)
        out.append(np.asarray(img))
    video = np.stack(out, axis=0)
    return video


def score_episode(evaluator, pred, gt_resized, device):
    '''
    Per-frame PSNR / SSIM / LPIPS, averaged over valid frames (all-black GT frames
    return the -999 sentinel and are excluded).
    :param pred: (T, 3, H, W) float tensor in [0, 1].
    :param gt_resized: (T, 3, H, W) float tensor in [0, 1] at the same resolution.
    :return metrics: dict with psnr / ssim / lpips / num_frames.
    '''
    rows = evaluator.compute(gt_resized.to(device), pred.to(device))
    rows = rows.detach().float().cpu()
    valid = rows[:, 0] > -900.0
    if valid.sum() == 0:
        raise ValueError('All GT frames are empty; no valid metrics')
    rows = rows[valid]

    metrics = {
        'psnr': float(rows[:, 0].mean()),
        'ssim': float(rows[:, 1].mean()),
        'lpips': float(rows[:, 2].mean()),
        'num_frames': int(rows.shape[0]),
    }
    return metrics


def save_comparison_video(input_frames, pred, gt_resized, path, fps):
    '''
    Horizontal input | pred | gt concat at the target-view resolution.
    :param input_frames: (T, H, W, 3) uint8 numpy (native input view).
    :param pred: (T, 3, H, W) float tensor in [0, 1].
    :param gt_resized: (T, 3, H, W) float tensor in [0, 1], same resolution as pred.
    '''
    import imageio.v2 as imageio
    import numpy as np

    hw = tuple(pred.shape[-2:])
    input_resized = infer_lib.resize_video(input_frames, hw).permute(1, 0, 2, 3)
    panels = [input_resized, pred, gt_resized]
    panels = [(p.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
              for p in panels]
    frames = list(np.concatenate(panels, axis=2))
    imageio.mimwrite(path, frames, fps=fps, quality=8)


def average_rows(rows):
    '''
    Macro-average a list of metric dicts.
    '''
    avg = {
        'psnr': sum(r['psnr'] for r in rows) / len(rows),
        'ssim': sum(r['ssim'] for r in rows) / len(rows),
        'lpips': sum(r['lpips'] for r in rows) / len(rows),
    }
    return avg


def format_table(per_split, groups):
    '''
    Per-dataset rows grouped into groups, each followed by the group macro-average.
    :param per_split: dict mapping split to {'episodes', 'psnr', 'ssim', 'lpips'}.
    :param groups: dict mapping group name to [split names].
    '''
    lines = []
    header = f'{"split":<32} {"#ep":>5} {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8}'
    lines.append(header)
    lines.append('=' * len(header))

    listed = set()
    group_avgs = {}
    for (group, members) in groups.items():
        present = [s for s in members if s in per_split]
        if not present:
            continue
        for split in present:
            r = per_split[split]
            lines.append(f'{split:<32} {r["episodes"]:>5d} {r["psnr"]:>8.2f} '
                         f'{r["ssim"]:>8.4f} {r["lpips"]:>8.4f}')
            listed.add(split)
        avg = average_rows([per_split[s] for s in present])
        group_avgs[group] = avg
        lines.append('-' * len(header))
        lines.append(f'{group + " (group avg)":<32} {"":>5} {avg["psnr"]:>8.2f} '
                     f'{avg["ssim"]:>8.4f} {avg["lpips"]:>8.4f}')
        lines.append('')

    leftover = [s for s in per_split if s not in listed]
    for split in leftover:
        r = per_split[split]
        lines.append(f'{split:<32} {r["episodes"]:>5d} {r["psnr"]:>8.2f} '
                     f'{r["ssim"]:>8.4f} {r["lpips"]:>8.4f}')

    table = '\n'.join(lines)
    return (table, group_avgs)


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # per-episode progress reaches redirected logs
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')  # lpips' VGG loader

    import torch
    from anyview.avb_dataset import AVBDataset
    from anyview.config import AnyViewConfig
    from anyview.cameras import resolve_scale_factor
    from anyview.metrics import RGBEvaluation
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    device = torch.device(args.device)
    config = AnyViewConfig(checkpoint_path=args.ckpt, tokenizer_path=args.tokenizer)

    (split_map, groups) = load_splits(args.avb_root)
    present = {s_ for (s_, v) in split_map.items()
               if os.path.isdir(os.path.join(args.avb_root, v.get('dir', s_) if isinstance(v, dict) else s_))}
    if len(present) < len(split_map):
        print(f'{len(split_map) - len(present)} splits listed in splits.json are not present in this '
              f'tree and are skipped')
    if args.splits.strip().lower() == 'all':
        splits = sorted(present)
    else:
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]
        unknown = [s for s in splits if s not in present]
        if unknown:
            raise ValueError(f'Splits {unknown} are not in this tree; available: {sorted(present)}')
    # Splits of restricted source datasets come without pixels in the public tier: skip them
    # up front (before the model loads) with one notice each.
    skipped_splits = []
    for split in list(splits):
        probe = AVBDataset(args.avb_root, split)
        if not probe.has_pixels(probe.episodes[0]):
            print(f'[{split}] skipped: this split comes without pixels (restricted source '
                  f'dataset); see docs/RESTRICTED_DATA.md to rebuild it')
            skipped_splits.append(split)
            splits.remove(split)
    if not splits:
        raise SystemExit('Nothing to evaluate: every requested split lacks pixels')
    print(f'Evaluating {len(splits)} splits: {splits}')

    print(f'Loading tokenizer from {args.tokenizer} ...')
    vae = load_vae(config.tokenizer_path, device)
    print(f'Loading DiT pipeline from {args.ckpt} ...')
    pipe = load_pipeline(config, device)

    evaluator = RGBEvaluation()  # scores at the 576 inference grid

    os.makedirs(args.out, exist_ok=True)

    per_episode = []
    per_split = {}
    num_failed = 0

    for split in splits:
        dataset = AVBDataset(args.avb_root, split)
        num_episodes = len(dataset)
        if args.stop_after > 0:
            num_episodes = min(args.stop_after, num_episodes)
        print(f'\n[{split}] {num_episodes} episodes')

        split_rows = []
        for idx in range(num_episodes):
            try:
                item = dataset[idx]
                episode = item.get('episode', f'{split}_ep{idx:03d}')

                if item.get('target_gt') is None:
                    print(f'  {episode}: no target_gt, skipping')
                    continue

                input_frames = as_uint8_video(item['input'])
                gt_frames = as_uint8_video(item['target_gt'])
                native_hw = (tuple(input_frames.shape[1:3]), tuple(gt_frames.shape[1:3]))
                if args.legacy_bake > 0:
                    input_frames = legacy_bake_video(input_frames, args.legacy_bake)
                    gt_frames = legacy_bake_video(gt_frames, args.legacy_bake)
                hw_target_native = tuple(gt_frames.shape[1:3])

                # View 0 = target, view 1 = input (AVBDataset sample key convention).
                cams = {
                    'intrinsics': {0: item['intrinsics_target'].float(),
                                   1: item['intrinsics_input'].float()},
                    'world2cam': {0: item['world2cam_target'].float(),
                                  1: item['world2cam_input'].float()},
                }
                cams['world2cam'] = infer_lib.reanchor_world2cam(cams['world2cam'])
                scale_factor = resolve_scale_factor(item)

                with torch.no_grad():
                    (pred, info) = infer_lib.generate_target_view(
                        pipe, vae, config, input_frames, cams, scale_factor,
                        seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
                        device=device, hw_target=hw_target_native, native_hw=native_hw)

                # GT resized to the snapped inference resolution (supervision-at-576
                # protocol); any further metric downscale happens inside the evaluator.
                gt_resized = infer_lib.resize_video(
                    gt_frames, tuple(info['hw_target'])).permute(1, 0, 2, 3)

                metrics = score_episode(evaluator, pred, gt_resized, device)
                row = {'split': split, 'episode': episode, **metrics}
                split_rows.append(row)
                per_episode.append(row)
                print(f'  {episode}: PSNR={metrics["psnr"]:.2f} '
                      f'SSIM={metrics["ssim"]:.4f} LPIPS={metrics["lpips"]:.4f}')

                if args.save_pred:
                    infer_lib.save_frames(pred, os.path.join(args.out, 'pred', split, episode))
                if args.save_videos:
                    video_dir = os.path.join(args.out, 'videos', split)
                    os.makedirs(video_dir, exist_ok=True)
                    save_comparison_video(
                        input_frames, pred, gt_resized,
                        os.path.join(video_dir, f'{episode}.mp4'), args.fps)

            except Exception as e:
                num_failed += 1
                print(f'  ERROR on {split} episode {idx}: {e}')
                traceback.print_exc()
                continue

        if split_rows:
            avg = average_rows(split_rows)
            per_split[split] = {'episodes': len(split_rows), **avg}

    if not per_split:
        raise RuntimeError('No episodes were evaluated successfully')

    (table, group_avgs) = format_table(per_split, groups)
    protocol = 'official (native frames)'
    if args.legacy_bake > 0:
        protocol = f'legacy-bake {args.legacy_bake} (paper reproduction)'
    print(f'\nProtocol: {protocol}; metrics at the 576 inference grid')
    print(table)
    if num_failed > 0:
        print(f'WARNING: {num_failed} episodes failed (see errors above); results are INCOMPLETE')

    results = {
        'ckpt': args.ckpt,
        'legacy_bake': args.legacy_bake,
        'num_steps': args.num_steps,
        'guidance': args.guidance,
        'seed': args.seed,
        'num_failed': num_failed,
        'incomplete': num_failed > 0,
        'skipped_splits': skipped_splits,
        'per_split': per_split,
        'groups': group_avgs,
        'per_episode': per_episode,
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, 'table.txt'), 'w') as f:
        f.write(table + '\n')

    csv_path = os.path.join(args.out, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('row,episodes,psnr,ssim,lpips\n')
        for (split, r) in per_split.items():
            f.write(f'{split},{r["episodes"]},{r["psnr"]:.4f},'
                    f'{r["ssim"]:.6f},{r["lpips"]:.6f}\n')
        for (group, avg) in group_avgs.items():
            f.write(f'{group} (group avg),,{avg["psnr"]:.4f},{avg["ssim"]:.6f},{avg["lpips"]:.6f}\n')

    print(f'Saved results.json, table.txt, results.csv to {args.out}')
    if num_failed > 0:
        raise SystemExit(2)  # partial aggregates must not pass as a complete evaluation


if __name__ == '__main__':
    main()
