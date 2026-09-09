# AnyView-DVS inference entry point: unified scene (input view + cameras) -> generated target view.
'''
Run 1->1 dynamic view synthesis on a single clip.

The episode is one unified scene (metadata.json + rgb/ + lowdim/, frames or
per-camera mp4): cam1 is the input view, cam0 the target view whose cameras are given. The
script VAE-encodes the input view, builds Plucker cams entries for both views, runs the
diffusion pipe, and VAE-decodes the generated target view (view 0) into an mp4 + pngs.

Example:
    python scripts/infer.py --episode path/to/episode \
        --ckpt model.pt --tokenizer tokenizer.pth --out out/demo
'''

import argparse
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)


MAX_FRAMES = 41  # single-chunk tokenizer limit (11 latent frames)


def parse_args():
    parser = argparse.ArgumentParser(
        description='AnyView-DVS inference: input view + cameras -> generated target view',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episode', type=str, required=True,
                        help='Unified scene directory: metadata.json + rgb/ + lowdim/ '
                             '(frames or per-camera mp4); cam1 = input, cam0 = target')
    parser.add_argument('--input-cam', type=str, default=None,
                        help='Input camera name (default: metadata specific.roles.input or cam1)')
    parser.add_argument('--target-cam', type=str, default=None,
                        help='Target camera name (default: metadata specific.roles.target or cam0)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='DiT checkpoint path (.pt)')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Video tokenizer (VAE) checkpoint path (.pth)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num-steps', type=int, default=35,
                        help='Diffusion sampling steps')
    parser.add_argument('--num-frames', type=int, default=None,
                        help='Clip length; must be 1 + 4k and at most 41 (e.g. 13, 29, 41). '
                             'Default: the longest such prefix of the episode')
    parser.add_argument('--guidance', type=float, default=0.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=0,
                        help='Sampling seed')
    parser.add_argument('--scale-factor', type=float, default=None,
                        help='Translation scale factor for cams; default resolves from '
                             'metadata specific.scale_factor or the source dataset name')
    parser.add_argument('--fps', type=int, default=12,
                        help='Output video frame rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device')
    args = parser.parse_args()
    return args


def apply_stride(val, stride, capped=False):
    '''
    Snap val to a multiple of stride (round to nearest; capped = never round up).
    '''
    if stride is None or stride == 1:
        return val
    rem = val % stride
    val2 = val // stride * stride
    if rem > stride / 2:
        val2 += stride
    if capped and val2 > val:
        val2 -= stride
    return val2


def resize_spec_shape(hw, target, stride=16):
    '''
    The training-data resize spec [-1, target, s<stride>]: longest side = target, the other
    side rounded, then both snapped to the stride with ties rounding DOWN.
    '''
    (height, width) = (int(hw[0]), int(hw[1]))
    if width > height:
        new_w = target
        new_h = round(height / width * target)
    else:
        new_h = target
        new_w = round(width / height * target)
    shape = (int(apply_stride(int(new_h), stride)), int(apply_stride(int(new_w), stride)))
    return shape


def snap_shape(height, width, target=576, stride=16, grid_res=384):
    '''
    Inference resolution for a native (H, W) frame. The checkpoint was trained and evaluated
    on frames stored at long side 384 and resized to 576 with the spec above, so the grid is
    computed in the same two stages (native -> 384 grid -> 576 grid) while the pixels are
    resized once, directly from the native frame. For 16:9 inputs this gives 576x304 (not
    576x320); 4:3 inputs give 576x432 either way. Intrinsics scale in memory with the resize.
    '''
    hw_grid = resize_spec_shape((height, width), grid_res, stride)
    shape = resize_spec_shape(hw_grid, target, stride)
    return shape


def resize_video(frames, hw):
    '''
    Resize a uint8 video with Lanczos (matches the eval-time PIL resize).
    :param frames: (T, H, W, 3) uint8 numpy array.
    :param hw: target (H, W).
    :return video: (3, T, H, W) float32 tensor in [0, 1].
    '''
    import numpy as np
    import torch
    from PIL import Image

    resized = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((hw[1], hw[0]), resample=Image.LANCZOS)
        resized.append(np.asarray(img))

    stacked = np.stack(resized, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    video = torch.from_numpy(stacked).permute(3, 0, 1, 2).contiguous()
    return video


def reanchor_world2cam(world2cam):
    '''
    Re-anchor poses so view 0, frame 0 becomes the world origin (identity), matching
    the zero-origin eval protocol. Idempotent when cameras are already anchored.
    :param world2cam: dict mapping view index to (T, 4, 4) tensor.
    '''
    import torch

    anchor_inv = torch.linalg.inv(world2cam[0][0])
    anchored = {}
    for (view, w2c) in world2cam.items():
        anchored[view] = w2c @ anchor_inv
    return anchored


def load_episode(episode_dir, num_frames=None, input_cam=None, target_cam=None):
    '''
    Read one unified scene: input-view frames plus the cameras of both views.
    :param num_frames (int or None): clip length from frame 0; None = all frames.
    :return (frames, cams, hw_target): frames (T, H, W, 3) uint8; cams dict with 'intrinsics'
        {0/1: (3, 3)}, 'world2cam' {0/1: (T, 4, 4)}, 'scale_factor', 'source'; hw_target =
        native (H, W) of the target camera, or None when unknown (then = input size).
    '''
    import torch
    from anyview.unified import UnifiedScene, world2cam_from_cam2world

    scene = UnifiedScene(episode_dir)
    roles = scene.specific.get('roles', {})
    cam_in = input_cam or roles.get('input', 'cam1')
    cam_out = target_cam or roles.get('target', 'cam0')
    for cam in (cam_in, cam_out):
        if cam not in scene.cameras:
            raise ValueError(f'{episode_dir}: camera {cam!r} not among {scene.cameras}; pick the '
                             f'input / target cameras with --input-cam / --target-cam')
    if not scene.has_rgb(cam_in):
        raise FileNotFoundError(
            f'{episode_dir}: no frames for camera {cam_in}. AnyViewBench splits from restricted '
            f'source datasets come without pixels; see docs/RESTRICTED_DATA.md')

    # Clip length must be 1 + 4k and at most MAX_FRAMES; by default use the longest such
    # prefix of the episode.
    if num_frames is None:
        T = min(scene.num_frames, MAX_FRAMES)
        T = T - (T - 1) % 4
        if T != scene.num_frames:
            print(f'Using the first {T} of {scene.num_frames} frames (clip length must be '
                  f'1 + 4k and at most {MAX_FRAMES}; pass --num-frames to choose)')
    else:
        T = int(num_frames)
        if T > scene.num_frames:
            raise ValueError(f'{episode_dir} has {scene.num_frames} frames, need {T}')
        if T < 1 or (T - 1) % 4 != 0 or T > MAX_FRAMES:
            raise ValueError(f'--num-frames must be 1 + 4k and at most {MAX_FRAMES} '
                             f'(e.g. 13, 29, 41), got {T}')
    indices = list(range(T))
    frames = scene.load_rgb(cam_in, indices)
    (K_in, c2w_in, _) = scene.load_lowdim(cam_in, indices)
    (K_out, c2w_out, _) = scene.load_lowdim(cam_out, indices)

    cams = {
        'intrinsics': {0: torch.from_numpy(K_out[0]), 1: torch.from_numpy(K_in[0])},
        'world2cam': {0: torch.from_numpy(world2cam_from_cam2world(c2w_out)),
                      1: torch.from_numpy(world2cam_from_cam2world(c2w_in))},
        'scale_factor': scene.specific.get('scale_factor', None),
        # Scenes without an AVB-style source block (e.g. Kubric-5D) resolve by dataset name.
        'source': scene.specific.get('source', {'dataset': scene.metadata.get('info', {}).get('name', '')}),
    }

    hw_target = None
    per_cam = scene.specific.get('resolution_per_camera', None)
    if per_cam is not None and cam_out in per_cam:
        hw_target = tuple(per_cam[cam_out])
    elif scene.has_rgb(cam_out):
        hw_target = tuple(scene.load_rgb(cam_out, [0]).shape[1:3])
    elif scene.metadata.get('resolution') is not None:
        hw_target = tuple(scene.metadata['resolution'])
    return (frames, cams, hw_target)


def prepare_cams_latent(vae, intrinsics, world2cam, hw_native, hw_infer, scale_factor, device):
    '''
    Build the 32-channel cams latent for one view: pixel-space Plucker channels
    (unit ray directions + moment), translation scaling, [-1, 1] clip, VAE encoding.
    '''
    from anyview.cameras import plucker_video, scale_intrinsics

    K_scaled = scale_intrinsics(intrinsics, hw_native, hw_infer)
    # plucker_video applies the translation scale_factor, bf16 cast, and [-1, 1] clamp itself.
    plucker = plucker_video(K_scaled, world2cam, hw_infer, scale_factor=scale_factor)
    plucker = plucker.to(device=device)

    cams_latent = vae.encode_cams(plucker.unsqueeze(0))
    return cams_latent


def generate_target_view(pipe, vae, config, frames, cams, scale_factor,
                         seed, num_steps, guidance, device, hw_target=None, native_hw=None):
    '''
    Core DVS forward pass shared by infer.py and eval_avb.py.
    :param frames: (T, H, W, 3) uint8 numpy array with the input-view (view 1) video.
    :param cams: dict from load_episode (world2cam already re-anchored).
    :param hw_target: native (H, W) of the target view; None = same as the input view.
    :param native_hw: ((H1, W1), (H0, W0)) sizes the intrinsics refer to, when frames /
        hw_target were pre-resized by the caller; None = the sizes given.
    :return (pred, info): pred = (T, 3, H0, W0) float32 tensor in [0, 1] on cpu;
        info = dict with the snapped per-view inference resolutions.
    '''
    import torch
    from anyview.logistics import build_dvs_entries, unpack_entries_from_streams

    (T, H1n, W1n, _) = frames.shape
    assert T >= 1 and (T - 1) % config.vae_temporal_factor == 0, \
        f'frame count must be 1 + {config.vae_temporal_factor}k (e.g. 13, 29, 41), got {T}'

    hw1 = snap_shape(H1n, W1n)
    if hw_target is None:
        hw0n = (H1n, W1n)
    else:
        hw0n = tuple(hw_target)
    hw0 = snap_shape(hw0n[0], hw0n[1])
    if native_hw is None:
        (hw1_native, hw0_native) = ((H1n, W1n), hw0n)
    else:
        (hw1_native, hw0_native) = (tuple(native_hw[0]), tuple(native_hw[1]))

    video1 = resize_video(frames, hw1)  # (3, T, H1, W1) in [0, 1]
    rgb1_latent = vae.encode_rgb(video1.unsqueeze(0).to(device))

    cams0_latent = prepare_cams_latent(
        vae, cams['intrinsics'][0], cams['world2cam'][0], hw0_native, hw0, scale_factor, device)
    cams1_latent = prepare_cams_latent(
        vae, cams['intrinsics'][1], cams['world2cam'][1], hw1_native, hw1, scale_factor, device)

    # Constant DVS masks are baked in here (rgb1 + both cams input, rgb0 output).
    entries = build_dvs_entries(None, rgb1_latent, cams0_latent, cams1_latent)
    # guidance is fixed at 0.0 inside the pipeline (text-free, no CFG); the arg is accepted
    # by the script for interface stability but only 0.0 is supported.
    assert guidance == 0.0, 'only guidance 0.0 is supported'
    samples = pipe.generate(entries, seed=seed, num_sampling_steps=num_steps)

    pred_entries = unpack_entries_from_streams(samples['y0_pred_streams'])
    pred_video = vae.decode_rgb(pred_entries['rgb0'])  # (1, 3, T, H0, W0) in [0, 1]
    pred = pred_video[0].to(torch.float32).clamp(0.0, 1.0).permute(1, 0, 2, 3).cpu()

    info = {'hw_input': list(hw1), 'hw_target': list(hw0)}
    return (pred, info)


def save_video(pred, path, fps):
    '''
    :param pred: (T, 3, H, W) float tensor in [0, 1].
    '''
    import imageio.v2 as imageio
    import numpy as np

    frames = (pred.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    imageio.mimwrite(path, list(frames), fps=fps, quality=8)


def save_frames(pred, frames_dir):
    '''
    :param pred: (T, 3, H, W) float tensor in [0, 1].
    '''
    import imageio.v2 as imageio
    import numpy as np

    os.makedirs(frames_dir, exist_ok=True)
    frames = (pred.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    for (t, frame) in enumerate(frames):
        imageio.imwrite(os.path.join(frames_dir, f'{t:06d}.png'), frame)


def main():
    args = parse_args()

    import torch
    from anyview.cameras import resolve_scale_factor
    from anyview.config import AnyViewConfig
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    device = torch.device(args.device)
    config = AnyViewConfig(checkpoint_path=args.ckpt, tokenizer_path=args.tokenizer)

    try:
        (frames, cams, hw_target) = load_episode(args.episode, args.num_frames,
                                                 args.input_cam, args.target_cam)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f'error: {e}')
    cams['world2cam'] = reanchor_world2cam(cams['world2cam'])
    scale_factor = resolve_scale_factor(cams, override=args.scale_factor)
    print(f'Loaded {frames.shape[0]} frames at {frames.shape[1]}x{frames.shape[2]}, '
          f'target native {hw_target}, scale_factor = {scale_factor}')

    print(f'Loading tokenizer from {args.tokenizer} ...')
    vae = load_vae(config.tokenizer_path, device)
    print(f'Loading DiT pipeline from {args.ckpt} ...')
    pipe = load_pipeline(config, device)

    with torch.no_grad():
        (pred, info) = generate_target_view(
            pipe, vae, config, frames, cams, scale_factor,
            seed=args.seed, num_steps=args.num_steps, guidance=args.guidance,
            device=device, hw_target=hw_target)

    os.makedirs(args.out, exist_ok=True)
    save_video(pred, os.path.join(args.out, 'pred.mp4'), args.fps)
    save_frames(pred, os.path.join(args.out, 'frames'))

    run_info = {
        'episode': args.episode,
        'ckpt': args.ckpt,
        'tokenizer': args.tokenizer,
        'seed': args.seed,
        'num_steps': args.num_steps,
        'guidance': args.guidance,
        'scale_factor': scale_factor,
        **info,
    }
    with open(os.path.join(args.out, 'info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)

    print(f'Done. Wrote pred.mp4 + {pred.shape[0]} pngs to {args.out}')


if __name__ == '__main__':
    main()
