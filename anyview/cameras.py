# Plucker camera channels for the DVS streams: 6-ch pixel-space plucker video
# (rays, cross) per view, VAE-encoded per 3-ch half and concatenated to the 32 cams channels.

'''
Camera (cams) entry construction.

The 32 cams channels of each stream are NOT a tiled or padded 6-channel embedding. The
pipeline builds a 6-channel plucker video (rays 0:3, cross 3:6) at PIXEL resolution over
all pixel frames, then encodes the two 3-channel halves separately with the video
tokenizer and concatenates the latents: (32, T_lat, H_lat, W_lat) per view, where the
temporal compression (41 pixel frames to 11 latent frames) is done by the VAE itself.

Conventions (these match the eval pipeline exactly):
- world2cam maps world points to camera points: x_cam = R @ x_world + t.
- Before plucker, all world2cam matrices of BOTH views are re-expressed relative to the
  target view (view 0) at frame 0, which becomes the identity (zero-origin normalization).
- The pixel grid uses integer coordinates 0 .. W-1 and 0 .. H-1 (align_corners=True).
- Rays are rotated to world space and then unit-normalized; cross = origin x rays.
- The cross channels are multiplied by a per-dataset metric scale factor (see
  DATASET_SCALE_FACTORS); the 6-channel video is cast to bfloat16 and clamped to [-1, 1]
  before VAE encoding. Rays are never scaled.
'''

import torch

# Per-dataset metric scale factor applied to the translation-dependent (cross) channels, keyed
# by source dataset name (driving 1/16, kubric 1/32, everything else 1/8); must match the values
# the checkpoint was trained with.
DATASET_SCALE_FACTORS = {
    # driving
    'argoverse2sync': 1.0 / 16.0,
    'ddad': 1.0 / 16.0,
    'lyftl5': 1.0 / 16.0,
    'pd4d': 1.0 / 16.0,
    'waymo': 1.0 / 16.0,
    # synthetic (oversized scenes)
    'kubric4d': 1.0 / 32.0,
    'kubric5d': 1.0 / 32.0,
    # robotics
    'droid': 1.0 / 8.0,
    'droidcalib': 1.0 / 8.0,
    'droidcalibv2': 1.0 / 8.0,
    'lbmv12': 1.0 / 8.0,
    # 4D / egocentric
    'assemblyhands': 1.0 / 8.0,
    'assemblyhands_v2': 1.0 / 8.0,
    'dycheck': 1.0 / 8.0,
    'egoexo4d': 1.0 / 8.0,
}


def resolve_scale_factor(sample, override=None):
    '''
    Scale factor of one scene: override > sample['scale_factor'] (metadata specific.scale_factor)
    > DATASET_SCALE_FACTORS[sample['source']['dataset']].
    '''
    if override is not None:
        return float(override)
    if sample.get('scale_factor') is not None:
        return float(sample['scale_factor'])

    source = sample.get('source') or {}
    dset = str(source.get('dataset', '')).lower()
    if dset in DATASET_SCALE_FACTORS:
        return DATASET_SCALE_FACTORS[dset]

    raise ValueError(
        f'Cannot resolve scale factor (source.dataset = {dset!r}); pass --scale-factor '
        f'explicitly or add specific.scale_factor to metadata.json. '
        f'Known datasets: {sorted(DATASET_SCALE_FACTORS.keys())}')


def invert_rigid(pose):
    '''
    Inverts one or more rigid 4x4 transforms via rotation transpose: [R | t] -> [R^T | -R^T t].
    Assumes an orthonormal rotation block (parity with the source pose inversion).
    :param pose: (..., 4, 4) tensor of float.
    :return inverse: (..., 4, 4) tensor of float32.
    '''
    pose = torch.as_tensor(pose, dtype=torch.float32)
    rot_T = pose[..., 0:3, 0:3].transpose(-2, -1)
    trans = pose[..., 0:3, 3:4]

    inverse = torch.zeros_like(pose)
    inverse[..., 0:3, 0:3] = rot_T
    inverse[..., 0:3, 3:4] = -rot_T @ trans
    inverse[..., 3, 3] = 1.0
    return inverse


def normalize_world2cam(world2cam_per_view):
    '''
    Zero-origin reference normalization applied at eval before plucker: re-express all
    world2cam matrices so that the TARGET view (view 0) at frame 0 becomes the identity /
    world origin. Relative geometry between all cameras is preserved.
    :param world2cam_per_view: list (V) of (T, 4, 4) tensors of float, view 0 (target) first.
    :return normalized: list (V) of (T, 4, 4) tensors of float32.
    '''
    ref = torch.as_tensor(world2cam_per_view[0], dtype=torch.float32)[0]
    ref_inv = invert_rigid(ref)  # (4, 4)

    normalized = []
    for world2cam in world2cam_per_view:
        world2cam = torch.as_tensor(world2cam, dtype=torch.float32)
        normalized.append(world2cam @ ref_inv)
    return normalized


def scale_intrinsics(intrinsics_3x3, orig_hw, new_hw):
    '''
    Rescales pinhole intrinsics for a resized image; purely multiplicative on (fx, cx) and
    (fy, cy), matching the source align_corners=True convention (no half-pixel offset).
    :param intrinsics_3x3: (3, 3) tensor of float.
    :param orig_hw: (H, W) of the resolution intrinsics_3x3 refers to.
    :param new_hw: (H, W) of the target resolution.
    :return scaled: (3, 3) tensor of float32.
    '''
    ratio_h = float(new_hw[0]) / float(orig_hw[0])
    ratio_w = float(new_hw[1]) / float(orig_hw[1])

    scaled = torch.as_tensor(intrinsics_3x3, dtype=torch.float32).clone()
    scaled[0, 0] = scaled[0, 0] * ratio_w
    scaled[0, 2] = scaled[0, 2] * ratio_w
    scaled[1, 1] = scaled[1, 1] * ratio_h
    scaled[1, 2] = scaled[1, 2] * ratio_h
    return scaled


def plucker_video(intrinsics_3x3, world2cam_4x4_per_frame, pixel_hw, scale_factor=1.0):
    '''
    Builds the raw 6-channel plucker video for one view at PIXEL resolution, exactly as the
    eval dataloader feeds it into the cams entries (before VAE encoding):
    per frame, rays = normalize(R^T @ K^-1 @ [x, y, 1]) and cross = origin x rays with
    origin = -R^T t; cross is scaled by scale_factor; the result is cast to bfloat16 and
    clamped to [-1, 1].
    :param intrinsics_3x3: (3, 3) tensor of float at pixel_hw resolution (zero skew).
    :param world2cam_4x4_per_frame: (T, 4, 4) tensor of float, already reference-normalized
        via normalize_world2cam.
    :param pixel_hw: (Hp, Wp) pixel resolution.
    :param scale_factor (float): per-dataset metric scale (see DATASET_SCALE_FACTORS).
    :return plucker: (6, T, Hp, Wp) tensor of bfloat16 in [-1, 1].
    '''
    K = torch.as_tensor(intrinsics_3x3, dtype=torch.float32)
    world2cam = torch.as_tensor(world2cam_4x4_per_frame, dtype=torch.float32)
    assert K.shape == (3, 3), f'Expected (3, 3) intrinsics, got {tuple(K.shape)}'
    assert world2cam.ndim == 3 and world2cam.shape[1:] == (4, 4), \
        f'Expected (T, 4, 4) world2cam, got {tuple(world2cam.shape)}'
    assert abs(float(K[0, 1])) < 1e-5, 'Nonzero skew is not supported'

    (T, Hp, Wp) = (world2cam.shape[0], int(pixel_hw[0]), int(pixel_hw[1]))
    device = world2cam.device
    K = K.to(device)

    # Integer pixel grid (align_corners=True): channel 0 = x, channel 1 = y, channel 2 = 1.
    ys, xs = torch.meshgrid(
        torch.arange(Hp, dtype=torch.float32, device=device),
        torch.arange(Wp, dtype=torch.float32, device=device),
        indexing='ij')
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # (3, Hp * Wp)

    # Analytic zero-skew inverse of K (parity with the source invert_intrinsics).
    inv_K = torch.eye(3, dtype=torch.float32, device=device)
    inv_K[0, 0] = 1.0 / K[0, 0]
    inv_K[1, 1] = 1.0 / K[1, 1]
    inv_K[0, 2] = -K[0, 2] / K[0, 0]
    inv_K[1, 2] = -K[1, 2] / K[1, 1]

    rays_cam = inv_K @ grid  # (3, Hp * Wp), camera-space directions at depth 1

    rot_T = world2cam[:, 0:3, 0:3].transpose(-2, -1)  # (T, 3, 3)
    trans = world2cam[:, 0:3, 3:4]  # (T, 3, 1)

    # Rotate rays to world space, then unit-normalize (this order matches the source).
    rays = rot_T @ rays_cam[None]  # (T, 3, Hp * Wp)
    rays = rays / torch.norm(rays, dim=1, keepdim=True)

    # Camera center in (normalized) world coordinates; cross = origin x rays per pixel.
    orig = (-rot_T @ trans).squeeze(-1)  # (T, 3)
    rays_flat = rays.permute(0, 2, 1)  # (T, Hp * Wp, 3)
    orig_flat = orig[:, None, :].expand_as(rays_flat)
    cross_flat = torch.cross(orig_flat, rays_flat, dim=-1)  # (T, Hp * Wp, 3)
    cross = cross_flat.permute(0, 2, 1)  # (T, 3, Hp * Wp)

    plucker = torch.cat([rays, cross], dim=1)  # (T, 6, Hp * Wp)
    plucker = plucker.reshape(T, 6, Hp, Wp).permute(1, 0, 2, 3)  # (6, T, Hp, Wp)

    # Metric scaling of the translation-dependent channels only, then bfloat16 + clamp
    # (same order as the source: scale in float32, cast, clamp).
    plucker = plucker.clone()
    plucker[3:6] = plucker[3:6] * scale_factor
    plucker = plucker.to(torch.bfloat16)
    plucker = torch.clamp(plucker, min=-1.0, max=1.0)
    return plucker


@torch.no_grad()
def plucker_channels(intrinsics_3x3, world2cam_4x4_per_frame, T, latent_hw, config, vae,
                     scale_factor=1.0):
    '''
    Builds the (32, T_lat, H_lat, W_lat) cams channels for one view: the 6-channel plucker
    video at pixel resolution is split into its rays (0:3) and cross (3:6) halves, each half
    is VAE-encoded to 16 latent channels, and the two latents are concatenated.
    :param intrinsics_3x3: (3, 3) tensor of float at pixel resolution (latent_hw * 8).
    :param world2cam_4x4_per_frame: (T, 4, 4) tensor of float, already reference-normalized
        via normalize_world2cam.
    :param T (int): pixel frame count; any valid clip length 1 + 4k (e.g. 13, 29, 41).
    :param latent_hw: (H_lat, W_lat) latent resolution.
    :param config (AnyViewConfig).
    :param vae (AnyViewVAE): tokenizer wrapper; both halves go through its raw [-1, 1]
        encode path (encode + sigma_data scaling, no RGB range remap).
    :param scale_factor (float): per-dataset metric scale (see DATASET_SCALE_FACTORS).
    :return cams_latent: (32, T_lat, H_lat, W_lat) tensor of bfloat16.
    '''
    assert T >= 1 and (T - 1) % config.vae_temporal_factor == 0, \
        f'Frame count must be 1 + {config.vae_temporal_factor}k, got {T}'
    assert world2cam_4x4_per_frame.shape[0] == T, \
        f'world2cam frame count {world2cam_4x4_per_frame.shape[0]} != T = {T}'

    (Hl, Wl) = (int(latent_hw[0]), int(latent_hw[1]))
    pixel_hw = (Hl * config.vae_spatial_factor, Wl * config.vae_spatial_factor)
    Tl = (T - 1) // config.vae_temporal_factor + 1

    plucker = plucker_video(intrinsics_3x3, world2cam_4x4_per_frame, pixel_hw,
                            scale_factor=scale_factor)
    # ^ (6, T, Hp, Wp) tensor of bfloat16 in [-1, 1].

    lat_rays = vae._encode_raw(plucker[None, 0:3])  # (1, 16, Tl, Hl, Wl)
    lat_cross = vae._encode_raw(plucker[None, 3:6])  # (1, 16, Tl, Hl, Wl)

    cams_latent = torch.cat([lat_rays, lat_cross], dim=1)[0]
    assert cams_latent.shape == (32, Tl, Hl, Wl), \
        f'Expected (32, {Tl}, {Hl}, {Wl}), got {tuple(cams_latent.shape)}'
    return cams_latent
