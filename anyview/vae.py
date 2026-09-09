# Cosmos video tokenizer wrapper: load tokenizer.pth, encode/decode RGB entries.

import torch

from .vendor import log
from .vendor.tokenizer import TokenizerInterface

# Latents are multiplied by sigma_data after encode (and divided before decode) to match the
# diffusion pipeline; the released checkpoint uses sigma_data = 1.0.
SIGMA_DATA = 1.0

# One clip = one chunk: the model operates on exactly 41 pixel frames (11 latent frames).
TOKENIZER_CHUNK_DURATION = 41


class AnyViewVAE:
    '''
    Minimal wrapper around the cosmos-predict2 Wan-style video tokenizer for RGB entries.
    Encode maps pixel videos in [0, 1] to 16-channel latents at (1 + (T-1)/4, H/8, W/8);
    decode maps latents back to pixel videos in [0, 1].
    '''

    def __init__(self, tokenizer_path: str, device: str = 'cuda'):
        '''
        :param tokenizer_path (str): Local path to the cosmos-predict2 tokenizer checkpoint
            (nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth).
        :param device (str): The underlying tokenizer weights always load onto CUDA.
        '''
        self.device = device
        self.tensor_kwargs = dict(device=device, dtype=torch.bfloat16)

        self.video_tokenizer = TokenizerInterface(
            chunk_duration=TOKENIZER_CHUNK_DURATION,
            load_mean_std=False,
            vae_pth=tokenizer_path,
        )

        # (T, H, W) downsampling factors between pixel and latent space: (4, 8, 8).
        self.vae_ratio = (self.video_tokenizer.temporal_compression_factor,
                          self.video_tokenizer.spatial_compression_factor,
                          self.video_tokenizer.spatial_compression_factor)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        '''
        Pixel frame count T to latent frame count 1 + (T-1)/4.
        '''
        num_latent_frames = self.video_tokenizer.get_latent_num_frames(num_pixel_frames)
        return num_latent_frames

    @torch.no_grad()
    def _encode_raw(self, raw_video):
        '''
        :param raw_video: (B, 3, Tp, Hp, Wp) tensor of float in [-1, 1].
        :return latent_video: (B, 16, Tl, Hl, Wl) tensor of bfloat16.
        '''
        (B, Cp, Tp, Hp, Wp) = raw_video.shape
        assert Cp == 3, \
            f'Raw video must have 3 channels, but got {Cp}'
        assert (Tp - 1) % self.vae_ratio[0] == 0, \
            f'raw Tp - 1 must be divisible by temporal_compression_factor: ' \
            f'{self.vae_ratio[0]}, but got {Tp}'
        assert Tp <= TOKENIZER_CHUNK_DURATION, \
            f'Single-chunk clips only: Tp must be <= {TOKENIZER_CHUNK_DURATION}, but got {Tp}'

        raw_video = raw_video.to(**self.tensor_kwargs, non_blocking=True)

        latent_video = self.video_tokenizer.encode(raw_video)
        latent_video = latent_video * SIGMA_DATA

        return latent_video

    @torch.no_grad()
    def encode(self, raw_video):
        '''
        :param raw_video: (B, 3, Tp, Hp, Wp) tensor of float in [0, 1] with RGB pixels.
        :return latent_video: (B, 16, Tl, Hl, Wl) tensor of bfloat16.
        '''
        assert raw_video.shape[1] == 3, 'RGB video must have 3 channels'

        if raw_video.isnan().any():
            nan_per_batch = raw_video.isnan().flatten(1).any(dim=1)
            log.error(f'NaN in RAW RGB before VAE encode! '
                      f'shape={raw_video.shape}, '
                      f'affected batch elements: {nan_per_batch.nonzero().flatten().tolist()}',
                      rank0_only=False)
            raw_video = raw_video.nan_to_num(0.0)

        raw_video = raw_video * 2.0 - 1.0  # Now values are in [-1, 1]
        latent_video = self._encode_raw(raw_video)
        return latent_video

    @torch.no_grad()
    def _decode_raw(self, latent_video):
        '''
        :param latent_video: (B, 16, Tl, Hl, Wl) tensor of float.
        :return raw_video: (B, 3, Tp, Hp, Wp) tensor of float in [-1, 1], same dtype as input.
        '''
        latent_video = latent_video / SIGMA_DATA
        raw_video = self.video_tokenizer.decode(latent_video)
        return raw_video

    @torch.no_grad()
    def decode(self, latent_video):
        '''
        :param latent_video: (B, 16, Tl, Hl, Wl) tensor of float.
        :return pixel_video: (B, 3, Tp, Hp, Wp) tensor of float in [0, 1] with RGB pixels.
        '''
        latent_video = latent_video.clone().detach()

        pixel_video = self._decode_raw(latent_video)  # Returned values are in [-1, 1]
        pixel_video = (pixel_video + 1.0) / 2.0  # Now values are in [0, 1]
        pixel_video = pixel_video.clamp(0.0, 1.0)

        return pixel_video


@torch.no_grad()
def cached_decode_rgb(my_dict, key, vae):
    '''
    Decodes the RGB latent entry my_dict[key] once, stores the pixels in the dict under
    key + '_cdec', and retrieves them automatically upon subsequent calls.
    :param key (str): rgb0 / rgb1.
    :return pixel_video: (B?, 3, Tp, Hp, Wp) tensor of float in [0, 1].
    '''
    if key + '_cdec' in my_dict:
        return my_dict[key + '_cdec']

    latent_video = my_dict[key]
    # ^ (B?, 16, Tl, Hl, Wl) tensor of float.

    has_batch_dim = (latent_video.ndim == 5)
    if not has_batch_dim:
        latent_video = latent_video[None]

    pixel_video = vae.decode(latent_video)
    # ^ (B, 3, Tp, Hp, Wp) tensor of float in [0, 1].

    if not has_batch_dim:
        pixel_video = pixel_video[0]

    my_dict[key + '_cdec'] = pixel_video  # on GPU
    return my_dict[key + '_cdec']


def load_vae(tokenizer_path, device='cuda'):
    '''
    Factory used by the scripts.
    '''
    vae = AnyViewVAE(tokenizer_path, device=str(device))
    return vae


# Script-facing aliases (the scripts speak encode_rgb / decode_rgb / encode_cams).
AnyViewVAE.encode_rgb = AnyViewVAE.encode
AnyViewVAE.decode_rgb = AnyViewVAE.decode


def _encode_cams(self, plucker_video):
    '''
    :param plucker_video: (B, 6, Tp, Hp, Wp) float in [-1, 1]; rays 0:3, cross 3:6.
    :return latent: (B, 32, Tl, Hl, Wl) -- each 3-ch half encoded separately, concatenated.
    '''
    assert plucker_video.shape[1] == 6, 'plucker video must have 6 channels (rays, cross)'
    latent_rays = self._encode_raw(plucker_video[:, 0:3])
    latent_cross = self._encode_raw(plucker_video[:, 3:6])
    latent = torch.cat([latent_rays, latent_cross], dim=1)
    return latent


AnyViewVAE.encode_cams = _encode_cams
