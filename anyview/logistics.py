# Entry <-> stream packing for the two fixed DVS streams (v0 target, v1 input), constant DVS masks.

from typing import Optional

import torch

# Latent stream layout, 52 channels per view; values are (channel_start, channel_end). The
# mask channels carry the mask VALUES as data and always count as inputs.
RGB_LATENT_CHANNELS = 16
CAMS_CHANNELS = 32
NUM_STREAM_CHANNELS = 52
NUM_VIEWS = 2

VIDEO_ENTRIES = [
    dict(
        rgb0=(0, 16),
        rgb0_input_mask=(16, 17),
        rgb0_output_mask=(17, 18),
        cams0=(18, 50),
        cams0_input_mask=(50, 51),
        cams0_output_mask=(51, 52),
    ),
    dict(
        rgb1=(0, 16),
        rgb1_input_mask=(16, 17),
        rgb1_output_mask=(17, 18),
        cams1=(18, 50),
        cams1_input_mask=(50, 51),
        cams1_output_mask=(51, 52),
    ),
]

MASK_MODES = ('input', 'output', 'supervise')


@torch.no_grad()
def build_dvs_entries(
    rgb0_latent: Optional[torch.Tensor],
    rgb1_latent: torch.Tensor,
    cams0: torch.Tensor,
    cams1: torch.Tensor,
) -> dict:
    '''
    Assemble the entries dict for 1->1 dynamic view synthesis with the constant DVS masks:
    view 1 (rgb + cams) and view 0 cams are inputs; view 0 rgb is the generated output.
    :param rgb0_latent: (B, 16, Tl, H0, W0) target-view GT latent, or None (zeros). Its content
        never reaches the network (input mask 0); it is only used for eval visuals / metrics.
    :param rgb1_latent: (B, 16, Tl, H1, W1) input-view latent.
    :param cams0: (B, 32, Tl, H0, W0) target-view Plucker embedding at latent resolution.
    :param cams1: (B, 32, Tl, H1, W1) input-view Plucker embedding at latent resolution.
    :return entries (dict): latents + masks + video_dims, ready for pack_streams_from_entries.
    The two views may differ in spatial size (portrait input, landscape target); B and Tl agree.
    '''
    (B, C, Tl, H1, W1) = rgb1_latent.shape
    assert C == RGB_LATENT_CHANNELS, \
        f'rgb1_latent must have {RGB_LATENT_CHANNELS} channels, but got {C} instead.'
    assert cams0.shape[0:3] == (B, CAMS_CHANNELS, Tl), \
        f'cams0 must be (B, {CAMS_CHANNELS}, {Tl}, H0, W0), but got {tuple(cams0.shape)} instead.'
    (H0, W0) = cams0.shape[3:5]
    if rgb0_latent is None:
        rgb0_latent = torch.zeros((B, C, Tl, H0, W0), device=rgb1_latent.device,
                                  dtype=rgb1_latent.dtype)
    assert rgb0_latent.shape == (B, C, Tl, H0, W0), \
        f'rgb0_latent must be {(B, C, Tl, H0, W0)}, but got {tuple(rgb0_latent.shape)} instead.'
    assert cams1.shape == (B, CAMS_CHANNELS, Tl, H1, W1), \
        f'cams1 must be {(B, CAMS_CHANNELS, Tl, H1, W1)}, but got {tuple(cams1.shape)} instead.'
    dims = [(Tl, H0, W0), (Tl, H1, W1)]  # view 0 = target, view 1 = input

    tensor_kwargs = dict(device=rgb1_latent.device, dtype=torch.bfloat16)
    entries = dict()
    entries['rgb0'] = rgb0_latent.to(**tensor_kwargs)
    entries['rgb1'] = rgb1_latent.to(**tensor_kwargs)
    entries['cams0'] = cams0.to(**tensor_kwargs)
    entries['cams1'] = cams1.to(**tensor_kwargs)

    # Constant masks: rgb0 is the only output entry; supervise mask = output mask.
    for k in ['rgb0', 'rgb1', 'cams0', 'cams1']:
        input_val = 0.0 if k == 'rgb0' else 1.0
        (Tv, Hv, Wv) = dims[int(k[-1])]
        input_mask = torch.full((B, 1, Tv, Hv, Wv), input_val, **tensor_kwargs)
        output_mask = 1.0 - input_mask
        entries[k + '_input_mask'] = input_mask
        entries[k + '_output_mask'] = output_mask
        entries[k + '_supervise_mask'] = output_mask

    entries['video_dims'] = dims
    return entries


@torch.no_grad()
def pack_streams_from_entries(entries: dict) -> tuple:
    '''
    Assemble the two 52-channel latent streams (v0, v1) and their per-channel mask dicts.
    :param entries (dict): as produced by build_dvs_entries.
    :return (streams, masks): streams maps 'v0' / 'v1' to (B, 52, Tl, H, W); masks maps
        'input' / 'output' / 'supervise' to such a dict.
    '''
    B = entries['rgb0'].shape[0]
    tensor_kwargs = dict(device=entries['rgb0'].device, dtype=entries['rgb0'].dtype)

    streams = dict()
    masks = {mode: dict() for mode in MASK_MODES}
    for v in range(NUM_VIEWS):
        (Tv, Hv, Wv) = entries['video_dims'][v]
        stream = torch.zeros((B, NUM_STREAM_CHANNELS, Tv, Hv, Wv), **tensor_kwargs)
        for (k, (c0, c1)) in VIDEO_ENTRIES[v].items():
            stream[:, c0:c1] = entries[k]
        streams[f'v{v}'] = stream

        # Mask channels count as inputs; the data channels take the per-entry masks.
        for mode in MASK_MODES:
            mask = torch.full_like(stream, 1.0 if mode == 'input' else 0.0)
            for k in (f'rgb{v}', f'cams{v}'):
                (c0, c1) = VIDEO_ENTRIES[v][k]
                mask[:, c0:c1] = entries[f'{k}_{mode}_mask']
            masks[mode][f'v{v}'] = mask

    return (streams, masks)


def unpack_entries_from_streams(streams: dict) -> dict:
    '''
    Slice the entries (latents and mask channels) back out of the streams.
    '''
    entries = dict()
    for v in range(NUM_VIEWS):
        if f'v{v}' in streams:
            assert streams[f'v{v}'].ndim == 5, \
                f'video stream v{v} must be (B, Cl, Tl, Hl, Wl), but got {streams[f"v{v}"].shape} instead.'
            for (k, (c0, c1)) in VIDEO_ENTRIES[v].items():
                entries[k] = streams[f'v{v}'][:, c0:c1]
    return entries


@torch.no_grad()
def assemble_network_inputs(x0_streams: dict, yt_streams: dict, masks: dict) -> dict:
    '''
    Network input per stream: conditioning inputs where the input mask is set, (scaled) noisy
    values where the output mask is set.
    '''
    xt_streams = {k: x0_streams[k] * masks['input'][k] + yt_streams[k] * masks['output'][k]
                  for k in x0_streams}
    return xt_streams


def assemble_clean_predictions(x0_streams: dict, y0_raw_pred_streams: dict, masks: dict) -> dict:
    '''
    Final prediction per stream: the non-augmented conditioning inputs where the input mask is
    set, the predicted outputs where the output mask is set.
    '''
    y0_pred_streams = {k: x0_streams[k] * masks['input'][k] + y0_raw_pred_streams[k] * masks['output'][k]
                       for k in y0_raw_pred_streams}
    return y0_pred_streams
