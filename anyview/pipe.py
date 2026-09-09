# AnyView diffusion pipeline: preconditioning, denoising, and sampling for the two-view model.
# sampling path (rectified flow, 2 fixed video streams, text-free conditioning).

from typing import Callable, Dict, Optional

import os

import torch

from .logistics import (
    assemble_clean_predictions,
    assemble_network_inputs,
    pack_streams_from_entries,
    unpack_entries_from_streams,
)
from .vendor.denoiser_scaling import RectifiedFlowScaling
from .vendor.misc import arch_invariant_rand
from .vendor.rectified_flow_scheduler import RectifiedFlowAB2Scheduler

# Sampling constants (match the released checkpoint's eval protocol).
DEFAULT_NUM_STEPS = 35
import re

SAC_WRAP = re.compile(r'\._checkpoint_wrapped_module')  # key segment added by activation checkpointing

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
SIGMA_ORDER = 7.0
SIGMA_DATA = 1.0
T_SCALING_FACTOR = 1.0

# The DiT was trained with T5-11B cross-attention; text-free inference conditions on the
# cached T5 embedding of the empty prompt "" (shape (1, 512, 1024) float32; only the first
# token position, the EOS token, is nonzero). This is NOT an all-zeros tensor.
T5_EMB_KEY = 't5_text_embeddings'


def load_empty_text_emb(path: str) -> torch.Tensor:
    '''
    Loads the cached empty-prompt T5 embedding used for text-free conditioning.
    :param path (str): torch.save file holding {'t5_text_embeddings': (1, 512, 1024) float32}
        (or the raw tensor); loaded with weights_only=True, so only tensors are accepted.
    :return emb (Tensor): (1, L, D) tensor of float.
    '''
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        emb = data[T5_EMB_KEY]
    else:
        emb = data
    assert torch.is_tensor(emb) and emb.ndim == 3 and emb.shape[0] == 1, \
        f'Expected (1, L, D) empty text embedding, got {getattr(emb, "shape", type(emb))}'
    return emb


def scheduler_step_streams(
    scheduler: RectifiedFlowAB2Scheduler,
    x0_pred_streams: dict[str, torch.Tensor],
    i: int,
    sample_streams: dict[str, torch.Tensor],
    x0_prev_streams: Optional[dict[str, torch.Tensor]] = None,
):
    '''
    Two step Adams-Bashforth (2-AB) evaluation in Rectified Flow form, applied separately per
    stream via RectifiedFlowAB2Scheduler.step().
    :return (sample_streams, x0_prev_streams): both dicts mapping stream name to tensor.
    '''
    keys = sample_streams.keys()
    if x0_prev_streams is None:
        x0_prev_streams = {k: None for k in keys}

    for k in keys:
        (sample_streams[k], x0_prev_streams[k]) = scheduler.step(
            x0_pred=x0_pred_streams[k],
            i=i,
            sample=sample_streams[k],
            x0_prev=x0_prev_streams[k],
        )

    return (sample_streams, x0_prev_streams)


class AnyViewPipeline:
    '''
    Diffusion sampling pipeline for 1->1 dynamic view synthesis. Consumes latent entries
    (see logistics.build_dvs_entries), runs the rectified flow sampling loop on the two
    packed streams (v0 = generated target view, v1 = input view), and returns the predicted
    clean streams and entries (rgb0 latent = generated view).
    '''

    def __init__(
        self,
        dit: torch.nn.Module,
        empty_text_emb: torch.Tensor,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        '''
        :param dit (nn.Module): AnyView DiT (see network.py), already on device in eval mode.
        :param empty_text_emb (Tensor): (1, L, D) cached T5 embedding of the empty prompt
            (see load_empty_text_emb).
        '''
        self.dit = dit
        self.tensor_kwargs = {'device': device, 'dtype': dtype}
        self.empty_text_emb = empty_text_emb.to(**self.tensor_kwargs)

        self.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            order=SIGMA_ORDER,
            t_scaling_factor=T_SCALING_FACTOR,
        )
        self.scaling = RectifiedFlowScaling(SIGMA_DATA, T_SCALING_FACTOR)

    def denoise(
        self,
        x0_streams: dict[str, torch.Tensor],
        yt_streams: dict[str, torch.Tensor],
        masks: dict[str, dict[str, torch.Tensor]],
        sigmas: dict[str, torch.Tensor],
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
    ):
        '''
        Single denoising step: preconditions, assembles masked network inputs, runs the DiT,
        and converts the network output to clean sample predictions.
        :param x0_streams (dict): Maps stream name to (B, C, T, H, W) tensor of float.
        :param yt_streams (dict): Maps stream name to (B, C, T, H, W) tensor of float.
        :param masks (dict): Maps 'input' / 'output' / ... to stream name to tensor.
        :param sigmas (dict): Maps stream name to (B, 1, 1, 1, 1) tensor of float.
        :param crossattn_emb (Tensor): (B, L, D) text conditioning tensor.
        :param fps (Tensor): (B) tensor; inert for this checkpoint (rope fps modulation is off).
        :return xt_assemb_streams (dict): Network input streams (masked combine).
        :return y0_pred_streams (dict): Predicted clean streams (masked combine).
        '''
        (cs_skip, cs_out, cs_in, cs_noise_pred) = (dict(), dict(), dict(), dict())

        for (k, v) in sigmas.items():
            sigma = v
            assert sigma.ndim == yt_streams[k].ndim, \
                f'sigma for stream {k} must be broadcast-shaped, got {sigma.shape}'
            (cs_skip[k], cs_out[k], cs_in[k], cs_noise_pred[k]) = self.scaling(sigma=sigma)
            # ^ values = (B, 1, T=1, 1, 1) tensors of float.

        # Preconditions always refer to outputs; conditioning frames carry no special noise
        # level (force_cond_noise_level=False semantics of the released checkpoint).
        cs_noise_assemb = cs_noise_pred

        # Scale the noisy observations.
        yt_scaled_streams = {k: cs_in[k] * v for (k, v) in yt_streams.items()}

        # Create the masked network input tensors by combining conditioning inputs and scaled
        # noisy target values.
        xt_assemb_streams = assemble_network_inputs(x0_streams, yt_scaled_streams, masks)

        # The whole network must operates in bfloat16 precision.
        xt_assemb_streams = {k: v.to(**self.tensor_kwargs) for (k, v) in xt_assemb_streams.items()}
        cs_noise_assemb = {k: v.to(**self.tensor_kwargs) for (k, v) in cs_noise_assemb.items()}

        net_output = self.dit(
            x_in_streams=xt_assemb_streams,
            timesteps=cs_noise_assemb,
            crossattn_emb=crossattn_emb,
            fps=fps,
        )

        # Calculate predicted clean samples based on the rectified flow parameterization.
        y0_raw_pred_streams = dict()

        for k in net_output.keys():
            net_output[k] = net_output[k].to(torch.float32)
            y0_raw_pred = cs_skip[k] * yt_streams[k] + cs_out[k] * net_output[k]
            y0_raw_pred_streams[k] = y0_raw_pred

        # Combine non-augmented conditioning inputs with predicted outputs.
        y0_pred_streams = assemble_clean_predictions(x0_streams, y0_raw_pred_streams, masks)

        return (xt_assemb_streams, y0_pred_streams)

    def create_y0_pred_fn(
        self,
        x0_streams: Dict[str, torch.Tensor],
        masks: Dict[str, Dict[str, torch.Tensor]],
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
    ) -> Callable:
        '''
        Returns the per-step sampling iteration function. Guidance is fixed to 0.0, so the
        classifier-free guidance branch of the source is dead and only the conditional
        denoise call remains.
        '''
        def y0_pred_fn(yt_streams: Dict[str, torch.Tensor], sigmas: Dict[str, torch.Tensor]):
            (xt_streams, y0_pred_streams) = self.denoise(
                x0_streams, yt_streams, masks, sigmas, crossattn_emb, fps=fps)
            return (xt_streams, y0_pred_streams)

        return y0_pred_fn

    @torch.no_grad()
    def generate(
        self,
        latent_entries: Dict[str, torch.Tensor],
        seed: int = 0,
        num_sampling_steps: int = DEFAULT_NUM_STEPS,
        sigma_max: Optional[float] = None,
        sigma_min: Optional[float] = None,
        fps: Optional[torch.Tensor] = None,
    ) -> Dict:
        '''
        Runs the full sampling loop for one batch of latent entries.
        :param latent_entries (dict): Latent entries from logistics.build_dvs_entries
            (rgb0/rgb1 latents, cams0/cams1 Plucker maps, constant masks).
        :param seed (int): Sampling seed; the starting noise of stream index v uses
            seed * 100 + v (architecture-invariant numpy RNG).
        :param num_sampling_steps (int).
        :param sigma_max (float): Optional scheduler override (default 80.0).
        :param sigma_min (float): Optional scheduler override (default 0.002).
        :param fps (Tensor): Optional (B) tensor; inert for this checkpoint.
        :return samples_dict (dict): x0_streams (ground truth), masks, xt_streams (first
            network input), y0_pred_streams / y0_pred_entries (prediction; rgb0 latent is the
            generated view), sampling_params.
        '''
        if sigma_max is not None and sigma_max > 0.0:
            self.scheduler.config.sigma_max = sigma_max
        if sigma_min is not None and sigma_min > 0.0:
            self.scheduler.config.sigma_min = sigma_min

        # Assemble all inputs, outputs, and masks.
        (x0_streams, masks) = pack_streams_from_entries(latent_entries)
        assert list(x0_streams.keys()) == ['v0', 'v1'], \
            f'Expected exactly streams [v0, v1], got {list(x0_streams.keys())} ' \
            f'(order matters for seed parity)'

        B = x0_streams['v0'].shape[0]
        device = x0_streams['v0'].device
        state_shapes = {k: v.shape[1:] for (k, v) in x0_streams.items()}

        # Text-free conditioning: cached T5 embedding of the empty prompt, repeated per sample.
        crossattn_emb = self.empty_text_emb.repeat(B, 1, 1)

        y0_pred_fn = self.create_y0_pred_fn(x0_streams, masks, crossattn_emb, fps=fps)

        # Starting noise per stream; the v-th stream uses seed * 100 + v.
        yt_noise_start_streams = {k:
            arch_invariant_rand(
                (B,) + tuple(state_shapes[k]),
                torch.float32,
                self.tensor_kwargs['device'],
                seed * 100 + v,
            )
            * self.scheduler.config.sigma_max for (v, k) in enumerate(state_shapes.keys())}

        # Sampling loop driven by RectifiedFlowAB2Scheduler:
        # construct sigma schedule (L + 1 entries including sigma_min) and timesteps.
        self.scheduler.set_timesteps(num_sampling_steps, device=device)

        sample_streams = {k: yt_noise_start_streams[k].to(dtype=torch.float32) for k in state_shapes}
        first_xt_streams = None
        y0_prev_streams = None

        for (i, _) in enumerate(self.scheduler.timesteps):
            # Current noise level (sigma_t).
            sigma_t = self.scheduler.sigmas[i].to(device, dtype=torch.float32)  # single float.
            sigma_in = {k: sigma_t.repeat(B, *([1] * len(state_shapes[k]))) for k in state_shapes}
            # ^ dict mapping stream name to (B, 1, 1, 1, 1) tensor of float.

            (xt_streams, y0_pred_streams) = y0_pred_fn(sample_streams, sigma_in)
            if first_xt_streams is None:
                first_xt_streams = xt_streams

            # Scheduler step updates the noisy sample and returns the cached y0.
            (sample_streams, y0_prev_streams) = scheduler_step_streams(
                self.scheduler,
                x0_pred_streams=y0_pred_streams,
                i=i,
                sample_streams=sample_streams,
                x0_prev_streams=y0_prev_streams)

        # Final clean pass at sigma_min.
        sigma_last = self.scheduler.sigmas[-1].to(device, dtype=torch.float32)  # single float.
        sigma_in = {k: sigma_last.repeat(B, *([1] * len(state_shapes[k]))) for k in state_shapes}
        (xt_streams, sample_streams) = y0_pred_fn(sample_streams, sigma_in)

        # Combine non-augmented conditioning inputs with predicted outputs.
        y0_pred_streams = assemble_clean_predictions(x0_streams, sample_streams, masks)

        # Unpack predictions back into entries; rgb0 latent = the generated target view.
        y0_pred_entries = unpack_entries_from_streams(y0_pred_streams)

        samples_dict = {
            'x0_streams': x0_streams,  # ground truth
            'masks': masks,
            'xt_streams': first_xt_streams,  # input (with full / starting noise)
            'y0_pred_streams': y0_pred_streams,  # prediction
            'y0_pred_entries': y0_pred_entries,  # prediction as entries (rgb0 = generated)
            ####
            'sampling_params': {
                'num_steps': num_sampling_steps,
                'guidance': 0.0,
                'seed': seed,
                'sigma_max': self.scheduler.config.sigma_max,
                'sigma_min': self.scheduler.config.sigma_min,
                'timesteps': self.scheduler.timesteps,
                'sigmas': self.scheduler.sigmas,
            },
        }

        return samples_dict


def load_dit(config, sac_mode='none'):
    '''
    Builds the DiT on cpu (fp32) and loads the checkpoint strictly (net. prefix stripped).
    '''
    from .network import AnyViewDiT

    dit = AnyViewDiT(config, sac_mode=sac_mode)
    state = torch.load(config.checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    # Training checkpoints store parameters under a net. prefix.
    state = {(k[4:] if k.startswith('net.') else k): v for (k, v) in state.items()}
    # With activation checkpointing the blocks are wrapped and their keys gain
    # '_checkpoint_wrapped_module.'; checkpoints are always stored in the plain layout.
    module_keys = list(dit.state_dict().keys())
    plain_to_module = {SAC_WRAP.sub('', mk): mk for mk in module_keys}
    mapped = {plain_to_module[k]: v for (k, v) in state.items() if k in plain_to_module}
    # Source keys that map to no module parameter are unexpected, except the fused-attention
    # bookkeeping buffers (_extra_state) that the training framework stored and this code
    # does not use.
    unexpected = [k for k in state if k not in plain_to_module and not k.endswith('_extra_state')]
    (missing, unexpected_torch) = dit.load_state_dict(mapped, strict=False)
    unexpected = sorted(set(unexpected) | set(unexpected_torch))
    missing = [SAC_WRAP.sub('', k) for k in missing]
    # blocks.N.sattn_rope buffers exist in the module for later-era key compatibility but
    # predate this checkpoint (unused in forward); everything else must match exactly.
    import re as _re
    benign = [k for k in missing if _re.fullmatch(r'blocks\.\d+\.sattn_rope', k)]
    missing = [k for k in missing if k not in benign]
    if missing or unexpected:
        raise RuntimeError(
            f'Checkpoint mismatch: missing={sorted(missing)[:8]} '
            f'unexpected={sorted(unexpected)[:8]} (counts {len(missing)}/{len(unexpected)})')
    return dit


def load_text_emb(config):
    '''
    Cached text-free conditioning embedding; relative paths resolve against the release root.
    '''
    text_emb_path = config.text_emb_path
    if not os.path.isabs(text_emb_path):
        release_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        text_emb_path = os.path.join(release_root, text_emb_path)
    empty_text_emb = load_empty_text_emb(text_emb_path)
    return empty_text_emb


def load_pipeline(config, device='cuda', dtype=torch.bfloat16):
    '''
    Factory used by the scripts: builds the DiT, loads the checkpoint strictly, loads the
    cached text-free embedding, and returns a ready AnyViewPipeline.
    '''
    dit = load_dit(config)
    dit = dit.to(device=device, dtype=dtype).eval()
    empty_text_emb = load_text_emb(config)
    pipeline = AnyViewPipeline(dit, empty_text_emb, device=str(device), dtype=dtype)
    return pipeline
