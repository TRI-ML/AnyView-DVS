# All fixed constants of AnyView-DVS: stream layout, 2B DiT dims, sampling, paths.

'''
Frozen configuration for AnyView-DVS inference.

Every field is a constant of the released system: the 2-view 1->1 dynamic view synthesis task,
the 52-channel stream layout, the 2B DiT dimensions, and the diffusion sampling settings.
Only the asset paths are expected to vary per machine.
'''

from dataclasses import dataclass


@dataclass(frozen=True)
class AnyViewConfig:
    '''
    Constants shared by network / logistics / pipe / vae / cameras / avb_dataset.
    View 1 carries the input video; view 0 is the generated target view.
    '''

    # ---- Views and stream layout --------------------------------------------------------------
    # Each view v is one stream 'v{v}' of shape (B, 52, T_lat, H_lat, W_lat).
    num_views: int = 2
    stream_channels: int = 52
    rgb_channels: tuple[int, int] = (0, 16)     # rgb latent (VAE, 16 ch)
    rgb_input_mask_channel: int = 16
    rgb_output_mask_channel: int = 17
    cams_channels: tuple[int, int] = (18, 50)   # Plucker embedding (32 ch)
    cams_input_mask_channel: int = 50
    cams_output_mask_channel: int = 51

    # Constant DVS masks, indexed by view (v0, v1). Cameras of both views are always given as
    # input and never predicted; rgb is input-only on view 1 and output-only on view 0.
    rgb_input_mask_value: tuple[float, float] = (0.0, 1.0)
    rgb_output_mask_value: tuple[float, float] = (1.0, 0.0)
    cams_input_mask_value: tuple[float, float] = (1.0, 1.0)
    cams_output_mask_value: tuple[float, float] = (0.0, 0.0)

    # ---- Frames and resolution ----------------------------------------------------------------
    num_frames: int = 41            # training clip length; inference takes any T = 1 + 4k
    state_t: int = 11               # latent frames = 1 + (num_frames - 1) / vae_temporal_factor
    vae_temporal_factor: int = 4
    vae_spatial_factor: int = 8     # H_lat = H_pix / 8, W_lat = W_pix / 8
    latent_channels: int = 16       # rgb latent channels (state_ch)
    resolution: int = 576           # short side at inference
    resolution_multiple: int = 16   # both pixel sides snapped to multiples of 16 (s16)
    fps: float = 10.0               # conditioner fps key; inert (rope fps modulation is off)

    # ---- Diffusion sampling (values the eval experiment resolves; RectifiedFlowAB2Scheduler) ---
    num_steps: int = 35
    guidance: float = 0.0           # CFG scale; 0 = single conditional branch
    seed: int = 0                   # initial noise per stream: arch_invariant_rand(seed * 100 + stream_index) * sigma_max
    sigma_max: float = 80.0
    sigma_min: float = 0.002
    scheduler_order: float = 7.0    # sigma schedule rho
    t_scaling_factor: float = 1.0   # rectified_flow_t_scaling_factor
    sigma_data: float = 1.0
    sigma_conditional: float = 0.0001  # noise level stamped on conditioning (input-masked) content
    cond_aug_sigma: float = 0.0     # conditioning augmentation noise; disabled

    # ---- Network: 2B DiT dims (verbatim from the cosmos-predict2 2B video2world net config) ----
    model_channels: int = 2048      # hidden size
    num_blocks: int = 28
    num_heads: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    # The pretrained Cosmos base operates on 16-channel rgb latents; each per-view PatchEmbed /
    # FinalLayer is widened to stream_channels (52) in the released checkpoint, with +1 input
    # channel for the constant zero padding mask (concat_padding_mask).
    in_channels: int = 16
    out_channels: int = 16
    concat_padding_mask: bool = True
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    atten_backend: str = 'minimal_a2a'
    pos_emb_cls: str = 'rope3d'
    pos_emb_learnable: bool = True
    pos_emb_interpolation: str = 'crop'
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 256
    rope_h_extrapolation_ratio: float = 3.0
    rope_w_extrapolation_ratio: float = 3.0
    rope_t_extrapolation_ratio: float = 1.0
    extra_per_block_abs_pos_emb: bool = False
    rope_enable_fps_modulation: bool = False

    # ---- Multi-view wiring (fixed; network.py bakes the matching legacy semantics: gate_mlp
    # gating and per-view t_embedder routing) ----------------------------------------------------
    video_concat_mode: str = 'view'         # views stay separate token sequences plus view embs
    video_proj_mode: str = 'per_view'       # separate PatchEmbed / FinalLayer per view
    view_timestep_mode: str = 'per_view'    # separate t_embedder per view
    view_emb_std: float = 0.06              # init std of the additive per-view embedding

    # ---- Text-free conditioning -----------------------------------------------------------------
    # No text encoder; cross-attention consumes a fixed precomputed T5 embedding of the
    # training-time default prompt, shape (1, text_emb_tokens, text_emb_channels).
    text_emb_tokens: int = 512
    text_emb_channels: int = 1024

    # ---- Asset paths (the only per-machine fields) ----------------------------------------------
    # DiT checkpoint: the released AnyView-DVS 2B weights (see README, Checkpoints).
    checkpoint_path: str = 'checkpoints/anyview_dvs_2b.pt'
    # Cosmos video tokenizer: nvidia/Cosmos-Predict2-2B-Video2World tokenizer.pth
    # (HuggingFace: nvidia/Cosmos-Predict2-2B-Video2World, checkpoints/.../tokenizer/tokenizer.pth).
    tokenizer_path: str = 'checkpoints/tokenizer.pth'
    # Default-prompt T5 embedding, saved via torch.save as
    # {'t5_text_embeddings': (1, 512, 1024), 't5_text_mask': (1, 512)}.
    text_emb_path: str = 'checkpoints/default_text_emb.pt'
