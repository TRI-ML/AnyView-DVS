# AnyView DiT: the two-view diffusion transformer, loaded from the released checkpoint without key remapping.
# snapshot (legacy_network_behavior=2); the officialHR2v state_dict loads with zero key remapping.

from typing import Optional

import torch
from einops import rearrange
from torch import nn

from .config import AnyViewConfig
from .vendor.action_video2world_dit import ActionConditionedMinimalV1LVGDiT
from .vendor.text2image_dit import (
    Block,
    FinalLayer,
    PatchEmbed,
    SACConfig,
    TimestepEmbedding,
    Timesteps,
)

# Fixed AnyView design: 2 views, 52 channels per stream
# ([0:16] rgb latent, [16:18] rgb in/out masks, [18:50] Plucker cams, [50:52] cams in/out masks).
NUM_VIEWS = 2
STREAM_CHANNELS = 52
VIEW_EMB_STD = 0.06  # init only; overwritten by the checkpoint


class AnyViewBlock(Block):
    '''
    Transformer block over both view streams: per-view AdaLN modulation and MLP,
    shared self- and cross-attention on the concatenated token sequences.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_dim = kwargs['rope_dim']  # 128
        # Unused at inference (AnyView has no sattn stream), but the trained checkpoint
        # carries blocks.N.sattn_rope, so the parameter must exist for key parity.
        self.sattn_rope = torch.nn.Parameter(torch.randn(1, 1, 1, self.rope_dim))

    def forward(
        self,
        x_B_T_H_W_D: list[torch.Tensor],  # per view: (B, T, H, W, D) bf16
        emb_B_T_D: list[torch.Tensor],  # per view: (B, T, D) bf16
        crossattn_emb: torch.Tensor,  # (B, NC, D) bf16
        rope_emb_L_1_1_D: list[torch.Tensor],  # per view: (T*H*W, 1, 1, D) f32
        adaln_lora_B_T_3D: list[torch.Tensor],  # per view: (B, T, D*3) bf16
        **leftover,
    ) -> list[torch.Tensor]:
        '''
        :return V_x: per view: (B, T, H, W, D) bf16.
        '''
        assert isinstance(x_B_T_H_W_D, list), 'AnyViewBlock expects per-view lists of tensors'

        cat_rope_emb_L_1_1_D = torch.cat(rope_emb_L_1_1_D, dim=0)  # (V*T*H*W, 1, 1, D) f32

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        V_used = len(x_B_T_H_W_D)
        V_x = list(x_B_T_H_W_D)  # shallow copy
        V_normalized_x = []
        V_gate_next = []
        V_shapes = []
        V_seqlens = [0]

        # ================================ SELF-ATTENTION

        for v in range(V_used):
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D[v]) + adaln_lora_B_T_3D[v]
            ).chunk(3, dim=-1)

            shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

            normalized_x_B_T_H_W_D = _fn(
                V_x[v],
                self.layer_norm_self_attn,
                scale_self_attn_B_T_1_1_D,
                shift_self_attn_B_T_1_1_D
            )

            normalized_x_B_L_D = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            (T, H, W) = V_x[v].shape[1:4]

            V_normalized_x.append(normalized_x_B_L_D)
            V_shapes.append((T, H, W))
            V_seqlens.append(V_seqlens[v] + T * H * W)
            V_gate_next.append(gate_self_attn_B_T_1_1_D)

        # Both streams attend jointly over the concatenated token sequence.
        cat_normalized_x_B_L_D = torch.cat(V_normalized_x, dim=1)  # (B, V*T*H*W, D) bf16
        cat_result_B_L_D = self.self_attn(
            cat_normalized_x_B_L_D,
            None,
            rope_emb=cat_rope_emb_L_1_1_D,
        )

        # ================================ CROSS-ATTENTION

        for v in range(V_used):
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D[v]) + adaln_lora_B_T_3D[v]
            ).chunk(3, dim=-1)

            shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

            result_B_L_D = cat_result_B_L_D[:, V_seqlens[v]:V_seqlens[v + 1]]  # (B, T*H*W, D) bf16

            (T, H, W) = V_shapes[v]
            result_B_T_H_W_D = rearrange(result_B_L_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)

            # this is gate_self_attn_B_T_1_1_D
            V_x[v] = V_x[v] + V_gate_next[v] * result_B_T_H_W_D

            normalized_x_B_T_H_W_D = _fn(
                V_x[v],
                self.layer_norm_cross_attn,
                scale_cross_attn_B_T_1_1_D,
                shift_cross_attn_B_T_1_1_D
            )

            normalized_x_B_L_D = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")

            V_normalized_x[v] = normalized_x_B_L_D
            V_gate_next[v] = gate_cross_attn_B_T_1_1_D

        cat_normalized_x_B_L_D = torch.cat(V_normalized_x, dim=1)  # (B, V*T*H*W, D) bf16
        cat_result_B_L_D = self.cross_attn(
            cat_normalized_x_B_L_D,
            crossattn_emb,
            rope_emb=cat_rope_emb_L_1_1_D,
        )

        # ================================ MLP

        for v in range(V_used):
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D[v]) + adaln_lora_B_T_3D[v]
            ).chunk(3, dim=-1)

            shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
            scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
            gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

            result_B_L_D = cat_result_B_L_D[:, V_seqlens[v]:V_seqlens[v + 1]]  # (B, T*H*W, D) bf16

            (T, H, W) = V_shapes[v]
            result_B_T_H_W_D = rearrange(result_B_L_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)

            # NOTE: residual-gate variant (block_gate_fix=False) matching the released checkpoint's
            # training runs: gate_mlp gates the cross-attn residual here and the mlp residual
            # below (gate_cross_attn is unused).
            V_x[v] = V_x[v] + gate_mlp_B_T_1_1_D * result_B_T_H_W_D

            normalized_x_B_T_H_W_D = _fn(
                V_x[v],
                self.layer_norm_mlp,
                scale_mlp_B_T_1_1_D,
                shift_mlp_B_T_1_1_D
            )

            # mlp = GPT2FeedForward (linear layers only), so per-view application is exact.
            result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)

            V_x[v] = V_x[v] + gate_mlp_B_T_1_1_D * result_B_T_H_W_D

        return V_x


class AnyViewDiT(ActionConditionedMinimalV1LVGDiT):
    '''
    2-view diffusion transformer for 1->1 dynamic view synthesis.
    Both view streams (v0 = generated target, v1 = input video) are patch-embedded per view,
    then processed jointly by AnyViewBlocks, then projected back to stream channels per view.
    Module attribute names match the Any4D training code so the released checkpoint
    (officialHR2v) loads without any key remapping.
    '''

    def __init__(self, config: AnyViewConfig = None, sac_mode: str = 'none'):
        '''
        :param sac_mode (str): activation checkpointing mode ('none' for inference; training
            uses the cosmos-predict2 2B setting, see scripts/train_dvs.py). Parameters and
            state-dict keys do not depend on it.
        '''
        # Cosmos-Predict2-2B video2world hyperparameters, fixed to match the released checkpoint.
        super().__init__(
            max_img_h=240,
            max_img_w=240,
            max_frames=128,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=True,
            model_channels=2048,
            num_blocks=28,
            num_heads=16,
            atten_backend='minimal_a2a',
            pos_emb_cls='rope3d',
            pos_emb_learnable=True,
            pos_emb_interpolation='crop',
            use_adaln_lora=True,
            adaln_lora_dim=256,
            rope_h_extrapolation_ratio=3.0,
            rope_w_extrapolation_ratio=3.0,
            rope_t_extrapolation_ratio=1.0,
            extra_per_block_abs_pos_emb=False,
            rope_enable_fps_modulation=False,
            sac_config=SACConfig(mode=sac_mode),
            # in/out projections cover the full 52-channel stream layout
            # (masks included; mask output channels are ignored on unpack):
            in_channels_override=STREAM_CHANNELS,
            out_channels_override=STREAM_CHANNELS,
            block_cls=AnyViewBlock,
            block_gate_fix=False,
            action_dim=0,  # no action conditioning; skips action embedder creation
        )
        # ^ this initializes x_embedder, pos_embedder, t_embedder, blocks, final_layer,
        # and t_embedding_norm (all for view 0 / shared).

        self.config = config

        # View 1 gets its own input projection, view embedding, timestep embedder, and
        # output projection ("newviews" = all views beyond the pretrained view 0).
        self.x_embedder_newviews = nn.ModuleList([
            PatchEmbed(
                spatial_patch_size=self.patch_spatial,  # = 2
                temporal_patch_size=self.patch_temporal,  # = 1
                in_channels=STREAM_CHANNELS,
                out_channels=self.model_channels,  # = 2048
            ) for _ in range(NUM_VIEWS - 1)
        ])

        # Unique viewpoint identifier, added to all view-1 tokens after input projection.
        self.view_embs_newviews = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.model_channels) * VIEW_EMB_STD)
            for _ in range(NUM_VIEWS - 1)
        ])

        self.t_embedder_newviews = nn.ModuleList([
            nn.Sequential(
                Timesteps(self.model_channels),
                TimestepEmbedding(self.model_channels, self.model_channels, use_adaln_lora=self.use_adaln_lora),
            ) for _ in range(NUM_VIEWS - 1)
        ])

        self.final_layer_newviews = nn.ModuleList([
            FinalLayer(
                hidden_size=self.model_channels,
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                out_channels=STREAM_CHANNELS,
                use_adaln_lora=self.use_adaln_lora,  # = True
                adaln_lora_dim=self.adaln_lora_dim,  # = 256
            ) for _ in range(NUM_VIEWS - 1)
        ])

    def forward(
        self,
        x_in_streams: dict[str, torch.Tensor],
        timesteps: dict[str, torch.Tensor],
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        **leftover,
    ) -> dict[str, torch.Tensor]:
        '''
        :param x_in_streams: maps 'v0' / 'v1' to (B, 52, T_lat, H_lat, W_lat) tensor.
        :param timesteps: maps 'v0' / 'v1' to (B, 1, 1 or T_lat, 1, 1) noise level tensor.
        :param crossattn_emb: (B, NC, 1024) text-free conditioning embedding.
        :param fps: (B) tensor; unused by rope (fps modulation disabled) but kept for parity.
        :return y_out_streams: maps 'v0' / 'v1' to (B, 52, T_lat, H_lat, W_lat) tensor.
        '''
        assert 'v0' in x_in_streams and 'v1' in x_in_streams, \
            f'AnyViewDiT expects exactly streams v0 and v1, got {sorted(x_in_streams.keys())}'

        my_x_embedders = [self.x_embedder, *self.x_embedder_newviews]
        my_final_layers = [self.final_layer, *self.final_layer_newviews]

        # Per-view input projection (patch embed), plus view embedding for view 1.
        x_proj_all = []
        timesteps_all = []

        for v in range(NUM_VIEWS):
            x_in_v = x_in_streams[f'v{v}']  # (B, C, T, H, W)
            timesteps_v = timesteps[f'v{v}'][:, 0, :, 0, 0]  # (B, 1 or T)

            x_proj_v = my_x_embedders[v](x_in_v)  # (B, T, H, W, D) bf16
            if v >= 1:
                x_proj_v = x_proj_v + self.view_embs_newviews[v - 1]

            T = x_proj_v.shape[1]
            if timesteps_v.shape[1] == 1:  # convert (B, 1) to (B, T) as needed
                timesteps_v = timesteps_v.repeat(1, T)

            x_proj_all.append(x_proj_v)
            timesteps_all.append(timesteps_v)

        # Timestep-embedding routing kept exactly as in training: every view after view 0 uses the
        # embedder of the last added view, which is what the released checkpoint expects.
        view_t_embedder = self.t_embedder_newviews[NUM_VIEWS - 2]

        # Positional (rope), diffusion timestep, and adaln embeddings, per view.
        rope_emb_all = []
        t_embedding_all = []
        t_emb_norm_all = []
        adaln_lora_all = []

        for v in range(NUM_VIEWS):
            rope_emb_v = self.pos_embedder(x_proj_all[v], fps=fps)  # (T*H*W, 1, 1, D) f32
            (t_embedding_v, adaln_lora_v) = view_t_embedder(timesteps_all[v])
            # ^ (B, T, D) bf16, (B, T, D*3) bf16
            t_emb_norm_v = self.t_embedding_norm(t_embedding_v)  # (B, T, D) bf16

            rope_emb_all.append(rope_emb_v)
            t_embedding_all.append(t_embedding_v)
            t_emb_norm_all.append(t_emb_norm_v)
            adaln_lora_all.append(adaln_lora_v)

        # Apply transformer blocks over both streams jointly.
        x_block_all = x_proj_all

        for block in self.blocks:
            x_block_all = block(
                x_B_T_H_W_D=x_block_all,  # per view: (B, T, H, W, D) bf16
                emb_B_T_D=t_emb_norm_all,  # per view: (B, T, D) bf16
                crossattn_emb=crossattn_emb,  # (B, NC, D) bf16
                rope_emb_L_1_1_D=rope_emb_all,  # per view: (T*H*W, 1, 1, 128) f32
                adaln_lora_B_T_3D=adaln_lora_all,  # per view: (B, T, D*3) bf16
            )

        # Per-view output projection back to the stream channel layout.
        y_out_streams = dict()

        for v in range(NUM_VIEWS):
            y_proj_v = my_final_layers[v](x_block_all[v], t_embedding_all[v], adaln_lora_B_T_3D=adaln_lora_all[v])
            y_proj_v = self.unpatchify(y_proj_v)  # (B, 52, T, H, W)
            y_out_streams[f'v{v}'] = y_proj_v

        return y_out_streams
