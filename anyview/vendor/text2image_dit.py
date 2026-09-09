# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Vendored from NVIDIA cosmos-predict2 (Apache 2.0), trimmed.
'''

# NOTE: transformer_engine removed. RMSNorm = the pure-torch class below (same `weight`
# key as te.pytorch.RMSNorm), rope = local GPT-NeoX apply_rotary_pos_emb, attention backends =
# torch / minimal_a2a (both torch SDPA). Context parallel, CUDA graphs, FSDP, the single-view
# forward, and the unused FourierFeatures / LearnablePosEmbAxis are dropped; all parameter and
# buffer names are unchanged.

import collections
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from . import log
from .a2a_cp import MinimalA2AAttnOp
from .model_weights_stats import WeightTrainingStat
from .selective_activation_checkpoint import SACConfig as _SACConfig


# selective activation checkpoint; only apply to the minimal v4 model. if there are change in the networks, some policy will not work as we expect.
def predict2_2B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 3 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_14B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 1 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def linear_selfattn_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


class CheckpointMode(str, Enum):
    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"
    LINEAR_SELFATTN = "linear_selfattn"
    PREDICT2_2B_720 = "predict2_2b_720"
    PREDICT2_14B_720 = "predict2_14b_720"

    def __str__(self) -> str:
        return self.value


@dataclass
class SACConfig(_SACConfig):
    def get_context_fn(self):
        if self.mode == CheckpointMode.LINEAR_SELFATTN:
            return linear_selfattn_context_fn
        elif self.mode == CheckpointMode.PREDICT2_2B_720:
            return predict2_2B_720_context_fn
        elif self.mode == CheckpointMode.PREDICT2_14B_720:
            return predict2_14B_720_context_fn
        else:
            # Reuse parent class implementation for other modes
            return super().get_context_fn()


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: weight is applied before the single cast back to x.dtype; this matches
        # te.pytorch.RMSNorm bit-for-bit in bf16 (upstream torch variant cast first: 74% exact).
        output = self._norm(x.float()) * self.weight.float()
        return output.type_as(x)


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


def torch_attention_op(q_B_S_H_D: torch.Tensor, k_B_S_H_D: torch.Tensor, v_B_S_H_D: torch.Tensor) -> torch.Tensor:
    """Computes multi-head attention using PyTorch's native implementation.

    This function provides a PyTorch backend alternative to Transformer Engine's attention operation.
    It rearranges the input tensors to match PyTorch's expected format, computes scaled dot-product
    attention, and rearranges the output back to the original format.

    The input tensor names use the following dimension conventions:

    - B: batch size
    - S: sequence length
    - H: number of attention heads
    - D: head dimension

    Args:
        q_B_S_H_D: Query tensor with shape (batch, seq_len, n_heads, head_dim)
        k_B_S_H_D: Key tensor with shape (batch, seq_len, n_heads, head_dim)
        v_B_S_H_D: Value tensor with shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Attention output tensor with shape (batch, seq_len, n_heads * head_dim)
    """
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    result_B_S_HD = rearrange(
        torch.nn.functional.scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D), "b h ... l -> b ... (h l)"
    )

    return result_B_S_HD


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    (x1, x2) = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# NOTE: replaces transformer_engine's apply_rotary_pos_emb(t, freqs, tensor_format="bshd",
# fused=True). VideoRopePosition3DEmb emits (L, 1, 1, D) angles with both halves duplicated
# (non-interleaved GPT-NeoX layout), so out = t * cos + rotate_half(t) * sin over half-splits.
# Like the TE fused kernel, the math runs in fp32 and the result is cast back to t.dtype.
def apply_rotary_pos_emb(t_B_S_H_D: torch.Tensor, freqs_L_1_1_D: torch.Tensor) -> torch.Tensor:
    (_, S, _, D) = t_B_S_H_D.shape
    assert freqs_L_1_1_D.shape[0] >= S, f"rope table has {freqs_L_1_1_D.shape[0]} positions, need {S}"
    assert freqs_L_1_1_D.shape[-1] == D, f"rope dim {freqs_L_1_1_D.shape[-1]} != head dim {D}"
    freqs_1_S_1_D = freqs_L_1_1_D[:S].transpose(0, 1).float()
    cos_1_S_1_D = torch.cos(freqs_1_S_1_D)
    sin_1_S_1_D = torch.sin(freqs_1_S_1_D)
    t_f32 = t_B_S_H_D.float()
    out_f32 = t_f32 * cos_1_S_1_D + _rotate_half(t_f32) * sin_1_S_1_D
    out = out_f32.to(t_B_S_H_D.dtype)
    return out


class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "minimal_a2a"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = "minimal_a2a",
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None  # self attention

        assert backend in ["torch", "minimal_a2a"], f"Invalid backend: {backend}"
        assert qkv_format == "bshd", f"Only qkv_format='bshd' is supported, got {qkv_format}"
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        if self.backend == "minimal_a2a":
            self.attn_op = MinimalA2AAttnOp()
        elif self.backend == "torch":
            self.attn_op = torch_attention_op

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope_emb: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb)
                k = apply_rotary_pos_emb(k, rope_emb)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        result = self.attn_op(q, k, v)  # [B, S, H, D]
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)


class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor]) -> torch.Tensor:
        """
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        return self._split_for_context_parallel(embeddings)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> Any:
        raise NotImplementedError

    def _split_for_context_parallel(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Context parallelism is not vendored; the embeddings are returned unchanged.
        return embeddings


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,  # 128
        len_h: int,  # 120
        len_w: int,  # 120
        len_t: int,  # 128
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,  # 3.0
        w_extrapolation_ratio: float = 1.0,  # 3.0
        t_extrapolation_ratio: float = 1.0,  # 1.0
        enable_fps_modulation: bool = True,  # False
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps  # 24
        self.max_h = len_h  # 120
        self.max_w = len_w  # 120
        self.max_t = len_t  # 128
        self.enable_fps_modulation = enable_fps_modulation  # False
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h  # 42
        self._dim_t = dim_t  # 44

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))  # 3.169401925648614
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))  # 3.169401925648614
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))  # 1.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            (T*H*W, 1, 1, head_dim) float tensor of rope angles.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor  # type: ignore
        w_theta = 10000.0 * w_ntk_factor  # type: ignore
        t_theta = 10000.0 * t_ntk_factor  # type: ignore

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:  # no
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert (
                uniform_fps or B == 1 or T == 1
            ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"

            # apply sequence scaling in temporal dimension
            if fps is None:  # image case
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self) -> int:
        return 0


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size: int,  # = 2
        temporal_patch_size: int,  # = 1
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        # NOTE: This corresponds to patchify, opposite of unpatchify
        # (though only the latter is a separate method)
        # NOTE: this is POSITION MINOR and CHANNEL MAJOR
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert (
            H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        ), f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,  # = 2
        temporal_patch_size: int,  # = 1
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D, scale_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d"), rearrange(
            scale_B_T_D, "b t d -> b t 1 1 d"
        )

        def _fn(
            _x_B_T_H_W_D: torch.Tensor,
            _norm_layer: nn.Module,
            _scale_B_T_1_1_D: torch.Tensor,
            _shift_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        x_B_T_H_W_D = _fn(x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D)

        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)

        return x_B_T_H_W_O


class Block(nn.Module):
    """
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,  # 2048
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "minimal_a2a",
        **leftover,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend)

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,  # (B, T, H, W, D) bf16
        emb_B_T_D: torch.Tensor,  # (B, T, D) bf16
        crossattn_emb: torch.Tensor,  # (B, N, D) bf16
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,  # (T*H*W, 1, 1, D) f32
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,  # (B, T, D*3) bf16
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,  # None
    ) -> torch.Tensor:
        # Single-view block; per-view lists of tensors are handled by the AnyViewBlock subclass.
        assert not isinstance(x_B_T_H_W_D, list), "Block.forward expects a single tensor"

        if extra_per_block_pos_emb is not None:  # no
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:  # yes
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)

        else:  # no
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        result_B_T_H_W_D = rearrange(
            # NOTE: Flattening and unflattening happens here (per layer)!
            self.self_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ), "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )

        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D,
                layer_norm_cross_attn,
                _scale_cross_attn_B_T_1_1_D,
                _shift_cross_attn_B_T_1_1_D
            )

            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ), "b (t h w) d -> b t h w d", t=T, h=H, w=W,
            )

            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D

        return x_B_T_H_W_D


class MiniTrainDIT(WeightTrainingStat):
    """
    A clean impl of DIT that can load and  reproduce the training results of the original DIT model in~(cosmos 1)
    A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.

    Only the module construction is vendored (forward is provided by the AnyView subclass).

    Args:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (tuple): Spatial resolution of patches for input processing.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        pos_emb_cls (str): Type of positional embeddings.
        pos_emb_learnable (bool): Whether positional embeddings are learnable.
        pos_emb_interpolation (str): Method for interpolating positional embeddings.
        min_fps (int): Minimum frames per second.
        max_fps (int): Maximum frames per second.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
        rope_h_extrapolation_ratio (float): Height extrapolation ratio for RoPE.
        rope_w_extrapolation_ratio (float): Width extrapolation ratio for RoPE.
        rope_t_extrapolation_ratio (float): Temporal extrapolation ratio for RoPE.
        extra_per_block_abs_pos_emb (bool): Must be False (LearnablePosEmbAxis is not vendored).
    """

    def __init__(
        self,
        max_img_h: int,  # 240
        max_img_w: int,  # 240
        max_frames: int,  # 128
        in_channels: int,
        out_channels: int,
        patch_spatial: int,  # 2
        patch_temporal: int,  # 1
        concat_padding_mask: bool = True,  # True
        # attention settings
        model_channels: int = 768,  # = 2048 (2B) / 5120 (14B)
        num_blocks: int = 10,  # = 28 (2B) / 36 (14B)
        num_heads: int = 16,  # = 16 (2B) / 40 (14B)
        mlp_ratio: float = 4.0,
        atten_backend: str = "minimal_a2a",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        # positional embedding settings
        pos_emb_cls: str = "sincos",  # "rope3d"
        pos_emb_learnable: bool = False,  # True
        pos_emb_interpolation: str = "crop",  # "crop"
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,  # True
        adaln_lora_dim: int = 256,  # 256
        rope_h_extrapolation_ratio: float = 1.0,  # 3.0
        rope_w_extrapolation_ratio: float = 1.0,  # 3.0
        rope_t_extrapolation_ratio: float = 1.0,  # 1.0
        extra_per_block_abs_pos_emb: bool = False,  # False
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,  # False
        sac_config: SACConfig = SACConfig(),  # every_n_blocks=1, mode="predict2_2b_720"
        # Any4D:
        in_channels_override: int = None,  # varies; first / existing viewpoint only
        out_channels_override: int = None,  # varies; first / existing viewpoint only
        block_cls: type = Block,
        block_gate_fix: bool = False,
        isolate_stub_streams: bool = False,
        **leftover,
    ) -> None:
        super().__init__()
        assert not extra_per_block_abs_pos_emb, "extra_per_block_abs_pos_emb (LearnablePosEmbAxis) is not vendored"
        self.max_img_h = max_img_h  # 240
        self.max_img_w = max_img_w  # 240
        self.max_frames = max_frames  # 128
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial  # 2
        self.patch_temporal = patch_temporal  # 1
        self.num_heads = num_heads  # = 16 (2B) / 40 (14B)
        self.num_blocks = num_blocks  # = 28 (2B) / 36 (14B)
        self.model_channels = model_channels  # = 2048 (2B) / 5120 (14B)
        self.concat_padding_mask = concat_padding_mask  # True
        self.atten_backend = atten_backend  # "minimal_a2a"
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls  # "rope3d"
        self.pos_emb_learnable = pos_emb_learnable  # True
        self.pos_emb_interpolation = pos_emb_interpolation  # "crop"
        self.min_fps = min_fps  # 1
        self.max_fps = max_fps  # 30
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio  # 3.0
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio  # 3.0
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio  # 1.0
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb  # False
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio  # 1.0
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio  # 1.0
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio  # 1.0
        self.rope_enable_fps_modulation = rope_enable_fps_modulation  # False

        # Any4D:
        self.in_channels_override = in_channels_override  # varies
        self.out_channels_override = out_channels_override  # varies
        self.block_cls = block_cls
        self.block_gate_fix = block_gate_fix
        self.isolate_stub_streams = isolate_stub_streams

        self.build_patch_embed()  # first / existing viewpoint only
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora  # True
        self.adaln_lora_dim = adaln_lora_dim  # 256
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        # Any4D
        self.rope_dim = model_channels // num_heads  # 2048 // 16 = 128

        self.blocks = nn.ModuleList(
            [
                block_cls(
                    x_dim=model_channels,  # 2048
                    context_dim=crossattn_emb_channels,  # 1024
                    num_heads=num_heads,  # = 16 (2B) / 40 (14B)
                    mlp_ratio=mlp_ratio,  # 4.0
                    use_adaln_lora=use_adaln_lora,  # True
                    adaln_lora_dim=adaln_lora_dim,  # 256
                    backend=atten_backend,  # "minimal_a2a"
                    rope_dim=self.rope_dim,  # 128
                    block_gate_fix=block_gate_fix,
                    isolate_stub_streams=isolate_stub_streams,
                )
                for _ in range(num_blocks)
            ]
        )

        if self.out_channels_override is not None:
            actual_out_channels = self.out_channels_override  # Any4D
        else:
            actual_out_channels = out_channels  # 16 (vanilla)

        self.final_layer = FinalLayer(  # first / existing viewpoint only
            hidden_size=self.model_channels,  # = 2048 (2B) / 5120 (14B)
            spatial_patch_size=self.patch_spatial,  # 2
            temporal_patch_size=self.patch_temporal,  # 1
            out_channels=actual_out_channels,  # varies
            use_adaln_lora=self.use_adaln_lora,  # = True
            adaln_lora_dim=self.adaln_lora_dim,  # = 256
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        self.init_weights()
        self.enable_selective_checkpoint(sac_config)

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def build_patch_embed(self) -> None:
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels

        if self.in_channels_override is not None:
            actual_in_channels = self.in_channels_override  # Any4D (bypass random +1s all over the place)
        else:
            actual_in_channels = in_channels  # 18 (vanilla)

        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,  # = 2
            temporal_patch_size=patch_temporal,  # = 1
            in_channels=actual_in_channels,  # = depends on surgery
            out_channels=model_channels,  # = 2048 (2B) / 5120 (14B)
        )

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":  # yes
            cls_type = VideoRopePosition3DEmb
        else:  # no
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.debug(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,  # = 2048 (2B) / 5120 (14B)
            len_h=self.max_img_h // self.patch_spatial,  # 240 // 2 = 120
            len_w=self.max_img_w // self.patch_spatial,  # 240 // 2 = 120
            len_t=self.max_frames // self.patch_temporal,  # 128 // 1 = 128
            max_fps=self.max_fps,  # 30
            min_fps=self.min_fps,  # 1
            is_learnable=self.pos_emb_learnable,  # True
            interpolation=self.pos_emb_interpolation,  # "crop"
            head_dim=self.model_channels // self.num_heads,  # 2048 // 16 = 128
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,  # 3.0
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,  # 3.0
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,  # 1.0
            enable_fps_modulation=self.rope_enable_fps_modulation,  # False
        )
        self.pos_embedder = cls_type(
            **kwargs,  # type: ignore
        )

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        # NOTE: this is POSITION MAJOR and CHANNEL MINOR, which is the INVERSE of patchify!
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.NONE:
            pass
        else:
            log.debug(
                f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(self.blocks)}"
            )
            _context_fn = sac_config.get_context_fn()
            for block_id, block in self.blocks.named_children():
                if int(block_id) % sac_config.every_n_blocks == 0:
                    log.debug(f"Enable selective checkpoint for block {block_id}")
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=_context_fn,
                        preserve_rng_state=False,
                    )
                    self.blocks.register_module(block_id, block)
            self.register_module(
                "final_layer",
                ptd_checkpoint_wrapper(
                    self.final_layer,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                ),
            )

        return self
