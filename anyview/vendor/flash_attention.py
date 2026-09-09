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

# NOTE: the flash_attn 2/3/4 kernels are dropped; flash_attention() always takes the
# upstream torch SDPA fallback, which is also what upstream does without flash_attn installed.

import warnings

import torch

__all__ = [
    "flash_attention",
]


def _match_gqa_heads(q, k, v):
    if q.size(2) == k.size(2):
        return q, k, v

    assert q.size(2) % k.size(2) == 0
    repeat_factor = q.size(2) // k.size(2)
    return q, k.repeat_interleave(repeat_factor, dim=2), v.repeat_interleave(repeat_factor, dim=2)


def _scaled_dot_product_attention_fallback(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes

    if window_size != (-1, -1):
        warnings.warn("Sliding-window attention is unavailable in the SDPA fallback; full attention is used.")

    q, k, v = _match_gqa_heads(q, k, v)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out_dtype = q.dtype
    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)

    if q_scale is not None:
        q = q * q_scale

    attn_mask = None
    if k_lens is not None:
        key_positions = torch.arange(k.size(-2), device=k.device)
        invalid_keys = key_positions.unsqueeze(0) >= k_lens.to(device=k.device).unsqueeze(1)
        attn_mask = torch.zeros(
            (q.size(0), 1, q.size(-2), k.size(-2)),
            device=q.device,
            dtype=q.dtype,
        )
        attn_mask = attn_mask.masked_fill(invalid_keys[:, None, None, :], float("-inf"))

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=causal,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )

    out = out.transpose(1, 2).contiguous()
    if q_lens is not None:
        query_positions = torch.arange(out.size(1), device=out.device)
        valid_queries = query_positions.unsqueeze(0) < q_lens.to(device=out.device).unsqueeze(1)
        out = out * valid_queries[:, :, None, None]

    return out.to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    """
    del version  # flash_attn kernels are not vendored
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    return _scaled_dot_product_attention_fallback(q=q, k=k, v=v, dtype=dtype)
