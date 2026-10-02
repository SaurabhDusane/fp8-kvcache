"""PyTorch reference decode attention: the ground truth for kernel tests.

``paged_decode_attention`` has the same interface every kernel implements:

    decode_attention(q, key_cache, value_cache, block_tables, context_lens, sm_scale,
                     k_scale=None, v_scale=None) -> out

- q: [batch, num_q_heads, head_dim], one query token per sequence (decode).
- key_cache / value_cache: logical [num_blocks, block_size, num_kv_heads, head_dim] (any
  strides), fp16/bf16/fp32, or float8_e4m3fn with k_scale / v_scale (granularity inferred from
  the scale's rank, see kvcache.cache.fp8).
- GQA: query head i attends with KV head i // (num_q_heads // num_kv_heads), the same mapping
  as HF's ``repeat_kv``.
- All math in fp32; returns fp32 [batch, num_q_heads, head_dim]. Tests cast as needed.

Deliberately simple (a Python loop over sequences, explicit gathers): clarity over speed.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from kvcache.cache import fp8 as fp8q


def _gather(cache: torch.Tensor, scale: torch.Tensor | None, blocks: torch.Tensor,
            n: int) -> torch.Tensor:
    """[n, num_kv_heads, head_dim] fp32 for the given physical blocks (dequantized if fp8)."""
    x = cache[blocks]  # [nblk, block_size, h, d]
    if cache.dtype == fp8q.FP8_DTYPE:
        if scale is None:
            raise ValueError("fp8 cache requires k_scale and v_scale")
        if scale.dim() >= 2:  # per-block / per-token scales follow the physical block
            scale = scale[blocks]
        x = fp8q.dequantize(x, scale)
    elif scale is not None:
        raise ValueError("scales are only valid with an fp8 cache")
    else:
        x = x.float()
    return x.reshape(-1, cache.shape[2], cache.shape[3])[:n]


def paged_decode_attention(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                           block_tables: torch.Tensor, context_lens: torch.Tensor,
                           sm_scale: float | None = None, k_scale: torch.Tensor | None = None,
                           v_scale: torch.Tensor | None = None) -> torch.Tensor:
    if q.dim() != 3:
        raise ValueError(f"q must be [batch, num_q_heads, head_dim], got {tuple(q.shape)}")
    batch, hq, d = q.shape
    nb, bs, hkv, dk = key_cache.shape
    if dk != d or value_cache.shape != key_cache.shape:
        raise ValueError("cache head_dim/shape mismatch")
    if hq % hkv:
        raise ValueError(f"num_q_heads {hq} not a multiple of num_kv_heads {hkv}")
    group = hq // hkv
    sm_scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    out = torch.empty(batch, hq, d, dtype=torch.float32, device=q.device)
    for b in range(batch):
        n = int(context_lens[b])
        if n < 1:
            raise ValueError(f"context_lens[{b}] = {n}; decode needs >= 1 token")
        nblk = math.ceil(n / bs)
        blocks = block_tables[b, :nblk].long()
        k = _gather(key_cache, k_scale, blocks, n)    # [n, hkv, d]
        v = _gather(value_cache, v_scale, blocks, n)  # [n, hkv, d]
        qb = q[b].float().view(hkv, group, d)         # q head = kv_head * group + g
        scores = torch.einsum("hgd,nhd->hgn", qb, k) * sm_scale
        p = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hgn,nhd->hgd", p, v).reshape(hq, d)
    return out


def dense_decode_attention_sdpa(q: torch.Tensor, keys: Sequence[torch.Tensor],
                                values: Sequence[torch.Tensor],
                                sm_scale: float | None = None) -> torch.Tensor:
    """Independent check via torch SDPA on dense per-sequence K/V ([len, num_kv_heads, head_dim]
    each), fp32. KV heads are expanded with repeat_interleave (HF repeat_kv mapping)."""
    batch, hq, d = q.shape
    out = torch.empty(batch, hq, d, dtype=torch.float32, device=q.device)
    for b in range(batch):
        k = keys[b].float().to(q.device)
        v = values[b].float().to(q.device)
        group = hq // k.shape[1]
        k = k.repeat_interleave(group, dim=1).transpose(0, 1)  # [hq, n, d]
        v = v.repeat_interleave(group, dim=1).transpose(0, 1)
        qb = q[b].float().unsqueeze(1)                         # [hq, 1, d]
        out[b] = F.scaled_dot_product_attention(qb, k, v, scale=sm_scale).squeeze(1)
    return out
