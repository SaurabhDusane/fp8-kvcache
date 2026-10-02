"""FP8 (e4m3fn) quantization for the paged KV cache.

Scale convention: ``scale = amax / 448`` (448 = largest finite e4m3fn value), so
``x ≈ dequantize(quantize(x, scale), scale) = fp8(x / scale) * scale``. Values are clamped to
±448 before the cast, so out-of-range inputs saturate instead of depending on the device's cast
behaviour. An all-zero group gets scale 1.0.

Granularity (chosen: per-KV-head; the others are supported here and in the reference so they can
be evaluated later). For a cache laid out logically as [num_blocks, block_size, num_kv_heads,
head_dim], the scale tensor shape identifies the granularity:

    "tensor"      shape []                                   one scale per layer (vLLM's scheme)
    "kv_head"     shape [num_kv_heads]                       one scale per KV head  <- default
    "block_head"  shape [num_blocks, num_kv_heads]           one scale per block per head
    "token_head"  shape [num_blocks, block_size, num_kv_heads]  one scale per token per head

Scales are static: computed once from the data used to build the cache. Later writes reuse them
and clamp (see ``quantize``), which is exactly where coarse static scales can clip.
"""

from __future__ import annotations

from typing import Literal

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
SCALE_DTYPE = torch.float32

Granularity = Literal["tensor", "kv_head", "block_head", "token_head"]
GRANULARITIES: tuple[str, ...] = ("tensor", "kv_head", "block_head", "token_head")

# Dims of a [num_blocks, block_size, num_kv_heads, head_dim] tensor reduced to get each scale.
_REDUCE_DIMS: dict[str, tuple[int, ...]] = {
    "tensor": (0, 1, 2, 3),
    "kv_head": (0, 1, 3),
    "block_head": (1, 3),
    "token_head": (3,),
}


def scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    amax = amax.to(SCALE_DTYPE)
    return torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))


def compute_scale(x: torch.Tensor, granularity: Granularity = "kv_head") -> torch.Tensor:
    """Scale for ``x`` laid out as [num_blocks, block_size, num_kv_heads, head_dim].

    Non-finite entries (e.g. poisoned padding) are ignored when taking amax.
    """
    if granularity not in _REDUCE_DIMS:
        raise ValueError(f"unknown granularity {granularity!r}; choose from {GRANULARITIES}")
    if x.dim() != 4:
        raise ValueError(f"expected [num_blocks, block_size, num_kv_heads, head_dim], got {tuple(x.shape)}")
    a = x.detach().float().abs()
    a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
    return scale_from_amax(a.amax(dim=_REDUCE_DIMS[granularity]))


def granularity_of(scale: torch.Tensor) -> Granularity:
    """Infer the granularity from the scale's rank (see module docstring)."""
    by_rank: dict[int, Granularity] = {0: "tensor", 1: "kv_head", 2: "block_head", 3: "token_head"}
    if scale.dim() not in by_rank:
        raise ValueError(f"scale must have rank 0-3, got shape {tuple(scale.shape)}")
    return by_rank[scale.dim()]


def broadcast_scale(scale: torch.Tensor, cache_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    """View ``scale`` so it broadcasts against a [num_blocks, block_size, num_kv_heads, head_dim]
    tensor, after checking its shape matches the cache."""
    nb, bs, h, _ = cache_shape
    g = granularity_of(scale)
    expected = {"tensor": (), "kv_head": (h,), "block_head": (nb, h), "token_head": (nb, bs, h)}[g]
    if tuple(scale.shape) != expected:
        raise ValueError(f"{g} scale must have shape {expected} for cache {tuple(cache_shape)}, "
                         f"got {tuple(scale.shape)}")
    view = {"tensor": (1, 1, 1, 1), "kv_head": (1, 1, h, 1),
            "block_head": (nb, 1, h, 1), "token_head": (nb, bs, h, 1)}[g]
    return scale.reshape(view)


def quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8(clamp(x / scale, ±448)) for x shaped [num_blocks, block_size, num_kv_heads, head_dim].

    Math in fp32. NaN inputs stay NaN (used to poison unused cache slots in tests).
    """
    s = broadcast_scale(scale.to(x.device, SCALE_DTYPE), x.shape)
    y = (x.float() / s).clamp(-FP8_MAX, FP8_MAX)
    return y.to(FP8_DTYPE)


def dequantize(q: torch.Tensor, scale: torch.Tensor,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """fp8 cache -> ``dtype`` (fp32 by default): q.float() * scale."""
    if q.dtype != FP8_DTYPE:
        raise TypeError(f"expected {FP8_DTYPE}, got {q.dtype}")
    s = broadcast_scale(scale.to(q.device, SCALE_DTYPE), q.shape)
    return (q.float() * s).to(dtype)


def quantize_tokens(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize new tokens [num_tokens, num_kv_heads, head_dim] with an existing "tensor" or
    "kv_head" scale (static scales: values beyond 448*scale are clamped)."""
    if granularity_of(scale) not in ("tensor", "kv_head"):
        raise ValueError("quantize_tokens supports only static tensor/kv_head scales")
    return quantize(x.unsqueeze(0), scale).squeeze(0)
