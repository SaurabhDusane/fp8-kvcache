"""Paged KV cache: allocation, block tables, and builders from dense per-sequence K/V.

Layout (chosen): logical **NHD** per block, i.e. ``[num_blocks, block_size, num_kv_heads,
head_dim]`` for the key cache and the value cache separately. This is the order vLLM V1 uses
inside a block, so a later vLLM integration can pass its cache as strided views without copying.

Consumers must be *stride-generic*: index the logical dims (block, token, head, dim) through the
tensor's strides and never assume contiguity. ``allocate_kv_cache(physical_layout="HND")``
allocates ``[num_blocks, num_kv_heads, block_size, head_dim]`` memory and returns it permuted to
the same logical NHD order, so HND can be benchmarked with no change to any consumer.

Conventions (shared with the kernels):
  - ``block_tables[b, i]`` (int32) is the physical block holding tokens
    ``[i*block_size, (i+1)*block_size)`` of sequence b; ``context_lens[b]`` (int32) its length.
  - Block 0 is reserved as the null block (as in vLLM V1): never assigned to a sequence, used to
    pad block-table rows. Builders fill it, all unused blocks and the unused tail slots of each
    sequence's last block with NaN ("poison"), so any read past ``context_lens`` shows up as NaN.
  - fp8 caches carry static ``k_scale``/``v_scale`` (see ``kvcache.cache.fp8``); default
    granularity is per KV head, shape ``[num_kv_heads]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from kvcache.cache import fp8 as fp8q

LOGICAL_LAYOUT = "NHD"
PhysicalLayout = Literal["NHD", "HND"]
NULL_BLOCK = 0


@dataclass
class PagedKVCache:
    key_cache: torch.Tensor      # logical [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor    # same shape/strides as key_cache
    block_tables: torch.Tensor   # [batch, max_blocks_per_seq] int32
    context_lens: torch.Tensor   # [batch] int32
    k_scale: torch.Tensor | None = None  # fp8 only; shape per kvcache.cache.fp8 granularity
    v_scale: torch.Tensor | None = None

    @property
    def num_blocks(self) -> int:
        return self.key_cache.shape[0]

    @property
    def block_size(self) -> int:
        return self.key_cache.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.key_cache.shape[2]

    @property
    def head_dim(self) -> int:
        return self.key_cache.shape[3]

    @property
    def is_fp8(self) -> bool:
        return self.key_cache.dtype == fp8q.FP8_DTYPE

    @property
    def scale_granularity(self) -> str | None:
        return None if self.k_scale is None else fp8q.granularity_of(self.k_scale)

    def strides(self) -> dict[str, tuple[int, int, int, int]]:
        """Element strides of the logical (block, token, head, dim) dims, for kernel launchers."""
        return {"key": tuple(self.key_cache.stride()), "value": tuple(self.value_cache.stride())}


def allocate_kv_cache(num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int,
                      dtype: torch.dtype = torch.float16, device: torch.device | str = "cpu",
                      physical_layout: PhysicalLayout = "NHD",
                      fill: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate (key_cache, value_cache), each logically [num_blocks, block_size, num_kv_heads,
    head_dim]. With ``physical_layout="HND"`` the memory is [num_blocks, num_kv_heads,
    block_size, head_dim] and the returned tensors are permuted (non-contiguous) views."""
    if physical_layout == "NHD":
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        perm = (0, 1, 2, 3)
    elif physical_layout == "HND":
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        perm = (0, 2, 1, 3)
    else:
        raise ValueError(f"physical_layout must be NHD or HND, got {physical_layout!r}")
    out = []
    for _ in range(2):
        t = torch.zeros(shape, dtype=dtype, device=device)
        if fill is not None:
            t.fill_(fill)
        out.append(t.permute(perm))
    return out[0], out[1]


def blocks_needed(context_len: int, block_size: int) -> int:
    return math.ceil(context_len / block_size)


def assign_blocks(context_lens: Sequence[int], block_size: int, num_blocks: int,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Random, non-contiguous physical blocks for each sequence -> block_tables [batch, max_blocks]
    int32. Block 0 (null) is never assigned and pads short rows."""
    need = [blocks_needed(n, block_size) for n in context_lens]
    if any(n < 1 for n in context_lens):
        raise ValueError("context lengths must be >= 1")
    if sum(need) > num_blocks - 1:
        raise ValueError(f"need {sum(need)} blocks + null block, only {num_blocks} allocated")
    perm = torch.randperm(num_blocks - 1, generator=generator) + 1  # skip the null block
    table = torch.full((len(need), max(need)), NULL_BLOCK, dtype=torch.int32)
    pos = 0
    for b, n in enumerate(need):
        table[b, :n] = perm[pos:pos + n].to(torch.int32)
        pos += n
    return table


def build_paged_kv_cache(keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor],
                         block_size: int, *, num_blocks: int | None = None,
                         kv_dtype: Literal["fp16", "bf16", "fp32", "fp8"] = "fp16",
                         scale_granularity: fp8q.Granularity = "kv_head",
                         physical_layout: PhysicalLayout = "NHD", seed: int = 0,
                         device: torch.device | str | None = None,
                         poison: bool = True) -> PagedKVCache:
    """Scatter dense per-sequence K/V (each [context_len, num_kv_heads, head_dim]) into a paged
    cache with random, non-contiguous block assignment.

    ``num_blocks`` defaults to 1 (null) + needed blocks + 25% spare, so free blocks sit between
    used ones. For ``kv_dtype="fp8"`` the scales are computed from the valid tokens only, at
    ``scale_granularity``, and the data is quantized with clamping.
    """
    if len(keys) != len(values) or not keys:
        raise ValueError("need the same, non-zero number of key and value tensors")
    h, d = keys[0].shape[1:]
    for k, v in zip(keys, values):
        if k.shape != v.shape or k.dim() != 3 or tuple(k.shape[1:]) != (h, d):
            raise ValueError(f"each K/V must be [len, {h}, {d}] and K/V shapes must match")
    device = torch.device(device) if device is not None else keys[0].device
    lens = [k.shape[0] for k in keys]
    need = sum(blocks_needed(n, block_size) for n in lens)
    if num_blocks is None:
        num_blocks = 1 + need + max(1, need // 4)
    gen = torch.Generator().manual_seed(seed)
    block_tables = assign_blocks(lens, block_size, num_blocks, gen)

    # Build in fp32 first (exact for fp16/bf16 inputs), then cast or quantize.
    kc32, vc32 = allocate_kv_cache(num_blocks, block_size, h, d, torch.float32, device,
                                   physical_layout, fill=float("nan") if poison else 0.0)
    for b, (k, v) in enumerate(zip(keys, values)):
        n = k.shape[0]
        nblk = blocks_needed(n, block_size)
        blocks = block_tables[b, :nblk].to(device=device, dtype=torch.long)
        pad = nblk * block_size - n
        kk = torch.cat([k.float(), k.new_full((pad, h, d), float("nan") if poison else 0.0).float()])
        vv = torch.cat([v.float(), v.new_full((pad, h, d), float("nan") if poison else 0.0).float()])
        kc32[blocks] = kk.to(device).view(nblk, block_size, h, d)
        vc32[blocks] = vv.to(device).view(nblk, block_size, h, d)

    k_scale = v_scale = None
    if kv_dtype == "fp8":
        k_scale = fp8q.compute_scale(kc32, scale_granularity)  # NaN poison ignored
        v_scale = fp8q.compute_scale(vc32, scale_granularity)
        kc_data, vc_data = fp8q.quantize(kc32, k_scale), fp8q.quantize(vc32, v_scale)
        dtype = fp8q.FP8_DTYPE
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[kv_dtype]
        kc_data, vc_data = kc32, vc32
    key_cache, value_cache = allocate_kv_cache(num_blocks, block_size, h, d, dtype, device,
                                               physical_layout)
    key_cache.copy_(kc_data)  # copy_ keeps the physical layout of the destination
    value_cache.copy_(vc_data)
    return PagedKVCache(key_cache=key_cache, value_cache=value_cache,
                        block_tables=block_tables.to(device),
                        context_lens=torch.tensor(lens, dtype=torch.int32, device=device),
                        k_scale=k_scale, v_scale=v_scale)


def gather_kv(cache: PagedKVCache, seq: int,
              dequantize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense (K, V) for one sequence, each [context_len, num_kv_heads, head_dim], read through
    the block table. fp8 caches are dequantized to fp32 unless ``dequantize=False``."""
    n = int(cache.context_lens[seq])
    nblk = blocks_needed(n, cache.block_size)
    blocks = cache.block_tables[seq, :nblk].long()
    out = []
    for data, scale in ((cache.key_cache, cache.k_scale), (cache.value_cache, cache.v_scale)):
        x = data[blocks]  # [nblk, block_size, h, d]
        if cache.is_fp8 and dequantize:
            assert scale is not None
            if scale.dim() >= 2:  # per-block/per-token scales are indexed by physical block
                scale = scale[blocks]
            x = fp8q.dequantize(x, scale)
        out.append(x.reshape(nblk * cache.block_size, cache.num_kv_heads, cache.head_dim)[:n])
    return out[0], out[1]
