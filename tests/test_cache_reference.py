"""Paged KV cache, FP8 utilities and reference decode attention.

Every test runs on CPU and again on CUDA (``gpu`` marker, auto-skipped without CUDA).
"""

from __future__ import annotations

import math

import pytest
import torch

from kvcache.cache import fp8 as fp8q
from kvcache.cache.paged import (
    NULL_BLOCK, allocate_kv_cache, assign_blocks, blocks_needed, build_paged_kv_cache, gather_kv,
)
from kvcache.reference.decode_attention import dense_decode_attention_sdpa, paged_decode_attention

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def make_seqs(lens, num_kv_heads, head_dim, seed=0, device="cpu", dtype=torch.float16):
    g = torch.Generator().manual_seed(seed)
    ks = [torch.randn(n, num_kv_heads, head_dim, generator=g).to(device, dtype) for n in lens]
    vs = [torch.randn(n, num_kv_heads, head_dim, generator=g).to(device, dtype) for n in lens]
    return ks, vs


def make_q(batch, num_q_heads, head_dim, seed=1, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, num_q_heads, head_dim, generator=g).to(device, torch.float16)


# --------------------------------------------------------------------------- fp8

def _fp8_error_bound(x: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    # e4m3: 3 mantissa bits -> half-ulp <= 2^-4 * |x| for normals (|x/scale| >= 2^-6);
    # subnormal spacing is 2^-9 * scale -> abs error <= 2^-10 * scale. Tiny slack for fp32 math.
    return torch.maximum(x.abs() * 2.0**-4, scale_b * 2.0**-10) * (1 + 1e-5) + 1e-12


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("granularity", fp8q.GRANULARITIES)
def test_fp8_roundtrip_error_bound(device, granularity) -> None:
    g = torch.Generator().manual_seed(0)
    # Heavy-tailed data with per-head magnitude differences (exercises normals and subnormals).
    x = torch.randn(6, 16, 4, 64, generator=g) * torch.tensor([0.01, 1.0, 30.0, 500.0]).view(1, 1, 4, 1)
    x[0, 0, 0, 0] = 0.0
    x = x.to(device)
    scale = fp8q.compute_scale(x, granularity)
    q = fp8q.quantize(x, scale)
    assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
    deq = fp8q.dequantize(q, scale)
    sb = fp8q.broadcast_scale(scale, x.shape)
    assert torch.all((deq - x).abs() <= _fp8_error_bound(x, sb))
    # The group max maps to exactly 448 -> dequantizes back to amax.
    assert torch.allclose(deq.abs().amax(), x.abs().amax(), rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_fp8_scale_shapes_and_granularity(device) -> None:
    x = torch.randn(5, 16, 3, 8, device=device)
    shapes = {"tensor": (), "kv_head": (3,), "block_head": (5, 3), "token_head": (5, 16, 3)}
    for gran, shape in shapes.items():
        s = fp8q.compute_scale(x, gran)
        assert tuple(s.shape) == shape and s.dtype == torch.float32
        assert fp8q.granularity_of(s) == gran
    ref = x.abs().amax(dim=(0, 1, 3)) / 448
    assert torch.allclose(fp8q.compute_scale(x, "kv_head"), ref)
    with pytest.raises(ValueError):
        fp8q.compute_scale(x, "per_channel")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fp8q.broadcast_scale(torch.ones(4, device=device), x.shape)  # wrong head count


@pytest.mark.parametrize("device", DEVICES)
def test_fp8_clamps_instead_of_overflowing(device) -> None:
    x = torch.tensor([1000.0, -1000.0, 0.5, 448.0], device=device).view(1, 4, 1, 1)
    scale = torch.ones(1, device=device)  # kv_head scale for 1 head
    deq = fp8q.dequantize(fp8q.quantize(x, scale), scale).flatten().tolist()
    assert deq == [448.0, -448.0, 0.5, 448.0]
    toks = fp8q.quantize_tokens(torch.full((3, 1, 2), 900.0, device=device), scale)
    assert toks.float().max().item() == 448.0
    with pytest.raises(ValueError):
        fp8q.quantize_tokens(torch.ones(3, 1, 2), torch.ones(2, 1))


@pytest.mark.parametrize("device", DEVICES)
def test_fp8_zero_group_and_nan_ignored(device) -> None:
    x = torch.zeros(2, 4, 2, 8, device=device)
    x[:, :, 1] = 3.0
    x[0, 0, 1, 0] = float("nan")  # poison must not affect the scale
    s = fp8q.compute_scale(x, "kv_head")
    assert s.tolist() == [1.0, pytest.approx(3.0 / 448)]
    deq = fp8q.dequantize(fp8q.quantize(x, s), s)
    assert torch.isnan(deq[0, 0, 1, 0]) and deq[:, :, 0].abs().max() == 0


# --------------------------------------------------------------------------- paged cache

def test_assign_blocks_random_noncontiguous() -> None:
    lens = [1, 15, 16, 17, 70]
    table = assign_blocks(lens, 16, num_blocks=40, generator=torch.Generator().manual_seed(0))
    assert table.dtype == torch.int32 and tuple(table.shape) == (5, 5)
    used = [table[b, :blocks_needed(n, 16)].tolist() for b, n in enumerate(lens)]
    flat = [x for row in used for x in row]
    assert NULL_BLOCK not in flat and len(set(flat)) == len(flat)
    assert all(table[b, len(row):].eq(NULL_BLOCK).all() for b, row in enumerate(used))
    long_row = used[-1]
    assert any(b2 != b1 + 1 for b1, b2 in zip(long_row, long_row[1:]))  # not one contiguous run
    with pytest.raises(ValueError):
        assign_blocks([100], 16, num_blocks=7)
    with pytest.raises(ValueError):
        assign_blocks([0], 16, num_blocks=7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_allocate_layouts(device, layout) -> None:
    k, v = allocate_kv_cache(10, 16, 4, 64, device=device, physical_layout=layout)
    assert tuple(k.shape) == (10, 16, 4, 64) and v.shape == k.shape
    if layout == "NHD":
        assert k.is_contiguous() and k.stride() == (16 * 4 * 64, 4 * 64, 64, 1)
    else:
        assert not k.is_contiguous() and k.stride() == (4 * 16 * 64, 64, 16 * 64, 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("block_size", [16, 32])
def test_build_and_gather_exact(device, layout, block_size) -> None:
    lens = [1, block_size - 1, block_size, block_size + 1, 3 * block_size + 5]
    ks, vs = make_seqs(lens, 2, 64, device=device)
    cache = build_paged_kv_cache(ks, vs, block_size, physical_layout=layout, seed=3)
    assert cache.key_cache.dtype == torch.float16 and cache.block_size == block_size
    assert cache.context_lens.tolist() == lens and cache.context_lens.dtype == torch.int32
    assert cache.block_tables.device.type == torch.device(device).type
    for b in range(len(lens)):
        k, v = gather_kv(cache, b)
        assert torch.equal(k, ks[b]) and torch.equal(v, vs[b])
    # Poison: the null block, unused blocks and the tail of each last block are NaN.
    assert torch.isnan(cache.key_cache[NULL_BLOCK]).all()
    last = cache.block_tables[0, 0].long()  # sequence of length 1: slots 1.. are padding
    assert torch.isnan(cache.key_cache[last, 1:]).all() and not torch.isnan(cache.key_cache[last, 0]).any()
    used = set(cache.block_tables.flatten().tolist())
    free = [i for i in range(cache.num_blocks) if i not in used]
    assert free and torch.isnan(cache.value_cache[free]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("granularity", fp8q.GRANULARITIES)
def test_build_fp8_cache(device, granularity) -> None:
    lens = [1, 17, 40]
    ks, vs = make_seqs(lens, 2, 64, device=device)
    cache = build_paged_kv_cache(ks, vs, 16, kv_dtype="fp8", scale_granularity=granularity)
    assert cache.is_fp8 and cache.scale_granularity == granularity
    # A finer scale is never larger than the per-tensor one, so the e4m3 bound computed with the
    # per-tensor scale holds for every granularity.
    s_k = torch.cat([x.float() for x in ks]).abs().max() / 448
    s_v = torch.cat([x.float() for x in vs]).abs().max() / 448
    for b in range(len(lens)):
        k, v = gather_kv(cache, b)
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        assert torch.all((k - ks[b].float()).abs() <= _fp8_error_bound(ks[b].float(), s_k))
        assert torch.all((v - vs[b].float()).abs() <= _fp8_error_bound(vs[b].float(), s_v))
    # Default granularity is per KV head.
    assert build_paged_kv_cache(ks, vs, 16, kv_dtype="fp8").k_scale.shape == (2,)


# --------------------------------------------------------------------------- reference attention

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("gqa", [1, 4, 6, 8])
@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_reference_matches_sdpa(device, head_dim, gqa, block_size) -> None:
    num_kv_heads = 2
    lens = [1, block_size + 3, 2 * block_size, 5 * block_size - 7, 37]  # ragged, incl. 1
    ks, vs = make_seqs(lens, num_kv_heads, head_dim, seed=head_dim + gqa, device=device)
    cache = build_paged_kv_cache(ks, vs, block_size, seed=block_size)
    q = make_q(len(lens), num_kv_heads * gqa, head_dim, device=device)
    out = paged_decode_attention(q, cache.key_cache, cache.value_cache, cache.block_tables,
                                 cache.context_lens)
    ref = dense_decode_attention_sdpa(q, ks, vs)
    assert out.dtype == torch.float32 and out.shape == q.shape
    assert torch.isfinite(out).all()  # poison was never read
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    # Length 1: the output is exactly that token's V for every query head in the group.
    torch.testing.assert_close(out[0], vs[0][0].float().repeat_interleave(gqa, 0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_paged_reference_layout_and_sm_scale(device) -> None:
    lens = [5, 33, 64]
    ks, vs = make_seqs(lens, 4, 64, device=device)
    q = make_q(3, 16, 64, device=device)
    outs = []
    for layout in ("NHD", "HND"):
        c = build_paged_kv_cache(ks, vs, 16, physical_layout=layout, seed=7)
        outs.append(paged_decode_attention(q, c.key_cache, c.value_cache, c.block_tables,
                                           c.context_lens, sm_scale=0.3))
    torch.testing.assert_close(outs[0], outs[1], atol=0, rtol=0)
    torch.testing.assert_close(outs[0], dense_decode_attention_sdpa(q, ks, vs, sm_scale=0.3),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("granularity", fp8q.GRANULARITIES)
def test_fp8_paged_reference_matches_sdpa_on_dequantized(device, granularity) -> None:
    """Kernel-error semantics: the fp8 path equals attention over the dequantized cache."""
    lens = [1, 19, 64, 100]
    ks, vs = make_seqs(lens, 2, 128, device=device)
    cache = build_paged_kv_cache(ks, vs, 16, kv_dtype="fp8", scale_granularity=granularity)
    q = make_q(len(lens), 12, 128, device=device)
    out = paged_decode_attention(q, cache.key_cache, cache.value_cache, cache.block_tables,
                                 cache.context_lens, k_scale=cache.k_scale, v_scale=cache.v_scale)
    dq = [gather_kv(cache, b) for b in range(len(lens))]
    ref = dense_decode_attention_sdpa(q, [k for k, _ in dq], [v for _, v in dq])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_reference_input_validation(device) -> None:
    ks, vs = make_seqs([4], 2, 64, device=device)
    c16 = build_paged_kv_cache(ks, vs, 16)
    c8 = build_paged_kv_cache(ks, vs, 16, kv_dtype="fp8")
    q = make_q(1, 4, 64, device=device)
    with pytest.raises(ValueError, match="requires k_scale"):
        paged_decode_attention(q, c8.key_cache, c8.value_cache, c8.block_tables, c8.context_lens)
    with pytest.raises(ValueError, match="only valid with an fp8"):
        paged_decode_attention(q, c16.key_cache, c16.value_cache, c16.block_tables,
                               c16.context_lens, k_scale=c8.k_scale, v_scale=c8.v_scale)
    with pytest.raises(ValueError, match="multiple"):
        paged_decode_attention(make_q(1, 3, 64, device=device), c16.key_cache, c16.value_cache,
                               c16.block_tables, c16.context_lens)
    with pytest.raises(ValueError, match=">= 1"):
        paged_decode_attention(q, c16.key_cache, c16.value_cache, c16.block_tables,
                               torch.zeros(1, dtype=torch.int32, device=device))


@pytest.mark.gpu
@pytest.mark.slow
def test_paged_reference_long_context_cuda() -> None:
    lens = [32768, 1000, 4097]
    ks, vs = make_seqs(lens, 2, 128, device="cuda")
    cache = build_paged_kv_cache(ks, vs, 16)
    q = make_q(3, 12, 128, device="cuda")
    out = paged_decode_attention(q, cache.key_cache, cache.value_cache, cache.block_tables,
                                 cache.context_lens)
    torch.testing.assert_close(out, dense_decode_attention_sdpa(q, ks, vs), atol=1e-5, rtol=1e-4)
