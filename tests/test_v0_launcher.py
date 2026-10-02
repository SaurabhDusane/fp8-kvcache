"""v0_fp16_paged launcher: registration and input validation (CPU; runs before any launch).

Valid inputs must either raise NotImplementedError (scaffold) or return a correctly shaped
output (once the body exists), so these tests keep working after the kernel is written.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from kvcache.cache.paged import build_paged_kv_cache  # noqa: E402
from kvcache.kernels import get_kernel  # noqa: E402

pytestmark = pytest.mark.interpreter  # CPU tensors need TRITON_INTERPRET=1


def _inputs(kv_dtype: str = "fp16", hkv: int = 2, gqa: int = 3, d: int = 64, bs: int = 16):
    g = torch.Generator().manual_seed(0)
    lens = [1, 20]
    ks = [torch.randn(n, hkv, d, generator=g).half() for n in lens]
    vs = [torch.randn(n, hkv, d, generator=g).half() for n in lens]
    cache = build_paged_kv_cache(ks, vs, bs, kv_dtype=kv_dtype)  # type: ignore[arg-type]
    q = torch.randn(len(lens), hkv * gqa, d, generator=g).half()
    return q, cache


def test_registered() -> None:
    spec = get_kernel("v0_fp16_paged")
    assert spec.supports_fp8 is False and "online softmax" in spec.description


def test_valid_inputs_reach_the_kernel() -> None:
    q, c = _inputs()
    fn = get_kernel("v0_fp16_paged").fn
    try:
        out = fn(q, c.key_cache, c.value_cache, c.block_tables, c.context_lens, 0.125)
    except NotImplementedError as exc:
        assert "kernel body not written yet" in str(exc)
    else:
        assert out.shape == q.shape and out.dtype == torch.float16


@pytest.mark.parametrize("mutate, error, match", [
    (lambda q, c: (q.float(), c), TypeError, "q must be float16"),
    (lambda q, c: (q, _with(c, key_cache=c.key_cache.float(), value_cache=c.value_cache.float())),
     TypeError, "fp16 caches only"),
    (lambda q, c: (q[:, :5], c), ValueError, "not a multiple"),
    (lambda q, c: (q, _with(c, block_tables=c.block_tables.long())), ValueError, "block_tables must be int32"),
    (lambda q, c: (q, _with(c, context_lens=c.context_lens[:1])), ValueError, "context_lens must be int32"),
    (lambda q, c: (q[..., :32], c), ValueError, "head_dim mismatch"),
    (lambda q, c: (q[0], c), ValueError, "q must be"),
], ids=["q_dtype", "cache_dtype", "gqa", "bt_dtype", "cl_shape", "head_dim", "q_rank"])
def test_validation_errors(mutate, error, match) -> None:
    q, c = _inputs()
    q, c = mutate(q, c)
    with pytest.raises(error, match=match):
        get_kernel("v0_fp16_paged").fn(q, c.key_cache, c.value_cache, c.block_tables,
                                       c.context_lens, 0.125)


def test_rejects_fp8_cache_and_scales() -> None:
    q, c8 = _inputs("fp8")
    fn = get_kernel("v0_fp16_paged").fn
    with pytest.raises(TypeError, match="fp16 caches only"):
        fn(q, c8.key_cache, c8.value_cache, c8.block_tables, c8.context_lens, 0.125,
           k_scale=c8.k_scale, v_scale=c8.v_scale)
    q, c = _inputs()
    with pytest.raises(ValueError, match="only for fp8"):
        fn(q, c.key_cache, c.value_cache, c.block_tables, c.context_lens, 0.125,
           k_scale=torch.ones(2), v_scale=torch.ones(2))


def test_rejects_non_power_of_two_block_size() -> None:
    q, c = _inputs(bs=24)
    with pytest.raises(ValueError, match="power of 2"):
        get_kernel("v0_fp16_paged").fn(q, c.key_cache, c.value_cache, c.block_tables,
                                       c.context_lens, 0.125)


def _with(cache, **kw):
    from dataclasses import replace

    return replace(cache, **kw)
