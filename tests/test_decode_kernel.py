"""Decode-attention kernel tests, written before any kernel exists.

Every registered kernel (``kvcache.kernels.load_kernels()``) is checked against the PyTorch
reference on the *same* cache (kernel error, CLAUDE.md): fp16 output vs fp32 reference at
atol = rtol = 1e-2. fp8 caches use per-KV-head scales; kernels that don't support fp8 skip those
cases. Zero registered kernels -> every parametrized test collects as one skipped item ("empty
parameter set").

Tiers:
  interpreter  TRITON_INTERPRET=1, CPU tensors, tiny shapes (auto-enabled without CUDA; on the
               GPU machine: ``TRITON_INTERPRET=1 pytest -m interpreter``)
  gpu          CUDA, the CLAUDE.md shape grid: head_dim {64,128} x GQA {1,4,6,8} x block size
               {16,32} x {fp16, fp8}, ragged batches 1-64 incl. length 1 and non-multiples of the
               block size, non-contiguous block tables; plus HND (stride-generic) layout cases
  slow         CUDA, contexts up to 32k
  captured     CUDA, real KV from tests/data (skips if scripts/capture_kv.py hasn't been run)

The harness itself (``check_kernel``) is validated on CPU with a reference-backed kernel and a
set of deliberately broken ones (bottom of the file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import pytest
import torch

from kvcache.cache.paged import build_paged_kv_cache
from kvcache.kernels import KernelSpec, load_kernels
from kvcache.reference.decode_attention import paged_decode_attention

KERNEL_ATOL = 1e-2
KERNEL_RTOL = 1e-2

KERNELS: list[KernelSpec] = load_kernels()


@dataclass(frozen=True)
class Case:
    lens: tuple[int, ...]
    num_kv_heads: int = 2
    gqa: int = 1
    head_dim: int = 64
    block_size: int = 16
    kv_dtype: str = "fp16"          # "fp16" | "fp8"
    layout: str = "NHD"             # physical layout; the logical view is always NHD
    sm_scale: float | None = None   # None -> head_dim ** -0.5
    seed: int = 0

    @property
    def id(self) -> str:
        lens = (f"b{len(self.lens)}max{max(self.lens)}" if len(self.lens) > 4
                else "L" + "-".join(map(str, self.lens)))
        s = "" if self.sm_scale is None else f"-sm{self.sm_scale:g}"
        return (f"{self.kv_dtype}-{self.layout}-d{self.head_dim}-hkv{self.num_kv_heads}-g{self.gqa}"
                f"-bs{self.block_size}-{lens}{s}")


def _ragged(n: int, max_len: int, seed: int) -> tuple[int, ...]:
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, max_len + 1, (n,), generator=g).tolist()
    lens[0], lens[-1] = 1, max_len  # always include length 1 and the max
    return tuple(lens)


# --------------------------------------------------------------------------- case grids

INTERPRETER_CASES = [
    Case(lens=(1, 17, 40), gqa=1, block_size=16),
    Case(lens=(5, 33), gqa=4, block_size=16),
    Case(lens=(31, 2, 64), gqa=6, block_size=32),
    Case(lens=(9,), num_kv_heads=1, gqa=8, block_size=16),
    Case(lens=(1, 17, 40), gqa=4, block_size=16, layout="HND"),
    Case(lens=(1, 17, 40), gqa=4, block_size=16, sm_scale=0.05),
]
INTERPRETER_CASES += [replace(c, kv_dtype="fp8") for c in INTERPRETER_CASES[:3]]


def _gpu_cases() -> list[Case]:
    cases = []
    batches = [
        lambda bs: dict(lens=(1,), num_kv_heads=2),                                   # batch 1, len 1
        lambda bs: dict(lens=(1, bs - 1, bs, bs + 1, 1000), num_kv_heads=2),           # edges
        lambda bs: dict(lens=_ragged(64, 2048, seed=bs), num_kv_heads=4),             # batch 64
    ]
    for head_dim in (64, 128):
        for gqa in (1, 4, 6, 8):
            for block_size in (16, 32):
                for kv_dtype in ("fp16", "fp8"):
                    for make in batches:
                        cases.append(Case(head_dim=head_dim, gqa=gqa, block_size=block_size,
                                          kv_dtype=kv_dtype, seed=len(cases), **make(block_size)))
    # Stride-generic: physical HND memory behind the same logical NHD view.
    for block_size in (16, 32):
        for kv_dtype in ("fp16", "fp8"):
            cases.append(Case(lens=(1, block_size + 3, 700), head_dim=128, gqa=6,
                              block_size=block_size, kv_dtype=kv_dtype, layout="HND"))
    cases.append(Case(lens=(3, 100), head_dim=128, gqa=6, sm_scale=0.02))
    return cases


GPU_CASES = _gpu_cases()
SLOW_CASES = [
    Case(lens=(32768,), head_dim=128, gqa=6, block_size=bs, kv_dtype=dt)
    for bs in (16, 32) for dt in ("fp16", "fp8")
] + [
    Case(lens=(8192, 1, 16384, 32767), head_dim=128, gqa=6, block_size=16, kv_dtype=dt)
    for dt in ("fp16", "fp8")
] + [Case(lens=_ragged(16, 32768, seed=7), head_dim=64, num_kv_heads=4, gqa=8, kv_dtype="fp16")]


# --------------------------------------------------------------------------- harness

def _bits(t: torch.Tensor) -> torch.Tensor:
    """Bitwise snapshot (NaN poison compares equal to itself)."""
    return t.detach().contiguous().view(torch.uint8).clone()


def build_inputs(case: Case, device: str | torch.device):
    g = torch.Generator().manual_seed(case.seed)
    hq = case.num_kv_heads * case.gqa
    ks = [torch.randn(n, case.num_kv_heads, case.head_dim, generator=g).half() for n in case.lens]
    vs = [torch.randn(n, case.num_kv_heads, case.head_dim, generator=g).half() for n in case.lens]
    q = torch.randn(len(case.lens), hq, case.head_dim, generator=g).half().to(device)
    cache = build_paged_kv_cache(ks, vs, case.block_size, kv_dtype=case.kv_dtype,  # type: ignore[arg-type]
                                 scale_granularity="kv_head", physical_layout=case.layout,  # type: ignore[arg-type]
                                 seed=case.seed, device=device)
    sm_scale = case.sm_scale if case.sm_scale is not None else 1.0 / math.sqrt(case.head_dim)
    return q, cache, sm_scale


def check_kernel(spec: KernelSpec, case: Case, device: str | torch.device) -> None:
    """Run ``spec`` on ``case`` and compare with the reference on the same cache."""
    if case.kv_dtype == "fp8" and not spec.supports_fp8:
        pytest.skip(f"{spec.name} does not support fp8 caches")
    q, cache, sm_scale = build_inputs(case, device)
    check_on_cache(spec, q, cache, sm_scale)


def check_on_cache(spec: KernelSpec, q: torch.Tensor, cache, sm_scale: float) -> None:
    args = (q, cache.key_cache, cache.value_cache, cache.block_tables, cache.context_lens, sm_scale)
    scales = dict(k_scale=cache.k_scale, v_scale=cache.v_scale) if cache.is_fp8 else {}
    before = [_bits(t) for t in (q, cache.key_cache, cache.value_cache, cache.block_tables,
                                 cache.context_lens)]
    try:
        out = spec(*args, **scales)
    except NotImplementedError as exc:  # scaffolded kernel: visible as a skip, suite stays green
        pytest.skip(f"{spec.name} not implemented yet: {exc}")
    if q.is_cuda:
        torch.cuda.synchronize()
    ref = paged_decode_attention(*args, **scales)

    assert isinstance(out, torch.Tensor), f"{spec.name} returned {type(out)}"
    assert out.shape == q.shape, f"shape {tuple(out.shape)} != {tuple(q.shape)}"
    assert out.dtype == q.dtype, f"dtype {out.dtype} != q.dtype {q.dtype}"
    assert out.device == q.device
    assert torch.isfinite(out).all(), "non-finite output: kernel read padding/poisoned slots?"
    torch.testing.assert_close(out.float(), ref, atol=KERNEL_ATOL, rtol=KERNEL_RTOL,
                               msg=lambda m: f"{spec.name}: kernel vs reference (same cache)\n{m}")
    after = [_bits(t) for t in (q, cache.key_cache, cache.value_cache, cache.block_tables,
                                cache.context_lens)]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "kernel modified its inputs"


def _kernel_params() -> list:
    return [pytest.param(k, id=k.name) for k in KERNELS]


def _case_params(cases: Sequence[Case]) -> list:
    return [pytest.param(c, id=c.id) for c in cases]


def _kernel_case_params(cases: Sequence[Case]) -> list:
    """kernel x case as ONE parameter list, so zero registered kernels gives a single skipped
    item per tier instead of one per case."""
    return [pytest.param(k, c, id=f"{k.name}-{c.id}") for k in KERNELS for c in cases]


# --------------------------------------------------------------------------- tiers

@pytest.mark.interpreter
@pytest.mark.parametrize("kernel, case", _kernel_case_params(INTERPRETER_CASES))
def test_kernel_interpreter(kernel: KernelSpec, case: Case) -> None:
    if case.kv_dtype == "fp8" and kernel.interpreter_fp8_skip_reason:
        pytest.skip(f"{kernel.name}: fp8 in the Triton interpreter: {kernel.interpreter_fp8_skip_reason}")
    check_kernel(kernel, case, "cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("kernel, case", _kernel_case_params(GPU_CASES))
def test_kernel_gpu(kernel: KernelSpec, case: Case) -> None:
    check_kernel(kernel, case, "cuda")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("kernel, case", _kernel_case_params(SLOW_CASES))
def test_kernel_long_context(kernel: KernelSpec, case: Case) -> None:
    check_kernel(kernel, case, "cuda")


@pytest.mark.gpu
@pytest.mark.parametrize("kv_dtype", ["fp16", "fp8"])
@pytest.mark.parametrize("kernel", _kernel_params())
def test_kernel_captured_kv(kernel: KernelSpec, kv_dtype: str, captured_kv_dir) -> None:
    from bench.kernels.kv_capture import load_captured_paged

    if kv_dtype == "fp8" and not kernel.supports_fp8:
        pytest.skip(f"{kernel.name} does not support fp8 caches")
    data_dir, manifest = captured_kv_dir
    for layer in manifest["layers"]:
        for block_size in (16, 32):
            cache, q, payload = load_captured_paged(layer, block_size, kv_dtype=kv_dtype,
                                                    device="cuda", data_dir=data_dir)
            check_on_cache(kernel, q, cache, payload["sm_scale"])


# --------------------------------------------------------------------------- harness self-tests

def _ref_kernel(q, k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale=None, v_scale=None):
    return paged_decode_attention(q, k_cache, v_cache, block_tables, context_lens, sm_scale,
                                  k_scale, v_scale).to(q.dtype)


def _drops_last_token(q, k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale=None, v_scale=None):
    lens = torch.clamp(context_lens - 1, min=1)
    return _ref_kernel(q, k_cache, v_cache, block_tables, lens, sm_scale, k_scale, v_scale)


def _wrong_gqa(q, k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale=None, v_scale=None):
    b, hq, d = q.shape
    hkv = k_cache.shape[2]
    perm = torch.arange(hq).view(hkv, hq // hkv).t().reshape(-1)  # tile instead of interleave
    out = _ref_kernel(q[:, perm], k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale, v_scale)
    return out[:, torch.argsort(perm)]


def _reads_padding(q, k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale=None, v_scale=None):
    bs = k_cache.shape[1]
    full = ((context_lens + bs - 1) // bs) * bs  # whole last block, incl. poisoned tail slots
    return _ref_kernel(q, k_cache, v_cache, block_tables, full.to(torch.int32), sm_scale, k_scale, v_scale)


def _ignores_scales(q, k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale=None, v_scale=None):
    ones = None if k_scale is None else torch.ones_like(k_scale)
    return _ref_kernel(q, k_cache, v_cache, block_tables, context_lens, sm_scale, ones, ones)


def _fp32_out(*args, **kwargs):
    return _ref_kernel(*args, **kwargs).float()


def _mutates_cache(q, k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale=None, v_scale=None):
    out = _ref_kernel(q, k_cache, v_cache, block_tables, context_lens, sm_scale, k_scale, v_scale)
    k_cache[0].zero_()  # null block
    return out


HARNESS_CASES = [Case(lens=(1, 17, 40), gqa=4), Case(lens=(5, 33), gqa=4, kv_dtype="fp8"),
                 Case(lens=(16, 3), gqa=2, layout="HND")]


@pytest.mark.parametrize("case", _case_params(HARNESS_CASES))
def test_harness_accepts_reference_kernel(case: Case) -> None:
    check_kernel(KernelSpec("reference", _ref_kernel, supports_fp8=True), case, "cpu")


@pytest.mark.parametrize("bad, case, match", [
    (_drops_last_token, HARNESS_CASES[0], "kernel vs reference"),
    (_wrong_gqa, HARNESS_CASES[0], "kernel vs reference"),
    (_reads_padding, HARNESS_CASES[0], "non-finite"),
    (_ignores_scales, HARNESS_CASES[1], "kernel vs reference"),
    (_fp32_out, HARNESS_CASES[0], "dtype"),
    (_mutates_cache, HARNESS_CASES[0], "modified its inputs"),
], ids=["drops_last_token", "wrong_gqa", "reads_padding", "ignores_scales", "fp32_out", "mutates_cache"])
def test_harness_rejects_broken_kernels(bad, case: Case, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        check_kernel(KernelSpec("broken", bad, supports_fp8=True), case, "cpu")


def test_harness_skips_unimplemented_kernel() -> None:
    def scaffold(*args, **kwargs):
        raise NotImplementedError("body not written")

    with pytest.raises(pytest.skip.Exception, match="not implemented yet: body not written"):
        check_kernel(KernelSpec("scaffold", scaffold), HARNESS_CASES[0], "cpu")


def test_harness_skips_fp8_for_fp16_only_kernel() -> None:
    with pytest.raises(pytest.skip.Exception, match="does not support fp8"):
        check_kernel(KernelSpec("fp16_only", _ref_kernel), HARNESS_CASES[1], "cpu")


def test_case_grids_cover_claude_md_shapes() -> None:
    g = GPU_CASES
    assert {c.head_dim for c in g} == {64, 128} and {c.gqa for c in g} == {1, 4, 6, 8}
    assert {c.block_size for c in g} == {16, 32} and {c.kv_dtype for c in g} == {"fp16", "fp8"}
    assert {c.layout for c in g} == {"NHD", "HND"}
    assert any(len(c.lens) == 64 for c in g) and any(c.lens == (1,) for c in g)
    assert all(1 in c.lens for c in g if len(c.lens) == 64)
    assert any(any(n % c.block_size for n in c.lens) for c in g)
    assert max(max(c.lens) for c in SLOW_CASES) == 32768
    assert all(max(c.lens) <= 64 for c in INTERPRETER_CASES)


def test_registry() -> None:
    from kvcache import kernels as K

    assert all(K._KERNEL_MODULE.match(m) for m in K.kernel_module_names())
    K.register_kernel("zz_test_tmp", supports_fp8=True)(_ref_kernel)
    try:
        spec = K.get_kernel("zz_test_tmp")
        assert spec.supports_fp8 and spec.scale_granularities == ("kv_head",)
        assert "zz_test_tmp" in [k.name for k in K.registered_kernels()]
        K.register_kernel("zz_test_tmp")(_ref_kernel)  # same fn: idempotent
        with pytest.raises(ValueError, match="twice"):
            K.register_kernel("zz_test_tmp")(_fp32_out)
    finally:
        K.unregister_kernel("zz_test_tmp")
    with pytest.raises(KeyError):
        K.get_kernel("zz_test_tmp")
