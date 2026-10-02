"""Decode-attention kernels: one file per version (v0_..., v1_...) plus this registry.

Every kernel implements the reference interface (see kvcache.reference.decode_attention):

    decode_attention(q, k_cache, v_cache, block_tables, context_lens, sm_scale,
                     k_scale=None, v_scale=None) -> out

  q             [batch, num_q_heads, head_dim], fp16
  k/v_cache     logical [num_blocks, block_size, num_kv_heads, head_dim] (NHD), ANY strides
                (stride-generic: physical HND views are passed too); fp16, or float8_e4m3fn
  block_tables  [batch, max_blocks_per_seq] int32 (padding entries point at null block 0)
  context_lens  [batch] int32, >= 1
  sm_scale      float (softmax scale, usually head_dim ** -0.5)
  k/v_scale     fp8 only: fp32 [num_kv_heads] (per-KV-head, static)
  out           [batch, num_q_heads, head_dim], q.dtype

Registering (in the kernel's own file):

    from kvcache.kernels import register_kernel

    @register_kernel("v0_fp16_paged", supports_fp8=False)
    def decode_attention(q, k_cache, v_cache, block_tables, context_lens, sm_scale,
                         k_scale=None, v_scale=None): ...

``load_kernels()`` imports every ``v<N>_*.py`` module in this package so its decorator runs;
tests and benchmarks then iterate ``registered_kernels()``. Import errors propagate on purpose:
a broken kernel file must fail loudly, not silently drop out of the test suite.

Triton decides interpreted vs compiled when ``@triton.jit`` runs, i.e. at import, so
TRITON_INTERPRET must be set before ``load_kernels()`` (tests/conftest.py handles this).
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from typing import Callable

KernelFn = Callable[..., "object"]


@dataclass(frozen=True)
class KernelSpec:
    name: str
    fn: KernelFn
    supports_fp8: bool = False
    # fp8 scale granularities the kernel accepts (kvcache.cache.fp8 names).
    scale_granularities: tuple[str, ...] = ("kv_head",)
    # Set when the Triton interpreter can't run the kernel's fp8 path; the reason is shown on skip.
    interpreter_fp8_skip_reason: str | None = None
    description: str = ""

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


_REGISTRY: dict[str, KernelSpec] = {}
_KERNEL_MODULE = re.compile(r"^v\d+_\w+$")
_loaded = False


def register_kernel(name: str, *, supports_fp8: bool = False,
                    scale_granularities: tuple[str, ...] = ("kv_head",),
                    interpreter_fp8_skip_reason: str | None = None,
                    description: str = "") -> Callable[[KernelFn], KernelFn]:
    def deco(fn: KernelFn) -> KernelFn:
        if name in _REGISTRY and _REGISTRY[name].fn is not fn:
            raise ValueError(f"kernel {name!r} registered twice")
        _REGISTRY[name] = KernelSpec(name, fn, supports_fp8, tuple(scale_granularities),
                                     interpreter_fp8_skip_reason, description or (fn.__doc__ or "").strip())
        return fn
    return deco


def unregister_kernel(name: str) -> None:
    """For tests that register temporary kernels."""
    _REGISTRY.pop(name, None)


def kernel_module_names() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if _KERNEL_MODULE.match(m.name))


def load_kernels() -> list[KernelSpec]:
    """Import all v<N>_*.py kernel modules (once) and return the registered kernels."""
    global _loaded
    if not _loaded:
        for mod in kernel_module_names():
            importlib.import_module(f"{__name__}.{mod}")
        _loaded = True
    return registered_kernels()


def registered_kernels() -> list[KernelSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_kernel(name: str) -> KernelSpec:
    load_kernels()
    if name not in _REGISTRY:
        raise KeyError(f"no kernel {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
