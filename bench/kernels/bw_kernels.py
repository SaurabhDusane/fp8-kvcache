"""Triton kernels for measuring achievable DRAM bandwidth (used by scripts/peak_bw.py).

These are measurement tools, not project kernels (those live in src/kvcache/kernels/).

- ``copy``: dst[i] = src[i]. DRAM traffic = read N + write N bytes.
- ``reduce_sum``: read-only. Each program walks the input with a grid-stride loop, accumulates
  in fp32 registers and writes ONE partial sum, so traffic = read N bytes (+ 4 bytes/program).
  This is the ceiling that matters for decode attention, which is read-dominated.

Interpreter mode (TRITON_INTERPRET=1) is decided when ``@triton.jit`` runs, i.e. at import:
set the env var before importing this module (see tests/test_peak_bw.py).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# fp16 elements per program iteration. 8192 x 2 B = 16 KiB; with 8 warps each thread handles
# 32 contiguous fp16 (64 B) -> several 128-bit vector loads in flight per thread.
COPY_BLOCK = 8192
REDUCE_BLOCK = 8192
NUM_WARPS = 8


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # int64 offsets: 1 GiB of fp16 is 2**29 elements; int64 keeps larger buffers safe too.
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


@triton.jit
def _reduce_sum_kernel(src_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    # Grid-stride loop: program p reads blocks p, p + nprog, p + 2*nprog, ... so every program
    # keeps streaming and the grid can be sized to the GPU (a few waves of SMs), not to N.
    for start in range(pid.to(tl.int64) * BLOCK, n_elements, nprog.to(tl.int64) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(src_ptr + offs, mask=offs < n_elements, other=0.0)
        acc += x.to(tl.float32)
    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))


def triton_copy(src: torch.Tensor, dst: torch.Tensor, block: int = COPY_BLOCK) -> None:
    assert src.is_contiguous() and dst.is_contiguous() and src.numel() == dst.numel()
    n = src.numel()
    _copy_kernel[(triton.cdiv(n, block),)](src, dst, n, BLOCK=block, num_warps=NUM_WARPS)


def triton_reduce_sum(src: torch.Tensor, partials: torch.Tensor,
                      block: int = REDUCE_BLOCK) -> torch.Tensor:
    """Write one fp32 partial sum per program into ``partials`` (its length = grid size).

    The total is ``partials.sum()``, done outside the timed region by the caller.
    """
    assert src.is_contiguous() and partials.dtype == torch.float32
    n = src.numel()
    grid = min(partials.numel(), triton.cdiv(n, block))
    _reduce_sum_kernel[(grid,)](src, partials, n, BLOCK=block, num_warps=NUM_WARPS)
    return partials[:grid]
