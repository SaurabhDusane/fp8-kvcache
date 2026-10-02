"""v0: FP16 paged decode attention, correctness first (no optimizations).

Status: SCAFFOLD. The launcher (validation, output, grid, registration) is complete; the kernel
body is Saurabh's to write. Until then the launcher raises NotImplementedError, which the test
suite reports as a skip ("not implemented yet"), so the suite stays green.

Design (v0):
  - One program per (sequence, query head): grid = (batch, num_q_heads).
  - Each program walks its sequence's logical blocks 0..ceil(context_len / BLOCK_SIZE)-1,
    looks up the physical block in the block table, loads that block's K and V rows for the
    query head's KV head (GQA: kv_head = q_head // num_queries_per_kv), and folds them into
    an online softmax kept in fp32.
  - Cache layout is logical NHD [num_blocks, block_size, num_kv_heads, head_dim] with ANY
    strides (all four strides are kernel arguments), so physical HND memory works unchanged.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from kvcache.kernels import register_kernel

KERNEL_NAME = "v0_fp16_paged"


@triton.jit
def _paged_decode_v0_kernel(
    # ---- pointers -------------------------------------------------------------------------
    out_ptr,             # [batch, num_q_heads, head_dim], q.dtype
    q_ptr,               # [batch, num_q_heads, head_dim], fp16
    k_cache_ptr,         # logical [num_blocks, block_size, num_kv_heads, head_dim], fp16
    v_cache_ptr,         # same logical shape as k_cache (strides may differ)
    block_tables_ptr,    # [batch, max_blocks_per_seq], int32
    context_lens_ptr,    # [batch], int32
    # ---- scalars --------------------------------------------------------------------------
    sm_scale,            # fp32 softmax scale (usually head_dim ** -0.5)
    num_queries_per_kv,  # GQA group size = num_q_heads // num_kv_heads
    # ---- strides, in elements (not bytes) ---------------------------------------------------
    stride_q_b, stride_q_h, stride_q_d,
    stride_o_b, stride_o_h, stride_o_d,
    stride_k_blk, stride_k_tok, stride_k_head, stride_k_d,
    stride_v_blk, stride_v_tok, stride_v_head, stride_v_d,
    stride_bt_b, stride_bt_blk,
    # ---- compile-time constants -------------------------------------------------------------
    HEAD_DIM: tl.constexpr,     # actual head_dim (64 or 128 for our models)
    BLOCK_D: tl.constexpr,      # head_dim rounded up to a power of 2 (tl.arange needs one);
                                # lanes d >= HEAD_DIM must be masked off
    BLOCK_SIZE: tl.constexpr,   # tokens per KV-cache block (16 or 32), a power of 2
):
    """Paged decode attention for one (sequence, query head) per program.

    Computes  out[b, h, :] = softmax(sm_scale * q[b, h, :] . K[b, :n, kvh, :]^T) @ V[b, :n, kvh, :]
    with n = context_lens[b] and kvh = h // num_queries_per_kv, where token t of sequence b lives
    in physical block block_tables[b, t // BLOCK_SIZE] at slot t % BLOCK_SIZE.

    Online softmax over blocks, all in fp32: keep a running max m, running denominator l and
    an unnormalized accumulator acc[BLOCK_D]; for each block's scores s:
        m_new = max(m, max(s));  alpha = exp(m - m_new);  p = exp(s - m_new)
        l = l * alpha + sum(p);  acc = acc * alpha + p @ V_block;  m = m_new
    and finally out = acc / l (cast to the output dtype).

    Masking: the last block is usually partial; slots t >= n must contribute exactly nothing
    (score -inf before the max/exp). Their memory holds stale data (NaN in tests), so loads of
    those slots must be masked too, not just their scores. Never read block-table entries past
    ceil(n / BLOCK_SIZE): they are padding (null block 0).
    """
    # ======================================================================================
    # Pointer arithmetic for our layout (everything is base + sum(index * stride)):
    #
    #   b  = tl.program_id(0)          sequence index
    #   h  = tl.program_id(1)          query head
    #   kvh = h // num_queries_per_kv  KV head shared by this query head's GQA group
    #
    #   q row:        q_ptr + b * stride_q_b + h * stride_q_h + d * stride_q_d,   d in [0, BLOCK_D)
    #   context len:  n = load(context_lens_ptr + b)
    #   block table:  physical block of logical block i:
    #                 load(block_tables_ptr + b * stride_bt_b + i * stride_bt_blk)
    #   K element:    k_cache_ptr + phys * stride_k_blk      <- which physical block
    #                             + t    * stride_k_tok      <- slot t in [0, BLOCK_SIZE)
    #                             + kvh  * stride_k_head     <- KV head
    #                             + d    * stride_k_d        <- channel
    #                 The (t, d) tile of one block is a [BLOCK_SIZE, BLOCK_D] 2-D pointer block:
    #                 t[:, None] * stride_k_tok + d[None, :] * stride_k_d.
    #                 For contiguous NHD, stride_k_d == 1 and stride_k_tok == num_kv_heads *
    #                 head_dim, so each row is head_dim contiguous elements and rows are spaced
    #                 by all KV heads' rows; for HND memory stride_k_tok == head_dim and the
    #                 tile is fully contiguous. The kernel doesn't care: it only uses strides.
    #   V element:    same with the stride_v_* strides.
    #   out row:      out_ptr + b * stride_o_b + h * stride_o_h + d * stride_o_d
    #
    # Overflow: phys * stride_k_blk can exceed 2**31 for a large cache (12 GB of fp16 is
    # ~6e9 elements). Cast phys to tl.int64 before multiplying by the block stride.
    #
    # Masks:  d < HEAD_DIM (BLOCK_D padding)  and  i * BLOCK_SIZE + t < n (partial last block).
    # ======================================================================================
    #
    # TODO(Saurabh): kernel body.
    #   1. Program ids, kvh, n; load q (fp16 -> fp32), masked on d < HEAD_DIM.
    #   2. Init m = -inf, l = 0, acc = zeros([BLOCK_D]) in fp32.
    #   3. For i in range(0, cdiv(n, BLOCK_SIZE)): load phys from the block table; load the
    #      K tile (masked), scores = sum(q[None, :] * k, axis=1) * sm_scale, set masked slots
    #      to -inf; online-softmax update; load the V tile (masked) and accumulate.
    #   4. Store acc / l as the output dtype, masked on d < HEAD_DIM.
    pass


def _validate(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
              block_tables: torch.Tensor, context_lens: torch.Tensor,
              k_scale: torch.Tensor | None, v_scale: torch.Tensor | None) -> None:
    if q.dim() != 3:
        raise ValueError(f"q must be [batch, num_q_heads, head_dim], got {tuple(q.shape)}")
    if q.dtype != torch.float16:
        raise TypeError(f"{KERNEL_NAME}: q must be float16, got {q.dtype}")
    if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("k_cache/v_cache must both be [num_blocks, block_size, num_kv_heads, "
                         f"head_dim]; got {tuple(k_cache.shape)} and {tuple(v_cache.shape)}")
    if k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
        raise TypeError(f"{KERNEL_NAME} supports fp16 caches only (fp8 is v1); "
                        f"got {k_cache.dtype}/{v_cache.dtype}")
    if k_scale is not None or v_scale is not None:
        raise ValueError(f"{KERNEL_NAME}: k_scale/v_scale are only for fp8 caches")
    batch, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q {head_dim} vs cache {cache_head_dim}")
    if num_q_heads % num_kv_heads:
        raise ValueError(f"num_q_heads {num_q_heads} is not a multiple of num_kv_heads {num_kv_heads}")
    if block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of 2 (tl.arange), got {block_size}")
    if block_tables.dtype != torch.int32 or block_tables.dim() != 2 or block_tables.shape[0] != batch:
        raise ValueError(f"block_tables must be int32 [batch={batch}, max_blocks], got "
                         f"{block_tables.dtype} {tuple(block_tables.shape)}")
    if context_lens.dtype != torch.int32 or tuple(context_lens.shape) != (batch,):
        raise ValueError(f"context_lens must be int32 [batch={batch}], got "
                         f"{context_lens.dtype} {tuple(context_lens.shape)}")
    devices = {t.device for t in (q, k_cache, v_cache, block_tables, context_lens)}
    if len(devices) != 1:
        raise ValueError(f"all inputs must be on one device, got {devices}")
    if not q.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU tensors need TRITON_INTERPRET=1 (Triton interpreter)")


@register_kernel(KERNEL_NAME, supports_fp8=False,
                 description="FP16 paged decode, one program per (seq, q head), online softmax")
def decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     block_tables: torch.Tensor, context_lens: torch.Tensor, sm_scale: float,
                     k_scale: torch.Tensor | None = None,
                     v_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Launcher: validate, allocate the output, launch one program per (sequence, query head)."""
    _validate(q, k_cache, v_cache, block_tables, context_lens, k_scale, v_scale)
    batch, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    out = torch.empty((batch, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    grid = (batch, num_q_heads)

    # TODO(Saurabh): delete this line once the kernel body is written.
    raise NotImplementedError(f"{KERNEL_NAME}: kernel body not written yet")

    _paged_decode_v0_kernel[grid](  # noqa: unreachable until the body exists
        out, q, k_cache, v_cache, block_tables, context_lens,
        float(sm_scale), num_q_heads // num_kv_heads,
        *q.stride(), *out.stride(), *k_cache.stride(), *v_cache.stride(), *block_tables.stride(),
        HEAD_DIM=head_dim, BLOCK_D=triton.next_power_of_2(head_dim), BLOCK_SIZE=block_size,
    )
    return out
