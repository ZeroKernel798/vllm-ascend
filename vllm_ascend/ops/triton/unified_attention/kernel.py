# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tree Unified Attention — Triton-Ascend kernel.

Replaces the CANN FIA (Fused Infer Attention) vendor kernel for tree attention
paths in speculative decoding.  Implements paged KV-cache flash attention with
optional ``qq_bias`` (additive bias on the query-suffix region), aligning
semantically with upstream vLLM's ``triton_unified_attention``.

Layout assumptions (TND family)
-------------------------------
* Q   : ``[num_tokens, num_heads, head_dim]``   — contiguous (TND)
* K/V : ``[num_blocks, block_size, num_kv_heads, head_dim]`` — paged
* block_table : ``[num_seqs, max_blocks]``  int32

For v0.1 we support a single sequence (``num_seqs == 1``) to keep the loop
simple while validating correctness.
"""

import torch
from vllm.triton_utils import tl, triton

# Prevent NaN from exp(-inf) arithmetic
_NEG_LARGE: float = -1e30
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
    ],
    key=[],
)
@triton.heuristics({
    "USE_QQ_BIAS": lambda args: args["qq_bias"] is not None,
})
@triton.jit(do_not_specialize=["scale"])
def tree_unified_attention_varlen_kernel(
    # ---- Q (TND: [total_tokens, H, D]) ----
    q_ptr, q_stride_t, q_stride_h,
    # ---- K/V cache (paged: [B, BS, Hkv, D]) ----
    k_cache_ptr, k_stride_b, k_stride_s, k_stride_h,
    v_cache_ptr, v_stride_b, v_stride_s, v_stride_h,
    # ---- Block table ([num_seqs, max_blocks] int32) ----
    block_table, bt_stride_seq,
    # ---- Per-sequence metadata arrays (int32, length num_seqs(+1)) ----
    cu_seqlens_q_ptr,               # [num_seqs + 1] cumulative query boundaries
    seq_lens_ptr,                   # [num_seqs] per-seq total KV length
    context_lens_ptr,               # [num_seqs] per-seq prefix length
    # ---- qq_bias ([max_q, max_q] fp32, nullable; shared across seqs) ----
    qq_bias, qq_bias_stride_0,
    # ---- Output (TND: [total_tokens, H, D]) ----
    o_ptr, o_stride_t, o_stride_h,
    # ---- Scalars ----
    scale,
    num_heads, num_kv_heads, block_size,
    num_blocks, bt_num_cols,
    # ---- Compile-time constants ----
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
):
    """Paged flash attention with optional ``qq_bias``, varlen batch.

    Grid: ``(cdiv(max_query_len, BLOCK_M), num_heads, num_seqs)``.
    Each program handles one head + one BLOCK_M chunk of the queries that
    belong to sequence ``program_id(2)``. Sequence boundaries come from
    ``cu_seqlens_q``; per-seq ``seq_len`` / ``context_len`` / ``block_table``
    row are indexed by the sequence id. Numerically identical to the
    single-sequence serial kernel applied per sequence.
    """
    pid_m = tl.program_id(0)        # token-group axis (within sequence)
    head_idx = tl.program_id(1)     # head axis
    seq_idx = tl.program_id(2)      # sequence axis

    # Sequence query range from cu_seqlens.
    q_seq_start = tl.load(cu_seqlens_q_ptr + seq_idx).to(tl.int32)
    q_seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1).to(tl.int32)
    seq_num_q = q_seq_end - q_seq_start

    # Per-seq scalars.
    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    context_len = tl.load(context_lens_ptr + seq_idx).to(tl.int32)

    q_heads_per_kv = num_heads // num_kv_heads
    kv_head = head_idx // q_heads_per_kv

    d_offs = tl.arange(0, head_dim)

    # Sequence-relative query indices for this tile.
    q_rel = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # [BLOCK_M] within-seq
    q_mask = q_rel < seq_num_q
    q_glob = q_seq_start + q_rel                          # global token index

    # ---- Load Q tile ----
    q_base = q_glob[:, None] * q_stride_t + head_idx * q_stride_h
    q_tile = tl.load(
        q_ptr + q_base + d_offs[None, :],
        mask=q_mask[:, None], other=0.0,
    ).to(tl.float32)

    NEG_LARGE: tl.constexpr = -1e30
    m_i = tl.full([BLOCK_M], NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    bt_row = block_table + seq_idx * bt_stride_seq

    # ---- 2-page unrolled tile loop (process 2 KV tiles per iteration) ----
    for k_abs_start in range(0, seq_len, BLOCK_N * 2):
        # ---- Tile 0: load K, QK^T, causal, bias ----
        k_abs0 = k_abs_start + tl.arange(0, BLOCK_N)
        k_end0 = tl.minimum(k_abs_start + BLOCK_N, seq_len)
        num_k0 = k_end0 - k_abs_start
        k_mask0 = tl.arange(0, BLOCK_N) < num_k0
        # Clamp the block-table column and the resulting physical block id.
        # ``seq_len`` can exceed the range actually backed by allocated blocks
        # (KV cache is far smaller than max_model_len * max_num_seqs), and the
        # unbacked columns hold stale ids.  Without clamping the address
        # ``phys * k_stride_b`` walks off the cache and the MTE unit raises
        # "DDR address out of range", killing the whole EngineCore.
        bt_col0 = tl.minimum(k_abs_start // block_size, bt_num_cols - 1)
        phys0 = tl.load(bt_row + bt_col0).to(tl.int32)
        phys0 = tl.maximum(0, tl.minimum(phys0, num_blocks - 1))
        kw0 = (k_abs_start % block_size) + tl.arange(0, BLOCK_N)
        k_tile0 = tl.load(k_cache_ptr + (phys0 * k_stride_b + kw0 * k_stride_s + kv_head * k_stride_h)[:, None] + d_offs[None, :],
                          mask=k_mask0[:, None], other=0.0, care_padding=True)
        s0 = tl.dot(q_tile, tl.trans(k_tile0).to(tl.float32)) * scale
        if (k_abs_start + num_k0) > context_len:
            qp = q_rel[:, None].to(tl.float32); cf = context_len.to(tl.float32)
            kf = k_abs0[None, :].to(tl.float32)
            s0 = tl.where(((kf < cf) | (qp >= (kf - cf))) & k_mask0[None, :], s0, NEG_LARGE)
        if USE_QQ_BIAS and (k_abs_start + num_k0) > context_len:
            kr0 = (k_abs0 - context_len).to(tl.int32)
            m0 = (kr0 >= 0) & (kr0 < qq_bias_stride_0) & q_mask[:, None] & (q_rel < qq_bias_stride_0)[:, None]
            s0 = s0 + tl.load(qq_bias + q_rel[:, None] * qq_bias_stride_0 + kr0[None, :], mask=m0, other=0.0)

        # ---- Tile 1: load, compute, and reduce in one block (if exists) ----
        k_abs1_start = k_abs_start + BLOCK_N
        if k_abs1_start < seq_len:
            k_abs1 = k_abs1_start + tl.arange(0, BLOCK_N)
            nk1 = tl.minimum(k_abs1_start + BLOCK_N, seq_len) - k_abs1_start
            km1 = tl.arange(0, BLOCK_N) < nk1
            bt_col1 = tl.minimum(k_abs1_start // block_size, bt_num_cols - 1)
            ph1 = tl.load(bt_row + bt_col1).to(tl.int32)
            ph1 = tl.maximum(0, tl.minimum(ph1, num_blocks - 1))
            kw1 = (k_abs1_start % block_size) + tl.arange(0, BLOCK_N)
            k_tile1 = tl.load(k_cache_ptr + (ph1 * k_stride_b + kw1 * k_stride_s + kv_head * k_stride_h)[:, None] + d_offs[None, :],
                              mask=km1[:, None], other=0.0, care_padding=True)
            s1 = tl.dot(q_tile, tl.trans(k_tile1).to(tl.float32)) * scale
            if (k_abs1_start + nk1) > context_len:
                qp = q_rel[:, None].to(tl.float32); cf = context_len.to(tl.float32)
                kf = k_abs1[None, :].to(tl.float32)
                s1 = tl.where(((kf < cf) | (qp >= (kf - cf))) & km1[None, :], s1, NEG_LARGE)
            if USE_QQ_BIAS and (k_abs1_start + nk1) > context_len:
                kr1 = (k_abs1 - context_len).to(tl.int32)
                m1 = (kr1 >= 0) & (kr1 < qq_bias_stride_0) & q_mask[:, None] & (q_rel < qq_bias_stride_0)[:, None]
                s1 = s1 + tl.load(qq_bias + q_rel[:, None] * qq_bias_stride_0 + kr1[None, :], mask=m1, other=0.0)
            # Softmax + PV for tile 1
            mij1 = tl.max(tl.where(km1[None, :], s1, NEG_LARGE), axis=1)
            mn1 = tl.maximum(m_i, mij1)
            a1 = tl.exp(m_i - mn1); p1 = tl.exp(s1 - mn1[:, None])
            l_i = l_i * a1 + tl.sum(p1, axis=1); m_i = mn1
            v1 = tl.load(v_cache_ptr + (ph1 * v_stride_b + kw1 * v_stride_s + kv_head * v_stride_h)[:, None] + d_offs[None, :],
                         mask=km1[:, None], other=0.0, care_padding=True)
            acc = acc * a1[:, None] + tl.dot(p1, v1.to(tl.float32))

        # ---- Softmax + PV for tile 0 ----
        mij0 = tl.max(tl.where(k_mask0[None, :], s0, NEG_LARGE), axis=1)
        mn0 = tl.maximum(m_i, mij0)
        a0 = tl.exp(m_i - mn0); p0 = tl.exp(s0 - mn0[:, None])
        l_i = l_i * a0 + tl.sum(p0, axis=1); m_i = mn0
        v0 = tl.load(v_cache_ptr + (phys0 * v_stride_b + kw0 * v_stride_s + kv_head * v_stride_h)[:, None] + d_offs[None, :],
                     mask=k_mask0[:, None], other=0.0, care_padding=True)
        acc = acc * a0[:, None] + tl.dot(p0, v0.to(tl.float32))

    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], acc)

    o_base = q_glob[:, None] * o_stride_t + head_idx * o_stride_h
    tl.store(
        o_ptr + o_base + d_offs[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=q_mask[:, None],
    )
