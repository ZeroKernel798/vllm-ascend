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
import triton.runtime.driver as driver


def get_npu_aicore_num() -> int:
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]


# ---------------------------------------------------------------------------
# qq_bias helper — semantic port of upstream's load_qq_bias_tile
# ---------------------------------------------------------------------------

@triton.jit
def load_qq_bias_tile(
    qq_bias_row_ptr: tl.tensor,    # pointer to qq_bias[row, :] start
    seq_offset: tl.tensor,         # [BLOCK_N] int32 — absolute key positions
    context_len: tl.tensor,        # scalar int32 — prefix length
    qq_bias_stride_0: tl.tensor,   # scalar int32 — stride = tree_len
) -> tl.tensor:
    """
    Load qq_bias values for keys that belong to the **query suffix** region.

    Only keys at positions ``>= context_len`` and ``< context_len + tree_len``
    receive a bias; prefix keys get ``0.0`` (no-op).

    Returns:
        ``[BLOCK_N]`` float32 — bias to add to attention scores.
    """
    key_rel_pos = seq_offset - context_len
    is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)
    return tl.load(
        qq_bias_row_ptr + key_rel_pos,
        mask=is_query_key,
        other=0.0,
    )


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------

# Prevent NaN from exp(-inf) arithmetic
_NEG_LARGE: float = -1e30


@triton.heuristics({
    "USE_QQ_BIAS": lambda args: args["qq_bias"] is not None,
})
@triton.jit(do_not_specialize=["scale"])
def tree_unified_attention_kernel(
    # ---- Q (TND: [N, H, D]) ----
    q_ptr,
    q_stride_t,                     # stride of token dim
    q_stride_h,                     # stride of head dim

    # ---- K cache (paged: [B, BS, Hkv, D]) ----
    k_cache_ptr,
    k_stride_b,                     # stride of block dim
    k_stride_s,                     # stride of position-within-block dim
    k_stride_h,                     # stride of kv-head dim

    # ---- V cache (paged: [B, BS, Hkv, D]) ----
    v_cache_ptr,
    v_stride_b,
    v_stride_s,
    v_stride_h,

    # ---- Block table ([num_seqs, max_blocks] int32) ----
    block_table,
    bt_stride_seq,

    # ---- Sequence length (scalar int) ----
    seq_len,

    # ---- qq_bias ([tree_len, tree_len] fp32 tensor, nullable) ----
    qq_bias,
    qq_bias_stride_0,

    # ---- Output (TND: [N, H, D]) ----
    o_ptr,
    o_stride_t,
    o_stride_h,

    # ---- Scalars (dynamic) ----
    scale,
    num_tokens,                     # N  (dynamic — varies per drafting step)
    num_heads,                      # H
    num_kv_heads,                   # Hkv
    block_size,                     # BS
    max_blocks_per_seq,             # max blocks in block_table
    context_len,                    # prefix length (for qq_bias)

    # ---- Compile-time constants ----
    head_dim: tl.constexpr,         # D  (must be constexpr for tl.arange)
    BLOCK_M: tl.constexpr,          # query token tile
    BLOCK_N: tl.constexpr,          # key tile within page
    USE_QQ_BIAS: tl.constexpr,      # compile-time flag
):
    """
    Paged flash attention with optional tree-attention ``qq_bias``.

    Grid layout
    -----------
    ``grid = (cdiv(num_tokens, BLOCK_M), num_heads)`` — 2D standard
    FlashAttention grid.  Each program handles one head and one
    BLOCK_M-sized chunk of query tokens.

    Inner loops
    -----------
    1. per KV page
    2.   per key block within page (BLOCK_N stride)
    """
    pid_m = tl.program_id(0)        # token-group axis
    head_idx = tl.program_id(1)     # head axis

    # GQA: number of query heads per KV head
    q_heads_per_kv = num_heads // num_kv_heads
    kv_head = head_idx // q_heads_per_kv

    # Feature dimension indices
    d_offs = tl.arange(0, head_dim)                # [D]
    d_mask = d_offs < head_dim

    # Token group: BLOCK_M tokens starting at q_start
    q_start = pid_m * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)        # [BLOCK_M]
    q_mask = q_offs < num_tokens                     # [BLOCK_M]

    # ---- Load Q tile [BLOCK_M, head_dim] in one vectorized access ----
    # TND layout: Q[q_idx, head_idx, d] == q_idx*q_stride_t + head_idx*q_stride_h + d
    q_base = q_offs[:, None] * q_stride_t + head_idx * q_stride_h
    q_tile = tl.load(
        q_ptr + q_base + d_offs[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)                                 # [BLOCK_M, head_dim]

    # ---- 2D online softmax state ----
    NEG_LARGE: tl.constexpr = -1e30
    m_i = tl.full([BLOCK_M], NEG_LARGE, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Number of pages = ceil(seq_len / block_size)
    num_pages = tl.cdiv(seq_len, block_size)

    # ---- Page loop (paged KV access) ----
    for page_idx in range(0, num_pages):
        # Physical block index from block table
        phys_block = tl.load(
            block_table + page_idx
        ).to(tl.int32)

        page_start = page_idx * block_size
        page_end = tl.minimum(page_start + block_size, seq_len)
        page_len = page_end - page_start

        # ---- Sub-block loop within page ----
        for k_start in range(0, page_len, BLOCK_N):
            k_end = tl.minimum(k_start + BLOCK_N, page_len)
            num_k = k_end - k_start

            # Absolute key positions for causal/bias
            k_abs = page_start + k_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
            k_within = k_start + tl.arange(0, BLOCK_N)               # [BLOCK_N]
            k_mask = tl.arange(0, BLOCK_N) < num_k                   # [BLOCK_N]

            # ---- Load K tile [BLOCK_N, head_dim] ----
            k_offs = (
                phys_block * k_stride_b
                + k_within * k_stride_s
                + kv_head * k_stride_h
            )
            k_tile = tl.load(
                k_cache_ptr + k_offs[:, None] + d_offs[None, :],
                mask=k_mask[:, None],
                other=0.0,
                care_padding=False,
            )                                    # [BLOCK_N, head_dim]

            # ---- QK^T: [BLOCK_M, head_dim] × [head_dim, BLOCK_N] → [BLOCK_M, BLOCK_N] ----
            # KEY OPTIMIZATION: one batched tl.dot instead of N separate [1,D]×[D,BN] calls
            kt = tl.trans(k_tile).to(tl.float32)  # [head_dim, BLOCK_N]
            s_2d = tl.dot(q_tile, kt) * scale      # [BLOCK_M, BLOCK_N]

            # ---- 2D Causal mask ----
            q_pos = q_offs[:, None].to(tl.float32)          # [BLOCK_M, 1]
            k_abs_2d = k_abs[None, :].to(tl.float32)        # [1, BLOCK_N]
            ctx_f = context_len.to(tl.float32)
            is_prefix = k_abs_2d < ctx_f                     # [1, BLOCK_N]
            is_draft_causal = q_pos >= (k_abs_2d - ctx_f)   # [BLOCK_M, BLOCK_N]
            causal_mask = (is_prefix | is_draft_causal) & k_mask[None, :]
            s_2d = tl.where(causal_mask, s_2d, NEG_LARGE)

            # ---- qq_bias (tree attention, per-row within BLOCK_M tile) ----
            if USE_QQ_BIAS:
                for i in range(BLOCK_M):
                    q_idx_i = q_start + i
                    if q_idx_i < num_tokens:
                        bias = load_qq_bias_tile(
                            qq_bias + q_idx_i * qq_bias_stride_0,
                            k_abs.to(tl.int32),
                            context_len,
                            qq_bias_stride_0,
                        )  # [BLOCK_N]
                        # Apply bias to row i only, at valid K positions
                        row_i = (tl.arange(0, BLOCK_M) == i)[:, None]  # [BLOCK_M, 1]
                        s_2d = tl.where(row_i & k_mask[None, :],
                                        s_2d + bias[None, :], s_2d)

            # ---- 2D Online softmax update ----
            m_ij = tl.max(tl.where(k_mask[None, :], s_2d, NEG_LARGE),
                          axis=1)                  # [BLOCK_M]
            m_new = tl.maximum(m_i, m_ij)           # [BLOCK_M]
            alpha = tl.exp(m_i - m_new)             # [BLOCK_M]
            p = tl.exp(s_2d - m_new[:, None])       # [BLOCK_M, BLOCK_N]
            l_i = l_i * alpha + tl.sum(p, axis=1)   # [BLOCK_M]
            m_i = m_new

            # ---- Load V tile [BLOCK_N, head_dim] & PV: [BLOCK_M, BLOCK_N] × [BLOCK_N, D] → [BLOCK_M, D] ----
            v_tile = tl.load(
                v_cache_ptr
                + (phys_block * v_stride_b
                   + k_within * v_stride_s
                   + kv_head * v_stride_h)[:, None]
                + d_offs[None, :],
                mask=k_mask[:, None],
                other=0.0,
                care_padding=False,
            )
            acc = acc * alpha[:, None] + tl.dot(p, v_tile.to(tl.float32))

    # ---- Final normalization ----
    acc = tl.where(l_i[:, None] > 0.0,
                   acc / l_i[:, None],
                   acc)

    # ---- Write output [BLOCK_M, head_dim] ----
    o_base = q_offs[:, None] * o_stride_t + head_idx * o_stride_h
    tl.store(
        o_ptr + o_base + d_offs[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=q_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Varlen kernel — single launch handles the whole batch (num_seqs >= 1)
# ---------------------------------------------------------------------------

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
    num_pages = tl.cdiv(seq_len, block_size)

    for page_idx in range(0, num_pages):
        phys_block = tl.load(bt_row + page_idx).to(tl.int32)
        page_start = page_idx * block_size
        page_end = tl.minimum(page_start + block_size, seq_len)
        page_len = page_end - page_start

        for k_start in range(0, page_len, BLOCK_N):
            k_end = tl.minimum(k_start + BLOCK_N, page_len)
            num_k = k_end - k_start
            k_abs = page_start + k_start + tl.arange(0, BLOCK_N)
            k_within = k_start + tl.arange(0, BLOCK_N)
            k_mask = tl.arange(0, BLOCK_N) < num_k

            k_offs = (
                phys_block * k_stride_b
                + k_within * k_stride_s
                + kv_head * k_stride_h
            )
            k_tile = tl.load(
                k_cache_ptr + k_offs[:, None] + d_offs[None, :],
                mask=k_mask[:, None], other=0.0, care_padding=False,
            )
            kt = tl.trans(k_tile).to(tl.float32)
            s_2d = tl.dot(q_tile, kt) * scale

            # Causal mask (relative positions; prefix always visible).
            q_pos = q_rel[:, None].to(tl.float32)
            k_abs_2d = k_abs[None, :].to(tl.float32)
            ctx_f = context_len.to(tl.float32)
            is_prefix = k_abs_2d < ctx_f
            is_draft_causal = q_pos >= (k_abs_2d - ctx_f)
            causal_mask = (is_prefix | is_draft_causal) & k_mask[None, :]
            s_2d = tl.where(causal_mask, s_2d, NEG_LARGE)

            if USE_QQ_BIAS:
                for i in range(BLOCK_M):
                    q_rel_i = pid_m * BLOCK_M + i
                    if q_rel_i < seq_num_q:
                        bias = load_qq_bias_tile(
                            qq_bias + q_rel_i * qq_bias_stride_0,
                            k_abs.to(tl.int32),
                            context_len,
                            qq_bias_stride_0,
                        )
                        row_i = (tl.arange(0, BLOCK_M) == i)[:, None]
                        s_2d = tl.where(row_i & k_mask[None, :],
                                        s_2d + bias[None, :], s_2d)

            m_ij = tl.max(tl.where(k_mask[None, :], s_2d, NEG_LARGE), axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s_2d - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            v_tile = tl.load(
                v_cache_ptr
                + (phys_block * v_stride_b
                   + k_within * v_stride_s
                   + kv_head * v_stride_h)[:, None]
                + d_offs[None, :],
                mask=k_mask[:, None], other=0.0, care_padding=False,
            )
            acc = acc * alpha[:, None] + tl.dot(p, v_tile.to(tl.float32))

    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], acc)

    o_base = q_glob[:, None] * o_stride_t + head_idx * o_stride_h
    tl.store(
        o_ptr + o_base + d_offs[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=q_mask[:, None],
    )


# ---------------------------------------------------------------------------
# FlashDecoding-style Split-KV kernel
# ---------------------------------------------------------------------------

@triton.heuristics({
    "USE_QQ_BIAS": lambda args: args["qq_bias"] is not None,
})
@triton.jit(do_not_specialize=["scale"])
def tree_attention_split_kv_kernel(
    # ---- Q (TND: [N, H, D]) ----
    q_ptr, q_stride_t, q_stride_h,

    # ---- K/V cache (paged) ----
    k_cache_ptr, k_stride_b, k_stride_s, k_stride_h,
    v_cache_ptr, v_stride_b, v_stride_s, v_stride_h,

    # ---- Block table ----
    block_table, bt_stride_seq,

    # ---- Sequence ----
    seq_len,

    # ---- qq_bias ----
    qq_bias, qq_bias_stride_0,

    # ---- Partial output buffers ----
    partial_m_ptr,      # [num_head_groups, num_seq_groups, num_tokens] fp32
    partial_l_ptr,      # same shape
    partial_acc_ptr,    # [num_head_groups, num_seq_groups, num_tokens, head_dim] fp32
    pm_stride_hg,       # stride of head_group dim
    pm_stride_sg,       # stride of seq_group dim
    pm_stride_t,        # stride of token dim
    pa_stride_hg, pa_stride_sg, pa_stride_t, pa_stride_d,

    # ---- Scalars ----
    scale,
    num_tokens, num_heads, num_kv_heads,
    block_size, max_blocks_per_seq, context_len,

    # ---- Compile-time ----
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,          # 1
    BLOCK_N: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    NUM_HEAD_GROUPS: tl.constexpr,
    NUM_SEQ_GROUPS: tl.constexpr,
):
    """
    Split-KV kernel: each core handles a (head_group, seq_group) pair.

    Core (head_group=g_h, seq_group=g_s) processes:
      - Heads in range [g_h * heads_per_group, (g_h+1) * heads_per_group)
      - Pages where page_idx % NUM_SEQ_GROUPS == g_s

    Stores partial (m_i, l_i, acc) to global memory buffers.
    """
    pid = tl.program_id(0)

    # Map pid to (head_group, seq_group)
    seq_group = pid // NUM_HEAD_GROUPS
    head_group = pid % NUM_HEAD_GROUPS

    # Head range for this core
    heads_per_group = tl.cdiv(num_heads, NUM_HEAD_GROUPS)
    head_start = head_group * heads_per_group
    head_end = tl.minimum(head_start + heads_per_group, num_heads)

    q_heads_per_kv = num_heads // num_kv_heads
    num_pages = tl.cdiv(seq_len, block_size)

    d_offs = tl.arange(0, head_dim)
    d_mask = d_offs < head_dim

    # ---- Per-head loop ----
    for head_idx in range(head_start, head_end):
        kv_head = head_idx // q_heads_per_kv
        # Head index within the head group (for partial buffer indexing)
        h_local = head_idx - head_start

        # ---- Per-query-token-group loop (BLOCK_M tokens at a time) ----
        for q_start in range(0, num_tokens, BLOCK_M):
            q_offs_m = q_start + tl.arange(0, BLOCK_M)     # [BLOCK_M]
            q_mask_m = q_offs_m < num_tokens                 # [BLOCK_M]

            # Load Q [BLOCK_M, head_dim]
            q_base_m = q_offs_m[:, None] * q_stride_t + head_idx * q_stride_h
            q_tile = tl.load(
                q_ptr + q_base_m + d_offs[None, :],
                mask=q_mask_m[:, None], other=0.0,
            ).to(tl.float32)                                 # [BLOCK_M, head_dim]

            # 2D online softmax state
            NEG_LARGE: tl.constexpr = -1e30
            m_i = tl.full([BLOCK_M], NEG_LARGE, dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc_m = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

            # ---- Page loop (only pages assigned to this seq_group) ----
            for page_idx in range(seq_group, num_pages, NUM_SEQ_GROUPS):
                phys_block = tl.load(
                    block_table + page_idx
                ).to(tl.int32)

                page_start = page_idx * block_size
                page_end = tl.minimum(page_start + block_size, seq_len)
                page_len = page_end - page_start

                # ---- Sub-block loop within page ----
                for k_start in range(0, page_len, BLOCK_N):
                    k_end = tl.minimum(k_start + BLOCK_N, page_len)
                    num_k = k_end - k_start

                    k_abs = page_start + k_start + tl.arange(0, BLOCK_N)
                    k_within = k_start + tl.arange(0, BLOCK_N)
                    k_mask = tl.arange(0, BLOCK_N) < num_k

                    # Load K tile [BLOCK_N, head_dim]
                    k_offs = (
                        phys_block * k_stride_b
                        + k_within * k_stride_s
                        + kv_head * k_stride_h
                    )
                    k_tile = tl.load(
                        k_cache_ptr + k_offs[:, None] + d_offs[None, :],
                        mask=k_mask[:, None], other=0.0,
                        care_padding=False,
                    )

                    # QK^T: [BLOCK_M, head_dim] × [head_dim, BLOCK_N] → [BLOCK_M, BLOCK_N]
                    kt = tl.trans(k_tile).to(tl.float32)
                    s_2d = tl.dot(q_tile, kt) * scale  # [BLOCK_M, BLOCK_N]

                    # 2D Causal mask
                    q_pos = q_offs_m[:, None].to(tl.float32)         # [BLOCK_M, 1]
                    k_abs_2d = k_abs[None, :].to(tl.float32)         # [1, BLOCK_N]
                    ctx_f = context_len.to(tl.float32)
                    is_prefix = k_abs_2d < ctx_f
                    is_draft_causal = q_pos >= (k_abs_2d - ctx_f)
                    causal_mask = (is_prefix | is_draft_causal) & k_mask[None, :]
                    s_2d = tl.where(causal_mask, s_2d, NEG_LARGE)

                    # qq_bias (per-row within BLOCK_M tile)
                    if USE_QQ_BIAS:
                        for i in range(BLOCK_M):
                            q_idx_i = q_start + i
                            if q_idx_i < num_tokens:
                                bias = load_qq_bias_tile(
                                    qq_bias + q_idx_i * qq_bias_stride_0,
                                    k_abs.to(tl.int32),
                                    context_len, qq_bias_stride_0,
                                )
                                row_i = (tl.arange(0, BLOCK_M) == i)[:, None]
                                s_2d = tl.where(row_i & k_mask[None, :],
                                                s_2d + bias[None, :], s_2d)

                    # 2D Online softmax update
                    m_ij = tl.max(tl.where(k_mask[None, :], s_2d, NEG_LARGE),
                                  axis=1)                    # [BLOCK_M]
                    m_new = tl.maximum(m_i, m_ij)
                    alpha = tl.exp(m_i - m_new)              # [BLOCK_M]
                    p = tl.exp(s_2d - m_new[:, None])        # [BLOCK_M, BLOCK_N]
                    l_i = l_i * alpha + tl.sum(p, axis=1)    # [BLOCK_M]
                    m_i = m_new

                    # PV: [BLOCK_M, BLOCK_N] × [BLOCK_N, head_dim] → [BLOCK_M, head_dim]
                    v_tile = tl.load(
                        v_cache_ptr
                        + (phys_block * v_stride_b
                           + k_within * v_stride_s
                           + kv_head * v_stride_h)[:, None]
                        + d_offs[None, :],
                        mask=k_mask[:, None], other=0.0,
                        care_padding=False,
                    )
                    acc_m = acc_m * alpha[:, None] + tl.dot(p, v_tile.to(tl.float32))

            # ---- Write partial results for all BLOCK_M query tokens ----
            q_write_offs = q_start + tl.arange(0, BLOCK_M)    # [BLOCK_M]
            q_write_mask = q_write_offs < num_tokens            # [BLOCK_M]

            pm_offs = (
                head_group * pm_stride_hg
                + seq_group * pm_stride_sg
                + h_local * num_tokens       # FIX: interleave by num_tokens
                + q_write_offs
            )
            tl.store(partial_m_ptr + pm_offs, m_i, mask=q_write_mask)
            tl.store(partial_l_ptr + pm_offs, l_i, mask=q_write_mask)

            pa_offs = (
                head_group * pa_stride_hg
                + seq_group * pa_stride_sg
                + (h_local * num_tokens + q_write_offs[:, None]) * pa_stride_t
                + d_offs[None, :]
            )
            tl.store(partial_acc_ptr + pa_offs, acc_m,
                     mask=q_write_mask[:, None] & d_mask[None, :])


# ---------------------------------------------------------------------------
# FlashDecoding Reduction kernel
# ---------------------------------------------------------------------------

@triton.jit
def tree_attention_reduce_kernel(
    # ---- Partial inputs ----
    partial_m_ptr,      # [num_head_groups, num_seq_groups, heads_per_group * num_tokens]
    partial_l_ptr,
    partial_acc_ptr,    # [num_head_groups, num_seq_groups, heads_per_group * num_tokens, head_dim]
    pm_stride_hg, pm_stride_sg, pm_stride_t,
    pa_stride_hg, pa_stride_sg, pa_stride_t, pa_stride_d,

    # ---- Output ----
    o_ptr,              # [num_tokens, num_heads, head_dim]
    o_stride_t, o_stride_h,

    # ---- Scalars ----
    num_tokens,
    num_heads,
    heads_per_group,    # heads covered by each head_group

    # ---- Compile-time ----
    head_dim: tl.constexpr,
    NUM_SEQ_GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr,   # tile for head_dim in reduction
):
    """
    Combine partial (m, l, acc) from all seq_groups.

    For each (head_group, head_local, query_token):
      m_final = max(m across seq_groups)
      l_final = sum(l_s * exp(m_s - m_final))
      acc_final = sum(acc_s * l_s * exp(m_s - m_final)) / l_final
    """
    pid = tl.program_id(0)

    # Map pid to (head_group, h_local, q_idx)
    total_tasks = NUM_SEQ_GROUPS * heads_per_group * num_tokens  # dummy, actually use 1D
    # pid maps to a specific (head_group, h_local, q_idx) to reduce
    # Total tasks: NUM_SEQ_GROUPS * heads_per_group * num_tokens
    # But we only need one reducer per (head_group, h_local, q_idx)
    # Actually, we need to reduce across seq_groups per (head_group, h_local, q_idx)
    # So total reduction tasks = NUM_HEAD_GROUPS * heads_per_group * num_tokens
    # But we don't have NUM_HEAD_GROUPS as constexpr. Instead, use grid = total_tasks
    # and map pid to (head_group, h_local, q_idx):

    # head_idx = pid // (num_tokens * heads_per_group)  # Not needed, reduce within head_group
    # Actually, each pid handles one (head_group * heads_per_group + h_local, q_idx) pair

    # Global head index
    head_idx = pid // num_tokens
    q_idx = pid % num_tokens

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    # Determine which head_group this head belongs to
    head_group = head_idx // heads_per_group
    h_local = head_idx % heads_per_group

    # Reduction: m_final = max over seq_groups, l_final = weighted sum
    # Use -1e30 instead of -inf to avoid NaN from tl.exp(-inf - (-inf))
    # in the dead branch of tl.where.
    NEG_LARGE = -1e30
    m_final = NEG_LARGE
    l_weighted = 0.0
    acc_final = tl.zeros([BLOCK_D], dtype=tl.float32)

    for sg in range(0, NUM_SEQ_GROUPS):
        pm_off = (
            head_group * pm_stride_hg
            + sg * pm_stride_sg
            + h_local * num_tokens     # FIX: interleave by num_tokens
            + q_idx
        )
        m_s = tl.load(partial_m_ptr + pm_off)
        l_s = tl.load(partial_l_ptr + pm_off)

        pa_off = (
            head_group * pa_stride_hg
            + sg * pa_stride_sg
            + (h_local * num_tokens + q_idx) * pa_stride_t
        )
        acc_s = tl.load(partial_acc_ptr + pa_off + d_offs, mask=d_mask, other=0.0)

        # Only process groups that actually contributed (l_s > 0)
        # Use is_first to detect the first real contribution
        is_first = (l_weighted == 0.0)
        has_contrib = l_s > 0.0

        # m_new: if first, use m_s directly; otherwise max(m_final, m_s)
        m_new = tl.where(
            has_contrib,
            tl.where(is_first, m_s, tl.maximum(m_final, m_s)),
            m_final,
        )
        # rescale: exp(m_final - m_new), but 1.0 for first contribution or no contribution
        rescale = tl.where(
            has_contrib,
            tl.where(is_first, 1.0, tl.exp(m_final - m_new)),
            1.0,
        )

        acc_final = acc_final * rescale
        l_weighted = l_weighted * rescale

        # Add new contribution
        weight = tl.where(has_contrib, l_s * tl.exp(m_s - m_new), 0.0)
        acc_final = tl.where(has_contrib, acc_final + acc_s * weight, acc_final)
        l_weighted = tl.where(has_contrib, l_weighted + weight, l_weighted)
        m_final = m_new

    # Final normalization
    acc_final = tl.where(l_weighted > 0.0, acc_final / l_weighted, acc_final)

    # Write output
    o_off = q_idx * o_stride_t + head_idx * o_stride_h
    tl.store(
        o_ptr + o_off + d_offs,
        acc_final.to(o_ptr.dtype.element_ty),
        mask=d_mask,
    )
