# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
High-level wrapper for tree_unified_attention.

Provides the public API that vllm-ascend's attention layer calls directly,
replacing ``torch_npu.npu_fused_infer_attention_score`` in the tree-attention
code path.
"""

from typing import Optional

import torch
from vllm.triton_utils import tl, triton

from .kernel import tree_unified_attention_varlen_kernel


def tree_unified_attention_varlen(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    max_query_len: int,
    qq_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    block_size: int = 256,
    num_kv_heads: Optional[int] = None,
) -> torch.Tensor:
    """Varlen paged tree attention — single launch for the whole batch.

    Mirrors upstream ``unified_attention`` varlen semantics: sequences are
    concatenated along dim 0 of ``q`` and delimited by ``cu_seqlens_q``; each
    sequence carries its own ``seq_len`` (total KV) / ``context_len`` (prefix)
    and ``block_table`` row.

    Args:
        q: ``[total_query_tokens, num_heads, head_dim]`` (TND), all sequences
            concatenated.
        k_cache / v_cache: paged ``[num_blocks, block_size, num_kv_heads, D]``.
        block_table: ``[num_seqs, max_blocks]`` int32.
        cu_seqlens_q: ``[num_seqs + 1]`` int32 cumulative query boundaries.
        seq_lens: ``[num_seqs]`` int32 per-seq total length.
        context_lens: ``[num_seqs]`` int32 per-seq prefix length
            (``seq_len - query_len``).
        max_query_len: max per-seq query length (grid sizing).
        qq_bias: optional ``[max_q, max_q]`` tree bias, shared across seqs;
            sliced per-row inside the kernel via the sequence-relative q index.
    """
    assert q.ndim == 3, f"q must be [N, H, D] TND, got {q.shape}"
    total_tokens, num_heads, head_dim = q.shape
    assert k_cache.ndim == 4, f"k_cache must be [B, BS, Hkv, D], got {k_cache.shape}"
    num_blocks, bs_cache, num_kv_heads_c, head_dim_c = k_cache.shape
    assert bs_cache == block_size, f"block_size mismatch: {block_size} vs {bs_cache}"
    assert head_dim_c == head_dim
    assert v_cache.shape == k_cache.shape
    assert block_table.ndim == 2, f"block_table must be 2D, got {block_table.shape}"

    num_seqs = block_table.shape[0]
    num_kv_heads_ = num_kv_heads or num_kv_heads_c
    assert num_kv_heads_c == num_kv_heads_

    if qq_bias is not None:
        assert qq_bias.ndim == 2, f"qq_bias must be [Q, Q], got {qq_bias.shape}"

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
    seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    context_lens = context_lens.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(q)

    # qq_bias forces BLOCK_M=1 (Ascend bishengir cannot compile the per-row
    # bias conditioning with BLOCK_M>1); otherwise batch query rows.
    BLOCK_M = 1 if qq_bias is not None else (4 if max_query_len >= 4 else 1)
    BLOCK_N = 128
    qq_bias_stride_0 = qq_bias.stride(0) if qq_bias is not None else 0

    grid = (triton.cdiv(max_query_len, BLOCK_M), num_heads, num_seqs)
    tree_unified_attention_varlen_kernel[grid](
        q, q.stride(0), q.stride(1),
        k_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache, v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table, block_table.stride(0),
        cu_seqlens_q, seq_lens, context_lens,
        qq_bias, qq_bias_stride_0,
        output, output.stride(0), output.stride(1),
        scale,
        num_heads, num_kv_heads_, block_size,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_QQ_BIAS=qq_bias is not None,
    )
    return output


def tree_unified_attention_multiseq(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_loc: torch.Tensor,
    qq_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    block_size: int = 256,
    num_kv_heads: Optional[int] = None,
) -> torch.Tensor:
    """Multi-sequence tree attention (``num_seqs >= 1``).

    Thin compatibility wrapper that assembles ``cu_seqlens_q`` / ``context_lens``
    from ``query_loc`` / ``seq_lens`` and delegates to
    :func:`tree_unified_attention_varlen`.

    Args:
        q: ``[total_query_tokens, num_heads, head_dim]`` (TND), all sequences
            concatenated along dim 0.
        k_cache / v_cache: paged ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        block_table: ``[num_seqs, max_blocks]`` int32 — one row per sequence.
        seq_lens: ``[num_seqs]`` int32 — per-sequence **total** length
            (context + query tokens).
        query_loc: ``[num_seqs + 1]`` int32 — cumulative query-token boundaries.
        qq_bias: optional ``[max_q_per_seq, max_q_per_seq]`` tree bias, shared
            across sequences.
    """
    num_seqs = block_table.shape[0]

    if isinstance(query_loc, torch.Tensor):
        q_bounds = query_loc.tolist()
    else:
        q_bounds = list(query_loc)
    assert len(q_bounds) == num_seqs + 1

    seq_lens_list = seq_lens.tolist()
    context_list = []
    max_q = 0
    for i in range(num_seqs):
        q_len_i = int(q_bounds[i + 1]) - int(q_bounds[i])
        context_list.append(int(seq_lens_list[i]) - q_len_i)
        max_q = max(max_q, q_len_i)

    cu_seqlens_q = torch.tensor(q_bounds, dtype=torch.int32, device=q.device)
    context_lens = torch.tensor(context_list, dtype=torch.int32, device=q.device)

    return tree_unified_attention_varlen(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens=seq_lens,
        context_lens=context_lens,
        max_query_len=max_q,
        qq_bias=qq_bias, scale=scale,
        block_size=block_size, num_kv_heads=num_kv_heads,
    )
