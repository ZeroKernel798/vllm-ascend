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

    # BLOCK_M = 16 forces bishengir to map tl.dot to a single Cube Mmad call
    # (16 = CUBE_BLOCK, avoids scalar-loop fallback seen in TTIR for BM < 16).
    # Inactive query rows are zeroed via causal mask in the kernel.
    BLOCK_M = 16
    qq_bias_stride_0 = qq_bias.stride(0) if qq_bias is not None else 0
    # BLOCK_M=16 may trigger compilation of the tl.load(qq_bias+...) path even
    # when USE_QQ_BIAS=False.  Feed a dummy tensor to satisfy the type-checker.
    _qq_bias = qq_bias if qq_bias is not None else torch.empty(1, 1, dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(max_query_len, BLOCK_M), num_heads, num_seqs)
    tree_unified_attention_varlen_kernel[grid](
        q, q.stride(0), q.stride(1),
        k_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache, v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table, block_table.stride(0),
        cu_seqlens_q, seq_lens, context_lens,
        _qq_bias, qq_bias_stride_0,
        output, output.stride(0), output.stride(1),
        scale,
        num_heads, num_kv_heads_, block_size,
        num_blocks, block_table.shape[1],
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        USE_QQ_BIAS=qq_bias is not None,
    )
    return output

