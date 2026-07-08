# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tree attention correctness test — port of upstream vLLM's
``tests/v1/spec_decode/test_tree_attention.py::test_tree_attn_correctness``
(snapshot ``be0dcc29d``, removed by PR #42121) to the Ascend Triton kernel.

Upstream idea (branch-vs-tree equivalence):
  1. Run the whole tree once through tree attention (with qq_bias).
  2. For each query node in the tree, take its visible branch (self + ancestors)
     per the tree mask, run it as a plain causal sequence through a reference
     attention, and assert the tree-attention output for that node matches.

Differences from upstream:
  * device ``npu`` instead of ``cuda``.
  * Tree attention runs through ``tree_unified_attention_multiseq`` (our
    in-house Triton-Ascend kernel) instead of the upstream TREE_ATTN backend.
  * Reference branch is computed with a dense PyTorch SDPA reference instead of
    a second vLLM backend.
  * Covers ``batch_size in [1, 16, 32]`` — exercises the num_seqs>1 path.
"""

import torch
import pytest

from vllm_ascend.ops.triton.unified_attention import (
    tree_unified_attention_multiseq,
)

# Upstream tree masks: row/col 0 is the implicit ROOT. 1 = visible, 0 = masked.
# tree_size_q = mask.shape[0] (includes root).
TREE_ATTN_MASKS = {
    # Chain: [(0,), (0,0), (0,0,0)]  -> 4x4 (root + 3)
    "chain": torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.int32,
    ),
    # Tree: [(0,), (1,), (0,0), (0,1), (1,0), (1,1)]  -> 7x7 (root + 6)
    "tree": torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1],
        ],
        dtype=torch.int32,
    ),
}


def _mask_to_bias(mask: torch.Tensor) -> torch.Tensor:
    """[T,T] 0/1 visibility -> 0/-inf additive bias (float32)."""
    bias = torch.zeros_like(mask, dtype=torch.float32)
    bias[mask == 0] = float("-inf")
    return bias


def _dense_ref_node(q_node, k_ctx, v_ctx, k_vis, v_vis, scale):
    """Dense SDPA reference for ONE tree query node.

    The node attends to: the full prefix context (all visible) + the set of
    tree tokens visible to it per the tree mask (``k_vis``/``v_vis``, all
    visible — they are exactly this node's ancestors + itself). No extra causal
    masking is needed because the visibility set is already the mask row.

    q_node:      [1, H, D]
    k_ctx/v_ctx: [ctx, Hkv, D]
    k_vis/v_vis: [V, Hkv, D]
    Returns: [1, H, D]
    """
    _, H, D = q_node.shape
    Hkv = k_ctx.shape[1]
    k = torch.cat([k_ctx, k_vis], dim=0)
    v = torch.cat([v_ctx, v_vis], dim=0)
    if Hkv < H:
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q_node.float(), k.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,khd->qhd", probs, v.float()).to(q_node.dtype)


@pytest.mark.parametrize("mask_name", list(TREE_ATTN_MASKS.keys()))
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(2, 2), (4, 2)])
@pytest.mark.parametrize("sequence_position", [16, 1024, 2048])
def test_tree_attn_correctness(
    mask_name, batch_size, num_heads, num_kv_heads, sequence_position
):
    """Branch-vs-tree equivalence on NPU, mirroring upstream."""
    device = torch.device("npu")
    torch.manual_seed(42)

    tree_mask = TREE_ATTN_MASKS[mask_name].to(device)
    tree_size_q = tree_mask.shape[0]
    dim_per_head = 128
    block_size = 32
    scale = dim_per_head ** (-0.5)
    seqlen_k = sequence_position + tree_size_q
    num_pages = (seqlen_k + block_size - 1) // block_size

    # Random q/k/v for the tree tokens: [B, tree_size_q, H, D]
    q = torch.randn(batch_size, tree_size_q, num_heads, dim_per_head,
                    device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, tree_size_q, num_kv_heads, dim_per_head,
                    device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, tree_size_q, num_kv_heads, dim_per_head,
                    device=device, dtype=torch.bfloat16)

    # Per-sequence disjoint paged KV cache.
    total_blocks = batch_size * num_pages
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, dim_per_head,
                          device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, dim_per_head,
                          device=device, dtype=torch.bfloat16)
    block_table = torch.zeros(batch_size, num_pages, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for p in range(num_pages):
            block_table[b, p] = b * num_pages + p

    # Write the tree tokens' k/v into the cache at positions
    # [sequence_position : seqlen_k] for every sequence.
    for b in range(batch_size):
        for t in range(tree_size_q):
            pos = sequence_position + t
            blk = block_table[b, pos // block_size].item()
            off = pos % block_size
            k_cache[blk, off] = k[b, t]
            v_cache[blk, off] = v[b, t]

    # ---- Whole-tree attention (qq_bias), batched ----
    qq_bias = _mask_to_bias(tree_mask)
    q_flat = q.reshape(batch_size * tree_size_q, num_heads, dim_per_head)
    seq_lens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    query_loc = [i * tree_size_q for i in range(batch_size + 1)]

    tree_out = tree_unified_attention_multiseq(
        q_flat, k_cache, v_cache, block_table, seq_lens, query_loc,
        qq_bias=qq_bias, scale=scale, block_size=block_size,
        num_kv_heads=num_kv_heads,
    ).reshape(batch_size, tree_size_q, num_heads, dim_per_head)

    # Prefix context (positions [0, sequence_position)) per sequence, dense.
    def ctx_kv(b):
        kc = k_cache.new_empty(sequence_position, num_kv_heads, dim_per_head)
        vc = v_cache.new_empty(sequence_position, num_kv_heads, dim_per_head)
        for pos in range(sequence_position):
            blk = block_table[b, pos // block_size].item()
            off = pos % block_size
            kc[pos] = k_cache[blk, off]
            vc[pos] = v_cache[blk, off]
        return kc, vc

    # ---- Verify each branch against dense reference ----
    for q_index in range(tree_size_q):
        branch_idx = torch.nonzero(tree_mask[q_index, :], as_tuple=True)[0]
        for b in range(batch_size):
            kc, vc = ctx_kv(b)
            q_node = q[b, q_index:q_index + 1]      # [1, H, D]
            k_vis = k[b, branch_idx]                 # [V, Hkv, D]
            v_vis = v[b, branch_idx]
            ref = _dense_ref_node(q_node, kc, vc, k_vis, v_vis, scale)
            got = tree_out[b, q_index:q_index + 1]
            assert torch.allclose(got.float(), ref.float(), atol=7.81e-3), (
                f"mismatch mask={mask_name} bs={batch_size} H={num_heads} "
                f"pos={sequence_position} q_index={q_index} b={b} "
                f"max_diff={(got.float()-ref.float()).abs().max().item():.5f}"
            )
