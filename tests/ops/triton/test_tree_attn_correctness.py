#!/usr/bin/env python3
"""Act 3 步骤 3 — Tree Attention 2D gather 正确性验证（layer 1+2）。

Layer 1: 随机输入，tree kernel 输出 vs torch reference（Per-row bias 应用）
Layer 2: tree kernel 输出 vs torch SDPA + 手工 tree causal mask

N_RANDOM_CASES: 随机 seed 数量
"""

import argparse
import pytest
import torch

from vllm_ascend.ops.triton.unified_attention import tree_unified_attention_varlen

_N_CASES = 16


def _mask_to_bias(mask: torch.Tensor) -> torch.Tensor:
    """[T,T] 0/1 visibility → 0/-inf additive bias."""
    bias = torch.zeros_like(mask, dtype=torch.float32)
    bias[mask == 0] = float("-inf")
    return bias


def _ref_per_row(q, k, v, qq_bias, scale, ctx):
    """Torch reference: per-row bias applied to pre-softmax scores."""
    T, H, D = q.shape
    _, Hkv, _ = k.shape
    rep = H // Hkv
    k2 = k.repeat_interleave(rep, dim=1) if Hkv < H else k
    v2 = v.repeat_interleave(rep, dim=1) if Hkv < H else v
    scores = torch.einsum("thd,Thd->thT", q.float(), k2.float()) * scale

    # Manual causal: prefix all visible, draft causal by relative position
    for i in range(T):
        q_abs = ctx + i
        for j in range(T):
            k_abs = j
            if k_abs >= ctx:  # draft region: causal
                if q_abs < k_abs:
                    scores[i, :, j] = float("-inf")
            # prefix: always visible

    # Per-row qq_bias
    if qq_bias is not None:
        for i in range(T):
            bias_i = qq_bias[i, :]  # [T]
            # Only apply to suffix keys (>= ctx)
            mask_i = torch.arange(T) >= ctx
            scores[i, :, :] += bias_i.unsqueeze(0) * mask_i.unsqueeze(0)

    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("thT,Thd->thd", probs, v2.float()).to(q.dtype)


def _ref_sdpa(q, k, v, tree_mask, scale, ctx):
    """SDPA reference with explicit tree causal mask."""
    T, H, D = q.shape
    _, Hkv, _ = k.shape
    rep = H // Hkv
    k2 = k.repeat_interleave(rep, dim=1) if Hkv < H else k
    v2 = v.repeat_interleave(rep, dim=1) if Hkv < H else v

    causal = torch.ones(T, T, dtype=torch.bool)
    for i in range(T):
        for j in range(T):
            if j < ctx:
                causal[i, j] = 1  # prefix: visible
            else:
                causal[i, j] = (i >= j - ctx)  # draft: causal

    mask_combined = tree_mask.bool() & causal
    attn_mask = torch.zeros(1, 1, T, T, device=q.device, dtype=q.dtype)
    attn_mask[:, :, ~mask_combined] = float("-inf")

    q_sdpa = q.transpose(0, 1).unsqueeze(0)  # [1, H, T, D]
    k_sdpa = k2.transpose(0, 1).unsqueeze(0)
    v_sdpa = v2.transpose(0, 1).unsqueeze(0)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, scale=scale,
    )
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


@pytest.mark.parametrize("seed", range(_N_CASES))
def test_layer1_vs_per_row_reference(seed):
    """Layer 1: tree kernel vs torch per-row bias reference."""
    device = torch.device("npu")
    torch.manual_seed(seed)

    T = 5          # total tokens (incl. root)
    ctx = 1        # prefix = root only
    H, Hkv = 4, 2  # GQA
    D = 128
    scale = D ** -0.5
    block_size = 32
    num_pages = (T + block_size - 1) // block_size

    q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(T, Hkv, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(T, Hkv, D, device=device, dtype=torch.bfloat16)

    # Tree mask: chain [(0,),(0,0),(0,0,0),(0,0,0,0)] → 5x5 upper-triangular
    tree_mask = torch.triu(torch.ones(T, T, dtype=torch.int32, device=device))
    bias = _mask_to_bias(tree_mask)

    # Paged KV cache
    k_cache = torch.randn(num_pages, block_size, Hkv, D,
                          device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(num_pages, block_size, Hkv, D,
                          device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    for t in range(T):
        blk = t // block_size
        off = t % block_size
        k_cache[blk, off] = k[t]
        v_cache[blk, off] = v[t]

    # Run kernel
    cu_q = torch.tensor([0, T], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([T], dtype=torch.int32, device=device)
    ctx_lens = torch.tensor([ctx], dtype=torch.int32, device=device)

    out_kernel = tree_unified_attention_varlen(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=block_table, cu_seqlens_q=cu_q,
        seq_lens=seq_lens, context_lens=ctx_lens,
        max_query_len=T, qq_bias=bias, scale=scale,
        block_size=block_size, num_kv_heads=Hkv,
    )

    # Reference
    ref = _ref_per_row(q.cpu(), k.cpu(), v.cpu(), bias.cpu(), scale, ctx)
    diff = (out_kernel.float().cpu() - ref).abs().max().item()
    assert diff < 1e-1, f"Layer 1 FAIL seed={seed} max_abs_diff={diff:.4f}"


@pytest.mark.parametrize("seed", range(_N_CASES))
def test_layer2_vs_sdpa_reference(seed):
    """Layer 2: tree kernel vs torch SDPA + tree causal."""
    device = torch.device("npu")
    torch.manual_seed(seed)

    T, ctx = 6, 1
    H, Hkv = 4, 2
    D = 128
    scale = D ** -0.5
    block_size = 32
    num_pages = (T + block_size - 1) // block_size

    q = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(T, Hkv, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(T, Hkv, D, device=device, dtype=torch.bfloat16)

    tree_mask = torch.triu(torch.ones(T, T, dtype=torch.int32, device=device))
    bias = _mask_to_bias(tree_mask)

    k_cache = torch.randn(num_pages, block_size, Hkv, D,
                          device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(num_pages, block_size, Hkv, D,
                          device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    for t in range(T):
        blk, off = t // block_size, t % block_size
        k_cache[blk, off], v_cache[blk, off] = k[t], v[t]

    cu_q = torch.tensor([0, T], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([T], dtype=torch.int32, device=device)
    ctx_lens = torch.tensor([ctx], dtype=torch.int32, device=device)

    out = tree_unified_attention_varlen(
        q=q, k_cache=k_cache, v_cache=v_cache,
        block_table=block_table, cu_seqlens_q=cu_q,
        seq_lens=seq_lens, context_lens=ctx_lens,
        max_query_len=T, qq_bias=bias, scale=scale,
        block_size=block_size, num_kv_heads=Hkv,
    )

    ref = _ref_sdpa(q.cpu(), k.cpu(), v.cpu(),
                    tree_mask.cpu().bool(), scale, ctx)
    diff = (out.float().cpu() - ref).abs().max().item()
    assert diff < 1e-1, f"Layer 2 FAIL seed={seed} max_abs_diff={diff:.4f}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", choices=["1", "2", "all"], default="all")
    args = p.parse_args()

    if args.layer in ("1", "all"):
        print("=== Layer 1: 2D gather vs per-row reference ===")
        for s in range(_N_CASES):
            test_layer1_vs_per_row_reference(s)
            print(f"  seed={s} PASS")
    if args.layer in ("2", "all"):
        print("=== Layer 2: tree kernel vs SDPA reference ===")
        for s in range(_N_CASES):
            test_layer2_vs_sdpa_reference(s)
            print(f"  seed={s} PASS")
    print("ALL PASS")
