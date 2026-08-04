#!/usr/bin/env python3
"""Act 3 步骤 3-4 — vLLM tree attention e2e 验证（修正 ctx 问题）"""
import argparse, os, sys, time
os.environ["VLLM_ASCEND_USE_TRITON_TREE_ATTENTION"] = "1"

import torch
import torch_npu; DEVICE = torch.device("npu")
from vllm_ascend.ops.triton.unified_attention import tree_unified_attention_varlen


def make_inputs(T, H, Hkv, D, ctx, BS=32):
    total = ctx + T; NP = (total + BS - 1) // BS
    q = torch.randn(T, H, D, device=DEVICE, dtype=torch.bfloat16)
    # Full k/v for ALL tokens (prefix + tree)
    k_all = torch.randn(total, Hkv, D, device=DEVICE, dtype=torch.bfloat16)
    v_all = torch.randn(total, Hkv, D, device=DEVICE, dtype=torch.bfloat16)
    kc = torch.zeros(NP, BS, Hkv, D, device=DEVICE, dtype=torch.bfloat16)
    vc = torch.zeros(NP, BS, Hkv, D, device=DEVICE, dtype=torch.bfloat16)
    for pos in range(total):
        pg, off = pos // BS, pos % BS
        kc[pg, off] = k_all[pos]; vc[pg, off] = v_all[pos]
    bt = torch.zeros(1, NP, dtype=torch.int32, device=DEVICE)
    for p in range(NP): bt[0, p] = p
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
    sl = torch.tensor([total], dtype=torch.int32, device=DEVICE)
    cl = torch.tensor([ctx], dtype=torch.int32, device=DEVICE)
    return q, kc, vc, bt, cu, sl, cl, k_all, v_all


def run_tree(T, H, Hkv, D, ctx, bias, scale):
    q, kc, vc, bt, cu, sl, cl, k_all, v_all = make_inputs(T, H, Hkv, D, ctx)
    torch.npu.synchronize(); t0 = time.perf_counter()
    out = tree_unified_attention_varlen(q=q, k_cache=kc, v_cache=vc, block_table=bt,
        cu_seqlens_q=cu, seq_lens=sl, context_lens=cl, max_query_len=T,
        qq_bias=bias, scale=scale, block_size=32, num_kv_heads=Hkv)
    torch.npu.synchronize(); ms = (time.perf_counter() - t0) * 1000
    return out, ms, q, k_all, v_all


def ref_attn(q_cpu, k_all, v_all, bias, scale, T, H, Hkv, ctx):
    """Full reference: prefix + tree tokens, causal mask, tree bias."""
    total = ctx + T
    k_all = k_all.float()
    v_all = v_all.float()
    k2 = k_all.repeat_interleave(H // Hkv, dim=1) if Hkv < H else k_all
    v2 = v_all.repeat_interleave(H // Hkv, dim=1) if Hkv < H else v_all
    scores = torch.einsum("thd,Thd->thT", q_cpu.float(), k2) * scale
    # Causal: prefix (j<ctx) visible, draft causal
    for i in range(T):
        for j in range(total):
            if j >= ctx and i < (j - ctx):
                scores[i, :, j] = -1e9
    # Tree bias on suffix keys
    if bias is not None:
        bc = bias.float()
        for i in range(T):
            for j in range(ctx, total):
                scores[i, :, j] += bc[i, j - ctx]
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("thT,Thd->thd", probs, v2)


def test_correctness():
    print("=== Layer 3: E2E correctness ===")
    H, Hkv, D = 8, 2, 128; scale = D ** -0.5
    configs = [
        ("T=3 ctx=4",      3, 4),
        ("T=4 ctx=8",      4, 8),
        ("T=5 ctx=16",     5, 16),
        ("T=4 ctx=4 GQA",  4, 4),
    ]
    passed, failed = 0, 0
    for name, T, ctx in configs:
        mask = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.int32))
        bias = torch.where(mask == 0,
                          torch.tensor(float('-inf'), device=DEVICE, dtype=torch.float32),
                          torch.zeros_like(mask, dtype=torch.float32))
        out, ms, q, k_all, v_all = run_tree(T, H, Hkv, D, ctx, bias, scale)
        has_nan = torch.isnan(out).any().item()
        if has_nan:
            print(f"  {name}: FAIL (NaN)"); failed += 1; continue
        ref = ref_attn(q.cpu(), k_all.cpu(), v_all.cpu(), bias.cpu(), scale, T, H, Hkv, ctx)
        diff = (out.float().cpu() - ref).abs().max().item()
        ok = "PASS" if diff < 6e-2 else "FAIL"
        print(f"  {name:14s} ms={ms:6.2f} diff={diff:.5f}  {ok}")
        if diff < 6e-2: passed += 1
        else: failed += 1
    print(f"\n{passed}/{passed+failed} PASSED")
    return failed == 0


def test_perf():
    print("\n=== Layer 4: Quick perf comparison ===")
    H, Hkv, D = 32, 8, 128; scale = D ** -0.5
    W, I = 3, 5
    ctx = 512
    for T in [1, 2, 4, 6, 8]:
        mask = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.int32))
        bias = torch.where(mask == 0, torch.tensor(float('-inf'), device=DEVICE, dtype=torch.float32),
                          torch.zeros_like(mask, dtype=torch.float32))
        for _ in range(W): run_tree(T, H, Hkv, D, ctx, bias, scale)
        times, times_lin = [], []
        for _ in range(I):
            _, ms, _, _, _ = run_tree(T, H, Hkv, D, ctx, bias, scale); times.append(ms)
        for _ in range(I):
            q, kc, vc, bt, cu, sl, cl, _, _ = make_inputs(T, H, Hkv, D, ctx)
            torch.npu.synchronize(); t0 = time.perf_counter()
            _ = tree_unified_attention_varlen(q=q, k_cache=kc, v_cache=vc, block_table=bt,
                cu_seqlens_q=cu, seq_lens=sl, context_lens=cl, max_query_len=T,
                qq_bias=None, scale=scale, block_size=32, num_kv_heads=Hkv)
            torch.npu.synchronize(); times_lin.append((time.perf_counter() - t0) * 1000)
        med = sorted(times)[len(times)//2]
        med_lin = sorted(times_lin)[len(times_lin)//2]
        r = med / med_lin if med_lin else 0
        print(f"  T={T} tree={med:.3f}ms linear={med_lin:.3f}ms ratio={r:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    if not test_correctness():
        print("\nCORRECTNESS FAILED"); sys.exit(1)
    print("\nCORRECTNESS ALL PASSED")
    if args.benchmark: test_perf()
