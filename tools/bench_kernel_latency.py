#!/usr/bin/env python3
"""Kernel latency benchmark: custom Triton vs CANN FIA vendor kernel."""
import json, sys, time, os, argparse
import torch
from vllm_ascend.ops.triton.unified_attention import tree_unified_attention_varlen


def bench(fn, *args, warmup=10, iters=50, **kwargs):
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(*args, **kwargs)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def make_paged_kv(num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
    return torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                       dtype=dtype, device=device)


def bench_triton(q, kc, vc, bt, sl_t, cq, cl, nt, qq_bias, scale, bs, nkv, iters):
    return bench(
        tree_unified_attention_varlen,
        q, kc, vc, bt, cq, sl_t, cl, nt,
        qq_bias=qq_bias, scale=scale, block_size=bs, num_kv_heads=nkv,
        warmup=5, iters=iters,
    )


def bench_cann_fia(q, kc, vc, bt, sl_t, nt, ctx, qq_bias, scale, bs, nkv, iters):
    """CANN FIA — transpose KV to [B, H, S, D] layout."""
    assert qq_bias is None, "CANN only supports non-tree attention"
    import torch_npu
    pad_len = kc.shape[0] * bs  # match KV cache total capacity
    mask = torch.zeros(1, 1, nt, pad_len, dtype=torch.bool, device=q.device)
    for i in range(nt):
        mask[0, 0, i, :ctx + i + 1] = True
    k_c = kc.transpose(1, 2).contiguous()
    v_c = vc.transpose(1, 2).contiguous()
    q_b = q.transpose(0, 1).unsqueeze(0)
    out = torch.empty_like(q_b)
    lse = torch.empty(1, q.shape[1], nt, dtype=torch.float32, device=q.device)
    return bench(
        torch_npu.npu_fused_infer_attention_score,
        query=q_b, key=k_c, value=v_c, atten_mask=mask, out=(out, lse),
        num_heads=q.shape[1], num_key_value_heads=nkv,
        scale=scale, input_layout="BNSD", block_size=bs,
        block_table=bt, actual_seq_lengths=sl_t, actual_seq_lengths_kv=sl_t,
        sparse_mode=0, warmup=3, iters=iters,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    print(f"=== Triton vs CANN FIA (Ascend 910B2C) ===\n")
    results = []
    BS = 256
    HD = 128

    CONFIGS = [
        ("decode_1T_2K",  1, 8, 4, 2048),
        ("decode_1T_4K",  1, 8, 4, 4096),
        ("decode_1T_8K",  1, 8, 4, 8192),
        ("decode_4T_2K",  4, 8, 4, 2048),
        ("prefill_128T",  128, 8, 4, 256),
    ]

    for name, nt, nh, nkv, sl in CONFIGS:
        dev = torch.device("npu")
        nb = (sl + BS - 1) // BS + 1
        ctx = sl - nt

        q = torch.randn(nt, nh, HD, dtype=torch.float16, device=dev)
        kc = make_paged_kv(nb, BS, nkv, HD, torch.float16, dev)
        vc = make_paged_kv(nb, BS, nkv, HD, torch.float16, dev)
        bt = torch.zeros(1, nb, dtype=torch.int32, device=dev)
        for i in range((sl + BS - 1) // BS):
            bt[0, i] = i

        sl_t = torch.tensor([sl], dtype=torch.int32, device=dev)
        cq = torch.tensor([0, nt], dtype=torch.int32, device=dev)
        cl = torch.tensor([ctx], dtype=torch.int32, device=dev)
        scale = 1.0 / (HD ** 0.5)

        ms_t = bench_triton(q, kc, vc, bt, sl_t, cq, cl, nt, None, scale, BS, nkv, args.iters)
        ms_c = bench_cann_fia(q, kc, vc, bt, sl_t, nt, ctx, None, scale, BS, nkv, args.iters)
        ratio = ms_t / ms_c
        tag = "√ triton faster" if ratio < 1 else "× slower"
        print(f"  {name:<18} t={ms_t:.4f}ms  c={ms_c:.4f}ms  {ratio:.2f}x {tag}")
        results.append({"name": name, "triton_ms": round(ms_t, 4), "cann_ms": round(ms_c, 4), "ratio": round(ratio, 2)})

    log_dir = "/root/vllm-ascend/.remote-logs/kernel_bench"
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(f"{log_dir}/triton_vs_cann_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)

    faster = sum(1 for r in results if r["ratio"] < 1)
    avg = sum(r["ratio"] for r in results) / len(results)
    print(f"\n  {faster}/{len(results)} faster, avg {avg:.2f}x")

if __name__ == "__main__":
    main()
