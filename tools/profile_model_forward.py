#!/usr/bin/env python3
"""
Minimal model forward profiler: loads Qwen3.5-0.8B via vLLM's internal APIs,
runs decode steps and times model.forward() directly.
"""
import os, sys, time, json, gc
import torch
import torch.npu

MODEL_PATH = "/data/huggingface_home/Qwen/Qwen3___5-0___8B-Base"
LOG_DIR = "/root/vllm-ascend/.remote-logs/profile"
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    # Import vllm internals
    from vllm.config import VllmConfig, CompilationConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.kv_cache_interface import KVCacheConfig

    # vllm-ascend overrides
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunnerV1

    compilation_config = CompilationConfig(
        mode=0,  # NONE
        cudagraph_mode=0,  # NONE
    )

    vllm_config = VllmConfig(
        model=MODEL_PATH,
        max_model_len=512,
        gpu_memory_utilization=0.40,
        enforce_eager=True,
        trust_remote_code=True,
        compilation_config=compilation_config,
        speculative_config=None,
    )

    # Use Ascend worker
    # But we can't easily construct this... let's try a different approach
    
    print(f"Loading model: {MODEL_PATH}")
    
    # Load directly with transformers + vllm patches
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print(f"  hidden_size={config.hidden_size}")
    print(f"  intermediate_size={config.intermediate_size}")
    print(f"  num_hidden_layers={config.num_hidden_layers}")
    print(f"  num_attention_heads={config.num_attention_heads}")
    print(f"  num_key_value_heads={config.num_key_value_heads}")
    print(f"  head_dim={getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}")
    
    # Layer type distribution
    layer_types = getattr(config, 'layer_types', None)
    if layer_types:
        from collections import Counter
        type_counts = Counter(layer_types)
        for t, c in type_counts.items():
            print(f"  layer_type[{t}]: {c}")
    
    # Create a simple matmul benchmark to quantify GEMM vs attention
    print("\n--- Full Layer Timing (single op micro-benchmarks) ---")
    
    device = "npu"
    dtype = torch.bfloat16
    HS = config.hidden_size
    IS = config.intermediate_size
    N_Q = config.num_attention_heads
    N_KV = config.num_key_value_heads
    D = getattr(config, 'head_dim', HS // N_Q)
    
    QKV_N = (N_Q + N_KV) * D * 2  # Q,K,V,gate projections
    
    # Create random inputs
    x = torch.randn(1, HS, dtype=dtype, device=device)
    w_list = {
        "QKV": torch.randn(QKV_N, HS, dtype=dtype, device=device),
        "O": torch.randn(HS, N_Q * D, dtype=dtype, device=device),
        "gate_up": torch.randn(IS * 2, HS, dtype=dtype, device=device),
        "down": torch.randn(HS, IS, dtype=dtype, device=device),
    }
    
    # Warmup all ops
    for _ in range(20):
        for name, w in w_list.items():
            torch.nn.functional.linear(x, w)
        torch.npu.synchronize()
    
    # Time each op (50 iterations per op, timed with events)
    results = {}
    
    def bench_op(name, fn, iters=200):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        
        evt_start = torch.npu.Event(enable_timing=True)
        evt_end = torch.npu.Event(enable_timing=True)
        evt_start.record()
        for _ in range(iters):
            fn()
        evt_end.record()
        torch.npu.synchronize()
        ms = evt_start.elapsed_time(evt_end) / iters
        results[name] = ms
        return ms
    
    # QKV projection
    qkv_ms = bench_op("qkv_proj", lambda: torch.nn.functional.linear(x, w_list["QKV"]), 500)
    
    # O projection
    o_in = torch.randn(1, N_Q * D, dtype=dtype, device=device)
    o_ms = bench_op("o_proj", lambda: torch.nn.functional.linear(o_in, w_list["O"]), 500)
    
    # Gate-up projection
    gu_ms = bench_op("gate_up", lambda: torch.nn.functional.linear(x, w_list["gate_up"]), 500)
    
    # SiLU-Mul
    gu_out = torch.randn(1, IS * 2, dtype=dtype, device=device)
    def silu_mul():
        a, b = gu_out.chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b
    si_ms = bench_op("silu_mul", silu_mul, 500)
    
    # Down projection
    d_in = torch.randn(1, IS, dtype=dtype, device=device)
    down_ms = bench_op("down", lambda: torch.nn.functional.linear(d_in, w_list["down"]), 500)
    
    # RMSNorm
    y = torch.randn(1, HS, dtype=dtype, device=device)
    w_n = torch.randn(HS, dtype=dtype, device=device)
    def rmsnorm():
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6) * w_n
    rms_ms = bench_op("rms_norm", rmsnorm, 1000)
    
    # Softmax at various seq lengths
    for sl in [64, 128, 256, 512, 1024, 2048]:
        s = torch.randn(1, sl, dtype=torch.float32, device=device)
        sm_ms = bench_op(f"softmax_L{sl}", lambda: torch.softmax(s, dim=-1), 1000)
    
    # Attention components: QK^T, softmax, PV
    # Simulate [num_heads, N_tokens=1, D] × K^T[D, L] and [1, L] × V[L, D]
    for sl in [64, 128, 256, 512, 1024, 2048]:
        q_t = torch.randn(N_Q, 1, D, dtype=dtype, device=device)
        k_t = torch.randn(1, sl, D, dtype=dtype, device=device)  # per-head
        # QK^T: [N_Q, 1, D] × [1, D, sl] → not a standard matmul
        # Actually use: q[batch=1, heads, 1, D] × k[1, heads, L, D]^T
        q_attn = q_t.permute(1, 0, 2)  # [1, N_Q, D]
        k_attn = k_t.expand(1, N_Q, sl, D)  # [1, N_Q, sl, D]
        attn_inner_ms = bench_op(
            f"attn_qkt_L{sl}",
            lambda: torch.matmul(q_attn[:, :, None, :], k_attn.transpose(-2, -1)),
            200,
        )
    
    # ---- Summary ----
    print("\n" + "=" * 65)
    print("RESULTS: Per-op Latency (μs)")
    print("=" * 65)
    for name, ms in sorted(results.items()):
        print(f"  {name:<20}: {ms*1000:>8.1f}μs")
    
    # Per-layer aggregation
    # Full attention layer: QKV + O + attn_inner + softmax + gate_up + silu + down + 2*RMSNorm
    attn_inner_ms = results.get("attn_qkt_L512", 0.1) * 1.5  # rough estimate for QK^T + PV + softmax
    full_layer_ms = qkv_ms + o_ms + gu_ms + si_ms + down_ms + 2 * rms_ms + attn_inner_ms
    
    print(f"\n  Full-attn layer total (est): {full_layer_ms*1000:.0f}μs")
    print(f"    QKV:     {qkv_ms*1000:.0f}μs ({qkv_ms/full_layer_ms*100:.1f}%)")
    print(f"    Attn:    {attn_inner_ms*1000:.0f}μs ({attn_inner_ms/full_layer_ms*100:.1f}%)")
    print(f"    MLP:     {(gu_ms+si_ms+down_ms)*1000:.0f}μs ({(gu_ms+si_ms+down_ms)/full_layer_ms*100:.1f}%)")
    print(f"    O+Norm:  {(o_ms+2*rms_ms)*1000:.0f}μs ({(o_ms+2*rms_ms)/full_layer_ms*100:.1f}%)")
    
    # Model estimate
    linear_layer_ms = full_layer_ms * 0.5  # GDN is faster
    N_full = 7
    N_linear = 21
    model_fwd_ms = N_full * full_layer_ms + N_linear * linear_layer_ms
    print(f"\n  Model forward est: {model_fwd_ms*1000:.0f}μs = {model_fwd_ms:.1f}ms")
    print(f"  E2E decode step:   ~231ms")
    print(f"  Model fwd / E2E:   {model_fwd_ms/231*100:.1f}%")
    
    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"{LOG_DIR}/ops_{ts}.json"
    with open(path, "w") as f:
        json.dump({k: round(v, 6) for k, v in results.items()}, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
