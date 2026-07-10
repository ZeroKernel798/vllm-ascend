#!/usr/bin/env python3
"""Operator-level precision microbench for torchao NPU fast-paths.

For each quantization type (int8wo / w8a8 / intx4wo), on a single
``[in_features, out_features]`` Linear layer, compute the relative error
of the native NPU kernel vs the fp32 ground-truth, and vs the portable
dequant path.  No model load required — this validates the numerical
fidelity of the kernels themselves.

Usage:
  python tools/benchmark_torchao_precision.py                              # default shapes
  python tools/benchmark_torchao_precision.py --in 4096 --out 4096 --tokens 256
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


DEVICE = "npu:0" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"


def _sdiv(a: Optional[float], b: Optional[float]) -> float:
    return a / b if a and b else 0.0


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.float(), b.float()
    return ((x - y).abs() / (y.abs() + 1e-8)).mean().item()


def _quantize_linear(
    weight: torch.Tensor,
    config: str = "int8wo",
) -> Tuple[torch.nn.Module, torch.Tensor]:
    """Return (quantized linear layer, dequantized fp32 reference weight)."""
    in_f, out_f = weight.shape[1], weight.shape[0]

    layer = torch.nn.Linear(in_f, out_f, bias=False)
    layer.weight.data.copy_(weight)
    ref = weight.float().clone()

    from torchao.quantization import (
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        IntxWeightOnlyConfig,
        quantize_,
    )
    if config == "int8wo":
        quantize_(layer, Int8WeightOnlyConfig())
    elif config == "w8a8":
        quantize_(layer, Int8DynamicActivationInt8WeightConfig())
    elif config == "intx4wo":
        quantize_(layer, IntxWeightOnlyConfig(weight_dtype=torch.int4))
    else:
        raise ValueError(f"Unknown config: {config}")

    layer = layer.to(DEVICE)
    ref = ref.to(DEVICE)
    return layer, ref


def compute_errors(
    in_features: int = 4096,
    out_features: int = 4096,
    tokens: int = 256,
    warmup: int = 5,
    iters: int = 50,
) -> dict:
    import time

    torch.manual_seed(0)
    x = torch.randn(tokens, in_features, device=DEVICE, dtype=torch.bfloat16)
    x_fp32 = x.float()

    def _sync():
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()

    def _timed(fn, *a, **kw):
        for _ in range(warmup):
            fn(*a, **kw)
        _sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*a, **kw)
        _sync()
        return (time.perf_counter() - t0) / iters * 1000  # ms per call

    results = {}
    for name, config in [("int8wo", "int8wo"), ("w8a8", "w8a8"), ("intx4wo", "intx4wo")]:
        try:
            weight = torch.randn(out_features, in_features)
            layer, ref_w = _quantize_linear(weight, config)

            # fp32 ground truth
            with torch.no_grad():
                y_ref = F.linear(x_fp32, ref_w).float()

            # dequant path (F.linear on bf16 activation → implicit dequant)
            lat_dq = _timed(lambda: F.linear(x, layer.weight))
            with torch.no_grad():
                y_dq = F.linear(x, layer.weight).float()

            # native NPU path
            try:
                import torch_npu  # noqa: F401
                from vllm_ascend.quantization.torchao_config import AscendTorchAOConfig, AscendTorchAOLinearMethod

                cfg = AscendTorchAOConfig()
                method = AscendTorchAOLinearMethod(cfg)
                method.process_weights_after_loading(layer)
                with torch.no_grad():
                    y_native = method.apply(layer, x).float()
                lat_native = _timed(lambda: method.apply(layer, x))
            except Exception:
                y_native = None
                lat_native = None

            err_dq = _rel_err(y_dq, y_ref)
            err_native = _rel_err(y_native, y_ref) if y_native is not None else None
            native_vs_dq = _rel_err(y_native, y_dq) if y_native is not None else None

            results[name] = {
                "dequant_vs_fp32": round(err_dq, 6),
                "native_vs_fp32": round(err_native, 6) if err_native else None,
                "native_vs_dequant": round(native_vs_dq, 6) if native_vs_dq else None,
                "lat_dequant_ms": round(lat_dq, 3),
                "lat_native_ms": round(lat_native, 3) if lat_native else None,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def main():
    import argparse

    p = argparse.ArgumentParser(description="TorchAO operator precision microbench")
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations for latency")
    p.add_argument("--iters", type=int, default=50, help="Timed iterations for latency")
    args = p.parse_args()

    print(f"=== Operator Precision + Latency: {args.in_features}x{args.out_features}, {args.tokens} tokens ===")
    results = compute_errors(args.in_features, args.out_features, args.tokens, args.warmup, args.iters)

    print(f"{'Type':<10} {'dequant_vs_fp32':>16} {'native_vs_fp32':>16} {'native_vs_dequant':>18} {'lat_deq(ms)':>12} {'lat_nat(ms)':>12}")
    print("-" * 80)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<10} ERROR: {r['error']}")
        else:
            ldq = f"{r['lat_dequant_ms']:.3f}" if r.get("lat_dequant_ms") else "-"
            lnat = f"{r['lat_native_ms']:.3f}" if r.get("lat_native_ms") else "-"
            print(f"{name:<10} {r['dequant_vs_fp32']:>16.6f} {r.get('native_vs_fp32', '-'):>16} {r.get('native_vs_dequant', '-'):>18} {ldq:>12} {lnat:>12}")

    print()
    print("Conclusion: if native_vs_fp32 ≈ dequant_vs_fp32, precision loss is from")
    print("the quantization algorithm itself — not from the NPU kernel implementation.")


if __name__ == "__main__":
    main()
