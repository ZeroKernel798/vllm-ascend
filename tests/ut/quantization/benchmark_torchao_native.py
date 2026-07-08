"""torchao native-kernel benchmark: bf16 vs int8wo(native ON) vs int8wo(native OFF).

Adapted from ccf-vllm-ascend/docs/torchao/scripts/benchmark_torchao.py.
Where the original compared bf16 vs int8wo (dequant path), this runs THREE
setups so we can see both the absolute speedup vs bf16 AND the native-kernel
gain over the dequant fallback:

  * bf16            : baseline
  * int8wo native=1 : npu_weight_quant_batchmatmul (P5-1 fast path)
  * int8wo native=0 : F.linear + dequant fallback (pre-optimization path)

Metrics per setup: NPU weight memory, prefill tps, decode tps, accuracy
(greedy token match + text similarity vs bf16).

Usage:
  python benchmark_torchao_native.py --model Qwen/Qwen3-0.6B --save out.json
"""
import argparse
import difflib
import gc
import json
import os
import sys
import time
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")
os.environ.setdefault("VLLM_ASCEND_TORCHAO_ALLOW_DEQUANT_FALLBACK", "1")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
# Some small models (gpt2) have max_position_embeddings=1024; allow our
# default benchmark max_model_len to be clamped rather than rejected.
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


ACCURACY_PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "Machine learning models are typically trained",
    "The theory of relativity was proposed by",
    "In the United States, the largest city is",
    "Climate change is caused by",
    "The Pacific Ocean is the",
    "Shakespeare wrote many famous plays including",
    "DNA stands for",
    "The speed of light in vacuum is",
]


def _resolve_local(model_id: str) -> str:
    base = "/data/huggingface_home/hub"
    safe = "models--" + model_id.replace("/", "--")
    snap_dir = os.path.join(base, safe, "snapshots")
    if os.path.isdir(snap_dir):
        snaps = [d for d in os.listdir(snap_dir) if not d.startswith(".")]
        if snaps:
            return os.path.join(snap_dir, snaps[0])
    return model_id


def free_memory():
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception:
        pass


def build_llm(model: str, *, quantize: bool, native: str, max_model_len: int = 1024):
    # native is "1" / "0"; only meaningful when quantize=True
    os.environ["VLLM_ASCEND_TORCHAO_NATIVE_KERNEL"] = native
    from vllm.entrypoints.llm import LLM

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        max_model_len=max_model_len,
        disable_log_stats=True,
    )
    if quantize:
        import torch as _torch
        from torchao.core.config import config_to_dict
        import torchao.quantization as tq

        # Build the torchao config matching VLLM_ASCEND_TORCHAO_CONFIG_TYPE so
        # this script can benchmark int8wo / w8a8 / w4a16, not just int8wo.
        ct = os.environ.get("VLLM_ASCEND_TORCHAO_CONFIG_TYPE", "int8wo")
        if ct == "w8a8":
            ao_cfg = tq.Int8DynamicActivationInt8WeightConfig()
        elif ct == "intx4wo":
            ao_cfg = tq.IntxWeightOnlyConfig(weight_dtype=_torch.int4)
        elif ct == "w4a8":
            ao_cfg = tq.Int8DynamicActivationIntxWeightConfig(weight_dtype=_torch.int4)
        else:  # int8wo (default)
            ao_cfg = tq.Int8WeightOnlyConfig()

        kwargs["quantization"] = "torchao"
        kwargs["hf_overrides"] = {
            "quantization_config_dict_json": json.dumps(config_to_dict(ao_cfg))
        }
    return LLM(**kwargs)


def query_npu_memory(llm) -> dict:
    def _probe(_model):
        import torch
        out = {}
        if hasattr(torch, "npu") and torch.npu.is_available():
            out["allocated_mb"] = torch.npu.memory_allocated() / (1024 ** 2)
            out["reserved_mb"] = torch.npu.memory_reserved() / (1024 ** 2)
            out["max_allocated_mb"] = torch.npu.max_memory_allocated() / (1024 ** 2)

        def real_storage_bytes(t):
            if hasattr(t, "__tensor_flatten__"):
                try:
                    names, _ctx = t.__tensor_flatten__()
                    return sum(real_storage_bytes(getattr(t, n)) for n in names)
                except Exception:
                    pass
            return t.numel() * t.element_size()

        try:
            logical = real = 0
            ql = ul = 0
            for _name, mod in _model.named_modules():
                w = getattr(mod, "weight", None)
                if w is None or not isinstance(w, torch.nn.Parameter):
                    continue
                try:
                    logical += w.numel() * w.element_size()
                    real += real_storage_bytes(w.data)
                    # native int8 path stores a plain int8 weight + weight_scale;
                    # count it as quantized via the layer flag if present.
                    if hasattr(w.data, "__tensor_flatten__") or getattr(mod, "_torchao_native_w8a16", False):
                        ql += 1
                    else:
                        ul += 1
                except Exception:
                    pass
            out["logical_param_mb"] = logical / (1024 ** 2)
            out["real_param_mb"] = real / (1024 ** 2)
            out["quantized_layers"] = ql
            out["unquantized_layers"] = ul
        except Exception as e:
            out["param_walk_error"] = str(e)
        return out

    try:
        result = llm.apply_model(_probe)
        return result[0] if isinstance(result, list) else result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def measure_accuracy(llm) -> list[dict]:
    from vllm.sampling_params import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    outs = llm.generate(ACCURACY_PROMPTS, sp)
    return [
        dict(prompt=o.prompt, text=o.outputs[0].text,
             token_ids=list(o.outputs[0].token_ids))
        for o in outs
    ]


def measure_prefill(llm, *, input_len: int, num_prompts: int) -> dict:
    from vllm.sampling_params import SamplingParams
    prompts = ["the " * input_len] * num_prompts
    sp = SamplingParams(temperature=0.0, max_tokens=4)
    llm.generate([prompts[0]], sp)  # warmup
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0
    in_tokens = sum(len(o.prompt_token_ids) for o in outs)
    return dict(elapsed_s=elapsed, input_tokens=in_tokens,
                prefill_tps=in_tokens / elapsed if elapsed > 0 else 0.0)


def measure_decode(llm, *, output_len: int, num_prompts: int) -> dict:
    from vllm.sampling_params import SamplingParams
    prompts = ["Once upon a time"] * num_prompts
    sp = SamplingParams(temperature=0.0, max_tokens=output_len)
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=4))  # warmup
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return dict(elapsed_s=elapsed, output_tokens=out_tokens,
                decode_tps=out_tokens / elapsed if elapsed > 0 else 0.0)


def run_setup(model, *, quantize, native, label, prefill_args, decode_args) -> dict:
    print(f"\n{'='*70}\n[{label}] BUILDING\n{'='*70}", flush=True)
    free_memory()
    result = dict(label=label, quantize=quantize, native=native)
    try:
        t0 = time.perf_counter()
        llm = build_llm(model, quantize=quantize, native=native)
        result["load_time_s"] = time.perf_counter() - t0

        def _reset_peak(_m):
            try:
                import torch
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            llm.apply_model(_reset_peak)
        except Exception:
            pass

        result["memory_post_load"] = query_npu_memory(llm)
        result["accuracy_outputs"] = measure_accuracy(llm)
        result["prefill"] = measure_prefill(llm, **prefill_args)
        result["decode"] = measure_decode(llm, **decode_args)
        result["memory_after_run"] = query_npu_memory(llm)
        del llm
    except Exception as e:
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        free_memory()
    return result


def compare_accuracy(ref_outs, outs) -> dict:
    n = min(len(ref_outs), len(outs))
    exact = first16 = first32 = 0
    ratios = []
    for i in range(n):
        b, q = ref_outs[i]["token_ids"], outs[i]["token_ids"]
        if b == q:
            exact += 1
        if b[:16] == q[:16]:
            first16 += 1
        if b[:32] == q[:32]:
            first32 += 1
        ratios.append(difflib.SequenceMatcher(None, ref_outs[i]["text"], outs[i]["text"]).ratio())
    return dict(n=n,
                exact_match_rate=exact / n if n else 0.0,
                first16_match_rate=first16 / n if n else 0.0,
                first32_match_rate=first32 / n if n else 0.0,
                avg_text_similarity=sum(ratios) / len(ratios) if ratios else 0.0)


def sdiv(a, b):
    return a / b if (a and b) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prefill-input-len", type=int, default=256)
    p.add_argument("--prefill-num", type=int, default=4)
    p.add_argument("--decode-output-len", type=int, default=128)
    p.add_argument("--decode-num", type=int, default=4)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    model = _resolve_local(args.model)
    pa = dict(input_len=args.prefill_input_len, num_prompts=args.prefill_num)
    da = dict(output_len=args.decode_output_len, num_prompts=args.decode_num)
    print(f"\n>>> Benchmark {args.model} -> {model}", flush=True)

    bf16 = run_setup(model, quantize=False, native="0", label="bf16", prefill_args=pa, decode_args=da)
    nat1 = run_setup(model, quantize=True, native="1", label="int8wo-native", prefill_args=pa, decode_args=da)
    nat0 = run_setup(model, quantize=True, native="0", label="int8wo-dequant", prefill_args=pa, decode_args=da)

    acc_nat1 = compare_accuracy(bf16.get("accuracy_outputs", []), nat1.get("accuracy_outputs", [])) if "error" not in bf16 and "error" not in nat1 else {}
    acc_nat0 = compare_accuracy(bf16.get("accuracy_outputs", []), nat0.get("accuracy_outputs", [])) if "error" not in bf16 and "error" not in nat0 else {}

    # ---- summary ----
    print("\n" + "=" * 70)
    print(f" SUMMARY: {args.model}")
    print("=" * 70)
    def dtps(r): return (r.get("decode") or {}).get("decode_tps")
    def ptps(r): return (r.get("prefill") or {}).get("prefill_tps")
    def realmb(r): return (r.get("memory_post_load") or {}).get("real_param_mb")
    bf, n1, n0 = dtps(bf16), dtps(nat1), dtps(nat0)
    print(f"  decode tps   bf16={bf}  native={n1}  dequant={n0}")
    print(f"  decode speedup  native/bf16={sdiv(n1,bf)}  dequant/bf16={sdiv(n0,bf)}  native/dequant={sdiv(n1,n0)}")
    print(f"  prefill tps  bf16={ptps(bf16)}  native={ptps(nat1)}  dequant={ptps(nat0)}")
    print(f"  real param MB  bf16={realmb(bf16)}  native={realmb(nat1)}  dequant={realmb(nat0)}")
    print(f"  acc native vs bf16:  first16={acc_nat1.get('first16_match_rate')}  sim={acc_nat1.get('avg_text_similarity')}")
    print(f"  errors: bf16={bf16.get('error')}  native={nat1.get('error')}  dequant={nat0.get('error')}")
    print("RESULTJSON " + json.dumps(dict(
        model=args.model,
        decode={"bf16": bf, "native": n1, "dequant": n0},
        prefill={"bf16": ptps(bf16), "native": ptps(nat1), "dequant": ptps(nat0)},
        real_mb={"bf16": realmb(bf16), "native": realmb(nat1), "dequant": realmb(nat0)},
        speedup={"native_vs_bf16": sdiv(n1, bf), "dequant_vs_bf16": sdiv(n0, bf), "native_vs_dequant": sdiv(n1, n0)},
        acc_native=acc_nat1, acc_dequant=acc_nat0,
        errors={"bf16": bf16.get("error"), "native": nat1.get("error"), "dequant": nat0.get("error")},
    ), default=str))

    if args.save:
        with open(args.save, "w") as f:
            json.dump(dict(model=args.model, bf16=bf16, native=nat1, dequant=nat0,
                           acc_native=acc_nat1, acc_dequant=acc_nat0), f, indent=2, default=str)
        print(f">>> saved {args.save}")


if __name__ == "__main__":
    main()
