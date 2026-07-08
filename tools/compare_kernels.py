#!/usr/bin/env python3
"""
Apples-to-Apples E2E comparison: Triton vs CANN on same hardware, same config.

Config matches vendor baseline (e2e-final-results.md):
  5 prompts, max_tokens=32, temperature=0, seed=42

Tests: baseline (no spec), linear_chain (MTP no tree), tree variants (Triton only).
"""
import json, os, signal, socket, subprocess, sys, time, urllib.request
from datetime import datetime

MODEL = "/data/huggingface_home/Qwen/Qwen3___5-0___8B-Base"
VENV = "/root/vllm-ascend/.venv"
LOG_DIR = "/root/vllm-ascend/.remote-logs/compare"

# Vendor's exact test prompts
PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    "The largest ocean on Earth is the",
    "In computer science, a binary tree is a data structure that",
]
MAX_TOKENS = 32  # matches vendor

TESTS = {
    "baseline_no_spec": {
        "name": "baseline",
        "spec": None,
        "num_spec_tokens": 0,
    },
    "mtp_linear_chain": {
        "name": "linear_chain",
        "spec": {"method": "mtp", "num_speculative_tokens": 3},
        "num_spec_tokens": 3,
    },
    "mtp_binary_tree": {
        "name": "binary_tree",
        "spec": {"method": "mtp", "num_speculative_tokens": 3,
                 "speculative_token_tree": "[(0,), (0,0), (0,1)]"},
        "num_spec_tokens": 3,
    },
    "mtp_ternary_tree": {
        "name": "ternary_tree",
        "spec": {"method": "mtp", "num_speculative_tokens": 4,
                 "speculative_token_tree": "[(0,), (0,0), (0,1), (0,2)]"},
        "num_spec_tokens": 4,
    },
    "mtp_mixed_shape": {
        "name": "mixed_shape",
        "spec": {"method": "mtp", "num_speculative_tokens": 5,
                 "speculative_token_tree": "[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]"},
        "num_spec_tokens": 5,
    },
}


def kill_all():
    for pat in ["vllm.entrypoints", "EngineCore"]:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(3)


def get_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_health(port, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def send_request(port, prompt):
    data = json.dumps({
        "prompt": prompt, "temperature": 0, "max_tokens": MAX_TOKENS, "seed": 42,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=180)
    elapsed_ms = (time.time() - t0) * 1000
    body = json.loads(resp.read())
    text = body["choices"][0]["text"]
    tokens = body.get("usage", {}).get("completion_tokens", 0)
    return text, tokens, elapsed_ms


def run_one_test(test_key, use_triton, gpu_mem="0.40"):
    test = TESTS[test_key]
    name = test["name"]
    spec = test["spec"]
    kernel_name = "TRITON" if use_triton else "CANN"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    port = get_free_port()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = f"{LOG_DIR}/serve-{name}-{kernel_name}-{timestamp}.log"

    print(f"\n{'='*60}")
    print(f"TEST: {name} [{kernel_name}]")
    print(f"  Port: {port}")
    print(f"  Log:  {log_file}")

    cmd = [
        f"{VENV}/bin/python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--max-model-len", "512",
        "--gpu-memory-utilization", gpu_mem,
        "--enforce-eager",
        "--trust-remote-code",
        "--port", str(port),
    ]
    if spec:
        cmd.extend(["--speculative-config", json.dumps(spec)])

    env = os.environ.copy()
    env["HF_HOME"] = "/data/huggingface_home"
    if use_triton:
        env["VLLM_ASCEND_TREE_TRITON"] = "1"

    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    ready = wait_health(port)
    if not ready:
        print("  RESULT: FAIL (server timeout)")
        proc.kill(); proc.wait()
        return {"test": name, "kernel": kernel_name, "status": "FAIL"}

    entries = []
    test_start = time.time()
    for i, prompt in enumerate(PROMPTS):
        try:
            text, tokens, latency_ms = send_request(port, prompt)
            ttft = round(latency_ms, 1)
            tp = tokens / (ttft / 1000) if ttft > 0 else 0
            print(f"  Prompt {i}: {tokens} tokens, {ttft:.0f}ms, {tp:.1f} tok/s")
            entries.append({
                "prompt": prompt[:40], "output": text,
                "tokens": tokens, "latency_ms": ttft, "tokens_per_sec": round(tp, 1),
            })
        except Exception as e:
            print(f"  Prompt {i}: ERROR - {e}")

    total_ms = (time.time() - test_start) * 1000

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    kill_all()
    time.sleep(3)

    total_tokens = sum(e["tokens"] for e in entries)
    overall_tps = total_tokens / (total_ms / 1000) if total_ms > 0 else 0

    result = {
        "test": name, "kernel": kernel_name, "spec_config": spec,
        "max_tokens": MAX_TOKENS, "num_prompts": len(PROMPTS),
        "entries": entries, "total_tokens": total_tokens,
        "total_latency_ms": round(total_ms, 1),
        "tokens_per_sec": round(overall_tps, 1),
        "status": "PASS" if entries else "FAIL",
        "warm_latency_ms": round(
            sum(e["latency_ms"] for e in entries[1:]) / max(len(entries) - 1, 1), 1
        ) if len(entries) > 1 else None,
    }
    print(f"  RESULT: {result['status']}  Total: {total_ms:.0f}ms  TPS: {overall_tps:.1f}")
    return result


def main():
    kill_all()

    # Select tests: "all" or specify names
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        test_keys = list(TESTS.keys())
    elif len(sys.argv) > 1:
        test_keys = [k for k in sys.argv[1:] if k in TESTS]
    else:
        test_keys = ["baseline_no_spec", "mtp_linear_chain"]

    # Kernel: triton or cann (for tree tests, triton always)
    use_triton = True
    if len(sys.argv) > 1 and sys.argv[-1] == "cann":
        use_triton = False

    all_results = []
    for key in test_keys:
        r = run_one_test(key, use_triton=use_triton)
        all_results.append(r)
        # Rate-limit to avoid NPU thermal issues
        time.sleep(5)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print(f"  {MAX_TOKENS} max_tokens × {len(PROMPTS)} prompts")
    print("="*60)
    print(f"{'Test':<20} {'Kernel':<8} {'Total':>8} {'Warm':>8} {'TPS':>8} {'Status':<6}")
    print("-"*60)
    for r in all_results:
        warm = f"{r['warm_latency_ms']:.0f}ms" if r['warm_latency_ms'] else "-"
        print(f"{r['test']:<20} {r['kernel']:<8} {r['total_latency_ms']:>7.0f}ms {warm:>8} {r['tokens_per_sec']:>7.1f} {r['status']:<6}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"{LOG_DIR}/comparison_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
