#!/usr/bin/env python3
"""Tree Attention E2E test — single test case at a time, manual cleanup."""
import json, os, signal, socket, subprocess, sys, time, urllib.request
from datetime import datetime

MODEL = os.environ.get(
    "TREE_E2E_MODEL",
    "/data/huggingface_home/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
)
VENV = "/root/vllm-ascend/.venv"
LOG_DIR = "/root/vllm-ascend/.remote-logs/speculative-token-tree"

PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
]

TESTS = {
    "baseline_no_spec": {
        "name": "baseline_no_spec",
        "spec": None,
        "tree": None,
        "num_spec_tokens": 0,
    },
    "mtp_linear_chain": {
        "name": "mtp_linear_chain",
        "spec": {"method": "mtp", "num_speculative_tokens": 3},
        "tree": None,
        "num_spec_tokens": 3,
    },
    "mtp_binary_tree": {
        "name": "mtp_binary_tree",
        "spec": {"method": "mtp", "num_speculative_tokens": 3, "speculative_token_tree": "[(0,), (0,0), (0,1)]"},
        "tree": "[(0,), (0,0), (0,1)]",
        "num_spec_tokens": 3,
    },
    "mtp_ternary_tree": {
        "name": "mtp_ternary_tree",
        "spec": {"method": "mtp", "num_speculative_tokens": 4, "speculative_token_tree": "[(0,), (0,0), (0,1), (0,2)]"},
        "tree": "[(0,), (0,0), (0,1), (0,2)]",
        "num_spec_tokens": 4,
    },
    "mtp_mixed_shape": {
        "name": "mtp_mixed_shape",
        "spec": {"method": "mtp", "num_speculative_tokens": 5, "speculative_token_tree": "[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]"},
        "tree": "[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]",
        "num_spec_tokens": 5,
    },
}

def kill_all_npu_procs():
    """Force-kill any stale NPU/VLLM processes before and after tests."""
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True)
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True)
    except Exception:
        pass
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
    data = json.dumps({"prompt": prompt, "temperature": 0, "max_tokens": 16, "seed": 42}).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    elapsed_ms = (time.time() - t0) * 1000
    body = json.loads(resp.read())
    return body["choices"][0]["text"], body.get("usage", {}).get("completion_tokens", 0), elapsed_ms

def run_one_test(test_name, use_triton=False, gpu_mem="0.40"):
    """Run a single test case with strict cleanup."""
    test = TESTS[test_name]
    name = test["name"]
    spec = test["spec"]
    tree = test["tree"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    port = get_free_port()
    kernel_name = "TRITON" if use_triton else "CANN"
    log_file = f"{LOG_DIR}/serve-{name}-{kernel_name}-{timestamp}.log"

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TEST: {name}  [{kernel_name}]")
    print(f"  Tree:  {tree or 'N/A'}")
    print(f"  Port:  {port}")
    print(f"  Log:   {log_file}")

    # Build command
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
    else:
        env.pop("VLLM_ASCEND_TREE_TRITON", None)

    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    print(f"  PID:   {proc.pid}")

    ready = wait_health(port)
    if not ready:
        print(f"  RESULT: FAIL (server timeout)")
        dump_error(log_file)
        proc.kill()
        proc.wait()
        return {"test": name, "kernel": kernel_name, "status": "FAIL", "reason": "server timeout"}

    test_entries = []
    test_start = time.time()

    for i, prompt in enumerate(PROMPTS):
        try:
            text, tokens, latency_ms = send_request(port, prompt)
            print(f"  Prompt {i}: \"{prompt}\"")
            print(f"    Output:  \"{text}\"")
            print(f"    Tokens:  {tokens}, Latency: {latency_ms:.0f}ms")
            test_entries.append({
                "prompt": prompt,
                "output": text,
                "tokens": tokens,
                "latency_ms": round(latency_ms, 1),
            })
        except Exception as e:
            print(f"  Prompt {i}: ERROR - {e}")

    total_ms = (time.time() - test_start) * 1000

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Aggressive cleanup
    kill_all_npu_procs()
    time.sleep(5)

    result = {
        "test": name,
        "kernel": kernel_name,
        "tree": tree,
        "spec_config": spec,
        "entries": test_entries,
        "total_latency_ms": round(total_ms, 1),
        "status": "PASS" if test_entries else "FAIL",
    }
    print(f"  RESULT: {result['status']}  Total: {total_ms:.0f}ms")
    return result


def dump_error(log_file):
    """Print last 10 error lines from server log."""
    try:
        with open(log_file) as f:
            lines = f.readlines()
        errs = [l for l in lines if "ERROR" in l or "Error" in l or "FATAL" in l]
        if errs:
            print(f"  Last errors from log:")
            for l in errs[-5:]:
                print(f"    {l.rstrip()[:200]}")
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_tree_e2e_test.py <test_name> [triton|cann]")
        print("Tests:", ", ".join(TESTS.keys()))
        sys.exit(1)

    test_name = sys.argv[1]
    if test_name not in TESTS:
        print(f"Unknown test: {test_name}")
        print("Available:", ", ".join(TESTS.keys()))
        sys.exit(1)

    use_triton = "triton" in sys.argv[2].lower() if len(sys.argv) > 2 else True

    # Cleanup before test
    print("Cleaning up stale NPU processes...")
    kill_all_npu_procs()

    result = run_one_test(test_name, use_triton=use_triton, gpu_mem="0.40")
    print(f"\nDone: {json.dumps(result, indent=2)}")
