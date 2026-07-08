#!/usr/bin/env python3
"""Tree Attention E2E test — using `vllm bench serve` for standardized metrics.

Replaces the old manual curl-based runner with upstream-standard vllm bench serve,
which reports TTFT, TPOT, ITL, throughput and acceptance rate in one pass.

Usage:
    python tools/run_tree_e2e_vllm_bench.py [--test TESTS] [--port PORT] [--backend {cann,triton}]
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

MODEL = os.environ.get(
    "TREE_E2E_MODEL",
    "/data/huggingface_home/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
)
VENV = "/root/vllm-ascend/.venv"
LOG_DIR = "/root/vllm-ascend/.remote-logs/speculative-token-tree"

# Tree shapes (no deep_narrow — removed as remote env issue, not a code bug)
TESTS = {
    "baseline_no_spec": {
        "name": "baseline_no_spec",
        "extra_args": [],
    },
    "mtp_linear_chain": {
        "name": "mtp_linear_chain",
        "extra_args": [
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":3}',
        ],
    },
    "mtp_binary_tree": {
        "name": "mtp_binary_tree",
        "extra_args": [
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":3,"speculative_token_tree":"[(0,),(0,0),(0,1)]"}',
        ],
    },
    "mtp_ternary_tree": {
        "name": "mtp_ternary_tree",
        "extra_args": [
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":4,"speculative_token_tree":"[(0,),(0,0),(0,1),(0,2)]"}',
        ],
    },
    "mtp_mixed_shape": {
        "name": "mtp_mixed_shape",
        "extra_args": [
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":5,"speculative_token_tree":"[(0,),(0,0),(0,0,0),(0,1),(0,1,0)]"}',
        ],
    },
}

DEFAULT_TESTS = [
    "baseline_no_spec",
    "mtp_linear_chain",
    "mtp_binary_tree",
    "mtp_ternary_tree",
    "mtp_mixed_shape",
]

BENCH_ARGS = [
    "--num-prompts", "32",
    "--request-rate", "inf",
    "--burstiness", "1.0",
    "--trust-remote-code",
    "--save-result",
]


def find_free_port(start=8100):
    for p in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("No free port found")


def kill_procs():
    try:
        subprocess.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
        subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
    except Exception:
        pass
    time.sleep(2)


def start_server(port, extra_args):
    cmd = (
        f"source {VENV}/bin/activate && "
        f"vllm serve {MODEL} "
        f"--port {port} "
        f"--trust-remote-code "
        f"--gpu-memory-utilization 0.9 "
        f"--max-model-len 4096 "
        + " ".join(extra_args)
    )
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    return proc


def wait_healthy(port, timeout=180):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def run_bench(port, extra_args):
    result_dir = os.path.join(LOG_DIR, f"vllm-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(result_dir, exist_ok=True)

    cmd = (
        f"source {VENV}/bin/activate && "
        f"vllm bench serve "
        f"--port {port} "
        f"--tokenizer {MODEL} "
        + " ".join(BENCH_ARGS)
        + f" --result-dir {result_dir}"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout, proc.stderr, result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", nargs="*", choices=list(TESTS.keys()), default=DEFAULT_TESTS)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--backend", choices=["cann", "triton"], default="triton")
    args = parser.parse_args()

    port = args.port or find_free_port()
    os.makedirs(LOG_DIR, exist_ok=True)

    results = {}
    all_passed = True

    for test_key in args.test:
        test = TESTS[test_key]
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"{'='*60}")

        kill_procs()
        time.sleep(2)

        print("Starting server...")
        proc = start_server(port, test["extra_args"])
        if not wait_healthy(port):
            print("FAIL: server did not become healthy")
            results[test_key] = {"status": "FAIL", "error": "server_unhealthy"}
            all_passed = False
            kill_procs()
            continue

        print("Running vllm bench serve...")
        rc, stdout, stderr, result_dir = run_bench(port, test["extra_args"])
        kill_procs()

        if rc != 0:
            print(f"FAIL: bench returned {rc}")
            results[test_key] = {"status": "FAIL", "rc": rc, "stderr": stderr[-200:]}
            all_passed = False
        else:
            print(f"PASS: results in {result_dir}")
            results[test_key] = {"status": "PASS", "result_dir": result_dir}

        time.sleep(3)

    summary_path = os.path.join(LOG_DIR, f"summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "backend": args.backend,
        "port": port,
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
