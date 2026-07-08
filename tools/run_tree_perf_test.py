#!/usr/bin/env python3
"""Tree Attention E2E perf test. Uses nohup for server launch (avoids subprocess buffering issues)."""
import json, os, socket, subprocess, sys, time, re
from datetime import datetime
from urllib.request import Request, urlopen

MODEL = "/data/huggingface_home/Qwen/Qwen3___5-0___8B-Base"
VENV = "/root/vllm-ascend/.venv"
LOG_DIR = "/root/vllm-ascend/.remote-logs/speculative-token-tree"
os.environ["HF_HOME"] = "/data/huggingface_home"

PROMPTS = [
    "The capital of France is", "Python is a programming language that",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =", "The Eiffel Tower is located in",
    "Machine learning is a subfield of",
]

TESTS = [
    {"name": "baseline", "spec": None, "tree": None, "ns": 0},
    {"name": "linear_chain", "spec": {"method":"mtp","num_speculative_tokens":3}, "tree": None, "ns": 3},
    {"name": "binary_tree", "spec": {"method":"mtp","num_speculative_tokens":3,"speculative_token_tree":"[(0,), (0,0), (0,1)]"}, "tree": "[(0,), (0,0), (0,1)]", "ns": 3},
    {"name": "ternary_tree", "spec": {"method":"mtp","num_speculative_tokens":4,"speculative_token_tree":"[(0,), (0,0), (0,1), (0,2)]"}, "tree": "[(0,), (0,0), (0,1), (0,2)]", "ns": 4},
    {"name": "mixed_shape", "spec": {"method":"mtp","num_speculative_tokens":5,"speculative_token_tree":"[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]"}, "tree": "[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]", "ns": 5},
]

def free_port():
    s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close(); return p

def killall():
    """Kill all vllm/api_server/EngineCore processes. NEVER kill self."""
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        for line in out.split("\n"):
            if any(k in line for k in ["api_server", "vllm.entrypoints",
                                         "VLLM::EngineCore"]):
                if "grep" not in line and "run_tree_perf" not in line:
                    pid = int(line.split()[1])
                    if pid != my_pid:
                        subprocess.run(["kill", "-9", str(pid)], stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(3)

def clear_hbm():
    """Clear torch NPU cache and wait for baseline HBM."""
    try:
        subprocess.run([f"{VENV}/bin/python", "-c",
            "import torch; torch.npu.empty_cache(); torch.npu.synchronize()"],
            timeout=15, capture_output=True)
    except Exception:
        pass
    time.sleep(2)

def wait_health(port, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urlopen(f"http://localhost:{port}/health", timeout=3).status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def send_request(port, prompt):
    data = json.dumps({"prompt": prompt, "temperature": 0, "max_tokens": 32, "seed": 42}).encode()
    r = Request(f"http://localhost:{port}/v1/completions", data=data,
                headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urlopen(r, timeout=180)
    ms = (time.time() - t0) * 1000
    body = json.loads(resp.read())
    return body["choices"][0]["text"], body.get("usage", {}).get("completion_tokens", 0), ms

def extract_accept_rate(log_file):
    try:
        with open(log_file) as f:
            text = f.read()
        # vLLM logs: spec_decode_acceptance_rate or similar
        m = re.search(r'acceptance_rate[:\s=]+([\d.]+)', text, re.I)
        if m: return float(m.group(1))
        m = re.search(r'forward.*accept[:\s=]*(\d+).*draft[:\s=]*(\d+)', text, re.I)
        if m: return int(m.group(1)) / int(m.group(2))
    except Exception:
        pass
    return None

# ── Main ──
results = []
baseline = None

for ti, test in enumerate(TESTS):
    name = test["name"]; spec = test["spec"]; port = free_port()
    log = f"{LOG_DIR}/perf-{name}.log"

    print(f"\n{'='*60}")
    print(f"[{ti+1}/{len(TESTS)}] {name}  tree={test['tree'] or 'N/A'}  port={port}")
    print(f"  {datetime.now().strftime('%H:%M:%S')}")

    # Step 1: kill everything
    killall()
    clear_hbm()

    # Step 2: start server (nohup, not subprocess.Popen)
    spec_flag = f"--speculative-config '{json.dumps(spec)}'" if spec else ""
    sh_cmd = (f"cd /root/vllm-ascend && source {VENV}/bin/activate && "
              f"HF_HOME=/data/huggingface_home nohup python -m vllm.entrypoints.openai.api_server "
              f"--model {MODEL} --max-model-len 512 --gpu-memory-utilization 0.30 "
              f"--enforce-eager --trust-remote-code --port {port} {spec_flag} "
              f"> {log} 2>&1 &")
    # Get server PID
    pid_out = subprocess.check_output(["bash", "-c", sh_cmd + " echo $!"], text=True).strip()
    print(f"  PID: {pid_out}")

    # Step 3: wait for ready
    if not wait_health(port):
        print(f"  FAILED to start")
        killall()
        continue
    print(f"  Ready, sending {len(PROMPTS)} prompts...")

    # Step 4: send requests
    entries = []; t0 = time.time(); total_tok = 0
    for p in PROMPTS:
        text, tok, ms = send_request(port, p)
        total_tok += tok
        entries.append({"prompt": p, "output": text.strip(), "tokens": tok, "latency_ms": round(ms, 1)})
        print(f"    [{tok}t, {ms:.0f}ms] {text.strip()[:60]}")
    total_sec = time.time() - t0

    # Step 5: kill & collect
    killall()
    accept = extract_accept_rate(log)

    r = {"test": name, "tree": test["tree"] or "N/A", "num_spec_tokens": test["ns"],
         "prompt_count": len(PROMPTS), "total_tokens": total_tok,
         "total_time_sec": round(total_sec, 2),
         "throughput_tok_s": round(total_tok / total_sec, 1) if total_sec > 0 else 0,
         "avg_latency_ms": round(sum(e["latency_ms"] for e in entries) / len(entries), 1),
         "acceptance_rate": round(accept, 3) if accept else None,
         "outputs": entries}
    if name == "baseline":
        baseline = r; r["speedup"] = 1.0
    elif baseline:
        r["speedup"] = round(baseline["throughput_tok_s"] / r["throughput_tok_s"], 2) if r["throughput_tok_s"] > 0 else None
    results.append(r)
    print(f"  tok/s={r['throughput_tok_s']}, avg_lat={r['avg_latency_ms']}ms, accept={r.get('acceptance_rate')}")

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
rf = f"{LOG_DIR}/perf-results-{ts}.json"
os.makedirs(LOG_DIR, exist_ok=True)
with open(rf, "w") as f:
    json.dump(results, f, indent=2)

# Summary table
print(f"\n{'='*78}")
print(f"{'Test':<16} {'Tree':<20} {'Spec':>4} {'Lat(ms)':>8} {'Tok/s':>8} {'Accept':>7} {'Speedup':>8} {'Correct':>7}")
print(f"{'-'*78}")
for r in results:
    tr = r["tree"][:18]; ac = f"{r['acceptance_rate']:.2f}" if r.get("acceptance_rate") else "N/A"
    sp = f"{r['speedup']:.2f}x" if r.get("speedup") else "N/A"
    # Correctness check vs baseline
    ok = "N/A"
    if baseline and r["test"] != "baseline":
        match = all(r["outputs"][i]["output"] == baseline["outputs"][i]["output"]
                    for i in range(min(len(r["outputs"]), len(baseline["outputs"]))))
        ok = "PASS" if match else "FAIL"
    print(f"{r['test']:<16} {tr:<20} {r['num_spec_tokens']:>4} {r['avg_latency_ms']:>8.0f} {r['throughput_tok_s']:>8.1f} {ac:>7} {sp:>8} {ok:>7}")

print(f"\nResults: {rf}")

# Step 3: final cleanup
killall()
clear_hbm()
