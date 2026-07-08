#!/bin/bash
# Tree Attention E2E test - runs on remote
set -euo pipefail

MODEL="/data/huggingface_home/Qwen/Qwen3___5-0___8B-Base"
VENV="/root/vllm-ascend/.venv"
LOG="/root/vllm-ascend/.remote-logs/speculative-token-tree"
RESULT="$LOG/results.json"
export HF_HOME=/data/huggingface_home
source "$VENV/bin/activate"

prompts=(
  "The capital of France is"
  "Python is a programming language that"
  "1 + 1 = 2, 2 + 2 = 4, 3 + 3 ="
  "The Eiffel Tower is located in"
  "Machine learning is a subfield of"
)

# Test configs: name|spec_args|tree_label
tests=(
  "baseline||N/A"
  "linear_chain|--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'|N/A"
  "binary_tree|--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"speculative_token_tree\":\"[(0,), (0,0), (0,1)]\"}'|[(0,),(0,0),(0,1)]"
  "ternary_tree|--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4,\"speculative_token_tree\":\"[(0,), (0,0), (0,1), (0,2)]\"}'|[(0,),(0,0),(0,1),(0,2)]"
  "deep_narrow|--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4,\"speculative_token_tree\":\"[(0,), (0,0), (0,0,0), (0,0,0,0)]\"}'|[(0,),(0,0),(0,0,0),(0,0,0,0)]"
  "mixed_shape|--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":5,\"speculative_token_tree\":\"[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]\"}'|[(0,),(0,0),(0,0,0),(0,1),(0,1,0)]"
)

cleanup() {
  echo "[cleanup] Killing servers..."
  pkill -9 -f "api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 3
  python3 -c 'import torch; torch.npu.empty_cache(); torch.npu.synchronize()' 2>/dev/null || true
  sleep 2
}

get_port() { python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"; }

echo '[]' > "$RESULT"
baseline_outputs=""

for entry in "${tests[@]}"; do
  IFS='|' read -r NAME SPEC_ARGS TREE_LABEL <<< "$entry"
  PORT=$(get_port)

  echo ""
  echo "===== $NAME | tree=$TREE_LABEL | port=$PORT ====="

  cleanup

  # Start server
  if [ -z "$SPEC_ARGS" ]; then
    nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --max-model-len 512 \
      --gpu-memory-utilization 0.30 --enforce-eager --trust-remote-code \
      --port "$PORT" > "$LOG/serve-$NAME.log" 2>&1 &
  else
    eval "nohup python -m vllm.entrypoints.openai.api_server \
      --model \"$MODEL\" --max-model-len 512 \
      --gpu-memory-utilization 0.30 --enforce-eager --trust-remote-code \
      --port \"$PORT\" $SPEC_ARGS \
      > \"$LOG/serve-$NAME.log\" 2>&1 &"
  fi
  SRV_PID=$!
  echo "  PID=$SRV_PID"

  # Wait for ready
  for i in $(seq 1 180); do
    if curl -sf -o /dev/null "http://localhost:$PORT/health" 2>/dev/null; then
      echo "  Ready after ${i}s"; break
    fi
    sleep 2
  done
  if ! curl -sf -o /dev/null "http://localhost:$PORT/health" 2>/dev/null; then
    echo "  FAILED to start"
    cleanup; continue
  fi

  # Send requests
  t0=$(date +%s%N)
  total_tok=0
  outputs_json="["
  for pi in "${!prompts[@]}"; do
    prompt="${prompts[$pi]}"
    rs=$(date +%s%N)
    resp=$(curl -sf "http://localhost:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"prompt\":\"$prompt\",\"temperature\":0,\"max_tokens\":32,\"seed\":42}")
    re=$(date +%s%N)
    lat=$(( (re - rs) / 1000000 ))
    text=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
    tok=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null)
    total_tok=$((total_tok + tok))
    echo "    [$tok t, ${lat}ms] $text"
    outputs_json+="{\"prompt\":\"$prompt\",\"output\":\"$text\",\"tokens\":$tok,\"latency_ms\":$lat},"
  done
  t1=$(date +%s%N)
  total_ms=$(( (t1 - t0) / 1000000 ))
  total_sec=$(python3 -c "print($total_ms / 1000)")
  throughput=$(python3 -c "print(round($total_tok / $total_sec, 1))")
  avg_lat=$(( total_ms / ${#prompts[@]} ))
  outputs_json="${outputs_json%,}]"

  if [ "$NAME" = "baseline" ]; then
    baseline_outputs="$outputs_json"
  fi

  # Kill server
  cleanup

  # Check correctness vs baseline
  correct="N/A"
  if [ "$NAME" != "baseline" ] && [ -n "$baseline_outputs" ]; then
    match=$(python3 -c "
import json
base=json.loads('''$baseline_outputs''')
spec=json.loads('''$outputs_json''')
all_ok=True
for b,s in zip(base,spec):
    if b['output'] != s['output']:
        all_ok=False; print('FAIL'); break
if all_ok: print('PASS')
")
    correct="$match"
  fi

  # Extract acceptance rate
  accept="null"
  if [ "$NAME" != "baseline" ]; then
    ar=$(grep -ioP 'acceptance_rate[:\s=]+[\d.]+' "$LOG/serve-$NAME.log" 2>/dev/null | tail -1 | grep -oP '[\d.]+' || echo "")
    [ -n "$ar" ] && accept="$ar"
  fi

  echo "  tok/s=$throughput, avg_lat=${avg_lat}ms, accept=$accept, correct=$correct"

  # Append to results
  python3 -c "
import json
with open('$RESULT') as f: data = json.load(f)
data.append({
    'test': '$NAME', 'tree': '$TREE_LABEL',
    'total_tokens': $total_tok, 'total_time_sec': round($total_sec,2),
    'throughput_tok_s': $throughput, 'avg_latency_ms': $avg_lat,
    'acceptance_rate': $accept, 'correctness': '$correct',
    'outputs': json.loads('''$outputs_json''')
})
with open('$RESULT','w') as f: json.dump(data,f,indent=2)
"
done

# Final summary
echo ""
echo "===== SUMMARY ====="
python3 -c "
import json
with open('$RESULT') as f: data = json.load(f)
base = data[0] if data else None
for r in data:
    sp = 'N/A'
    if base and r['test'] != 'baseline' and r['throughput_tok_s'] > 0:
        sp = round(base['throughput_tok_s'] / r['throughput_tok_s'], 2)
    print(f\"{r['test']:<16} tree={r['tree']:<22} tok/s={r['throughput_tok_s']:>6} lat={r['avg_latency_ms']:>5}ms accept={r.get('acceptance_rate','N/A'):>5} correct={r.get('correctness','N/A'):>5} speedup={sp}\")
"
echo "Results: $RESULT"

cleanup
