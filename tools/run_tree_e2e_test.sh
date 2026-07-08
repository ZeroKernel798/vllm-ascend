#!/bin/bash
# Tree Attention E2E test runner — tests all tree shapes, collects accuracy + perf
set -e

REMOTE_ROOT="/root/vllm-ascend"
VENV="${REMOTE_ROOT}/.venv"
LOG_DIR="${REMOTE_ROOT}/.remote-logs/speculative-token-tree"
MODEL="/data/huggingface_home/Qwen/Qwen3___5-0___8B-Base"

mkdir -p "$LOG_DIR"

# Import venv
source "${VENV}/bin/activate"
export HF_HOME=/data/huggingface_home

PROMPTS=(
  "The capital of France is"
  "Python is a programming language that"
  "1 + 1 = 2, 2 + 2 = 4, 3 + 3 ="
)

# Test configurations: name, spec_json, num_spec_tokens, tree_str
declare -A TESTS
TESTS=(
  ["baseline_no_spec"]='{}:0:'
  ["mtp_linear_chain"]='{"method":"mtp","num_speculative_tokens":3}:3:'
  ["mtp_binary_tree"]='{"method":"mtp","num_speculative_tokens":3,"speculative_token_tree":"[(0,), (0,0), (0,1)]"}:3:[(0,), (0,0), (0,1)]'
  ["mtp_ternary_tree"]='{"method":"mtp","num_speculative_tokens":4,"speculative_token_tree":"[(0,), (0,0), (0,1), (0,2)]"}:4:[(0,), (0,0), (0,1), (0,2)]'
  ["mtp_deep_narrow"]='{"method":"mtp","num_speculative_tokens":4,"speculative_token_tree":"[(0,), (0,0), (0,0,0), (0,0,0,0)]"}:4:[(0,), (0,0), (0,0,0), (0,0,0,0)]'
  ["mtp_mixed_shape"]='{"method":"mtp","num_speculative_tokens":5,"speculative_token_tree":"[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]"}:5:[(0,), (0,0), (0,0,0), (0,1), (0,1,0)]'
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${LOG_DIR}/e2e-tree-perf-${TIMESTAMP}.json"
echo '{"tests":[]}' > "$RESULTS_FILE"

# Get a free port
get_free_port() {
  python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

for TEST_NAME in "${!TESTS[@]}"; do
  IFS=':' read -r SPEC_JSON NUM_SPEC TOKENS TREE_STR <<< "${TESTS[$TEST_NAME]}"
  PORT=$(get_free_port)
  LOG_FILE="${LOG_DIR}/serve-${TEST_NAME}-${TIMESTAMP}.log"

  echo ""
  echo "============================================================"
  echo "TEST: $TEST_NAME"
  echo "  Config: $SPEC_JSON"
  echo "  Tree: $TREE_STR"
  echo "  Port: $PORT"
  echo "  Log:  $LOG_FILE"
  echo "============================================================"

  # Start server
  cd "$REMOTE_ROOT"
  if [ "$TEST_NAME" = "baseline_no_spec" ]; then
    nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --max-model-len 512 \
      --gpu-memory-utilization 0.70 \
      --enforce-eager \
      --trust-remote-code \
      --port "$PORT" \
      > "$LOG_FILE" 2>&1 &
    SERVE_PID=$!
  else
    nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --speculative-config "$SPEC_JSON" \
      --max-model-len 512 \
      --gpu-memory-utilization 0.70 \
      --enforce-eager \
      --trust-remote-code \
      --port "$PORT" \
      > "$LOG_FILE" 2>&1 &
    SERVE_PID=$!
  fi

  echo "  PID: $SERVE_PID"

  # Wait for server ready
  for i in $(seq 1 180); do
    if curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" 2>/dev/null | grep -q 200; then
      echo "  Server ready after ${i}s"
      break
    fi
    sleep 1
  done

  if ! curl -s -o /dev/null "http://localhost:${PORT}/health" 2>/dev/null; then
    echo "  ERROR: Server failed to start"
    kill $SERVE_PID 2>/dev/null
    continue
  fi

  # Run test prompts and collect results
  TEST_START=$(date +%s%N)
  declare -a OUTPUTS
  declare -a LATENCIES
  declare -a TOKEN_COUNTS

  for idx in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$idx]}"
    REQ_START=$(date +%s%N)

    RESPONSE=$(curl -s "http://localhost:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"prompt\":\"${PROMPT}\",\"temperature\":0,\"max_tokens\":16,\"seed\":42}")

    REQ_END=$(date +%s%N)
    REQ_MS=$(( (REQ_END - REQ_START) / 1000000 ))

    TEXT=$(echo "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["text"])' 2>/dev/null || echo "ERROR")
    TOKENS=$(echo "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("usage",{}).get("completion_tokens","N/A"))' 2>/dev/null || echo "N/A")

    echo "  Prompt $idx: \"${PROMPT}\""
    echo "    Output: ${TEXT}"
    echo "    Tokens: ${TOKENS}, Latency: ${REQ_MS}ms"

    OUTPUTS[$idx]="$TEXT"
    LATENCIES[$idx]=$REQ_MS
    TOKEN_COUNTS[$idx]="$TOKENS"
  done

  TEST_END=$(date +%s%N)
  TEST_TOTAL_MS=$(( (TEST_END - TEST_START) / 1000000 ))

  # Collect NPU metrics from log
  NPU_METRICS=$(python3 -c "
import re, json
try:
    with open('$LOG_FILE') as f:
        log = f.read()
    ttft = re.findall(r'request_time.*?(\d+\.?\d*)', log)
    print(json.dumps({'ttft_samples': ttft[:5] if ttft else []}))
except:
    print('{}')
" 2>/dev/null || echo '{}')

  # Kill server
  kill $SERVE_PID 2>/dev/null
  sleep 3
  kill -9 $SERVE_PID 2>/dev/null
  sleep 2

  # Append result
  python3 -c "
import json

result = {
    'test': '$TEST_NAME',
    'spec_config': json.loads('''$SPEC_JSON'''),
    'tree': '''$TREE_STR''',
    'outputs': list(zip('${PROMPTS[@]}'.split(), '${OUTPUTS[@]}'.split(), [${LATENCIES[@]}], '${TOKEN_COUNTS[@]}'.split())),
    'total_latency_ms': $TEST_TOTAL_MS,
    'npu_metrics': json.loads('''$NPU_METRICS'''),
}

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['tests'].append(result)
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"

  echo "  [DONE] Total: ${TEST_TOTAL_MS}ms"

  # Wait for NPU to cool down
  sleep 5
done

echo ""
echo "============================================================"
echo "ALL TESTS COMPLETE"
echo "Results: $RESULTS_FILE"
echo "============================================================"
