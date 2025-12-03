#!/usr/bin/env bash

# Proxy Setting (optional)
# export ALL_PROXY=http://127.0.0.1:7890

# set -euo pipefail

# Network connectivity test (OpenAI / Azure OpenAI)
test_llm_api() {
    echo "🔍 Testing LLM API connectivity..."
    local url="https://api.openai.com"
    if [ -n "${AZURE_OPENAI_ENDPOINT:-}" ]; then
        url="${AZURE_OPENAI_ENDPOINT}"
    fi

    local result
    result=$(curl -v --max-time 3 "$url" 2>&1 || true)
    if echo "$result" | grep -qiE "Connected to|SSL connection using|HTTP/.* 200|HTTP/.* 401|HTTP/.* 404"; then
        echo -e "\033[1;34m✅ Network Test PASSED - $url is reachable\033[0m"
        return 0
    else
        echo -e "\033[1;31m❌ Network Test FAILED - Cannot reach $url within 3s\033[0m"
        echo -e "\033[1;31m💡 Check your proxy or network settings\033[0m"
        return 1
    fi
}

# Run the network test (can comment out if not needed)
test_llm_api || true

# Change directory to repo root
cd $(dirname $0)/..
set -a; source .env; set +a;

# [Please EDIT below] Configuration for this run
# --------------------------
SCRIPT=eval/inference_agent_tool_mode.py
# 1021 new sys
MODEL_NAME=geovista-rl-12k-7b
BENCHMARK=geobench
export VLLM_PORT=12004
export VLLM_HOST="localhost"
# max examples to inference
NUM_SAMPLES="${INFER_NUM_SAMPLES:-1500}"
# ---------------------

# Defaults (can be overridden via env vars)
DATASET_ROOT="${INFER_DATASET_ROOT:-.temp/datasets/${BENCHMARK}}"
OUTPUT_DIR=".temp/outputs/${BENCHMARK}/${MODEL_NAME}"
TEMP_DIR="${INFER_TEMP_DIR:-.temp/inference_crop_imgs}"

mkdir -p "$OUTPUT_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="${INFER_OUTPUT:-$OUTPUT_DIR/inference_${timestamp}.jsonl}"
LOG_PATH="$OUTPUT_DIR/run_${timestamp}.log"

echo "➡️  Dataset root: $DATASET_ROOT"
echo "➡️  Output file : $OUTPUT_PATH"
echo "➡️  Script      : $SCRIPT"
echo "➡️  Temp dir    : $TEMP_DIR"
echo "➡️  VLLM port   : $VLLM_PORT"
echo "➡️  VLLM host   : $VLLM_HOST"
if [ -n "$NUM_SAMPLES" ]; then
    echo "➡️  Num samples : $NUM_SAMPLES"
fi

# Build command
CMD=(python3 $SCRIPT
    --output "$OUTPUT_PATH"
    --temp_dir "$TEMP_DIR"
    --num_workers 16
)

if [ -n "$NUM_SAMPLES" ]; then
    CMD+=(--num_samples "$NUM_SAMPLES")
fi

# Run and tee logs
echo "🚀 Running: ${CMD[*]}"
"${CMD[@]}" | tee "$LOG_PATH" 2>&1


echo "✅ Results saved to: $OUTPUT_PATH"
echo "📝 Log saved to     : $LOG_PATH"
