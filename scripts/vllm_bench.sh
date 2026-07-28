#!/usr/bin/env bash
set -euo pipefail

export VLLM_LOGGING_LEVEL=ERROR

declare -A MODELS=(
  ["qwen"]="Qwen/Qwen2.5-7B-Instruct"
  ["llama"]="meta-llama/Llama-3.1-8B-Instruct"
  ["mistral"]="mistralai/Mistral-7B-Instruct-v0.3"
  ["gemma"]="google/gemma-2-9b-it"
  ["deepseek"]="deepseek-ai/deepseek-llm-7b-chat"
)

RESULTS_DIR="./bench_results"
mkdir -p "$RESULTS_DIR"

# Engine-level flags mirrored 1:1 from the single-prompt script's LLM(...) defaults.
# Verify these flag names against your installed vLLM version with:
#   vllm bench throughput --help
#   vllm bench latency --help
COMMON_ENGINE_FLAGS=(
  --attention-backend TRITON_ATTN
  --gpu-memory-utilization 0.8
  --dtype bfloat16
  --max-model-len 1024
  --max-num-seqs 8
  --enforce-eager
  # trust_remote_code=False is already the vllm default, so no flag needed
)

# This vLLM version uses --random-input-len / --random-output-len / --random-prefix-len
# for the random dataset (not --input-len / --output-len, which triggered a "both specified"
# warning and got silently overridden by unset --random-* defaults, which is what caused the
# max_model_len assertion). --random-prefix-len 0 avoids adding extra prefix tokens on top.
RANDOM_DATASET_FLAGS=(
  --dataset-name random
  --random-input-len 512
  --random-output-len 128
  --random-prefix-len 0
  --random-range-ratio 0
)

# --- offline batched throughput (no server needed) ---
run_throughput() {
  local name=$1 model=$2
  echo "=== throughput: $name ($model) ==="
  vllm bench throughput \
    --model "$model" \
    "${RANDOM_DATASET_FLAGS[@]}" \
    --num-prompts 10000 \
    "${COMMON_ENGINE_FLAGS[@]}" \
    2>&1 | tee "$RESULTS_DIR/${name}_throughput.log"
}

# --- single-batch latency (closest apples-to-apples vs. the single-prompt script) ---
run_latency() {
  local name=$1 model=$2
  echo "=== latency: $name ($model) ==="
  vllm bench latency \
    --model "$model" \
    --input-len 512 \
    --output-len 128 \
    --batch-size 1 \
    --num-iters-warmup 3 \
    --num-iters 32 \
    "${COMMON_ENGINE_FLAGS[@]}" \
    2>&1 | tee "$RESULTS_DIR/${name}_latency.log"
}

for name in "${!MODELS[@]}"; do
  model="${MODELS[$name]}"
  echo ""
  echo "########## $name ($model) ##########"
  run_throughput "$name" "$model"
  run_latency "$name" "$model"
done

echo ""
echo "Done. Logs in $RESULTS_DIR/"
