#!/usr/bin/env bash
# Stage 6: same fine-sweep methodology as sweep_fine.sh (shared rate grid,
# per-rate seed, save-result), parametrized by an output-dir tag so the
# uncapped same-machine baseline and the --max-num-seqs-capped run land in
# separate, directly comparable directories instead of overwriting
# stage1_fine's original (different-machine) results.
#
# Usage: ulimit -n 65536 && sweep_stage6.sh <port> <tag>
#   sweep_stage6.sh 8001 uncapped_thisbox
#   sweep_stage6.sh 8001 maxseqs84
set -euo pipefail

PORT=$1
TAG=$2
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
RATES=(8 12 16 20 24 32 40 48 56 64)
OUTDIR=~/inference-profiling/results/stage6_${TAG}/vllm
mkdir -p "$OUTDIR"

for RATE in "${RATES[@]}"; do
  NUM_PROMPTS=$(python3 -c "print(max(200, round(60*$RATE)))")
  echo "=== vllm rate=$RATE num_prompts=$NUM_PROMPTS tag=$TAG ==="
  vllm bench serve \
    --backend openai \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --random-range-ratio 0 \
    --request-rate "$RATE" \
    --num-prompts "$NUM_PROMPTS" \
    --seed "$RATE" \
    --temperature 0 \
    --save-result --save-detailed \
    --result-dir "$OUTDIR" \
    --result-filename "rate_${RATE}.json"
done

echo "=== stage6 ($TAG) sweep complete: $OUTDIR ==="
