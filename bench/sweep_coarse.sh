#!/usr/bin/env bash
# Stage 1 coarse pass: wide log-spaced request-rate grid, run per engine.
#
# The server is never restarted between rate points, so --seed is set to
# $RATE rather than a fixed value: reusing the same seed at every point
# would make later, larger num_prompts runs replay earlier points' prompt
# content verbatim against a server with prefix/radix caching on by default,
# leaking a caching effect into what's supposed to be a pure throughput
# measurement.
#
# Requires a raised client-side file-descriptor limit before invoking this
# script (`ulimit -n 65536`) — at high request rates the benchmark client's
# own aiohttp connection pool can exceed the default 1024 open-file limit,
# which previously showed up as false "failed" requests that were actually
# the client running out of sockets, not the server rejecting load.
#
# Usage: ulimit -n 65536 && sweep_coarse.sh <vllm|sglang> <port>
set -euo pipefail

ENGINE=$1
PORT=$2
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
RATES=(1 2 4 8 16 32 64)
INPUT_LEN=512
OUTPUT_LEN=128
OUTDIR=~/inference-profiling/results/stage1_coarse/$ENGINE
mkdir -p "$OUTDIR"

for RATE in "${RATES[@]}"; do
  # Duration target: >=200 requests (floor) or >=60s of sustained load,
  # whichever needs more prompts at this rate -- so low rates get a longer
  # wall-clock window for statistical stability instead of a too-short,
  # noisy sample.
  NUM_PROMPTS=$(python3 -c "print(max(200, round(60*$RATE)))")
  echo "=== $ENGINE rate=$RATE num_prompts=$NUM_PROMPTS ==="

  if [ "$ENGINE" = "vllm" ]; then
    vllm bench serve \
      --backend openai \
      --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len $INPUT_LEN \
      --random-output-len $OUTPUT_LEN \
      --random-range-ratio 0 \
      --request-rate "$RATE" \
      --num-prompts "$NUM_PROMPTS" \
      --seed "$RATE" \
      --temperature 0 \
      --save-result --save-detailed \
      --result-dir "$OUTDIR" \
      --result-filename "rate_${RATE}.json"
  elif [ "$ENGINE" = "sglang" ]; then
    python3 -m sglang.bench_serving \
      --backend sglang-oai \
      --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len $INPUT_LEN \
      --random-output-len $OUTPUT_LEN \
      --random-range-ratio 0 \
      --request-rate "$RATE" \
      --num-prompts "$NUM_PROMPTS" \
      --seed "$RATE" \
      --temperature 0 \
      --output-file "$OUTDIR/rate_${RATE}.json" \
      --output-details
  else
    echo "unknown engine: $ENGINE" >&2
    exit 1
  fi
done

echo "=== $ENGINE coarse sweep complete: $OUTDIR ==="
