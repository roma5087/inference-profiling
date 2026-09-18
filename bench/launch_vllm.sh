#!/usr/bin/env bash
# Launches vLLM with the FlashInfer attention backend (Stage 4's validated
# config) and an optional --max-num-seqs cap (Stage 6's isolated variable).
# Logs the full boot to stdout/a file so the KV-cache-sizing line
# ("GPU KV cache size: N tokens") is captured as a reviewable artifact
# rather than just asserted in conversation -- confirm it reads ~465-470K
# tokens before trusting any sweep run against this server. A value near
# 24,624 tokens means the KV-cache-sizing bug documented in the README's
# Stage 0 troubleshooting (cold FlashInfer-autotune-cache run) has
# recurred; restart the server once (autotune cache is now warm) before
# proceeding.
#
# Usage: launch_vllm.sh <port> [max-num-seqs]
#   launch_vllm.sh 8001            # uncapped (vLLM default max-num-seqs=256)
#   launch_vllm.sh 8001 84         # capped to 84 concurrent requests
set -euo pipefail

PORT=$1
MAX_NUM_SEQS=${2:-}
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

export CUDA_HOME=~/venv-vllm/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
source ~/venv-vllm/bin/activate

ARGS=(--port "$PORT" --attention-backend FLASHINFER)
if [ -n "$MAX_NUM_SEQS" ]; then
  ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

vllm serve "$MODEL" "${ARGS[@]}"
