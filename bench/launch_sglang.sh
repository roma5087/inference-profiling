#!/usr/bin/env bash
# Launches SGLang with metrics enabled (required for Stage 6's /metrics
# polling -- SGLang does not expose /metrics by default, unlike vLLM).
#
# Usage: launch_sglang.sh <port>
set -euo pipefail

PORT=$1
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

export CUDA_HOME=~/venv-sglang/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
source ~/venv-sglang/bin/activate

python3 -m sglang.launch_server --model-path "$MODEL" --port "$PORT" --enable-metrics
