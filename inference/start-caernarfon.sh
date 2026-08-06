#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
PORT="${CAERNARFON_PORT:-8002}"
GPU_MEMORY="${CAERNARFON_GPU_MEMORY:-0.44}"

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "$ROOT/inference/logs" "$ROOT/inference/run"

exec "$ROOT/inference/.venv/bin/vllm" serve britllm/britllm-3b-v0.1 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name britllm/britllm-3b-v0.1 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization "$GPU_MEMORY" \
  --generation-config vllm \
  --seed 1 \
  --chat-template "$ROOT/inference/templates/caernarfon-chat.jinja" \
  --chat-template-content-format string
