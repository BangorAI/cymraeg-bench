#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
PORT="${MWYDRYN_7B_PORT:-8006}"
MODEL_ID="${MWYDRYN_7B_MODEL:-BangorAI/mwydryn-7b-fersiwn-2}"
GPU_MEMORY="${MWYDRYN_7B_GPU_MEMORY:-0.92}"
MAX_MODEL_LEN="${MWYDRYN_7B_MAX_MODEL_LEN:-4096}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

exec "$ROOT/inference/.venv/bin/vllm" serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$MODEL_ID" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY" \
  --chat-template "$ROOT/inference/templates/mistral-cymraeg-chat.jinja" \
  --generation-config vllm
