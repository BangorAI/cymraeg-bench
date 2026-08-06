#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
PORT="${MISTRAL_CYMRAEG_PORT:-8005}"
MODEL_ID="${MISTRAL_CYMRAEG_MODEL:-BangorAI/Mistral-7B-Cymraeg-Welsh-v2}"
GPU_MEMORY="${MISTRAL_CYMRAEG_GPU_MEMORY:-0.92}"
MAX_MODEL_LEN="${MISTRAL_CYMRAEG_MAX_MODEL_LEN:-4096}"

# Roedd Xet yn gyson araf ar a4090; mae'r llwybr HTTP safonol yn gyflymach.
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
