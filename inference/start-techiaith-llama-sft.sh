#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
PORT="${TECHIAITH_LLAMA_SFT_PORT:-8007}"
MODEL_ID="${TECHIAITH_LLAMA_SFT_MODEL:-techiaith/llama-3.2-1b-welsh-sft}"
MODEL_ROOT="${TECHIAITH_LLAMA_SFT_MODEL_ROOT:-$ROOT/inference/models/techiaith-llama-3.2-1b-welsh-sft}"
GPU_MEMORY="${TECHIAITH_LLAMA_SFT_GPU_MEMORY:-0.88}"
MAX_MODEL_LEN="${TECHIAITH_LLAMA_SFT_MAX_MODEL_LEN:-4096}"

# Mae'r repo'n cynnwys addasydd yn y gwraidd a model BF16 llawn yn sft/.
# Lawrlwythir dim ond y model llawn er mwyn osgoi dibyniaeth ar Unsloth/PEFT.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ ! -f "$MODEL_ROOT/sft/model.safetensors" ]]; then
  mkdir -p "$MODEL_ROOT"
  "$ROOT/inference/.venv/bin/hf" download "$MODEL_ID" \
    --include 'sft/*' \
    --local-dir "$MODEL_ROOT"
fi

exec "$ROOT/inference/.venv/bin/vllm" serve "$MODEL_ROOT/sft" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$MODEL_ID" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY" \
  --chat-template "$ROOT/inference/templates/techiaith-llama-sft-chat.jinja" \
  --generation-config vllm
