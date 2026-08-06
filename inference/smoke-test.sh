#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
MODEL="${1:?Defnydd: inference/smoke-test.sh caernarfon|mwydryn|mistral-cymraeg|mwydryn-7b|techiaith-llama-sft}"

case "$MODEL" in
  caernarfon)
    PORT="${CAERNARFON_PORT:-8002}"
    API_MODEL="britllm/britllm-3b-v0.1"
    ;;
  mwydryn)
    PORT="${MWYDRYN_PORT:-8003}"
    API_MODEL="BangorAI/phi2-mwydryn-1"
    ;;
  mistral-cymraeg)
    PORT="${MISTRAL_CYMRAEG_PORT:-8005}"
    API_MODEL="BangorAI/Mistral-7B-Cymraeg-Welsh-v2"
    ;;
  mwydryn-7b)
    PORT="${MWYDRYN_7B_PORT:-8006}"
    API_MODEL="BangorAI/mwydryn-7b-fersiwn-2"
    ;;
  techiaith-llama-sft)
    PORT="${TECHIAITH_LLAMA_SFT_PORT:-8007}"
    API_MODEL="techiaith/llama-3.2-1b-welsh-sft"
    ;;
  *)
    echo "Model anhysbys: $MODEL" >&2
    exit 2
    ;;
esac

curl --silent --show-error --fail \
  --header 'Content-Type: application/json' \
  --data "{\"model\":\"$API_MODEL\",\"messages\":[{\"role\":\"system\",\"content\":\"Ateba yn Gymraeg glir.\"},{\"role\":\"user\",\"content\":\"Beth yw prifddinas Cymru? Ateba mewn un frawddeg.\"}],\"temperature\":0,\"seed\":1,\"max_tokens\":48}" \
  "http://127.0.0.1:$PORT/v1/chat/completions"
echo
