#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
MODEL="${1:?Defnydd: inference/stop.sh caernarfon|mwydryn|mistral-cymraeg|mwydryn-7b|techiaith-llama-sft|all}"

stop_one() {
  local name="$1"
  local pid_file="$ROOT/inference/run/$name.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "Nid oes PID wedi'i gadw ar gyfer $name."
    return
  fi
  local pid
  pid="$(<"$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -- "-$pid"
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "Mae $name yn dal yn fyw; yn gorfodi'r grŵp proses i stopio." >&2
      kill -KILL -- "-$pid"
    fi
  fi
  rm -f "$pid_file"
  echo "Wedi stopio $name."
}

case "$MODEL" in
  caernarfon|mwydryn|mistral-cymraeg|mwydryn-7b|techiaith-llama-sft) stop_one "$MODEL" ;;
  all)
    stop_one caernarfon
    stop_one mwydryn
    stop_one mistral-cymraeg
    stop_one mwydryn-7b
    stop_one techiaith-llama-sft
    ;;
  *)
    echo "Model anhysbys: $MODEL" >&2
    exit 2
    ;;
esac
