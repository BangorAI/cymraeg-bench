#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
MODEL="${1:?Defnydd: inference/start.sh caernarfon|mwydryn|mistral-cymraeg|mwydryn-7b|techiaith-llama-sft}"
STARTUP_TIMEOUT="${AISTEDDFOD_START_TIMEOUT:-1800}"

export PATH="$ROOT/inference/.venv/bin:$PATH"

case "$MODEL" in
  caernarfon)
    PORT="${CAERNARFON_PORT:-8002}"
    ;;
  mwydryn)
    PORT="${MWYDRYN_PORT:-8003}"
    ;;
  mistral-cymraeg)
    PORT="${MISTRAL_CYMRAEG_PORT:-8005}"
    STARTUP_TIMEOUT="${AISTEDDFOD_START_TIMEOUT:-3600}"
    ;;
  mwydryn-7b)
    PORT="${MWYDRYN_7B_PORT:-8006}"
    STARTUP_TIMEOUT="${AISTEDDFOD_START_TIMEOUT:-3600}"
    ;;
  techiaith-llama-sft)
    PORT="${TECHIAITH_LLAMA_SFT_PORT:-8007}"
    STARTUP_TIMEOUT="${AISTEDDFOD_START_TIMEOUT:-3600}"
    ;;
  *)
    echo "Model anhysbys: $MODEL" >&2
    exit 2
    ;;
esac

mkdir -p "$ROOT/inference/logs" "$ROOT/inference/run"

if [[ -f "$ROOT/inference/run/$MODEL.pid" ]] && kill -0 "$(<"$ROOT/inference/run/$MODEL.pid")" 2>/dev/null; then
  echo "Mae $MODEL eisoes yn rhedeg (PID $(<"$ROOT/inference/run/$MODEL.pid"))."
  exit 0
fi

nohup setsid "$ROOT/inference/start-$MODEL.sh" \
  >"$ROOT/inference/logs/$MODEL.log" 2>&1 </dev/null &
PID=$!
echo "$PID" >"$ROOT/inference/run/$MODEL.pid"
echo "Yn cychwyn $MODEL ar borth $PORT (PID $PID)."

for _ in $(seq 1 $((STARTUP_TIMEOUT / 2))); do
  if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "Mae $MODEL yn barod."
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Methodd $MODEL â chychwyn. Gweler $ROOT/inference/logs/$MODEL.log" >&2
    tail -n 80 "$ROOT/inference/logs/$MODEL.log" >&2
    exit 1
  fi
  sleep 2
done

echo "Daeth yr amser aros i ben. Gweler $ROOT/inference/logs/$MODEL.log" >&2
exit 1
