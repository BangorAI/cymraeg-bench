#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"
RUN_PREFIX="${LOCAL_BENCH_PREFIX:-cymraegbench-v0.1-lleol}"
MAX_CASES="${LOCAL_BENCH_MAX_CASES:-100}"
MAX_ATTEMPTS="${LOCAL_BENCH_MAX_ATTEMPTS:-3}"
MODEL_FILTER="${LOCAL_BENCH_MODELS:-}"

export PATH="$HOME/.local/bin:$PATH"

cd "$ROOT"
mkdir -p inference/logs runs

# Enw'r gwasanaeth : ID yr harnais : nifer y gweithwyr. Mae gweinydd Mwydryn
# yn cynhyrchu un ymateb ar y tro; gall vLLM drin dau gais yn gyfochrog.
models=(
  "mistral-cymraeg:bangor-mistral-cymraeg-v2:2"
  "mwydryn:mwydryn:1"
  "caernarfon:caernarfon-3b:2"
  "mwydryn-7b:mwydryn-7b-v2:2"
  "techiaith-llama-sft:techiaith-llama-3-2-1b-sft:4"
)

for spec in "${models[@]}"; do
  IFS=: read -r service model workers <<<"$spec"
  if [[ -n "$MODEL_FILTER" && ",$MODEL_FILTER," != *",$model,"* ]]; then
    continue
  fi
  run_id="$RUN_PREFIX-$model"

  echo "== $model: cychwyn y gwasanaeth =="
  inference/stop.sh all
  inference/start.sh "$service"

  echo "== $model: rhedeg hyd at $MAX_CASES achos fesul set =="
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "== $model: cynnig $attempt/$MAX_ATTEMPTS =="
    set +e
    uv run --locked cymraeg-bench run \
      --run-id "$run_id" \
      --models "$model" \
      --max-cases "$MAX_CASES" \
      --seed 1 \
      --workers "$workers"
    run_status=$?
    set -e
    if [[ ! -f "runs/$run_id.sqlite3" ]]; then
      echo "Methodd y rhediad cyn creu cronfa ddata (cod $run_status)." >&2
      continue
    fi

    # Ar dymheredd 0 gyda had sefydlog, bydd ailadrodd allbwn sydd wedi taro'r
    # terfyn tocynnau yn union yr un fath. Sgoriwch hwnnw'n sero nawr, ac
    # ailbrofwch wallau seilwaith yn unig.
    set +e
    uv run --locked cymraeg-bench finalize-output-errors "runs/$run_id.sqlite3"
    attempt_finalize_status=$?
    set -e
    if [[ "$attempt_finalize_status" -eq 0 ]]; then
      break
    fi
    echo "Mae gwallau seilwaith ar ôl; ailbrofir y rheini'n unig."
  done

  set +e
  uv run --locked cymraeg-bench finalize-output-errors "runs/$run_id.sqlite3"
  finalize_status=$?
  set -e
  uv run --locked cymraeg-bench report "runs/$run_id.sqlite3"
  uv run --locked cymraeg-bench ccc-report "runs/$run_id.sqlite3"
  if [[ "$finalize_status" -eq 0 ]]; then
    uv run --locked cymraeg-bench leaderboard \
      "runs/$run_id.sqlite3" --output-dir "results/$run_id"
  else
    echo "Mae gwall seilwaith yn weddill; ni chrëwyd sgorfwrdd ar gyfer $model."
  fi
done

inference/stop.sh all
echo "Cwblhawyd y pum model lleol."
