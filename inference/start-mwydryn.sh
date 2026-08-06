#!/usr/bin/env bash
set -euo pipefail

ROOT="${AISTEDDFOD_ROOT:-$HOME/aisteddfod}"

# Mae trosglwyddiadau Xet yn araf iawn ar a4090; defnyddia'r llwybr HTTP
# safonol, sy'n gallu ailddechrau ffeiliau .incomplete yn y cache Hugging Face.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="$ROOT/inference/mwydryn-compat${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$ROOT/inference/logs" "$ROOT/inference/run"

exec "$ROOT/inference/.venv/bin/python" "$ROOT/inference/mwydryn_server.py"
