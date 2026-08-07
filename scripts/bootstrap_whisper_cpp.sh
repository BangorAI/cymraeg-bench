#!/usr/bin/env bash
set -euo pipefail

GWREIDDYN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config/voice-runtimes.env
source "$GWREIDDYN/config/voice-runtimes.env"
COD="${WHISPER_CPP_DIR:-$GWREIDDYN/vendor/whisper.cpp}"
ADEILAD="$COD/build"

if [[ ! -d "$COD/.git" ]]; then
  mkdir -p "$(dirname "$COD")"
  git clone --filter=blob:none "$WHISPER_CPP_REPO" "$COD"
fi
git -C "$COD" fetch origin "$WHISPER_CPP_COMMIT" --depth 1
git -C "$COD" checkout --detach "$WHISPER_CPP_COMMIT"

cmake -S "$COD" -B "$ADEILAD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DWHISPER_BUILD_EXAMPLES=ON
cmake --build "$ADEILAD" --parallel "${WHISPER_CPP_BUILD_JOBS:-8}"

CLI="$ADEILAD/bin/whisper-cli"
if [[ ! -x "$CLI" ]]; then
  echo "Heb ganfod whisper-cli ar ôl build: $CLI" >&2
  exit 2
fi
echo "$CLI"
