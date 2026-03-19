#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/sha256-3603045b543e1c0dfb27f126a3642e7f805e480b84c4781e3d848ace971cba7a"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

exec "$REPO_ROOT/build-host-cuda-128/bin/llama-server" \
  -m "$MODEL_PATH" \
  --port 8080 \
  --ctx-size 4096 \
  --api-surface openai \
  -ngl 999
