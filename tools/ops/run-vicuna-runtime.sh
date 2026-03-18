#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/sha256-3603045b543e1c0dfb27f126a3642e7f805e480b84c4781e3d848ace971cba7a"

cd "$REPO_ROOT"
source "$REPO_ROOT/.envrc"

export PATH="/usr/local/cuda-12.8/bin:$PATH"
export CUDACXX="/usr/local/cuda-12.8/bin/nvcc"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

exec "$REPO_ROOT/build-host-cuda-128/bin/llama-server" \
  -m "$MODEL_PATH" \
  --port 8080 \
  --ctx-size 4096 \
  --api-surface openai \
  -ngl 999
