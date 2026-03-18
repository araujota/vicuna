#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/sha256-3603045b543e1c0dfb27f126a3642e7f805e480b84c4781e3d848ace971cba7a"

cd "$REPO_ROOT"
source "$REPO_ROOT/.envrc"

export PATH="/usr/local/cuda-12.8/bin:$PATH"
export CUDACXX="/usr/local/cuda-12.8/bin/nvcc"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export VICUNA_BASH_TOOL_ENABLED="${VICUNA_BASH_TOOL_ENABLED:-1}"
export VICUNA_BASH_TOOL_PATH="${VICUNA_BASH_TOOL_PATH:-$(command -v bash)}"
export VICUNA_BASH_TOOL_WORKDIR="${VICUNA_BASH_TOOL_WORKDIR:-$REPO_ROOT}"
export VICUNA_BASH_TOOL_TIMEOUT_MS="${VICUNA_BASH_TOOL_TIMEOUT_MS:-15000}"
export VICUNA_BASH_TOOL_MAX_STDOUT_BYTES="${VICUNA_BASH_TOOL_MAX_STDOUT_BYTES:-16384}"
export VICUNA_BASH_TOOL_MAX_STDERR_BYTES="${VICUNA_BASH_TOOL_MAX_STDERR_BYTES:-8192}"
export VICUNA_BASH_TOOL_LOGIN_SHELL="${VICUNA_BASH_TOOL_LOGIN_SHELL:-1}"
export VICUNA_BASH_TOOL_INHERIT_ENV="${VICUNA_BASH_TOOL_INHERIT_ENV:-1}"

exec "$REPO_ROOT/build-host-cuda-128/bin/llama-server" \
  -m "$MODEL_PATH" \
  --port 8080 \
  --ctx-size 4096 \
  --api-surface openai \
  -ngl 999
