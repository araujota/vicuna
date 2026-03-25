#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -f "$REPO_ROOT/.envrc" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.envrc"
fi

export PATH="/usr/local/cuda-12.8/bin:$PATH"
export CUDACXX="${CUDACXX:-/usr/local/cuda-12.8/bin/nvcc}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

export VICUNA_DEEPSEEK_BASE_URL="${VICUNA_DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export VICUNA_DEEPSEEK_MODEL="${VICUNA_DEEPSEEK_MODEL:-deepseek-reasoner}"
export VICUNA_DEEPSEEK_TIMEOUT_MS="${VICUNA_DEEPSEEK_TIMEOUT_MS:-60000}"
export TELEGRAM_BRIDGE_MODEL="${TELEGRAM_BRIDGE_MODEL:-$VICUNA_DEEPSEEK_MODEL}"
