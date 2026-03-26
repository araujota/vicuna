#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"
"$REPO_ROOT/tools/ops/sync-openclaw-runtime-state.sh"

if [[ -z "${VICUNA_DEEPSEEK_API_KEY:-}" ]]; then
  printf '[vicuna-runtime] error: VICUNA_DEEPSEEK_API_KEY is required\n' >&2
  exit 1
fi

exec "$REPO_ROOT/build-host-cuda-128/bin/llama-server" \
  --port "${VICUNA_RUNTIME_PORT:-8080}" \
  --api-surface openai \
  --no-webui
