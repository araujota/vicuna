#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

host_mode="${VICUNA_HOST_INFERENCE_MODE:-standard}"
runpod_role="${VICUNA_RUNPOD_INFERENCE_ROLE:-disabled}"
if [[ "$host_mode" == "experimental" && "$runpod_role" == "host" ]]; then
  :
elif [[ -z "${VICUNA_DEEPSEEK_API_KEY:-}" ]]; then
  printf '[vicuna-runtime] error: VICUNA_DEEPSEEK_API_KEY is required\n' >&2
  exit 1
fi

exec "${VICUNA_RUNTIME_BIN:-$REPO_ROOT/build/bin/llama-server}" \
  --port "${VICUNA_RUNTIME_PORT:-8080}" \
  --api-surface openai \
  --no-webui
