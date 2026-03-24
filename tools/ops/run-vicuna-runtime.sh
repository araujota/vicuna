#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

if [[ ! -f "$VICUNA_RUNTIME_MODEL_PATH" ]]; then
  "$REPO_ROOT/tools/ops/fetch-runtime-model.sh"
fi

if [[ ! -f "$VICUNA_RUNTIME_MODEL_PATH" ]]; then
  printf '[vicuna-runtime] error: managed runtime model missing at %s\n' "$VICUNA_RUNTIME_MODEL_PATH" >&2
  exit 1
fi

exec "$REPO_ROOT/build-host-cuda-128/bin/llama-server" \
  -m "$VICUNA_RUNTIME_MODEL_PATH" \
  --alias "$VICUNA_RUNTIME_MODEL_ALIAS" \
  --jinja \
  --chat-template-file "$VICUNA_RUNTIME_MODEL_CHAT_TEMPLATE_FILE" \
  --reasoning-format "$VICUNA_RUNTIME_MODEL_REASONING_FORMAT" \
  --reasoning-budget -1 \
  --port 8080 \
  --ctx-size "$VICUNA_RUNTIME_CTX_SIZE" \
  --api-surface openai \
  -ngl 999
