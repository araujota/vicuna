#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

MODEL_PATH="${VICUNA_RUNTIME_MODEL_PATH:-}"
MODEL_URL="${VICUNA_RUNTIME_MODEL_URL:-}"

usage() {
    cat <<'EOF'
Usage: fetch-runtime-model.sh

Downloads the managed runtime GGUF into the configured local model path if it
is not already present.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "$MODEL_PATH" || -z "$MODEL_URL" ]]; then
    printf '[vicuna-model-fetch] error: VICUNA_RUNTIME_MODEL_PATH and VICUNA_RUNTIME_MODEL_URL must be set\n' >&2
    exit 1
fi

if [[ -f "$MODEL_PATH" ]]; then
    printf '[vicuna-model-fetch] model already present: %s\n' "$MODEL_PATH"
    exit 0
fi

mkdir -p "$(dirname "$MODEL_PATH")"
tmp_path="${MODEL_PATH}.partial"
trap 'rm -f "$tmp_path"' INT TERM ERR

printf '[vicuna-model-fetch] downloading %s\n' "$MODEL_URL"
curl --fail --location --retry 5 --retry-delay 5 --output "$tmp_path" "$MODEL_URL"
mv "$tmp_path" "$MODEL_PATH"
trap - INT TERM ERR
printf '[vicuna-model-fetch] ready: %s\n' "$MODEL_PATH"
