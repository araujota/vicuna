#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

exec python3 "$REPO_ROOT/tools/policy-learning/cli.py" serve-registry \
  --host "${VICUNA_POLICY_REGISTRY_HOST:-127.0.0.1}" \
  --port "${VICUNA_POLICY_REGISTRY_PORT:-18081}" \
  --registry-dir "${VICUNA_POLICY_REGISTRY_DIR:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/policy-registry}" \
  --model-name "${VICUNA_POLICY_MODEL_NAME:-vicuna-governance}" \
  --default-alias "${VICUNA_POLICY_DEFAULT_ALIAS:-candidate}" \
  --fallback-alias "${VICUNA_POLICY_FALLBACK_ALIAS:-champion}"
