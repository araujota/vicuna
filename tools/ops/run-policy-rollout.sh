#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

POLICY_PYTHON_BIN="${VICUNA_POLICY_PYTHON_BIN:-python3}"

exec "$POLICY_PYTHON_BIN" "$REPO_ROOT/tools/policy-learning/cli.py" advance-rollout \
  --server "${VICUNA_POLICY_SERVER_URL:-http://127.0.0.1:${VICUNA_RUNTIME_PORT:-8080}}" \
  --registry-dir "${VICUNA_POLICY_REGISTRY_DIR:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/policy-registry}" \
  --model-name "${VICUNA_POLICY_MODEL_NAME:-vicuna-governance}" \
  --runtime-env-file "${VICUNA_POLICY_LIVE_ROLLOUT_RUNTIME_ENV_FILE:-/etc/vicuna/vicuna.env}" \
  --runtime-service "${VICUNA_POLICY_LIVE_ROLLOUT_RUNTIME_SERVICE:-vicuna-runtime.service}" \
  --state-path "${VICUNA_POLICY_LIVE_ROLLOUT_STATE_PATH:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/policy-rollout/state.json}" \
  --journal-dir "${VICUNA_POLICY_LIVE_ROLLOUT_JOURNAL_DIR:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/policy-rollout/journal}" \
  --shadow-min-requests "${VICUNA_POLICY_SHADOW_MIN_REQUESTS:-25}" \
  --shadow-max-disagreement-rate "${VICUNA_POLICY_SHADOW_MAX_DISAGREEMENT_RATE:-0.25}" \
  --shadow-max-candidate-failure-rate "${VICUNA_POLICY_SHADOW_MAX_CANDIDATE_FAILURE_RATE:-0.10}"
