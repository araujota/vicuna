#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

POLICY_SERVER_URL="${VICUNA_POLICY_SERVER_URL:-http://127.0.0.1:${VICUNA_RUNTIME_PORT:-8080}}"
POLICY_DATASET_DIR="${VICUNA_POLICY_DATASET_DIR:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/policy-datasets/nightly}"
POLICY_DATASET_ID="${VICUNA_POLICY_DATASET_ID:-vicuna-governance-nightly-v1}"
POLICY_REGISTRY_DIR="${VICUNA_POLICY_REGISTRY_DIR:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/policy-registry}"
POLICY_RUN_ROOT="${VICUNA_POLICY_RUN_ROOT:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/policy-runs}"
POLICY_MODEL_NAME="${VICUNA_POLICY_MODEL_NAME:-vicuna-governance}"
POLICY_LIMIT="${VICUNA_POLICY_LIMIT:-512}"
POLICY_TIMEOUT_MS="${VICUNA_POLICY_TIMEOUT_MS:-5000}"
POLICY_MIN_RECORD_COUNT="${VICUNA_POLICY_MIN_RECORD_COUNT:-25}"
POLICY_MIN_EXACT_MATCH_RATE="${VICUNA_POLICY_MIN_EXACT_MATCH_RATE:-0.55}"
POLICY_MAX_INVALID_ACTION_RATE="${VICUNA_POLICY_MAX_INVALID_ACTION_RATE:-0.0}"
POLICY_MIN_REWARD_DELTA="${VICUNA_POLICY_MIN_REWARD_DELTA:-0.0}"
POLICY_PYTHON_BIN="${VICUNA_POLICY_PYTHON_BIN:-python3}"

exec "$POLICY_PYTHON_BIN" "$REPO_ROOT/tools/policy-learning/cli.py" nightly-batch \
  --server "$POLICY_SERVER_URL" \
  --dataset-dir "$POLICY_DATASET_DIR" \
  --dataset-id "$POLICY_DATASET_ID" \
  --registry-dir "$POLICY_REGISTRY_DIR" \
  --model-name "$POLICY_MODEL_NAME" \
  --run-root "$POLICY_RUN_ROOT" \
  --limit "$POLICY_LIMIT" \
  --timeout-ms "$POLICY_TIMEOUT_MS" \
  --min-record-count "$POLICY_MIN_RECORD_COUNT" \
  --min-exact-match-rate "$POLICY_MIN_EXACT_MATCH_RATE" \
  --max-invalid-action-rate "$POLICY_MAX_INVALID_ACTION_RATE" \
  --min-reward-delta "$POLICY_MIN_REWARD_DELTA"
