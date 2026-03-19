#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

unset VICUNA_RUNTIME_STATE_PATH
unset VICUNA_RUNTIME_STATE_BACKUP_DIR
unset VICUNA_BASH_TOOL_ENABLED
unset VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

EXPECTED_STATE_PATH="$REPO_ROOT/.cache/vicuna/runtime-state.json"
EXPECTED_BACKUP_DIR="$REPO_ROOT/.cache/vicuna/runtime-state-backups"
EXPECTED_CATALOG_PATH="$REPO_ROOT/.cache/vicuna/openclaw-catalog.json"

[[ "$VICUNA_RUNTIME_STATE_PATH" == "$EXPECTED_STATE_PATH" ]] || {
    printf 'expected runtime state path %s, got %s\n' "$EXPECTED_STATE_PATH" "$VICUNA_RUNTIME_STATE_PATH" >&2
    exit 1
}

[[ "$VICUNA_RUNTIME_STATE_BACKUP_DIR" == "$EXPECTED_BACKUP_DIR" ]] || {
    printf 'expected backup dir %s, got %s\n' "$EXPECTED_BACKUP_DIR" "$VICUNA_RUNTIME_STATE_BACKUP_DIR" >&2
    exit 1
}

[[ "$VICUNA_BASH_TOOL_ENABLED" == "1" ]] || {
    printf 'expected bash tool enabled by default, got %s\n' "$VICUNA_BASH_TOOL_ENABLED" >&2
    exit 1
}

[[ "$VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH" == "$EXPECTED_CATALOG_PATH" ]] || {
    printf 'expected OpenClaw catalog path %s, got %s\n' "$EXPECTED_CATALOG_PATH" "$VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH" >&2
    exit 1
}

printf 'runtime-env defaults ok\n'
