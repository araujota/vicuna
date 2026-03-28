#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

TASKS_DIR="${VICUNA_ONGOING_TASKS_DIR:-${VICUNA_STATE_ROOT:-/var/lib/vicuna}/ongoing-tasks}"
TASK_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks-dir)
            TASKS_DIR="$2"
            shift 2
            ;;
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        *)
            printf '[vicuna-ongoing-task] unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$TASK_ID" ]]; then
    printf '[vicuna-ongoing-task] --task-id is required\n' >&2
    exit 2
fi

NODE_BIN="${VICUNA_OPENCLAW_NODE_BIN:-${TELEGRAM_BRIDGE_NODE_BIN:-$(command -v node)}}"
if [[ -z "$NODE_BIN" ]]; then
    printf '[vicuna-ongoing-task] node is required\n' >&2
    exit 2
fi

FLOCK_BIN="${VICUNA_ONGOING_TASKS_FLOCK_BIN:-/usr/bin/flock}"
ONGOING_TASKS_CLI="$VICUNA_REPO_ROOT/tools/openclaw-harness/dist/ongoing-tasks.js"
SECRETS_PATH="${VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH:-/etc/vicuna/openclaw-tool-secrets.json}"
TMP_ROOT="${VICUNA_ONGOING_TASKS_TMPDIR:-$TASKS_DIR/tmp}"
LOCK_FILE="$TASKS_DIR/locks/$TASK_ID.lock"

mkdir -p "$TASKS_DIR" "$TASKS_DIR/locks" "$TASKS_DIR/logs" "$TASKS_DIR/executions" "$TMP_ROOT"

PAYLOAD_BASE64="$("$NODE_BIN" -e 'process.stdout.write(Buffer.from(JSON.stringify({action:"execute", task_id: process.argv[1]})).toString("base64"))' "$TASK_ID")"

export TMPDIR="$TMP_ROOT"
export TMP="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$TMP_ROOT/runtime}"
mkdir -p "$XDG_RUNTIME_DIR"

if ! "$FLOCK_BIN" -n "$LOCK_FILE" \
    "$NODE_BIN" "$ONGOING_TASKS_CLI" \
        "--payload-base64=$PAYLOAD_BASE64" \
        "--secrets-path=$SECRETS_PATH"; then
    status=$?
    if [[ $status -eq 1 ]]; then
        printf '[vicuna-ongoing-task] deferred overlap for %s\n' "$TASK_ID"
        exit 0
    fi
    exit "$status"
fi
