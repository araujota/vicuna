#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runpod-runtime-common.sh"

CAPTURE_HOST_DIR="${VICUNA_RUNPOD_CAPTURE_SYNC_HOST_DIR:-$VICUNA_STATE_ROOT/experimental-capture/live}"
CAPTURE_REMOTE_DIR="${VICUNA_RUNPOD_CAPTURE_SYNC_REMOTE_DIR:-/workspace/llama-state-clean/capture}"
CAPTURE_STATE_PATH="${VICUNA_RUNPOD_CAPTURE_SYNC_STATE_PATH:-$VICUNA_STATE_ROOT/runpod-capture-sync/state.json}"
CAPTURE_LOCK_PATH="${VICUNA_RUNPOD_CAPTURE_SYNC_LOCK_PATH:-$VICUNA_STATE_ROOT/runpod-capture-sync/sync.lock}"
CAPTURE_OWNER="${VICUNA_SERVICE_USER:-vicuna}"
CAPTURE_GROUP="${VICUNA_SERVICE_GROUP:-$CAPTURE_OWNER}"

log() {
    printf '[vicuna-runpod-capture-sync] %s\n' "$*" >&2
}

ensure_dirs() {
    mkdir -p "$CAPTURE_HOST_DIR" "$(dirname "$CAPTURE_STATE_PATH")" "$(dirname "$CAPTURE_LOCK_PATH")"
}

ensure_lock() {
    exec 9>"$CAPTURE_LOCK_PATH"
    if ! flock -n 9; then
        log "sync already running; exiting"
        exit 0
    fi
}

read_state_value() {
    local file_name="$1"
    python3 - "$CAPTURE_STATE_PATH" "$file_name" <<'PY'
import json
import pathlib
import sys

state_path = pathlib.Path(sys.argv[1])
file_name = sys.argv[2]
if not state_path.exists():
    print(0)
    raise SystemExit(0)
state = json.loads(state_path.read_text(encoding="utf-8"))
value = (((state.get("files") or {}).get(file_name) or {}).get("line_count"))
print(int(value or 0))
PY
}

write_state_value() {
    local file_name="$1"
    local line_count="$2"
    local pod_id="$3"
    python3 - "$CAPTURE_STATE_PATH" "$file_name" "$line_count" "$pod_id" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

state_path = pathlib.Path(sys.argv[1])
file_name = sys.argv[2]
line_count = int(sys.argv[3])
pod_id = sys.argv[4]

state = {}
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))

files = state.setdefault("files", {})
files[file_name] = {"line_count": line_count}
state["pod_id"] = pod_id
state["updated_at"] = datetime.now(timezone.utc).isoformat()

tmp_path = state_path.with_suffix(".tmp")
tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp_path.replace(state_path)
PY
}

remote_line_count() {
    local pod_id="$1"
    local remote_file="$2"
    runpod_pod_bash "$pod_id" <<EOF
python3 - "$remote_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)

count = 0
with path.open("r", encoding="utf-8") as handle:
    for count, _ in enumerate(handle, start=1):
        pass
print(count)
PY
EOF
}

fetch_new_lines() {
    local pod_id="$1"
    local remote_file="$2"
    local start_line="$3"
    runpod_pod_bash "$pod_id" <<EOF
python3 - "$remote_file" "$start_line" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
start_line = int(sys.argv[2])
if not path.exists():
    raise SystemExit(0)

with path.open("r", encoding="utf-8") as handle:
    for index, line in enumerate(handle, start=1):
        if index > start_line:
            sys.stdout.write(line)
PY
EOF
}

append_file_delta() {
    local pod_id="$1"
    local file_name="$2"
    local host_path="$CAPTURE_HOST_DIR/$file_name"
    local remote_path="$CAPTURE_REMOTE_DIR/$file_name"
    local current_count
    current_count="$(remote_line_count "$pod_id" "$remote_path" | tail -n 1 | tr -d '\r')"
    [[ -n "$current_count" ]] || current_count=0

    local last_synced
    last_synced="$(read_state_value "$file_name")"
    local reset_detected=0
    if (( current_count < last_synced )); then
        log "$file_name shrank from $last_synced to $current_count; resetting offset"
        last_synced=0
        reset_detected=1
    fi

    if (( current_count == last_synced )); then
        if (( reset_detected )); then
            write_state_value "$file_name" "$current_count" "$pod_id"
        fi
        log "$file_name has no new rows"
        return 0
    fi

    local tmp_payload
    tmp_payload="$(mktemp "${TMPDIR:-/tmp}/vicuna-runpod-capture-${file_name}.XXXXXX")"
    fetch_new_lines "$pod_id" "$remote_path" "$last_synced" >"$tmp_payload"
    if [[ -s "$tmp_payload" ]]; then
        cat "$tmp_payload" >>"$host_path"
        chown "$CAPTURE_OWNER:$CAPTURE_GROUP" "$host_path"
        log "$file_name appended $(wc -l <"$tmp_payload" | tr -d ' ') rows"
    else
        log "$file_name had no transferable payload despite line-count delta"
    fi
    rm -f "$tmp_payload"

    write_state_value "$file_name" "$current_count" "$pod_id"
}

main() {
    ensure_dirs
    ensure_lock
    runpod_ensure_local_requirements

    local pod_id
    pod_id="$(runpod_find_pod_id || true)"
    if [[ -z "$pod_id" ]]; then
        log "no pod named ${RUNPOD_POD_NAME} found; exiting"
        exit 0
    fi

    local status
    status="$(runpod_pod_status "$pod_id" || true)"
    if [[ "$status" != "RUNNING" ]]; then
        log "pod $pod_id status is ${status:-unknown}; exiting"
        exit 0
    fi

    append_file_delta "$pod_id" "transitions.jsonl"
    append_file_delta "$pod_id" "decode_traces.jsonl"
    append_file_delta "$pod_id" "emotive_traces.jsonl"
}

main "$@"
