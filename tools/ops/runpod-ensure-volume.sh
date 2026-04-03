#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runpod-runtime-common.sh"

runpod_ensure_local_requirements

if [[ "$RUNPOD_VOLUME_MODE" == "pod" ]]; then
    runpod_log "skipping network volume because RUNPOD_VOLUME_MODE=pod"
    exit 0
fi

volume_id="$(runpod_find_volume_id)"
if [[ -n "$volume_id" ]]; then
    runpod_log "using existing volume $RUNPOD_VOLUME_NAME ($volume_id)"
    printf '%s\n' "$volume_id"
    exit 0
fi

runpod_log "creating network volume $RUNPOD_VOLUME_NAME in $RUNPOD_DATA_CENTER_ID (${RUNPOD_VOLUME_SIZE_GB}GB)"
if ! create_payload="$(runpodctl network-volume create \
    --name "$RUNPOD_VOLUME_NAME" \
    --size "$RUNPOD_VOLUME_SIZE_GB" \
    --data-center-id "$RUNPOD_DATA_CENTER_ID" 2>&1)"; then
    if [[ "$RUNPOD_VOLUME_MODE" == "auto" ]]; then
        runpod_log "network volume creation failed; falling back to pod-attached persistent volume"
        exit 0
    fi
    printf '%s\n' "$create_payload" >&2
    runpod_die "network volume creation failed"
fi
volume_id="$(printf '%s\n' "$create_payload" | runpod_python_json_field "data.get('id') or data.get('networkVolumeId') or data.get('id', '') if isinstance(data, dict) else ''")"
[[ -n "$volume_id" ]] || runpod_die "network volume creation did not return an id"
printf '%s\n' "$volume_id"
