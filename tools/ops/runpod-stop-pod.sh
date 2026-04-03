#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runpod-runtime-common.sh"

runpod_ensure_local_requirements
pod_id="$(runpod_find_pod_id)"
if [[ -z "$pod_id" ]]; then
    runpod_log "no pod named ${RUNPOD_POD_NAME} exists; nothing to stop"
    exit 0
fi

runpod_log "stopping pod ${RUNPOD_POD_NAME} (${pod_id})"
runpodctl pod stop "$pod_id" >/dev/null
printf '%s\n' "$pod_id"
