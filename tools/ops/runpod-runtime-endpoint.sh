#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runpod-runtime-common.sh"

runpod_ensure_local_requirements
pod_id="${1:-}"
if [[ -z "$pod_id" ]]; then
    pod_id="$(runpod_find_pod_id)"
fi
[[ -n "$pod_id" ]] || runpod_die "no pod named ${RUNPOD_POD_NAME} exists"
runpod_wait_for_pod_running "$pod_id"

endpoint_json="$(runpod_runtime_endpoint_json "$pod_id")"
for _ in $(seq 1 ${RUNPOD_POD_HTTP_TIMEOUT_SECONDS}); do
    if [[ -n "$endpoint_json" ]] && python3 - "$endpoint_json" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
raise SystemExit(0 if payload.get("url") else 1)
PY
    then
        printf '%s\n' "$endpoint_json"
        exit 0
    fi
    sleep 1
    endpoint_json="$(runpod_runtime_endpoint_json "$pod_id")"
done

printf '%s\n' "$endpoint_json" >&2
runpod_die "runtime endpoint is unavailable for pod ${pod_id}"
