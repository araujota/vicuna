#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runpod-runtime-common.sh"

pod_ports_include_runtime() {
    local pod_id="$1"
    local payload
    payload="$(runpod_get_pod_json "$pod_id")"
    RUNPOD_POD_JSON="$payload" RUNPOD_RUNTIME_PORT="$RUNPOD_RUNTIME_PORT" python3 - <<'PY'
import json
import os

obj = json.loads(os.environ["RUNPOD_POD_JSON"])
runtime_port = str(os.environ["RUNPOD_RUNTIME_PORT"])
needle = f"{runtime_port}/tcp"
ports = obj.get("ports") or []
for item in ports:
    if str(item).strip().lower() == needle:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

runpod_ensure_local_requirements
runpod_ensure_ssh_key
volume_id="$("$REPO_ROOT/tools/ops/runpod-ensure-volume.sh")"

pod_id="$(runpod_find_pod_id)"
if [[ -n "$pod_id" ]]; then
    if ! pod_ports_include_runtime "$pod_id"; then
        runpod_log "updating existing pod $RUNPOD_POD_NAME ($pod_id) to expose ${RUNPOD_RUNTIME_PORT}/tcp"
        runpodctl pod update "$pod_id" --ports "22/tcp,${RUNPOD_RUNTIME_PORT}/tcp" >/dev/null
    fi
    status="$(runpod_pod_status "$pod_id" || true)"
    if [[ "$status" != "RUNNING" && "$status" != "Running" && "$status" != "running" ]]; then
        runpod_log "starting existing pod $RUNPOD_POD_NAME ($pod_id)"
        runpodctl pod start "$pod_id" >/dev/null
    else
        runpod_log "using existing running pod $RUNPOD_POD_NAME ($pod_id)"
    fi
    runpod_wait_for_pod_running "$pod_id"
    printf '%s\n' "$pod_id"
    exit 0
fi

client_balance="$(runpod_client_balance)"
if [[ -n "$client_balance" ]]; then
    if ! python3 -c "import sys; raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)" "$client_balance"; then
        runpod_die "RunPod account balance is too low to create a pod; add funds or reuse an existing pod"
    fi
fi

runpod_log "creating pod $RUNPOD_POD_NAME on $RUNPOD_GPU_ID"
create_cmd=(runpodctl pod create
    --name "$RUNPOD_POD_NAME" \
    --gpu-id "$RUNPOD_GPU_ID" \
    --volume-mount-path "$RUNPOD_VOLUME_MOUNT_PATH" \
    --ports "22/tcp,${RUNPOD_RUNTIME_PORT}/tcp")
if [[ -n "$RUNPOD_TEMPLATE_ID" ]]; then
    create_cmd+=(--template-id "$RUNPOD_TEMPLATE_ID")
else
    create_cmd+=(--image "$RUNPOD_IMAGE")
fi
if [[ -n "$RUNPOD_DATA_CENTER_ID" ]]; then
    create_cmd+=(--data-center-ids "$RUNPOD_DATA_CENTER_ID")
fi
if [[ -n "$RUNPOD_CLOUD_TYPE" ]]; then
    create_cmd+=(--cloud-type "$RUNPOD_CLOUD_TYPE")
fi
if [[ "$RUNPOD_PUBLIC_IP" == "1" || "$RUNPOD_PUBLIC_IP" == "true" ]]; then
    create_cmd+=(--public-ip)
fi
if [[ -n "$volume_id" ]]; then
    create_cmd+=(--network-volume-id "$volume_id")
else
    create_cmd+=(--volume-in-gb "$RUNPOD_VOLUME_SIZE_GB")
fi
create_payload="$("${create_cmd[@]}")"
pod_id="$(printf '%s\n' "$create_payload" | runpod_python_json_field "data.get('id') or data.get('podId') or data.get('pod_id') if isinstance(data, dict) else ''")"
[[ -n "$pod_id" ]] || runpod_die "pod creation did not return an id"
runpod_wait_for_pod_running "$pod_id"
printf '%s\n' "$pod_id"
