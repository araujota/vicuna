#!/usr/bin/env bash
set -euo pipefail

RUNPOD_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$RUNPOD_SCRIPT_ROOT}"

export RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-vicuna-llamacpp-smoke-secure}"
export RUNPOD_VOLUME_NAME="${RUNPOD_VOLUME_NAME:-vicuna-llamacpp-smoke}"
export RUNPOD_VOLUME_MODE="${RUNPOD_VOLUME_MODE:-auto}"
export RUNPOD_TEMPLATE_ID="${RUNPOD_TEMPLATE_ID:-}"
export RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
export RUNPOD_GPU_ID="${RUNPOD_GPU_ID:-NVIDIA A100 SXM}"
export RUNPOD_DATA_CENTER_ID="${RUNPOD_DATA_CENTER_ID:-}"
export RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
export RUNPOD_PUBLIC_IP="${RUNPOD_PUBLIC_IP:-0}"
export RUNPOD_VOLUME_SIZE_GB="${RUNPOD_VOLUME_SIZE_GB:-80}"
export RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-20}"
export RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/workspace}"
export RUNPOD_RUNTIME_HOST="${RUNPOD_RUNTIME_HOST:-0.0.0.0}"
export RUNPOD_RUNTIME_PORT="${RUNPOD_RUNTIME_PORT:-8080}"
export RUNPOD_RUNTIME_AUTH_TOKEN="${RUNPOD_RUNTIME_AUTH_TOKEN:-${VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN:-runpod-node-smoke-token}}"
export RUNPOD_LOCAL_ARTIFACT_ROOT="${RUNPOD_LOCAL_ARTIFACT_ROOT:-${REPO_ROOT}/.cache/vicuna/runpod}"
export RUNPOD_POD_HTTP_TIMEOUT_SECONDS="${RUNPOD_POD_HTTP_TIMEOUT_SECONDS:-120}"
export RUNPOD_SSH_PRIVATE_KEY_PATH="${RUNPOD_SSH_PRIVATE_KEY_PATH:-${HOME}/.runpod/ssh/RunPod-Key-Go}"
export RUNPOD_RUNTIME_SCHEME="${RUNPOD_RUNTIME_SCHEME:-http}"

runpod_log() {
    printf '[vicuna-runpod] %s\n' "$*" >&2
}

runpod_die() {
    printf '[vicuna-runpod] error: %s\n' "$*" >&2
    exit 1
}

runpod_require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || runpod_die "required command not found: $cmd"
}

runpod_python_json_field() {
    local expression="$1"
    python3 -c "import json,sys; data=json.load(sys.stdin); value=($expression); print('' if value is None else value)"
}

runpod_load_api_key() {
    if [[ -z "${RUNPOD_API_KEY:-}" && -r "${HOME}/.zshrc" ]]; then
        local extracted_key
        extracted_key="$(python3 - <<'PY'
import pathlib, re
path = pathlib.Path.home() / ".zshrc"
text = path.read_text(errors="ignore")
patterns = [
    r'export\s+RUNPOD_API_KEY="([^"]+)"',
    r"export\s+RUNPOD_API_KEY='([^']+)'",
    r'export\s+RUNPOD_API_KEY=([^\s#]+)',
]
for pattern in patterns:
    match = re.search(pattern, text)
    if match:
        print(match.group(1))
        break
PY
)"
        if [[ -n "${extracted_key}" ]]; then
            export RUNPOD_API_KEY="${extracted_key}"
        else
            set +u
            set -a
            # shellcheck disable=SC1090
            source "${HOME}/.zshrc" >/dev/null 2>&1 || true
            set +a
            set -u
        fi
    fi
    if [[ -z "${RUNPOD_API_KEY:-}" && ! -r "${HOME}/.runpod/config.toml" ]]; then
        runpod_die "RUNPOD_API_KEY is not configured and ~/.runpod/config.toml is not readable"
    fi
    if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
        export RUNPOD_API_KEY
    fi
}

runpod_ensure_local_requirements() {
    runpod_load_api_key
    runpod_require_cmd runpodctl
    runpod_require_cmd python3
    runpod_require_cmd ssh
    runpod_require_cmd scp
    runpod_require_cmd tar
    mkdir -p "${RUNPOD_LOCAL_ARTIFACT_ROOT}"
}

runpod_find_volume_id() {
    local payload
    payload="$(runpodctl network-volume list)"
    RUNPOD_JSON_PAYLOAD="$payload" python3 - "$RUNPOD_VOLUME_NAME" <<'PY'
import json, os, sys
target = sys.argv[1]
for item in json.loads(os.environ["RUNPOD_JSON_PAYLOAD"]):
    if item.get("name") == target:
        print(item.get("id", ""))
        break
PY
}

runpod_find_pod_id() {
    local payload
    payload="$(runpodctl pod list -a)"
    RUNPOD_JSON_PAYLOAD="$payload" python3 - "$RUNPOD_POD_NAME" <<'PY'
import json, os, sys
target = sys.argv[1]
for item in json.loads(os.environ["RUNPOD_JSON_PAYLOAD"]):
    if item.get("name") == target:
        print(item.get("id", ""))
        break
PY
}

runpod_get_pod_json() {
    local pod_id="$1"
    runpodctl pod get "$pod_id" --include-network-volume
}

runpod_get_pod_rest_json() {
    local pod_id="$1"
    runpod_load_api_key
    curl -fsS \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        "https://rest.runpod.io/v1/pods/${pod_id}"
}

runpod_runtime_endpoint_json() {
    local pod_id="$1"
    local pod_json
    pod_json="$(runpod_get_pod_json "$pod_id")"
    local pod_rest_json
    pod_rest_json="$(runpod_get_pod_rest_json "$pod_id" || true)"
    RUNPOD_POD_JSON="$pod_json" RUNPOD_POD_REST_JSON="$pod_rest_json" RUNPOD_RUNTIME_PORT="$RUNPOD_RUNTIME_PORT" RUNPOD_RUNTIME_SCHEME="$RUNPOD_RUNTIME_SCHEME" python3 - <<'PY'
import json
import os
import sys

obj = json.loads(os.environ["RUNPOD_POD_JSON"])
raw_rest = os.environ.get("RUNPOD_POD_REST_JSON", "").strip()
rest_obj = json.loads(raw_rest) if raw_rest else {}
runtime_port = str(os.environ["RUNPOD_RUNTIME_PORT"])
scheme = os.environ.get("RUNPOD_RUNTIME_SCHEME", "http")

public_ip = (
    rest_obj.get("publicIp")
    or rest_obj.get("publicIP")
    or rest_obj.get("ip")
    or obj.get("publicIp")
    or obj.get("publicIP")
    or obj.get("ip")
)
port_mappings = rest_obj.get("portMappings") or obj.get("portMappings") or {}
external_port = None

if isinstance(port_mappings, dict):
    value = port_mappings.get(runtime_port)
    if value is not None:
        external_port = value

def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)

for candidate in walk(obj):
    if external_port is not None:
        break
    keys = {str(k).lower(): v for k, v in candidate.items()}
    internal = keys.get("privateport")
    if internal is None:
        internal = keys.get("internalport")
    if internal is None:
        internal = keys.get("containerport")
    if internal is None:
        internal = keys.get("port")
    if str(internal) != runtime_port:
        continue
    external_port = (
        keys.get("publicport")
        or keys.get("externalport")
        or keys.get("hostport")
        or keys.get("publishedport")
    )

payload = {
    "pod_id": obj.get("id"),
    "public_ip": public_ip,
    "runtime_port": int(runtime_port),
    "external_port": int(external_port) if external_port is not None else None,
    "scheme": scheme,
    "url": None,
}
if public_ip and external_port is not None:
    payload["url"] = f"{scheme}://{public_ip}:{int(external_port)}"

print(json.dumps(payload))
PY
}

runpod_user_json() {
    runpodctl user
}

runpod_client_balance() {
    local payload
    payload="$(runpod_user_json)"
    printf '%s\n' "$payload" | runpod_python_json_field "data.get('clientBalance') if isinstance(data, dict) else ''"
}

runpod_pod_status() {
    local pod_id="$1"
    local payload
    payload="$(runpod_get_pod_json "$pod_id")"
    RUNPOD_JSON_PAYLOAD="$payload" python3 - <<'PY'
import json, os
obj = json.loads(os.environ["RUNPOD_JSON_PAYLOAD"])
for key in ("desiredStatus", "status", "lastStatus", "machineStatus"):
    value = obj.get(key)
    if isinstance(value, str) and value:
        print(value)
        break
PY
}

runpod_ensure_ssh_key() {
    local pubkey="${RUNPOD_SSH_PUBLIC_KEY_PATH:-$HOME/.ssh/id_ed25519.pub}"
    [[ -r "$pubkey" ]] || runpod_die "public SSH key not found: $pubkey"
    local add_output
    if add_output="$(runpodctl ssh add-key --key-file "$pubkey" 2>&1)"; then
        return 0
    fi
    if grep -Eiq 'already exists|duplicate|already added' <<<"$add_output"; then
        return 0
    fi
    printf '%s\n' "$add_output" >&2
    runpod_die "failed to add local SSH key to RunPod account"
}

runpod_wait_for_pod_running() {
    local pod_id="$1"
    local attempt
    for attempt in $(seq 1 120); do
        local status
        status="$(runpod_pod_status "$pod_id" || true)"
        case "$status" in
            RUNNING|Running|running)
                return 0
                ;;
        esac
        sleep 5
    done
    runpod_die "pod did not reach RUNNING state within timeout: $pod_id"
}

runpod_wait_for_ssh_info() {
    local pod_id="$1"
    local attempt
    for attempt in $(seq 1 60); do
        local ssh_info_json
        ssh_info_json="$(runpodctl ssh info "$pod_id" 2>/dev/null || true)"
        if [[ -n "$ssh_info_json" ]] && RUNPOD_SSH_INFO_JSON="$ssh_info_json" python3 - <<'PY'
import json
import os
import sys

raw = os.environ.get("RUNPOD_SSH_INFO_JSON", "")
if not raw.strip():
    raise SystemExit(1)
try:
    info = json.loads(raw)
except Exception:
    raise SystemExit(1)

def dig(obj, *path):
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

for value in (
    info.get("sshCommand"),
    info.get("ssh_command"),
    info.get("command"),
    dig(info, "ssh", "command"),
    dig(info, "ssh", "sshCommand"),
    info.get("host"),
    info.get("hostname"),
    info.get("ip"),
    info.get("publicIp"),
    dig(info, "ssh", "host"),
    dig(info, "ssh", "hostname"),
):
    if isinstance(value, str) and value:
        raise SystemExit(0)

raise SystemExit(1)
PY
        then
            printf '%s\n' "$ssh_info_json"
            return 0
        fi
        sleep 5
    done
    runpod_die "RunPod SSH info did not become ready for pod: $pod_id"
}

runpod_connection_exec() {
    local mode="$1"
    local pod_id="$2"
    shift 2
    local ssh_info_json
    ssh_info_json="$(runpod_wait_for_ssh_info "$pod_id")"
    RUNPOD_SSH_INFO_JSON="$ssh_info_json" RUNPOD_EXEC_MODE="$mode" python3 - "$@" <<'PY'
import json
import os
import shlex
import subprocess
import sys

info = json.loads(os.environ["RUNPOD_SSH_INFO_JSON"])
mode = os.environ["RUNPOD_EXEC_MODE"]

def dig(obj, *path):
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

def coalesce(*values):
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None

command = coalesce(
    info.get("sshCommand"),
    info.get("ssh_command"),
    info.get("command"),
    dig(info, "ssh", "command"),
    dig(info, "ssh", "sshCommand"),
)

host = coalesce(
    info.get("host"),
    info.get("hostname"),
    info.get("ip"),
    info.get("publicIp"),
    dig(info, "ssh", "host"),
    dig(info, "ssh", "hostname"),
)
user = coalesce(
    info.get("user"),
    info.get("username"),
    dig(info, "ssh", "user"),
    dig(info, "ssh", "username"),
) or "root"
key_path = coalesce(
    info.get("keyPath"),
    info.get("identityFile"),
    dig(info, "ssh", "keyPath"),
    dig(info, "ssh", "identityFile"),
)
override_key_path = os.environ.get("RUNPOD_SSH_PRIVATE_KEY_PATH")
if override_key_path and os.path.exists(override_key_path):
    key_path = override_key_path
port = info.get("port") or dig(info, "ssh", "port") or 22

ssh_base = None
if isinstance(command, str) and command.startswith("ssh "):
    ssh_base = shlex.split(command)
    injected = []
    saw_strict = False
    saw_known_hosts = False
    saw_identity = False
    i = 0
    while i < len(ssh_base):
        token = ssh_base[i]
        injected.append(token)
        if token == "-i" and i + 1 < len(ssh_base):
            value = ssh_base[i + 1]
            if override_key_path and os.path.exists(override_key_path):
                value = override_key_path
            injected.append(value)
            saw_identity = True
            i += 2
            continue
        if token == "-o" and i + 1 < len(ssh_base):
            option = ssh_base[i + 1]
            injected.append(option)
            if option == "StrictHostKeyChecking=no":
                saw_strict = True
            if option == "UserKnownHostsFile=/dev/null":
                saw_known_hosts = True
            i += 2
            continue
        i += 1
    if not saw_strict:
        injected[1:1] = ["-o", "StrictHostKeyChecking=no"]
    if not saw_known_hosts:
        injected[1:1] = ["-o", "UserKnownHostsFile=/dev/null"]
    if not saw_identity and override_key_path and os.path.exists(override_key_path):
        injected[1:1] = ["-i", override_key_path]
    ssh_base = injected
else:
    if not host:
        raise SystemExit("RunPod SSH info did not contain a usable host")
    ssh_base = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]
    if key_path:
        ssh_base += ["-i", key_path]
    if port:
        ssh_base += ["-p", str(port)]
    ssh_base += [f"{user}@{host}"]

def scp_base():
    base = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    parsed = list(ssh_base)
    i = 0
    remote = None
    while i < len(parsed):
        token = parsed[i]
        if token == "ssh":
            i += 1
            continue
        if token in {"-i", "-p", "-J", "-o", "-F", "-l"} and i + 1 < len(parsed):
            mapped = "-P" if token == "-p" else token
            base += [mapped, parsed[i + 1]]
            i += 2
            continue
        if token.startswith("-"):
            base.append(token)
            i += 1
            continue
        remote = token
        i += 1
    if remote is None:
        raise SystemExit("failed to derive remote host for scp")
    return base, remote

if mode == "exec":
    remote_cmd = os.environ["RUNPOD_REMOTE_CMD"]
    subprocess.run(ssh_base + [remote_cmd], check=True)
elif mode == "bash":
    script = os.environ["RUNPOD_REMOTE_SCRIPT"]
    subprocess.run(ssh_base + ["bash", "-se"], input=script, text=True, check=True)
elif mode == "scp_to":
    local_path = os.environ["RUNPOD_LOCAL_PATH"]
    remote_path = os.environ["RUNPOD_REMOTE_PATH"]
    base, remote = scp_base()
    subprocess.run(base + [local_path, f"{remote}:{remote_path}"], check=True)
elif mode == "scp_from":
    local_path = os.environ["RUNPOD_LOCAL_PATH"]
    remote_path = os.environ["RUNPOD_REMOTE_PATH"]
    base, remote = scp_base()
    subprocess.run(base + [f"{remote}:{remote_path}", local_path], check=True)
else:
    raise SystemExit(f"unsupported mode: {mode}")
PY
}

runpod_pod_exec() {
    local pod_id="$1"
    shift
    local remote_cmd="$*"
    RUNPOD_REMOTE_CMD="$remote_cmd" runpod_connection_exec exec "$pod_id"
}

runpod_pod_bash() {
    local pod_id="$1"
    local remote_script
    remote_script="$(cat)"
    RUNPOD_REMOTE_SCRIPT="$remote_script" runpod_connection_exec bash "$pod_id"
}

runpod_copy_to_pod() {
    local pod_id="$1"
    local local_path="$2"
    local remote_path="$3"
    RUNPOD_LOCAL_PATH="$local_path" RUNPOD_REMOTE_PATH="$remote_path" runpod_connection_exec scp_to "$pod_id"
}

runpod_copy_from_pod() {
    local pod_id="$1"
    local remote_path="$2"
    local local_path="$3"
    RUNPOD_LOCAL_PATH="$local_path" RUNPOD_REMOTE_PATH="$remote_path" runpod_connection_exec scp_from "$pod_id"
}
