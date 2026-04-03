#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SYSTEM_ENV_FILE="${VICUNA_SYSTEM_ENV_FILE:-/etc/vicuna/vicuna.env}"
RUNTIME_SERVICE="${VICUNA_RUNTIME_SERVICE_NAME:-vicuna-runtime.service}"
MISTRAL_RELAY_SERVICE="${VICUNA_RUNPOD_MISTRAL_RELAY_SERVICE_NAME:-vicuna-runpod-mistral-relay.service}"
RUNPOD_CREDENTIAL_DIR="${RUNPOD_CREDENTIAL_DIR:-/etc/vicuna/runpod}"
RUNPOD_TUNNEL_LOCAL_PORT="${RUNPOD_TUNNEL_LOCAL_PORT:-18080}"
MISTRAL_RELAY_HOST="${VICUNA_RUNPOD_MISTRAL_RELAY_HOST:-127.0.0.1}"
MISTRAL_RELAY_PORT="${VICUNA_RUNPOD_MISTRAL_RELAY_PORT:-18082}"
RUNTIME_PORT_DEFAULT="${VICUNA_RUNTIME_PORT:-8080}"
RUNPOD_POD_NAME_DEFAULT="${RUNPOD_POD_NAME_DEFAULT:-vicuna-llamacpp-smoke-secure}"
RUNPOD_RELAY_TIMEOUT_MS_DEFAULT="${RUNPOD_RELAY_TIMEOUT_MS_DEFAULT:-1800000}"

usage() {
    cat <<'EOF'
Usage: host-inference-mode.sh <status|toggle|set> [standard|experimental] [--json]

Toggle the deployed host between standard DeepSeek serving and experimental
RunPod-backed relay serving.
EOF
}

die() {
    printf '[vicuna-host-mode] error: %s\n' "$*" >&2
    exit 1
}

require_root() {
    if (( EUID != 0 )); then
        die "run this helper as root"
    fi
}

load_env_file() {
    if [[ -r "$SYSTEM_ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$SYSTEM_ENV_FILE"
        set +a
    fi
}

env_get() {
    local key="$1"
    python3 - "$SYSTEM_ENV_FILE" "$key" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    raise SystemExit(0)
for line in path.read_text(encoding="utf-8").splitlines():
    if line.startswith(f"{key}="):
        print(line.split("=", 1)[1])
        raise SystemExit(0)
PY
}

env_set() {
    local key="$1"
    local value="$2"
    python3 - "$SYSTEM_ENV_FILE" "$key" "$value" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = []
if path.exists():
    lines = path.read_text(encoding="utf-8").splitlines()
updated = False
for index, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[index] = f"{key}={value}"
        updated = True
        break
if not updated:
    lines.append(f"{key}={value}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

runtime_port() {
    local value
    value="$(env_get "VICUNA_RUNTIME_PORT")"
    if [[ -n "$value" ]]; then
        printf '%s\n' "$value"
        return
    fi
    printf '%s\n' "$RUNTIME_PORT_DEFAULT"
}

resolved_runpod_pod_name() {
    local value
    value="${RUNPOD_POD_NAME:-}"
    if [[ -z "$value" ]]; then
        value="$(env_get "RUNPOD_POD_NAME")"
    fi
    if [[ -z "$value" ]]; then
        value="$RUNPOD_POD_NAME_DEFAULT"
    fi
    printf '%s\n' "$value"
}

resolved_runpod_relay_timeout_ms() {
    local value
    value="${VICUNA_RUNPOD_INFERENCE_TIMEOUT_MS:-}"
    if [[ -z "$value" ]]; then
        value="$(env_get "VICUNA_RUNPOD_INFERENCE_TIMEOUT_MS")"
    fi
    if [[ -z "$value" ]]; then
        value="$RUNPOD_RELAY_TIMEOUT_MS_DEFAULT"
    fi
    printf '%s\n' "$value"
}

current_mode() {
    local value
    value="$(env_get "VICUNA_HOST_INFERENCE_MODE")"
    if [[ "$value" == "experimental" ]]; then
        printf 'experimental\n'
        return
    fi
    printf 'standard\n'
}

ensure_runtime_auth_token() {
    local token
    token="$(env_get "VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN")"
    if [[ -z "$token" ]]; then
        token="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(24))
PY
)"
        env_set "VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN" "$token"
    fi
    env_set "RUNPOD_RUNTIME_AUTH_TOKEN" "$token"
    printf '%s\n' "$token"
}

ensure_runpod_credentials() {
    local api_key="${RUNPOD_API_KEY:-}"
    if [[ -z "$api_key" ]]; then
        api_key="$(env_get "RUNPOD_API_KEY")"
    fi
    [[ -n "$api_key" ]] || die "RUNPOD_API_KEY is not configured on the host"

    local private_key_path="${RUNPOD_SSH_PRIVATE_KEY_PATH:-}"
    local public_key_path="${RUNPOD_SSH_PUBLIC_KEY_PATH:-}"
    if [[ -z "$private_key_path" ]]; then
        private_key_path="$(env_get "RUNPOD_SSH_PRIVATE_KEY_PATH")"
    fi
    if [[ -z "$public_key_path" ]]; then
        public_key_path="$(env_get "RUNPOD_SSH_PUBLIC_KEY_PATH")"
    fi

    if [[ -z "$private_key_path" ]]; then
        private_key_path="${RUNPOD_CREDENTIAL_DIR}/RunPod-Key-Go"
    fi
    if [[ -z "$public_key_path" ]]; then
        public_key_path="${private_key_path}.pub"
    fi

    [[ -r "$private_key_path" ]] || die "RunPod SSH private key is missing: $private_key_path"
    [[ -r "$public_key_path" ]] || die "RunPod SSH public key is missing: $public_key_path"

    env_set "RUNPOD_API_KEY" "$api_key"
    env_set "RUNPOD_SSH_PRIVATE_KEY_PATH" "$private_key_path"
    env_set "RUNPOD_SSH_PUBLIC_KEY_PATH" "$public_key_path"
    export RUNPOD_API_KEY="$api_key"
    export RUNPOD_SSH_PRIVATE_KEY_PATH="$private_key_path"
    export RUNPOD_SSH_PUBLIC_KEY_PATH="$public_key_path"
}

wait_for_runtime_health() {
    local port="$1"
    for _ in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    die "runtime did not become healthy on port ${port}"
}

restart_runtime_service() {
    systemctl daemon-reload
    systemctl restart "$RUNTIME_SERVICE"
    wait_for_runtime_health "$(runtime_port)"
}

restart_mistral_relay_service() {
    systemctl daemon-reload
    systemctl restart "$MISTRAL_RELAY_SERVICE"
}

stop_mistral_relay_service() {
    systemctl stop "$MISTRAL_RELAY_SERVICE" >/dev/null 2>&1 || true
}

set_standard_env() {
    env_set "VICUNA_HOST_INFERENCE_MODE" "standard"
    env_set "VICUNA_RUNPOD_INFERENCE_ROLE" "disabled"
}

set_experimental_env() {
    local runtime_url="$1"
    local auth_token="$2"
    local pod_id="${3:-}"
    local relay_timeout_ms
    relay_timeout_ms="$(resolved_runpod_relay_timeout_ms)"
    env_set "VICUNA_HOST_INFERENCE_MODE" "experimental"
    env_set "VICUNA_RUNPOD_INFERENCE_ROLE" "host"
    env_set "VICUNA_RUNPOD_INFERENCE_URL" "http://${MISTRAL_RELAY_HOST}:${MISTRAL_RELAY_PORT}"
    env_set "VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN" "$auth_token"
    env_set "RUNPOD_RUNTIME_AUTH_TOKEN" "$auth_token"
    env_set "VICUNA_RUNPOD_INFERENCE_TIMEOUT_MS" "$relay_timeout_ms"
    env_set "VICUNA_RUNPOD_MISTRAL_UPSTREAM_URL" "$runtime_url"
    env_set "VICUNA_RUNPOD_MISTRAL_RELAY_HOST" "$MISTRAL_RELAY_HOST"
    env_set "VICUNA_RUNPOD_MISTRAL_RELAY_PORT" "$MISTRAL_RELAY_PORT"
    env_set "RUNPOD_POD_NAME" "$(resolved_runpod_pod_name)"
    if [[ -n "$pod_id" ]]; then
        env_set "RUNPOD_ACTIVE_POD_ID" "$pod_id"
    fi
}

mode_status_json() {
    local previous_mode="$1"
    local current_mode_value="$2"
    local pod_action="$3"
    local pod_id="${4:-}"
    local tunnel_port="${5:-0}"
    local runtime_url="${6:-}"
    local public_ip="${7:-}"
    local external_port="${8:-}"
    python3 - "$previous_mode" "$current_mode_value" "$pod_action" "$pod_id" "$tunnel_port" "$RUNTIME_SERVICE" "$runtime_url" "$public_ip" "$external_port" <<'PY'
import json
import sys
print(json.dumps({
    "ok": True,
    "previous_mode": sys.argv[1],
    "current_mode": sys.argv[2],
    "pod_action": sys.argv[3],
    "pod_id": sys.argv[4] or None,
    "tunnel_local_port": int(sys.argv[5]),
    "runtime_service": sys.argv[6],
    "runtime_url": sys.argv[7] or None,
    "public_ip": sys.argv[8] or None,
    "external_port": int(sys.argv[9]) if sys.argv[9] else None,
}))
PY
}

set_experimental_mode() {
    load_env_file
    ensure_runpod_credentials
    local auth_token
    auth_token="$(ensure_runtime_auth_token)"
    local pod_id
    pod_id="$("$REPO_ROOT/tools/ops/runpod-ensure-pod.sh" | tail -n 1)"
    local endpoint_json
    endpoint_json="$("$REPO_ROOT/tools/ops/runpod-runtime-endpoint.sh" "$pod_id")"
    local runtime_url
    runtime_url="$(python3 - "$endpoint_json" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
print(data.get("url", ""))
PY
)"
    [[ -n "$runtime_url" ]] || die "RunPod direct runtime endpoint is unavailable for pod $pod_id"
    local public_ip
    public_ip="$(python3 - "$endpoint_json" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
print(data.get("public_ip", ""))
PY
)"
    local external_port
    external_port="$(python3 - "$endpoint_json" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
value = data.get("external_port")
print("" if value is None else value)
PY
)"
    set_experimental_env "$runtime_url" "$auth_token" "$pod_id"
    restart_mistral_relay_service
    restart_runtime_service
    mode_status_json "$1" "experimental" "started" "$pod_id" 0 "$runtime_url" "$public_ip" "$external_port"
}

set_standard_mode() {
    load_env_file
    set_standard_env
    stop_mistral_relay_service
    restart_runtime_service
    local pod_id=""
    pod_id="$("$REPO_ROOT/tools/ops/runpod-stop-pod.sh" 2>/dev/null | tail -n 1 || true)"
    mode_status_json "$1" "standard" "stopped" "$pod_id" 0 "" "" ""
}

COMMAND="${1:-}"
TARGET_MODE="${2:-}"
JSON_MODE=0
for arg in "$@"; do
    if [[ "$arg" == "--json" ]]; then
        JSON_MODE=1
    fi
done

require_root

case "$COMMAND" in
    status)
        previous="$(current_mode)"
        if (( JSON_MODE )); then
            mode_status_json "$previous" "$previous" "none" "" 0
        else
            printf '%s\n' "$previous"
        fi
        ;;
    toggle)
        previous="$(current_mode)"
        if [[ "$previous" == "experimental" ]]; then
            set_standard_mode "$previous"
        else
            set_experimental_mode "$previous"
        fi
        ;;
    set)
        [[ "$TARGET_MODE" == "standard" || "$TARGET_MODE" == "experimental" ]] || die "set requires standard or experimental"
        previous="$(current_mode)"
        if [[ "$TARGET_MODE" == "$previous" ]]; then
            if (( JSON_MODE )); then
                mode_status_json "$previous" "$previous" "none" "" 0
            else
                printf '%s\n' "$previous"
            fi
            exit 0
        fi
        if [[ "$TARGET_MODE" == "experimental" ]]; then
            set_experimental_mode "$previous"
        else
            set_standard_mode "$previous"
        fi
        ;;
    -h|--help|"")
        usage
        [[ -n "$COMMAND" ]] || exit 1
        ;;
    *)
        die "unknown command: $COMMAND"
        ;;
esac
