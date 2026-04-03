#!/usr/bin/env bash
set -euo pipefail

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$SCRIPT_REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

SYSTEM_ENV_FILE="${VICUNA_SYSTEM_ENV_FILE:-/etc/vicuna/vicuna.env}"
if [[ -r "$SYSTEM_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SYSTEM_ENV_FILE"
fi

CURRENT_REPO_ROOT="$SCRIPT_REPO_ROOT"
CONFIGURED_REPO_ROOT="${VICUNA_REPO_ROOT:-${REPO_ROOT:-$CURRENT_REPO_ROOT}}"
SYSTEMD_SCOPE="${VICUNA_SYSTEMD_SCOPE:-}"
RUNTIME_SERVICE="${VICUNA_RUNTIME_SERVICE_NAME:-vicuna-runtime.service}"
BRIDGE_SERVICE="${VICUNA_TELEGRAM_BRIDGE_SERVICE_NAME:-vicuna-telegram-bridge.service}"
WEBGL_RENDERER_SERVICE="${VICUNA_WEBGL_RENDERER_SERVICE_NAME:-vicuna-webgl-renderer.service}"
BUILD_DIR="${VICUNA_RUNTIME_BUILD_DIR:-build-host-cuda-128}"
PORT="${VICUNA_RUNTIME_PORT:-8080}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: rebuild-vicuna-runtime.sh [--dry-run]

Rebuild and restart the provider-first Vicuña runtime service.
EOF
}

log() {
    printf '[vicuna-rebuild] %s\n' "$*"
}

OPS_OPERATION_ID="rebuild_$(date +%s%3N)_$$"

log_event() {
    local event="$1"
    local message="$2"
    shift 2
    local fields=(
        --field "operation_id=\"$OPS_OPERATION_ID\""
        --field "target_surface=\"runtime\""
    )
    local field
    for field in "$@"; do
        fields+=(--field "$field")
    done
    python3 "$CONFIGURED_REPO_ROOT/tools/ops/service_log_router.py" event \
        --service ops \
        --event "$event" \
        --message "$message" \
        --log-root "${VICUNA_LOG_ROOT:-/var/log/vicuna}" \
        --retention-days "${VICUNA_LOG_RETENTION_DAYS:-7}" \
        "${fields[@]}" >/dev/null 2>&1 || true
}

die() {
    printf '[vicuna-rebuild] error: %s\n' "$*" >&2
    log_event "runtime_rebuild_failed" "$*" 'result="failed"'
    exit 1
}

run_cmd() {
    if (( DRY_RUN )); then
        printf '[vicuna-rebuild] dry-run:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

scope_unit_cat() {
    local scope="$1"
    local service_name="$2"
    if [[ "$scope" == "system" ]]; then
        systemctl cat "$service_name" 2>/dev/null
        return
    fi
    systemctl --user cat "$service_name" 2>/dev/null
}

scope_has_service() {
    local scope="$1"
    local service_name="$2"
    scope_unit_cat "$scope" "$service_name" >/dev/null
}

extract_unit_repo_root() {
    local scope="$1"
    local service_name="$2"
    local rendered
    rendered="$(scope_unit_cat "$scope" "$service_name" || true)"
    [[ -n "$rendered" ]] || return 0
    local working_directory
    working_directory="$(printf '%s\n' "$rendered" | awk -F= '/^WorkingDirectory=/ { print $2; exit }')"
    if [[ -n "$working_directory" ]]; then
        printf '%s\n' "$working_directory"
        return 0
    fi
    printf '%s\n' "$rendered" |
        sed -n 's|^ExecStart=\([^[:space:]]*\)/tools/ops/.*$|\1|p' |
        head -n 1
}

resolve_systemd_scope() {
    if [[ -n "$SYSTEMD_SCOPE" ]]; then
        case "$SYSTEMD_SCOPE" in
            system|user)
                printf '%s\n' "$SYSTEMD_SCOPE"
                return 0
                ;;
            *)
                die "unsupported VICUNA_SYSTEMD_SCOPE=$SYSTEMD_SCOPE"
                ;;
        esac
    fi

    local have_system=0
    local have_user=0
    scope_has_service system "$RUNTIME_SERVICE" && have_system=1
    scope_has_service user "$RUNTIME_SERVICE" && have_user=1

    if (( have_system && have_user )); then
        die "both system and user Vicuña runtime units are installed; run install-vicuna-system-service.sh to converge the host first"
    fi
    if (( have_system )); then
        printf 'system\n'
        return 0
    fi
    if (( have_user )); then
        printf 'user\n'
        return 0
    fi

    die "could not determine the installed Vicuña service scope; set VICUNA_SYSTEMD_SCOPE explicitly or install the system services first"
}

systemctl_cmd() {
    if [[ "$SYSTEMD_SCOPE" == "system" ]]; then
        if (( EUID == 0 )); then
            systemctl "$@"
        elif [[ ! -t 0 ]]; then
            sudo -S systemctl "$@"
        else
            sudo systemctl "$@"
        fi
        return
    fi
    systemctl --user "$@"
}

assert_repo_root_alignment() {
    if [[ "$CURRENT_REPO_ROOT" != "$CONFIGURED_REPO_ROOT" ]]; then
        die "this checkout is $CURRENT_REPO_ROOT but the configured live repo root is $CONFIGURED_REPO_ROOT; run rebuild from the configured root or reinstall the system services"
    fi

    local runtime_root bridge_root renderer_root
    runtime_root="$(extract_unit_repo_root "$SYSTEMD_SCOPE" "$RUNTIME_SERVICE")"
    bridge_root="$(extract_unit_repo_root "$SYSTEMD_SCOPE" "$BRIDGE_SERVICE")"
    renderer_root="$(extract_unit_repo_root "$SYSTEMD_SCOPE" "$WEBGL_RENDERER_SERVICE")"

    [[ -n "$runtime_root" ]] || die "could not resolve installed repo root for $RUNTIME_SERVICE"
    [[ "$runtime_root" == "$CONFIGURED_REPO_ROOT" ]] ||
        die "$RUNTIME_SERVICE is installed from $runtime_root but the configured repo root is $CONFIGURED_REPO_ROOT; run install-vicuna-system-service.sh to converge the host"

    if [[ -n "$bridge_root" && "$bridge_root" != "$CONFIGURED_REPO_ROOT" ]]; then
        die "$BRIDGE_SERVICE is installed from $bridge_root but the configured repo root is $CONFIGURED_REPO_ROOT; run install-vicuna-system-service.sh to converge the host"
    fi

    if [[ -n "$renderer_root" && "$renderer_root" != "$CONFIGURED_REPO_ROOT" ]]; then
        die "$WEBGL_RENDERER_SERVICE is installed from $renderer_root but the configured repo root is $CONFIGURED_REPO_ROOT; run install-vicuna-system-service.sh to converge the host"
    fi
}

assert_scope_convergence() {
    if [[ "$SYSTEMD_SCOPE" == "system" ]] &&
            (scope_has_service user "$RUNTIME_SERVICE" ||
             scope_has_service user "$BRIDGE_SERVICE" ||
             scope_has_service user "$WEBGL_RENDERER_SERVICE"); then
        die "stale user-scoped Vicuña units are still installed; run install-vicuna-system-service.sh to remove them before rebuilding the system deployment"
    fi
}

runtime_port_listeners() {
    ss -ltnp 2>/dev/null | awk -v port=":""$PORT" '$4 ~ (port "$") { print }'
}

ensure_port_released() {
    if (( DRY_RUN )); then
        return 0
    fi

    local attempts=10
    local listeners=""
    for ((i = 0; i < attempts; ++i)); do
        listeners="$(runtime_port_listeners || true)"
        if [[ -z "$listeners" ]]; then
            return 0
        fi
        sleep 1
    done

    die "runtime port $PORT is still owned after stopping $RUNTIME_SERVICE: $listeners"
}

wait_for_health() {
    local attempts=60
    for ((i = 0; i < attempts; ++i)); do
        if curl --silent --show-error "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

while (($# > 0)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
    shift
done

SYSTEMD_SCOPE="$(resolve_systemd_scope)"
assert_scope_convergence
assert_repo_root_alignment

log "repo=$CONFIGURED_REPO_ROOT scope=$SYSTEMD_SCOPE service=$RUNTIME_SERVICE build_dir=$BUILD_DIR"
log_event "runtime_rebuild_started" "runtime rebuild started" \
    "scope=\"$SYSTEMD_SCOPE\"" \
    "repo_root=\"$CONFIGURED_REPO_ROOT\"" \
    "build_dir=\"$BUILD_DIR\""
run_cmd systemctl_cmd stop "$RUNTIME_SERVICE"
ensure_port_released
run_cmd cmake -S "$CONFIGURED_REPO_ROOT" -B "$CONFIGURED_REPO_ROOT/$BUILD_DIR" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
run_cmd cmake --build "$CONFIGURED_REPO_ROOT/$BUILD_DIR" --target llama-server -j 12
run_cmd systemctl_cmd reset-failed "$RUNTIME_SERVICE"
run_cmd systemctl_cmd start "$RUNTIME_SERVICE"

if (( DRY_RUN )); then
    exit 0
fi

wait_for_health || die "runtime health endpoint did not recover after restart"

log "rebuild complete"
log_event "runtime_rebuild_finished" "runtime rebuild finished" 'result="succeeded"'
