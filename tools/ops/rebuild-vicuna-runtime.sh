#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

SYSTEM_ENV_FILE="${VICUNA_SYSTEM_ENV_FILE:-/etc/vicuna/vicuna.env}"
if [[ -r "$SYSTEM_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SYSTEM_ENV_FILE"
fi

SYSTEMD_SCOPE="${VICUNA_SYSTEMD_SCOPE:-user}"
RUNTIME_SERVICE="${VICUNA_RUNTIME_SERVICE_NAME:-vicuna-runtime.service}"
BRIDGE_SERVICE="${VICUNA_TELEGRAM_BRIDGE_SERVICE_NAME:-vicuna-telegram-bridge.service}"
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

run_cmd() {
    if (( DRY_RUN )); then
        printf '[vicuna-rebuild] dry-run:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

systemctl_cmd() {
    if [[ "$SYSTEMD_SCOPE" == "system" ]]; then
        if (( EUID == 0 )); then
            systemctl "$@"
        else
            sudo systemctl "$@"
        fi
        return
    fi
    systemctl --user "$@"
}

restart_bridge_if_present() {
    if (( DRY_RUN )); then
        run_cmd systemctl_cmd try-restart "$BRIDGE_SERVICE"
        return 0
    fi

    if systemctl_cmd status "$BRIDGE_SERVICE" >/dev/null 2>&1; then
        systemctl_cmd try-restart "$BRIDGE_SERVICE" || true
    fi
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
            printf '[vicuna-rebuild] error: unknown option: %s\n' "$1" >&2
            exit 1
            ;;
    esac
    shift
done

log "repo=$REPO_ROOT service=$RUNTIME_SERVICE build_dir=$BUILD_DIR"
run_cmd systemctl_cmd stop "$RUNTIME_SERVICE"
run_cmd cmake -S "$REPO_ROOT" -B "$REPO_ROOT/$BUILD_DIR" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
run_cmd cmake --build "$REPO_ROOT/$BUILD_DIR" --target llama-server -j 12
run_cmd systemctl_cmd reset-failed "$RUNTIME_SERVICE"
run_cmd systemctl_cmd start "$RUNTIME_SERVICE"
restart_bridge_if_present

if (( DRY_RUN )); then
    exit 0
fi

wait_for_health || {
    printf '[vicuna-rebuild] error: runtime health endpoint did not recover after restart\n' >&2
    exit 1
}

log "rebuild complete"
