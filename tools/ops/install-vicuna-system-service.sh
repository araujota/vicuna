#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SERVICE_USER="${VICUNA_SERVICE_USER:-vicuna}"
SERVICE_GROUP="${VICUNA_SERVICE_GROUP:-vicuna}"
INTERACTIVE_OWNER="${VICUNA_INTERACTIVE_OWNER:-$(stat -c %U "$REPO_ROOT")}"
SYSTEMD_DIR="${VICUNA_SYSTEMD_DIR:-/etc/systemd/system}"
ETC_DIR="${VICUNA_ETC_DIR:-/etc/vicuna}"
ENV_FILE="${VICUNA_SYSTEM_ENV_FILE:-$ETC_DIR/vicuna.env}"
STATE_ROOT="${VICUNA_STATE_ROOT:-/var/lib/vicuna}"
LOG_ROOT="${VICUNA_LOG_ROOT:-/var/log/vicuna}"
TELEGRAM_STATE_PATH="${TELEGRAM_BRIDGE_STATE_PATH:-$STATE_ROOT/telegram-bridge-state.json}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: install-vicuna-system-service.sh [--dry-run]

Install the provider-first Vicuña runtime and retained Telegram bridge as
systemd services.
EOF
}

log() {
    printf '[vicuna-system-install] %s\n' "$*"
}

die() {
    printf '[vicuna-system-install] error: %s\n' "$*" >&2
    exit 1
}

run_cmd() {
    if (( DRY_RUN )); then
        printf '[vicuna-system-install] dry-run:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

require_root() {
    if (( ! DRY_RUN && EUID != 0 )); then
        die "run this script as root"
    fi
}

node_version_ge() {
    local version="$1"
    version="${version#v}"
    local major minor patch
    IFS='.' read -r major minor patch <<<"$version"
    major="${major:-0}"
    minor="${minor:-0}"
    if (( major > 20 )); then
        return 0
    fi
    if (( major < 20 )); then
        return 1
    fi
    (( minor >= 16 ))
}

resolve_node_bin() {
    if [[ -n "${TELEGRAM_BRIDGE_NODE_BIN:-}" && -x "${TELEGRAM_BRIDGE_NODE_BIN:-}" ]]; then
        printf '%s\n' "$TELEGRAM_BRIDGE_NODE_BIN"
        return 0
    fi

    if command -v node >/dev/null 2>&1; then
        local current_node
        current_node="$(command -v node)"
        if node_version_ge "$("$current_node" -v)"; then
            printf '%s\n' "$current_node"
            return 0
        fi
    fi

    die "could not resolve Node.js >= 20.16; set TELEGRAM_BRIDGE_NODE_BIN"
}

ensure_group() {
    getent group "$SERVICE_GROUP" >/dev/null 2>&1 || run_cmd groupadd --system "$SERVICE_GROUP"
}

ensure_user() {
    if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
        run_cmd useradd \
            --system \
            --gid "$SERVICE_GROUP" \
            --home-dir "$STATE_ROOT" \
            --create-home \
            --shell /usr/sbin/nologin \
            "$SERVICE_USER"
    fi

    for extra_group in video render; do
        if getent group "$extra_group" >/dev/null 2>&1; then
            run_cmd usermod -a -G "$extra_group" "$SERVICE_USER"
        fi
    done
}

grant_parent_access() {
    local target="$1"
    while [[ "$target" != "/" ]]; do
        run_cmd setfacl -m "u:${SERVICE_USER}:rx" "$target"
        target="$(dirname "$target")"
    done
}

grant_repo_access() {
    grant_parent_access "$REPO_ROOT"
    run_cmd setfacl -R -m "u:${SERVICE_USER}:rwx" "$REPO_ROOT"
    run_cmd bash -lc "find '$REPO_ROOT' -type d -print0 | xargs -0 -r setfacl -m 'd:u:${SERVICE_USER}:rwx'"
}

write_env_file() {
    local node_bin="$1"
    run_cmd install -d -m 0755 "$ETC_DIR"
    if (( DRY_RUN )); then
        cat <<EOF
VICUNA_SYSTEMD_SCOPE=system
VICUNA_SYSTEM_ENV_FILE=$ENV_FILE
VICUNA_SERVICE_USER=$SERVICE_USER
VICUNA_SERVICE_GROUP=$SERVICE_GROUP
VICUNA_RUNTIME_SERVICE_NAME=vicuna-runtime.service
VICUNA_REPO_ROOT=$REPO_ROOT
REPO_ROOT=$REPO_ROOT
VICUNA_RUNTIME_BUILD_DIR=${VICUNA_RUNTIME_BUILD_DIR:-build-host-cuda-128}
VICUNA_RUNTIME_PORT=${VICUNA_RUNTIME_PORT:-8080}
VICUNA_DEEPSEEK_BASE_URL=${VICUNA_DEEPSEEK_BASE_URL:-https://api.deepseek.com}
VICUNA_DEEPSEEK_MODEL=${VICUNA_DEEPSEEK_MODEL:-deepseek-reasoner}
VICUNA_DEEPSEEK_TIMEOUT_MS=${VICUNA_DEEPSEEK_TIMEOUT_MS:-60000}
VICUNA_DEEPSEEK_API_KEY=${VICUNA_DEEPSEEK_API_KEY:-}
TELEGRAM_BRIDGE_STATE_PATH=$TELEGRAM_STATE_PATH
TELEGRAM_BRIDGE_NODE_BIN=$node_bin
EOF
        return 0
    fi

    cat >"$ENV_FILE" <<EOF
VICUNA_SYSTEMD_SCOPE=system
VICUNA_SYSTEM_ENV_FILE=$ENV_FILE
VICUNA_SERVICE_USER=$SERVICE_USER
VICUNA_SERVICE_GROUP=$SERVICE_GROUP
VICUNA_RUNTIME_SERVICE_NAME=vicuna-runtime.service
VICUNA_REPO_ROOT=$REPO_ROOT
REPO_ROOT=$REPO_ROOT
VICUNA_RUNTIME_BUILD_DIR=${VICUNA_RUNTIME_BUILD_DIR:-build-host-cuda-128}
VICUNA_RUNTIME_PORT=${VICUNA_RUNTIME_PORT:-8080}
VICUNA_DEEPSEEK_BASE_URL=${VICUNA_DEEPSEEK_BASE_URL:-https://api.deepseek.com}
VICUNA_DEEPSEEK_MODEL=${VICUNA_DEEPSEEK_MODEL:-deepseek-reasoner}
VICUNA_DEEPSEEK_TIMEOUT_MS=${VICUNA_DEEPSEEK_TIMEOUT_MS:-60000}
VICUNA_DEEPSEEK_API_KEY=${VICUNA_DEEPSEEK_API_KEY:-}
TELEGRAM_BRIDGE_STATE_PATH=$TELEGRAM_STATE_PATH
TELEGRAM_BRIDGE_NODE_BIN=$node_bin
EOF
    chmod 0644 "$ENV_FILE"
}

install_unit() {
    local template="$1"
    local destination="$2"
    if (( DRY_RUN )); then
        sed \
            -e "s|@VICUNA_REPO_ROOT@|$REPO_ROOT|g" \
            -e "s|@VICUNA_SERVICE_USER@|$SERVICE_USER|g" \
            -e "s|@VICUNA_SERVICE_GROUP@|$SERVICE_GROUP|g" \
            -e "s|@VICUNA_ENV_FILE@|$ENV_FILE|g" \
            "$template"
        return 0
    fi

    sed \
        -e "s|@VICUNA_REPO_ROOT@|$REPO_ROOT|g" \
        -e "s|@VICUNA_SERVICE_USER@|$SERVICE_USER|g" \
        -e "s|@VICUNA_SERVICE_GROUP@|$SERVICE_GROUP|g" \
        -e "s|@VICUNA_ENV_FILE@|$ENV_FILE|g" \
        "$template" >"$destination"
    chmod 0644 "$destination"
}

disable_user_service_if_possible() {
    local service_name="$1"
    local owner_uid
    owner_uid="$(id -u "$INTERACTIVE_OWNER" 2>/dev/null || true)"
    [[ -n "$owner_uid" ]] || return 0
    [[ -S "/run/user/$owner_uid/bus" ]] || return 0
    if (( DRY_RUN )); then
        log "would disable user service $service_name for $INTERACTIVE_OWNER"
        return 0
    fi
    runuser -u "$INTERACTIVE_OWNER" -- env \
        XDG_RUNTIME_DIR="/run/user/$owner_uid" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$owner_uid/bus" \
        systemctl --user disable --now "$service_name" >/dev/null 2>&1 || true
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

require_root

NODE_BIN="$(resolve_node_bin)"

log "repo=$REPO_ROOT service_user=$SERVICE_USER interactive_owner=$INTERACTIVE_OWNER"
ensure_group
ensure_user
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$STATE_ROOT" "$LOG_ROOT"
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$(dirname "$TELEGRAM_STATE_PATH")"
grant_repo_access
write_env_file "$NODE_BIN"
install_unit "$REPO_ROOT/tools/ops/systemd/vicuna-runtime.system.service" "$SYSTEMD_DIR/vicuna-runtime.service"
install_unit "$REPO_ROOT/tools/ops/systemd/vicuna-telegram-bridge.system.service" "$SYSTEMD_DIR/vicuna-telegram-bridge.service"
disable_user_service_if_possible "vicuna-telegram-bridge.service"
disable_user_service_if_possible "vicuna-runtime.service"
run_cmd systemctl daemon-reload
run_cmd systemctl enable vicuna-runtime.service vicuna-telegram-bridge.service
run_cmd systemctl restart vicuna-runtime.service
run_cmd systemctl restart vicuna-telegram-bridge.service

log "system-service install complete"
log "runtime unit: $SYSTEMD_DIR/vicuna-runtime.service"
log "bridge unit: $SYSTEMD_DIR/vicuna-telegram-bridge.service"
log "env file: $ENV_FILE"
