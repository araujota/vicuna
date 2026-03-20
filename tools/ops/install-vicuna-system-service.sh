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
CACHE_ROOT="${VICUNA_CACHE_ROOT:-/var/cache/vicuna}"
LOG_ROOT="${VICUNA_LOG_ROOT:-/var/log/vicuna}"
RUNTIME_STATE_PATH="${VICUNA_RUNTIME_STATE_PATH:-$STATE_ROOT/runtime-state.json}"
RUNTIME_BACKUP_DIR="${VICUNA_RUNTIME_STATE_BACKUP_DIR:-$STATE_ROOT/runtime-state-backups}"
OPENCLAW_CATALOG_PATH="${VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH:-$CACHE_ROOT/openclaw-catalog.json}"
TELEGRAM_STATE_PATH="${TELEGRAM_BRIDGE_STATE_PATH:-$STATE_ROOT/telegram-bridge-state.json}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: install-vicuna-system-service.sh [--dry-run]

Create or update a dedicated `vicuna` service account, grant it ACL-based
access to the repository worktree, install system-level runtime and Telegram
bridge units, and enable them through systemd.

Environment overrides:
  VICUNA_SERVICE_USER
  VICUNA_SERVICE_GROUP
  VICUNA_INTERACTIVE_OWNER
  VICUNA_SYSTEMD_DIR
  VICUNA_ETC_DIR
  VICUNA_SYSTEM_ENV_FILE
  VICUNA_STATE_ROOT
  VICUNA_CACHE_ROOT
  VICUNA_LOG_ROOT
  VICUNA_RUNTIME_STATE_PATH
  VICUNA_RUNTIME_STATE_BACKUP_DIR
  VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH
  TELEGRAM_BRIDGE_STATE_PATH
  TELEGRAM_BRIDGE_NODE_BIN
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
    if (( DRY_RUN )); then
        return 0
    fi
    if (( EUID != 0 )); then
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

    local owner_home
    owner_home="$(getent passwd "$INTERACTIVE_OWNER" | cut -d: -f6)"
    if [[ -n "$owner_home" && -d "$owner_home/.nvm/versions/node" ]]; then
        local candidate
        candidate="$(find "$owner_home/.nvm/versions/node" -mindepth 2 -maxdepth 2 -path '*/bin/node' | sort -V | tail -n 1)"
        if [[ -n "$candidate" && -x "$candidate" ]] && node_version_ge "$("$candidate" -v)"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    die "could not resolve Node.js >= 20.16; set TELEGRAM_BRIDGE_NODE_BIN"
}

resolve_codex_bin() {
    if [[ -n "${VICUNA_CODEX_TOOL_PATH:-}" && -x "${VICUNA_CODEX_TOOL_PATH:-}" ]]; then
        printf '%s\n' "$VICUNA_CODEX_TOOL_PATH"
        return 0
    fi

    if command -v codex >/dev/null 2>&1; then
        printf '%s\n' "$(command -v codex)"
        return 0
    fi

    local owner_home
    owner_home="$(getent passwd "$INTERACTIVE_OWNER" | cut -d: -f6)"
    if [[ -n "$owner_home" ]]; then
        local candidate=""
        candidate="$(runuser -u "$INTERACTIVE_OWNER" -- bash -lc 'command -v codex' 2>/dev/null || true)"
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi

        for candidate in \
            "$owner_home/.local/bin/codex" \
            "$owner_home/.npm-global/bin/codex" \
            "/usr/local/bin/codex"
        do
            if [[ -x "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    fi

    printf '\n'
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
    local codex_bin="$2"
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
VICUNA_RUNTIME_STATE_PATH=$RUNTIME_STATE_PATH
VICUNA_RUNTIME_STATE_BACKUP_DIR=$RUNTIME_BACKUP_DIR
VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=$OPENCLAW_CATALOG_PATH
VICUNA_BASH_TOOL_WORKDIR=$REPO_ROOT
VICUNA_CODEX_TOOL_PATH=$codex_bin
VICUNA_CODEX_TOOL_WORKDIR=$REPO_ROOT
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
VICUNA_RUNTIME_STATE_PATH=$RUNTIME_STATE_PATH
VICUNA_RUNTIME_STATE_BACKUP_DIR=$RUNTIME_BACKUP_DIR
VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=$OPENCLAW_CATALOG_PATH
VICUNA_BASH_TOOL_WORKDIR=$REPO_ROOT
VICUNA_CODEX_TOOL_PATH=$codex_bin
VICUNA_CODEX_TOOL_WORKDIR=$REPO_ROOT
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

sync_runtime_catalog() {
    local node_bin="$1"
    local codex_bin="$2"

    run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$(dirname "$OPENCLAW_CATALOG_PATH")"

    local -a sync_env=(
        env
        "REPO_ROOT=$REPO_ROOT"
        "VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=$OPENCLAW_CATALOG_PATH"
    )
    if [[ -n "$codex_bin" ]]; then
        sync_env+=("VICUNA_CODEX_TOOL_PATH=$codex_bin")
    fi

    run_cmd "${sync_env[@]}" "$node_bin" "$REPO_ROOT/tools/openclaw-harness/dist/index.js" sync-runtime-catalog
    if [[ -f "$OPENCLAW_CATALOG_PATH" ]]; then
        run_cmd chown "$SERVICE_USER:$SERVICE_GROUP" "$OPENCLAW_CATALOG_PATH"
    fi
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
CODEX_BIN="$(resolve_codex_bin)"

log "repo=$REPO_ROOT service_user=$SERVICE_USER interactive_owner=$INTERACTIVE_OWNER"
log "state_root=$STATE_ROOT cache_root=$CACHE_ROOT log_root=$LOG_ROOT node_bin=$NODE_BIN codex_bin=${CODEX_BIN:-<missing>}"

ensure_group
ensure_user
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$STATE_ROOT" "$CACHE_ROOT" "$LOG_ROOT"
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$(dirname "$RUNTIME_STATE_PATH")" "$RUNTIME_BACKUP_DIR" "$(dirname "$TELEGRAM_STATE_PATH")"
grant_repo_access
write_env_file "$NODE_BIN" "$CODEX_BIN"
sync_runtime_catalog "$NODE_BIN" "$CODEX_BIN"
install_unit "$REPO_ROOT/tools/ops/systemd/vicuna-runtime.system.service" "$SYSTEMD_DIR/vicuna-runtime.service"
install_unit "$REPO_ROOT/tools/ops/systemd/vicuna-telegram-bridge.system.service" "$SYSTEMD_DIR/vicuna-telegram-bridge.service"

disable_user_service_if_possible "vicuna-telegram-bridge.service"
disable_user_service_if_possible "vicuna-runtime.service"

run_cmd systemctl daemon-reload
run_cmd systemctl enable vicuna-runtime.service vicuna-telegram-bridge.service
run_cmd systemctl restart vicuna-runtime.service
run_cmd systemctl restart vicuna-telegram-bridge.service

log "system-service migration complete"
log "runtime unit: $SYSTEMD_DIR/vicuna-runtime.service"
log "bridge unit: $SYSTEMD_DIR/vicuna-telegram-bridge.service"
log "env file: $ENV_FILE"
