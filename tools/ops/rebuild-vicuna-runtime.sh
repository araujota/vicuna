#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

SYSTEM_ENV_FILE="${VICUNA_SYSTEM_ENV_FILE:-/etc/vicuna/vicuna.env}"
if [[ -z "${VICUNA_SYSTEMD_SCOPE:-}" && -r "$SYSTEM_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SYSTEM_ENV_FILE"
fi

SYSTEMD_SCOPE="${VICUNA_SYSTEMD_SCOPE:-user}"
if [[ "$SYSTEMD_SCOPE" == "system" && -r "$SYSTEM_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SYSTEM_ENV_FILE"
fi

RUNTIME_SERVICE="${VICUNA_RUNTIME_SERVICE_NAME:-vicuna-runtime.service}"
BUILD_DIR="build-host-cuda-128"
PORT="8080"
ALLOW_STATE_RESET=0
ALLOW_BUSY_STOP=0
DRY_RUN=0
SKIP_BACKUP=0

usage() {
    cat <<'EOF'
Usage: rebuild-vicuna-runtime.sh [options]

Safely rebuild and relaunch the Vicuña runtime while preserving runtime snapshot
state by default.

Options:
  --allow-state-reset   Allow rebuild even when persisted runtime state may not
                        be preserved. Use only for schema-breaking changes to
                        the persisted self-model / functional surfaces.
  --allow-busy-stop     Permit stop/restart while the runtime still reports
                        active or pending work.
  --skip-backup         Do not copy the runtime snapshot into the backup
                        directory before restart.
  --dry-run             Print the planned actions without stopping or building.
  -h, --help            Show this help text.
EOF
}

log() {
    printf '[vicuna-rebuild] %s\n' "$*"
}

die() {
    printf '[vicuna-rebuild] error: %s\n' "$*" >&2
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

health_json() {
    curl --silent --show-error "http://127.0.0.1:${PORT}/health"
}

json_expr() {
    local json_input="$1"
    local expression="$2"
    printf '%s' "$json_input" | node -e '
const fs = require("fs");
const data = JSON.parse(fs.readFileSync(0, "utf8"));
const expr = process.argv[1];
const value = expr.split(".").reduce((acc, key) => acc == null ? undefined : acc[key], data);
if (value === undefined) {
  process.exit(3);
}
if (typeof value === "object") {
  process.stdout.write(JSON.stringify(value));
} else {
  process.stdout.write(String(value));
}
' "$expression"
}

wait_for_health() {
    local attempts=60
    local delay=1
    local health=""
    for ((i = 0; i < attempts; ++i)); do
        if health="$(health_json 2>/dev/null)" && [[ -n "$health" ]]; then
            printf '%s' "$health"
            return 0
        fi
        sleep "$delay"
    done
    return 1
}

wait_for_persistence_ready() {
    local attempts=60
    local delay=1
    local health=""
    for ((i = 0; i < attempts; ++i)); do
        if health="$(health_json 2>/dev/null)" && [[ -n "$health" ]]; then
            local runtime_enabled runtime_healthy restored_path restore_attempted restore_success
            runtime_enabled="$(json_expr "$health" "runtime_persistence.enabled" || true)"
            runtime_healthy="$(json_expr "$health" "runtime_persistence.healthy" || true)"
            restored_path="$(json_expr "$health" "runtime_persistence.path" || true)"
            restore_attempted="$(json_expr "$health" "runtime_persistence.restore_attempted" || true)"
            restore_success="$(json_expr "$health" "runtime_persistence.restore_success" || true)"

            if [[ "$runtime_enabled" == "true" &&
                  "$runtime_healthy" == "true" &&
                  "$restored_path" == "$VICUNA_RUNTIME_STATE_PATH" ]]; then
                if (( had_snapshot )); then
                    if [[ "$restore_attempted" == "true" && "$restore_success" == "true" ]]; then
                        printf '%s' "$health"
                        return 0
                    fi
                else
                    printf '%s' "$health"
                    return 0
                fi
            fi
        fi
        sleep "$delay"
    done
    return 1
}

service_active() {
    systemctl_cmd is-active --quiet "$RUNTIME_SERVICE"
}

wait_for_service_stop() {
    local attempts=30
    for ((i = 0; i < attempts; ++i)); do
        if ! service_active; then
            return 0
        fi
        sleep 1
    done
    return 1
}

copy_backup() {
    local source_path="$1"
    local suffix="$2"
    [[ -f "$source_path" ]] || return 0
    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$VICUNA_RUNTIME_STATE_BACKUP_DIR"
    local dest_path="$VICUNA_RUNTIME_STATE_BACKUP_DIR/$(basename "$source_path").${timestamp}.${suffix}"
    cp "$source_path" "$dest_path"
    log "backed up $(basename "$source_path") -> $dest_path"
}

while (($# > 0)); do
    case "$1" in
        --allow-state-reset)
            ALLOW_STATE_RESET=1
            ;;
        --allow-busy-stop)
            ALLOW_BUSY_STOP=1
            ;;
        --skip-backup)
            SKIP_BACKUP=1
            ;;
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

log "repo=$REPO_ROOT service=$RUNTIME_SERVICE build_dir=$BUILD_DIR"
log "runtime_state_path=$VICUNA_RUNTIME_STATE_PATH"

if [[ -z "${VICUNA_RUNTIME_STATE_PATH:-}" ]] && (( ! ALLOW_STATE_RESET )); then
    die "VICUNA_RUNTIME_STATE_PATH must be configured unless --allow-state-reset is used"
fi

had_snapshot=0
if [[ -n "${VICUNA_RUNTIME_STATE_PATH:-}" && -f "$VICUNA_RUNTIME_STATE_PATH" ]]; then
    had_snapshot=1
fi

if service_active; then
    log "runtime service is active; inspecting health before rebuild"
    health="$(health_json)" || die "failed to query runtime health"

    runtime_enabled="$(json_expr "$health" "runtime_persistence.enabled" || true)"
    runtime_healthy="$(json_expr "$health" "runtime_persistence.healthy" || true)"
    waiting_tasks="$(json_expr "$health" "waiting_active_tasks" || true)"
    bash_pending="$(json_expr "$health" "external_bash_pending" || true)"
    memory_pending="$(json_expr "$health" "external_hard_memory_pending" || true)"

    if (( ! ALLOW_STATE_RESET )); then
        [[ "$runtime_enabled" == "true" ]] || die "runtime persistence is not enabled; refusing rebuild without --allow-state-reset"
        [[ "$runtime_healthy" == "true" ]] || die "runtime persistence is unhealthy; refusing rebuild"
    fi

    if (( ! ALLOW_BUSY_STOP )); then
        [[ "${waiting_tasks:-0}" == "0" ]] || die "runtime has waiting active tasks; use --allow-busy-stop to override"
        [[ "${bash_pending:-0}" == "0" ]] || die "runtime has pending bash work; use --allow-busy-stop to override"
        [[ "${memory_pending:-0}" == "0" ]] || die "runtime has pending hard-memory work; use --allow-busy-stop to override"
    fi
fi

run_cmd systemctl_cmd stop "$RUNTIME_SERVICE"
if (( ! DRY_RUN )); then
    wait_for_service_stop || die "runtime service did not stop cleanly"
fi

if (( ! SKIP_BACKUP )) && [[ -n "${VICUNA_RUNTIME_STATE_PATH:-}" ]]; then
    copy_backup "$VICUNA_RUNTIME_STATE_PATH" "pre-rebuild"
    copy_backup "${VICUNA_RUNTIME_STATE_PATH}.provenance.jsonl" "provenance"
fi

run_cmd cmake -S "$REPO_ROOT" -B "$REPO_ROOT/$BUILD_DIR" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
run_cmd cmake --build "$REPO_ROOT/$BUILD_DIR" --target llama-server -j 12

run_cmd systemctl_cmd reset-failed "$RUNTIME_SERVICE"
run_cmd systemctl_cmd start "$RUNTIME_SERVICE"

if (( DRY_RUN )); then
    exit 0
fi

if (( ALLOW_STATE_RESET )); then
    health="$(wait_for_health)" || die "runtime health endpoint did not recover after restart"
else
    health="$(wait_for_persistence_ready)" || die "runtime persistence did not become healthy after restart"
fi

runtime_enabled="$(json_expr "$health" "runtime_persistence.enabled" || true)"
runtime_healthy="$(json_expr "$health" "runtime_persistence.healthy" || true)"
restore_attempted="$(json_expr "$health" "runtime_persistence.restore_attempted" || true)"
restore_success="$(json_expr "$health" "runtime_persistence.restore_success" || true)"
restored_path="$(json_expr "$health" "runtime_persistence.path" || true)"

if (( ! ALLOW_STATE_RESET )); then
    [[ "$runtime_enabled" == "true" ]] || die "runtime came back without persistence enabled"
    [[ "$runtime_healthy" == "true" ]] || die "runtime came back with unhealthy persistence"
    [[ "$restored_path" == "$VICUNA_RUNTIME_STATE_PATH" ]] || die "runtime restored from unexpected snapshot path"
    if (( had_snapshot )); then
        [[ "$restore_attempted" == "true" ]] || die "runtime did not attempt restore from the preserved snapshot"
        [[ "$restore_success" == "true" ]] || die "runtime restore did not succeed"
    fi
fi

log "rebuild complete"
log "runtime persistence enabled=$runtime_enabled healthy=$runtime_healthy restore_attempted=$restore_attempted restore_success=$restore_success"
log "runtime health recovered on http://127.0.0.1:${PORT}/health"
