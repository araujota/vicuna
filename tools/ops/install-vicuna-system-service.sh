#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SERVICE_USER="${VICUNA_SERVICE_USER:-vicuna}"
SERVICE_GROUP="${VICUNA_SERVICE_GROUP:-vicuna}"
INTERACTIVE_OWNER="${VICUNA_INTERACTIVE_OWNER:-}"
INTERACTIVE_HOME="${VICUNA_INTERACTIVE_HOME:-}"
INTERACTIVE_HOME_ROOTS="${VICUNA_INTERACTIVE_HOME_ROOTS:-/home /root}"
SYSTEMD_DIR="${VICUNA_SYSTEMD_DIR:-/etc/systemd/system}"
ETC_DIR="${VICUNA_ETC_DIR:-/etc/vicuna}"
ENV_FILE="${VICUNA_SYSTEM_ENV_FILE:-$ETC_DIR/vicuna.env}"
STATE_ROOT="${VICUNA_STATE_ROOT:-/var/lib/vicuna}"
LOG_ROOT="${VICUNA_LOG_ROOT:-/var/log/vicuna}"
HOST_SHELL_ROOT="${VICUNA_HOST_SHELL_ROOT:-/home/vicuna/home}"
SYSTEM_HOST_SHELL_ROOT="${VICUNA_SYSTEM_HOST_SHELL_ROOT:-$HOST_SHELL_ROOT}"
TELEGRAM_STATE_PATH="${VICUNA_SYSTEM_TELEGRAM_STATE_PATH:-$STATE_ROOT/telegram-bridge-state.json}"
OPENCLAW_SECRETS_PATH="${VICUNA_SYSTEM_OPENCLAW_TOOL_FABRIC_SECRETS_PATH:-$STATE_ROOT/openclaw-tool-secrets.json}"
OPENCLAW_CATALOG_PATH="${VICUNA_SYSTEM_OPENCLAW_TOOL_FABRIC_CATALOG_PATH:-$STATE_ROOT/openclaw-catalog.json}"
HARD_MEMORY_DIR="${VICUNA_HARD_MEMORY_DIR:-$SYSTEM_HOST_SHELL_ROOT/memories}"
ONGOING_TASKS_DIR="${VICUNA_ONGOING_TASKS_DIR:-$STATE_ROOT/ongoing-tasks}"
ONGOING_TASKS_TMPDIR="${VICUNA_ONGOING_TASKS_TMPDIR:-$ONGOING_TASKS_DIR/tmp}"
SKILLS_DIR="${VICUNA_SKILLS_DIR:-$SYSTEM_HOST_SHELL_ROOT/skills}"
DOCS_DIR="${VICUNA_DOCS_DIR:-$SYSTEM_HOST_SHELL_ROOT/docs}"
HEURISTICS_DIR="${VICUNA_HEURISTICS_DIR:-$SYSTEM_HOST_SHELL_ROOT/heuristics}"
HEURISTIC_MEMORY_PATH="${VICUNA_HEURISTIC_MEMORY_PATH:-$HEURISTICS_DIR/vicuna-heuristic-memory.json}"
POLICY_DATASET_DIR="${VICUNA_POLICY_DATASET_DIR:-$STATE_ROOT/policy-datasets/nightly}"
POLICY_REGISTRY_DIR="${VICUNA_POLICY_REGISTRY_DIR:-$STATE_ROOT/policy-registry}"
POLICY_RUN_ROOT="${VICUNA_POLICY_RUN_ROOT:-$STATE_ROOT/policy-runs}"
RUNTIME_SERVICE="${VICUNA_RUNTIME_SERVICE_NAME:-vicuna-runtime.service}"
BRIDGE_SERVICE="${VICUNA_TELEGRAM_BRIDGE_SERVICE_NAME:-vicuna-telegram-bridge.service}"
WEBGL_RENDERER_SERVICE="${VICUNA_WEBGL_RENDERER_SERVICE_NAME:-vicuna-webgl-renderer.service}"
RUNTIME_PORT="${VICUNA_RUNTIME_PORT:-8080}"
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

path_owner_name() {
    local target="$1"
    local owner=""
    owner="$(stat -c %U "$target" 2>/dev/null || true)"
    if [[ -n "$owner" ]]; then
        printf '%s\n' "$owner"
        return 0
    fi

    owner="$(stat -f %Su "$target" 2>/dev/null || true)"
    if [[ -n "$owner" ]]; then
        printf '%s\n' "$owner"
        return 0
    fi

    printf 'UNKNOWN\n'
}

require_root() {
    if (( ! DRY_RUN && EUID != 0 )); then
        die "run this script as root"
    fi
}

resolve_interactive_home() {
    if [[ -n "$INTERACTIVE_OWNER" ]]; then
        local passwd_line
        passwd_line="$(getent passwd "$INTERACTIVE_OWNER" 2>/dev/null || true)"
        if [[ -n "$passwd_line" ]]; then
            printf '%s\n' "${passwd_line##*:}"
            return 0
        fi
    fi

    if [[ -n "$INTERACTIVE_HOME" ]]; then
        printf '%s\n' "$INTERACTIVE_HOME"
        return 0
    fi

    if [[ -n "$INTERACTIVE_OWNER" && "$INTERACTIVE_OWNER" != "UNKNOWN" ]]; then
        printf '/home/%s\n' "$INTERACTIVE_OWNER"
        return 0
    fi

    printf '\n'
}

resolve_interactive_uid() {
    id -u "$INTERACTIVE_OWNER" 2>/dev/null || true
}

interactive_user_systemd_bus() {
    local owner_uid="$1"
    [[ -n "$owner_uid" ]] || return 1
    [[ -S "/run/user/$owner_uid/bus" ]]
}

interactive_user_unit_path() {
    local home_dir="$1"
    local service_name="$2"
    printf '%s/.config/systemd/user/%s\n' "$home_dir" "$service_name"
}

interactive_user_wants_path() {
    local home_dir="$1"
    local service_name="$2"
    printf '%s/.config/systemd/user/default.target.wants/%s\n' "$home_dir" "$service_name"
}

user_home_has_vicuna_units() {
    local home_dir="$1"
    local service_name
    for service_name in "$RUNTIME_SERVICE" "$BRIDGE_SERVICE" "$WEBGL_RENDERER_SERVICE"; do
        if [[ -f "$(interactive_user_unit_path "$home_dir" "$service_name")" ]]; then
            return 0
        fi
    done
    return 1
}

user_home_owner() {
    local home_dir="$1"
    if [[ "$home_dir" == "/root" ]]; then
        printf 'root\n'
        return 0
    fi
    basename "$home_dir"
}

declare -a USER_SERVICE_OWNERS=()
declare -a USER_SERVICE_HOMES=()

register_user_service_home() {
    local home_dir="$1"
    local owner="${2:-}"

    local existing_home
    for existing_home in "${USER_SERVICE_HOMES[@]:-}"; do
        if [[ "$existing_home" == "$home_dir" ]]; then
            return 0
        fi
    done

    if [[ -z "$owner" ]]; then
        owner="$(user_home_owner "$home_dir")"
    fi
    [[ -n "$owner" ]] || return 0

    USER_SERVICE_HOMES+=("$home_dir")
    USER_SERVICE_OWNERS+=("$owner")
}

collect_user_service_candidates() {
    USER_SERVICE_OWNERS=()
    USER_SERVICE_HOMES=()

    if [[ -n "$INTERACTIVE_HOME" ]]; then
        register_user_service_home "$INTERACTIVE_HOME" "$INTERACTIVE_OWNER"
    fi

    if [[ -n "$INTERACTIVE_OWNER" && "$INTERACTIVE_OWNER" != "UNKNOWN" ]]; then
        local owner_home
        owner_home="$(resolve_interactive_home)"
        if [[ -n "$owner_home" ]]; then
            register_user_service_home "$owner_home" "$INTERACTIVE_OWNER"
        fi
    fi

    local home_root home_dir
    for home_root in $INTERACTIVE_HOME_ROOTS; do
        if [[ "$home_root" == "/root" ]]; then
            if user_home_has_vicuna_units "/root"; then
                register_user_service_home "/root" "root"
            fi
            continue
        fi

        [[ -d "$home_root" ]] || continue
        for home_dir in "$home_root"/*; do
            [[ -d "$home_dir" ]] || continue
            if user_home_has_vicuna_units "$home_dir"; then
                register_user_service_home "$home_dir" ""
            fi
        done
    done
}

system_unit_path() {
    local service_name="$1"
    printf '%s/%s\n' "$SYSTEMD_DIR" "$service_name"
}

system_override_dir() {
    local service_name="$1"
    printf '%s/%s.d\n' "$SYSTEMD_DIR" "$service_name"
}

existing_env_value() {
    local key="$1"
    [[ -r "$ENV_FILE" ]] || return 1
    python3 - "$ENV_FILE" "$key" <<'PY'
import pathlib
import sys

env_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]

try:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            print(line.split("=", 1)[1])
            sys.exit(0)
except FileNotFoundError:
    pass

sys.exit(1)
PY
}

resolved_env_value() {
    local key="$1"
    local default_value="${2-}"
    local current_value="${!key-}"
    if [[ -n "$current_value" ]]; then
        printf '%s\n' "$current_value"
        return 0
    fi

    local existing_value=""
    existing_value="$(existing_env_value "$key" || true)"
    if [[ -n "$existing_value" ]]; then
        printf '%s\n' "$existing_value"
        return 0
    fi

    printf '%s\n' "$default_value"
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

    local existing_node_bin=""
    existing_node_bin="$(existing_env_value "TELEGRAM_BRIDGE_NODE_BIN" || true)"
    if [[ -n "$existing_node_bin" && -x "$existing_node_bin" ]] && node_version_ge "$("$existing_node_bin" -v)"; then
        printf '%s\n' "$existing_node_bin"
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

resolve_ffmpeg_bin() {
    if [[ -n "${TELEGRAM_BRIDGE_FFMPEG_BIN:-}" && -x "${TELEGRAM_BRIDGE_FFMPEG_BIN:-}" ]]; then
        printf '%s\n' "$TELEGRAM_BRIDGE_FFMPEG_BIN"
        return 0
    fi

    local existing_ffmpeg_bin=""
    existing_ffmpeg_bin="$(existing_env_value "TELEGRAM_BRIDGE_FFMPEG_BIN" || true)"
    if [[ -n "$existing_ffmpeg_bin" && -x "$existing_ffmpeg_bin" ]]; then
        printf '%s\n' "$existing_ffmpeg_bin"
        return 0
    fi

    local candidate
    for candidate in /usr/bin/ffmpeg /opt/homebrew/bin/ffmpeg; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if command -v ffmpeg >/dev/null 2>&1; then
        command -v ffmpeg
        return 0
    fi

    die "could not resolve ffmpeg; set TELEGRAM_BRIDGE_FFMPEG_BIN"
}

resolve_chromium_bin() {
    if [[ -n "${VICUNA_WEBGL_RENDERER_CHROMIUM_BIN:-}" && -x "${VICUNA_WEBGL_RENDERER_CHROMIUM_BIN:-}" ]]; then
        printf '%s\n' "$VICUNA_WEBGL_RENDERER_CHROMIUM_BIN"
        return 0
    fi

    local existing_chromium_bin=""
    existing_chromium_bin="$(existing_env_value "VICUNA_WEBGL_RENDERER_CHROMIUM_BIN" || true)"
    if [[ -n "$existing_chromium_bin" && -x "$existing_chromium_bin" ]]; then
        printf '%s\n' "$existing_chromium_bin"
        return 0
    fi

    local candidate
    for candidate in /snap/bin/chromium /usr/bin/chromium /usr/bin/chromium-browser /usr/bin/google-chrome-stable; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    die "could not resolve Chromium; set VICUNA_WEBGL_RENDERER_CHROMIUM_BIN"
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

ensure_service_writable_file() {
    local target_path="$1"
    local mode="$2"
    if (( DRY_RUN )); then
        run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$(dirname "$target_path")"
        run_cmd install -m "$mode" -o "$SERVICE_USER" -g "$SERVICE_GROUP" /dev/null "$target_path"
        return 0
    fi

    install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$(dirname "$target_path")"
    if [[ -e "$target_path" ]]; then
        chown "$SERVICE_USER:$SERVICE_GROUP" "$target_path"
        chmod "$mode" "$target_path"
        return 0
    fi
    install -m "$mode" -o "$SERVICE_USER" -g "$SERVICE_GROUP" /dev/null "$target_path"
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
    local ffmpeg_bin="$2"
    local chromium_bin="$3"
    local runtime_build_dir runtime_port deepseek_base_url deepseek_model deepseek_timeout_ms
    local deepseek_api_key telegram_bot_token telegram_bridge_node_bin telegram_bridge_ffmpeg_bin
    local telegram_bridge_video_encoder telegram_bridge_render_backend webgl_renderer_url webgl_renderer_host
    local webgl_renderer_port webgl_renderer_ffmpeg_bin webgl_renderer_video_encoder
    local webgl_renderer_max_concurrent_renders webgl_renderer_gpu_memory_budget_mb webgl_renderer_mandatory_gpu
    local ongoing_tasks_runner_script ongoing_tasks_runtime_url ongoing_tasks_runtime_model
    local openclaw_node_bin tavily_api_key radarr_api_key radarr_base_url sonarr_api_key sonarr_base_url
    local chaptarr_api_key chaptarr_base_url vicuna_api_key
    local policy_mode policy_candidate_url policy_timeout_ms policy_dataset_dir policy_registry_dir
    local policy_run_root policy_model_name policy_server_url policy_registry_host policy_registry_port
    local policy_default_alias policy_fallback_alias policy_limit policy_min_record_count
    local policy_min_exact_match_rate policy_max_invalid_action_rate policy_min_reward_delta

    runtime_build_dir="$(resolved_env_value "VICUNA_RUNTIME_BUILD_DIR" "build-host-cuda-128")"
    runtime_port="$(resolved_env_value "VICUNA_RUNTIME_PORT" "8080")"
    deepseek_base_url="$(resolved_env_value "VICUNA_DEEPSEEK_BASE_URL" "https://api.deepseek.com")"
    deepseek_model="$(resolved_env_value "VICUNA_DEEPSEEK_MODEL" "deepseek-chat")"
    deepseek_timeout_ms="$(resolved_env_value "VICUNA_DEEPSEEK_TIMEOUT_MS" "60000")"
    deepseek_api_key="$(resolved_env_value "VICUNA_DEEPSEEK_API_KEY")"
    telegram_bot_token="$(resolved_env_value "TELEGRAM_BOT_TOKEN")"
    telegram_bridge_node_bin="$(resolved_env_value "TELEGRAM_BRIDGE_NODE_BIN" "$node_bin")"
    telegram_bridge_ffmpeg_bin="$(resolved_env_value "TELEGRAM_BRIDGE_FFMPEG_BIN" "$ffmpeg_bin")"
    telegram_bridge_video_encoder="$(resolved_env_value "TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER" "h264_nvenc")"
    telegram_bridge_render_backend="$(resolved_env_value "TELEGRAM_BRIDGE_RENDER_BACKEND" "chromium_webgl")"
    webgl_renderer_url="$(resolved_env_value "VICUNA_WEBGL_RENDERER_URL" "http://127.0.0.1:8091")"
    webgl_renderer_host="$(resolved_env_value "VICUNA_WEBGL_RENDERER_HOST" "127.0.0.1")"
    webgl_renderer_port="$(resolved_env_value "VICUNA_WEBGL_RENDERER_PORT" "8091")"
    webgl_renderer_ffmpeg_bin="$(resolved_env_value "VICUNA_WEBGL_RENDERER_FFMPEG_BIN" "$ffmpeg_bin")"
    webgl_renderer_video_encoder="$(resolved_env_value "VICUNA_WEBGL_RENDERER_FFMPEG_VIDEO_ENCODER" "$telegram_bridge_video_encoder")"
    webgl_renderer_max_concurrent_renders="$(resolved_env_value "VICUNA_WEBGL_RENDERER_MAX_CONCURRENT_RENDERS" "1")"
    webgl_renderer_gpu_memory_budget_mb="$(resolved_env_value "VICUNA_WEBGL_RENDERER_GPU_MEMORY_BUDGET_MB" "1024")"
    webgl_renderer_mandatory_gpu="$(resolved_env_value "VICUNA_WEBGL_RENDERER_MANDATORY_GPU" "1")"
    ongoing_tasks_runner_script="$(resolved_env_value "VICUNA_ONGOING_TASKS_RUNNER_SCRIPT" "$REPO_ROOT/tools/ops/run-ongoing-task-cron.sh")"
    ongoing_tasks_runtime_url="$(resolved_env_value "VICUNA_ONGOING_TASKS_RUNTIME_URL" "http://127.0.0.1:${runtime_port}/v1/chat/completions")"
    ongoing_tasks_runtime_model="$(resolved_env_value "VICUNA_ONGOING_TASKS_RUNTIME_MODEL" "$deepseek_model")"
    openclaw_node_bin="$(resolved_env_value "VICUNA_OPENCLAW_NODE_BIN" "$telegram_bridge_node_bin")"
    tavily_api_key="$(resolved_env_value "TAVILY_API_KEY")"
    radarr_api_key="$(resolved_env_value "RADARR_API_KEY")"
    radarr_base_url="$(resolved_env_value "RADARR_BASE_URL" "http://10.0.0.218:7878")"
    sonarr_api_key="$(resolved_env_value "SONARR_API_KEY")"
    sonarr_base_url="$(resolved_env_value "SONARR_BASE_URL" "http://10.0.0.218:8989")"
    chaptarr_api_key="$(resolved_env_value "CHAPTARR_API_KEY")"
    chaptarr_base_url="$(resolved_env_value "CHAPTARR_BASE_URL" "http://10.0.0.218:8789")"
    vicuna_api_key="$(resolved_env_value "VICUNA_API_KEY")"
    policy_mode="$(resolved_env_value "VICUNA_POLICY_MODE")"
    policy_candidate_url="$(resolved_env_value "VICUNA_POLICY_CANDIDATE_URL")"
    policy_timeout_ms="$(resolved_env_value "VICUNA_POLICY_TIMEOUT_MS" "500")"
    policy_dataset_dir="$(resolved_env_value "VICUNA_POLICY_DATASET_DIR" "$POLICY_DATASET_DIR")"
    policy_registry_dir="$(resolved_env_value "VICUNA_POLICY_REGISTRY_DIR" "$POLICY_REGISTRY_DIR")"
    policy_run_root="$(resolved_env_value "VICUNA_POLICY_RUN_ROOT" "$POLICY_RUN_ROOT")"
    policy_model_name="$(resolved_env_value "VICUNA_POLICY_MODEL_NAME" "vicuna-governance")"
    policy_server_url="$(resolved_env_value "VICUNA_POLICY_SERVER_URL" "http://127.0.0.1:${runtime_port}")"
    policy_registry_host="$(resolved_env_value "VICUNA_POLICY_REGISTRY_HOST" "127.0.0.1")"
    policy_registry_port="$(resolved_env_value "VICUNA_POLICY_REGISTRY_PORT" "18081")"
    policy_default_alias="$(resolved_env_value "VICUNA_POLICY_DEFAULT_ALIAS" "candidate")"
    policy_fallback_alias="$(resolved_env_value "VICUNA_POLICY_FALLBACK_ALIAS" "champion")"
    policy_limit="$(resolved_env_value "VICUNA_POLICY_LIMIT" "512")"
    policy_min_record_count="$(resolved_env_value "VICUNA_POLICY_MIN_RECORD_COUNT" "25")"
    policy_min_exact_match_rate="$(resolved_env_value "VICUNA_POLICY_MIN_EXACT_MATCH_RATE" "0.55")"
    policy_max_invalid_action_rate="$(resolved_env_value "VICUNA_POLICY_MAX_INVALID_ACTION_RATE" "0.0")"
    policy_min_reward_delta="$(resolved_env_value "VICUNA_POLICY_MIN_REWARD_DELTA" "0.0")"

    run_cmd install -d -m 0755 "$ETC_DIR"
    if (( DRY_RUN )); then
        cat <<EOF
VICUNA_SYSTEMD_SCOPE=system
VICUNA_SYSTEM_ENV_FILE=$ENV_FILE
VICUNA_SERVICE_USER=$SERVICE_USER
VICUNA_SERVICE_GROUP=$SERVICE_GROUP
VICUNA_RUNTIME_SERVICE_NAME=$RUNTIME_SERVICE
VICUNA_TELEGRAM_BRIDGE_SERVICE_NAME=$BRIDGE_SERVICE
VICUNA_WEBGL_RENDERER_SERVICE_NAME=$WEBGL_RENDERER_SERVICE
VICUNA_REPO_ROOT=$REPO_ROOT
REPO_ROOT=$REPO_ROOT
VICUNA_RUNTIME_BUILD_DIR=$runtime_build_dir
VICUNA_RUNTIME_PORT=$runtime_port
VICUNA_DEEPSEEK_BASE_URL=$deepseek_base_url
VICUNA_DEEPSEEK_MODEL=$deepseek_model
VICUNA_DEEPSEEK_TIMEOUT_MS=$deepseek_timeout_ms
VICUNA_DEEPSEEK_API_KEY=$deepseek_api_key
TELEGRAM_BOT_TOKEN=$telegram_bot_token
TELEGRAM_BRIDGE_STATE_PATH=$TELEGRAM_STATE_PATH
TELEGRAM_BRIDGE_NODE_BIN=$telegram_bridge_node_bin
TELEGRAM_BRIDGE_FFMPEG_BIN=$telegram_bridge_ffmpeg_bin
TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER=$telegram_bridge_video_encoder
TELEGRAM_BRIDGE_RENDER_BACKEND=$telegram_bridge_render_backend
VICUNA_WEBGL_RENDERER_URL=$webgl_renderer_url
VICUNA_WEBGL_RENDERER_HOST=$webgl_renderer_host
VICUNA_WEBGL_RENDERER_PORT=$webgl_renderer_port
VICUNA_WEBGL_RENDERER_CHROMIUM_BIN=$chromium_bin
VICUNA_WEBGL_RENDERER_FFMPEG_BIN=$webgl_renderer_ffmpeg_bin
VICUNA_WEBGL_RENDERER_FFMPEG_VIDEO_ENCODER=$webgl_renderer_video_encoder
VICUNA_WEBGL_RENDERER_MAX_CONCURRENT_RENDERS=$webgl_renderer_max_concurrent_renders
VICUNA_WEBGL_RENDERER_GPU_MEMORY_BUDGET_MB=$webgl_renderer_gpu_memory_budget_mb
VICUNA_WEBGL_RENDERER_MANDATORY_GPU=$webgl_renderer_mandatory_gpu
VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH=$OPENCLAW_SECRETS_PATH
VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=$OPENCLAW_CATALOG_PATH
VICUNA_STATE_ROOT=$STATE_ROOT
VICUNA_HARD_MEMORY_DIR=$HARD_MEMORY_DIR
VICUNA_SKILLS_DIR=$SKILLS_DIR
VICUNA_ONGOING_TASKS_DIR=$ONGOING_TASKS_DIR
VICUNA_ONGOING_TASKS_TMPDIR=$ONGOING_TASKS_TMPDIR
VICUNA_ONGOING_TASKS_RUNNER_SCRIPT=$ongoing_tasks_runner_script
VICUNA_ONGOING_TASKS_RUNTIME_URL=$ongoing_tasks_runtime_url
VICUNA_ONGOING_TASKS_RUNTIME_MODEL=$ongoing_tasks_runtime_model
VICUNA_OPENCLAW_NODE_BIN=$openclaw_node_bin
VICUNA_HOST_SHELL_ROOT=$SYSTEM_HOST_SHELL_ROOT
VICUNA_SKILLS_DIR=$SKILLS_DIR
VICUNA_DOCS_DIR=$DOCS_DIR
VICUNA_HEURISTIC_MEMORY_PATH=$HEURISTIC_MEMORY_PATH
VICUNA_POLICY_MODE=$policy_mode
VICUNA_POLICY_CANDIDATE_URL=$policy_candidate_url
VICUNA_POLICY_TIMEOUT_MS=$policy_timeout_ms
VICUNA_POLICY_DATASET_DIR=$policy_dataset_dir
VICUNA_POLICY_REGISTRY_DIR=$policy_registry_dir
VICUNA_POLICY_RUN_ROOT=$policy_run_root
VICUNA_POLICY_MODEL_NAME=$policy_model_name
VICUNA_POLICY_SERVER_URL=$policy_server_url
VICUNA_POLICY_REGISTRY_HOST=$policy_registry_host
VICUNA_POLICY_REGISTRY_PORT=$policy_registry_port
VICUNA_POLICY_DEFAULT_ALIAS=$policy_default_alias
VICUNA_POLICY_FALLBACK_ALIAS=$policy_fallback_alias
VICUNA_POLICY_LIMIT=$policy_limit
VICUNA_POLICY_MIN_RECORD_COUNT=$policy_min_record_count
VICUNA_POLICY_MIN_EXACT_MATCH_RATE=$policy_min_exact_match_rate
VICUNA_POLICY_MAX_INVALID_ACTION_RATE=$policy_max_invalid_action_rate
VICUNA_POLICY_MIN_REWARD_DELTA=$policy_min_reward_delta
TAVILY_API_KEY=$tavily_api_key
RADARR_API_KEY=$radarr_api_key
RADARR_BASE_URL=$radarr_base_url
SONARR_API_KEY=$sonarr_api_key
SONARR_BASE_URL=$sonarr_base_url
CHAPTARR_API_KEY=$chaptarr_api_key
CHAPTARR_BASE_URL=$chaptarr_base_url
VICUNA_API_KEY=$vicuna_api_key
EOF
        return 0
    fi

    cat >"$ENV_FILE" <<EOF
VICUNA_SYSTEMD_SCOPE=system
VICUNA_SYSTEM_ENV_FILE=$ENV_FILE
VICUNA_SERVICE_USER=$SERVICE_USER
VICUNA_SERVICE_GROUP=$SERVICE_GROUP
VICUNA_RUNTIME_SERVICE_NAME=$RUNTIME_SERVICE
VICUNA_TELEGRAM_BRIDGE_SERVICE_NAME=$BRIDGE_SERVICE
VICUNA_WEBGL_RENDERER_SERVICE_NAME=$WEBGL_RENDERER_SERVICE
VICUNA_REPO_ROOT=$REPO_ROOT
REPO_ROOT=$REPO_ROOT
VICUNA_RUNTIME_BUILD_DIR=$runtime_build_dir
VICUNA_RUNTIME_PORT=$runtime_port
VICUNA_DEEPSEEK_BASE_URL=$deepseek_base_url
VICUNA_DEEPSEEK_MODEL=$deepseek_model
VICUNA_DEEPSEEK_TIMEOUT_MS=$deepseek_timeout_ms
VICUNA_DEEPSEEK_API_KEY=$deepseek_api_key
TELEGRAM_BOT_TOKEN=$telegram_bot_token
TELEGRAM_BRIDGE_STATE_PATH=$TELEGRAM_STATE_PATH
TELEGRAM_BRIDGE_NODE_BIN=$telegram_bridge_node_bin
TELEGRAM_BRIDGE_FFMPEG_BIN=$telegram_bridge_ffmpeg_bin
TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER=$telegram_bridge_video_encoder
TELEGRAM_BRIDGE_RENDER_BACKEND=$telegram_bridge_render_backend
VICUNA_WEBGL_RENDERER_URL=$webgl_renderer_url
VICUNA_WEBGL_RENDERER_HOST=$webgl_renderer_host
VICUNA_WEBGL_RENDERER_PORT=$webgl_renderer_port
VICUNA_WEBGL_RENDERER_CHROMIUM_BIN=$chromium_bin
VICUNA_WEBGL_RENDERER_FFMPEG_BIN=$webgl_renderer_ffmpeg_bin
VICUNA_WEBGL_RENDERER_FFMPEG_VIDEO_ENCODER=$webgl_renderer_video_encoder
VICUNA_WEBGL_RENDERER_MAX_CONCURRENT_RENDERS=$webgl_renderer_max_concurrent_renders
VICUNA_WEBGL_RENDERER_GPU_MEMORY_BUDGET_MB=$webgl_renderer_gpu_memory_budget_mb
VICUNA_WEBGL_RENDERER_MANDATORY_GPU=$webgl_renderer_mandatory_gpu
VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH=$OPENCLAW_SECRETS_PATH
VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=$OPENCLAW_CATALOG_PATH
VICUNA_STATE_ROOT=$STATE_ROOT
VICUNA_HARD_MEMORY_DIR=$HARD_MEMORY_DIR
VICUNA_SKILLS_DIR=$SKILLS_DIR
VICUNA_ONGOING_TASKS_DIR=$ONGOING_TASKS_DIR
VICUNA_ONGOING_TASKS_TMPDIR=$ONGOING_TASKS_TMPDIR
VICUNA_ONGOING_TASKS_RUNNER_SCRIPT=$ongoing_tasks_runner_script
VICUNA_ONGOING_TASKS_RUNTIME_URL=$ongoing_tasks_runtime_url
VICUNA_ONGOING_TASKS_RUNTIME_MODEL=$ongoing_tasks_runtime_model
VICUNA_OPENCLAW_NODE_BIN=$openclaw_node_bin
VICUNA_HOST_SHELL_ROOT=$SYSTEM_HOST_SHELL_ROOT
VICUNA_SKILLS_DIR=$SKILLS_DIR
VICUNA_DOCS_DIR=$DOCS_DIR
VICUNA_HEURISTIC_MEMORY_PATH=$HEURISTIC_MEMORY_PATH
VICUNA_POLICY_MODE=$policy_mode
VICUNA_POLICY_CANDIDATE_URL=$policy_candidate_url
VICUNA_POLICY_TIMEOUT_MS=$policy_timeout_ms
VICUNA_POLICY_DATASET_DIR=$policy_dataset_dir
VICUNA_POLICY_REGISTRY_DIR=$policy_registry_dir
VICUNA_POLICY_RUN_ROOT=$policy_run_root
VICUNA_POLICY_MODEL_NAME=$policy_model_name
VICUNA_POLICY_SERVER_URL=$policy_server_url
VICUNA_POLICY_REGISTRY_HOST=$policy_registry_host
VICUNA_POLICY_REGISTRY_PORT=$policy_registry_port
VICUNA_POLICY_DEFAULT_ALIAS=$policy_default_alias
VICUNA_POLICY_FALLBACK_ALIAS=$policy_fallback_alias
VICUNA_POLICY_LIMIT=$policy_limit
VICUNA_POLICY_MIN_RECORD_COUNT=$policy_min_record_count
VICUNA_POLICY_MIN_EXACT_MATCH_RATE=$policy_min_exact_match_rate
VICUNA_POLICY_MAX_INVALID_ACTION_RATE=$policy_max_invalid_action_rate
VICUNA_POLICY_MIN_REWARD_DELTA=$policy_min_reward_delta
TAVILY_API_KEY=$tavily_api_key
RADARR_API_KEY=$radarr_api_key
RADARR_BASE_URL=$radarr_base_url
SONARR_API_KEY=$sonarr_api_key
SONARR_BASE_URL=$sonarr_base_url
CHAPTARR_API_KEY=$chaptarr_api_key
CHAPTARR_BASE_URL=$chaptarr_base_url
VICUNA_API_KEY=$vicuna_api_key
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

stop_system_service_if_present() {
    local service_name="$1"
    local unit_path
    unit_path="$(system_unit_path "$service_name")"
    [[ -f "$unit_path" ]] || return 0
    run_cmd systemctl stop "$service_name"
}

reset_failed_service_if_present() {
    local service_name="$1"
    if (( DRY_RUN )); then
        run_cmd systemctl reset-failed "$service_name"
        return 0
    fi
    systemctl reset-failed "$service_name" >/dev/null 2>&1 || true
}

clear_system_override() {
    local service_name="$1"
    local override_dir
    override_dir="$(system_override_dir "$service_name")"
    if (( DRY_RUN )); then
        run_cmd rm -rf "$override_dir"
        return 0
    fi
    [[ -d "$override_dir" ]] || return 0
    rm -rf "$override_dir"
}

disable_and_remove_user_service() {
    local service_name="$1"
    local owner_name="$2"
    local owner_uid="$3"
    local home_dir="$4"
    local unit_path wants_path
    unit_path="$(interactive_user_unit_path "$home_dir" "$service_name")"
    wants_path="$(interactive_user_wants_path "$home_dir" "$service_name")"

    if (( DRY_RUN )); then
        if interactive_user_systemd_bus "$owner_uid"; then
            log "would disable user service $service_name for $owner_name"
        fi
        run_cmd rm -f "$unit_path" "$wants_path"
        return 0
    fi

    if interactive_user_systemd_bus "$owner_uid"; then
        runuser -u "$owner_name" -- env \
            XDG_RUNTIME_DIR="/run/user/$owner_uid" \
            DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$owner_uid/bus" \
            systemctl --user disable --now "$service_name" >/dev/null 2>&1 || true
    fi

    rm -f "$unit_path" "$wants_path"
}

reload_user_systemd_if_present() {
    local owner_name="$1"
    local owner_uid="$2"
    if (( DRY_RUN )); then
        return 0
    fi
    interactive_user_systemd_bus "$owner_uid" || return 0
    runuser -u "$owner_name" -- env \
        XDG_RUNTIME_DIR="/run/user/$owner_uid" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$owner_uid/bus" \
        systemctl --user daemon-reload >/dev/null 2>&1 || true
}

runtime_port_listeners() {
    ss -ltnp 2>/dev/null | awk -v port=":""$RUNTIME_PORT" '$4 ~ (port "$") { print }'
}

assert_runtime_port_clear() {
    if (( DRY_RUN )); then
        return 0
    fi

    local listeners=""
    listeners="$(runtime_port_listeners || true)"
    [[ -z "$listeners" ]] || die "runtime port $RUNTIME_PORT is still owned after convergence cleanup: $listeners"
}

wait_for_health() {
    local attempts=60
    for ((i = 0; i < attempts; ++i)); do
        if curl --silent --show-error "http://127.0.0.1:${RUNTIME_PORT}/health" >/dev/null 2>&1; then
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

require_root

NODE_BIN="$(resolve_node_bin)"
FFMPEG_BIN="$(resolve_ffmpeg_bin)"
CHROMIUM_BIN="$(resolve_chromium_bin)"
if [[ -z "$INTERACTIVE_OWNER" ]]; then
    INTERACTIVE_OWNER="$(path_owner_name "$REPO_ROOT")"
fi
if [[ -z "$INTERACTIVE_HOME" ]]; then
    INTERACTIVE_HOME="$(resolve_interactive_home)"
fi
collect_user_service_candidates

log "repo=$REPO_ROOT service_user=$SERVICE_USER interactive_owner=$INTERACTIVE_OWNER user_service_candidates=${#USER_SERVICE_HOMES[@]}"
ensure_group
ensure_user
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$STATE_ROOT" "$LOG_ROOT"
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$HARD_MEMORY_DIR"
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" \
    "$ONGOING_TASKS_DIR" "$ONGOING_TASKS_TMPDIR" "$SKILLS_DIR" "$DOCS_DIR" "$HEURISTICS_DIR" \
    "$POLICY_DATASET_DIR" "$POLICY_REGISTRY_DIR" "$POLICY_RUN_ROOT"
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$(dirname "$TELEGRAM_STATE_PATH")"
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$(dirname "$SYSTEM_HOST_SHELL_ROOT")"
run_cmd install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$SYSTEM_HOST_SHELL_ROOT"
ensure_service_writable_file "$TELEGRAM_STATE_PATH" 0640
ensure_service_writable_file "$OPENCLAW_SECRETS_PATH" 0600
ensure_service_writable_file "$OPENCLAW_CATALOG_PATH" 0640
grant_repo_access

stop_system_service_if_present "$BRIDGE_SERVICE"
stop_system_service_if_present "$WEBGL_RENDERER_SERVICE"
stop_system_service_if_present "$RUNTIME_SERVICE"

for idx in "${!USER_SERVICE_HOMES[@]}"; do
    owner_name="${USER_SERVICE_OWNERS[$idx]}"
    home_dir="${USER_SERVICE_HOMES[$idx]}"
    owner_uid="$(id -u "$owner_name" 2>/dev/null || true)"
    disable_and_remove_user_service "$BRIDGE_SERVICE" "$owner_name" "$owner_uid" "$home_dir"
    disable_and_remove_user_service "$WEBGL_RENDERER_SERVICE" "$owner_name" "$owner_uid" "$home_dir"
    disable_and_remove_user_service "$RUNTIME_SERVICE" "$owner_name" "$owner_uid" "$home_dir"
    reload_user_systemd_if_present "$owner_name" "$owner_uid"
done

clear_system_override "$BRIDGE_SERVICE"
clear_system_override "$WEBGL_RENDERER_SERVICE"
clear_system_override "$RUNTIME_SERVICE"
assert_runtime_port_clear

write_env_file "$NODE_BIN" "$FFMPEG_BIN" "$CHROMIUM_BIN"
install_unit "$REPO_ROOT/tools/ops/systemd/vicuna-runtime.system.service" "$(system_unit_path "$RUNTIME_SERVICE")"
install_unit "$REPO_ROOT/tools/ops/systemd/vicuna-telegram-bridge.system.service" "$(system_unit_path "$BRIDGE_SERVICE")"
install_unit "$REPO_ROOT/tools/ops/systemd/vicuna-webgl-renderer.system.service" "$(system_unit_path "$WEBGL_RENDERER_SERVICE")"
run_cmd systemctl daemon-reload
reset_failed_service_if_present "$RUNTIME_SERVICE"
reset_failed_service_if_present "$WEBGL_RENDERER_SERVICE"
reset_failed_service_if_present "$BRIDGE_SERVICE"
run_cmd systemctl enable "$RUNTIME_SERVICE" "$WEBGL_RENDERER_SERVICE" "$BRIDGE_SERVICE"
run_cmd systemctl restart "$RUNTIME_SERVICE"
run_cmd systemctl restart "$WEBGL_RENDERER_SERVICE"
run_cmd systemctl restart "$BRIDGE_SERVICE"

if (( DRY_RUN )); then
    log "system-service install complete"
    exit 0
fi

wait_for_health || die "runtime health endpoint did not recover after install"

log "system-service install complete"
log "runtime unit: $(system_unit_path "$RUNTIME_SERVICE")"
log "webgl renderer unit: $(system_unit_path "$WEBGL_RENDERER_SERVICE")"
log "bridge unit: $(system_unit_path "$BRIDGE_SERVICE")"
log "env file: $ENV_FILE"
