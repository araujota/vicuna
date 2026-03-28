#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
preserved_env_vars=(
    REPO_ROOT
    VICUNA_SYSTEM_ENV_FILE
    TELEGRAM_BRIDGE_NODE_BIN
    TELEGRAM_BRIDGE_FFMPEG_BIN
    TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER
    VICUNA_WEBGL_RENDERER_HOST
    VICUNA_WEBGL_RENDERER_PORT
    VICUNA_WEBGL_RENDERER_CHROMIUM_BIN
    VICUNA_WEBGL_RENDERER_FFMPEG_BIN
    VICUNA_WEBGL_RENDERER_FFMPEG_VIDEO_ENCODER
    VICUNA_WEBGL_RENDERER_MAX_CONCURRENT_RENDERS
    VICUNA_WEBGL_RENDERER_GPU_MEMORY_BUDGET_MB
    VICUNA_WEBGL_RENDERER_MANDATORY_GPU
    VICUNA_WEBGL_RENDERER_TIMEOUT_MS
    VICUNA_WEBGL_RENDERER_STARTUP_TIMEOUT_MS
    VICUNA_WEBGL_RENDERER_STATE_ROOT
    VICUNA_WEBGL_RENDERER_TEMP_ROOT
    __NV_PRIME_RENDER_OFFLOAD
    __VK_LAYER_NV_optimus
    __GLX_VENDOR_LIBRARY_NAME
    DRI_PRIME
    VK_DRIVER_FILES
    VK_ICD_FILENAMES
    TMPDIR
    TMP
    TEMP
    XDG_RUNTIME_DIR
)

restore_preserved_env() {
    local var_name="$1"
    local saved_var="__saved_${var_name}"
    local saved_value="${!saved_var:-__unset__}"
    if [[ "$saved_value" != "__unset__" ]]; then
        export "$var_name=$saved_value"
    fi
    unset "$saved_var"
}

for var_name in "${preserved_env_vars[@]}"; do
    saved_var="__saved_${var_name}"
    printf -v "$saved_var" '%s' "${!var_name-__unset__}"
done

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

for var_name in "${preserved_env_vars[@]}"; do
    restore_preserved_env "$var_name"
done

MIN_NODE_MAJOR=20
MIN_NODE_MINOR=16

node_version_ge() {
    local version="$1"
    version="${version#v}"
    local major minor patch
    IFS='.' read -r major minor patch <<<"$version"
    major="${major:-0}"
    minor="${minor:-0}"
    if (( major > MIN_NODE_MAJOR )); then
        return 0
    fi
    if (( major < MIN_NODE_MAJOR )); then
        return 1
    fi
    (( minor >= MIN_NODE_MINOR ))
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

    printf '[vicuna-webgl-renderer] error: Node.js >= %d.%d is required; set TELEGRAM_BRIDGE_NODE_BIN.\n' \
        "$MIN_NODE_MAJOR" "$MIN_NODE_MINOR" >&2
    return 1
}

resolve_ffmpeg_bin() {
    if [[ -n "${VICUNA_WEBGL_RENDERER_FFMPEG_BIN:-}" && -x "${VICUNA_WEBGL_RENDERER_FFMPEG_BIN:-}" ]]; then
        printf '%s\n' "$VICUNA_WEBGL_RENDERER_FFMPEG_BIN"
        return 0
    fi
    if [[ -n "${TELEGRAM_BRIDGE_FFMPEG_BIN:-}" && -x "${TELEGRAM_BRIDGE_FFMPEG_BIN:-}" ]]; then
        printf '%s\n' "$TELEGRAM_BRIDGE_FFMPEG_BIN"
        return 0
    fi
    for candidate in /usr/bin/ffmpeg /opt/homebrew/bin/ffmpeg; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    printf '[vicuna-webgl-renderer] error: ffmpeg is required; set VICUNA_WEBGL_RENDERER_FFMPEG_BIN.\n' >&2
    return 1
}

resolve_chromium_bin() {
    if [[ -n "${VICUNA_WEBGL_RENDERER_CHROMIUM_BIN:-}" && -x "${VICUNA_WEBGL_RENDERER_CHROMIUM_BIN:-}" ]]; then
        printf '%s\n' "$VICUNA_WEBGL_RENDERER_CHROMIUM_BIN"
        return 0
    fi
    for candidate in /snap/bin/chromium /usr/bin/chromium /usr/bin/chromium-browser /usr/bin/google-chrome-stable; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    printf '[vicuna-webgl-renderer] error: Chromium is required; set VICUNA_WEBGL_RENDERER_CHROMIUM_BIN.\n' >&2
    return 1
}

configure_gpu_env() {
    local nvidia_icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
    if [[ ! -f "$nvidia_icd" ]]; then
        return 0
    fi

    export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
    export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
    export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
    export DRI_PRIME="${DRI_PRIME:-1}"
    export VK_DRIVER_FILES="${VK_DRIVER_FILES:-$nvidia_icd}"
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$nvidia_icd}"
}

configure_runtime_dirs() {
    local service_home="${HOME:-/var/lib/vicuna}"
    local state_root="${VICUNA_WEBGL_RENDERER_STATE_ROOT:-$service_home/webgl-renderer}"
    local runtime_dir="$state_root/xdg-runtime"
    local tmp_dir="$state_root/tmp"
    local temp_root="${VICUNA_WEBGL_RENDERER_TEMP_ROOT:-$state_root/renderer-artifacts}"

    mkdir -p "$runtime_dir" "$tmp_dir" "$temp_root"
    chmod 700 "$runtime_dir" "$tmp_dir" "$temp_root"

    export VICUNA_WEBGL_RENDERER_STATE_ROOT="$state_root"
    export VICUNA_WEBGL_RENDERER_TEMP_ROOT="$temp_root"
    export XDG_RUNTIME_DIR="$runtime_dir"
    export TMPDIR="$tmp_dir"
    export TMP="$tmp_dir"
    export TEMP="$tmp_dir"
}

NODE_BIN="$(resolve_node_bin)"
export VICUNA_WEBGL_RENDERER_FFMPEG_BIN="$(resolve_ffmpeg_bin)"
export VICUNA_WEBGL_RENDERER_CHROMIUM_BIN="$(resolve_chromium_bin)"
export VICUNA_WEBGL_RENDERER_FFMPEG_VIDEO_ENCODER="${VICUNA_WEBGL_RENDERER_FFMPEG_VIDEO_ENCODER:-${TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER:-h264_nvenc}}"
configure_gpu_env
configure_runtime_dirs

exec "$NODE_BIN" "$REPO_ROOT/tools/telegram-bridge/emotive-webgl-renderer-service.mjs"
