#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
preserved_env_vars=(
    REPO_ROOT
    VICUNA_SYSTEM_ENV_FILE
    TELEGRAM_BOT_TOKEN
    TELEGRAM_BRIDGE_VICUNA_BASE_URL
    TELEGRAM_BRIDGE_MODEL
    TELEGRAM_BRIDGE_STATE_PATH
    TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS
    TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES
    TELEGRAM_BRIDGE_MAX_TOKENS
    TELEGRAM_BRIDGE_MAX_DOCUMENT_CHARS
    TELEGRAM_BRIDGE_NODE_BIN
    SUPERMEMORY_API_KEY
    SUPERMEMORY_BASE_URL
    TAVILY_API_KEY
    RADARR_API_KEY
    RADARR_BASE_URL
    SONARR_API_KEY
    SONARR_BASE_URL
    CHAPTARR_API_KEY
    CHAPTARR_BASE_URL
    VICUNA_OPENCLAW_NODE_BIN
    VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH
    VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH
    VICUNA_API_KEY
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

    if [[ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]]; then
        # shellcheck disable=SC1090
        source "${NVM_DIR:-$HOME/.nvm}/nvm.sh"
        local nvm_node=""
        nvm_node="$(nvm which 20 2>/dev/null || true)"
        if [[ -n "$nvm_node" && -x "$nvm_node" ]] && node_version_ge "$("$nvm_node" -v)"; then
            printf '%s\n' "$nvm_node"
            return 0
        fi
    fi

    printf '[telegram-bridge] error: Node.js >= %d.%d is required; set TELEGRAM_BRIDGE_NODE_BIN or install a supported Node runtime.\n' \
        "$MIN_NODE_MAJOR" "$MIN_NODE_MINOR" >&2
    return 1
}

NODE_BIN="$(resolve_node_bin)"
"$REPO_ROOT/tools/ops/sync-openclaw-runtime-state.sh"

exec "$NODE_BIN" "$REPO_ROOT/tools/telegram-bridge/index.mjs"
