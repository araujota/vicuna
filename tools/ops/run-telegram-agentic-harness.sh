#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

resolve_node_bin() {
    if [[ -n "${TELEGRAM_BRIDGE_NODE_BIN:-}" && -x "${TELEGRAM_BRIDGE_NODE_BIN:-}" ]]; then
        printf '%s\n' "$TELEGRAM_BRIDGE_NODE_BIN"
        return 0
    fi
    if command -v node >/dev/null 2>&1; then
        printf '%s\n' "$(command -v node)"
        return 0
    fi
    if [[ -x "/home/tyler-araujo/.nvm/versions/node/v20.19.5/bin/node" ]]; then
        printf '%s\n' "/home/tyler-araujo/.nvm/versions/node/v20.19.5/bin/node"
        return 0
    fi
    printf '[telegram-agentic-harness] error: Node.js is required.\n' >&2
    return 1
}

NODE_BIN="$(resolve_node_bin)"
exec "$NODE_BIN" "$REPO_ROOT/tools/ops/telegram-agentic-harness.mjs" "$@"
