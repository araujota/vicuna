#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

unset VICUNA_RUNTIME_STATE_PATH
unset VICUNA_RUNTIME_STATE_BACKUP_DIR
unset VICUNA_BASH_TOOL_ENABLED
unset VICUNA_BASH_TOOL_LOGIN_SHELL
unset VICUNA_BASH_TOOL_MAX_CHILD_PROCESSES
unset VICUNA_BASH_TOOL_ALLOWED_COMMANDS
unset VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH
unset VICUNA_RUNTIME_MODEL_DIR
unset VICUNA_RUNTIME_MODEL_NAME
unset VICUNA_RUNTIME_MODEL_PATH
unset VICUNA_RUNTIME_MODEL_URL
unset VICUNA_RUNTIME_MODEL_ALIAS
unset VICUNA_RUNTIME_MODEL_CHAT_TEMPLATE_FILE
unset VICUNA_RUNTIME_MODEL_REASONING_FORMAT
unset VICUNA_RUNTIME_CTX_SIZE
unset TELEGRAM_BRIDGE_MODEL

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

EXPECTED_STATE_PATH="$REPO_ROOT/.cache/vicuna/runtime-state.json"
EXPECTED_BACKUP_DIR="$REPO_ROOT/.cache/vicuna/runtime-state-backups"
EXPECTED_CATALOG_PATH="$REPO_ROOT/.cache/vicuna/openclaw-catalog.json"
EXPECTED_MODEL_PATH="$REPO_ROOT/models/runtime/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"
EXPECTED_TEMPLATE_PATH="$REPO_ROOT/models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja"

[[ "$VICUNA_RUNTIME_STATE_PATH" == "$EXPECTED_STATE_PATH" ]] || {
    printf 'expected runtime state path %s, got %s\n' "$EXPECTED_STATE_PATH" "$VICUNA_RUNTIME_STATE_PATH" >&2
    exit 1
}

[[ "$VICUNA_RUNTIME_STATE_BACKUP_DIR" == "$EXPECTED_BACKUP_DIR" ]] || {
    printf 'expected backup dir %s, got %s\n' "$EXPECTED_BACKUP_DIR" "$VICUNA_RUNTIME_STATE_BACKUP_DIR" >&2
    exit 1
}

[[ "$VICUNA_BASH_TOOL_ENABLED" == "1" ]] || {
    printf 'expected bash tool enabled by default, got %s\n' "$VICUNA_BASH_TOOL_ENABLED" >&2
    exit 1
}

[[ "$VICUNA_BASH_TOOL_LOGIN_SHELL" == "0" ]] || {
    printf 'expected bash tool login shell disabled by default, got %s\n' "$VICUNA_BASH_TOOL_LOGIN_SHELL" >&2
    exit 1
}

[[ "$VICUNA_BASH_TOOL_MAX_CHILD_PROCESSES" == "4096" ]] || {
    printf 'expected bash tool max child processes 4096 by default, got %s\n' "$VICUNA_BASH_TOOL_MAX_CHILD_PROCESSES" >&2
    exit 1
}

[[ -z "$VICUNA_BASH_TOOL_ALLOWED_COMMANDS" ]] || {
    printf 'expected empty bash tool allowlist by default, got %s\n' "$VICUNA_BASH_TOOL_ALLOWED_COMMANDS" >&2
    exit 1
}

[[ "$VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH" == "$EXPECTED_CATALOG_PATH" ]] || {
    printf 'expected OpenClaw catalog path %s, got %s\n' "$EXPECTED_CATALOG_PATH" "$VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH" >&2
    exit 1
}

[[ "$VICUNA_RUNTIME_MODEL_PATH" == "$EXPECTED_MODEL_PATH" ]] || {
    printf 'expected runtime model path %s, got %s\n' "$EXPECTED_MODEL_PATH" "$VICUNA_RUNTIME_MODEL_PATH" >&2
    exit 1
}

[[ "$VICUNA_RUNTIME_MODEL_ALIAS" == "vicuna-runtime" ]] || {
    printf 'expected runtime model alias vicuna-runtime, got %s\n' "$VICUNA_RUNTIME_MODEL_ALIAS" >&2
    exit 1
}

[[ "$VICUNA_RUNTIME_MODEL_CHAT_TEMPLATE_FILE" == "$EXPECTED_TEMPLATE_PATH" ]] || {
    printf 'expected runtime model template %s, got %s\n' "$EXPECTED_TEMPLATE_PATH" "$VICUNA_RUNTIME_MODEL_CHAT_TEMPLATE_FILE" >&2
    exit 1
}

[[ "$VICUNA_RUNTIME_MODEL_REASONING_FORMAT" == "deepseek" ]] || {
    printf 'expected runtime model reasoning format deepseek, got %s\n' "$VICUNA_RUNTIME_MODEL_REASONING_FORMAT" >&2
    exit 1
}

[[ "$VICUNA_RUNTIME_CTX_SIZE" == "16384" ]] || {
    printf 'expected runtime context size 16384 by default, got %s\n' "$VICUNA_RUNTIME_CTX_SIZE" >&2
    exit 1
}

[[ "$VICUNA_BASH_TOOL_MAX_STDOUT_BYTES" == "819200" ]] || {
    printf 'expected bash tool stdout budget 819200 by default, got %s\n' "$VICUNA_BASH_TOOL_MAX_STDOUT_BYTES" >&2
    exit 1
}

[[ "$TELEGRAM_BRIDGE_MODEL" == "vicuna-runtime" ]] || {
    printf 'expected telegram bridge model vicuna-runtime, got %s\n' "$TELEGRAM_BRIDGE_MODEL" >&2
    exit 1
}

printf 'runtime-env defaults ok\n'
