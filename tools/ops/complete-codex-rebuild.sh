#!/usr/bin/env bash
set -euo pipefail

PENDING_MESSAGE_FILE=""
FINAL_MESSAGE_FILE=""
REBUILD_SCRIPT=""

usage() {
    cat <<'EOF'
Usage: complete-codex-rebuild.sh --pending-message-file PATH --final-message-file PATH --rebuild-script PATH
EOF
}

while (($# > 0)); do
    case "$1" in
        --pending-message-file)
            PENDING_MESSAGE_FILE="${2:-}"
            shift 2
            ;;
        --final-message-file)
            FINAL_MESSAGE_FILE="${2:-}"
            shift 2
            ;;
        --rebuild-script)
            REBUILD_SCRIPT="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[codex-rebuild-helper] unknown option: $1" >&2
            exit 1
            ;;
    esac
done

[[ -n "$PENDING_MESSAGE_FILE" ]] || { echo "[codex-rebuild-helper] missing --pending-message-file" >&2; exit 1; }
[[ -n "$FINAL_MESSAGE_FILE" ]] || { echo "[codex-rebuild-helper] missing --final-message-file" >&2; exit 1; }
[[ -n "$REBUILD_SCRIPT" ]] || { echo "[codex-rebuild-helper] missing --rebuild-script" >&2; exit 1; }

BASE_MESSAGE=""
if [[ -f "$PENDING_MESSAGE_FILE" ]]; then
    BASE_MESSAGE="$(cat "$PENDING_MESSAGE_FILE")"
fi
mkdir -p "$(dirname "$FINAL_MESSAGE_FILE")"

if "$REBUILD_SCRIPT" --allow-busy-stop; then
    {
        printf 'REBUILD_STATUS=ok\n'
        printf '%s\n' "$BASE_MESSAGE"
    } > "$FINAL_MESSAGE_FILE"
else
    {
        printf 'REBUILD_STATUS=failed\n'
        printf '%s\n' "$BASE_MESSAGE"
    } > "$FINAL_MESSAGE_FILE"
    exit 1
fi
