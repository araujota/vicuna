#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if (( $# < 2 )); then
    printf 'usage: %s <service> <command> [args...]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

SERVICE="$1"
shift

exec python3 "$REPO_ROOT/tools/ops/service_log_router.py" run \
    --service "$SERVICE" \
    --log-root "${VICUNA_LOG_ROOT:-/var/log/vicuna}" \
    --retention-days "${VICUNA_LOG_RETENTION_DAYS:-7}" \
    -- "$@"
