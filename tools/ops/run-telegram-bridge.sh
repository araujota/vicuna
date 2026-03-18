#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/.envrc"

exec /usr/bin/env node "$REPO_ROOT/tools/telegram-bridge/index.mjs"
