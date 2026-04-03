#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

exec python3 "$REPO_ROOT/tools/ops/runpod-mistral-relay.py"
