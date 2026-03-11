#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_UPSTREAM_DIR="${HOME}/.cache/supermemory-upstream"
DEFAULT_WORKDIR="${SCRIPT_DIR}/workdir/mcp-worker"
DEFAULT_WRANGLER_FALLBACK="/Users/tyleraraujo/arcagent/worker/cloudflare/node_modules/wrangler/bin/wrangler.js"
DEFAULT_API_URL="https://api.supermemory.ai"
DEFAULT_WORKER_NAME="vicuna-supermemory-mcp"
DEFAULT_MIN_FREE_MB=512

MODE="prepare"
UPSTREAM_DIR="${SUPERMEMORY_UPSTREAM_DIR:-$DEFAULT_UPSTREAM_DIR}"
WORKDIR="${SUPERMEMORY_MCP_WORKDIR:-$DEFAULT_WORKDIR}"
API_URL="${SUPERMEMORY_API_URL:-$DEFAULT_API_URL}"
WORKER_NAME="${SUPERMEMORY_MCP_WORKER_NAME:-$DEFAULT_WORKER_NAME}"
WRANGLER_BIN="${WRANGLER_BIN:-}"
MIN_FREE_MB="${SUPERMEMORY_MCP_MIN_FREE_MB:-$DEFAULT_MIN_FREE_MB}"
SKIP_INSTALL=0

usage() {
    cat <<'EOF'
Usage:
  bootstrap-mcp-worker.sh [prepare|dry-run|deploy] [options]

Options:
  --upstream-dir PATH   Cache directory for the upstream supermemory repo
  --workdir PATH        Generated local MCP worker workspace
  --api-url URL         API_URL injected into the generated worker config
  --worker-name NAME    Cloudflare worker name for the generated config
  --wrangler-bin PATH   Explicit Wrangler binary or wrangler.js path
  --skip-install        Skip npm install after copying the workspace
  -h, --help            Show this help

Environment overrides:
  SUPERMEMORY_UPSTREAM_DIR
  SUPERMEMORY_MCP_WORKDIR
  SUPERMEMORY_API_URL
  SUPERMEMORY_MCP_WORKER_NAME
  SUPERMEMORY_MCP_MIN_FREE_MB
  WRANGLER_BIN

Notes:
  - Full Supermemory API self-hosting is documented upstream as enterprise-only.
  - This bootstrap prepares the public-source MCP worker slice and can point it
    at either the hosted API or a future private/self-hosted API_URL.
EOF
}

log() {
    printf '[supermemory-selfhost] %s\n' "$*"
}

fail() {
    printf '[supermemory-selfhost] error: %s\n' "$*" >&2
    exit 1
}

run_wrangler() {
    if [[ "${WRANGLER_BIN}" == *.js ]]; then
        node "${WRANGLER_BIN}" "$@"
    else
        "${WRANGLER_BIN}" "$@"
    fi
}

resolve_wrangler() {
    if [[ -n "${WRANGLER_BIN}" ]]; then
        [[ -e "${WRANGLER_BIN}" ]] || fail "configured WRANGLER_BIN does not exist: ${WRANGLER_BIN}"
        return
    fi

    if command -v wrangler >/dev/null 2>&1; then
        WRANGLER_BIN="$(command -v wrangler)"
        return
    fi

    if [[ -f "${DEFAULT_WRANGLER_FALLBACK}" ]]; then
        WRANGLER_BIN="${DEFAULT_WRANGLER_FALLBACK}"
        return
    fi

    fail "could not resolve wrangler from PATH or fallback path"
}

require_tools() {
    command -v git >/dev/null 2>&1 || fail "git is required"
    command -v node >/dev/null 2>&1 || fail "node is required"
    command -v npm >/dev/null 2>&1 || fail "npm is required"
}

check_auth() {
    log "verifying Cloudflare auth via Wrangler"
    run_wrangler whoami >/dev/null
}

check_free_space() {
    local target_dir available_kb available_mb
    target_dir="$(dirname "${WORKDIR}")"
    mkdir -p "${target_dir}"
    available_kb="$(df -Pk "${target_dir}" | tail -1 | awk '{print $4}')"
    available_mb="$(( available_kb / 1024 ))"

    if (( available_mb < MIN_FREE_MB )); then
        fail "only ${available_mb} MiB free under ${target_dir}; need at least ${MIN_FREE_MB} MiB for npm install, or rerun with --skip-install"
    fi
}

sync_upstream() {
    if [[ -d "${UPSTREAM_DIR}/.git" ]]; then
        log "refreshing upstream repo in ${UPSTREAM_DIR}"
        git -C "${UPSTREAM_DIR}" fetch --depth 1 origin main
        git -C "${UPSTREAM_DIR}" checkout main >/dev/null 2>&1
        git -C "${UPSTREAM_DIR}" reset --hard origin/main >/dev/null
    else
        log "cloning upstream supermemory repo into ${UPSTREAM_DIR}"
        mkdir -p "$(dirname "${UPSTREAM_DIR}")"
        git clone --depth 1 https://github.com/supermemoryai/supermemory.git "${UPSTREAM_DIR}"
    fi
}

prepare_workspace() {
    local source_dir="${UPSTREAM_DIR}/apps/mcp"
    [[ -d "${source_dir}" ]] || fail "missing upstream MCP source at ${source_dir}"

    log "preparing workspace at ${WORKDIR}"
    rm -rf "${WORKDIR}"
    mkdir -p "${WORKDIR}"
    cp -R "${source_dir}/." "${WORKDIR}/"

    WORKDIR="${WORKDIR}" node <<'EOF'
const fs = require('fs');
const path = require('path');

const workdir = process.env.WORKDIR;
const indexPath = path.join(workdir, 'src', 'index.ts');
let source = fs.readFileSync(indexPath, 'utf8');

source = source.replace(
`type Bindings = {
\tMCP_SERVER: DurableObjectNamespace
\tAPI_URL?: string
\tPOSTHOG_API_KEY?: string
}`,
`type Bindings = {
\tMCP_SERVER: DurableObjectNamespace
\tAPI_URL?: string
\tPUBLIC_RESOURCE_URL?: string
\tPOSTHOG_API_KEY?: string
}`
);

source = source.replace(
`app.get("/.well-known/oauth-protected-resource", (c) => {
\tconst apiUrl = c.env.API_URL || DEFAULT_API_URL
\tconst resourceUrl =
\t\tc.env.API_URL === "http://localhost:8787"
\t\t\t? "http://localhost:8788"
\t\t\t: "https://mcp.supermemory.ai"
`,
`app.get("/.well-known/oauth-protected-resource", (c) => {
\tconst apiUrl = c.env.API_URL || DEFAULT_API_URL
\tconst requestOrigin = new URL(c.req.url).origin
\tconst resourceUrl =
\t\tc.env.PUBLIC_RESOURCE_URL ||
\t\t(c.env.API_URL === "http://localhost:8787"
\t\t\t? "http://localhost:8788"
\t\t\t: requestOrigin)
`
);

fs.writeFileSync(indexPath, source);
EOF

    cat > "${WORKDIR}/wrangler.vicuna.jsonc" <<EOF
{
  "\$schema": "node_modules/wrangler/config-schema.json",
  "name": "${WORKER_NAME}",
  "main": "src/index.ts",
  "compatibility_date": "2025-01-01",
  "compatibility_flags": ["nodejs_compat"],
  "workers_dev": true,
  "rules": [{ "type": "Text", "globs": ["**/*.html"], "fallthrough": false }],
  "vars": {
    "API_URL": "${API_URL}",
    "PUBLIC_RESOURCE_URL": "https://${WORKER_NAME}.araujota97.workers.dev"
  },
  "durable_objects": {
    "bindings": [
      {
        "name": "MCP_SERVER",
        "class_name": "SupermemoryMCP"
      }
    ]
  },
  "migrations": [
    {
      "tag": "v1",
      "new_sqlite_classes": ["SupermemoryMCP"]
    }
  ],
  "observability": {
    "enabled": true
  }
}
EOF

    cat > "${WORKDIR}/.dev.vars" <<EOF
API_URL=${API_URL}
EOF

    mkdir -p "${WORKDIR}/dist"
    cp "${WORKDIR}/mcp-app.html" "${WORKDIR}/dist/mcp-app.html"

    if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
        check_free_space
        log "installing MCP worker runtime dependencies with npm"
        npm install --omit=dev --no-fund --no-audit --prefix "${WORKDIR}"
    else
        log "skipping npm install"
    fi
}

deploy_workspace() {
    log "running Wrangler ${MODE} for ${WORKER_NAME}"
    if [[ "${MODE}" == "dry-run" ]]; then
        run_wrangler deploy \
            --config "${WORKDIR}/wrangler.vicuna.jsonc" \
            --cwd "${WORKDIR}" \
            --name "${WORKER_NAME}" \
            --dry-run
    else
        run_wrangler deploy \
            --config "${WORKDIR}/wrangler.vicuna.jsonc" \
            --cwd "${WORKDIR}" \
            --name "${WORKER_NAME}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        prepare|dry-run|deploy)
            MODE="$1"
            shift
            ;;
        --upstream-dir)
            UPSTREAM_DIR="$2"
            shift 2
            ;;
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --worker-name)
            WORKER_NAME="$2"
            shift 2
            ;;
        --wrangler-bin)
            WRANGLER_BIN="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

require_tools
resolve_wrangler
check_auth
sync_upstream
prepare_workspace

if [[ "${MODE}" == "dry-run" || "${MODE}" == "deploy" ]]; then
    deploy_workspace
else
    log "prepare complete"
    log "workspace: ${WORKDIR}"
    log "config: ${WORKDIR}/wrangler.vicuna.jsonc"
fi
