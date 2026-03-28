#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

OPENCLAW_INDEX_JS="${VICUNA_OPENCLAW_INDEX_JS:-$REPO_ROOT/tools/openclaw-harness/dist/index.js}"
OPENCLAW_HARNESS_DIR="${VICUNA_OPENCLAW_HARNESS_DIR:-$REPO_ROOT/tools/openclaw-harness}"

log() {
  printf '[vicuna-openclaw-sync] %s\n' "$*" >&2
}

resolve_node_bin() {
  if [[ -n "${VICUNA_OPENCLAW_NODE_BIN:-}" && -x "${VICUNA_OPENCLAW_NODE_BIN:-}" ]]; then
    printf '%s\n' "$VICUNA_OPENCLAW_NODE_BIN"
    return 0
  fi
  if [[ -n "${TELEGRAM_BRIDGE_NODE_BIN:-}" && -x "${TELEGRAM_BRIDGE_NODE_BIN:-}" ]]; then
    printf '%s\n' "$TELEGRAM_BRIDGE_NODE_BIN"
    return 0
  fi
  if command -v node >/dev/null 2>&1; then
    command -v node
    return 0
  fi
  printf '[vicuna-openclaw-sync] error: could not resolve a Node.js binary\n' >&2
  return 1
}

NODE_BIN="$(resolve_node_bin)"

resolve_npm_command() {
  if [[ -n "${VICUNA_OPENCLAW_NPM_CLI:-}" && -f "${VICUNA_OPENCLAW_NPM_CLI:-}" ]]; then
    printf '%s\n' "$NODE_BIN"
    printf '%s\n' "${VICUNA_OPENCLAW_NPM_CLI}"
    return 0
  fi

  local node_prefix
  node_prefix="$(CDPATH='' cd -- "$(dirname -- "$NODE_BIN")/.." && pwd)"
  local derived_cli="$node_prefix/lib/node_modules/npm/bin/npm-cli.js"
  if [[ -f "$derived_cli" ]]; then
    printf '%s\n' "$NODE_BIN"
    printf '%s\n' "$derived_cli"
    return 0
  fi

  if command -v npm >/dev/null 2>&1; then
    printf '%s\n' "npm"
    printf '%s\n' ""
    return 0
  fi

  printf '[vicuna-openclaw-sync] error: could not resolve an npm CLI for %s\n' "$NODE_BIN" >&2
  return 1
}

run_npm() {
  local command_bin="$1"
  local command_arg="$2"
  shift 2
  if [[ "$command_bin" == "npm" ]]; then
    npm "$@"
    return
  fi
  "$command_bin" "$command_arg" "$@"
}

ensure_openclaw_entrypoint() {
  if [[ -f "$OPENCLAW_INDEX_JS" ]]; then
    return 0
  fi

  local expected_dir="$OPENCLAW_HARNESS_DIR/dist"
  local actual_dir
  actual_dir="$(dirname "$OPENCLAW_INDEX_JS")"
  if [[ "$actual_dir" != "$expected_dir" ]]; then
    printf '[vicuna-openclaw-sync] error: %s is missing and is outside the managed harness dist directory %s\n' \
      "$OPENCLAW_INDEX_JS" "$expected_dir" >&2
    return 1
  fi

  if [[ ! -f "$OPENCLAW_HARNESS_DIR/package.json" ]]; then
    printf '[vicuna-openclaw-sync] error: %s is missing and %s does not contain package.json\n' \
      "$OPENCLAW_INDEX_JS" "$OPENCLAW_HARNESS_DIR" >&2
    return 1
  fi

  local npm_command
  local npm_arg
  local npm_parts
  npm_parts="$(resolve_npm_command)"
  npm_command="${npm_parts%%$'\n'*}"
  npm_arg=""
  if [[ "$npm_parts" == *$'\n'* ]]; then
    npm_arg="${npm_parts#*$'\n'}"
  fi
  if [[ -z "$npm_command" ]]; then
    return 1
  fi

  local install_verb="install"
  if [[ -f "$OPENCLAW_HARNESS_DIR/package-lock.json" ]]; then
    install_verb="ci"
  fi

  log "bootstrapping OpenClaw harness because $OPENCLAW_INDEX_JS is missing"
  run_npm "$npm_command" "$npm_arg" "$install_verb" --prefix "$OPENCLAW_HARNESS_DIR" >/dev/null
  run_npm "$npm_command" "$npm_arg" run --prefix "$OPENCLAW_HARNESS_DIR" build >/dev/null

  if [[ ! -f "$OPENCLAW_INDEX_JS" ]]; then
    printf '[vicuna-openclaw-sync] error: expected harness entrypoint %s after build, but it is still missing\n' \
      "$OPENCLAW_INDEX_JS" >&2
    return 1
  fi
}

ensure_openclaw_entrypoint

SECRETS_PATH="${VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH:-$REPO_ROOT/.cache/vicuna/openclaw-tool-secrets.json}"
CATALOG_PATH="${VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH:-$REPO_ROOT/.cache/vicuna/openclaw-catalog.json}"
mkdir -p "$(dirname "$SECRETS_PATH")" "$(dirname "$CATALOG_PATH")"

run_openclaw() {
  "$NODE_BIN" "$OPENCLAW_INDEX_JS" "$@" >/dev/null
}

if [[ -n "${TAVILY_API_KEY:-}" ]]; then
  run_openclaw install-tavily "$TAVILY_API_KEY"
fi
if [[ -n "${RADARR_API_KEY:-}" ]]; then
  run_openclaw install-radarr "$RADARR_API_KEY" "${RADARR_BASE_URL:-http://10.0.0.218:7878}"
fi
if [[ -n "${SONARR_API_KEY:-}" ]]; then
  run_openclaw install-sonarr "$SONARR_API_KEY" "${SONARR_BASE_URL:-http://10.0.0.218:8989}"
fi
if [[ -n "${CHAPTARR_API_KEY:-}" ]]; then
  run_openclaw install-chaptarr "$CHAPTARR_API_KEY" "${CHAPTARR_BASE_URL:-http://10.0.0.218:8789}"
fi

"$NODE_BIN" <<'EOF'
const fs = require("fs");
const path = require("path");

const trim = (value) => typeof value === "string" ? value.trim() : "";
const secretsPath = trim(process.env.VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH);
if (!secretsPath) {
  process.exit(0);
}

let secrets = {};
if (fs.existsSync(secretsPath)) {
  const raw = fs.readFileSync(secretsPath, "utf8").trim();
  if (raw) {
    secrets = JSON.parse(raw);
  }
}
if (!secrets.tools || typeof secrets.tools !== "object") {
  secrets.tools = {};
}

const runtimeIdentity = trim(process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY) || "vicuna";
const stateRoot = trim(process.env.VICUNA_STATE_ROOT) || "/var/lib/vicuna";
const hostShellRoot = trim(process.env.VICUNA_HOST_SHELL_ROOT) || "/home/vicuna/home";

const upsert = (toolId, fields) => {
  const next = { ...(secrets.tools[toolId] || {}) };
  for (const [key, value] of Object.entries(fields)) {
    if (value !== undefined && value !== null && value !== "") {
      next[key] = value;
    }
  }
  if (Object.keys(next).length > 0) {
    secrets.tools[toolId] = next;
  }
};

upsert("ongoing_tasks", {
  task_dir: trim(process.env.VICUNA_ONGOING_TASKS_DIR) || `${stateRoot}/ongoing-tasks`,
  runner_script: trim(process.env.VICUNA_ONGOING_TASKS_RUNNER_SCRIPT),
  crontab_bin: trim(process.env.VICUNA_ONGOING_TASKS_CRONTAB_BIN),
  flock_bin: trim(process.env.VICUNA_ONGOING_TASKS_FLOCK_BIN),
  temp_dir: trim(process.env.VICUNA_ONGOING_TASKS_TMPDIR) || `${stateRoot}/ongoing-tasks/tmp`,
  runtime_url: trim(process.env.VICUNA_ONGOING_TASKS_RUNTIME_URL),
  runtime_model: trim(process.env.VICUNA_ONGOING_TASKS_RUNTIME_MODEL) || trim(process.env.VICUNA_DEEPSEEK_MODEL),
  runtime_api_key: trim(process.env.VICUNA_API_KEY),
  host_user: trim(process.env.VICUNA_ONGOING_TASKS_HOST_USER) || "vicuna",
});

upsert("parsed_documents", {
  docs_dir: trim(process.env.VICUNA_DOCS_DIR) || `${hostShellRoot}/docs`,
});

upsert("telegram_relay", {
  base_url:
    trim(process.env.VICUNA_TELEGRAM_RELAY_BASE_URL) ||
    trim(process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL) ||
    "http://127.0.0.1:8080",
  auth_token: trim(process.env.VICUNA_API_KEY),
  default_chat_scope: trim(process.env.VICUNA_TELEGRAM_DEFAULT_CHAT_SCOPE),
});

fs.mkdirSync(path.dirname(secretsPath), { recursive: true });
fs.writeFileSync(secretsPath, `${JSON.stringify(secrets, null, 2)}\n`, "utf8");
EOF

run_openclaw sync-runtime-catalog
