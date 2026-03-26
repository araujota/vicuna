#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# shellcheck disable=SC1091
source "$REPO_ROOT/tools/ops/runtime-env.sh"

OPENCLAW_INDEX_JS="${VICUNA_OPENCLAW_INDEX_JS:-$REPO_ROOT/tools/openclaw-harness/dist/index.js}"

if [[ ! -f "$OPENCLAW_INDEX_JS" ]]; then
  printf '[vicuna-openclaw-sync] skip: %s is missing\n' "$OPENCLAW_INDEX_JS" >&2
  exit 0
fi

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

const supermemoryBaseUrl =
  trim(process.env.SUPERMEMORY_BASE_URL) ||
  trim(process.env.VICUNA_ONGOING_TASKS_BASE_URL) ||
  trim(process.env.VICUNA_PARSED_DOCUMENTS_BASE_URL);
const supermemoryAuth =
  trim(process.env.SUPERMEMORY_API_KEY) ||
  trim(process.env.VICUNA_ONGOING_TASKS_AUTH_TOKEN) ||
  trim(process.env.VICUNA_PARSED_DOCUMENTS_AUTH_TOKEN);
const runtimeIdentity = trim(process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY) || "vicuna";
const ongoingTasksContainerTag = trim(process.env.VICUNA_ONGOING_TASKS_CONTAINER_TAG) || `${runtimeIdentity}-ongoing-tasks`;
const parsedDocumentsContainerTag =
  trim(process.env.TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG) || `${runtimeIdentity}-telegram-documents`;

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
  base_url: supermemoryBaseUrl,
  auth_token: supermemoryAuth,
  container_tag: ongoingTasksContainerTag,
  runtime_identity: trim(process.env.VICUNA_ONGOING_TASKS_RUNTIME_IDENTITY) || runtimeIdentity,
  registry_key: trim(process.env.VICUNA_ONGOING_TASKS_REGISTRY_KEY),
  registry_title: trim(process.env.VICUNA_ONGOING_TASKS_REGISTRY_TITLE),
});

upsert("parsed_documents", {
  base_url: supermemoryBaseUrl,
  auth_token: supermemoryAuth,
  container_tag: parsedDocumentsContainerTag,
  runtime_identity: runtimeIdentity,
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
