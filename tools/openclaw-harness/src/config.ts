import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type OpenClawToolSecrets = {
  tools?: {
    hard_memory?: {
      memory_dir?: string;
      runtime_identity?: string;
    };
    skills?: {
      skills_dir?: string;
    };
    tavily?: {
      api_key?: string;
    };
    radarr?: {
      base_url?: string;
      api_key?: string;
    };
    sonarr?: {
      base_url?: string;
      api_key?: string;
    };
    chaptarr?: {
      base_url?: string;
      api_key?: string;
      legal_importer_url?: string;
      legal_importer_wait_ms?: number;
      legal_importer_poll_ms?: number;
    };
    ongoing_tasks?: {
      task_dir?: string;
      runner_script?: string;
      crontab_bin?: string;
      flock_bin?: string;
      temp_dir?: string;
      runtime_url?: string;
      runtime_model?: string;
      runtime_api_key?: string;
      host_user?: string;
    };
    parsed_documents?: {
      docs_dir?: string;
      default_threshold?: number;
      short_query_threshold?: number;
      max_results?: number;
    };
    telegram_relay?: {
      base_url?: string;
      auth_token?: string;
      default_chat_scope?: string;
    };
  };
};

export type OpenClawPaths = {
  repoRoot: string;
  stateDir: string;
  secretsPath: string;
  runtimeCatalogPath: string;
  hardMemoryDir: string;
  skillsDir: string;
  ongoingTasksDir: string;
  docsDir: string;
  hostShellRoot: string;
  hardMemoryWrapperPath: string;
  skillsWrapperPath: string;
  hostShellWrapperPath: string;
  tavilyWrapperPath: string;
  radarrWrapperPath: string;
  sonarrWrapperPath: string;
  chaptarrWrapperPath: string;
  ongoingTasksWrapperPath: string;
  parsedDocumentsWrapperPath: string;
  telegramRelayWrapperPath: string;
};

export type OpenClawServarrToolId = "radarr" | "sonarr";
export type OpenClawMediaToolId = OpenClawServarrToolId | "chaptarr";

export const DEFAULT_RADARR_BASE_URL = "http://10.0.0.218:7878";
export const DEFAULT_SONARR_BASE_URL = "http://10.0.0.218:8989";
export const DEFAULT_CHAPTARR_BASE_URL = "http://10.0.0.218:8789";

function moduleDir(): string {
  return path.dirname(fileURLToPath(import.meta.url));
}

export function defaultRepoRoot(): string {
  return path.resolve(moduleDir(), "../../..");
}

export function defaultHostShellRoot(repoRoot = defaultRepoRoot()): string {
  const configured = process.env.VICUNA_HOST_SHELL_ROOT?.trim();
  if (configured) {
    return configured;
  }

  const hostPath = "/home/vicuna/home";
  if (fs.existsSync(hostPath)) {
    return hostPath;
  }

  return path.join(repoRoot, ".cache", "vicuna", "host-shell-home");
}

export function defaultStateRoot(repoRoot = defaultRepoRoot()): string {
  const configured = process.env.VICUNA_STATE_ROOT?.trim();
  if (configured) {
    return configured;
  }

  if ((process.env.VICUNA_SYSTEMD_SCOPE?.trim() || "") === "system" || fs.existsSync("/var/lib/vicuna")) {
    return "/var/lib/vicuna";
  }

  return path.join(repoRoot, ".cache", "vicuna");
}

export function defaultHardMemoryDir(repoRoot = defaultRepoRoot()): string {
  const configured = process.env.VICUNA_HARD_MEMORY_DIR?.trim();
  if (configured) {
    return configured;
  }

  return path.join(defaultHostShellRoot(repoRoot), "memories");
}

export function defaultSkillsDir(repoRoot = defaultRepoRoot()): string {
  const configured = process.env.VICUNA_SKILLS_DIR?.trim();
  if (configured) {
    return configured;
  }

  return path.join(defaultHostShellRoot(repoRoot), "skills");
}

export function defaultOngoingTasksDir(repoRoot = defaultRepoRoot()): string {
  const configured = process.env.VICUNA_ONGOING_TASKS_DIR?.trim();
  if (configured) {
    return configured;
  }

  return path.join(defaultStateRoot(repoRoot), "ongoing-tasks");
}

export function defaultDocsDir(repoRoot = defaultRepoRoot()): string {
  const configured = process.env.VICUNA_DOCS_DIR?.trim();
  if (configured) {
    return configured;
  }

  return path.join(defaultHostShellRoot(repoRoot), "docs");
}

export function defaultPaths(repoRoot = defaultRepoRoot()): OpenClawPaths {
  const stateDir = defaultStateRoot(repoRoot);
  const secretsPath =
    process.env.VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH?.trim() ||
    path.join(defaultStateRoot(repoRoot), "openclaw-tool-secrets.json");
  const runtimeCatalogPath =
    process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH?.trim() ||
    path.join(defaultStateRoot(repoRoot), "openclaw-catalog.json");
  return {
    repoRoot,
    stateDir,
    secretsPath,
    runtimeCatalogPath,
    hardMemoryDir: defaultHardMemoryDir(repoRoot),
    skillsDir: defaultSkillsDir(repoRoot),
    ongoingTasksDir: defaultOngoingTasksDir(repoRoot),
    docsDir: defaultDocsDir(repoRoot),
    hostShellRoot: defaultHostShellRoot(repoRoot),
    hardMemoryWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "hard-memory-api"),
    skillsWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "skills-api"),
    hostShellWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "host-shell-api"),
    tavilyWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "tavily-web-search"),
    radarrWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "radarr-api"),
    sonarrWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "sonarr-api"),
    chaptarrWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "chaptarr-api"),
    ongoingTasksWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "ongoing-tasks-api"),
    parsedDocumentsWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "parsed-documents-search"),
    telegramRelayWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "telegram-relay-api")
  };
}

export function ensureParentDir(filePath: string): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

export function loadToolSecrets(secretsPath: string): OpenClawToolSecrets {
  if (!fs.existsSync(secretsPath)) {
    return {};
  }
  const raw = fs.readFileSync(secretsPath, "utf8");
  if (!raw.trim()) {
    return {};
  }
  return JSON.parse(raw) as OpenClawToolSecrets;
}

export function saveToolSecrets(secretsPath: string, secrets: OpenClawToolSecrets): void {
  ensureParentDir(secretsPath);
  fs.writeFileSync(secretsPath, `${JSON.stringify(secrets, null, 2)}\n`, "utf8");
}

export function upsertTavilyApiKey(secrets: OpenClawToolSecrets, apiKey: string): OpenClawToolSecrets {
  const next: OpenClawToolSecrets = {
    ...secrets,
    tools: {
      ...secrets.tools,
      tavily: {
        ...secrets.tools?.tavily,
        api_key: apiKey
      }
    }
  };
  return next;
}

export function upsertApiToolConfig(
  secrets: OpenClawToolSecrets,
  toolId: OpenClawMediaToolId,
  apiKey: string,
  baseUrl?: string
): OpenClawToolSecrets {
  const existing =
    toolId === "radarr"
      ? secrets.tools?.radarr
      : toolId === "sonarr"
        ? secrets.tools?.sonarr
        : secrets.tools?.chaptarr;
  const nextConfig = {
    ...existing,
    api_key: apiKey,
    ...(baseUrl ? { base_url: baseUrl } : {})
  };
  return {
    ...secrets,
    tools: {
      ...secrets.tools,
      [toolId]: nextConfig
    }
  };
}

export function upsertServarrConfig(
  secrets: OpenClawToolSecrets,
  toolId: OpenClawServarrToolId,
  apiKey: string,
  baseUrl?: string
): OpenClawToolSecrets {
  return upsertApiToolConfig(secrets, toolId, apiKey, baseUrl);
}
