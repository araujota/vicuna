import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type OpenClawToolSecrets = {
  tools?: {
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
    };
  };
};

export type OpenClawPaths = {
  repoRoot: string;
  stateDir: string;
  secretsPath: string;
  runtimeCatalogPath: string;
  tavilyWrapperPath: string;
  radarrWrapperPath: string;
  sonarrWrapperPath: string;
  chaptarrWrapperPath: string;
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

export function defaultPaths(repoRoot = defaultRepoRoot()): OpenClawPaths {
  const stateDir = path.join(repoRoot, ".cache", "vicuna");
  const runtimeCatalogPath =
    process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH?.trim() ||
    path.join(stateDir, "openclaw-catalog.json");
  return {
    repoRoot,
    stateDir,
    secretsPath: path.join(stateDir, "openclaw-tool-secrets.json"),
    runtimeCatalogPath,
    tavilyWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "tavily-web-search"),
    radarrWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "radarr-api"),
    sonarrWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "sonarr-api"),
    chaptarrWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "chaptarr-api")
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
