import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type OpenClawToolSecrets = {
  tools?: {
    tavily?: {
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
};

function moduleDir(): string {
  return path.dirname(fileURLToPath(import.meta.url));
}

export function defaultRepoRoot(): string {
  return path.resolve(moduleDir(), "../../..");
}

export function defaultPaths(repoRoot = defaultRepoRoot()): OpenClawPaths {
  const stateDir = path.join(repoRoot, ".cache", "vicuna");
  return {
    repoRoot,
    stateDir,
    secretsPath: path.join(stateDir, "openclaw-tool-secrets.json"),
    runtimeCatalogPath: path.join(stateDir, "openclaw-catalog.json"),
    tavilyWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "tavily-web-search")
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
