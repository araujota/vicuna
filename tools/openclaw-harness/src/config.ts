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
  telegramRuntimeToolsSnapshotPath: string;
  hardMemoryDir: string;
  skillsDir: string;
  hostShellRoot: string;
  hardMemoryWrapperPath: string;
  skillsWrapperPath: string;
  hostShellWrapperPath: string;
  telegramRelayWrapperPath: string;
};

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

export function defaultPaths(repoRoot = defaultRepoRoot()): OpenClawPaths {
  const stateDir = defaultStateRoot(repoRoot);
  return {
    repoRoot,
    stateDir,
    secretsPath:
      process.env.VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH?.trim() ||
      path.join(stateDir, "openclaw-tool-secrets.json"),
    runtimeCatalogPath:
      process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH?.trim() ||
      path.join(stateDir, "openclaw-catalog.json"),
    telegramRuntimeToolsSnapshotPath:
      process.env.VICUNA_TELEGRAM_RUNTIME_TOOLS_SNAPSHOT_PATH?.trim() ||
      path.join(stateDir, "telegram-runtime-tools.json"),
    hardMemoryDir: defaultHardMemoryDir(repoRoot),
    skillsDir: defaultSkillsDir(repoRoot),
    hostShellRoot: defaultHostShellRoot(repoRoot),
    hardMemoryWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "hard-memory-api"),
    skillsWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "skills-api"),
    hostShellWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "host-shell-api"),
    telegramRelayWrapperPath: path.join(repoRoot, "tools", "openclaw-harness", "bin", "telegram-relay-api"),
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
