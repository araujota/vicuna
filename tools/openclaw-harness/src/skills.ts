import fs from "node:fs";
import path from "node:path";
import { Buffer } from "node:buffer";

import { defaultPaths, loadToolSecrets } from "./config.js";
import { isCliEntrypoint } from "./cli-entrypoint.js";

export type SkillsInvocation = {
  action?: unknown;
  name?: unknown;
  content?: unknown;
  overwrite?: unknown;
};

export type SkillsConfig = {
  skillsDir: string;
};

export type SkillsResponseEnvelope = {
  family: "skills";
  action: "read" | "create" | "unknown";
  ok: boolean;
  name?: string;
  path?: string;
  content?: string;
  created?: boolean;
  updated?: boolean;
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class SkillsToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

function trimString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeSkillStem(value: string): string {
  const withoutExtension = value.toLowerCase().endsWith(".md") ? value.slice(0, -3) : value;
  return withoutExtension.trim().toLowerCase();
}

function safeSkillFilename(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  if (!normalized) {
    throw new SkillsToolError("invalid_argument", "skill name did not contain any safe filename characters");
  }
  return `${normalized}.md`;
}

function ensureSkillsDir(skillsDir: string): void {
  fs.mkdirSync(skillsDir, { recursive: true });
}

function listSkillMarkdownFiles(skillsDir: string): string[] {
  if (!fs.existsSync(skillsDir)) {
    return [];
  }
  return fs
    .readdirSync(skillsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
    .map((entry) => path.join(skillsDir, entry.name))
    .sort((left, right) => path.basename(left).localeCompare(path.basename(right)));
}

function resolveSkillPathByName(skillsDir: string, requestedName: string): string {
  const requestedStem = normalizeSkillStem(requestedName);
  const matches = listSkillMarkdownFiles(skillsDir).filter((filePath) => {
    const filename = path.basename(filePath);
    return normalizeSkillStem(filename) === requestedStem;
  });
  if (matches.length === 0) {
    throw new SkillsToolError("not_found", `skill not found: ${requestedName}`);
  }
  if (matches.length > 1) {
    throw new SkillsToolError("ambiguous_name", `multiple skills matched: ${requestedName}`, {
      matches: matches.map((item) => path.basename(item)),
    });
  }
  return matches[0];
}

function errorEnvelope(error: unknown, action: SkillsResponseEnvelope["action"]): SkillsResponseEnvelope {
  if (error instanceof SkillsToolError) {
    return {
      family: "skills",
      action,
      ok: false,
      error: {
        kind: error.kind,
        message: error.message,
        ...(error.details ? { details: error.details } : {}),
      },
    };
  }
  return {
    family: "skills",
    action,
    ok: false,
    error: {
      kind: "unhandled_error",
      message: error instanceof Error ? error.message : String(error),
    },
  };
}

export function parseCliInvocation(argv: string[]): { payload: SkillsInvocation; secretsPath: string } {
  let payloadBase64 = "";
  let secretsPath = "";
  for (const arg of argv) {
    if (arg.startsWith("--payload-base64=")) {
      payloadBase64 = arg.slice("--payload-base64=".length);
      continue;
    }
    if (arg.startsWith("--secrets-path=")) {
      secretsPath = arg.slice("--secrets-path=".length);
    }
  }
  if (!payloadBase64) {
    throw new SkillsToolError("missing_payload", "missing required --payload-base64 argument");
  }
  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new SkillsToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new SkillsToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }
  return {
    payload: payload as SkillsInvocation,
    secretsPath,
  };
}

export function resolveSkillsConfig(secretsPath: string): SkillsConfig {
  const secrets = loadToolSecrets(secretsPath);
  return {
    skillsDir: trimString(secrets.tools?.skills?.skills_dir) ?? defaultPaths().skillsDir,
  };
}

function requireAction(payload: SkillsInvocation): "read" | "create" {
  const action = trimString(payload.action)?.toLowerCase();
  if (action === "read" || action === "create") {
    return action;
  }
  if (trimString(payload.content)) {
    return "create";
  }
  if (trimString(payload.name)) {
    return "read";
  }
  throw new SkillsToolError("invalid_payload", "skills payload must include an explicit action or a readable name/content shape");
}

export async function handleSkills(payload: SkillsInvocation, config: SkillsConfig): Promise<SkillsResponseEnvelope> {
  const action = requireAction(payload);
  ensureSkillsDir(config.skillsDir);

  if (action === "read") {
    const name = trimString(payload.name);
    if (!name) {
      throw new SkillsToolError("missing_argument", "skill_read requires a non-empty name");
    }
    const filePath = resolveSkillPathByName(config.skillsDir, name);
    return {
      family: "skills",
      action,
      ok: true,
      name: path.basename(filePath, ".md"),
      path: filePath,
      content: fs.readFileSync(filePath, "utf8"),
    };
  }

  const name = trimString(payload.name);
  const content = trimString(payload.content);
  if (!name) {
    throw new SkillsToolError("missing_argument", "skill_create requires a non-empty name");
  }
  if (!content) {
    throw new SkillsToolError("missing_argument", "skill_create requires non-empty markdown content");
  }
  const overwrite = payload.overwrite === undefined ? false : Boolean(payload.overwrite);
  const filename = safeSkillFilename(name);
  const filePath = path.join(config.skillsDir, filename);
  const alreadyExists = fs.existsSync(filePath);
  if (alreadyExists && !overwrite) {
    throw new SkillsToolError("already_exists", `skill already exists: ${filename}`);
  }
  const tempPath = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  fs.writeFileSync(tempPath, `${content.trim()}\n`, "utf8");
  fs.renameSync(tempPath, filePath);
  return {
    family: "skills",
    action,
    ok: true,
    name: path.basename(filePath, ".md"),
    path: filePath,
    created: !alreadyExists,
    updated: alreadyExists,
  };
}

export async function runSkillsCli(argv: string[]): Promise<void> {
  try {
    const { payload, secretsPath } = parseCliInvocation(argv);
    const config = resolveSkillsConfig(secretsPath);
    const response = await handleSkills(payload, config);
    process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(error, "unknown"), null, 2)}\n`);
    process.exitCode = 1;
  }
}

if (isCliEntrypoint(import.meta.url, process.argv[1])) {
  await runSkillsCli(process.argv.slice(2));
}
