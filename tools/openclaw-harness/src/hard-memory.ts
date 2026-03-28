import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { Buffer } from "node:buffer";

import { defaultPaths, loadToolSecrets } from "./config.js";

export type HardMemoryInvocation = {
  query?: unknown;
  limit?: unknown;
  domain?: unknown;
  memories?: unknown;
  containerTag?: unknown;
};

export type HardMemoryConfig = {
  memoryDir: string;
  runtimeIdentity: string;
};

export type HardMemoryQueryResult = {
  record_id: string;
  title?: string;
  key?: string;
  kind?: string;
  domain?: string;
  tags: string[];
  score: number;
  excerpt: string;
  content: string;
  path: string;
  updated_at: string;
};

export type HardMemoryResponseEnvelope = {
  family: "hard_memory";
  action: "read" | "write" | "unknown";
  ok: boolean;
  count?: number;
  written?: number;
  created?: number;
  updated?: number;
  results?: HardMemoryQueryResult[];
  files?: Array<{ record_id: string; path: string }>;
  warnings?: string[];
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

type JsonRecord = Record<string, unknown>;

type StoredMemoryDocument = {
  recordId: string;
  title?: string;
  key?: string;
  kind?: string;
  domain?: string;
  tags: string[];
  importance?: number;
  confidence?: number;
  gainBias?: number;
  allostaticRelevance?: number;
  isStatic?: boolean;
  runtimeIdentity?: string;
  containerTag?: string;
  createdAt: string;
  updatedAt: string;
  content: string;
  path: string;
};

type WriteMemoryInput = {
  content: string;
  title?: string;
  key?: string;
  kind?: string;
  domain?: string;
  tags: string[];
  importance?: number;
  confidence?: number;
  gainBias?: number;
  allostaticRelevance?: number;
  isStatic?: boolean;
};

export class HardMemoryToolError extends Error {
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

function asRecord(value: unknown): JsonRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : undefined;
}

function normalizeTags(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return Array.from(new Set(value.map((item) => trimString(item)).filter((item): item is string => Boolean(item))));
}

function optionalNumber(value: unknown, field: string): number | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new HardMemoryToolError("invalid_argument", `${field} must be a finite number`);
  }
  return value;
}

function optionalBoolean(value: unknown, field: string): boolean | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new HardMemoryToolError("invalid_argument", `${field} must be a boolean`);
  }
  return value;
}

function nowIso(): string {
  return new Date().toISOString();
}

function slugify(value: string): string {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
  return normalized || "memory";
}

function parseScalar(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.startsWith("\"") || trimmed === "true" || trimmed === "false" || trimmed === "null" || /^-?\d+(\.\d+)?$/.test(trimmed)) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return trimmed;
    }
  }
  return trimmed;
}

function formatScalar(value: string | number | boolean | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function fileStemForMemory(memory: WriteMemoryInput): string {
  const stable = trimString(memory.key);
  if (stable) {
    return slugify(stable);
  }
  const title = trimString(memory.title);
  if (title) {
    return `${Date.now()}-${slugify(title)}`;
  }
  return `${Date.now()}-${randomUUID().slice(0, 8)}`;
}

function parseWriteInput(value: unknown): WriteMemoryInput {
  const record = asRecord(value);
  if (!record) {
    throw new HardMemoryToolError("invalid_argument", "each memory must be an object");
  }
  const content = trimString(record.content);
  if (!content) {
    throw new HardMemoryToolError("missing_argument", "each memory requires non-empty content");
  }
  return {
    content,
    title: trimString(record.title),
    key: trimString(record.key),
    kind: trimString(record.kind),
    domain: trimString(record.domain),
    tags: normalizeTags(record.tags),
    importance: optionalNumber(record.importance, "importance"),
    confidence: optionalNumber(record.confidence, "confidence"),
    gainBias: optionalNumber(record.gainBias, "gainBias"),
    allostaticRelevance: optionalNumber(record.allostaticRelevance, "allostaticRelevance"),
    isStatic: optionalBoolean(record.isStatic, "isStatic"),
  };
}

function parsePositiveLimit(value: unknown): number {
  if (value === undefined || value === null) {
    return 5;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new HardMemoryToolError("invalid_argument", "limit must be a positive integer");
  }
  return Math.min(value, 20);
}

function requireAction(payload: HardMemoryInvocation): "read" | "write" {
  if (Array.isArray(payload.memories)) {
    return "write";
  }
  if (typeof payload.query === "string") {
    return "read";
  }
  throw new HardMemoryToolError("invalid_payload", "hard_memory payload must include either memories[] or query");
}

function ensureMemoryDir(memoryDir: string): void {
  fs.mkdirSync(memoryDir, { recursive: true });
}

function serializeDocument(document: StoredMemoryDocument): string {
  const lines: string[] = ["---"];
  const scalarEntries: Array<[string, string | number | boolean | undefined]> = [
    ["record_id", document.recordId],
    ["title", document.title],
    ["key", document.key],
    ["kind", document.kind],
    ["domain", document.domain],
    ["importance", document.importance],
    ["confidence", document.confidence],
    ["gain_bias", document.gainBias],
    ["allostatic_relevance", document.allostaticRelevance],
    ["is_static", document.isStatic],
    ["runtime_identity", document.runtimeIdentity],
    ["container_tag", document.containerTag],
    ["created_at", document.createdAt],
    ["updated_at", document.updatedAt],
  ];
  for (const [key, value] of scalarEntries) {
    const formatted = formatScalar(value);
    if (formatted !== undefined) {
      lines.push(`${key}: ${formatted}`);
    }
  }
  lines.push("tags:");
  for (const tag of document.tags) {
    lines.push(`  - ${JSON.stringify(tag)}`);
  }
  lines.push("---", "", document.content.trim(), "");
  return lines.join("\n");
}

function parseDocument(filePath: string): StoredMemoryDocument {
  const raw = fs.readFileSync(filePath, "utf8");
  if (!raw.startsWith("---\n")) {
    throw new HardMemoryToolError("invalid_document", "markdown memory is missing frontmatter", { path: filePath });
  }
  const end = raw.indexOf("\n---\n", 4);
  if (end < 0) {
    throw new HardMemoryToolError("invalid_document", "markdown memory frontmatter is not closed", { path: filePath });
  }
  const frontmatterLines = raw.slice(4, end).split("\n");
  const metadata: Record<string, unknown> = {};
  let activeArrayKey = "";
  for (const line of frontmatterLines) {
    if (!line.trim()) {
      continue;
    }
    if (line.startsWith("  - ")) {
      if (!activeArrayKey) {
        throw new HardMemoryToolError("invalid_document", "frontmatter list item had no active key", { path: filePath });
      }
      const current = Array.isArray(metadata[activeArrayKey]) ? (metadata[activeArrayKey] as unknown[]) : [];
      current.push(parseScalar(line.slice(4)));
      metadata[activeArrayKey] = current;
      continue;
    }
    const separator = line.indexOf(":");
    if (separator < 0) {
      throw new HardMemoryToolError("invalid_document", "frontmatter line was missing ':'", { path: filePath, line });
    }
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    if (!value) {
      metadata[key] = [];
      activeArrayKey = key;
      continue;
    }
    metadata[key] = parseScalar(value);
    activeArrayKey = "";
  }
  const createdAt = trimString(metadata.created_at) ?? nowIso();
  const updatedAt = trimString(metadata.updated_at) ?? createdAt;
  return {
    recordId: trimString(metadata.record_id) ?? path.basename(filePath, ".md"),
    title: trimString(metadata.title),
    key: trimString(metadata.key),
    kind: trimString(metadata.kind),
    domain: trimString(metadata.domain),
    tags: normalizeTags(metadata.tags),
    importance: typeof metadata.importance === "number" ? metadata.importance : undefined,
    confidence: typeof metadata.confidence === "number" ? metadata.confidence : undefined,
    gainBias: typeof metadata.gain_bias === "number" ? metadata.gain_bias : undefined,
    allostaticRelevance: typeof metadata.allostatic_relevance === "number" ? metadata.allostatic_relevance : undefined,
    isStatic: typeof metadata.is_static === "boolean" ? metadata.is_static : undefined,
    runtimeIdentity: trimString(metadata.runtime_identity),
    containerTag: trimString(metadata.container_tag),
    createdAt,
    updatedAt,
    content: raw.slice(end + 5).trim(),
    path: filePath,
  };
}

function writeDocument(filePath: string, document: StoredMemoryDocument): void {
  const tempPath = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  fs.writeFileSync(tempPath, serializeDocument(document), "utf8");
  fs.renameSync(tempPath, filePath);
}

function excerptForQuery(content: string, query: string): string {
  const normalized = content.trim();
  if (!normalized) {
    return "";
  }
  const lowerContent = normalized.toLowerCase();
  const lowerQuery = query.toLowerCase();
  const index = lowerContent.indexOf(lowerQuery);
  if (index < 0) {
    return normalized.slice(0, 200);
  }
  return normalized.slice(Math.max(0, index - 60), Math.min(normalized.length, index + lowerQuery.length + 120));
}

function tokenize(value: string): string[] {
  return value.toLowerCase().split(/[^a-z0-9]+/g).filter(Boolean);
}

function lexicalScore(document: StoredMemoryDocument, query: string): number {
  const lowerQuery = query.toLowerCase();
  const tokens = tokenize(query);
  const title = (document.title ?? "").toLowerCase();
  const key = (document.key ?? "").toLowerCase();
  const kind = (document.kind ?? "").toLowerCase();
  const domain = (document.domain ?? "").toLowerCase();
  const content = document.content.toLowerCase();
  const tags = document.tags.map((tag) => tag.toLowerCase());

  let score = 0;
  if (title.includes(lowerQuery)) score += 8;
  if (key.includes(lowerQuery)) score += 8;
  if (kind.includes(lowerQuery)) score += 5;
  if (domain.includes(lowerQuery)) score += 5;
  if (content.includes(lowerQuery)) score += 6;
  for (const token of tokens) {
    if (title.includes(token)) score += 2;
    if (key.includes(token)) score += 2;
    if (kind.includes(token)) score += 1.5;
    if (domain.includes(token)) score += 1.5;
    if (tags.includes(token)) score += 2;
    if (content.includes(token)) score += 1;
  }
  return score;
}

function listMarkdownFiles(memoryDir: string): string[] {
  if (!fs.existsSync(memoryDir)) {
    return [];
  }
  return fs
    .readdirSync(memoryDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
    .map((entry) => path.join(memoryDir, entry.name));
}

export function parseCliInvocation(argv: string[]): { payload: HardMemoryInvocation; secretsPath: string } {
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
    throw new HardMemoryToolError("missing_payload", "missing required --payload-base64 argument");
  }
  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new HardMemoryToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new HardMemoryToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }
  return {
    payload: payload as HardMemoryInvocation,
    secretsPath,
  };
}

export function resolveHardMemoryConfig(secretsPath: string): HardMemoryConfig {
  const secrets = loadToolSecrets(secretsPath);
  const configured = secrets.tools?.hard_memory;
  return {
    memoryDir: trimString(configured?.memory_dir) ?? defaultPaths().hardMemoryDir,
    runtimeIdentity:
      trimString(configured?.runtime_identity) ??
      trimString(process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY) ??
      "vicuna",
  };
}

export async function handleHardMemory(payload: HardMemoryInvocation, config: HardMemoryConfig): Promise<HardMemoryResponseEnvelope> {
  const action = requireAction(payload);
  ensureMemoryDir(config.memoryDir);

  if (action === "write") {
    const warnings: string[] = [];
    const files: Array<{ record_id: string; path: string }> = [];
    const containerTag = trimString(payload.containerTag);
    const memories = Array.isArray(payload.memories) ? payload.memories : [];
    if (memories.length === 0) {
      throw new HardMemoryToolError("missing_argument", "hard_memory_write requires at least one memory");
    }
    let created = 0;
    let updated = 0;
    for (const item of memories) {
      const parsed = parseWriteInput(item);
      const stem = fileStemForMemory(parsed);
      const filePath = path.join(config.memoryDir, `${stem}.md`);
      let existing: StoredMemoryDocument | undefined;
      if (fs.existsSync(filePath)) {
        try {
          existing = parseDocument(filePath);
        } catch (error) {
          warnings.push(error instanceof Error ? error.message : String(error));
        }
      }
      const document: StoredMemoryDocument = {
        recordId: existing?.recordId ?? parsed.key ?? stem,
        title: parsed.title,
        key: parsed.key,
        kind: parsed.kind,
        domain: parsed.domain,
        tags: parsed.tags,
        importance: parsed.importance,
        confidence: parsed.confidence,
        gainBias: parsed.gainBias,
        allostaticRelevance: parsed.allostaticRelevance,
        isStatic: parsed.isStatic,
        runtimeIdentity: config.runtimeIdentity,
        containerTag,
        createdAt: existing?.createdAt ?? nowIso(),
        updatedAt: nowIso(),
        content: parsed.content,
        path: filePath,
      };
      writeDocument(filePath, document);
      files.push({ record_id: document.recordId, path: filePath });
      if (existing) {
        updated += 1;
      } else {
        created += 1;
      }
    }
    return {
      family: "hard_memory",
      action,
      ok: true,
      written: files.length,
      created,
      updated,
      files,
      warnings,
    };
  }

  const query = trimString(payload.query);
  if (!query) {
    throw new HardMemoryToolError("missing_argument", "hard_memory_read requires a non-empty query");
  }
  const limit = parsePositiveLimit(payload.limit);
  const domainFilter = trimString(payload.domain)?.toLowerCase();
  const warnings: string[] = [];
  const results = listMarkdownFiles(config.memoryDir)
    .flatMap((filePath) => {
      try {
        const document = parseDocument(filePath);
        if (domainFilter && (document.domain ?? "").toLowerCase() !== domainFilter) {
          return [];
        }
        const score = lexicalScore(document, query);
        if (score <= 0) {
          return [];
        }
        return [{
          record_id: document.recordId,
          ...(document.title ? { title: document.title } : {}),
          ...(document.key ? { key: document.key } : {}),
          ...(document.kind ? { kind: document.kind } : {}),
          ...(document.domain ? { domain: document.domain } : {}),
          tags: document.tags,
          score,
          excerpt: excerptForQuery(document.content, query),
          content: document.content,
          path: filePath,
          updated_at: document.updatedAt,
        } satisfies HardMemoryQueryResult];
      } catch (error) {
        warnings.push(error instanceof Error ? error.message : String(error));
        return [];
      }
    })
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      return right.updated_at.localeCompare(left.updated_at);
    })
    .slice(0, limit);

  return {
    family: "hard_memory",
    action,
    ok: true,
    count: results.length,
    results,
    warnings,
  };
}

export function errorEnvelope(action: "read" | "write" | "unknown", error: unknown): HardMemoryResponseEnvelope {
  if (error instanceof HardMemoryToolError) {
    return {
      family: "hard_memory",
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
    family: "hard_memory",
    action,
    ok: false,
    error: {
      kind: "unexpected_error",
      message: error instanceof Error ? error.message : String(error),
    },
  };
}

export async function runHardMemoryCli(argv: string[]): Promise<void> {
  let action: "read" | "write" | "unknown" = "unknown";
  try {
    const { payload, secretsPath } = parseCliInvocation(argv);
    action = requireAction(payload);
    const config = resolveHardMemoryConfig(secretsPath);
    const response = await handleHardMemory(payload, config);
    process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(action, error), null, 2)}\n`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  void runHardMemoryCli(process.argv.slice(2));
}
