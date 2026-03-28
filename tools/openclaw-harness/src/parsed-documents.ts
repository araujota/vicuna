import { Buffer } from "node:buffer";
import fs from "node:fs";
import path from "node:path";

import { defaultDocsDir, defaultRepoRoot, loadToolSecrets } from "./config.js";

export type ParsedDocumentsInvocation = {
  query?: unknown;
  limit?: unknown;
  threshold?: unknown;
};

export type ParsedDocumentsConfig = {
  docsDir: string;
  defaultThreshold: number;
  shortQueryThreshold: number;
  maxResults: number;
};

export type ParsedDocumentSearchResult = {
  document_title: string;
  chunk_text: string;
  similarity: number;
  chunk_index: number;
  link_key?: string;
  source_path?: string;
};

export type ParsedDocumentsResponseEnvelope = {
  family: "parsed_documents";
  action: "search_chunks";
  ok: boolean;
  count?: number;
  items?: ParsedDocumentSearchResult[];
  threshold?: number;
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class ParsedDocumentsToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

type JsonRecord = Record<string, unknown>;

const DEFAULT_MAX_RESULTS = 5;
const MAX_RESULTS = 8;
const DEFAULT_THRESHOLD = 0.58;
const DEFAULT_SHORT_QUERY_THRESHOLD = 0.68;

function trimString(value: string | undefined | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function asRecord(value: unknown): JsonRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : undefined;
}

function requireQuery(payload: ParsedDocumentsInvocation): string {
  if (typeof payload.query !== "string" || payload.query.trim().length === 0) {
    throw new ParsedDocumentsToolError("missing_argument", "query is required");
  }
  return payload.query.trim();
}

function optionalInteger(value: unknown, fieldName: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new ParsedDocumentsToolError("invalid_argument", `${fieldName} must be a positive integer`, {
      field: fieldName,
      value,
    });
  }
  return value;
}

function optionalThreshold(value: unknown, fieldName: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new ParsedDocumentsToolError("invalid_argument", `${fieldName} must be a number in the range [0, 1]`, {
      field: fieldName,
      value,
    });
  }
  return value;
}

function queryTokenCount(query: string): number {
  return query.split(/\s+/).filter(Boolean).length;
}

function effectiveThreshold(query: string, override: number | undefined, config: ParsedDocumentsConfig): number {
  if (override !== undefined) {
    return override;
  }
  const trimmed = query.trim();
  if (queryTokenCount(trimmed) <= 3 || trimmed.length < 24) {
    return config.shortQueryThreshold;
  }
  return config.defaultThreshold;
}

function boundedLimit(requested: number | undefined, configuredMax: number): number {
  return Math.max(1, Math.min(requested ?? DEFAULT_MAX_RESULTS, configuredMax, MAX_RESULTS));
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/i)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function lexicalScore(query: string, chunkText: string): number {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) {
    return 0;
  }
  const chunkTokens = new Set(tokenize(chunkText));
  let matched = 0;
  for (const token of queryTokens) {
    if (chunkTokens.has(token)) {
      matched += 1;
    }
  }
  const tokenScore = matched / queryTokens.length;
  const phraseBonus = chunkText.toLowerCase().includes(query.toLowerCase()) ? 0.25 : 0;
  return Math.min(1, Number((tokenScore + phraseBonus).toFixed(4)));
}

type StoredParsedChunk = {
  chunk_index: number;
  contextual_text: string;
  source_text: string | undefined;
  document_title: string;
  link_key: string | undefined;
};

type StoredBundle = {
  sourcePath?: string;
  chunks: StoredParsedChunk[];
};

function readJsonFile(filePath: string): unknown {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readBundle(bundleDir: string): StoredBundle {
  const metadataPath = path.join(bundleDir, "metadata.json");
  const chunksPath = path.join(bundleDir, "chunks.json");
  const metadata = fs.existsSync(metadataPath) ? asRecord(readJsonFile(metadataPath)) : undefined;
  const sourcePath =
    trimString(typeof metadata?.source_path === "string" ? metadata.source_path : undefined) ??
    trimString(typeof metadata?.sourcePath === "string" ? metadata.sourcePath : undefined);
  if (!fs.existsSync(chunksPath)) {
    return { sourcePath, chunks: [] };
  }
  const rawChunks = readJsonFile(chunksPath);
  const chunksArray = Array.isArray(rawChunks) ? rawChunks : [];
  const chunks = chunksArray
    .map((entry, index) => {
      const record = asRecord(entry);
      if (!record) {
        return null;
      }
      const contextualText =
        trimString(typeof record.contextual_text === "string" ? record.contextual_text : undefined) ??
        trimString(typeof record.contextualText === "string" ? record.contextualText : undefined) ??
        trimString(typeof record.chunk_text === "string" ? record.chunk_text : undefined) ??
        trimString(typeof record.chunkText === "string" ? record.chunkText : undefined);
      if (!contextualText) {
        return null;
      }
      const chunkIndexRaw = record.chunk_index ?? record.chunkIndex ?? index;
      const chunkIndex =
        typeof chunkIndexRaw === "number" && Number.isInteger(chunkIndexRaw) && chunkIndexRaw >= 0
          ? chunkIndexRaw
          : index;
      const documentTitle =
        trimString(typeof record.document_title === "string" ? record.document_title : undefined) ??
        trimString(typeof record.documentTitle === "string" ? record.documentTitle : undefined) ??
        trimString(typeof metadata?.document_title === "string" ? metadata.document_title : undefined) ??
        trimString(typeof metadata?.documentTitle === "string" ? metadata.documentTitle : undefined) ??
        path.basename(bundleDir);
      return {
        chunk_index: chunkIndex,
        contextual_text: contextualText,
        source_text:
          trimString(typeof record.source_text === "string" ? record.source_text : undefined) ??
          trimString(typeof record.sourceText === "string" ? record.sourceText : undefined),
        document_title: documentTitle,
        link_key:
          trimString(typeof record.link_key === "string" ? record.link_key : undefined) ??
          trimString(typeof record.linkKey === "string" ? record.linkKey : undefined),
      } satisfies StoredParsedChunk;
    })
    .filter((entry): entry is StoredParsedChunk => entry !== null);
  return { sourcePath, chunks };
}

export function parseCliInvocation(argv: string[]): { payload: ParsedDocumentsInvocation; secretsPath: string } {
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
    throw new ParsedDocumentsToolError("missing_payload", "missing required --payload-base64 argument");
  }
  if (!secretsPath) {
    throw new ParsedDocumentsToolError("missing_secrets_path", "missing required --secrets-path argument");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new ParsedDocumentsToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new ParsedDocumentsToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }

  return {
    payload: payload as ParsedDocumentsInvocation,
    secretsPath,
  };
}

export function resolveParsedDocumentsConfig(secretsPath: string): ParsedDocumentsConfig {
  const secrets = loadToolSecrets(secretsPath);
  const configured = secrets.tools?.parsed_documents;
  const repoRoot = defaultRepoRoot();
  return {
    docsDir:
      trimString(configured?.docs_dir) ??
      trimString(process.env.VICUNA_DOCS_DIR) ??
      defaultDocsDir(repoRoot),
    defaultThreshold:
      typeof configured?.default_threshold === "number" && Number.isFinite(configured.default_threshold)
        ? configured.default_threshold
        : DEFAULT_THRESHOLD,
    shortQueryThreshold:
      typeof configured?.short_query_threshold === "number" && Number.isFinite(configured.short_query_threshold)
        ? configured.short_query_threshold
        : DEFAULT_SHORT_QUERY_THRESHOLD,
    maxResults:
      typeof configured?.max_results === "number" && Number.isInteger(configured.max_results)
        ? Math.max(1, configured.max_results)
        : DEFAULT_MAX_RESULTS,
  };
}

export async function handleParsedDocuments(
  payload: ParsedDocumentsInvocation,
  config: ParsedDocumentsConfig,
): Promise<ParsedDocumentsResponseEnvelope> {
  const query = requireQuery(payload);
  const limit = boundedLimit(optionalInteger(payload.limit, "limit"), config.maxResults);
  const threshold = effectiveThreshold(query, optionalThreshold(payload.threshold, "threshold"), config);
  if (!fs.existsSync(config.docsDir)) {
    return {
      family: "parsed_documents",
      action: "search_chunks",
      ok: true,
      count: 0,
      items: [],
      threshold,
    };
  }

  const bundleDirs = fs
    .readdirSync(config.docsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(config.docsDir, entry.name));

  const matches: ParsedDocumentSearchResult[] = [];
  for (const bundleDir of bundleDirs) {
    const bundle = readBundle(bundleDir);
    for (const chunk of bundle.chunks) {
      const similarity = lexicalScore(query, chunk.contextual_text);
      if (similarity < threshold) {
        continue;
      }
      matches.push({
        document_title: chunk.document_title,
        chunk_text: chunk.contextual_text,
        similarity,
        chunk_index: chunk.chunk_index,
        ...(chunk.link_key ? { link_key: chunk.link_key } : {}),
        ...(bundle.sourcePath ? { source_path: bundle.sourcePath } : {}),
      });
    }
  }

  matches.sort((lhs, rhs) => {
    if (rhs.similarity !== lhs.similarity) {
      return rhs.similarity - lhs.similarity;
    }
    if (lhs.document_title !== rhs.document_title) {
      return lhs.document_title.localeCompare(rhs.document_title);
    }
    return lhs.chunk_index - rhs.chunk_index;
  });

  const items = matches.slice(0, limit);
  return {
    family: "parsed_documents",
    action: "search_chunks",
    ok: true,
    count: items.length,
    items,
    threshold,
  };
}

export function errorEnvelope(error: unknown): ParsedDocumentsResponseEnvelope {
  if (error instanceof ParsedDocumentsToolError) {
    return {
      family: "parsed_documents",
      action: "search_chunks",
      ok: false,
      error: {
        kind: error.kind,
        message: error.message,
        ...(error.details ? { details: error.details } : {}),
      },
    };
  }

  return {
    family: "parsed_documents",
    action: "search_chunks",
    ok: false,
    error: {
      kind: "unexpected_error",
      message: error instanceof Error ? error.message : String(error),
    },
  };
}

export async function runParsedDocumentsCli(argv: string[]): Promise<void> {
  try {
    const { payload, secretsPath } = parseCliInvocation(argv);
    const config = resolveParsedDocumentsConfig(secretsPath);
    const response = await handleParsedDocuments(payload, config);
    process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(error), null, 2)}\n`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  void runParsedDocumentsCli(process.argv.slice(2));
}
