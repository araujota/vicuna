import { Buffer } from "node:buffer";

import { loadToolSecrets } from "./config.js";

export type ParsedDocumentsInvocation = {
  query?: unknown;
  limit?: unknown;
  threshold?: unknown;
};

export type ParsedDocumentsConfig = {
  baseUrl: string;
  authToken?: string;
  containerTag: string;
  runtimeIdentity: string;
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

const DEFAULT_HARD_MEMORY_BASE_URL = "https://api.supermemory.ai";
const DEFAULT_CONTAINER_TAG = "vicuna-telegram-documents";
const DEFAULT_RUNTIME_IDENTITY = "vicuna";
const DEFAULT_MAX_RESULTS = 5;
const MAX_RESULTS = 8;
const DEFAULT_THRESHOLD = 0.58;
const DEFAULT_SHORT_QUERY_THRESHOLD = 0.68;

function trimString(value: string | undefined | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
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

function effectiveThreshold(
  query: string,
  override: number | undefined,
  config: ParsedDocumentsConfig
): number {
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

function searchFilters() {
  return {
    AND: [
      { key: "contentKind", value: "parsed_chunk", filterType: "metadata" },
      { key: "source", value: "telegram_bridge", filterType: "metadata" },
    ],
  };
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
  const runtimeIdentity =
    trimString(configured?.runtime_identity) ??
    trimString(process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY) ??
    DEFAULT_RUNTIME_IDENTITY;
  return {
    baseUrl: normalizeBaseUrl(
      trimString(configured?.base_url) ??
        trimString(process.env.SUPERMEMORY_BASE_URL) ??
        DEFAULT_HARD_MEMORY_BASE_URL
    ),
    authToken: trimString(configured?.auth_token) ?? trimString(process.env.SUPERMEMORY_API_KEY),
    containerTag:
      trimString(configured?.container_tag) ??
      trimString(process.env.TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG) ??
      DEFAULT_CONTAINER_TAG,
    runtimeIdentity,
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

async function hardMemoryRequest(
  config: ParsedDocumentsConfig,
  path: string,
  body: unknown
): Promise<JsonRecord> {
  if (!config.authToken) {
    throw new ParsedDocumentsToolError(
      "missing_auth_token",
      "missing parsed-documents auth token in OpenClaw tool secrets or SUPERMEMORY_API_KEY"
    );
  }

  let response: Response;
  try {
    response = await fetch(new URL(path, `${config.baseUrl}/`), {
      method: "POST",
      headers: {
        accept: "application/json",
        "content-type": "application/json",
        Authorization: `Bearer ${config.authToken}`,
        "x-supermemory-api-key": config.authToken,
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new ParsedDocumentsToolError("network_error", `failed to reach parsed-documents backend: ${(error as Error).message}`, {
      base_url: config.baseUrl,
      path,
    });
  }

  const responseText = await response.text();
  if (!response.ok) {
    throw new ParsedDocumentsToolError("http_error", `parsed-documents request failed with HTTP ${response.status}`, {
      status: response.status,
      body: responseText.slice(0, 400),
      path,
    });
  }

  try {
    return JSON.parse(responseText) as JsonRecord;
  } catch (error) {
    throw new ParsedDocumentsToolError("invalid_json", "parsed-documents backend returned non-JSON content", {
      body: responseText.slice(0, 400),
      parse_error: (error as Error).message,
      path,
    });
  }
}

function projectSearchResult(result: unknown, threshold: number): ParsedDocumentSearchResult | null {
  const record = asRecord(result);
  const metadata = asRecord(record?.metadata);
  if (metadata?.contentKind !== "parsed_chunk" || metadata?.source !== "telegram_bridge") {
    return null;
  }
  const similarity = typeof record?.similarity === "number" && Number.isFinite(record.similarity)
    ? record.similarity
    : NaN;
  if (!Number.isFinite(similarity) || similarity < threshold) {
    return null;
  }

  const documentTitle =
    trimString(typeof metadata?.documentTitle === "string" ? metadata.documentTitle : undefined) ??
    trimString(typeof metadata?.telegramFileName === "string" ? metadata.telegramFileName : undefined);
  const chunkText =
    trimString(typeof record?.memory === "string" ? record.memory : undefined) ??
    trimString(typeof record?.chunk === "string" ? record.chunk : undefined);
  if (!documentTitle || !chunkText) {
    return null;
  }

  return {
    document_title: documentTitle,
    chunk_text: chunkText,
    similarity,
    chunk_index:
      typeof metadata?.chunkIndex === "number" && Number.isInteger(metadata.chunkIndex)
        ? metadata.chunkIndex
        : 0,
    link_key: trimString(typeof metadata?.linkKey === "string" ? metadata.linkKey : undefined),
  };
}

function isSearchResult(value: ParsedDocumentSearchResult | null): value is ParsedDocumentSearchResult {
  return value !== null;
}

export async function handleParsedDocuments(
  payload: ParsedDocumentsInvocation,
  config: ParsedDocumentsConfig
): Promise<ParsedDocumentsResponseEnvelope> {
  const query = requireQuery(payload);
  const requestedLimit = optionalInteger(payload.limit, "limit");
  const thresholdOverride = optionalThreshold(payload.threshold, "threshold");
  const threshold = effectiveThreshold(query, thresholdOverride, config);
  const limit = boundedLimit(requestedLimit, config.maxResults);

  const response = await hardMemoryRequest(config, "/v4/search", {
    q: query,
    containerTag: config.containerTag,
    filters: searchFilters(),
    limit: Math.max(limit * 2, limit + 2),
    rerank: true,
    rewriteQuery: queryTokenCount(query) >= 4,
    searchMode: "memories",
    threshold,
  });
  const results = Array.isArray(response.results) ? response.results : [];
  const items = results
    .map((result) => projectSearchResult(result, threshold))
    .filter(isSearchResult)
    .slice(0, limit);

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
