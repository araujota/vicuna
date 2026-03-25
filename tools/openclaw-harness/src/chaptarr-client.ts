import { Buffer } from "node:buffer";

import {
  DEFAULT_CHAPTARR_BASE_URL,
  loadToolSecrets,
} from "./config.js";

export type ChaptarrInvocation = {
  action: string;
  [key: string]: unknown;
};

export type ChaptarrServiceConfig = {
  service: "chaptarr";
  baseUrl: string;
  apiKey?: string;
  legalImporterBaseUrl?: string;
  legalImporterWaitMs?: number;
  legalImporterPollMs?: number;
};

export type ChaptarrResponseEnvelope = {
  service: "chaptarr";
  action: string;
  base_url: string;
  ok: boolean;
  request?: {
    method: string;
    path: string;
    query?: Record<string, unknown>;
  };
  data?: unknown;
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class ChaptarrToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

export type ChaptarrCliContext = {
  payload: ChaptarrInvocation;
  config: ChaptarrServiceConfig;
  service: "chaptarr";
};

export const FIXED_CHAPTARR_MEDIA_TYPE = "ebook";
export const FIXED_CHAPTARR_ROOT_FOLDER_PATH = "/books";

const VALID_MEDIA_TYPES = new Set([FIXED_CHAPTARR_MEDIA_TYPE]);

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function deriveLegalImporterBaseUrl(chaptarrBaseUrl: string): string | undefined {
  try {
    const url = new URL(chaptarrBaseUrl);
    url.port = "5000";
    url.pathname = "";
    url.search = "";
    url.hash = "";
    return normalizeBaseUrl(url.toString());
  } catch {
    return undefined;
  }
}

export function resolveChaptarrConfig(secretsPath: string): ChaptarrServiceConfig {
  const secrets = loadToolSecrets(secretsPath);
  const baseUrl = normalizeBaseUrl(secrets.tools?.chaptarr?.base_url?.trim() || DEFAULT_CHAPTARR_BASE_URL);
  const apiKey = secrets.tools?.chaptarr?.api_key?.trim();
  const configuredLegalImporterUrl = secrets.tools?.chaptarr?.legal_importer_url?.trim();
  const legalImporterBaseUrl = configuredLegalImporterUrl
    ? normalizeBaseUrl(configuredLegalImporterUrl)
    : deriveLegalImporterBaseUrl(baseUrl);
  const rawWaitMs = secrets.tools?.chaptarr?.legal_importer_wait_ms;
  const rawPollMs = secrets.tools?.chaptarr?.legal_importer_poll_ms;
  const legalImporterWaitMs =
    typeof rawWaitMs === "number" && Number.isFinite(rawWaitMs) && rawWaitMs > 0
      ? Math.trunc(rawWaitMs)
      : 120000;
  const legalImporterPollMs =
    typeof rawPollMs === "number" && Number.isFinite(rawPollMs) && rawPollMs > 0
      ? Math.trunc(rawPollMs)
      : 5000;
  return {
    service: "chaptarr",
    baseUrl,
    apiKey: apiKey && apiKey.length > 0 ? apiKey : undefined
    ,
    legalImporterBaseUrl: legalImporterBaseUrl && legalImporterBaseUrl.length > 0 ? legalImporterBaseUrl : undefined,
    legalImporterWaitMs,
    legalImporterPollMs,
  };
}

export function parseCliInvocation(argv: string[]): { payload: ChaptarrInvocation; secretsPath: string } {
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
    throw new ChaptarrToolError("missing_payload", "missing required --payload-base64 argument");
  }
  if (!secretsPath) {
    throw new ChaptarrToolError("missing_secrets_path", "missing required --secrets-path argument");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new ChaptarrToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new ChaptarrToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }

  return {
    payload: payload as ChaptarrInvocation,
    secretsPath
  };
}

export function requireAction(payload: ChaptarrInvocation): string {
  const action = typeof payload.action === "string" ? payload.action.trim() : "";
  if (!action) {
    throw new ChaptarrToolError("missing_action", "tool payload requires a non-empty action");
  }
  return action;
}

export function canonicalizeChaptarrAction(action: string): string {
  return action === "inspect" ? "list_downloaded_books" : action;
}

export function optionalString(payload: ChaptarrInvocation, key: string): string | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new ChaptarrToolError("invalid_argument", `${key} must be a string`);
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function optionalInteger(payload: ChaptarrInvocation, key: string): number | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ChaptarrToolError("invalid_argument", `${key} must be an integer`);
  }
  return value > 0 ? value : undefined;
}

export function optionalBoolean(payload: ChaptarrInvocation, key: string): boolean | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ChaptarrToolError("invalid_argument", `${key} must be a boolean`);
  }
  return value;
}

export function optionalIntegerArray(payload: ChaptarrInvocation, key: string): number[] | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "number" || !Number.isInteger(entry))) {
    throw new ChaptarrToolError("invalid_argument", `${key} must be an array of integers`);
  }
  return value as number[];
}

export function optionalStringArray(payload: ChaptarrInvocation, key: string): string[] | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string")) {
    throw new ChaptarrToolError("invalid_argument", `${key} must be an array of strings`);
  }
  const items = value.map((entry) => entry.trim()).filter((entry) => entry.length > 0);
  return items.length > 0 ? items : undefined;
}

export function requireString(payload: ChaptarrInvocation, key: string, helpText?: string): string {
  const value = optionalString(payload, key);
  if (!value) {
    throw new ChaptarrToolError("missing_argument", helpText ?? `${key} is required`);
  }
  return value;
}

export function requireInteger(payload: ChaptarrInvocation, key: string, helpText?: string): number {
  const value = optionalInteger(payload, key);
  if (value === undefined) {
    throw new ChaptarrToolError("missing_argument", helpText ?? `${key} is required`);
  }
  return value;
}

export function resolveMediaTypes(payload: ChaptarrInvocation): string[] {
  const mediaType = optionalString(payload, "media_type");
  if (!mediaType) {
    return [FIXED_CHAPTARR_MEDIA_TYPE];
  }
  if (!VALID_MEDIA_TYPES.has(mediaType)) {
    throw new ChaptarrToolError("invalid_argument", "media_type must be ebook");
  }
  return [mediaType];
}

export function successEnvelope(
  action: string,
  baseUrl: string,
  method: string,
  path: string,
  query: Record<string, unknown> | undefined,
  data: unknown
): ChaptarrResponseEnvelope {
  return {
    service: "chaptarr",
    action,
    base_url: baseUrl,
    ok: true,
    request: {
      method,
      path,
      ...(query && Object.keys(query).length > 0 ? { query } : {})
    },
    data
  };
}

export function errorEnvelope(action: string, baseUrl: string, error: unknown): ChaptarrResponseEnvelope {
  if (error instanceof ChaptarrToolError) {
    return {
      service: "chaptarr",
      action,
      base_url: baseUrl,
      ok: false,
      error: {
        kind: error.kind,
        message: error.message,
        ...(error.details ? { details: error.details } : {})
      }
    };
  }
  const message = error instanceof Error ? error.message : String(error);
  return {
    service: "chaptarr",
    action,
    base_url: baseUrl,
    ok: false,
    error: {
      kind: "unexpected_error",
      message
    }
  };
}

function appendQuery(url: URL, query: Record<string, unknown> | undefined): void {
  if (!query) {
    return;
  }
  for (const [key, rawValue] of Object.entries(query)) {
    if (rawValue === undefined || rawValue === null || rawValue === "") {
      continue;
    }
    if (Array.isArray(rawValue)) {
      for (const entry of rawValue) {
        url.searchParams.append(key, String(entry));
      }
      continue;
    }
    url.searchParams.set(key, String(rawValue));
  }
}

export async function chaptarrRequestJson(
  config: ChaptarrServiceConfig,
  method: string,
  path: string,
  query?: Record<string, unknown>,
  body?: unknown
): Promise<unknown> {
  if (!config.apiKey) {
    throw new ChaptarrToolError(
      "missing_api_key",
      "missing Chaptarr API key in OpenClaw tool secrets",
      { service: config.service }
    );
  }

  const url = new URL(path, `${config.baseUrl}/`);
  appendQuery(url, query);

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: {
        accept: "application/json",
        "content-type": "application/json",
        "X-Api-Key": config.apiKey
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) })
    });
  } catch (error) {
    throw new ChaptarrToolError(
      "network_error",
      `failed to reach Chaptarr: ${(error as Error).message}`,
      { base_url: config.baseUrl }
    );
  }

  const responseText = await response.text();
  if (!response.ok) {
    const kind = response.status === 401 || response.status === 403 ? "authorization_failed" : "http_error";
    throw new ChaptarrToolError(
      kind,
      `Chaptarr request failed with HTTP ${response.status}`,
      {
        status: response.status,
        body: responseText.slice(0, 400),
        path
      }
    );
  }

  if (!responseText.trim()) {
    return null;
  }

  try {
    return JSON.parse(responseText) as unknown;
  } catch (error) {
    throw new ChaptarrToolError(
      "invalid_json",
      "Chaptarr returned non-JSON content",
      { body: responseText.slice(0, 400), parse_error: (error as Error).message }
    );
  }
}

export function chooseLookupCandidate<T extends Record<string, unknown>>(
  candidates: T[],
  matchers: Array<(candidate: T) => boolean>
): T {
  for (const matcher of matchers) {
    const match = candidates.find(matcher);
    if (match) {
      return match;
    }
  }
  if (candidates.length === 0) {
    throw new ChaptarrToolError("lookup_no_match", "lookup returned no candidates");
  }
  return candidates[0];
}

export async function runChaptarrCli(
  argv: string[],
  handler: (context: ChaptarrCliContext) => Promise<ChaptarrResponseEnvelope>
): Promise<void> {
  let payload: ChaptarrInvocation = { action: "unknown" };
  let config: ChaptarrServiceConfig = {
    service: "chaptarr",
    baseUrl: DEFAULT_CHAPTARR_BASE_URL,
    apiKey: undefined,
    legalImporterBaseUrl: deriveLegalImporterBaseUrl(DEFAULT_CHAPTARR_BASE_URL),
    legalImporterWaitMs: 120000,
    legalImporterPollMs: 5000,
  };
  let action = "unknown";

  try {
    const parsed = parseCliInvocation(argv);
    payload = parsed.payload;
    action = requireAction(payload);
    config = resolveChaptarrConfig(parsed.secretsPath);
    const response = await handler({ payload, config, service: "chaptarr" });
    process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(action, config.baseUrl, error), null, 2)}\n`);
  }
}
