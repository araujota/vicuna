import { Buffer } from "node:buffer";

import {
  DEFAULT_RADARR_BASE_URL,
  DEFAULT_SONARR_BASE_URL,
  loadToolSecrets,
  type OpenClawServarrToolId,
} from "./config.js";

export type ServarrService = OpenClawServarrToolId;

export type ServarrInvocation = {
  action: string;
  [key: string]: unknown;
};

export type ServarrServiceConfig = {
  service: ServarrService;
  baseUrl: string;
  apiKey?: string;
};

export type ServarrResponseEnvelope = {
  service: ServarrService;
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

export class ServarrToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

export type ServarrCliContext = {
  payload: ServarrInvocation;
  config: ServarrServiceConfig;
  service: ServarrService;
};

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function defaultBaseUrlForService(service: ServarrService): string {
  return service === "radarr" ? DEFAULT_RADARR_BASE_URL : DEFAULT_SONARR_BASE_URL;
}

export function resolveServarrConfig(service: ServarrService, secretsPath: string): ServarrServiceConfig {
  const secrets = loadToolSecrets(secretsPath);
  const serviceSecrets = service === "radarr" ? secrets.tools?.radarr : secrets.tools?.sonarr;
  const baseUrl = normalizeBaseUrl(serviceSecrets?.base_url?.trim() || defaultBaseUrlForService(service));
  const apiKey = serviceSecrets?.api_key?.trim();
  return {
    service,
    baseUrl,
    apiKey: apiKey && apiKey.length > 0 ? apiKey : undefined
  };
}

export function parseCliInvocation(argv: string[]): { payload: ServarrInvocation; secretsPath: string } {
  let payloadBase64 = "";
  let secretsPath = "";
  for (const arg of argv) {
    if (arg.startsWith("--payload-base64=")) {
      payloadBase64 = arg.slice("--payload-base64=".length);
      continue;
    }
    if (arg.startsWith("--secrets-path=")) {
      secretsPath = arg.slice("--secrets-path=".length);
      continue;
    }
  }

  if (!payloadBase64) {
    throw new ServarrToolError("missing_payload", "missing required --payload-base64 argument");
  }
  if (!secretsPath) {
    throw new ServarrToolError("missing_secrets_path", "missing required --secrets-path argument");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new ServarrToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new ServarrToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }

  return {
    payload: payload as ServarrInvocation,
    secretsPath
  };
}

export function requireAction(payload: ServarrInvocation): string {
  const action = typeof payload.action === "string" ? payload.action.trim() : "";
  if (!action) {
    throw new ServarrToolError("missing_action", "tool payload requires a non-empty action");
  }
  return action;
}

export function optionalString(payload: ServarrInvocation, key: string): string | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new ServarrToolError("invalid_argument", `${key} must be a string`);
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function optionalInteger(payload: ServarrInvocation, key: string): number | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ServarrToolError("invalid_argument", `${key} must be an integer`);
  }
  return value;
}

export function optionalBoolean(payload: ServarrInvocation, key: string): boolean | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ServarrToolError("invalid_argument", `${key} must be a boolean`);
  }
  return value;
}

export function optionalIntegerArray(payload: ServarrInvocation, key: string): number[] | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "number" || !Number.isInteger(entry))) {
    throw new ServarrToolError("invalid_argument", `${key} must be an array of integers`);
  }
  return value as number[];
}

export function requireString(payload: ServarrInvocation, key: string, helpText?: string): string {
  const value = optionalString(payload, key);
  if (!value) {
    throw new ServarrToolError("missing_argument", helpText ?? `${key} is required`);
  }
  return value;
}

export function requireInteger(payload: ServarrInvocation, key: string, helpText?: string): number {
  const value = optionalInteger(payload, key);
  if (value === undefined) {
    throw new ServarrToolError("missing_argument", helpText ?? `${key} is required`);
  }
  return value;
}

export function successEnvelope(
  service: ServarrService,
  action: string,
  baseUrl: string,
  method: string,
  path: string,
  query: Record<string, unknown> | undefined,
  data: unknown
): ServarrResponseEnvelope {
  return {
    service,
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

export function errorEnvelope(
  service: ServarrService,
  action: string,
  baseUrl: string,
  error: unknown
): ServarrResponseEnvelope {
  if (error instanceof ServarrToolError) {
    return {
      service,
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
    service,
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

export async function servarrRequestJson(
  config: ServarrServiceConfig,
  method: string,
  path: string,
  query?: Record<string, unknown>,
  body?: unknown
): Promise<unknown> {
  if (!config.apiKey) {
    throw new ServarrToolError(
      "missing_api_key",
      `missing ${config.service} API key in OpenClaw tool secrets`,
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
    throw new ServarrToolError(
      "network_error",
      `failed to reach ${config.service}: ${(error as Error).message}`,
      { service: config.service, base_url: config.baseUrl }
    );
  }

  const responseText = await response.text();
  if (!response.ok) {
    const kind = response.status === 401 || response.status === 403 ? "authorization_failed" : "http_error";
    throw new ServarrToolError(
      kind,
      `${config.service} request failed with HTTP ${response.status}`,
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
    throw new ServarrToolError(
      "invalid_json",
      `${config.service} returned non-JSON content`,
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
    throw new ServarrToolError("lookup_no_match", "lookup returned no candidates");
  }
  return candidates[0];
}

export async function runServarrCli(
  service: ServarrService,
  argv: string[],
  handler: (context: ServarrCliContext) => Promise<ServarrResponseEnvelope>
): Promise<void> {
  let payload: ServarrInvocation = { action: "unknown" };
  let config: ServarrServiceConfig = {
    service,
    baseUrl: defaultBaseUrlForService(service),
    apiKey: undefined
  };
  let action = "unknown";

  try {
    const parsed = parseCliInvocation(argv);
    payload = parsed.payload;
    action = requireAction(payload);
    config = resolveServarrConfig(service, parsed.secretsPath);
    const response = await handler({ payload, config, service });
    process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(service, action, config.baseUrl, error), null, 2)}\n`);
  }
}
