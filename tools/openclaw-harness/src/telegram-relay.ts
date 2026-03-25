import { Buffer } from "node:buffer";

import { loadToolSecrets } from "./config.js";

export type TelegramRelayInvocation = {
  text?: unknown;
  chat_scope?: unknown;
  reply_to_message_id?: unknown;
  intent?: unknown;
  dedupe_key?: unknown;
  urgency?: unknown;
};

export type TelegramRelayConfig = {
  baseUrl: string;
  authToken?: string;
  defaultChatScope?: string;
};

export type TelegramRelayResponseEnvelope = {
  family: "telegram";
  action: "relay";
  ok: boolean;
  sequence_number?: number;
  chat_scope?: string;
  deduplicated?: boolean;
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class TelegramRelayToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

const DEFAULT_VICUNA_BASE_URL = "http://127.0.0.1:8080";

function trimString(value: string | undefined | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function requireString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TelegramRelayToolError("missing_argument", `${fieldName} is required`);
  }
  return value.trim();
}

function optionalString(value: unknown, fieldName: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new TelegramRelayToolError("invalid_argument", `${fieldName} must be a string`);
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

function optionalPositiveInteger(value: unknown, fieldName: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new TelegramRelayToolError("invalid_argument", `${fieldName} must be a positive integer`, {
      field: fieldName,
      value,
    });
  }
  return value;
}

function optionalNonNegativeNumber(value: unknown, fieldName: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new TelegramRelayToolError("invalid_argument", `${fieldName} must be a non-negative number`, {
      field: fieldName,
      value,
    });
  }
  return value;
}

export function parseCliInvocation(argv: string[]): { payload: TelegramRelayInvocation; secretsPath: string } {
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
    throw new TelegramRelayToolError("missing_payload", "missing required --payload-base64 argument");
  }
  if (!secretsPath) {
    throw new TelegramRelayToolError("missing_secrets_path", "missing required --secrets-path argument");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new TelegramRelayToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new TelegramRelayToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }

  return {
    payload: payload as TelegramRelayInvocation,
    secretsPath,
  };
}

export function resolveTelegramRelayConfig(secretsPath: string): TelegramRelayConfig {
  const secrets = loadToolSecrets(secretsPath);
  const configured = secrets.tools?.telegram_relay;
  return {
    baseUrl: normalizeBaseUrl(
      trimString(configured?.base_url) ??
        trimString(process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL) ??
        DEFAULT_VICUNA_BASE_URL
    ),
    authToken: trimString(configured?.auth_token) ?? trimString(process.env.VICUNA_API_KEY),
    defaultChatScope:
      trimString(configured?.default_chat_scope) ?? trimString(process.env.TELEGRAM_DEFAULT_CHAT_SCOPE),
  };
}

function buildRelayRequest(payload: TelegramRelayInvocation, config: TelegramRelayConfig) {
  const text = requireString(payload.text, "text");
  const chatScope = optionalString(payload.chat_scope, "chat_scope") ?? config.defaultChatScope;
  if (!chatScope) {
    throw new TelegramRelayToolError(
      "missing_chat_scope",
      "telegram relay requires chat_scope or a configured default_chat_scope"
    );
  }

  const request: Record<string, unknown> = {
    kind: "message",
    text,
    chat_scope: chatScope,
  };
  const replyToMessageId = optionalPositiveInteger(payload.reply_to_message_id, "reply_to_message_id");
  if (replyToMessageId) {
    request.reply_to_message_id = replyToMessageId;
  }
  const intent = optionalString(payload.intent, "intent");
  if (intent) {
    request.intent = intent;
  }
  const dedupeKey = optionalString(payload.dedupe_key, "dedupe_key");
  if (dedupeKey) {
    request.dedupe_key = dedupeKey;
  }
  const urgency = optionalNonNegativeNumber(payload.urgency, "urgency");
  if (urgency !== undefined) {
    request.urgency = urgency;
  }
  return request;
}

export async function handleTelegramRelay(
  payload: TelegramRelayInvocation,
  config: TelegramRelayConfig
): Promise<TelegramRelayResponseEnvelope> {
  const request = buildRelayRequest(payload, config);
  const headers: Record<string, string> = {
    "content-type": "application/json",
  };
  if (config.authToken) {
    headers.authorization = `Bearer ${config.authToken}`;
  }

  const response = await fetch(`${config.baseUrl}/v1/telegram/outbox`, {
    method: "POST",
    headers,
    body: JSON.stringify(request),
  });

  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    throw new TelegramRelayToolError("server_error", `telegram relay returned non-JSON response: ${(error as Error).message}`);
  }

  const record = body && typeof body === "object" && !Array.isArray(body) ? (body as Record<string, unknown>) : {};
  if (!response.ok) {
    const errorRecord =
      record.error && typeof record.error === "object" && !Array.isArray(record.error)
        ? (record.error as Record<string, unknown>)
        : undefined;
    throw new TelegramRelayToolError(
      errorRecord && typeof errorRecord.type === "string" ? errorRecord.type : "server_error",
      errorRecord && typeof errorRecord.message === "string"
        ? errorRecord.message
        : `telegram relay request failed with status ${response.status}`,
      { status: response.status }
    );
  }

  if (typeof record.sequence_number !== "number" || !Number.isInteger(record.sequence_number) || record.sequence_number < 1) {
    throw new TelegramRelayToolError("server_error", "telegram relay response did not include a valid sequence_number");
  }
  if (typeof record.chat_scope !== "string" || record.chat_scope.trim().length === 0) {
    throw new TelegramRelayToolError("server_error", "telegram relay response did not include a valid chat_scope");
  }

  return {
    family: "telegram",
    action: "relay",
    ok: true,
    sequence_number: record.sequence_number,
    chat_scope: record.chat_scope.trim(),
    deduplicated: record.deduplicated === true,
  };
}

export function errorEnvelope(error: unknown): TelegramRelayResponseEnvelope {
  if (error instanceof TelegramRelayToolError) {
    return {
      family: "telegram",
      action: "relay",
      ok: false,
      error: {
        kind: error.kind,
        message: error.message,
        details: error.details,
      },
    };
  }
  return {
    family: "telegram",
    action: "relay",
    ok: false,
    error: {
      kind: "server_error",
      message: error instanceof Error ? error.message : "unknown telegram relay error",
    },
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const { payload, secretsPath } = parseCliInvocation(process.argv.slice(2));
    const config = resolveTelegramRelayConfig(secretsPath);
    const result = await handleTelegramRelay(payload, config);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(error))}\n`);
    process.exitCode = 1;
  }
}
