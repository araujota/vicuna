import { Buffer } from "node:buffer";
import { randomUUID } from "node:crypto";

import { loadToolSecrets } from "./config.js";

export type OngoingTaskUnit = "hours" | "days" | "weeks";

export type OngoingTasksInvocation = {
  action: string;
  [key: string]: unknown;
};

export type OngoingTaskFrequency = {
  interval: number;
  unit: OngoingTaskUnit;
};

export type OngoingTaskRecord = {
  task_id: string;
  task_text: string;
  frequency: OngoingTaskFrequency;
  created_at: string;
  updated_at: string;
  last_done_at: string | null;
  active: boolean;
};

export type OngoingTaskRegistry = {
  schema_version: number;
  updated_at: string;
  tasks: OngoingTaskRecord[];
};

export type OngoingTaskSummary = {
  task_id: string;
  task_text: string;
  frequency: OngoingTaskFrequency;
  last_done_at: string | null;
  next_due_at: string;
  due_now: boolean;
  active: boolean;
};

export type OngoingTasksConfig = {
  baseUrl: string;
  authToken?: string;
  containerTag: string;
  runtimeIdentity: string;
  registryKey: string;
  registryTitle: string;
  queryThreshold: number;
};

export type OngoingTasksResponseEnvelope = {
  family: "ongoing_tasks";
  action: string;
  ok: boolean;
  task?: OngoingTaskSummary;
  tasks?: OngoingTaskSummary[];
  count?: number;
  due_only?: boolean;
  deleted_task_id?: string;
  remaining_count?: number;
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class OngoingTasksToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

type JsonRecord = Record<string, unknown>;
type Clock = () => Date;

const DEFAULT_HARD_MEMORY_BASE_URL = "https://api.supermemory.ai";
const DEFAULT_RUNTIME_IDENTITY = "vicuna";
const DEFAULT_REGISTRY_KEY = "ongoing-tasks-registry";
const DEFAULT_REGISTRY_TITLE = "Ongoing task registry";
const REGISTRY_SCHEMA_VERSION = 1;

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

function emptyRegistry(now: string): OngoingTaskRegistry {
  return {
    schema_version: REGISTRY_SCHEMA_VERSION,
    updated_at: now,
    tasks: [],
  };
}

function currentIso(clock: Clock): string {
  return clock().toISOString();
}

function parseIsoTimestamp(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new OngoingTasksToolError("invalid_argument", `${fieldName} must be a non-empty ISO-8601 timestamp`);
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    throw new OngoingTasksToolError("invalid_argument", `${fieldName} must be a valid ISO-8601 timestamp`, {
      field: fieldName,
      value,
    });
  }
  return parsed.toISOString();
}

function parseTaskUnit(value: unknown, fieldName: string): OngoingTaskUnit {
  if (value !== "hours" && value !== "days" && value !== "weeks") {
    throw new OngoingTasksToolError(
      "invalid_argument",
      `${fieldName} must be one of hours, days, or weeks`,
      { field: fieldName, value }
    );
  }
  return value;
}

function parsePositiveInteger(value: unknown, fieldName: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new OngoingTasksToolError("invalid_argument", `${fieldName} must be an integer greater than or equal to 1`, {
      field: fieldName,
      value,
    });
  }
  return value;
}

function requireNonEmptyString(payload: OngoingTasksInvocation, fieldName: string): string {
  const value = payload[fieldName];
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new OngoingTasksToolError("missing_argument", `${fieldName} is required`);
  }
  return value.trim();
}

function optionalString(payload: OngoingTasksInvocation, fieldName: string): string | undefined {
  const value = payload[fieldName];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new OngoingTasksToolError("invalid_argument", `${fieldName} must be a string`);
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function optionalBoolean(payload: OngoingTasksInvocation, fieldName: string): boolean | undefined {
  const value = payload[fieldName];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new OngoingTasksToolError("invalid_argument", `${fieldName} must be a boolean`);
  }
  return value;
}

function optionalInteger(payload: OngoingTasksInvocation, fieldName: string): number | undefined {
  const value = payload[fieldName];
  if (value === undefined || value === null) {
    return undefined;
  }
  return parsePositiveInteger(value, fieldName);
}

function requireAction(payload: OngoingTasksInvocation): string {
  const action = optionalString(payload, "action");
  if (!action) {
    throw new OngoingTasksToolError("missing_action", "tool payload requires a non-empty action");
  }
  return action;
}

export function parseCliInvocation(argv: string[]): { payload: OngoingTasksInvocation; secretsPath: string } {
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
    throw new OngoingTasksToolError("missing_payload", "missing required --payload-base64 argument");
  }
  if (!secretsPath) {
    throw new OngoingTasksToolError("missing_secrets_path", "missing required --secrets-path argument");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new OngoingTasksToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new OngoingTasksToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }

  return {
    payload: payload as OngoingTasksInvocation,
    secretsPath,
  };
}

export function resolveOngoingTasksConfig(secretsPath: string): OngoingTasksConfig {
  const secrets = loadToolSecrets(secretsPath);
  const configured = secrets.tools?.ongoing_tasks;
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
    containerTag: trimString(configured?.container_tag) ?? runtimeIdentity,
    runtimeIdentity,
    registryKey: trimString(configured?.registry_key) ?? DEFAULT_REGISTRY_KEY,
    registryTitle: trimString(configured?.registry_title) ?? DEFAULT_REGISTRY_TITLE,
    queryThreshold:
      typeof configured?.query_threshold === "number" && Number.isFinite(configured.query_threshold)
        ? configured.query_threshold
        : 0,
  };
}

function frequencyWindowMs(frequency: OngoingTaskFrequency): number {
  switch (frequency.unit) {
    case "hours":
      return frequency.interval * 60 * 60 * 1000;
    case "days":
      return frequency.interval * 24 * 60 * 60 * 1000;
    case "weeks":
      return frequency.interval * 7 * 24 * 60 * 60 * 1000;
  }
}

function summarizeTask(task: OngoingTaskRecord, now: Date): OngoingTaskSummary {
  const nextDueAt =
    task.last_done_at === null
      ? task.created_at
      : new Date(new Date(task.last_done_at).getTime() + frequencyWindowMs(task.frequency)).toISOString();
  return {
    task_id: task.task_id,
    task_text: task.task_text,
    frequency: task.frequency,
    last_done_at: task.last_done_at,
    next_due_at: nextDueAt,
    due_now: task.active && new Date(nextDueAt).getTime() <= now.getTime(),
    active: task.active,
  };
}

function validateTaskRecord(record: unknown): OngoingTaskRecord {
  const task = asRecord(record);
  const frequency = asRecord(task?.frequency);
  if (!task || !frequency) {
    throw new OngoingTasksToolError("invalid_registry", "stored ongoing-task record was not an object");
  }

  const taskId = typeof task.task_id === "string" && task.task_id.trim().length > 0 ? task.task_id.trim() : undefined;
  const taskText =
    typeof task.task_text === "string" && task.task_text.trim().length > 0 ? task.task_text.trim() : undefined;
  if (!taskId || !taskText) {
    throw new OngoingTasksToolError("invalid_registry", "stored ongoing-task record was missing task_id or task_text");
  }

  return {
    task_id: taskId,
    task_text: taskText,
    frequency: {
      interval: parsePositiveInteger(frequency.interval, "frequency.interval"),
      unit: parseTaskUnit(frequency.unit, "frequency.unit"),
    },
    created_at: parseIsoTimestamp(task.created_at, "created_at"),
    updated_at: parseIsoTimestamp(task.updated_at, "updated_at"),
    last_done_at: task.last_done_at === null ? null : parseIsoTimestamp(task.last_done_at, "last_done_at"),
    active: typeof task.active === "boolean" ? task.active : true,
  };
}

function parseRegistryContent(rawContent: string): OngoingTaskRegistry {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawContent) as unknown;
  } catch (error) {
    throw new OngoingTasksToolError("invalid_registry", "stored ongoing-task registry was not valid JSON", {
      parse_error: (error as Error).message,
    });
  }

  const registry = asRecord(parsed);
  if (!registry || !Array.isArray(registry.tasks)) {
    throw new OngoingTasksToolError("invalid_registry", "stored ongoing-task registry was missing its tasks array");
  }

  return {
    schema_version:
      typeof registry.schema_version === "number" && Number.isInteger(registry.schema_version)
        ? registry.schema_version
        : REGISTRY_SCHEMA_VERSION,
    updated_at: parseIsoTimestamp(registry.updated_at, "updated_at"),
    tasks: registry.tasks.map((task) => validateTaskRecord(task)),
  };
}

async function hardMemoryRequest(
  config: OngoingTasksConfig,
  path: string,
  body: unknown
): Promise<JsonRecord> {
  if (!config.authToken) {
    throw new OngoingTasksToolError(
      "missing_auth_token",
      "missing ongoing-tasks hard-memory auth token in OpenClaw tool secrets or SUPERMEMORY_API_KEY"
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
    throw new OngoingTasksToolError("network_error", `failed to reach hard memory: ${(error as Error).message}`, {
      base_url: config.baseUrl,
      path,
    });
  }

  const responseText = await response.text();
  if (!response.ok) {
    throw new OngoingTasksToolError("http_error", `hard memory request failed with HTTP ${response.status}`, {
      status: response.status,
      body: responseText.slice(0, 400),
      path,
    });
  }

  try {
    return JSON.parse(responseText) as JsonRecord;
  } catch (error) {
    throw new OngoingTasksToolError("invalid_json", "hard memory returned non-JSON content", {
      body: responseText.slice(0, 400),
      parse_error: (error as Error).message,
      path,
    });
  }
}

export async function loadOngoingTaskRegistry(
  config: OngoingTasksConfig,
  clock: Clock = () => new Date()
): Promise<OngoingTaskRegistry> {
  const response = await hardMemoryRequest(config, "/v4/profile", {
    containerTag: config.containerTag,
    q: config.registryKey,
    threshold: config.queryThreshold,
  });
  const searchResults = asRecord(response.searchResults);
  const results = Array.isArray(searchResults?.results) ? searchResults.results : [];
  const matchingRegistries: OngoingTaskRegistry[] = [];

  for (const result of results) {
    const item = asRecord(result);
    const metadata = asRecord(item?.metadata);
    const metadataKey = typeof metadata?.key === "string" ? metadata.key : undefined;
    const title = typeof item?.title === "string" ? item.title : typeof metadata?.title === "string" ? metadata.title : undefined;
    if (metadataKey !== config.registryKey && title !== config.registryTitle) {
      continue;
    }

    const rawContent =
      typeof item?.memory === "string"
        ? item.memory
        : typeof item?.chunk === "string"
          ? item.chunk
          : typeof item?.content === "string"
            ? item.content
            : undefined;
    if (!rawContent) {
      throw new OngoingTasksToolError("invalid_registry", "stored ongoing-task registry was missing its content text");
    }

    matchingRegistries.push(parseRegistryContent(rawContent));
  }

  if (matchingRegistries.length === 0) {
    return emptyRegistry(currentIso(clock));
  }

  matchingRegistries.sort((lhs, rhs) => rhs.updated_at.localeCompare(lhs.updated_at));
  return matchingRegistries[0];
}

export async function saveOngoingTaskRegistry(
  config: OngoingTasksConfig,
  registry: OngoingTaskRegistry,
  clock: Clock = () => new Date()
): Promise<void> {
  const normalizedRegistry: OngoingTaskRegistry = {
    schema_version: REGISTRY_SCHEMA_VERSION,
    updated_at: currentIso(clock),
    tasks: registry.tasks,
  };

  await hardMemoryRequest(config, "/v4/memories", {
    containerTag: config.containerTag,
    memories: [{
      content: JSON.stringify(normalizedRegistry),
      metadata: {
        source: "vicuna",
        runtimeIdentity: config.runtimeIdentity,
        kind: "tool_observation",
        domain: "strategy",
        key: config.registryKey,
        title: config.registryTitle,
        tags: ["ongoing_tasks", "registry"],
        importance: 0.8,
        confidence: 1,
        gainBias: 0.3,
        allostaticRelevance: 0,
      },
    }],
  });
}

function findTaskIndex(registry: OngoingTaskRegistry, taskId: string): number {
  return registry.tasks.findIndex((task) => task.task_id === taskId);
}

function listTaskSummaries(
  registry: OngoingTaskRegistry,
  now: Date,
  dueOnly: boolean
): OngoingTaskSummary[] {
  return registry.tasks
    .map((task) => summarizeTask(task, now))
    .filter((task) => task.active)
    .filter((task) => !dueOnly || task.due_now)
    .sort((lhs, rhs) => lhs.next_due_at.localeCompare(rhs.next_due_at));
}

export async function handleOngoingTasks(
  payload: OngoingTasksInvocation,
  config: OngoingTasksConfig,
  clock: Clock = () => new Date()
): Promise<OngoingTasksResponseEnvelope> {
  const action = requireAction(payload);
  const now = clock();
  const nowIso = now.toISOString();
  const registry = await loadOngoingTaskRegistry(config, clock);

  if (action === "create") {
    const task: OngoingTaskRecord = {
      task_id: randomUUID(),
      task_text: requireNonEmptyString(payload, "task_text"),
      frequency: {
        interval: parsePositiveInteger(payload.interval, "interval"),
        unit: parseTaskUnit(payload.unit, "unit"),
      },
      created_at: nowIso,
      updated_at: nowIso,
      last_done_at: null,
      active: true,
    };
    registry.tasks.push(task);
    await saveOngoingTaskRegistry(config, registry, clock);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      task: summarizeTask(task, now),
    };
  }

  if (action === "get") {
    const taskId = optionalString(payload, "task_id");
    if (taskId) {
      const index = findTaskIndex(registry, taskId);
      if (index < 0) {
        throw new OngoingTasksToolError("lookup_no_match", `no ongoing task matched ${taskId}`, { task_id: taskId });
      }
      return {
        family: "ongoing_tasks",
        action,
        ok: true,
        task: summarizeTask(registry.tasks[index], now),
      };
    }

    const dueOnly = optionalBoolean(payload, "due_only") ?? false;
    const tasks = listTaskSummaries(registry, now, dueOnly);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      tasks,
      count: tasks.length,
      due_only: dueOnly,
    };
  }

  const taskId = requireNonEmptyString(payload, "task_id");
  const taskIndex = findTaskIndex(registry, taskId);
  if (taskIndex < 0) {
    throw new OngoingTasksToolError("lookup_no_match", `no ongoing task matched ${taskId}`, { task_id: taskId });
  }

  const task = registry.tasks[taskIndex];
  if (action === "edit") {
    const taskText = optionalString(payload, "task_text");
    const interval = optionalInteger(payload, "interval");
    const unit = payload.unit === undefined ? undefined : parseTaskUnit(payload.unit, "unit");
    const active = optionalBoolean(payload, "active");

    task.task_text = taskText ?? task.task_text;
    task.frequency = {
      interval: interval ?? task.frequency.interval,
      unit: unit ?? task.frequency.unit,
    };
    task.active = active ?? task.active;
    task.updated_at = nowIso;
    registry.tasks[taskIndex] = task;
    await saveOngoingTaskRegistry(config, registry, clock);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      task: summarizeTask(task, now),
    };
  }

  if (action === "complete") {
    const completedAt = payload.completed_at === undefined ? nowIso : parseIsoTimestamp(payload.completed_at, "completed_at");
    task.last_done_at = completedAt;
    task.updated_at = nowIso;
    registry.tasks[taskIndex] = task;
    await saveOngoingTaskRegistry(config, registry, clock);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      task: summarizeTask(task, now),
    };
  }

  if (action === "delete") {
    registry.tasks.splice(taskIndex, 1);
    await saveOngoingTaskRegistry(config, registry, clock);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      deleted_task_id: taskId,
      remaining_count: registry.tasks.length,
    };
  }

  throw new OngoingTasksToolError("unsupported_action", `unsupported ongoing-tasks action: ${action}`);
}

export function errorEnvelope(action: string, error: unknown): OngoingTasksResponseEnvelope {
  if (error instanceof OngoingTasksToolError) {
    return {
      family: "ongoing_tasks",
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
    family: "ongoing_tasks",
    action,
    ok: false,
    error: {
      kind: "unexpected_error",
      message: error instanceof Error ? error.message : String(error),
    },
  };
}

export async function runOngoingTasksCli(argv: string[], clock: Clock = () => new Date()): Promise<void> {
  let action = "unknown";
  try {
    const { payload, secretsPath } = parseCliInvocation(argv);
    action = requireAction(payload);
    const config = resolveOngoingTasksConfig(secretsPath);
    const response = await handleOngoingTasks(payload, config, clock);
    process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(action, error), null, 2)}\n`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  void runOngoingTasksCli(process.argv.slice(2));
}
