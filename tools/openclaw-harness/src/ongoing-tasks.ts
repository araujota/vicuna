import { Buffer } from "node:buffer";
import { execFile as execFileCallback } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

import { defaultOngoingTasksDir, defaultRepoRoot, loadToolSecrets } from "./config.js";

const execFile = promisify(execFileCallback);

export type OngoingTaskUnit = "minutes" | "hours" | "days" | "weeks";

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
  schedule_expression: string;
  lock_path: string;
  log_path: string;
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
  schedule_expression: string;
};

export type OngoingTasksConfig = {
  tasksDir: string;
  runnerScript: string;
  crontabBin: string;
  flockBin: string;
  tempDir: string;
  runtimeUrl: string;
  runtimeModel: string;
  runtimeApiKey?: string;
  hostUser: string;
  managedCrontabPath?: string;
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
  installed?: boolean;
  removed?: boolean;
  schedule_expression?: string;
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
type ExecFileLike = (
  command: string,
  args: string[],
  options?: { env?: NodeJS.ProcessEnv }
) => Promise<{ stdout: string; stderr: string }>;
type FetchLike = typeof fetch;

const REGISTRY_SCHEMA_VERSION = 2;
const DEFAULT_HOST_USER = "vicuna";
const DEFAULT_CRONTAB_BIN = "/usr/bin/crontab";
const DEFAULT_FLOCK_BIN = "/usr/bin/flock";
const DEFAULT_RUNTIME_URL = "http://127.0.0.1:8080/v1/chat/completions";
const DEFAULT_RUNTIME_MODEL = "deepseek-chat";

function trimString(value: string | undefined | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function asRecord(value: unknown): JsonRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : undefined;
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
  if (value !== "minutes" && value !== "hours" && value !== "days" && value !== "weeks") {
    throw new OngoingTasksToolError(
      "invalid_argument",
      `${fieldName} must be one of minutes, hours, days, or weeks`,
      { field: fieldName, value },
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

function registryPath(config: OngoingTasksConfig): string {
  return path.join(config.tasksDir, "registry.json");
}

function lockDir(config: OngoingTasksConfig): string {
  return path.join(config.tasksDir, "locks");
}

function logDir(config: OngoingTasksConfig): string {
  return path.join(config.tasksDir, "logs");
}

function ensureDir(dirPath: string): void {
  fs.mkdirSync(dirPath, { recursive: true });
}

function ensureTaskDirs(config: OngoingTasksConfig): void {
  ensureDir(config.tasksDir);
  ensureDir(lockDir(config));
  ensureDir(logDir(config));
  ensureDir(config.tempDir);
}

function emptyRegistry(nowIso: string): OngoingTaskRegistry {
  return {
    schema_version: REGISTRY_SCHEMA_VERSION,
    updated_at: nowIso,
    tasks: [],
  };
}

function loadRegistry(config: OngoingTasksConfig, clock: Clock): OngoingTaskRegistry {
  ensureTaskDirs(config);
  const filePath = registryPath(config);
  if (!fs.existsSync(filePath)) {
    return emptyRegistry(currentIso(clock));
  }
  const raw = fs.readFileSync(filePath, "utf8").trim();
  if (!raw) {
    return emptyRegistry(currentIso(clock));
  }
  const parsed = JSON.parse(raw) as JsonRecord;
  const schemaVersion = typeof parsed.schema_version === "number" ? parsed.schema_version : REGISTRY_SCHEMA_VERSION;
  const tasksRaw = Array.isArray(parsed.tasks) ? parsed.tasks : [];
  const tasks: OngoingTaskRecord[] = tasksRaw.map((entry) => {
    const record = asRecord(entry);
    if (!record) {
      throw new OngoingTasksToolError("invalid_registry", "ongoing task registry contains a non-object task entry");
    }
    const frequencyRecord = asRecord(record.frequency);
    if (!frequencyRecord) {
      throw new OngoingTasksToolError("invalid_registry", "ongoing task registry contains an invalid frequency object");
    }
    return {
      task_id: String(record.task_id ?? "").trim(),
      task_text: String(record.task_text ?? "").trim(),
      frequency: {
        interval: parsePositiveInteger(frequencyRecord.interval, "frequency.interval"),
        unit: parseTaskUnit(frequencyRecord.unit, "frequency.unit"),
      },
      created_at: parseIsoTimestamp(record.created_at, "created_at"),
      updated_at: parseIsoTimestamp(record.updated_at, "updated_at"),
      last_done_at: record.last_done_at == null ? null : parseIsoTimestamp(record.last_done_at, "last_done_at"),
      active: record.active === false ? false : true,
      schedule_expression: String(record.schedule_expression ?? "").trim() || tickExpressionForUnit(parseTaskUnit(frequencyRecord.unit, "frequency.unit")),
      lock_path: String(record.lock_path ?? "").trim(),
      log_path: String(record.log_path ?? "").trim(),
    };
  }).filter((record) => record.task_id && record.task_text);

  return {
    schema_version: schemaVersion,
    updated_at: parseIsoTimestamp(parsed.updated_at ?? currentIso(clock), "updated_at"),
    tasks,
  };
}

function saveRegistry(config: OngoingTasksConfig, registry: OngoingTaskRegistry, clock: Clock): void {
  ensureTaskDirs(config);
  registry.updated_at = currentIso(clock);
  fs.writeFileSync(registryPath(config), `${JSON.stringify(registry, null, 2)}\n`, "utf8");
}

function frequencyWindowMs(frequency: OngoingTaskFrequency): number {
  const unitMs =
    frequency.unit === "minutes"
      ? 60_000
      : frequency.unit === "hours"
        ? 60 * 60_000
        : frequency.unit === "days"
          ? 24 * 60 * 60_000
          : 7 * 24 * 60 * 60_000;
  return frequency.interval * unitMs;
}

function dueAnchorMs(task: OngoingTaskRecord): number {
  return Date.parse(task.last_done_at ?? task.created_at);
}

function summarizeTask(task: OngoingTaskRecord, now: Date): OngoingTaskSummary {
  const windowMs = frequencyWindowMs(task.frequency);
  const nextDueMs = dueAnchorMs(task) + windowMs;
  return {
    task_id: task.task_id,
    task_text: task.task_text,
    frequency: task.frequency,
    last_done_at: task.last_done_at,
    next_due_at: new Date(nextDueMs).toISOString(),
    due_now: task.active && now.getTime() >= nextDueMs,
    active: task.active,
    schedule_expression: task.schedule_expression,
  };
}

function listTaskSummaries(registry: OngoingTaskRegistry, now: Date, dueOnly: boolean): OngoingTaskSummary[] {
  return registry.tasks
    .map((task) => summarizeTask(task, now))
    .filter((task) => !dueOnly || task.due_now);
}

function findTaskIndex(registry: OngoingTaskRegistry, taskId: string): number {
  return registry.tasks.findIndex((task) => task.task_id === taskId);
}

function tickExpressionForUnit(unit: OngoingTaskUnit): string {
  switch (unit) {
    case "minutes":
      return "* * * * *";
    case "hours":
      return "0 * * * *";
    case "days":
      return "0 0 * * *";
    case "weeks":
      return "0 0 * * 0";
  }
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\"'\"'`)}'`;
}

function markerStart(taskId: string): string {
  return `# >>> VICUNA MANAGED TASK ${taskId} >>>`;
}

function markerEnd(taskId: string): string {
  return `# <<< VICUNA MANAGED TASK ${taskId} <<<`;
}

function buildCronEntry(task: OngoingTaskRecord, config: OngoingTasksConfig): string[] {
  const command = [
    shellQuote(config.runnerScript),
    "--tasks-dir",
    shellQuote(config.tasksDir),
    "--task-id",
    shellQuote(task.task_id),
  ].join(" ");
  const redirect = `>> ${shellQuote(task.log_path)} 2>&1`;
  return [
    markerStart(task.task_id),
    `${task.schedule_expression} ${command} ${redirect}`,
    markerEnd(task.task_id),
  ];
}

function stripManagedBlock(lines: string[], taskId: string): string[] {
  const start = markerStart(taskId);
  const end = markerEnd(taskId);
  const output: string[] = [];
  let skipping = false;
  for (const line of lines) {
    if (line.trim() === start) {
      skipping = true;
      continue;
    }
    if (skipping && line.trim() === end) {
      skipping = false;
      continue;
    }
    if (!skipping) {
      output.push(line);
    }
  }
  return output;
}

async function readInstalledCrontab(config: OngoingTasksConfig, execFileImpl: ExecFileLike): Promise<string[]> {
  if (config.managedCrontabPath) {
    if (!fs.existsSync(config.managedCrontabPath)) {
      return [];
    }
    return fs.readFileSync(config.managedCrontabPath, "utf8").split(/\r?\n/);
  }

  try {
    const result = await execFileImpl(config.crontabBin, ["-l"], {
      env: {
        ...process.env,
        TMPDIR: config.tempDir,
        HOME: `/home/${config.hostUser}`,
      },
    });
    return result.stdout.split(/\r?\n/);
  } catch (error) {
    const stderr = String((error as { stderr?: string }).stderr ?? "");
    if (/no crontab for/i.test(stderr)) {
      return [];
    }
    throw new OngoingTasksToolError("crontab_read_failed", `failed to read ${config.hostUser} crontab`, {
      stderr: stderr.trim(),
    });
  }
}

async function installCrontab(lines: string[], config: OngoingTasksConfig, execFileImpl: ExecFileLike): Promise<void> {
  const normalized = `${lines.filter((line, index, all) => !(index === all.length - 1 && line === "")).join("\n")}\n`;
  if (config.managedCrontabPath) {
    ensureDir(path.dirname(config.managedCrontabPath));
    fs.writeFileSync(config.managedCrontabPath, normalized, "utf8");
    return;
  }

  ensureTaskDirs(config);
  const tempPath = path.join(config.tempDir, `crontab-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.txt`);
  fs.writeFileSync(tempPath, normalized, "utf8");
  try {
    await execFileImpl(config.crontabBin, [tempPath], {
      env: {
        ...process.env,
        TMPDIR: config.tempDir,
        HOME: `/home/${config.hostUser}`,
      },
    });
  } catch (error) {
    throw new OngoingTasksToolError("crontab_write_failed", `failed to install ${config.hostUser} crontab`, {
      stderr: String((error as { stderr?: string }).stderr ?? "").trim(),
    });
  } finally {
    fs.rmSync(tempPath, { force: true });
  }
}

async function upsertCronEntry(task: OngoingTaskRecord, config: OngoingTasksConfig, execFileImpl: ExecFileLike): Promise<void> {
  const lines = await readInstalledCrontab(config, execFileImpl);
  const updated = stripManagedBlock(lines, task.task_id);
  if (task.active) {
    updated.push(...buildCronEntry(task, config));
  }
  await installCrontab(updated, config, execFileImpl);
}

async function removeCronEntry(taskId: string, config: OngoingTasksConfig, execFileImpl: ExecFileLike): Promise<void> {
  const lines = await readInstalledCrontab(config, execFileImpl);
  const updated = stripManagedBlock(lines, taskId);
  await installCrontab(updated, config, execFileImpl);
}

function buildTaskRecord(
  taskId: string,
  taskText: string,
  frequency: OngoingTaskFrequency,
  active: boolean,
  nowIso: string,
  config: OngoingTasksConfig,
): OngoingTaskRecord {
  return {
    task_id: taskId,
    task_text: taskText,
    frequency,
    created_at: nowIso,
    updated_at: nowIso,
    last_done_at: null,
    active,
    schedule_expression: tickExpressionForUnit(frequency.unit),
    lock_path: path.join(lockDir(config), `${taskId}.lock`),
    log_path: path.join(logDir(config), `${taskId}.log`),
  };
}

function executionLogPath(config: OngoingTasksConfig, taskId: string): string {
  return path.join(config.tasksDir, "executions", `${taskId}.jsonl`);
}

function appendExecutionRecord(
  config: OngoingTasksConfig,
  taskId: string,
  payload: Record<string, unknown>,
): void {
  const filePath = executionLogPath(config, taskId);
  ensureDir(path.dirname(filePath));
  fs.appendFileSync(filePath, `${JSON.stringify(payload)}\n`, "utf8");
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
  const repoRoot = defaultRepoRoot();
  const tasksDir =
    trimString(configured?.task_dir) ??
    trimString(process.env.VICUNA_ONGOING_TASKS_DIR) ??
    defaultOngoingTasksDir(repoRoot);
  return {
    tasksDir,
    runnerScript:
      trimString(configured?.runner_script) ??
      trimString(process.env.VICUNA_ONGOING_TASKS_RUNNER_SCRIPT) ??
      path.join(repoRoot, "tools", "ops", "run-ongoing-task-cron.sh"),
    crontabBin:
      trimString(configured?.crontab_bin) ??
      trimString(process.env.VICUNA_ONGOING_TASKS_CRONTAB_BIN) ??
      DEFAULT_CRONTAB_BIN,
    flockBin:
      trimString(configured?.flock_bin) ??
      trimString(process.env.VICUNA_ONGOING_TASKS_FLOCK_BIN) ??
      DEFAULT_FLOCK_BIN,
    tempDir:
      trimString(configured?.temp_dir) ??
      trimString(process.env.VICUNA_ONGOING_TASKS_TMPDIR) ??
      path.join(tasksDir, "tmp"),
    runtimeUrl:
      trimString(configured?.runtime_url) ??
      trimString(process.env.VICUNA_ONGOING_TASKS_RUNTIME_URL) ??
      DEFAULT_RUNTIME_URL,
    runtimeModel:
      trimString(configured?.runtime_model) ??
      trimString(process.env.VICUNA_ONGOING_TASKS_RUNTIME_MODEL) ??
      trimString(process.env.VICUNA_DEEPSEEK_MODEL) ??
      DEFAULT_RUNTIME_MODEL,
    runtimeApiKey:
      trimString(configured?.runtime_api_key) ??
      trimString(process.env.VICUNA_API_KEY),
    hostUser:
      trimString(configured?.host_user) ??
      trimString(process.env.VICUNA_ONGOING_TASKS_HOST_USER) ??
      DEFAULT_HOST_USER,
    managedCrontabPath:
      trimString(process.env.VICUNA_ONGOING_TASKS_CRONTAB_FILE),
  };
}

export async function handleOngoingTasks(
  payload: OngoingTasksInvocation,
  config: OngoingTasksConfig,
  clock: Clock = () => new Date(),
  execFileImpl: ExecFileLike = execFile,
  fetchImpl: FetchLike = fetch,
): Promise<OngoingTasksResponseEnvelope> {
  const action = requireAction(payload);
  const now = clock();
  const nowIso = now.toISOString();
  const registry = loadRegistry(config, clock);

  if (action === "create") {
    const taskText = requireNonEmptyString(payload, "task_text");
    const interval = parsePositiveInteger(payload.interval, "interval");
    const unit = parseTaskUnit(payload.unit, "unit");
    const taskId = optionalString(payload, "task_id") ?? randomUUID();
    const active = optionalBoolean(payload, "active") ?? true;
    const existingIndex = findTaskIndex(registry, taskId);
    const record = buildTaskRecord(taskId, taskText, { interval, unit }, active, nowIso, config);
    if (existingIndex >= 0) {
      record.created_at = registry.tasks[existingIndex].created_at;
      record.last_done_at = registry.tasks[existingIndex].last_done_at;
      registry.tasks[existingIndex] = record;
    } else {
      registry.tasks.push(record);
    }
    saveRegistry(config, registry, clock);
    await upsertCronEntry(record, config, execFileImpl);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      task: summarizeTask(record, now),
      installed: record.active,
      schedule_expression: record.schedule_expression,
    };
  }

  if (action === "get" || action === "get_due") {
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

    const dueOnly = action === "get_due" ? true : (optionalBoolean(payload, "due_only") ?? false);
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
    task.schedule_expression = tickExpressionForUnit(task.frequency.unit);
    task.active = active ?? task.active;
    task.updated_at = nowIso;
    registry.tasks[taskIndex] = task;
    saveRegistry(config, registry, clock);
    if (task.active) {
      await upsertCronEntry(task, config, execFileImpl);
    } else {
      await removeCronEntry(task.task_id, config, execFileImpl);
    }
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      task: summarizeTask(task, now),
      installed: task.active,
      schedule_expression: task.schedule_expression,
    };
  }

  if (action === "complete") {
    const completedAt = payload.completed_at === undefined ? nowIso : parseIsoTimestamp(payload.completed_at, "completed_at");
    task.last_done_at = completedAt;
    task.updated_at = nowIso;
    registry.tasks[taskIndex] = task;
    saveRegistry(config, registry, clock);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      task: summarizeTask(task, now),
    };
  }

  if (action === "delete") {
    registry.tasks.splice(taskIndex, 1);
    saveRegistry(config, registry, clock);
    await removeCronEntry(taskId, config, execFileImpl);
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      deleted_task_id: taskId,
      remaining_count: registry.tasks.length,
      removed: true,
    };
  }

  if (action === "execute") {
    if (!task.active) {
      appendExecutionRecord(config, task.task_id, {
        triggered_at: nowIso,
        status: "skipped_inactive",
      });
      return {
        family: "ongoing_tasks",
        action,
        ok: true,
        task: summarizeTask(task, now),
      };
    }
    const summary = summarizeTask(task, now);
    if (!summary.due_now) {
      appendExecutionRecord(config, task.task_id, {
        triggered_at: nowIso,
        status: "skipped_not_due",
        next_due_at: summary.next_due_at,
      });
      return {
        family: "ongoing_tasks",
        action,
        ok: true,
        task: summary,
      };
    }

    const requestBody = {
      model: config.runtimeModel,
      stream: false,
      messages: [
        {
          role: "system",
          content: task.task_text,
        },
      ],
    };

    let response: Response;
    try {
      response = await fetchImpl(config.runtimeUrl, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(config.runtimeApiKey ? { Authorization: `Bearer ${config.runtimeApiKey}` } : {}),
        },
        body: JSON.stringify(requestBody),
      });
    } catch (error) {
      appendExecutionRecord(config, task.task_id, {
        triggered_at: nowIso,
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
      });
      throw new OngoingTasksToolError("runtime_request_failed", "failed to reach the runtime for scheduled execution", {
        runtime_url: config.runtimeUrl,
        error: error instanceof Error ? error.message : String(error),
      });
    }

    const responseText = await response.text();
    if (!response.ok) {
      appendExecutionRecord(config, task.task_id, {
        triggered_at: nowIso,
        status: "failed",
        status_code: response.status,
        body: responseText.slice(0, 400),
      });
      throw new OngoingTasksToolError("runtime_http_error", `scheduled runtime execution failed with HTTP ${response.status}`, {
        runtime_url: config.runtimeUrl,
        status: response.status,
      });
    }

    task.last_done_at = nowIso;
    task.updated_at = nowIso;
    registry.tasks[taskIndex] = task;
    saveRegistry(config, registry, clock);
    appendExecutionRecord(config, task.task_id, {
      triggered_at: nowIso,
      status: "success",
    });
    return {
      family: "ongoing_tasks",
      action,
      ok: true,
      task: summarizeTask(task, now),
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
