import { Buffer } from "node:buffer";
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

import { defaultHostShellRoot, defaultRepoRoot } from "./config.js";

const execFileAsync = promisify(execFile);

const DEFAULT_TIMEOUT_MS = 8_000;
const MAX_TIMEOUT_MS = 30_000;
const MAX_OUTPUT_BYTES = 128 * 1024;
const MAX_PREVIEW_LINES = 24;
const MAX_PREVIEW_LINE_CHARS = 240;
const MAX_CHANGED_PATHS = 128;
const MAX_SNAPSHOT_ENTRIES = 4_096;

type JsonRecord = Record<string, unknown>;

export type HostShellInvocation = {
  command?: unknown;
  purpose?: unknown;
  working_directory?: unknown;
  timeout_ms?: unknown;
};

export type HostShellConfig = {
  workspaceRoot: string;
  defaultTimeoutMs: number;
  maxTimeoutMs: number;
};

type SnapshotEntry = {
  kind: "file" | "directory" | "symlink" | "other";
  size: number;
  mtimeMs: number;
};

type SnapshotState = {
  entries: Map<string, SnapshotEntry>;
  truncated: boolean;
};

type OutputKind = "empty" | "json" | "path" | "path_list" | "text" | "binary";

export type HostShellOutputSummary = {
  kind: OutputKind;
  line_count: number;
  byte_count: number;
  truncated: boolean;
  preview_lines: string[];
  parsed_json?: unknown;
  detected_paths?: string[];
};

export type HostShellResponseEnvelope = {
  family: "host_shell";
  action: "execute";
  ok: boolean;
  status: "completed" | "failed" | "timed_out";
  summary: string;
  command: string;
  purpose: string;
  workspace_root: string;
  working_directory: string;
  duration_ms: number;
  exit_code: number;
  stdout: HostShellOutputSummary;
  stderr: HostShellOutputSummary;
  workspace_diff: {
    created_paths: string[];
    modified_paths: string[];
    deleted_paths: string[];
    snapshot_truncated: boolean;
  };
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

export class HostShellToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

function trimString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function requireCommand(payload: HostShellInvocation): string {
  const command = trimString(payload.command);
  if (!command) {
    throw new HostShellToolError("missing_argument", "command is required");
  }
  return command;
}

function requirePurpose(payload: HostShellInvocation): string {
  const purpose = trimString(payload.purpose);
  if (!purpose) {
    throw new HostShellToolError("missing_argument", "purpose is required");
  }
  return purpose;
}

function normalizeTimeoutMs(raw: unknown, config: HostShellConfig): number {
  if (raw === undefined || raw === null) {
    return config.defaultTimeoutMs;
  }
  if (typeof raw !== "number" || !Number.isInteger(raw) || raw < 1) {
    throw new HostShellToolError("invalid_argument", "timeout_ms must be a positive integer", {
      field: "timeout_ms",
      value: raw,
    });
  }
  if (raw > config.maxTimeoutMs) {
    throw new HostShellToolError("invalid_argument", `timeout_ms must not exceed ${config.maxTimeoutMs}`, {
      field: "timeout_ms",
      value: raw,
      max_timeout_ms: config.maxTimeoutMs,
    });
  }
  return raw;
}

function normalizeRelativeWorkingDirectory(raw: unknown): string {
  if (raw === undefined || raw === null || raw === "") {
    return ".";
  }
  if (typeof raw !== "string") {
    throw new HostShellToolError("invalid_argument", "working_directory must be a string", {
      field: "working_directory",
      value: raw,
    });
  }
  const trimmed = raw.trim();
  if (!trimmed) {
    return ".";
  }
  if (path.isAbsolute(trimmed)) {
    throw new HostShellToolError("invalid_argument", "working_directory must be relative to the host shell workspace", {
      field: "working_directory",
      value: trimmed,
    });
  }
  const normalized = path.posix.normalize(trimmed.replaceAll("\\", "/"));
  if (normalized === ".." || normalized.startsWith("../")) {
    throw new HostShellToolError("invalid_argument", "working_directory must stay inside the host shell workspace", {
      field: "working_directory",
      value: trimmed,
    });
  }
  return normalized === "." ? "." : normalized.replace(/^\.\/+/, "");
}

function ensureWorkspaceRoot(workspaceRoot: string): void {
  if (!workspaceRoot.trim()) {
    throw new HostShellToolError("invalid_config", "host shell workspace root is empty");
  }
  if (!fs.existsSync(workspaceRoot)) {
    throw new HostShellToolError("workspace_missing", "host shell workspace root does not exist", {
      workspace_root: workspaceRoot,
    });
  }
  const stats = fs.statSync(workspaceRoot);
  if (!stats.isDirectory()) {
    throw new HostShellToolError("workspace_invalid", "host shell workspace root is not a directory", {
      workspace_root: workspaceRoot,
    });
  }
}

function resolveWorkingDirectory(workspaceRoot: string, relativeWorkingDirectory: string): { absolute: string; relative: string } {
  const absolute = path.resolve(workspaceRoot, relativeWorkingDirectory);
  const relative = path.relative(workspaceRoot, absolute);
  if (relative === ".." || relative.startsWith(`..${path.sep}`)) {
    throw new HostShellToolError("invalid_argument", "working_directory resolved outside the host shell workspace", {
      working_directory: relativeWorkingDirectory,
      workspace_root: workspaceRoot,
    });
  }
  return {
    absolute,
    relative: relative ? relative.replaceAll(path.sep, "/") : ".",
  };
}

function entryKind(stats: fs.Stats): SnapshotEntry["kind"] {
  if (stats.isFile()) {
    return "file";
  }
  if (stats.isDirectory()) {
    return "directory";
  }
  if (stats.isSymbolicLink()) {
    return "symlink";
  }
  return "other";
}

function snapshotWorkspace(root: string): SnapshotState {
  const entries = new Map<string, SnapshotEntry>();
  const stack = [root];
  let truncated = false;

  while (stack.length > 0) {
    const current = stack.pop()!;
    let directoryEntries: fs.Dirent[];
    try {
      directoryEntries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }

    directoryEntries.sort((lhs, rhs) => lhs.name.localeCompare(rhs.name));
    for (const entry of directoryEntries) {
      const absolute = path.join(current, entry.name);
      const relative = path.relative(root, absolute).replaceAll(path.sep, "/");
      if (!relative) {
        continue;
      }
      let stats: fs.Stats;
      try {
        stats = fs.lstatSync(absolute);
      } catch {
        continue;
      }
      entries.set(relative, {
        kind: entryKind(stats),
        size: stats.size,
        mtimeMs: Math.trunc(stats.mtimeMs),
      });
      if (entries.size >= MAX_SNAPSHOT_ENTRIES) {
        truncated = true;
        return { entries, truncated };
      }
      if (entry.isDirectory()) {
        stack.push(absolute);
      }
    }
  }

  return { entries, truncated };
}

function diffSnapshots(before: SnapshotState, after: SnapshotState) {
  const createdPaths: string[] = [];
  const modifiedPaths: string[] = [];
  const deletedPaths: string[] = [];

  for (const [relative, nextEntry] of after.entries) {
    const previousEntry = before.entries.get(relative);
    if (!previousEntry) {
      createdPaths.push(relative);
      continue;
    }
    if (
      previousEntry.kind !== nextEntry.kind ||
      previousEntry.size !== nextEntry.size ||
      previousEntry.mtimeMs !== nextEntry.mtimeMs
    ) {
      modifiedPaths.push(relative);
    }
  }

  for (const relative of before.entries.keys()) {
    if (!after.entries.has(relative)) {
      deletedPaths.push(relative);
    }
  }

  return {
    created_paths: createdPaths.slice(0, MAX_CHANGED_PATHS),
    modified_paths: modifiedPaths.slice(0, MAX_CHANGED_PATHS),
    deleted_paths: deletedPaths.slice(0, MAX_CHANGED_PATHS),
    snapshot_truncated:
      before.truncated ||
      after.truncated ||
      createdPaths.length > MAX_CHANGED_PATHS ||
      modifiedPaths.length > MAX_CHANGED_PATHS ||
      deletedPaths.length > MAX_CHANGED_PATHS,
  };
}

function safePreviewLine(line: string): string {
  return line.length > MAX_PREVIEW_LINE_CHARS ? `${line.slice(0, MAX_PREVIEW_LINE_CHARS - 1)}…` : line;
}

function looksBinary(buffer: Buffer): boolean {
  if (buffer.length === 0) {
    return false;
  }
  const sampleLength = Math.min(buffer.length, 512);
  for (let index = 0; index < sampleLength; index += 1) {
    if (buffer[index] === 0) {
      return true;
    }
  }
  return false;
}

function previewLines(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0)
    .slice(0, MAX_PREVIEW_LINES)
    .map(safePreviewLine);
}

function detectedPathsFromLines(lines: string[], workspaceRoot: string): string[] {
  const detected = new Set<string>();
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    if (trimmed.includes("\t")) {
      continue;
    }
    if (trimmed.length > 240) {
      continue;
    }
    if (trimmed.startsWith("{") || trimmed.startsWith("[") || trimmed.startsWith("<")) {
      continue;
    }
    const normalized = trimmed.replaceAll("\\", "/");
    if (normalized === "." || normalized === "..") {
      detected.add(normalized);
      continue;
    }
    if (path.isAbsolute(normalized)) {
      const relative = path.relative(workspaceRoot, normalized);
      if (relative === ".." || relative.startsWith(`..${path.sep}`)) {
        continue;
      }
      detected.add(relative ? relative.replaceAll(path.sep, "/") : ".");
      continue;
    }
    if (!normalized.includes(" ")) {
      detected.add(path.posix.normalize(normalized));
    }
  }
  return [...detected].slice(0, MAX_PREVIEW_LINES);
}

function summarizeOutput(buffer: Buffer, workspaceRoot: string): HostShellOutputSummary {
  const byteCount = buffer.length;
  if (byteCount === 0) {
    return {
      kind: "empty",
      line_count: 0,
      byte_count: 0,
      truncated: false,
      preview_lines: [],
    };
  }

  if (looksBinary(buffer)) {
    return {
      kind: "binary",
      line_count: 0,
      byte_count: byteCount,
      truncated: byteCount >= MAX_OUTPUT_BYTES,
      preview_lines: [],
    };
  }

  const text = buffer.toString("utf8");
  const lines = text.split(/\r?\n/);
  const lineCount = lines.filter((line) => line.length > 0).length;
  const trimmed = text.trim();
  const previews = previewLines(text);

  if (!trimmed) {
    return {
      kind: "empty",
      line_count: lineCount,
      byte_count: byteCount,
      truncated: byteCount >= MAX_OUTPUT_BYTES,
      preview_lines: [],
    };
  }

  try {
    return {
      kind: "json",
      line_count: lineCount,
      byte_count: byteCount,
      truncated: byteCount >= MAX_OUTPUT_BYTES,
      preview_lines: previews,
      parsed_json: JSON.parse(trimmed),
    };
  } catch {
  }

  const detectedPaths = detectedPathsFromLines(previews, workspaceRoot);
  if (detectedPaths.length === 1 && previews.length === 1) {
    return {
      kind: "path",
      line_count: lineCount,
      byte_count: byteCount,
      truncated: byteCount >= MAX_OUTPUT_BYTES,
      preview_lines: previews,
      detected_paths: detectedPaths,
    };
  }
  if (detectedPaths.length >= 2 && detectedPaths.length === previews.length) {
    return {
      kind: "path_list",
      line_count: lineCount,
      byte_count: byteCount,
      truncated: byteCount >= MAX_OUTPUT_BYTES,
      preview_lines: previews,
      detected_paths: detectedPaths,
    };
  }

  return {
    kind: "text",
    line_count: lineCount,
    byte_count: byteCount,
    truncated: byteCount >= MAX_OUTPUT_BYTES,
    preview_lines: previews,
    ...(detectedPaths.length > 0 ? { detected_paths: detectedPaths } : {}),
  };
}

function buildSummary(
  status: HostShellResponseEnvelope["status"],
  workingDirectory: string,
  durationMs: number,
  stdout: HostShellOutputSummary,
  stderr: HostShellOutputSummary,
  diff: HostShellResponseEnvelope["workspace_diff"],
): string {
  const parts = [
    status === "completed"
      ? `Command succeeded in ${durationMs}ms from '${workingDirectory}'.`
      : status === "timed_out"
        ? `Command timed out after ${durationMs}ms from '${workingDirectory}'.`
        : `Command failed in ${durationMs}ms from '${workingDirectory}'.`,
  ];
  const changedCount = diff.created_paths.length + diff.modified_paths.length + diff.deleted_paths.length;
  if (changedCount > 0) {
    parts.push(
      `Workspace changes: ${diff.created_paths.length} created, ${diff.modified_paths.length} modified, ${diff.deleted_paths.length} deleted.`
    );
  } else {
    parts.push("Workspace changes: none detected.");
  }
  if (stdout.kind !== "empty") {
    parts.push(`Stdout: ${stdout.line_count} line(s), kind=${stdout.kind}.`);
  }
  if (stderr.kind !== "empty") {
    parts.push(`Stderr: ${stderr.line_count} line(s), kind=${stderr.kind}.`);
  }
  if (diff.snapshot_truncated) {
    parts.push("Workspace diff was truncated.");
  }
  return parts.join(" ");
}

export function parseCliInvocation(argv: string[]): { payload: HostShellInvocation } {
  let payloadBase64 = "";
  for (const arg of argv) {
    if (arg.startsWith("--payload-base64=")) {
      payloadBase64 = arg.slice("--payload-base64=".length);
    }
  }

  if (!payloadBase64) {
    throw new HostShellToolError("missing_payload", "missing required --payload-base64 argument");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new HostShellToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new HostShellToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }

  return {
    payload: payload as HostShellInvocation,
  };
}

export function resolveHostShellConfig(repoRoot = defaultRepoRoot()): HostShellConfig {
  return {
    workspaceRoot: defaultHostShellRoot(repoRoot),
    defaultTimeoutMs: DEFAULT_TIMEOUT_MS,
    maxTimeoutMs: MAX_TIMEOUT_MS,
  };
}

function errorEnvelope(error: HostShellToolError, payload?: Partial<HostShellResponseEnvelope>): HostShellResponseEnvelope {
  return {
    family: "host_shell",
    action: "execute",
    ok: false,
    status: error.kind === "timeout" ? "timed_out" : "failed",
    summary: error.message,
    command: payload?.command ?? "",
    purpose: payload?.purpose ?? "",
    workspace_root: payload?.workspace_root ?? "",
    working_directory: payload?.working_directory ?? ".",
    duration_ms: payload?.duration_ms ?? 0,
    exit_code: payload?.exit_code ?? 1,
    stdout: payload?.stdout ?? {
      kind: "empty",
      line_count: 0,
      byte_count: 0,
      truncated: false,
      preview_lines: [],
    },
    stderr: payload?.stderr ?? {
      kind: "empty",
      line_count: 0,
      byte_count: 0,
      truncated: false,
      preview_lines: [],
    },
    workspace_diff: payload?.workspace_diff ?? {
      created_paths: [],
      modified_paths: [],
      deleted_paths: [],
      snapshot_truncated: false,
    },
    error: {
      kind: error.kind,
      message: error.message,
      ...(error.details ? { details: error.details } : {}),
    },
  };
}

export async function handleHostShell(
  payload: HostShellInvocation,
  config = resolveHostShellConfig(),
): Promise<HostShellResponseEnvelope> {
  const command = requireCommand(payload);
  const purpose = requirePurpose(payload);
  ensureWorkspaceRoot(config.workspaceRoot);
  const timeoutMs = normalizeTimeoutMs(payload.timeout_ms, config);
  const workingDirectory = resolveWorkingDirectory(
    config.workspaceRoot,
    normalizeRelativeWorkingDirectory(payload.working_directory),
  );
  if (!fs.existsSync(workingDirectory.absolute)) {
    throw new HostShellToolError("working_directory_missing", "working_directory does not exist inside the host shell workspace", {
      working_directory: workingDirectory.relative,
      workspace_root: config.workspaceRoot,
    });
  }

  const beforeSnapshot = snapshotWorkspace(config.workspaceRoot);
  const startedAt = Date.now();

  try {
    const result = await execFileAsync(
      "bash",
      ["-lc", command],
      {
        cwd: workingDirectory.absolute,
        env: {
          ...process.env,
          HOME: config.workspaceRoot,
        },
        timeout: timeoutMs,
        maxBuffer: MAX_OUTPUT_BYTES,
        encoding: "buffer",
      },
    );

    const afterSnapshot = snapshotWorkspace(config.workspaceRoot);
    const stdout = summarizeOutput(result.stdout as Buffer, config.workspaceRoot);
    const stderr = summarizeOutput(result.stderr as Buffer, config.workspaceRoot);
    const workspaceDiff = diffSnapshots(beforeSnapshot, afterSnapshot);
    const durationMs = Date.now() - startedAt;

    return {
      family: "host_shell",
      action: "execute",
      ok: true,
      status: "completed",
      summary: buildSummary("completed", workingDirectory.relative, durationMs, stdout, stderr, workspaceDiff),
      command,
      purpose,
      workspace_root: config.workspaceRoot,
      working_directory: workingDirectory.relative,
      duration_ms: durationMs,
      exit_code: 0,
      stdout,
      stderr,
      workspace_diff: workspaceDiff,
    };
  } catch (error) {
    const durationMs = Date.now() - startedAt;
    const nodeError = error as NodeJS.ErrnoException & {
      code?: string | number;
      signal?: NodeJS.Signals | null;
      stdout?: Buffer;
      stderr?: Buffer;
      killed?: boolean;
    };
    const afterSnapshot = snapshotWorkspace(config.workspaceRoot);
    const stdout = summarizeOutput(Buffer.from(nodeError.stdout ?? []), config.workspaceRoot);
    const stderr = summarizeOutput(Buffer.from(nodeError.stderr ?? []), config.workspaceRoot);
    const workspaceDiff = diffSnapshots(beforeSnapshot, afterSnapshot);
    const timedOut = nodeError.killed || nodeError.signal === "SIGTERM";
    const wrapped = new HostShellToolError(
      timedOut ? "timeout" : "command_failed",
      timedOut
        ? `shell command exceeded timeout of ${timeoutMs}ms`
        : `shell command failed${nodeError.code !== undefined ? ` with exit code ${String(nodeError.code)}` : ""}`,
      {
        command,
        timeout_ms: timeoutMs,
        code: nodeError.code,
        signal: nodeError.signal ?? undefined,
      },
    );
    return errorEnvelope(wrapped, {
      command,
      purpose,
      workspace_root: config.workspaceRoot,
      working_directory: workingDirectory.relative,
      duration_ms: durationMs,
      exit_code: typeof nodeError.code === "number" ? nodeError.code : 1,
      stdout,
      stderr,
      workspace_diff: workspaceDiff,
    });
  }
}

export async function runHostShellCli(argv: string[]): Promise<HostShellResponseEnvelope> {
  const { payload } = parseCliInvocation(argv);
  return handleHostShell(payload, resolveHostShellConfig());
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runHostShellCli(process.argv.slice(2))
    .then((response) => {
      process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
    })
    .catch((error) => {
      const wrapped =
        error instanceof HostShellToolError
          ? error
          : new HostShellToolError("unexpected_error", error instanceof Error ? error.message : String(error));
      process.stdout.write(`${JSON.stringify(errorEnvelope(wrapped), null, 2)}\n`);
      process.exitCode = 1;
    });
}
