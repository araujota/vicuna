import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import {
  handleHostShell,
  HostShellToolError,
  parseHostShellCliInvocation,
  resolveHostShellConfig,
} from "../src/index.js";

function makeWorkspaceRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-host-shell-"));
}

test("parseHostShellCliInvocation decodes the payload", () => {
  const payload = {
    command: "pwd",
    purpose: "Inspect the workspace root.",
    working_directory: "logs",
    timeout_ms: 2000,
  };
  const encoded = Buffer.from(JSON.stringify(payload), "utf8").toString("base64");
  const parsed = parseHostShellCliInvocation([`--payload-base64=${encoded}`]);
  assert.deepEqual(parsed.payload, payload);
});

test("resolveHostShellConfig honors VICUNA_HOST_SHELL_ROOT", () => {
  const configuredRoot = makeWorkspaceRoot();
  process.env.VICUNA_HOST_SHELL_ROOT = configuredRoot;
  try {
    const config = resolveHostShellConfig("/tmp/vicuna-openclaw-test");
    assert.equal(config.workspaceRoot, configuredRoot);
    assert.equal(config.defaultTimeoutMs, 8000);
    assert.equal(config.maxTimeoutMs, 30000);
  } finally {
    delete process.env.VICUNA_HOST_SHELL_ROOT;
  }
});

test("handleHostShell summarizes JSON output and file changes", async () => {
  const workspaceRoot = makeWorkspaceRoot();
  const response = await handleHostShell(
    {
      command: "mkdir -p logs && printf '{\"ok\":true,\"files\":[\"logs/out.txt\"]}\\n' && printf 'hello\\n' > logs/out.txt",
      purpose: "Create one file and report it.",
    },
    {
      workspaceRoot,
      defaultTimeoutMs: 8000,
      maxTimeoutMs: 30000,
    },
  );

  assert.equal(response.ok, true);
  assert.equal(response.status, "completed");
  assert.equal(response.working_directory, ".");
  assert.equal(response.stdout.kind, "json");
  assert.deepEqual(response.stdout.parsed_json, { ok: true, files: ["logs/out.txt"] });
  assert.equal(response.workspace_diff.created_paths.includes("logs"), true);
  assert.equal(response.workspace_diff.created_paths.includes("logs/out.txt"), true);
  assert.equal(fs.readFileSync(path.join(workspaceRoot, "logs", "out.txt"), "utf8"), "hello\n");
});

test("handleHostShell summarizes path-list stdout without raw shell dumps", async () => {
  const workspaceRoot = makeWorkspaceRoot();
  fs.mkdirSync(path.join(workspaceRoot, "docs"), { recursive: true });
  fs.writeFileSync(path.join(workspaceRoot, "docs", "one.txt"), "a\n", "utf8");
  fs.writeFileSync(path.join(workspaceRoot, "docs", "two.txt"), "b\n", "utf8");

  const response = await handleHostShell(
    {
      command: "find docs -maxdepth 1 -type f | sort",
      purpose: "List files in docs.",
    },
    {
      workspaceRoot,
      defaultTimeoutMs: 8000,
      maxTimeoutMs: 30000,
    },
  );

  assert.equal(response.ok, true);
  assert.equal(response.stdout.kind, "path_list");
  assert.deepEqual(response.stdout.detected_paths, ["docs/one.txt", "docs/two.txt"]);
  assert.deepEqual(response.workspace_diff, {
    created_paths: [],
    modified_paths: [],
    deleted_paths: [],
    snapshot_truncated: false,
  });
});

test("handleHostShell rejects working directories that escape the workspace", async () => {
  const workspaceRoot = makeWorkspaceRoot();
  await assert.rejects(
    () =>
      handleHostShell(
        {
          command: "pwd",
          purpose: "Try to escape.",
          working_directory: "../outside",
        },
        {
          workspaceRoot,
          defaultTimeoutMs: 8000,
          maxTimeoutMs: 30000,
        },
      ),
    (error: unknown) =>
      error instanceof HostShellToolError &&
      error.kind === "invalid_argument" &&
      /inside the host shell workspace/i.test(error.message),
  );
});

test("handleHostShell returns a timed_out envelope when the command exceeds timeout", async () => {
  const workspaceRoot = makeWorkspaceRoot();
  const response = await handleHostShell(
    {
      command: "sleep 1",
      purpose: "Force a timeout.",
      timeout_ms: 50,
    },
    {
      workspaceRoot,
      defaultTimeoutMs: 50,
      maxTimeoutMs: 30000,
    },
  );

  assert.equal(response.ok, false);
  assert.equal(response.status, "timed_out");
  assert.equal(response.error?.kind, "timeout");
  assert.match(response.summary, /timeout/i);
});
