import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { handleHardMemory, resolveHardMemoryConfig } from "../src/hard-memory.js";

test("resolveHardMemoryConfig prefers explicit env and falls back to host-shell-root memories", () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-hard-memory-config-"));
  const secretsPath = path.join(tempRoot, "openclaw-tool-secrets.json");
  fs.writeFileSync(secretsPath, "{}\n", "utf8");

  const previousMemoryDir = process.env.VICUNA_HARD_MEMORY_DIR;
  const previousHostShellRoot = process.env.VICUNA_HOST_SHELL_ROOT;
  try {
    delete process.env.VICUNA_HARD_MEMORY_DIR;
    process.env.VICUNA_HOST_SHELL_ROOT = "/srv/vicuna-home";
    assert.equal(resolveHardMemoryConfig(secretsPath).memoryDir, "/srv/vicuna-home/memories");

    process.env.VICUNA_HARD_MEMORY_DIR = "/tmp/vicuna-memories";
    assert.equal(resolveHardMemoryConfig(secretsPath).memoryDir, "/tmp/vicuna-memories");
  } finally {
    if (previousMemoryDir === undefined) {
      delete process.env.VICUNA_HARD_MEMORY_DIR;
    } else {
      process.env.VICUNA_HARD_MEMORY_DIR = previousMemoryDir;
    }
    if (previousHostShellRoot === undefined) {
      delete process.env.VICUNA_HOST_SHELL_ROOT;
    } else {
      process.env.VICUNA_HOST_SHELL_ROOT = previousHostShellRoot;
    }
  }
});

test("handleHardMemory writes markdown files, reads them back, and keyed writes update in place", async () => {
  const memoryDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-hard-memory-store-"));
  const config = { memoryDir, runtimeIdentity: "vicuna" };

  const writeResponse = await handleHardMemory({
    memories: [{
      content: "Remember the markdown backend.",
      title: "Markdown backend",
      key: "markdown-backend",
      kind: "DECISION",
      domain: "runtime",
      tags: ["memory", "markdown"],
      importance: 0.9,
    }],
  }, config);
  assert.equal(writeResponse.ok, true);
  assert.equal(writeResponse.written, 1);
  assert.equal(writeResponse.created, 1);

  const files = fs.readdirSync(memoryDir).filter((entry) => entry.endsWith(".md"));
  assert.deepEqual(files, ["markdown-backend.md"]);
  const storedText = fs.readFileSync(path.join(memoryDir, files[0]), "utf8");
  assert.match(storedText, /^---\n/);
  assert.match(storedText, /key: "markdown-backend"/);
  assert.match(storedText, /Remember the markdown backend\./);

  const readResponse = await handleHardMemory({ query: "markdown backend", limit: 3 }, config);
  assert.equal(readResponse.ok, true);
  assert.equal(readResponse.count, 1);
  assert.equal(readResponse.results?.[0].key, "markdown-backend");
  assert.match(String(readResponse.results?.[0].excerpt ?? ""), /markdown backend/i);

  const updateResponse = await handleHardMemory({
    memories: [{
      content: "Remember the markdown backend is now durable.",
      key: "markdown-backend",
      kind: "DECISION",
      domain: "runtime",
      tags: ["memory", "markdown"],
    }],
  }, config);
  assert.equal(updateResponse.ok, true);
  assert.equal(updateResponse.updated, 1);
  assert.deepEqual(fs.readdirSync(memoryDir).filter((entry) => entry.endsWith(".md")), ["markdown-backend.md"]);
  assert.match(
    fs.readFileSync(path.join(memoryDir, "markdown-backend.md"), "utf8"),
    /Remember the markdown backend is now durable\./,
  );
});

test("handleHardMemory skips malformed markdown files and returns warnings", async () => {
  const memoryDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-hard-memory-warnings-"));
  const config = { memoryDir, runtimeIdentity: "vicuna" };

  fs.writeFileSync(path.join(memoryDir, "broken.md"), "not frontmatter", "utf8");
  await handleHardMemory({
    memories: [{
      content: "Remember the runtime is provider-first.",
      key: "provider-first",
      domain: "runtime",
      tags: ["provider"],
    }],
  }, config);

  const response = await handleHardMemory({ query: "provider-first", limit: 5 }, config);
  assert.equal(response.ok, true);
  assert.equal(response.count, 1);
  assert.equal(response.results?.[0].key, "provider-first");
  assert.equal((response.warnings ?? []).length, 1);
});
