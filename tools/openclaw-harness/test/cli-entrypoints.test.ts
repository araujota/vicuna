import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";

import { isCliEntrypoint } from "../src/cli-entrypoint.js";

function packageRoot(): string {
  return path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
}

function tsxBin(): string {
  return path.join(packageRoot(), "node_modules", ".bin", "tsx");
}

function encodePayload(payload: object): string {
  return Buffer.from(JSON.stringify(payload), "utf8").toString("base64");
}

function writeSecrets(tempDir: string): string {
  const memoryDir = path.join(tempDir, "memories");
  const skillsDir = path.join(tempDir, "skills");
  const tasksDir = path.join(tempDir, "ongoing-tasks");
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");
  fs.mkdirSync(memoryDir, { recursive: true });
  fs.mkdirSync(skillsDir, { recursive: true });
  fs.mkdirSync(tasksDir, { recursive: true });
  fs.writeFileSync(
    secretsPath,
    JSON.stringify({
      tools: {
        hard_memory: {
          memory_dir: memoryDir,
        },
        skills: {
          skills_dir: skillsDir,
        },
        ongoing_tasks: {
          task_dir: tasksDir,
          runner_script: "/tmp/run-ongoing-task-cron.sh",
          host_user: "vicuna",
        },
      },
    }),
    "utf8",
  );
  return secretsPath;
}

test("CLI entrypoints write stdout when executed as standalone scripts", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-cli-entrypoints-"));
  const secretsPath = writeSecrets(tempDir);
  const cliEnv = {
    ...process.env,
    VICUNA_ONGOING_TASKS_CRONTAB_FILE: path.join(tempDir, "vicuna.crontab"),
  };
  try {
    const hardMemoryStdout = execFileSync(
      tsxBin(),
      [
        path.join(packageRoot(), "src", "hard-memory.ts"),
        `--secrets-path=${secretsPath}`,
        `--payload-base64=${encodePayload({
          memories: [{ key: "cli-memory", content: "CLI smoke memory." }],
        })}`,
      ],
      { encoding: "utf8", env: cliEnv },
    );
    const hardMemoryResponse = JSON.parse(hardMemoryStdout);
    assert.equal(hardMemoryResponse.ok, true);
    assert.equal(hardMemoryResponse.written, 1);

    const skillsStdout = execFileSync(
      tsxBin(),
      [
        path.join(packageRoot(), "src", "skills.ts"),
        `--secrets-path=${secretsPath}`,
        `--payload-base64=${encodePayload({
          action: "create",
          name: "CLI Skill",
          content: "# CLI Skill\n\nUse the CLI path.\n",
        })}`,
      ],
      { encoding: "utf8", env: cliEnv },
    );
    const skillsResponse = JSON.parse(skillsStdout);
    assert.equal(skillsResponse.ok, true);
    assert.equal(skillsResponse.created, true);

    const ongoingStdout = execFileSync(
      tsxBin(),
      [
        path.join(packageRoot(), "src", "ongoing-tasks.ts"),
        `--secrets-path=${secretsPath}`,
        `--payload-base64=${encodePayload({
          action: "create",
          task_text: "Run the recurring system prompt.",
          interval: 1,
          unit: "days",
        })}`,
      ],
      { encoding: "utf8", env: cliEnv },
    );
    const ongoingResponse = JSON.parse(ongoingStdout);
    assert.equal(ongoingResponse.ok, true);
    assert.equal(ongoingResponse.installed, true);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test("CLI entrypoint detection accepts symlinked invocation paths", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-cli-symlink-"));
  const realDir = path.join(tempDir, "real");
  const realScript = path.join(realDir, "hard-memory.ts");
  const symlinkDir = path.join(tempDir, "symlinked");
  const symlinkScript = path.join(symlinkDir, "hard-memory.ts");
  fs.mkdirSync(realDir, { recursive: true });
  fs.writeFileSync(realScript, "// test fixture\n", "utf8");
  fs.symlinkSync(realDir, symlinkDir, "dir");
  try {
    assert.equal(isCliEntrypoint(new URL(`file://${realScript}`).href, symlinkScript), true);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
