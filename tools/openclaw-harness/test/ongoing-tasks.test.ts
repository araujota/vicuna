import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
import test from "node:test";

import {
  handleOngoingTasks,
  type OngoingTasksConfig,
} from "../src/ongoing-tasks.js";

function createConfig(tempDir: string): OngoingTasksConfig {
  return {
    tasksDir: path.join(tempDir, "ongoing-tasks"),
    runnerScript: "/tmp/run-ongoing-task-cron.sh",
    crontabBin: "/usr/bin/crontab",
    flockBin: "/usr/bin/flock",
    tempDir: path.join(tempDir, "ongoing-tasks", "tmp"),
    runtimeUrl: "http://127.0.0.1:8080/v1/chat/completions",
    runtimeModel: "deepseek-chat",
    hostUser: "vicuna",
    managedCrontabPath: path.join(tempDir, "vicuna.crontab"),
  };
}

test("ongoing task create and delete manage one Vicuña-owned crontab block", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-ongoing-"));
  const config = createConfig(tempDir);
  try {
    const created = await handleOngoingTasks({
      action: "create",
      task_text: "Every week, summarize the local Radarr backlog.",
      interval: 1,
      unit: "weeks",
    }, config);

    assert.equal(created.ok, true);
    assert.equal(created.task?.schedule_expression, "0 0 * * 0");

    const crontabText = fs.readFileSync(config.managedCrontabPath!, "utf8");
    assert.match(crontabText, /VICUNA MANAGED TASK/);
    assert.match(crontabText, /run-ongoing-task-cron\.sh/);
    assert.match(crontabText, /--task-id/);

    const deleted = await handleOngoingTasks({
      action: "delete",
      task_id: created.task?.task_id,
    }, config);

    assert.equal(deleted.ok, true);
    const prunedCrontab = fs.readFileSync(config.managedCrontabPath!, "utf8");
    assert.doesNotMatch(prunedCrontab, /VICUNA MANAGED TASK/);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test("scheduled execute sends the stored recurring prompt as a system message and updates completion time", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-ongoing-exec-"));
  const config = createConfig(tempDir);
  let clockMs = Date.parse("2026-03-27T12:00:00Z");
  const clock = () => new Date(clockMs);
  const fetchCalls: unknown[] = [];
  try {
    const created = await handleOngoingTasks({
      action: "create",
      task_text: "Review the local docs directory and report the most recent uploads.",
      interval: 1,
      unit: "minutes",
    }, config, clock);

    clockMs += 61_000;
    const executed = await handleOngoingTasks({
      action: "execute",
      task_id: created.task?.task_id,
    }, config, clock, undefined, async (url, init) => {
      fetchCalls.push({ url, init });
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    });

    assert.equal(executed.ok, true);
    assert.equal(fetchCalls.length, 1);
    const request = fetchCalls[0] as { url: string; init?: RequestInit };
    assert.equal(request.url, "http://127.0.0.1:8080/v1/chat/completions");
    const body = JSON.parse(String(request.init?.body));
    assert.equal(body.messages[0].role, "system");
    assert.equal(body.messages[0].content, "Review the local docs directory and report the most recent uploads.");
    assert.equal(executed.task?.last_done_at, "2026-03-27T12:01:01.000Z");
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test("scheduled execute skips fetch when the recurring task is not due yet", async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-ongoing-skip-"));
  const config = createConfig(tempDir);
  try {
    const created = await handleOngoingTasks({
      action: "create",
      task_text: "Check for new bridge errors.",
      interval: 1,
      unit: "hours",
    }, config, () => new Date("2026-03-27T12:00:00Z"));

    let called = false;
    const executed = await handleOngoingTasks({
      action: "execute",
      task_id: created.task?.task_id,
    }, config, () => new Date("2026-03-27T12:30:00Z"), undefined, async () => {
      called = true;
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    });

    assert.equal(executed.ok, true);
    assert.equal(called, false);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
