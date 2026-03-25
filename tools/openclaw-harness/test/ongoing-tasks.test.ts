import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import assert from "node:assert/strict";
import test from "node:test";

import {
  errorEnvelope,
  handleOngoingTasks,
  type OngoingTasksConfig,
  type OngoingTasksInvocation,
} from "../src/ongoing-tasks.js";

type MockHardMemoryState = {
  registryText?: string;
  writeBodies: unknown[];
};

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const body = Buffer.concat(chunks).toString("utf8");
  return body ? JSON.parse(body) : {};
}

function writeJson(res: ServerResponse, statusCode: number, payload: unknown): void {
  const encoded = Buffer.from(JSON.stringify(payload), "utf8");
  res.statusCode = statusCode;
  res.setHeader("content-type", "application/json");
  res.setHeader("content-length", String(encoded.length));
  res.end(encoded);
}

async function startMockHardMemory(state: MockHardMemoryState): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server = createServer(async (req, res) => {
    const body = await readJson(req);
    if (req.method === "POST" && req.url === "/v4/profile") {
      const searchResults = state.registryText
        ? {
            results: [{
              id: "memory-registry-1",
              title: "Ongoing task registry",
              memory: state.registryText,
              metadata: {
                key: "ongoing-tasks-registry",
                title: "Ongoing task registry",
              },
            }],
          }
        : { results: [] };
      writeJson(res, 200, { searchResults });
      return;
    }

    if (req.method === "POST" && req.url === "/v4/memories") {
      state.writeBodies.push(body);
      const payload = body as {
        memories?: Array<{ content?: string }>;
      };
      state.registryText = payload.memories?.[0]?.content;
      writeJson(res, 200, { ok: true });
      return;
    }

    writeJson(res, 404, { error: "not found" });
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("expected TCP server address");
  }

  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: async () => {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      });
    },
  };
}

function config(baseUrl: string): OngoingTasksConfig {
  return {
    baseUrl,
    authToken: "test-token",
    containerTag: "vicuna",
    runtimeIdentity: "vicuna",
    registryKey: "ongoing-tasks-registry",
    registryTitle: "Ongoing task registry",
    queryThreshold: 0,
  };
}

function fixedClock(iso: string): () => Date {
  return () => new Date(iso);
}

test("ongoing-task registry supports create, due polling, complete, edit, and delete with compact summaries", async () => {
  const state: MockHardMemoryState = { writeBodies: [] };
  const mock = await startMockHardMemory(state);
  try {
    const taskConfig = config(mock.baseUrl);
    const create = await handleOngoingTasks(
      {
        action: "create",
        task_text:
          "Every few days, analyze the movies we have saved in Radarr and recommend one I might like.",
        interval: 3,
        unit: "days",
      },
      taskConfig,
      fixedClock("2026-03-25T12:00:00.000Z")
    );

    assert.equal(create.ok, true);
    assert.equal(create.family, "ongoing_tasks");
    assert.ok(create.task);
    assert.equal(create.task.task_text.includes("Radarr"), true);
    assert.equal(create.task.frequency.interval, 3);
    assert.equal(create.task.frequency.unit, "days");
    assert.equal(create.task.due_now, true);
    assert.equal(create.task.next_due_at, "2026-03-25T12:00:00.000Z");
    assert.equal("base_url" in create, false);
    assert.equal("request" in create, false);

    const taskId = create.task.task_id;
    const due = await handleOngoingTasks(
      { action: "get", due_only: true },
      taskConfig,
      fixedClock("2026-03-25T12:30:00.000Z")
    );
    assert.equal(due.ok, true);
    assert.equal(due.due_only, true);
    assert.equal(due.count, 1);
    assert.equal(due.tasks?.[0]?.task_id, taskId);

    const complete = await handleOngoingTasks(
      { action: "complete", task_id: taskId, completed_at: "2026-03-25T13:00:00.000Z" },
      taskConfig,
      fixedClock("2026-03-25T13:00:00.000Z")
    );
    assert.equal(complete.ok, true);
    assert.ok(complete.task);
    assert.equal(complete.task.last_done_at, "2026-03-25T13:00:00.000Z");
    assert.equal(complete.task.next_due_at, "2026-03-28T13:00:00.000Z");
    assert.equal(complete.task.due_now, false);

    const notYetDue = await handleOngoingTasks(
      { action: "get", due_only: true },
      taskConfig,
      fixedClock("2026-03-26T13:00:00.000Z")
    );
    assert.equal(notYetDue.ok, true);
    assert.equal(notYetDue.count, 0);

    const edited = await handleOngoingTasks(
      { action: "edit", task_id: taskId, interval: 1, unit: "days" },
      taskConfig,
      fixedClock("2026-03-27T13:05:00.000Z")
    );
    assert.equal(edited.ok, true);
    assert.ok(edited.task);
    assert.equal(edited.task.frequency.interval, 1);
    assert.equal(edited.task.frequency.unit, "days");
    assert.equal(edited.task.due_now, true);
    assert.equal(edited.task.next_due_at, "2026-03-26T13:00:00.000Z");

    const deleted = await handleOngoingTasks(
      { action: "delete", task_id: taskId },
      taskConfig,
      fixedClock("2026-03-27T13:10:00.000Z")
    );
    assert.equal(deleted.ok, true);
    assert.equal(deleted.deleted_task_id, taskId);
    assert.equal(deleted.remaining_count, 0);

    const remaining = await handleOngoingTasks(
      { action: "get" },
      taskConfig,
      fixedClock("2026-03-27T13:15:00.000Z")
    );
    assert.equal(remaining.ok, true);
    assert.equal(remaining.count, 0);
    assert.equal(state.writeBodies.length >= 4, true);
  } finally {
    await mock.close();
  }
});

test("ongoing-task registry surfaces typed invalid-registry errors without leaking raw storage payloads", async () => {
  const state: MockHardMemoryState = {
    registryText: "{not valid json",
    writeBodies: [],
  };
  const mock = await startMockHardMemory(state);
  try {
    await assert.rejects(
      () => handleOngoingTasks({ action: "get" } satisfies OngoingTasksInvocation, config(mock.baseUrl)),
      /stored ongoing-task registry was not valid JSON/
    );

    try {
      await handleOngoingTasks({ action: "get" }, config(mock.baseUrl));
      assert.fail("expected invalid registry to throw");
    } catch (error) {
      const envelope = errorEnvelope("get", error);
      assert.equal(envelope.ok, false);
      assert.equal(envelope.error?.kind, "invalid_registry");
      assert.equal(envelope.error?.message.includes("valid JSON"), true);
      assert.equal("tasks" in envelope, false);
      assert.equal("deleted_task_id" in envelope, false);
    }
  } finally {
    await mock.close();
  }
});
