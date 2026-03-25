import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import assert from "node:assert/strict";
import test from "node:test";

import {
  errorEnvelope,
  handleTelegramRelay,
  resolveTelegramRelayConfig,
  type TelegramRelayConfig,
  type TelegramRelayInvocation,
} from "../src/telegram-relay.js";

type MockTelegramRelayState = {
  requests: unknown[];
  nextSequenceNumber: number;
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

async function startMockTelegramRelay(
  state: MockTelegramRelayState
): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server = createServer(async (req, res) => {
    if (req.method === "POST" && req.url === "/v1/telegram/outbox") {
      const body = await readJson(req);
      state.requests.push(body);
      const request = body as Record<string, unknown>;
      writeJson(res, 200, {
        ok: true,
        queued: true,
        deduplicated: request.dedupe_key === "same-message",
        sequence_number: state.nextSequenceNumber++,
        chat_scope: request.chat_scope,
        stored_items: state.requests.length,
        next_sequence_number: state.nextSequenceNumber,
      });
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

function config(baseUrl: string, defaultChatScope?: string): TelegramRelayConfig {
  return {
    baseUrl,
    authToken: "test-api-key",
    defaultChatScope,
  };
}

test("telegram relay queues one compact outbox message and supports default chat scope fallback", async () => {
  const state: MockTelegramRelayState = { requests: [], nextSequenceNumber: 1 };
  const mock = await startMockTelegramRelay(state);
  try {
    const explicit = await handleTelegramRelay(
      {
        text: "Done. I queued the movie recommendation run.",
        chat_scope: "7502424413",
        reply_to_message_id: 91,
        dedupe_key: "same-message",
        intent: "conclusion",
        urgency: 0.4,
      },
      config(mock.baseUrl)
    );

    assert.equal(explicit.ok, true);
    assert.equal(explicit.sequence_number, 1);
    assert.equal(explicit.chat_scope, "7502424413");
    assert.equal(explicit.deduplicated, true);

    const firstRequest = state.requests[0] as Record<string, unknown>;
    assert.equal(firstRequest.kind, "message");
    assert.equal(firstRequest.chat_scope, "7502424413");
    assert.equal(firstRequest.reply_to_message_id, 91);
    assert.equal(firstRequest.intent, "conclusion");

    const fallback = await handleTelegramRelay(
      {
        text: "Follow-up sent through the default chat scope.",
      },
      config(mock.baseUrl, "12345")
    );

    assert.equal(fallback.ok, true);
    assert.equal(fallback.sequence_number, 2);
    assert.equal(fallback.chat_scope, "12345");

    const secondRequest = state.requests[1] as Record<string, unknown>;
    assert.equal(secondRequest.chat_scope, "12345");
  } finally {
    await mock.close();
  }
});

test("telegram relay returns typed missing-chat-scope errors", async () => {
  try {
    await handleTelegramRelay({ text: "Need a chat id." } satisfies TelegramRelayInvocation, config("http://127.0.0.1:8080"));
    assert.fail("expected missing chat scope to throw");
  } catch (error) {
    const envelope = errorEnvelope(error);
    assert.equal(envelope.ok, false);
    assert.equal(envelope.error?.kind, "missing_chat_scope");
    assert.equal(envelope.error?.message.includes("default_chat_scope"), true);
  }
});

test("telegram relay config resolves provider URL, auth, and default chat scope from env fallback", () => {
  process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL = "http://127.0.0.1:8099/";
  process.env.VICUNA_API_KEY = "relay-token";
  process.env.TELEGRAM_DEFAULT_CHAT_SCOPE = "555";

  const config = resolveTelegramRelayConfig("/tmp/does-not-exist.json");
  assert.equal(config.baseUrl, "http://127.0.0.1:8099");
  assert.equal(config.authToken, "relay-token");
  assert.equal(config.defaultChatScope, "555");

  delete process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL;
  delete process.env.VICUNA_API_KEY;
  delete process.env.TELEGRAM_DEFAULT_CHAT_SCOPE;
});
