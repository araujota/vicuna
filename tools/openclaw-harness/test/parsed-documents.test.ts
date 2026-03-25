import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import assert from "node:assert/strict";
import test from "node:test";

import {
  errorEnvelope,
  handleParsedDocuments,
  resolveParsedDocumentsConfig,
  type ParsedDocumentsConfig,
} from "../src/parsed-documents.js";

type MockSearchState = {
  requests: unknown[];
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

async function startMockSearch(state: MockSearchState): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server = createServer(async (req, res) => {
    if (req.method === "POST" && req.url === "/v4/search") {
      const body = await readJson(req);
      state.requests.push(body);
      writeJson(res, 200, {
        results: [
          {
            id: "chunk_1",
            similarity: 0.81,
            memory: "Christopher Nolan science-fiction influence chunk",
            metadata: {
              contentKind: "parsed_chunk",
              source: "telegram_bridge",
              documentTitle: "movie-notes.md",
              chunkIndex: 0,
              linkKey: "telegram-doc-1",
            },
          },
          {
            id: "chunk_2",
            similarity: 0.63,
            memory: "Loose but still relevant movie preference chunk",
            metadata: {
              contentKind: "parsed_chunk",
              source: "telegram_bridge",
              documentTitle: "movie-notes.md",
              chunkIndex: 1,
              linkKey: "telegram-doc-1",
            },
          },
          {
            id: "chunk_3",
            similarity: 0.92,
            memory: "wrong kind",
            metadata: {
              contentKind: "source_file",
              source: "telegram_bridge",
              documentTitle: "raw.pdf",
              chunkIndex: 0,
              linkKey: "telegram-doc-raw",
            },
          },
        ],
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

function config(baseUrl: string): ParsedDocumentsConfig {
  return {
    baseUrl,
    authToken: "test-token",
    containerTag: "vicuna-telegram-documents",
    runtimeIdentity: "vicuna",
    defaultThreshold: 0.58,
    shortQueryThreshold: 0.68,
    maxResults: 5,
  };
}

test("parsed-document search returns compact labeled chunks and filters by explicit threshold policy", async () => {
  const state: MockSearchState = { requests: [] };
  const mock = await startMockSearch(state);
  try {
    const longQuery = await handleParsedDocuments(
      { query: "Find notes about Christopher Nolan science fiction preferences" },
      config(mock.baseUrl)
    );

    assert.equal(longQuery.ok, true);
    assert.equal(longQuery.threshold, 0.58);
    assert.equal(longQuery.count, 2);
    assert.deepEqual(longQuery.items?.[0], {
      document_title: "movie-notes.md",
      chunk_text: "Christopher Nolan science-fiction influence chunk",
      similarity: 0.81,
      chunk_index: 0,
      link_key: "telegram-doc-1",
    });

    const shortQuery = await handleParsedDocuments(
      { query: "Nolan" },
      config(mock.baseUrl)
    );

    assert.equal(shortQuery.ok, true);
    assert.equal(shortQuery.threshold, 0.68);
    assert.equal(shortQuery.count, 1);

    const request = state.requests[0] as Record<string, unknown>;
    assert.equal(request.containerTag, "vicuna-telegram-documents");
    assert.equal(request.searchMode, "memories");
    assert.equal(request.rerank, true);
  } finally {
    await mock.close();
  }
});

test("parsed-document search config resolves env fallback", () => {
  process.env.SUPERMEMORY_BASE_URL = "https://memory.example.com/";
  process.env.SUPERMEMORY_API_KEY = "memory-token";
  process.env.TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG = "telegram-docs";

  const config = resolveParsedDocumentsConfig("/tmp/does-not-exist.json");
  assert.equal(config.baseUrl, "https://memory.example.com");
  assert.equal(config.authToken, "memory-token");
  assert.equal(config.containerTag, "telegram-docs");

  delete process.env.SUPERMEMORY_BASE_URL;
  delete process.env.SUPERMEMORY_API_KEY;
  delete process.env.TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG;
});

test("parsed-document search returns typed payload errors", async () => {
  try {
    await handleParsedDocuments({ query: "" }, config("http://127.0.0.1:8080"));
    assert.fail("expected empty query to throw");
  } catch (error) {
    const envelope = errorEnvelope(error);
    assert.equal(envelope.ok, false);
    assert.equal(envelope.error?.kind, "missing_argument");
  }
});
