import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
import test from "node:test";

import {
  errorEnvelope,
  handleParsedDocuments,
  resolveParsedDocumentsConfig,
  type ParsedDocumentsConfig,
} from "../src/parsed-documents.js";

function config(docsDir: string): ParsedDocumentsConfig {
  return {
    docsDir,
    defaultThreshold: 0.58,
    shortQueryThreshold: 0.68,
    maxResults: 5,
  };
}

function writeBundle(root: string, bundleId: string, payload: { title: string; sourcePath: string; chunks: Array<{ text: string; index: number; linkKey?: string }> }): void {
  const bundleDir = path.join(root, bundleId);
  fs.mkdirSync(bundleDir, { recursive: true });
  fs.writeFileSync(path.join(bundleDir, "metadata.json"), `${JSON.stringify({
    bundle_id: bundleId,
    document_title: payload.title,
    source_path: payload.sourcePath,
  }, null, 2)}\n`);
  fs.writeFileSync(path.join(bundleDir, "chunks.json"), `${JSON.stringify(
    payload.chunks.map((chunk) => ({
      chunk_index: chunk.index,
      contextual_text: chunk.text,
      document_title: payload.title,
      link_key: chunk.linkKey,
    })),
    null,
    2,
  )}\n`);
}

test("parsed-document search returns compact labeled chunks from the local docs corpus", async () => {
  const docsDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-docs-"));
  try {
    writeBundle(docsDir, "telegram-doc-1", {
      title: "movie-notes.md",
      sourcePath: path.join(docsDir, "telegram-doc-1", "source", "movie-notes.md"),
      chunks: [
        { index: 0, text: "Christopher Nolan science fiction preferences chunk", linkKey: "telegram-doc-1" },
        { index: 1, text: "Loose but still relevant movie preference chunk", linkKey: "telegram-doc-1" },
      ],
    });

    const longQuery = await handleParsedDocuments(
      { query: "Find notes about Christopher Nolan science fiction preferences" },
      config(docsDir),
    );

    assert.equal(longQuery.ok, true);
    assert.equal(longQuery.threshold, 0.58);
    assert.equal(longQuery.count, 1);
    assert.deepEqual(longQuery.items?.[0], {
      document_title: "movie-notes.md",
      chunk_text: "Christopher Nolan science fiction preferences chunk",
      similarity: 0.625,
      chunk_index: 0,
      link_key: "telegram-doc-1",
      source_path: path.join(docsDir, "telegram-doc-1", "source", "movie-notes.md"),
    });

    const shortQuery = await handleParsedDocuments(
      { query: "Nolan" },
      config(docsDir),
    );

    assert.equal(shortQuery.ok, true);
    assert.equal(shortQuery.threshold, 0.68);
    assert.equal(shortQuery.count, 1);
  } finally {
    fs.rmSync(docsDir, { recursive: true, force: true });
  }
});

test("parsed-document config resolves the local docs root from env", () => {
  process.env.VICUNA_DOCS_DIR = "/tmp/vicuna-docs";

  const resolved = resolveParsedDocumentsConfig("/tmp/does-not-exist.json");
  assert.equal(resolved.docsDir, "/tmp/vicuna-docs");

  delete process.env.VICUNA_DOCS_DIR;
});

test("parsed-document search returns typed payload errors", async () => {
  try {
    await handleParsedDocuments({ query: "" }, config("/tmp/vicuna-docs"));
    assert.fail("expected empty query to throw");
  } catch (error) {
    const envelope = errorEnvelope(error);
    assert.equal(envelope.ok, false);
    assert.equal(envelope.error?.kind, "missing_argument");
  }
});
