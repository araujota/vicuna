import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  appendProactiveId,
  appendChatTranscriptMessage,
  buildTelegramDocumentDescriptor,
  buildTelegramDocumentLinkage,
  buildTelegramFileUrl,
  detectTelegramDocumentLogicalType,
  extractChatCompletionText,
  extractResponseText,
  getChatTranscript,
  ingestTelegramDocumentMessage,
  loadState,
  normalizeDocumentPlainText,
  normalizeState,
  parseSseChunk,
  saveState,
  sanitizeAssistantRelayText,
  splitSseBuffer,
} from './lib.mjs';

test('extractResponseText returns assistant output text', () => {
  const text = extractResponseText({
    output: [
      {
        role: 'assistant',
        content: [
          { type: 'output_text', text: 'first' },
          { type: 'output_text', text: 'second' },
        ],
      },
    ],
  });

  assert.equal(text, 'first\n\nsecond');
});

test('extractChatCompletionText reads assistant content', () => {
  const text = extractChatCompletionText({
    choices: [
      {
        message: {
          content: 'reply',
        },
      },
    ],
  });

  assert.equal(text, 'reply');
});

test('extractChatCompletionText strips vicuna tool-call xml', () => {
  const text = extractChatCompletionText({
    choices: [
      {
        message: {
          content: 'I should search.\n<vicuna_tool_call tool="web_search"><arg name="query" type="string">GDP</arg></vicuna_tool_call>',
        },
      },
    ],
  });

  assert.equal(text, 'I should search.');
});

test('sanitizeAssistantRelayText strips known tool-call xml blocks', () => {
  const text = sanitizeAssistantRelayText(
    'prefix\n<tool_call>{"name":"search"}</tool_call>\n<minimax:tool_call><invoke name="x"></invoke></minimax:tool_call>\nsuffix',
  );

  assert.equal(text, 'prefix\n\nsuffix');
});

test('detectTelegramDocumentLogicalType supports pdf doc and docx', () => {
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'application/pdf',
    file_name: 'report.bin',
  }), 'pdf');
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'application/msword',
    file_name: 'report.doc',
  }), 'doc');
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'application/octet-stream',
    file_name: 'report.docx',
  }), 'docx');
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'application/zip',
    file_name: 'report.zip',
  }), 'unsupported');
});

test('buildTelegramDocumentDescriptor normalizes telegram document metadata', () => {
  const descriptor = buildTelegramDocumentDescriptor({
    chat: { id: 42 },
    message_id: 9,
    document: {
      file_id: 'file_1',
      file_unique_id: 'uniq_1',
      file_name: 'Quarterly Report.PDF',
      mime_type: 'application/pdf',
      file_size: 1234,
    },
  });

  assert.deepEqual(descriptor, {
    chatId: '42',
    messageId: '9',
    fileId: 'file_1',
    fileUniqueId: 'uniq_1',
    fileName: 'Quarterly Report.PDF',
    mimeType: 'application/pdf',
    fileSize: 1234,
    extension: 'pdf',
    logicalType: 'pdf',
  });
});

test('normalizeDocumentPlainText collapses whitespace and truncates', () => {
  const normalized = normalizeDocumentPlainText(' first \r\n\r\n\r\nsecond \f third ', { maxChars: 12 });

  assert.equal(normalized.text, 'first\n\nsecon');
  assert.equal(normalized.truncated, true);
});

test('buildTelegramDocumentLinkage creates shared metadata for raw and text documents', () => {
  const linkage = buildTelegramDocumentLinkage({
    chatId: '1001',
    messageId: '7',
    fileId: 'file_1',
    fileUniqueId: 'uniq_1',
    fileName: 'memo.docx',
    mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    logicalType: 'docx',
  });

  assert.match(linkage.linkKey, /^telegram-doc-/);
  assert.equal(linkage.containerTag, 'telegram-chat-1001');
  assert.equal(linkage.rawMetadata.linkKey, linkage.textMetadata.linkKey);
  assert.equal(linkage.rawMetadata.contentKind, 'source_file');
  assert.equal(linkage.textMetadata.contentKind, 'extracted_text');
});

test('buildTelegramFileUrl creates telegram file endpoint', () => {
  assert.equal(
    buildTelegramFileUrl('token123', '/documents/file.pdf'),
    'https://api.telegram.org/file/bottoken123/documents/file.pdf',
  );
});

test('ingestTelegramDocumentMessage persists raw and extracted documents with shared linkage', async () => {
  const calls = {
    upload: [],
    update: [],
    add: [],
  };
  const supermemoryClient = {
    documents: {
      async uploadFile(payload) {
        calls.upload.push(payload);
        return { id: 'raw_doc_1' };
      },
      async update(id, payload) {
        calls.update.push({ id, payload });
        return { id, status: 'done' };
      },
      async add(payload) {
        calls.add.push(payload);
        return { id: 'text_doc_1' };
      },
    },
  };

  const result = await ingestTelegramDocumentMessage({
    message: {
      chat: { id: 77 },
      message_id: 10,
      document: {
        file_id: 'file_1',
        file_unique_id: 'uniq_1',
        file_name: 'report.pdf',
        mime_type: 'application/pdf',
      },
    },
    maxDocumentChars: 50,
    resolveTelegramFile: async (fileId) => {
      assert.equal(fileId, 'file_1');
      return { file_path: 'docs/report.pdf' };
    },
    downloadTelegramFile: async (filePath) => {
      assert.equal(filePath, 'docs/report.pdf');
      return Buffer.from('pdf bytes');
    },
    extractPdfText: async () => 'Alpha\n\n\nBeta',
    extractWordText: async () => {
      throw new Error('should not be called');
    },
    supermemoryClient,
    toFileFactory: async (buffer, fileName) => ({ buffer: Buffer.from(buffer), fileName }),
  });

  assert.equal(result.ok, true);
  assert.equal(result.rawDocumentId, 'raw_doc_1');
  assert.equal(result.textDocumentId, 'text_doc_1');
  assert.match(result.transcriptText, /\[Document: report\.pdf\]/);
  assert.match(result.transcriptText, /Alpha\n\nBeta/);
  assert.equal(calls.upload.length, 1);
  assert.equal(calls.update.length, 1);
  assert.equal(calls.add.length, 1);

  const rawMetadata = JSON.parse(calls.upload[0].metadata);
  assert.equal(rawMetadata.contentKind, 'source_file');
  assert.equal(calls.update[0].payload.metadata.linkKey, rawMetadata.linkKey);
  assert.equal(calls.add[0].metadata.linkKey, rawMetadata.linkKey);
  assert.equal(calls.add[0].containerTag, calls.update[0].payload.containerTag);
});

test('ingestTelegramDocumentMessage rejects unsupported documents cleanly', async () => {
  const result = await ingestTelegramDocumentMessage({
    message: {
      chat: { id: 77 },
      message_id: 10,
      document: {
        file_id: 'file_1',
        file_name: 'archive.zip',
        mime_type: 'application/zip',
      },
    },
    supermemoryClient: {},
  });

  assert.equal(result.ok, false);
  assert.equal(result.userError, 'Only plain text, PDF, DOC, and DOCX messages are supported right now.');
});

test('ingestTelegramDocumentMessage requires supermemory for document processing', async () => {
  const result = await ingestTelegramDocumentMessage({
    message: {
      chat: { id: 77 },
      message_id: 10,
      document: {
        file_id: 'file_1',
        file_name: 'report.pdf',
        mime_type: 'application/pdf',
      },
    },
    supermemoryClient: null,
  });

  assert.equal(result.ok, false);
  assert.equal(result.userError, 'SUPERMEMORY_API_KEY is required for Telegram document ingestion.');
});

test('ingestTelegramDocumentMessage surfaces partial failure after raw upload', async () => {
  const supermemoryClient = {
    documents: {
      async uploadFile() {
        return { id: 'raw_doc_1' };
      },
      async update() {
        return { id: 'raw_doc_1', status: 'done' };
      },
      async add() {
        throw new Error('text add failed');
      },
    },
  };

  const result = await ingestTelegramDocumentMessage({
    message: {
      chat: { id: 77 },
      message_id: 10,
      document: {
        file_id: 'file_1',
        file_unique_id: 'uniq_1',
        file_name: 'report.pdf',
        mime_type: 'application/pdf',
      },
    },
    resolveTelegramFile: async () => ({ file_path: 'docs/report.pdf' }),
    downloadTelegramFile: async () => Buffer.from('pdf bytes'),
    extractPdfText: async () => 'Alpha',
    extractWordText: async () => 'unused',
    supermemoryClient,
    toFileFactory: async () => ({ name: 'report.pdf' }),
  });

  assert.equal(result.ok, false);
  assert.equal(result.partialFailure, true);
  assert.match(result.userError, /partially failed after raw file storage/);
});

test('ingestTelegramDocumentMessage surfaces extraction failures directly', async () => {
  const result = await ingestTelegramDocumentMessage({
    message: {
      chat: { id: 77 },
      message_id: 10,
      document: {
        file_id: 'file_1',
        file_name: 'report.doc',
        mime_type: 'application/msword',
      },
    },
    resolveTelegramFile: async () => ({ file_path: 'docs/report.doc' }),
    downloadTelegramFile: async () => Buffer.from('doc bytes'),
    extractPdfText: async () => 'unused',
    extractWordText: async () => {
      throw new Error('DOC/DOCX extraction requires /usr/bin/textutil on the bridge host.');
    },
    supermemoryClient: { documents: {} },
    toFileFactory: async () => ({ name: 'report.doc' }),
  });

  assert.equal(result.ok, false);
  assert.match(result.userError, /textutil/);
});

test('parseSseChunk parses response events', () => {
  const events = parseSseChunk([
    'event: response.created',
    'data: {"type":"response.created"}',
    '',
    'event: response.completed',
    'data: {"type":"response.completed","response":{"id":"resp_1","output":[{"role":"assistant","content":[{"text":"hello"}]}]}}',
    '',
  ].join('\n'));

  assert.equal(events.length, 2);
  assert.equal(events[0].event, 'response.created');
  assert.equal(events[1].data.response.id, 'resp_1');
});

test('splitSseBuffer leaves incomplete trailing event in remainder', () => {
  const chunk = [
    'event: response.created',
    'data: {"type":"response.created"}',
    '',
    'event: response.completed',
    'data: {"type":"response.completed"}',
  ].join('\n');

  const { complete, remainder } = splitSseBuffer(chunk);
  assert.match(complete, /response\.created/);
  assert.match(remainder, /response\.completed/);
});

test('normalizeState bounds and deduplicates persisted values', () => {
  const state = normalizeState({
    telegramOffset: '12',
    chatIds: ['1', 1, '2'],
    proactiveResponseIds: ['a', 'a', 'b'],
    chatSessions: {
      1: {
        messages: [
          { role: 'user', content: ' first ' },
          { role: 'assistant', content: 'second' },
          { role: 'tool', content: 'ignored' },
        ],
      },
    },
  });

  assert.equal(state.telegramOffset, 12);
  assert.deepEqual(state.chatIds, ['1', '2']);
  assert.deepEqual(state.proactiveResponseIds, ['a', 'b']);
  assert.deepEqual(state.chatSessions['1'].messages, [
    { role: 'user', content: 'first' },
    { role: 'assistant', content: 'second' },
  ]);
});

test('appendProactiveId maintains bounded dedupe ids', () => {
  let state = { telegramOffset: 0, chatIds: [], proactiveResponseIds: [] };
  state = appendProactiveId(state, 'resp_1');
  state = appendProactiveId(state, 'resp_1');
  state = appendProactiveId(state, 'resp_2');

  assert.deepEqual(state.proactiveResponseIds, ['resp_1', 'resp_2']);
});

test('appendChatTranscriptMessage isolates chats and trims bounded history', () => {
  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 1, 'user', 'hello', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 1, 'assistant', 'hi', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 2, 'user', 'other chat', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 1, 'user', 'follow-up', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 1, 'assistant', 'answer', { maxHistoryMessages: 3 });

  assert.deepEqual(getChatTranscript(state, 1), [
    { role: 'assistant', content: 'hi' },
    { role: 'user', content: 'follow-up' },
    { role: 'assistant', content: 'answer' },
  ]);
  assert.deepEqual(getChatTranscript(state, 2), [
    { role: 'user', content: 'other chat' },
  ]);
});

test('saveState persists chatSessions transcript history', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'vicuna-telegram-bridge-'));
  const statePath = path.join(dir, 'state.json');

  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 7502424413, 'user', 'first', { maxHistoryMessages: 4 });
  state = appendChatTranscriptMessage(state, 7502424413, 'assistant', 'second', { maxHistoryMessages: 4 });

  await saveState(statePath, state, { maxHistoryMessages: 4 });
  const persisted = JSON.parse(await readFile(statePath, 'utf8'));
  const reloaded = await loadState(statePath, { maxHistoryMessages: 4 });

  assert.deepEqual(persisted.chatSessions['7502424413'].messages, [
    { role: 'user', content: 'first' },
    { role: 'assistant', content: 'second' },
  ]);
  assert.deepEqual(getChatTranscript(reloaded, 7502424413), [
    { role: 'user', content: 'first' },
    { role: 'assistant', content: 'second' },
  ]);
});
