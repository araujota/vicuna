import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  appendProactiveId,
  appendChatTranscriptMessage,
  bootstrapTelegramOutboxOffset,
  buildTelegramDocumentDescriptor,
  buildTelegramDocumentLinkage,
  buildTelegramFileUrl,
  buildTelegramRelaySystemPrompt,
  createTelegramConversation,
  detectTelegramDocumentLogicalType,
  extractChatCompletionText,
  extractResponseText,
  getConversationForTelegramMessage,
  getChatTranscript,
  getLatestConversationId,
  getTelegramRequestDeltaMessages,
  ingestTelegramDocumentMessage,
  isTelegramCarryableAssistantMessage,
  isTelegramReplyTargetErrorMessage,
  isTelegramTerminalDeliveryErrorMessage,
  loadState,
  normalizeDocumentPlainText,
  normalizeTelegramOutboxItem,
  normalizeState,
  parseSseChunk,
  recordTelegramOutboxDeliveryReceipt,
  reconcileTelegramOutboxOffset,
  retainCoherentTranscriptWindow,
  resolveTelegramConversationForMessage,
  resolveTelegramConversationForOutbound,
  saveState,
  sanitizeAssistantRelayText,
  setTelegramOutboxCheckpoint,
  shouldBootstrapTelegramOutboxOffset,
  splitSseBuffer,
  summarizeChatCompletion,
  updateTelegramOffset,
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

test('extractChatCompletionText returns empty string for tool-call-only completion', () => {
  const text = extractChatCompletionText({
    choices: [
      {
        finish_reason: 'tool_calls',
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'exec', arguments: '{}' },
            },
          ],
        },
      },
    ],
  });

  assert.equal(text, '');
});

test('summarizeChatCompletion reports non-text completion shape', () => {
  const summary = summarizeChatCompletion({
    choices: [
      {
        finish_reason: 'tool_calls',
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'exec', arguments: '{}' },
            },
          ],
          reasoning_content: 'internal',
        },
      },
    ],
  });

  assert.deepEqual(summary, {
    finishReason: 'tool_calls',
    messageRole: 'assistant',
    contentType: 'string',
    textLength: 0,
    contentPartCount: 0,
    toolCallCount: 1,
    reasoningLength: 8,
  });
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

test('buildTelegramRelaySystemPrompt prefers exposed media tools over stale refusals', () => {
  const prompt = buildTelegramRelaySystemPrompt();

  assert.match(prompt, /Maintain continuity from the runtime-managed Telegram dialogue state and the current user turn\./);
  assert.match(prompt, /Sonarr, Radarr, Chaptarr, or another relevant live tool/);
  assert.match(prompt, /use the relevant tool instead of claiming you lack access/i);
  assert.match(prompt, /cannot interact with external systems/i);
  assert.match(prompt, /treat that as stale and use the currently available tool on this turn/i);
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

test('appendChatTranscriptMessage isolates chats and trims bounded history coherently', () => {
  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 1, 'user', 'hello', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 1, 'assistant', 'hi', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 2, 'user', 'other chat', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 1, 'user', 'follow-up', { maxHistoryMessages: 3 });
  state = appendChatTranscriptMessage(state, 1, 'assistant', 'answer', { maxHistoryMessages: 3 });

  assert.deepEqual(getChatTranscript(state, 1), [
    { role: 'user', content: 'follow-up' },
    { role: 'assistant', content: 'answer' },
  ]);
  assert.deepEqual(getChatTranscript(state, 2), [
    { role: 'user', content: 'other chat' },
  ]);
});

test('appendChatTranscriptMessage can filter transcript continuity by conversation id', () => {
  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 1, 'user', 'first thread user', {
    maxHistoryMessages: 8,
    conversationId: 'tc1',
    telegramMessageId: 11,
  });
  state = appendChatTranscriptMessage(state, 1, 'assistant', 'first thread assistant', {
    maxHistoryMessages: 8,
    conversationId: 'tc1',
    telegramMessageId: 12,
  });
  state = appendChatTranscriptMessage(state, 1, 'user', 'second thread user', {
    maxHistoryMessages: 8,
    conversationId: 'tc2',
    telegramMessageId: 21,
  });

  assert.deepEqual(getChatTranscript(state, 1, { conversationId: 'tc1' }), [
    { role: 'user', content: 'first thread user', conversationId: 'tc1', telegramMessageId: 11 },
    { role: 'assistant', content: 'first thread assistant', conversationId: 'tc1', telegramMessageId: 12 },
  ]);
  assert.deepEqual(getChatTranscript(state, 1, { conversationId: 'tc2' }), [
    { role: 'user', content: 'second thread user', conversationId: 'tc2', telegramMessageId: 21 },
  ]);
  assert.equal(getConversationForTelegramMessage(state, 1, 12), 'tc1');
  assert.equal(getLatestConversationId(state, 1), 'tc2');
});

test('terminal Telegram error replies are not carryable transcript history', () => {
  assert.equal(
    isTelegramCarryableAssistantMessage('I ran into a problem while working on that request and could not complete it: vicuna chat failed: 400 {"error":{"code":400,"message":"request (22963 tokens) exceeds the available context size (16384 tokens), try increasing it","type":"exceed_context_size_error"}}'),
    false,
  );
  assert.equal(
    isTelegramCarryableAssistantMessage('The import has started for Dante\'s Inferno.'),
    true,
  );

  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 1, 'user', 'hello', { maxHistoryMessages: 8 });
  state = appendChatTranscriptMessage(
    state,
    1,
    'assistant',
    'I ran into a problem while working on that request and could not complete it: vicuna chat failed: 500 {"error":{"code":500,"message":"Failed to continue authoritative ReAct turn after exhausted terminal-policy repair","type":"server_error"}}',
    { maxHistoryMessages: 8 },
  );
  state = appendChatTranscriptMessage(state, 1, 'assistant', 'clean reply', { maxHistoryMessages: 8 });

  assert.deepEqual(getChatTranscript(state, 1), [
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'clean reply' },
  ]);
});

test('getTelegramRequestDeltaMessages keeps only the newest user turn', () => {
  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 7502424413, 'user', 'first user', {
    maxHistoryMessages: 8,
    conversationId: 'tc3',
    telegramMessageId: 101,
  });
  state = appendChatTranscriptMessage(state, 7502424413, 'assistant', 'first assistant', {
    maxHistoryMessages: 8,
    conversationId: 'tc3',
    telegramMessageId: 102,
  });
  state = appendChatTranscriptMessage(state, 7502424413, 'user', 'latest user', {
    maxHistoryMessages: 8,
    conversationId: 'tc3',
    telegramMessageId: 103,
  });

  assert.deepEqual(getTelegramRequestDeltaMessages(state, 7502424413, { conversationId: 'tc3' }), [
    { role: 'user', content: 'latest user' },
  ]);
});

test('resolveTelegramConversationForMessage prefers reply linkage and otherwise reuses latest conversation', () => {
  let state = normalizeState({});
  ({ state } = createTelegramConversation(state, 42));
  state = appendChatTranscriptMessage(state, 42, 'assistant', 'Which Archer?', {
    maxHistoryMessages: 8,
    conversationId: 'tc1',
    telegramMessageId: 99,
  });

  const explicitReply = resolveTelegramConversationForMessage(state, {
    chat: { id: 42 },
    message_id: 100,
    reply_to_message: { message_id: 99 },
    text: 'the 2009 one',
  });
  assert.equal(explicitReply.conversationId, 'tc1');
  assert.equal(explicitReply.reason, 'reply_to_message');

  const plainFollowUp = resolveTelegramConversationForMessage(explicitReply.state, {
    chat: { id: 42 },
    message_id: 101,
    text: 'and delete it',
  });
  assert.equal(plainFollowUp.conversationId, 'tc1');
  assert.equal(plainFollowUp.reason, 'latest');
});

test('resolveTelegramConversationForOutbound reuses reply-linked conversation anchors', () => {
  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 5, 'user', 'delete Archer', {
    maxHistoryMessages: 8,
    conversationId: 'tc9',
    telegramMessageId: 401,
  });

  const resolved = resolveTelegramConversationForOutbound(state, 5, {
    replyToMessageId: 401,
  });

  assert.equal(resolved.conversationId, 'tc9');
  assert.equal(resolved.reason, 'reply_to_message');
});

test('retainCoherentTranscriptWindow drops leading assistant orphan after bounded trim', () => {
  const retained = retainCoherentTranscriptWindow([
    { role: 'user', content: 'u1' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'u2' },
    { role: 'assistant', content: 'a2' },
    { role: 'user', content: 'u3' },
  ], { maxHistoryMessages: 4 });

  assert.deepEqual(retained, [
    { role: 'user', content: 'u2' },
    { role: 'assistant', content: 'a2' },
    { role: 'user', content: 'u3' },
  ]);
});

test('appendChatTranscriptMessage preserves newest user turn without leading assistant orphan', () => {
  let state = normalizeState({});
  for (let index = 1; index <= 6; index += 1) {
    state = appendChatTranscriptMessage(state, 1, 'user', `u${index}`, { maxHistoryMessages: 12 });
    state = appendChatTranscriptMessage(state, 1, 'assistant', `a${index}`, { maxHistoryMessages: 12 });
  }

  state = appendChatTranscriptMessage(state, 1, 'user', 'u7', { maxHistoryMessages: 12 });

  assert.deepEqual(getChatTranscript(state, 1), [
    { role: 'user', content: 'u2' },
    { role: 'assistant', content: 'a2' },
    { role: 'user', content: 'u3' },
    { role: 'assistant', content: 'a3' },
    { role: 'user', content: 'u4' },
    { role: 'assistant', content: 'a4' },
    { role: 'user', content: 'u5' },
    { role: 'assistant', content: 'a5' },
    { role: 'user', content: 'u6' },
    { role: 'assistant', content: 'a6' },
    { role: 'user', content: 'u7' },
  ]);
});

test('normalizeState repairs persisted chat sessions that start with assistant-only orphan', () => {
  const state = normalizeState({
    chatSessions: {
      1: {
        messages: [
          { role: 'assistant', content: 'a1' },
          { role: 'user', content: 'u2' },
          { role: 'assistant', content: 'a2' },
        ],
      },
    },
  }, { maxHistoryMessages: 3 });

  assert.deepEqual(getChatTranscript(state, 1), [
    { role: 'user', content: 'u2' },
    { role: 'assistant', content: 'a2' },
  ]);
});

test('saveState persists chatSessions transcript history', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'vicuna-telegram-bridge-'));
  const statePath = path.join(dir, 'state.json');

  let state = normalizeState({});
  state = appendChatTranscriptMessage(state, 7502424413, 'user', 'first', {
    maxHistoryMessages: 4,
    conversationId: 'tc1',
    telegramMessageId: 101,
  });
  state = appendChatTranscriptMessage(state, 7502424413, 'assistant', 'second', {
    maxHistoryMessages: 4,
    conversationId: 'tc1',
    telegramMessageId: 102,
  });
  state = setTelegramOutboxCheckpoint(state, 21);
  state = recordTelegramOutboxDeliveryReceipt(state, {
    sequenceNumber: 21,
    chatId: '7502424413',
    replyToMessageId: 101,
    deliveryMode: 'reply',
    telegramMessageId: 102,
    deliveredAtMs: 123456,
  });

  await saveState(statePath, state, { maxHistoryMessages: 4 });
  const persisted = JSON.parse(await readFile(statePath, 'utf8'));
  const reloaded = await loadState(statePath, { maxHistoryMessages: 4 });

  assert.deepEqual(persisted.chatSessions['7502424413'].messages, [
    { role: 'user', content: 'first', conversationId: 'tc1', telegramMessageId: 101 },
    { role: 'assistant', content: 'second', conversationId: 'tc1', telegramMessageId: 102 },
  ]);
  assert.equal(persisted.schemaVersion >= 2, true);
  assert.equal(persisted.telegramOutboxCheckpointInitialized, true);
  assert.deepEqual(persisted.telegramOutboxDeliveryReceipt, {
    sequenceNumber: 21,
    chatId: '7502424413',
    replyToMessageId: 101,
    deliveryMode: 'reply',
    telegramMessageId: 102,
    deliveredAtMs: 123456,
  });
  assert.deepEqual(persisted.chatConversationState['7502424413'], {
    latestConversationId: 'tc1',
    messageToConversation: {
      '101': 'tc1',
      '102': 'tc1',
    },
  });
  assert.deepEqual(getChatTranscript(reloaded, 7502424413), [
    { role: 'user', content: 'first', conversationId: 'tc1', telegramMessageId: 101 },
    { role: 'assistant', content: 'second', conversationId: 'tc1', telegramMessageId: 102 },
  ]);
});

test('updateTelegramOffset remains monotonic alongside transcript updates', () => {
  let state = normalizeState({});
  state = updateTelegramOffset(state, 40);
  state = appendChatTranscriptMessage(state, 1, 'user', 'hello', { maxHistoryMessages: 4 });
  state = updateTelegramOffset(state, 39);

  assert.equal(state.telegramOffset, 41);
  assert.deepEqual(getChatTranscript(state, 1), [
    { role: 'user', content: 'hello' },
  ]);
});

test('reconcileTelegramOutboxOffset rewinds when the runtime outbox sequence resets behind persisted state', () => {
  const nextOffset = reconcileTelegramOutboxOffset(8, {
    stored_items: 1,
    oldest_sequence: 1,
    newest_sequence: 1,
  });

  assert.equal(nextOffset, 0);
});

test('reconcileTelegramOutboxOffset keeps the current offset when the runtime outbox has not fallen behind it', () => {
  const nextOffset = reconcileTelegramOutboxOffset(1, {
    stored_items: 1,
    oldest_sequence: 1,
    newest_sequence: 3,
  });

  assert.equal(nextOffset, 1);
});

test('reconcileTelegramOutboxOffset advances into the retained outbox window when the saved offset is too old', () => {
  const nextOffset = reconcileTelegramOutboxOffset(2, {
    stored_items: 4,
    oldest_sequence: 10,
    newest_sequence: 13,
  });

  assert.equal(nextOffset, 9);
});

test('reconcileTelegramOutboxOffset resets to zero when the runtime outbox is empty', () => {
  const nextOffset = reconcileTelegramOutboxOffset(7, {
    stored_items: 0,
    oldest_sequence: 0,
    newest_sequence: 0,
  });

  assert.equal(nextOffset, 0);
});

test('shouldBootstrapTelegramOutboxOffset is true for fresh state with retained backlog', () => {
  const state = normalizeState({});

  assert.equal(shouldBootstrapTelegramOutboxOffset(state, {
    stored_items: 4,
    oldest_sequence: 10,
    newest_sequence: 13,
  }), true);
});

test('bootstrapTelegramOutboxOffset fast-forwards to newest retained sequence and marks checkpoint initialized', () => {
  const state = bootstrapTelegramOutboxOffset(normalizeState({}), {
    stored_items: 4,
    oldest_sequence: 10,
    newest_sequence: 13,
  });

  assert.equal(state.telegramOutboxOffset, 13);
  assert.equal(state.telegramOutboxCheckpointInitialized, true);
});

test('setTelegramOutboxCheckpoint preserves explicit resume progress', () => {
  const state = setTelegramOutboxCheckpoint(normalizeState({}), 9);

  assert.equal(state.telegramOutboxOffset, 9);
  assert.equal(state.telegramOutboxCheckpointInitialized, true);
});

test('isTelegramReplyTargetErrorMessage identifies reply-target failures', () => {
  assert.equal(isTelegramReplyTargetErrorMessage('telegram sendMessage failed: 400 {"ok":false,"description":"Bad Request: reply message not found"}'), true);
  assert.equal(isTelegramReplyTargetErrorMessage('telegram sendMessage failed: 403 {"ok":false,"description":"Forbidden: bot was blocked by the user"}'), false);
});

test('isTelegramTerminalDeliveryErrorMessage identifies terminal chat delivery failures', () => {
  assert.equal(isTelegramTerminalDeliveryErrorMessage('telegram sendMessage failed: 400 {"ok":false,"description":"Bad Request: chat not found"}'), true);
  assert.equal(isTelegramTerminalDeliveryErrorMessage('telegram sendMessage failed: 403 {"ok":false,"description":"Forbidden: bot was blocked by the user"}'), true);
  assert.equal(isTelegramTerminalDeliveryErrorMessage('telegram sendMessage failed: 403 {"ok":false,"description":"Forbidden: bot is not a member of the channel chat"}'), true);
  assert.equal(isTelegramTerminalDeliveryErrorMessage('telegram sendMessage failed: 400 {"ok":false,"description":"Bad Request: reply message not found"}'), false);
  assert.equal(isTelegramTerminalDeliveryErrorMessage('telegram sendMessage failed: 500 {"ok":false,"description":"Internal Server Error"}'), false);
});

test('normalizeTelegramOutboxItem validates message follow-ups', () => {
  assert.deepEqual(normalizeTelegramOutboxItem({
    sequence_number: 12,
    kind: 'message',
    chat_scope: '7502424413',
    text: 'done',
    reply_to_message_id: 91,
  }), {
    ok: true,
    skippable: true,
    sequenceNumber: 12,
    kind: 'message',
    chatId: '7502424413',
    text: 'done',
    replyToMessageId: 91,
  });
});

test('normalizeTelegramOutboxItem marks malformed sequenced items as skippable faults', () => {
  assert.deepEqual(normalizeTelegramOutboxItem({
    sequence_number: 14,
    kind: 'message',
    chat_scope: '',
    text: '',
  }), {
    ok: false,
    skippable: true,
    sequenceNumber: 14,
    kind: 'message',
    error: 'runtime telegram outbox message item was incomplete',
  });
});

test('normalizeTelegramOutboxItem marks missing sequence metadata as non-skippable', () => {
  assert.deepEqual(normalizeTelegramOutboxItem({
    kind: 'ask_with_options',
    chat_scope: '7502424413',
    question: '',
    options: ['A'],
  }), {
    ok: false,
    skippable: false,
    sequenceNumber: 0,
    kind: 'ask_with_options',
    error: 'runtime telegram outbox ask_with_options item was incomplete',
  });
});

test('normalizeTelegramOutboxItem validates approval_request prompts', () => {
  assert.deepEqual(normalizeTelegramOutboxItem({
    sequence_number: 22,
    kind: 'approval_request',
    approval_id: 'appr_7f8c2d',
    chat_scope: '7502424413',
    question: 'Approve deleting Archer from Sonarr?',
    options: ['allow', 'disallow'],
    reply_to_message_id: 243,
  }), {
    ok: true,
    skippable: true,
    sequenceNumber: 22,
    kind: 'approval_request',
    approvalId: 'appr_7f8c2d',
    chatId: '7502424413',
    question: 'Approve deleting Archer from Sonarr?',
    options: ['allow', 'disallow'],
    replyToMessageId: 243,
  });
});

test('normalizeState preserves approval prompt metadata and drops malformed approval prompts', () => {
  const state = normalizeState({
    pendingOptionPrompts: {
      a1: {
        kind: 'approval_request',
        approvalId: 'appr_1',
        chatId: '7502424413',
        question: 'Approve Sonarr delete?',
        options: ['allow', 'disallow'],
        conversationId: 'tc1',
        telegramMessageId: 321,
        createdAtMs: 10,
      },
      a2: {
        kind: 'approval_request',
        approvalId: '',
        chatId: '7502424413',
        question: 'Broken approval',
        options: ['allow', 'disallow'],
        createdAtMs: 20,
      },
    },
  });

  assert.deepEqual(state.pendingOptionPrompts, {
    a1: {
      kind: 'approval_request',
      approvalId: 'appr_1',
      chatId: '7502424413',
      question: 'Approve Sonarr delete?',
      options: ['allow', 'disallow'],
      conversationId: 'tc1',
      telegramMessageId: 321,
      createdAtMs: 10,
    },
  });
});
