import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  appendProactiveId,
  appendChatTranscriptMessage,
  extractChatCompletionText,
  extractResponseText,
  getChatTranscript,
  loadState,
  normalizeState,
  parseSseChunk,
  saveState,
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
