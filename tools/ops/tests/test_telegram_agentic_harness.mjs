import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { readFile } from 'node:fs/promises';

import {
  buildHostHeaders,
  captureAdvanced,
  extractAssistantText,
  isCaptureSatisfied,
  isDeliverySatisfied,
  parseHarnessConfig,
  snapshotCaptureState,
} from '../telegram-agentic-harness.mjs';

test('parseHarnessConfig normalizes defaults and requests', () => {
  const config = parseHarnessConfig({
    chatId: '7502424413',
    requests: [
      { text: 'First prompt' },
      { id: 'second', text: 'Second prompt', waitForVideo: false },
    ],
  }, {
    TELEGRAM_BRIDGE_VICUNA_BASE_URL: 'http://127.0.0.1:8080',
    TELEGRAM_BRIDGE_MODEL: 'vicuna-experimental',
  });

  assert.equal(config.chatId, '7502424413');
  assert.equal(config.requests.length, 2);
  assert.equal(config.requests[0].id, 'turn_1');
  assert.equal(config.requests[0].visibleText, 'First prompt');
  assert.equal(config.requests[1].waitForVideo, false);
  assert.equal(config.deferredDelivery, true);
});

test('buildHostHeaders emits telegram-scoped forwarding headers', () => {
  const headers = buildHostHeaders({
    chatId: '7502424413',
    conversationId: 'tc1',
    promptMessageId: 123,
    historyTurns: 4,
    requestId: 'abc',
    deferredDelivery: true,
    apiKey: 'secret',
  });
  assert.equal(headers['X-Vicuna-Telegram-Chat-Id'], '7502424413');
  assert.equal(headers['X-Vicuna-Telegram-Conversation-Id'], 'tc1');
  assert.equal(headers['X-Vicuna-Telegram-Message-Id'], '123');
  assert.equal(headers['X-Vicuna-Telegram-History-Turns'], '4');
  assert.equal(headers['X-Vicuna-Telegram-Deferred-Delivery'], '1');
  assert.equal(headers.Authorization, 'Bearer secret');
});

test('extractAssistantText pulls plain assistant content from host responses', () => {
  assert.equal(extractAssistantText({
    choices: [{ message: { content: 'Ready.' } }],
  }), 'Ready.');
  assert.equal(extractAssistantText({
    choices: [{ message: { content: null } }],
  }), '');
});

test('delivery satisfaction accepts queued async video follow-up once text is delivered', () => {
  const baseState = {
    telegramOutboxDeliveryReceipt: {
      sequenceNumber: 41,
      chatId: '7502424413',
      replyToMessageId: 333,
      deliveryMode: 'reply',
      telegramMessageId: 444,
      deliveredAtMs: 1,
      animation: {
        requested: true,
        status: 'sent',
        stage: 'complete',
        keyframeCount: 10,
        durationSeconds: 2.4,
      },
    },
    pendingVideoDeliveries: {},
  };
  const satisfied = isDeliverySatisfied(baseState, {
    chatId: '7502424413',
    promptMessageId: 333,
    minimumSequenceNumber: 40,
    waitForVideo: true,
  });
  assert.equal(satisfied.done, true);

  const pending = isDeliverySatisfied({
    ...baseState,
    telegramOutboxDeliveryReceipt: {
      ...baseState.telegramOutboxDeliveryReceipt,
      animation: {
        requested: true,
        status: 'queued',
        stage: 'render',
        keyframeCount: 10,
        durationSeconds: 2.4,
      },
    },
    pendingVideoDeliveries: {
      job1: {
        sequenceNumber: 41,
      },
    },
  }, {
    chatId: '7502424413',
    promptMessageId: 333,
    minimumSequenceNumber: 40,
    waitForVideo: true,
  });
  assert.equal(pending.done, true);
  assert.equal(pending.reason, 'video_queued');
});

test('capture advancement requires all three host JSONL surfaces', async () => {
  const fixtureDir = path.join(process.cwd(), 'tools', 'ops', 'tests', 'fixtures-agentic-harness');
  await import('node:fs/promises').then(async ({ mkdir, writeFile, rm }) => {
    await rm(fixtureDir, { recursive: true, force: true });
    await mkdir(fixtureDir, { recursive: true });
    await writeFile(path.join(fixtureDir, 'transitions.jsonl'), '{"a":1}\n');
    await writeFile(path.join(fixtureDir, 'decode_traces.jsonl'), '{"a":1}\n{"b":2}\n');
    await writeFile(path.join(fixtureDir, 'emotive_traces.jsonl'), '{"a":1}\n');
  });
  const before = await snapshotCaptureState(fixtureDir);
  await import('node:fs/promises').then(async ({ writeFile }) => {
    await writeFile(path.join(fixtureDir, 'transitions.jsonl'), '{"a":1}\n{"b":2}\n');
    await writeFile(path.join(fixtureDir, 'decode_traces.jsonl'), '{"a":1}\n{"b":2}\n{"c":3}\n');
    await writeFile(path.join(fixtureDir, 'emotive_traces.jsonl'), '{"a":1}\n{"b":2}\n');
  });
  const after = await snapshotCaptureState(fixtureDir);
  assert.deepEqual(captureAdvanced(before, after), {
    transitions: true,
    decode: true,
    emotive: true,
  });
  assert.equal(isCaptureSatisfied(before, after), true);
});

test('sample request file stays in safe non-media domains', async () => {
  const samplePath = path.join(process.cwd(), 'tools', 'ops', 'examples', 'telegram-agentic-training-sample.json');
  const sample = JSON.parse(await readFile(samplePath, 'utf8'));
  const joined = sample.requests.map((entry) => String(entry.text ?? '')).join('\n').toLowerCase();
  assert.equal(joined.includes('delete movie'), false);
  assert.equal(joined.includes('add movie'), false);
  assert.equal(joined.includes('delete book'), false);
  assert.equal(joined.includes('add book'), false);
});
