#!/usr/bin/env node

import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  buildTelegramChatCompletionRequest,
  loadState,
  DEFAULT_TELEGRAM_BRIDGE_REQUEST_TIMEOUT_MS,
} from '../telegram-bridge/lib.mjs';

const DEFAULT_MODEL = 'vicuna-experimental';
const DEFAULT_CHAT_COMPLETION_TIMEOUT_MS = DEFAULT_TELEGRAM_BRIDGE_REQUEST_TIMEOUT_MS;
const DEFAULT_DELIVERY_TIMEOUT_MS = 300000;
const DEFAULT_CAPTURE_TIMEOUT_MS = 120000;
const DEFAULT_POLL_INTERVAL_MS = 500;
const DEFAULT_CAPTURE_DIR = '/var/lib/vicuna/experimental-capture/live';
const DEFAULT_BRIDGE_STATE_PATH = '/var/lib/vicuna/telegram-bridge-state.json';
const MAX_REQUESTS_PER_RUN = 256;

function parseInteger(value, fallback) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseBoolean(value, fallback = false) {
  if (typeof value === 'boolean') {
    return value;
  }
  const normalized = String(value ?? '').trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (['1', 'true', 'yes', 'on'].includes(normalized)) {
    return true;
  }
  if (['0', 'false', 'no', 'off'].includes(normalized)) {
    return false;
  }
  return fallback;
}

function nowIso() {
  return new Date().toISOString();
}

function createHarnessRequestId(prefix = 'tgh') {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function requireString(value, fieldName) {
  const normalized = String(value ?? '').trim();
  if (!normalized) {
    throw new Error(`${fieldName} must be a non-empty string`);
  }
  return normalized;
}

function ensureRequestList(value) {
  if (!Array.isArray(value) || value.length <= 0) {
    throw new Error('requests must be a non-empty array');
  }
  if (value.length > MAX_REQUESTS_PER_RUN) {
    throw new Error(`requests exceeds max supported batch size of ${MAX_REQUESTS_PER_RUN}`);
  }
  return value;
}

export function parseHarnessConfig(raw, env = process.env) {
  const payload = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const chatId = requireString(
    payload.chatId ?? payload.chat_id ?? env.TELEGRAM_AGENTIC_HARNESS_CHAT_ID,
    'chatId',
  );
  const requests = ensureRequestList(payload.requests);
  const conversationId = String(
    payload.conversationId ?? payload.conversation_id ?? env.TELEGRAM_AGENTIC_HARNESS_CONVERSATION_ID ?? '',
  ).trim();
  const hostBaseUrl = requireString(
    payload.hostBaseUrl ?? payload.host_base_url ?? env.TELEGRAM_BRIDGE_VICUNA_BASE_URL ?? 'http://127.0.0.1:8080',
    'hostBaseUrl',
  ).replace(/\/+$/, '');
  const model = String(payload.model ?? env.TELEGRAM_BRIDGE_MODEL ?? DEFAULT_MODEL).trim() || DEFAULT_MODEL;
  const maxTokens = Math.max(32, parseInteger(payload.maxTokens ?? payload.max_tokens ?? env.TELEGRAM_BRIDGE_MAX_TOKENS, 1024));
  const deferredDelivery = parseBoolean(payload.deferredDelivery ?? payload.deferred_delivery, true);
  const verifyCapture = parseBoolean(payload.verifyCapture ?? payload.verify_capture, true);
  const captureDir = String(payload.captureDir ?? payload.capture_dir ?? env.VICUNA_EXPERIMENTAL_CAPTURE_DIR ?? DEFAULT_CAPTURE_DIR).trim() || DEFAULT_CAPTURE_DIR;
  const statePath = String(payload.statePath ?? payload.state_path ?? env.TELEGRAM_BRIDGE_STATE_PATH ?? DEFAULT_BRIDGE_STATE_PATH).trim() || DEFAULT_BRIDGE_STATE_PATH;
  const requestTimeoutMs = Math.max(1000, parseInteger(payload.requestTimeoutMs ?? payload.request_timeout_ms ?? env.TELEGRAM_AGENTIC_HARNESS_REQUEST_TIMEOUT_MS, DEFAULT_CHAT_COMPLETION_TIMEOUT_MS));
  const deliveryTimeoutMs = Math.max(1000, parseInteger(payload.deliveryTimeoutMs ?? payload.delivery_timeout_ms ?? env.TELEGRAM_AGENTIC_HARNESS_DELIVERY_TIMEOUT_MS, DEFAULT_DELIVERY_TIMEOUT_MS));
  const captureTimeoutMs = Math.max(1000, parseInteger(payload.captureTimeoutMs ?? payload.capture_timeout_ms ?? env.TELEGRAM_AGENTIC_HARNESS_CAPTURE_TIMEOUT_MS, DEFAULT_CAPTURE_TIMEOUT_MS));
  const pollIntervalMs = Math.max(100, parseInteger(payload.pollIntervalMs ?? payload.poll_interval_ms ?? env.TELEGRAM_AGENTIC_HARNESS_POLL_INTERVAL_MS, DEFAULT_POLL_INTERVAL_MS));
  const outputPath = String(payload.outputPath ?? payload.output_path ?? '').trim();
  return {
    chatId,
    conversationId,
    hostBaseUrl,
    model,
    maxTokens,
    deferredDelivery,
    verifyCapture,
    captureDir,
    statePath,
    requestTimeoutMs,
    deliveryTimeoutMs,
    captureTimeoutMs,
    pollIntervalMs,
    outputPath,
    requests: requests.map((entry, index) => {
      const item = entry && typeof entry === 'object' && !Array.isArray(entry) ? entry : {};
      const text = requireString(item.text ?? item.prompt, `requests[${index}].text`);
      return {
        id: String(item.id ?? `turn_${index + 1}`).trim() || `turn_${index + 1}`,
        text,
        visibleText: String(item.visibleText ?? item.visible_text ?? text).trim() || text,
        waitForVideo: parseBoolean(item.waitForVideo ?? item.wait_for_video, true),
        verifyCapture: parseBoolean(item.verifyCapture ?? item.verify_capture, verifyCapture),
      };
    }),
  };
}

export function buildHostHeaders({
  chatId,
  conversationId = '',
  promptMessageId,
  historyTurns,
  requestId,
  deferredDelivery = true,
  apiKey = '',
}) {
  const headers = {
    'Content-Type': 'application/json',
    'X-Client-Request-Id': requestId,
    'X-Vicuna-Telegram-Chat-Id': String(chatId),
    'X-Vicuna-Telegram-Message-Id': String(Math.max(0, Number(promptMessageId ?? 0) || 0)),
    'X-Vicuna-Telegram-History-Turns': String(Math.max(1, Number(historyTurns ?? 1) || 1)),
  };
  if (deferredDelivery) {
    headers['X-Vicuna-Telegram-Deferred-Delivery'] = '1';
  }
  if (conversationId) {
    headers['X-Vicuna-Telegram-Conversation-Id'] = conversationId;
  }
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

export function extractAssistantText(responseBody) {
  const content = responseBody?.choices?.[0]?.message?.content;
  return typeof content === 'string' ? content.trim() : '';
}

async function readJsonFile(filePath) {
  return JSON.parse(await readFile(filePath, 'utf8'));
}

async function fileLineCount(filePath) {
  try {
    const text = await readFile(filePath, 'utf8');
    const trimmed = text.trim();
    return trimmed ? trimmed.split('\n').length : 0;
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return 0;
    }
    throw error;
  }
}

export async function snapshotCaptureState(captureDir) {
  const transitionsPath = path.join(captureDir, 'transitions.jsonl');
  const decodePath = path.join(captureDir, 'decode_traces.jsonl');
  const emotivePath = path.join(captureDir, 'emotive_traces.jsonl');
  return {
    dir: captureDir,
    transitionsPath,
    decodePath,
    emotivePath,
    transitions: await fileLineCount(transitionsPath),
    decode: await fileLineCount(decodePath),
    emotive: await fileLineCount(emotivePath),
  };
}

export function captureAdvanced(before, after) {
  return {
    transitions: after.transitions > before.transitions,
    decode: after.decode > before.decode,
    emotive: after.emotive > before.emotive,
  };
}

export function isCaptureSatisfied(before, after) {
  const advanced = captureAdvanced(before, after);
  return advanced.transitions && advanced.decode && advanced.emotive;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function telegramRequest(botToken, method, payload) {
  const response = await fetch(`https://api.telegram.org/bot${botToken}/${method}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok || !body.ok) {
    throw new Error(`telegram ${method} failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body.result;
}

async function sendVisiblePromptMessage(botToken, chatId, text) {
  return telegramRequest(botToken, 'sendMessage', {
    chat_id: chatId,
    text: String(text ?? '').trim(),
  });
}

export async function assertExperimentalMode(hostBaseUrl, apiKey = '') {
  const response = await fetch(`${hostBaseUrl}/health`, {
    method: 'GET',
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`host /health failed: ${response.status} ${JSON.stringify(body)}`);
  }
  const mode = String(body?.runpod_inference?.host_inference_mode ?? '');
  const role = String(body?.runpod_inference?.role ?? '');
  const enabled = body?.runpod_inference?.enabled === true;
  if (!(enabled && role === 'host' && mode === 'experimental')) {
    throw new Error(`host is not in experimental RunPod relay mode (enabled=${enabled}, role=${role}, mode=${mode})`);
  }
  return body;
}

function findPendingVideoForSequence(state, sequenceNumber) {
  const deliveries = Object.values(state?.pendingVideoDeliveries ?? {});
  return deliveries.find((entry) => Number(entry?.sequenceNumber ?? 0) === Number(sequenceNumber ?? 0)) ?? null;
}

export function isDeliverySatisfied(state, {
  chatId,
  promptMessageId,
  minimumSequenceNumber = 0,
  waitForVideo = true,
}) {
  const receipt = state?.telegramOutboxDeliveryReceipt ?? null;
  if (!receipt) {
    return { done: false, reason: 'missing_receipt', receipt: null, pendingVideo: null };
  }
  if (String(receipt.chatId ?? '') !== String(chatId)) {
    return { done: false, reason: 'receipt_chat_mismatch', receipt, pendingVideo: null };
  }
  if (Number(receipt.sequenceNumber ?? 0) <= Number(minimumSequenceNumber ?? 0)) {
    return { done: false, reason: 'receipt_not_advanced', receipt, pendingVideo: null };
  }
  if (Number(receipt.replyToMessageId ?? 0) !== Number(promptMessageId ?? 0)) {
    return { done: false, reason: 'reply_anchor_mismatch', receipt, pendingVideo: null };
  }
  const pendingVideo = findPendingVideoForSequence(state, receipt.sequenceNumber);
  const animation = receipt.animation ?? null;
  if (!waitForVideo) {
    return { done: true, reason: 'text_delivered', receipt, pendingVideo };
  }
  if (!animation || animation.requested !== true) {
    return { done: true, reason: pendingVideo ? 'video_pending_async' : 'no_video_requested', receipt, pendingVideo };
  }
  const animationStatus = String(animation.status ?? '').trim().toLowerCase();
  if (['queued', 'render', 'sent', 'skipped', 'failed'].includes(animationStatus)) {
    return { done: true, reason: `video_${animation.status}`, receipt, pendingVideo };
  }
  return { done: false, reason: 'video_not_terminal', receipt, pendingVideo };
}

export async function waitForDeliveryState({
  statePath,
  chatId,
  promptMessageId,
  minimumSequenceNumber = 0,
  waitForVideo = true,
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  timeoutMs = DEFAULT_DELIVERY_TIMEOUT_MS,
}) {
  const startedAt = Date.now();
  for (;;) {
    const state = await loadState(statePath);
    const result = isDeliverySatisfied(state, {
      chatId,
      promptMessageId,
      minimumSequenceNumber,
      waitForVideo,
    });
    if (result.done) {
      return {
        ...result,
        state,
        elapsedMs: Date.now() - startedAt,
      };
    }
    if ((Date.now() - startedAt) >= timeoutMs) {
      throw new Error(`delivery wait timed out after ${timeoutMs}ms (${result.reason})`);
    }
    await sleep(pollIntervalMs);
  }
}

export async function waitForCaptureAdvance({
  captureDir,
  before,
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  timeoutMs = DEFAULT_CAPTURE_TIMEOUT_MS,
}) {
  const startedAt = Date.now();
  for (;;) {
    const after = await snapshotCaptureState(captureDir);
    if (isCaptureSatisfied(before, after)) {
      return {
        before,
        after,
        advanced: captureAdvanced(before, after),
        elapsedMs: Date.now() - startedAt,
      };
    }
    if ((Date.now() - startedAt) >= timeoutMs) {
      const advanced = captureAdvanced(before, after);
      throw new Error(`capture verification timed out after ${timeoutMs}ms (transitions=${advanced.transitions}, decode=${advanced.decode}, emotive=${advanced.emotive})`);
    }
    await sleep(pollIntervalMs);
  }
}

async function invokeHostTurn({
  hostBaseUrl,
  apiKey,
  chatId,
  conversationId,
  promptMessageId,
  transcript,
  model,
  maxTokens,
  deferredDelivery,
  requestTimeoutMs,
  requestId,
}) {
  const historyTurns = Math.max(1, Math.ceil(transcript.length / 2));
  const response = await fetch(`${hostBaseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: buildHostHeaders({
      chatId,
      conversationId,
      promptMessageId,
      historyTurns,
      requestId,
      deferredDelivery,
      apiKey,
    }),
    body: JSON.stringify(buildTelegramChatCompletionRequest({
      model,
      transcript,
      maxTokens,
    })),
    signal: AbortSignal.timeout(requestTimeoutMs),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`host chat completions failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return {
    requestId: response.headers.get('x-client-request-id') || response.headers.get('x-request-id') || requestId,
    body,
  };
}

async function ensureReadable(pathname) {
  const info = await stat(pathname);
  if (!info.isFile()) {
    throw new Error(`${pathname} is not a file`);
  }
}

export async function runHarness(config, env = process.env) {
  const botToken = requireString(env.TELEGRAM_BOT_TOKEN, 'TELEGRAM_BOT_TOKEN');
  const apiKey = String(env.VICUNA_API_KEY ?? '').trim();
  await assertExperimentalMode(config.hostBaseUrl, apiKey);

  const transcript = [];
  const report = {
    schemaVersion: 'vicuna.telegram_agentic_harness.v1',
    startedAt: nowIso(),
    chatId: config.chatId,
    conversationId: config.conversationId,
    hostBaseUrl: config.hostBaseUrl,
    model: config.model,
    turns: [],
  };

  let minimumSequenceNumber = 0;
  for (const turn of config.requests) {
    const beforeState = await loadState(config.statePath);
    const beforeReceipt = beforeState.telegramOutboxDeliveryReceipt ?? null;
    minimumSequenceNumber = Math.max(minimumSequenceNumber, Number(beforeReceipt?.sequenceNumber ?? 0) || 0);
    const captureBefore = turn.verifyCapture ? await snapshotCaptureState(config.captureDir) : null;

    const sentPrompt = await sendVisiblePromptMessage(botToken, config.chatId, turn.visibleText);
    const promptMessageId = Number(sentPrompt?.message_id ?? 0) || 0;
    if (promptMessageId <= 0) {
      throw new Error(`failed to obtain Telegram prompt message id for turn ${turn.id}`);
    }

    transcript.push({
      role: 'user',
      content: turn.text,
    });

    const startedAtMs = Date.now();
    const hostResult = await invokeHostTurn({
      hostBaseUrl: config.hostBaseUrl,
      apiKey,
      chatId: config.chatId,
      conversationId: config.conversationId,
      promptMessageId,
      transcript,
      model: config.model,
      maxTokens: config.maxTokens,
      deferredDelivery: config.deferredDelivery,
      requestTimeoutMs: config.requestTimeoutMs,
      requestId: createHarnessRequestId('hreq'),
    });

    const assistantText = extractAssistantText(hostResult.body);
    if (assistantText) {
      transcript.push({
        role: 'assistant',
        content: assistantText,
      });
    }

    const delivery = await waitForDeliveryState({
      statePath: config.statePath,
      chatId: config.chatId,
      promptMessageId,
      minimumSequenceNumber,
      waitForVideo: turn.waitForVideo,
      pollIntervalMs: config.pollIntervalMs,
      timeoutMs: config.deliveryTimeoutMs,
    });
    minimumSequenceNumber = Math.max(minimumSequenceNumber, Number(delivery.receipt?.sequenceNumber ?? 0) || 0);

    const capture = turn.verifyCapture
      ? await waitForCaptureAdvance({
        captureDir: config.captureDir,
        before: captureBefore,
        pollIntervalMs: config.pollIntervalMs,
        timeoutMs: config.captureTimeoutMs,
      })
      : null;

    report.turns.push({
      id: turn.id,
      promptText: turn.text,
      visiblePromptText: turn.visibleText,
      promptTelegramMessageId: promptMessageId,
      hostRequestId: hostResult.requestId,
      responseText: assistantText,
      queuedTelegramDelivery: hostResult.body?.vicuna_telegram_delivery?.queued === true,
      runtimeDelivery: hostResult.body?.vicuna_telegram_delivery ?? null,
      deliveryReceipt: delivery.receipt,
      deliveryElapsedMs: delivery.elapsedMs,
      pendingVideo: delivery.pendingVideo,
      capture,
      assistantTranscriptSkipped: !assistantText,
      elapsedMs: Date.now() - startedAtMs,
    });
  }

  report.completedAt = nowIso();
  report.turnCount = report.turns.length;
  return report;
}

function usage() {
  console.error('usage: telegram-agentic-harness.mjs <input.json> [output.json]');
}

async function main(argv = process.argv.slice(2), env = process.env) {
  if (argv.length < 1) {
    usage();
    process.exitCode = 1;
    return;
  }
  const inputPath = path.resolve(argv[0]);
  const outputPathArg = argv[1] ? path.resolve(argv[1]) : '';
  await ensureReadable(inputPath);
  const raw = await readJsonFile(inputPath);
  const config = parseHarnessConfig({
    ...raw,
    ...(outputPathArg ? { outputPath: outputPathArg } : {}),
  }, env);
  const report = await runHarness(config, env);
  const outputPath = config.outputPath || outputPathArg;
  if (outputPath) {
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  main().catch((error) => {
    console.error(String(error?.stack ?? error));
    process.exitCode = 1;
  });
}
