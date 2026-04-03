import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { access, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import {
  buildEmotiveAnimationRenderPlan,
  extractEmotiveAnimationTerminalMoment,
  prependEmotiveAnimationStartMoment,
  renderEmotiveAnimationMp4,
} from './emotive-animation-render.mjs';
import {
  buildWebglRenderTimeoutMs,
  getWebglRendererHealth,
  renderEmotiveAnimationViaWebglService,
} from './emotive-webgl-renderer-client.mjs';
import { handleBridgeControlCommand } from './control-commands.mjs';
import {
  appendProactiveId,
  appendChatTranscriptMessage,
  buildTelegramPlainTextFallbackPayload,
  buildTelegramAnimationDeliveryPlan,
  buildTelegramBridgeRequestTimeoutMs,
  buildTelegramMessageChunkPlan,
  buildTelegramChatCompletionRequest,
  buildTelegramFileUrl,
  bootstrapTelegramOutboxOffset,
  deleteTelegramPendingVideoDelivery,
  deletePendingOptionPrompt,
  enqueueTelegramPendingVideoDelivery,
  extractChatCompletionText,
  extractResponseText,
  formatTelegramMessage,
  getPendingOptionPrompt,
  getChatTranscript,
  getTelegramPendingVideoDelivery,
  ingestTelegramDocumentMessage,
  isTelegramReplyTargetErrorMessage,
  isTelegramTerminalDeliveryErrorMessage,
  listTelegramPendingVideoDeliveries,
  loadState,
  normalizeTelegramRichTextPayload,
  normalizeTelegramOutboxItem,
  parseInteger,
  parseSseChunk,
  recordTelegramOutboxDeliveryReceipt,
  reconcileTelegramOutboxOffset,
  registerChat,
  resolveTelegramConversationForMessage,
  resolveTelegramConversationForOutbound,
  saveState,
  getTelegramEmotiveAnimationState,
  setTelegramEmotiveAnimationState,
  setTelegramOutboxCheckpoint,
  shouldSuppressDeferredTelegramFailureMessage,
  setPendingOptionPrompt,
  shouldBootstrapTelegramOutboxOffset,
  splitSseBuffer,
  TELEGRAM_BRIDGE_ASK_OUTBOX_POLL_IDLE_MS,
  DEFAULT_TELEGRAM_BRIDGE_REQUEST_TIMEOUT_MS,
  TELEGRAM_BRIDGE_SELF_EMIT_ACTIVE_DELAY_MS,
  TELEGRAM_BRIDGE_SELF_EMIT_ERROR_DELAY_MS,
  TELEGRAM_BRIDGE_WATCHDOG_DELAY_MS,
  updateTelegramPendingVideoDelivery,
  updateTelegramOffset,
} from './lib.mjs';

const execFileAsync = promisify(execFile);
const bridgeDir = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_TELEGRAM_BRIDGE_MAX_TOKENS = 1024;
const MAX_TELEGRAM_BRIDGE_MAX_TOKENS = 4096;

const env = {
  telegramBotToken: process.env.TELEGRAM_BOT_TOKEN ?? '',
  vicunaBaseUrl: (process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL ?? 'http://127.0.0.1:8080').replace(/\/+$/, ''),
  model: process.env.TELEGRAM_BRIDGE_MODEL ?? process.env.VICUNA_DEEPSEEK_MODEL ?? 'deepseek-chat',
  statePath: process.env.TELEGRAM_BRIDGE_STATE_PATH ?? '/tmp/vicuna-telegram-bridge-state.json',
  pollTimeoutSeconds: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS, 30)),
  requestTimeoutMs: buildTelegramBridgeRequestTimeoutMs(
    process.env.TELEGRAM_BRIDGE_REQUEST_TIMEOUT_MS ?? DEFAULT_TELEGRAM_BRIDGE_REQUEST_TIMEOUT_MS,
  ),
  maxHistoryMessages: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES, 12)),
  maxTokens: (() => {
    const configured = parseInteger(process.env.TELEGRAM_BRIDGE_MAX_TOKENS, DEFAULT_TELEGRAM_BRIDGE_MAX_TOKENS);
    return Math.max(32, Math.min(MAX_TELEGRAM_BRIDGE_MAX_TOKENS, configured));
  })(),
  maxDocumentChars: Math.max(256, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_DOCUMENT_CHARS, 12000)),
  selfEmitAfter: Math.max(0, parseInteger(process.env.TELEGRAM_BRIDGE_SELF_EMIT_AFTER, 0)),
  replayRetainedOutbox: String(process.env.TELEGRAM_BRIDGE_REPLAY_RETAINED_OUTBOX ?? '').trim() === '1',
  vicunaApiKey: process.env.VICUNA_API_KEY ?? '',
  hardMemoryRuntimeIdentity: (process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY ?? 'vicuna').trim() || 'vicuna',
  docsRoot: (
    process.env.VICUNA_DOCS_DIR ??
    path.join(process.env.VICUNA_HOST_SHELL_ROOT ?? '/home/vicuna/home', 'docs')
  ).trim(),
  documentContainerTag: (
    process.env.TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG ??
    `${(process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY ?? 'vicuna').trim() || 'vicuna'}-telegram-documents`
  ).trim() || 'vicuna-telegram-documents',
  doclingPythonBin: process.env.TELEGRAM_BRIDGE_DOCLING_PYTHON_BIN ?? 'python3',
  doclingParserScriptPath:
    process.env.TELEGRAM_BRIDGE_DOCLING_PARSER_SCRIPT_PATH ?? path.join(bridgeDir, 'docling-parse.py'),
  ffmpegBin: process.env.TELEGRAM_BRIDGE_FFMPEG_BIN ?? 'ffmpeg',
  ffmpegVideoEncoder: process.env.TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER ?? 'h264_nvenc',
  renderBackend: (process.env.TELEGRAM_BRIDGE_RENDER_BACKEND ?? 'cpu_canvas').trim() || 'cpu_canvas',
  webglRendererUrl: (process.env.VICUNA_WEBGL_RENDERER_URL ?? 'http://127.0.0.1:8091').replace(/\/+$/, ''),
  webglRenderTimeoutMs: Math.max(1000, parseInteger(process.env.TELEGRAM_BRIDGE_WEBGL_RENDER_TIMEOUT_MS, 120000)),
  webglRenderMaxAttempts: Math.max(1, Math.min(5, parseInteger(process.env.TELEGRAM_BRIDGE_WEBGL_RENDER_MAX_ATTEMPTS, 3))),
  videoSpoolDir: (
    process.env.TELEGRAM_BRIDGE_VIDEO_SPOOL_DIR ??
    path.join(path.dirname(process.env.TELEGRAM_BRIDGE_STATE_PATH ?? '/tmp/vicuna-telegram-bridge-state.json'), 'telegram-video-spool')
  ).trim(),
  videoDeliveryRetryBaseMs: Math.max(250, parseInteger(process.env.TELEGRAM_BRIDGE_VIDEO_RETRY_BASE_MS, 1000)),
  videoDeliveryMaxAttempts: Math.max(0, parseInteger(process.env.TELEGRAM_BRIDGE_VIDEO_MAX_ATTEMPTS, 0)),
  videoPollIdleMs: Math.max(100, parseInteger(process.env.TELEGRAM_BRIDGE_VIDEO_POLL_IDLE_MS, 500)),
};

if (!env.telegramBotToken) {
  throw new Error('TELEGRAM_BOT_TOKEN is required');
}

let state = await loadState(env.statePath, { maxHistoryMessages: env.maxHistoryMessages });
let selfEmitStreamActive = false;
let selfEmitLastConnectedAt = 0;
let selfEmitLastEventAt = 0;
let selfEmitGeneration = 0;

const telegramBaseUrl = `https://api.telegram.org/bot${env.telegramBotToken}`;
const vicunaUrl = new URL(env.vicunaBaseUrl);
const sseHttpModule = vicunaUrl.protocol === 'https:' ? https : http;

function log(message, extra = undefined) {
  const payload = {
    schema_version: 'vicuna.service_event.v1',
    timestamp_ms: Date.now(),
    service: 'telegram-bridge',
    event: 'log',
    message,
  };
  if (extra !== undefined) {
    if (extra && typeof extra === 'object' && !Array.isArray(extra)) {
      Object.assign(payload, extra);
    } else {
      payload.payload = extra;
    }
  }
  console.log(JSON.stringify(payload));
}

function createBridgeRequestId(prefix = 'tg') {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function logEvent(event, fields = {}) {
  console.log(JSON.stringify({
    schema_version: 'vicuna.service_event.v1',
    timestamp_ms: Date.now(),
    service: 'telegram-bridge',
    event,
    ...fields,
  }));
}

function transcriptSummary(chatId, conversationId = '') {
  const transcript = getChatTranscript(state, chatId, conversationId ? { conversationId } : {});
  return {
    chatId: String(chatId),
    ...(conversationId ? { conversationId: String(conversationId) } : {}),
    messageCount: transcript.length,
    roles: transcript.map((entry) => entry.role).join(','),
    lastPreview: transcript.length > 0 ? transcript[transcript.length - 1].content.slice(0, 120) : '',
  };
}

function vicunaHeaders() {
  const headers = {
    'Content-Type': 'application/json',
  };
  if (env.vicunaApiKey) {
    headers.Authorization = `Bearer ${env.vicunaApiKey}`;
  }
  return headers;
}

async function postVicunaJson(pathname, payload, options = {}) {
  const response = await fetch(`${env.vicunaBaseUrl}${pathname}`, {
    method: 'POST',
    headers: {
      ...vicunaHeaders(),
      ...(options.headers ?? {}),
    },
    body: JSON.stringify(payload ?? {}),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`vicuna ${pathname} failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body;
}

async function telegramRequest(method, payload, options = {}) {
  const multipart = options.multipart === true;
  let response;
  try {
    response = await fetch(`${telegramBaseUrl}/${method}`, {
      method: 'POST',
      ...(multipart
        ? { body: payload }
        : {
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        }),
    });
  } catch (error) {
    const causeCode = typeof error?.cause?.code === 'string' && error.cause.code.trim()
      ? ` (${error.cause.code.trim()})`
      : '';
    throw new Error(`telegram ${method} transport failed: ${String(error?.message ?? error)}${causeCode}`);
  }
  const body = await response.json();
  if (!response.ok || !body.ok) {
    throw new Error(`telegram ${method} failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body.result;
}

function isTelegramTransportFailureMessage(message) {
  const value = String(message ?? '').trim();
  return value.startsWith('telegram sendVideo transport failed:') || value === 'fetch failed';
}

async function telegramSendVideoRequestWithRetry(buildForm, context = {}) {
  const maxAttempts = 3;
  let lastError = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await telegramRequest('sendVideo', buildForm(), { multipart: true });
    } catch (error) {
      lastError = error;
      if (!isTelegramTransportFailureMessage(error?.message) || attempt >= maxAttempts) {
        throw error;
      }
      log('Telegram video upload transport failed; retrying', {
        ...context,
        attempt,
        maxAttempts,
        error: String(error?.message ?? error),
      });
      await delay(250 * attempt);
    }
  }
  throw lastError ?? new Error('telegram sendVideo transport failed: retry budget exhausted');
}

async function downloadTelegramFile(filePath) {
  const response = await fetch(buildTelegramFileUrl(env.telegramBotToken, filePath));
  if (!response.ok) {
    throw new Error(`telegram file download failed: ${response.status}`);
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  return buffer;
}

async function parseTelegramDocumentWithDocling(fileBuffer, descriptor) {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'vicuna-telegram-doc-'));
  const safeFileName = path.basename(String(descriptor?.fileName ?? 'telegram-document'));
  const inputPath = path.join(tempDir, safeFileName);
  try {
    await access(env.doclingParserScriptPath);
    await writeFile(inputPath, fileBuffer);
    const { stdout, stderr } = await execFileAsync(env.doclingPythonBin, [
      env.doclingParserScriptPath,
      inputPath,
    ], {
      maxBuffer: 32 * 1024 * 1024,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });
    try {
      return JSON.parse(stdout);
    } catch (error) {
      throw new Error(
        `Docling parser returned invalid JSON: ${stderr?.trim() || (error instanceof Error ? error.message : String(error))}`,
      );
    }
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      if (error.path === env.doclingParserScriptPath) {
        throw new Error(`Docling parser helper is missing at ${env.doclingParserScriptPath}.`);
      }
      throw new Error(`Docling parsing requires ${env.doclingPythonBin} on the bridge host.`);
    }
    const stderr = typeof error?.stderr === 'string' ? error.stderr.trim() : '';
    const stdout = typeof error?.stdout === 'string' ? error.stdout.trim() : '';
    throw new Error(stderr || stdout || error.message || 'Docling parsing failed.');
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function sendTelegramMessage(chatId, text, extra = {}) {
  const messageText = String(text ?? '').trim();
  if (!messageText) {
    return;
  }
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      ...extra,
      text: messageText,
    },
    text: messageText,
  });
  return (await sendTelegramTextPayloadWithFallback(chatId, {
    chat_id: chatId,
    ...normalized.payload,
  }, {
    replyToMessageId: Number(extra?.reply_to_message_id ?? extra?.reply_parameters?.message_id ?? 0) || 0,
    fallbackText: messageText,
  })).sent;
}

function appendTelegramMultipartField(form, key, value) {
  if (value === undefined || value === null) {
    return;
  }
  if (typeof value === 'object' && !(value instanceof Blob)) {
    form.set(key, JSON.stringify(value));
    return;
  }
  form.set(key, String(value));
}

function cloneTelegramPayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return {};
  }
  return JSON.parse(JSON.stringify(payload));
}

function getTelegramReplyAnchorId(payload, fallbackReplyToMessageId = 0) {
  const payloadReplyToMessageId = Number(payload?.reply_to_message_id ?? 0) || 0;
  if (payloadReplyToMessageId > 0) {
    return payloadReplyToMessageId;
  }
  const replyParametersMessageId = Number(payload?.reply_parameters?.message_id ?? 0) || 0;
  if (replyParametersMessageId > 0) {
    return replyParametersMessageId;
  }
  return Math.max(0, Number(fallbackReplyToMessageId ?? 0) || 0);
}

function withoutTelegramReplyAnchor(payload) {
  const nextPayload = cloneTelegramPayload(payload);
  delete nextPayload.reply_to_message_id;
  delete nextPayload.reply_parameters;
  return nextPayload;
}

function getTelegramResultMessageId(result) {
  if (Array.isArray(result)) {
    return Number(result[0]?.message_id ?? 0) || 0;
  }
  return Number(result?.message_id ?? 0) || 0;
}

function getPrimaryTelegramMessageId(telegramMessageIds, sent) {
  const ids = Array.isArray(telegramMessageIds)
    ? telegramMessageIds.map((value) => Number(value ?? 0) || 0).filter((value) => value > 0)
    : [];
  if (ids.length > 0) {
    return ids[ids.length - 1];
  }
  return getTelegramResultMessageId(sent);
}

function isTelegramEntityParseErrorMessage(message) {
  const normalized = String(message ?? '').trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return normalized.includes("can't parse entities") ||
    normalized.includes('cant parse entities') ||
    normalized.includes('unsupported start tag') ||
    normalized.includes('unexpected end tag') ||
    normalized.includes('must be escaped') ||
    normalized.includes('entity beginning at byte offset');
}

async function sendTelegramTextPayloadWithFallback(chatId, payload, options = {}) {
  const replyToMessageId = Math.max(0, Number(options.replyToMessageId ?? 0) || 0);
  const fallbackText = typeof options.fallbackText === 'string' ? options.fallbackText : '';
  const attemptPayload = cloneTelegramPayload(payload);
  const requestedReplyToMessageId = getTelegramReplyAnchorId(attemptPayload, replyToMessageId);
  if (requestedReplyToMessageId > 0 && attemptPayload.reply_to_message_id === undefined && attemptPayload.reply_parameters === undefined) {
    attemptPayload.reply_parameters = {
      message_id: requestedReplyToMessageId,
    };
  }
  try {
    return {
      sent: await telegramRequest('sendMessage', attemptPayload),
      requestedReplyToMessageId,
      fallbackMode: 'none',
      fallbackError: '',
    };
  } catch (error) {
    if (isTelegramReplyTargetErrorMessage(error?.message)) {
      const fallbackError = error.message;
      log('Telegram follow-up reply target rejected; retrying without reply anchor', {
        method: 'sendMessage',
        chatId: String(chatId),
        replyToMessageId: requestedReplyToMessageId,
        error: fallbackError,
      });
      return {
        sent: await telegramRequest('sendMessage', withoutTelegramReplyAnchor(attemptPayload)),
        requestedReplyToMessageId,
        fallbackMode: 'reply_anchor_removed',
        fallbackError,
      };
    }
    if (!isTelegramEntityParseErrorMessage(error?.message)) {
      throw error;
    }
    const fallbackError = error.message;
    const plainTextPayload = buildTelegramPlainTextFallbackPayload({
      telegramMethod: 'sendMessage',
      telegramPayload: attemptPayload,
      text: fallbackText,
    }).payload;
    plainTextPayload.chat_id = chatId;
    if (requestedReplyToMessageId > 0 && plainTextPayload.reply_to_message_id === undefined && plainTextPayload.reply_parameters === undefined) {
      plainTextPayload.reply_parameters = {
        message_id: requestedReplyToMessageId,
      };
    }
    log('Telegram rich-text payload was rejected; retrying as plain text', {
      method: 'sendMessage',
      chatId: String(chatId),
      replyToMessageId: requestedReplyToMessageId,
      error: fallbackError,
    });
    try {
      return {
        sent: await telegramRequest('sendMessage', plainTextPayload),
        requestedReplyToMessageId,
        fallbackMode: 'plain_text',
        fallbackError,
      };
    } catch (fallbackSendError) {
      if (!isTelegramReplyTargetErrorMessage(fallbackSendError?.message)) {
        throw fallbackSendError;
      }
      log('Telegram plain-text fallback reply target rejected; retrying without reply anchor', {
        method: 'sendMessage',
        chatId: String(chatId),
        replyToMessageId: requestedReplyToMessageId,
        error: fallbackSendError.message,
      });
      return {
        sent: await telegramRequest('sendMessage', withoutTelegramReplyAnchor(plainTextPayload)),
        requestedReplyToMessageId,
        fallbackMode: 'plain_text_no_reply',
        fallbackError: fallbackSendError.message,
      };
    }
  }
}

async function sendTelegramChunkedTextOutboxMessage(chatId, payload, replyToMessageId = 0, textSource = '') {
  const chunkPlan = buildTelegramMessageChunkPlan({
    telegramPayload: payload,
    text: textSource,
  });
  if (!Array.isArray(chunkPlan.chunks) || chunkPlan.chunks.length <= 0) {
    throw new Error('Telegram outbox message payload was missing text.');
  }

  const requestedReplyToMessageId = getTelegramReplyAnchorId(payload, replyToMessageId);
  const sentMessages = [];
  let fallbackError = '';
  let usedFallbackNoReply = false;
  let usedPlainTextFallback = false;
  for (let index = 0; index < chunkPlan.chunks.length; index += 1) {
    const chunkPayload = cloneTelegramPayload(chunkPlan.chunks[index].payload);
    chunkPayload.chat_id = chatId;
    const sendResult = await sendTelegramTextPayloadWithFallback(chatId, chunkPayload, {
      replyToMessageId: index === 0 ? requestedReplyToMessageId : 0,
      fallbackText: chunkPlan.chunks[index].sourceText,
    });
    sentMessages.push(sendResult.sent);
    if (sendResult.fallbackMode === 'reply_anchor_removed' || sendResult.fallbackMode === 'plain_text_no_reply') {
      usedFallbackNoReply = true;
    }
    if (sendResult.fallbackMode === 'plain_text' || sendResult.fallbackMode === 'plain_text_no_reply') {
      usedPlainTextFallback = true;
    }
    if (sendResult.fallbackError) {
      fallbackError = sendResult.fallbackError;
    }
  }

  const telegramMessageIds = sentMessages
    .map((value) => Number(value?.message_id ?? 0) || 0)
    .filter((value) => value > 0);
  const deliveryMode = chunkPlan.chunks.length > 1
    ? (
      usedFallbackNoReply
        ? 'chunked_fallback_no_reply'
        : requestedReplyToMessageId > 0
          ? 'chunked_reply'
          : 'chunked_no_reply'
    )
    : (
      usedFallbackNoReply
        ? 'fallback_no_reply'
        : usedPlainTextFallback
          ? 'plain_text_fallback'
        : requestedReplyToMessageId > 0
          ? 'reply'
          : 'no_reply'
    );

  return {
    sent: sentMessages.length === 1 ? sentMessages[0] : sentMessages,
    sentMessages,
    telegramMessageIds,
    telegramMessageId: getPrimaryTelegramMessageId(telegramMessageIds, sentMessages),
    deliveryMode,
    requestedReplyToMessageId: requestedReplyToMessageId > 0 ? requestedReplyToMessageId : 0,
    fallbackError,
    chunkCount: sentMessages.length,
  };
}

async function sendTelegramOutboxMessage(chatId, method, payload, replyToMessageId = 0, textSource = '') {
  if (method === 'sendMessage') {
    return await sendTelegramChunkedTextOutboxMessage(chatId, payload, replyToMessageId, textSource);
  }
  const normalizedPayload = normalizeTelegramRichTextPayload({
    telegramMethod: method,
    telegramPayload: payload,
  }).payload;
  const deliveryPayload = cloneTelegramPayload(normalizedPayload);
  deliveryPayload.chat_id = chatId;

  const requestedReplyToMessageId = getTelegramReplyAnchorId(deliveryPayload, replyToMessageId);
  if (requestedReplyToMessageId > 0 && deliveryPayload.reply_to_message_id === undefined && deliveryPayload.reply_parameters === undefined) {
    deliveryPayload.reply_parameters = {
      message_id: requestedReplyToMessageId,
    };
  }

  if (requestedReplyToMessageId <= 0) {
    const sent = await telegramRequest(method, deliveryPayload);
    return {
      sent,
      deliveryMode: 'no_reply',
      requestedReplyToMessageId: 0,
      fallbackError: '',
    };
  }

  try {
    const sent = await telegramRequest(method, deliveryPayload);
    return {
      sent,
      deliveryMode: 'reply',
      requestedReplyToMessageId,
      fallbackError: '',
    };
  } catch (error) {
    if (!isTelegramReplyTargetErrorMessage(error?.message)) {
      throw error;
    }
    const fallbackError = error.message;
    log('Telegram follow-up reply target rejected; retrying without reply anchor', {
      method,
      chatId: String(chatId),
      replyToMessageId: requestedReplyToMessageId,
      error: fallbackError,
    });
    const sent = await telegramRequest(method, withoutTelegramReplyAnchor(deliveryPayload));
    return {
      sent,
      sentMessages: [sent],
      telegramMessageIds: [getTelegramResultMessageId(sent)].filter((value) => value > 0),
      telegramMessageId: getTelegramResultMessageId(sent),
      deliveryMode: 'fallback_no_reply',
      requestedReplyToMessageId,
      fallbackError,
    };
  }
}

async function sendTelegramVideoAttachment(chatId, videoPath, payload = {}, replyToMessageId = 0) {
  const deliveryPayload = cloneTelegramPayload(payload);
  deliveryPayload.chat_id = chatId;
  if (deliveryPayload.supports_streaming === undefined) {
    deliveryPayload.supports_streaming = true;
  }

  const requestedReplyToMessageId = getTelegramReplyAnchorId(deliveryPayload, replyToMessageId);
  if (requestedReplyToMessageId > 0 && deliveryPayload.reply_to_message_id === undefined && deliveryPayload.reply_parameters === undefined) {
    deliveryPayload.reply_parameters = {
      message_id: requestedReplyToMessageId,
    };
  }
  const videoBuffer = await readFile(videoPath);
  const fileName = path.basename(videoPath) || 'emotive-animation.mp4';

  const buildForm = (requestPayload) => {
    const form = new FormData();
    for (const [key, value] of Object.entries(requestPayload)) {
      appendTelegramMultipartField(form, key, value);
    }
    form.set('video', new Blob([videoBuffer], { type: 'video/mp4' }), fileName);
    return form;
  };

  if (requestedReplyToMessageId <= 0) {
    try {
      const sent = await telegramSendVideoRequestWithRetry(
        () => buildForm(deliveryPayload),
        { chatId: String(chatId), replyToMessageId: 0 },
      );
      return {
        sent,
        deliveryMode: 'no_reply',
        requestedReplyToMessageId: 0,
        fallbackError: '',
      };
    } catch (error) {
      error.stage = 'upload';
      throw error;
    }
  }

  try {
    const sent = await telegramSendVideoRequestWithRetry(
      () => buildForm(deliveryPayload),
      { chatId: String(chatId), replyToMessageId: requestedReplyToMessageId },
    );
    return {
      sent,
      deliveryMode: 'reply',
      requestedReplyToMessageId,
      fallbackError: '',
    };
  } catch (error) {
    if (!isTelegramReplyTargetErrorMessage(error?.message)) {
      error.stage = 'upload';
      throw error;
    }
    const fallbackError = error.message;
    log('Telegram video reply target rejected; retrying without reply anchor', {
      chatId: String(chatId),
      replyToMessageId: requestedReplyToMessageId,
      error: fallbackError,
    });
    try {
      const sent = await telegramSendVideoRequestWithRetry(
        () => buildForm(withoutTelegramReplyAnchor(deliveryPayload)),
        { chatId: String(chatId), replyToMessageId: 0, fallbackFromReplyAnchor: true },
      );
      return {
        sent,
        deliveryMode: 'fallback_no_reply',
        requestedReplyToMessageId,
        fallbackError,
      };
    } catch (fallbackErrorResponse) {
      fallbackErrorResponse.stage = 'upload';
      throw fallbackErrorResponse;
    }
  }
}

async function renderTelegramEmotiveAnimation(bundle, options = {}) {
  const requestedOutputPath = String(options.outputPath ?? '').trim();
  const requestId = String(options.requestId ?? '').trim();
  const outputPath = requestedOutputPath || path.join(
    await mkdtemp(path.join(os.tmpdir(), 'vicuna-telegram-animation-')),
    'emotive-animation.mp4',
  );
  const ownsTempDir = !requestedOutputPath;
  const tempDir = ownsTempDir ? path.dirname(outputPath) : '';
  try {
    const result = env.renderBackend === 'chromium_webgl'
      ? await renderEmotiveAnimationViaWebglService(bundle, {
        serviceUrl: env.webglRendererUrl,
        outputPath,
        requestId,
        timeoutMs: Number(options.timeoutMs ?? env.webglRenderTimeoutMs) || env.webglRenderTimeoutMs,
        maxAttempts: env.webglRenderMaxAttempts,
      })
      : await renderEmotiveAnimationMp4(bundle, {
        ffmpegBin: env.ffmpegBin,
        videoEncoder: env.ffmpegVideoEncoder,
        outputPath,
      });
    return {
      ...result,
      tempDir: ownsTempDir ? tempDir : '',
      outputPath,
    };
  } catch (error) {
    if (ownsTempDir) {
      await rm(tempDir, { recursive: true, force: true });
    } else {
      await removeTelegramVideoArtifact(outputPath);
    }
    throw error;
  }
}

function createTelegramPendingVideoJobId(sequenceNumber) {
  const normalizedSequence = Math.max(0, Number(sequenceNumber ?? 0) || 0);
  return `tv_${normalizedSequence.toString(36)}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function buildTelegramVideoRetryDelayMs(attemptCount) {
  const normalizedAttemptCount = Math.max(1, Number(attemptCount ?? 1) || 1);
  return Math.min(300000, env.videoDeliveryRetryBaseMs * (2 ** (normalizedAttemptCount - 1)));
}

function buildQueuedAnimationReceipt(bundle) {
  const renderPlan = buildEmotiveAnimationRenderPlan(bundle);
  return {
    requested: true,
    status: 'queued',
    stage: 'queued',
    keyframeCount: renderPlan?.keyframeCount ?? 0,
      durationSeconds: renderPlan?.durationSeconds ?? 0,
  };
}

async function ensureTelegramVideoSpoolDir() {
  await mkdir(env.videoSpoolDir, { recursive: true });
}

function buildTelegramVideoArtifactPath(jobId) {
  const normalizedJobId = String(jobId ?? '').trim().replace(/[^a-zA-Z0-9._-]+/g, '_') || 'pending-video';
  return path.join(env.videoSpoolDir, `${normalizedJobId}.mp4`);
}

async function hasUsableTelegramVideoArtifact(artifactPath) {
  const normalizedPath = String(artifactPath ?? '').trim();
  if (!normalizedPath) {
    return false;
  }
  try {
    const info = await stat(normalizedPath);
    return info.isFile() && info.size > 0;
  } catch {
    return false;
  }
}

async function removeTelegramVideoArtifact(artifactPath) {
  const normalizedPath = String(artifactPath ?? '').trim();
  if (!normalizedPath) {
    return;
  }
  await rm(normalizedPath, { force: true }).catch(() => {});
}

async function isWebglRendererReady() {
  if (env.renderBackend !== 'chromium_webgl') {
    return true;
  }
  try {
    const health = await getWebglRendererHealth(env.webglRendererUrl);
    return health?.status === 'ok' && health?.browser?.ready === true;
  } catch {
    return false;
  }
}

function isRetryableVideoDeliveryError(error) {
  const message = String(error?.message ?? error).trim();
  if (!message) {
    return true;
  }
  if (isTelegramTerminalDeliveryErrorMessage(message)) {
    return false;
  }
  return true;
}

async function processPendingTelegramVideoDelivery(job) {
  const currentJob = getTelegramPendingVideoDelivery(state, job.jobId, {
    maxHistoryMessages: env.maxHistoryMessages,
  });
  if (!currentJob) {
    return false;
  }

  const resolvedConversation = resolveTelegramConversationForOutbound(state, currentJob.chatId, {
    preferredConversationId: currentJob.conversationId,
    replyToMessageId: currentJob.replyToMessageId,
  });
  state = resolvedConversation.state;
  const priorAnimationState = getTelegramEmotiveAnimationState(state, currentJob.chatId, {
    conversationId: resolvedConversation.conversationId,
  });
  const renderBundle = prependEmotiveAnimationStartMoment(currentJob.bundle, priorAnimationState?.lastMoment);
  const persistedArtifactPath = String(currentJob.artifactPath ?? '').trim();
  const artifactPath = (await hasUsableTelegramVideoArtifact(persistedArtifactPath))
    ? persistedArtifactPath
    : buildTelegramVideoArtifactPath(currentJob.jobId);
  const artifactReady = await hasUsableTelegramVideoArtifact(artifactPath);
  let renderResult = null;
  try {
    if (!artifactReady && env.renderBackend === 'chromium_webgl' && !(await isWebglRendererReady())) {
      const nextAttemptAtMs = Date.now() + Math.min(15000, buildTelegramVideoRetryDelayMs(Math.max(1, currentJob.attemptCount + 1)));
      state = updateTelegramPendingVideoDelivery(state, currentJob.jobId, {
        stage: 'render',
        nextAttemptAtMs,
        updatedAtMs: Date.now(),
        lastError: 'renderer_not_ready',
        artifactPath: '',
        artifactReadyAtMs: 0,
      }, {
        maxHistoryMessages: env.maxHistoryMessages,
      });
      state = recordTelegramOutboxDeliveryReceipt(state, {
        sequenceNumber: currentJob.sequenceNumber,
        chatId: currentJob.chatId,
        replyToMessageId: currentJob.replyToMessageId,
        deliveryMode: currentJob.deliveryMode || 'no_reply',
        telegramMessageId: currentJob.textTelegramMessageId || currentJob.replyToMessageId,
        ...(Array.isArray(currentJob.telegramMessageIds) && currentJob.telegramMessageIds.length > 1
          ? {
            telegramMessageIds: currentJob.telegramMessageIds,
            chunkCount: currentJob.chunkCount ?? currentJob.telegramMessageIds.length,
          }
          : {}),
        deliveredAtMs: Date.now(),
        animation: {
          requested: true,
          status: 'queued',
          stage: 'render',
          keyframeCount: buildEmotiveAnimationRenderPlan(renderBundle)?.keyframeCount ?? 0,
          durationSeconds: buildEmotiveAnimationRenderPlan(renderBundle)?.durationSeconds ?? 0,
          failureReason: 'renderer_not_ready',
        },
      }, {
        maxHistoryMessages: env.maxHistoryMessages,
      });
      await persistState();
      return false;
    }

    if (!artifactReady) {
      await ensureTelegramVideoSpoolDir();
      state = updateTelegramPendingVideoDelivery(state, currentJob.jobId, {
        attemptCount: currentJob.attemptCount + 1,
        stage: 'render',
        nextAttemptAtMs: Date.now(),
        updatedAtMs: Date.now(),
        lastError: '',
        artifactPath,
        artifactReadyAtMs: 0,
      }, {
        maxHistoryMessages: env.maxHistoryMessages,
      });
      await persistState();

      renderResult = await renderTelegramEmotiveAnimation(renderBundle, {
        outputPath: artifactPath,
        requestId: currentJob.jobId,
        timeoutMs: buildWebglRenderTimeoutMs(renderBundle, {
          minTimeoutMs: env.webglRenderTimeoutMs,
        }),
      });
      state = updateTelegramPendingVideoDelivery(state, currentJob.jobId, {
        stage: 'upload',
        nextAttemptAtMs: Date.now(),
        updatedAtMs: Date.now(),
        artifactPath,
        artifactReadyAtMs: Date.now(),
        lastError: '',
      }, {
        maxHistoryMessages: env.maxHistoryMessages,
      });
      await persistState();
    } else {
      state = updateTelegramPendingVideoDelivery(state, currentJob.jobId, {
        attemptCount: currentJob.attemptCount + 1,
        stage: 'upload',
        nextAttemptAtMs: Date.now(),
        updatedAtMs: Date.now(),
        artifactPath,
        artifactReadyAtMs: Math.max(0, Number(currentJob.artifactReadyAtMs ?? Date.now()) || Date.now()),
        lastError: '',
      }, {
        maxHistoryMessages: env.maxHistoryMessages,
      });
      await persistState();
    }

    const videoReplyAnchorId = currentJob.textTelegramMessageId > 0
      ? currentJob.textTelegramMessageId
      : currentJob.replyToMessageId;
    const videoDelivery = await sendTelegramVideoAttachment(
      currentJob.chatId,
      artifactPath,
      currentJob.videoPayload,
      videoReplyAnchorId,
    );
    const videoTelegramMessageId = getTelegramResultMessageId(videoDelivery.sent);
    const terminalMoment = extractEmotiveAnimationTerminalMoment(renderBundle);
    state = setTelegramEmotiveAnimationState(state, currentJob.chatId, {
      lastMoment: terminalMoment,
      lastRenderedAtMs: Date.now(),
    }, {
      maxHistoryMessages: env.maxHistoryMessages,
      conversationId: resolvedConversation.conversationId,
    });
    state = recordTelegramOutboxDeliveryReceipt(state, {
      sequenceNumber: currentJob.sequenceNumber,
      chatId: currentJob.chatId,
      replyToMessageId: currentJob.replyToMessageId,
      deliveryMode: currentJob.deliveryMode || 'no_reply',
      telegramMessageId: currentJob.textTelegramMessageId || videoTelegramMessageId,
      ...(Array.isArray(currentJob.telegramMessageIds) && currentJob.telegramMessageIds.length > 1
        ? {
          telegramMessageIds: currentJob.telegramMessageIds,
          chunkCount: currentJob.chunkCount ?? currentJob.telegramMessageIds.length,
        }
        : {}),
      deliveredAtMs: Date.now(),
      animation: {
        requested: true,
        status: 'sent',
        stage: 'complete',
        keyframeCount: renderResult?.keyframeCount ?? buildEmotiveAnimationRenderPlan(renderBundle)?.keyframeCount ?? 0,
        durationSeconds: renderResult?.durationSeconds ?? buildEmotiveAnimationRenderPlan(renderBundle)?.durationSeconds ?? 0,
        telegramMessageId: videoTelegramMessageId,
      },
    }, {
      maxHistoryMessages: env.maxHistoryMessages,
    });
    state = deleteTelegramPendingVideoDelivery(state, currentJob.jobId, {
      maxHistoryMessages: env.maxHistoryMessages,
    });
    await persistState();
    log('delivered Telegram emotive animation as async follow-up video', {
      chatId: currentJob.chatId,
      conversationId: resolvedConversation.conversationId,
      sequenceNumber: currentJob.sequenceNumber,
      jobId: currentJob.jobId,
      textTelegramMessageId: currentJob.textTelegramMessageId,
      videoTelegramMessageId,
      keyframeCount: renderResult.keyframeCount,
      durationSeconds: renderResult.durationSeconds,
      fallbackError: videoDelivery.fallbackError || undefined,
    });
    await removeTelegramVideoArtifact(artifactPath);
    return true;
  } catch (error) {
    const nextAttemptCount = currentJob.attemptCount + 1;
    const shouldRetry = isRetryableVideoDeliveryError(error)
      && (env.videoDeliveryMaxAttempts === 0 || nextAttemptCount < env.videoDeliveryMaxAttempts);
    const nextAttemptAtMs = shouldRetry
      ? Date.now() + buildTelegramVideoRetryDelayMs(nextAttemptCount)
      : 0;
    const failureStage = String(error?.stage ?? (artifactReady ? 'upload' : 'render')).trim() || 'render';
    const keepArtifact = failureStage === 'upload' && await hasUsableTelegramVideoArtifact(artifactPath);
    state = shouldRetry
      ? updateTelegramPendingVideoDelivery(state, currentJob.jobId, {
        attemptCount: nextAttemptCount,
        stage: failureStage,
        nextAttemptAtMs,
        updatedAtMs: Date.now(),
        lastError: String(error?.message ?? error),
        artifactPath: keepArtifact ? artifactPath : '',
        artifactReadyAtMs: keepArtifact
          ? Math.max(0, Number(currentJob.artifactReadyAtMs ?? Date.now()) || Date.now())
          : 0,
      }, {
        maxHistoryMessages: env.maxHistoryMessages,
      })
      : deleteTelegramPendingVideoDelivery(state, currentJob.jobId, {
        maxHistoryMessages: env.maxHistoryMessages,
      });
    state = recordTelegramOutboxDeliveryReceipt(state, {
      sequenceNumber: currentJob.sequenceNumber,
      chatId: currentJob.chatId,
      replyToMessageId: currentJob.replyToMessageId,
      deliveryMode: currentJob.deliveryMode || 'no_reply',
      telegramMessageId: currentJob.textTelegramMessageId || currentJob.replyToMessageId,
      ...(Array.isArray(currentJob.telegramMessageIds) && currentJob.telegramMessageIds.length > 1
        ? {
          telegramMessageIds: currentJob.telegramMessageIds,
          chunkCount: currentJob.chunkCount ?? currentJob.telegramMessageIds.length,
        }
        : {}),
      deliveredAtMs: Date.now(),
      animation: {
        requested: true,
        status: shouldRetry ? 'queued' : 'failed',
        stage: failureStage,
        keyframeCount: renderResult?.keyframeCount ?? buildEmotiveAnimationRenderPlan(renderBundle)?.keyframeCount ?? 0,
        durationSeconds: renderResult?.durationSeconds ?? buildEmotiveAnimationRenderPlan(renderBundle)?.durationSeconds ?? 0,
        failureReason: String(error?.message ?? error),
      },
    }, {
      maxHistoryMessages: env.maxHistoryMessages,
    });
    await persistState();
    log(
      shouldRetry
        ? 'queued Telegram emotive animation retry'
        : 'failed Telegram emotive animation after bounded retries',
      {
        chatId: currentJob.chatId,
        conversationId: resolvedConversation.conversationId,
        sequenceNumber: currentJob.sequenceNumber,
        jobId: currentJob.jobId,
        attemptCount: nextAttemptCount,
        maxAttempts: env.videoDeliveryMaxAttempts || 'unbounded',
        stage: failureStage,
        nextAttemptAtMs: shouldRetry ? nextAttemptAtMs : undefined,
        error: String(error?.message ?? error),
      },
    );
    if (!shouldRetry || !keepArtifact) {
      await removeTelegramVideoArtifact(artifactPath);
    }
    return !shouldRetry;
  } finally {
    if (renderResult?.tempDir) {
      await rm(renderResult.tempDir, { recursive: true, force: true });
    }
  }
}

async function pollTelegramPendingVideoLoop() {
  for (;;) {
    try {
      const jobs = listTelegramPendingVideoDeliveries(state, {
        maxHistoryMessages: env.maxHistoryMessages,
      });
      const now = Date.now();
      const nextJob = jobs.find((job) => Math.max(0, Number(job?.nextAttemptAtMs ?? 0) || 0) <= now);
      if (!nextJob) {
        await delay(env.videoPollIdleMs);
        continue;
      }
      await processPendingTelegramVideoDelivery(nextJob);
    } catch (error) {
      log(`Telegram pending video polling error: ${error.message}`);
      await delay(env.videoPollIdleMs);
    }
  }
}

async function sendTelegramPromptMessage(chatId, text, promptMarkup, replyToMessageId = 0) {
  const normalizedReplyToMessageId = Math.max(0, Number(replyToMessageId ?? 0) || 0);
  const payload = {
    reply_markup: promptMarkup,
  };
  if (normalizedReplyToMessageId <= 0) {
    return {
      sent: await sendTelegramMessage(chatId, text, payload),
      deliveryMode: 'no_reply',
      requestedReplyToMessageId: 0,
      fallbackError: '',
    };
  }

  try {
    return {
      sent: await sendTelegramMessage(chatId, text, {
        ...payload,
        reply_to_message_id: normalizedReplyToMessageId,
      }),
      deliveryMode: 'reply',
      requestedReplyToMessageId: normalizedReplyToMessageId,
      fallbackError: '',
    };
  } catch (error) {
    if (!isTelegramReplyTargetErrorMessage(error?.message)) {
      throw error;
    }
    const fallbackError = error.message;
    log('Telegram prompt reply target rejected; retrying without reply anchor', {
      chatId: String(chatId),
      replyToMessageId: normalizedReplyToMessageId,
      error: fallbackError,
    });
    return {
      sent: await sendTelegramMessage(chatId, text, payload),
      deliveryMode: 'fallback_no_reply',
      requestedReplyToMessageId: normalizedReplyToMessageId,
      fallbackError,
    };
  }
}

async function clearPromptReplyMarkup(chatId, telegramMessageId) {
  const normalizedMessageId = Math.max(0, Number(telegramMessageId ?? 0) || 0);
  if (!chatId || normalizedMessageId <= 0) {
    return;
  }
  await telegramRequest('editMessageReplyMarkup', {
    chat_id: chatId,
    message_id: normalizedMessageId,
    reply_markup: { inline_keyboard: [] },
  }).catch(() => {});
}

async function interruptDmnApprovalsForChat(chatId, telegramMessageId = 0) {
  const response = await postVicunaJson('/v1/telegram/interruption', {
    chat_scope: String(chatId),
    telegram_message_id: Math.max(0, Number(telegramMessageId ?? 0) || 0),
    interrupt_kind: 'new_user_message',
  });
  const cancelledApprovalIds = Array.isArray(response?.cancelled_approval_ids)
    ? response.cancelled_approval_ids.map((value) => String(value ?? '').trim()).filter(Boolean)
    : [];
  if (cancelledApprovalIds.length === 0) {
    return;
  }

  const nextPromptEntries = [];
  for (const [promptId, prompt] of Object.entries(state.pendingOptionPrompts ?? {})) {
    if (
      prompt?.kind === 'approval_request' &&
      String(prompt?.chatId ?? '') === String(chatId) &&
      cancelledApprovalIds.includes(String(prompt?.approvalId ?? '').trim())
    ) {
      await clearPromptReplyMarkup(chatId, prompt.telegramMessageId);
      continue;
    }
    nextPromptEntries.push([promptId, prompt]);
  }

  state = {
    ...state,
    pendingOptionPrompts: Object.fromEntries(nextPromptEntries),
  };
  await persistState();
  log('cancelled pending runtime approval prompts for new Telegram input', {
    chatId: String(chatId),
    telegramMessageId: Math.max(0, Number(telegramMessageId ?? 0) || 0),
    cancelledApprovalIds,
  });
}

function buildOptionCallbackData(promptId, optionIndex) {
  return `vopt:${promptId}:${optionIndex}`;
}

function parseOptionCallbackData(value) {
  const match = /^vopt:([A-Za-z0-9_-]+):([0-9]+)$/.exec(String(value ?? '').trim());
  if (!match) {
    return null;
  }
  return {
    promptId: match[1],
    optionIndex: Number.parseInt(match[2], 10),
  };
}

async function ensureTelegramLongPolling() {
  await telegramRequest('deleteWebhook', {
    drop_pending_updates: false,
  });
}

async function broadcastToChats(text) {
  const messageText = String(text ?? '').trim();
  if (!messageText || state.chatIds.length === 0) {
    return;
  }
  await Promise.allSettled(
    state.chatIds.map((chatId) =>
      sendTelegramMessage(chatId, messageText),
    ),
  );
}

async function callVicunaForTelegramMessage(chatId, messageId = 0, options = {}) {
  const conversationId = typeof options?.conversationId === 'string' ? options.conversationId.trim() : '';
  const transcript = getChatTranscript(state, chatId, conversationId ? { conversationId } : {});
  const historyTurns = Math.max(1, Math.ceil(transcript.length / 2));
  const deferredDelivery = options?.deferredDelivery === true;
  const messages = transcript.length > 0 ? transcript : [];
  const requestId = typeof options?.requestId === 'string' && options.requestId.trim()
    ? options.requestId.trim()
    : createBridgeRequestId('vicuna');

  if (messages.length === 0) {
    throw new Error('Telegram turn forwarding requires a non-empty transcript');
  }

  const requestStartedAt = Date.now();
  logEvent('vicuna_request_started', {
    requestId,
    chatId: String(chatId),
    conversationId,
    messageId,
    deferredDelivery,
    carriedTranscriptMessageCount: transcript.length,
    maxTokens: env.maxTokens,
  });
  log('forwarding Telegram transcript to Vicuna', {
    chatId: String(chatId),
    conversationId,
    carriedTranscriptMessageCount: transcript.length,
    roles: transcript.map((entry) => entry.role),
    maxTokens: env.maxTokens,
    deferredDelivery,
  });

  const response = await fetch(`${env.vicunaBaseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      ...vicunaHeaders(),
      'X-Client-Request-Id': requestId,
      'X-Vicuna-Telegram-Chat-Id': String(chatId),
      ...(deferredDelivery ? { 'X-Vicuna-Telegram-Deferred-Delivery': '1' } : {}),
      'X-Vicuna-Telegram-Message-Id': String(messageId || 0),
      'X-Vicuna-Telegram-History-Turns': String(historyTurns),
      ...(conversationId ? { 'X-Vicuna-Telegram-Conversation-Id': conversationId } : {}),
    },
    body: JSON.stringify(buildTelegramChatCompletionRequest({
      model: env.model,
      transcript: messages,
      maxTokens: env.maxTokens,
    })),
    signal: AbortSignal.timeout(env.requestTimeoutMs),
  });
  const body = await response.json();
  const runtimeRequestId = response.headers.get('x-client-request-id')
    || response.headers.get('x-request-id')
    || requestId;
  if (!response.ok) {
    logEvent('vicuna_request_failed', {
      requestId: runtimeRequestId,
      chatId: String(chatId),
      conversationId,
      messageId,
      elapsedMs: Date.now() - requestStartedAt,
      status: response.status,
      deferredDelivery,
      error: `vicuna chat failed: ${response.status} ${JSON.stringify(body)}`,
    });
    throw new Error(`vicuna chat failed: ${response.status} ${JSON.stringify(body)}`);
  }
  logEvent('vicuna_request_finished', {
    requestId: runtimeRequestId,
    chatId: String(chatId),
    conversationId,
    messageId,
    elapsedMs: Date.now() - requestStartedAt,
    status: response.status,
    deferredDelivery,
    finishReason: body?.choices?.[0]?.finish_reason ?? '',
    toolCallCount: Array.isArray(body?.choices?.[0]?.message?.tool_calls) ? body.choices[0].message.tool_calls.length : 0,
    queuedTelegramDelivery: body?.vicuna_telegram_delivery?.queued === true,
  });
  if (body?.vicuna_telegram_delivery?.handled === true) {
    return '';
  }

  const reply = extractChatCompletionText(body);
  if (reply) {
    return reply;
  }

  const toolCalls = Array.isArray(body?.choices?.[0]?.message?.tool_calls) ? body.choices[0].message.tool_calls.length : 0;
  throw new Error(
    `Telegram runtime returned no queued delivery and no relayable assistant text (finish_reason=${String(body?.choices?.[0]?.finish_reason ?? '')}, tool_calls=${toolCalls})`,
  );
}

async function fetchVicunaRequestTraceSummary(requestId) {
  const id = String(requestId ?? '').trim();
  if (!id) {
    return null;
  }
  const response = await fetch(`${env.vicunaBaseUrl}/v1/debug/request-traces?request_id=${encodeURIComponent(id)}&limit=200`, {
    method: 'GET',
    headers: vicunaHeaders(),
    signal: AbortSignal.timeout(Math.min(env.requestTimeoutMs, 5000)),
  });
  if (!response.ok) {
    return null;
  }
  const body = await response.json();
  const items = Array.isArray(body?.items) ? body.items : [];
  const latest = items.length > 0 ? items[items.length - 1] : null;
  return latest && typeof latest === 'object'
    ? {
      latestEvent: String(latest.event ?? ''),
      latestComponent: String(latest.component ?? ''),
      count: items.length,
    }
    : null;
}

async function runDeferredTelegramTurn({ chatId, messageId = 0, conversationId = '' }) {
  const requestId = createBridgeRequestId('deferred');
  try {
    const reply = await callVicunaForTelegramMessage(chatId, messageId, {
      deferredDelivery: true,
      conversationId,
      requestId,
    });
    if (reply) {
      const resolved = resolveTelegramConversationForOutbound(state, chatId, {
        preferredConversationId: conversationId,
        replyToMessageId: messageId,
      });
      state = resolved.state;
      const sent = await sendTelegramMessage(
        chatId,
        reply,
        messageId ? { reply_to_message_id: messageId } : {},
      ).catch(() => null);
      if (sent?.message_id) {
        state = appendChatTranscriptMessage(
          state,
          chatId,
          'assistant',
          reply,
          {
            maxHistoryMessages: env.maxHistoryMessages,
            conversationId: resolved.conversationId,
            telegramMessageId: Number(sent.message_id) || 0,
          },
        );
        await persistState();
      }
      log('deferred Telegram turn fell back to direct bridge delivery', {
        chatId: String(chatId),
        messageId,
        conversationId,
      });
      logEvent('deferred_turn_bridge_fallback_delivery', {
        requestId,
        chatId: String(chatId),
        messageId,
        conversationId,
      });
    }
  } catch (error) {
    const classification = error?.name === 'TimeoutError'
      ? 'timeout'
      : (String(error?.message ?? '').includes('fetch failed') ? 'transport' : 'runtime');
    const traceSummary = await fetchVicunaRequestTraceSummary(requestId).catch(() => null);
    const suppressFailureReply = shouldSuppressDeferredTelegramFailureMessage({
      classification,
      latestTraceEvent: traceSummary?.latestEvent ?? '',
    });
    logEvent('deferred_turn_failed', {
      requestId,
      chatId: String(chatId),
      messageId,
      conversationId,
      classification,
      errorName: String(error?.name ?? ''),
      error: error.message,
      traceSummary,
      suppressFailureReply,
    });
    log('deferred Telegram turn failed', {
      chatId: String(chatId),
      messageId,
      conversationId,
      error: error.message,
      traceSummary,
      suppressFailureReply,
    });
    if (suppressFailureReply) {
      return;
    }
    const resolved = resolveTelegramConversationForOutbound(state, chatId, {
      preferredConversationId: conversationId,
      replyToMessageId: messageId,
    });
    state = resolved.state;
    const sent = await sendTelegramMessage(
      chatId,
      `I ran into a problem while working on that request and could not complete it: ${error.message}`,
      messageId ? { reply_to_message_id: messageId } : {},
    ).catch(() => null);
    if (sent?.message_id) {
      state = appendChatTranscriptMessage(
        state,
        chatId,
        'assistant',
        `I ran into a problem while working on that request and could not complete it: ${error.message}`,
        {
          maxHistoryMessages: env.maxHistoryMessages,
          conversationId: resolved.conversationId,
          telegramMessageId: Number(sent.message_id) || 0,
        },
      );
      await persistState();
    }
  }
}

async function persistState() {
  await saveState(env.statePath, state, { maxHistoryMessages: env.maxHistoryMessages });
}

async function handleTelegramMessage(message) {
  if (!message?.chat?.id) {
    return;
  }
  state = registerChat(state, message.chat.id);
  await persistState();
  await interruptDmnApprovalsForChat(message.chat.id, message.message_id);

  const text = typeof message.text === 'string' ? message.text.trim() : '';
  const hasDocument = Boolean(message.document?.file_id);

  if (text === '/start') {
    await sendTelegramMessage(
      message.chat.id,
      'Telegram relay connected. Your messages will be forwarded to the local Vicuña runtime, and proactive system emissions will be relayed here.',
      { reply_to_message_id: message.message_id },
    );
    return;
  }

  if (await handleBridgeControlCommand({
    repoRoot: path.resolve(bridgeDir, '..', '..'),
    text,
    message,
    sendTelegramMessage,
  })) {
    return;
  }

  if (hasDocument) {
    const captionText = typeof message.caption === 'string' ? message.caption.trim() : '';
    const resolvedConversation = resolveTelegramConversationForMessage(state, message, {
      maxHistoryMessages: env.maxHistoryMessages,
    });
    state = resolvedConversation.state;
    const ingestion = await ingestTelegramDocumentMessage({
      message,
      maxDocumentChars: env.maxDocumentChars,
      maxDocumentChunks: 128,
      resolveTelegramFile: (fileId) => telegramRequest('getFile', { file_id: fileId }),
      downloadTelegramFile,
      parseDocument: ({ fileBuffer, descriptor }) => parseTelegramDocumentWithDocling(fileBuffer, descriptor),
      docsRoot: env.docsRoot,
      documentContainerTag: env.documentContainerTag,
    });

    if (!ingestion.ok) {
      await sendTelegramMessage(
        message.chat.id,
        ingestion.userError,
        { reply_to_message_id: message.message_id },
      );
      return;
    }

    state = appendChatTranscriptMessage(
      state,
      message.chat.id,
      'user',
      captionText ? `${captionText}\n\n${ingestion.transcriptText}` : ingestion.transcriptText,
      {
        maxHistoryMessages: env.maxHistoryMessages,
        conversationId: resolvedConversation.conversationId,
        telegramMessageId: message.message_id,
      },
    );
    await persistState();
    log('appended Telegram document user turn', transcriptSummary(message.chat.id, resolvedConversation.conversationId));

    void runDeferredTelegramTurn({
      chatId: message.chat.id,
      messageId: message.message_id,
      conversationId: resolvedConversation.conversationId,
    });
    return;
  }

  if (!text) {
    await sendTelegramMessage(
      message.chat.id,
      'Only plain text or Docling-supported document uploads such as PDF and DOCX are supported right now.',
      { reply_to_message_id: message.message_id },
    );
    return;
  }

  const resolvedConversation = resolveTelegramConversationForMessage(state, message, {
    maxHistoryMessages: env.maxHistoryMessages,
  });
  state = resolvedConversation.state;
  state = appendChatTranscriptMessage(
    state,
    message.chat.id,
    'user',
    text,
    {
      maxHistoryMessages: env.maxHistoryMessages,
      conversationId: resolvedConversation.conversationId,
      telegramMessageId: message.message_id,
    },
  );
  await persistState();
  log('appended Telegram user turn', {
    ...transcriptSummary(message.chat.id, resolvedConversation.conversationId),
    continuitySource: resolvedConversation.reason,
    replyToMessageId: Number(message?.reply_to_message?.message_id ?? 0) || 0,
  });

  void runDeferredTelegramTurn({
    chatId: message.chat.id,
    messageId: message.message_id,
    conversationId: resolvedConversation.conversationId,
  });
}

async function handleTelegramCallbackQuery(callbackQuery) {
  const parsed = parseOptionCallbackData(callbackQuery?.data);
  if (!parsed) {
    await telegramRequest('answerCallbackQuery', {
      callback_query_id: callbackQuery.id,
      text: 'This option selection is no longer available.',
      show_alert: false,
    });
    return;
  }

  const prompt = getPendingOptionPrompt(state, parsed.promptId);
  if (!prompt || parsed.optionIndex < 0 || parsed.optionIndex >= prompt.options.length) {
    await telegramRequest('answerCallbackQuery', {
      callback_query_id: callbackQuery.id,
      text: 'That option prompt has expired.',
      show_alert: false,
    });
    return;
  }

  const selectedOption = prompt.options[parsed.optionIndex];
  const chatId = callbackQuery?.message?.chat?.id ?? prompt.chatId;
  const messageId = callbackQuery?.message?.message_id ?? prompt.telegramMessageId ?? 0;

  if (prompt.kind === 'approval_request') {
    let approvalResponse = null;
    try {
      approvalResponse = await postVicunaJson('/v1/telegram/approval', {
        approval_id: prompt.approvalId,
        chat_scope: String(chatId),
        decision: selectedOption,
        selected_option_index: parsed.optionIndex,
        selected_option_label: selectedOption,
        telegram_message_id: Number(messageId) || 0,
        telegram_callback_query_id: String(callbackQuery?.id ?? ''),
        decision_source: 'callback_query',
      });
    } catch (error) {
      await telegramRequest('answerCallbackQuery', {
        callback_query_id: callbackQuery.id,
        text: `Relay error: ${error.message}`,
        show_alert: false,
      });
      throw error;
    }

    const approvalOk = approvalResponse?.ok === true;
    await telegramRequest('answerCallbackQuery', {
      callback_query_id: callbackQuery.id,
      text: approvalOk
        ? `Selected: ${selectedOption}`
        : String(approvalResponse?.error ?? 'That approval prompt has expired.'),
      show_alert: false,
    });

    if (
      approvalOk ||
      ['expired', 'superseded', 'denied', 'cancelled', 'dispatch_failed'].includes(String(approvalResponse?.status ?? ''))
    ) {
      await clearPromptReplyMarkup(chatId, messageId);
      state = deletePendingOptionPrompt(state, parsed.promptId, { maxHistoryMessages: env.maxHistoryMessages });
      await persistState();
    }
    return;
  }

  await telegramRequest('answerCallbackQuery', {
    callback_query_id: callbackQuery.id,
    text: `Selected: ${selectedOption}`,
    show_alert: false,
  });

  await clearPromptReplyMarkup(chatId, messageId);

  state = deletePendingOptionPrompt(state, parsed.promptId, { maxHistoryMessages: env.maxHistoryMessages });
  state = appendChatTranscriptMessage(
    state,
    chatId,
    'user',
    `Selected option for "${prompt.question}": ${selectedOption}`,
    {
      maxHistoryMessages: env.maxHistoryMessages,
      conversationId: prompt.conversationId,
    },
  );
  await persistState();
  log('appended Telegram option-selection user turn', transcriptSummary(chatId, prompt.conversationId));

  const reply = await callVicunaForTelegramMessage(chatId, messageId, {
    conversationId: prompt.conversationId,
  });

  if (reply) {
    const sent = await sendTelegramMessage(
      chatId,
      reply,
      messageId ? { reply_to_message_id: messageId } : {},
    );
    state = appendChatTranscriptMessage(
      state,
      chatId,
      'assistant',
      reply,
      {
        maxHistoryMessages: env.maxHistoryMessages,
        conversationId: prompt.conversationId,
        telegramMessageId: Number(sent?.message_id ?? 0) || 0,
      },
    );
    await persistState();
    log('appended Telegram assistant turn after option selection', transcriptSummary(chatId, prompt.conversationId));
  } else {
    log('Telegram option-selection turn produced no direct assistant text; assuming tool-delivered or deferred output', {
      chatId: String(chatId),
      messageId,
      promptId: parsed.promptId,
      conversationId: prompt.conversationId,
    });
  }
}

async function pollTelegramAskOutboxLoop() {
  for (;;) {
    try {
      const response = await fetch(`${env.vicunaBaseUrl}/v1/telegram/outbox?after=${state.telegramOutboxOffset}`, {
        headers: vicunaHeaders(),
      });
      if (!response.ok) {
        throw new Error(`vicuna telegram outbox failed: ${response.status}`);
      }
      const body = await response.json();
      if (!env.replayRetainedOutbox && shouldBootstrapTelegramOutboxOffset(state, body)) {
        const bootstrapOffset = Math.max(0, Number(body?.newest_sequence ?? 0) || 0);
        state = bootstrapTelegramOutboxOffset(state, body, { maxHistoryMessages: env.maxHistoryMessages });
        await persistState();
        log('initialized Telegram outbox checkpoint without backlog replay', {
          nextOffset: bootstrapOffset,
          storedItems: Number(body?.stored_items ?? 0) || 0,
          oldestSequence: Number(body?.oldest_sequence ?? 0) || 0,
          newestSequence: Number(body?.newest_sequence ?? 0) || 0,
        });
        await delay(TELEGRAM_BRIDGE_ASK_OUTBOX_POLL_IDLE_MS);
        continue;
      }
      if (env.replayRetainedOutbox && !state.telegramOutboxCheckpointInitialized) {
        state = setTelegramOutboxCheckpoint(state, state.telegramOutboxOffset, { maxHistoryMessages: env.maxHistoryMessages });
        await persistState();
        log('initialized Telegram outbox checkpoint for explicit backlog replay', {
          currentOffset: state.telegramOutboxOffset,
          storedItems: Number(body?.stored_items ?? 0) || 0,
        });
      }
      const previousOutboxOffset = state.telegramOutboxOffset;
      const reconciledOffset = reconcileTelegramOutboxOffset(state.telegramOutboxOffset, body);
      if (reconciledOffset !== state.telegramOutboxOffset) {
        state = setTelegramOutboxCheckpoint(state, reconciledOffset, { maxHistoryMessages: env.maxHistoryMessages });
        await persistState();
        log('reset Telegram outbox offset after runtime sequence change', {
          previousOffset: Number(previousOutboxOffset ?? 0),
          nextOffset: reconciledOffset,
          storedItems: Number(body?.stored_items ?? 0) || 0,
          oldestSequence: Number(body?.oldest_sequence ?? 0) || 0,
          newestSequence: Number(body?.newest_sequence ?? 0) || 0,
        });
        continue;
      }
      const items = Array.isArray(body?.items) ? body.items : [];
      for (const item of items) {
        const normalizedItem = normalizeTelegramOutboxItem(item);
        if (!normalizedItem.ok) {
          log('skipping malformed runtime telegram outbox item', {
            error: normalizedItem.error,
            kind: normalizedItem.kind,
            sequenceNumber: normalizedItem.sequenceNumber,
          });
          if (normalizedItem.skippable) {
            state = setTelegramOutboxCheckpoint(
              state,
              Math.max(state.telegramOutboxOffset, normalizedItem.sequenceNumber),
              { maxHistoryMessages: env.maxHistoryMessages },
            );
            await persistState();
            continue;
          }
          throw new Error(normalizedItem.error);
        }

        const sequenceNumber = normalizedItem.sequenceNumber;
        try {
          if (normalizedItem.kind === 'message') {
            const chatId = normalizedItem.chatId;
            const text = normalizedItem.text;
            const replyToMessageId = normalizedItem.replyToMessageId;
            const telegramMethod = normalizedItem.telegramMethod;
            const telegramPayload = normalizedItem.telegramPayload;
            const emotiveAnimation = normalizedItem.emotiveAnimation;
            const resolvedConversation = resolveTelegramConversationForOutbound(state, chatId, {
              replyToMessageId,
            });
            state = resolvedConversation.state;
            let delivery;
            let sent;
            let telegramMessageId = 0;
            let deliveredTelegramMethod = telegramMethod;
            let animationReceipt = null;
            let textDelivery = null;
            const priorAnimationState = getTelegramEmotiveAnimationState(state, chatId, {
              conversationId: resolvedConversation.conversationId,
            });
            const renderBundle = emotiveAnimation
              ? prependEmotiveAnimationStartMoment(emotiveAnimation, priorAnimationState?.lastMoment)
              : null;
            const normalizedMessagePayload = telegramMethod === 'sendMessage'
              ? normalizeTelegramRichTextPayload({
                telegramMethod,
                telegramPayload,
                text,
              }).payload
              : telegramPayload;

            if (emotiveAnimation && telegramMethod === 'sendMessage') {
              const animationDeliveryPlan = buildTelegramAnimationDeliveryPlan({
                telegramMethod,
                telegramPayload: normalizedMessagePayload,
                text,
              });
              if (!animationDeliveryPlan.ok) {
                animationReceipt = {
                  requested: true,
                  status: 'failed',
                  stage: 'caption',
                  keyframeCount: buildEmotiveAnimationRenderPlan(renderBundle)?.keyframeCount ?? 0,
                  durationSeconds: buildEmotiveAnimationRenderPlan(renderBundle)?.durationSeconds ?? 0,
                  failureReason:
                    `emotive animation primary sendVideo payload was unsupported: ${animationDeliveryPlan.reason}`,
                };
                log('Telegram emotive animation spoiler-video fallback preserved text delivery', {
                  chatId,
                  conversationId: resolvedConversation.conversationId,
                  sequenceNumber,
                  stage: animationReceipt.stage,
                  error: animationReceipt.failureReason,
                  keyframeCount: animationReceipt.keyframeCount,
                  durationSeconds: animationReceipt.durationSeconds,
                });
                delivery = await sendTelegramOutboxMessage(
                  chatId,
                  telegramMethod,
                  animationDeliveryPlan.messagePayload ?? normalizedMessagePayload,
                  replyToMessageId,
                  text,
                );
                sent = delivery.sent;
                telegramMessageId = delivery.telegramMessageId || getPrimaryTelegramMessageId(delivery.telegramMessageIds, sent);
              } else {
                textDelivery = await sendTelegramOutboxMessage(
                  chatId,
                  telegramMethod,
                  animationDeliveryPlan.messagePayload,
                  replyToMessageId,
                  text,
                );
                delivery = textDelivery;
                sent = textDelivery.sent;
                telegramMessageId = textDelivery.telegramMessageId || getPrimaryTelegramMessageId(textDelivery.telegramMessageIds, sent);
                deliveredTelegramMethod = 'sendMessage';
                const queuedAnimationReceipt = buildQueuedAnimationReceipt(renderBundle);
                animationReceipt = queuedAnimationReceipt;
                state = enqueueTelegramPendingVideoDelivery(state, {
                  jobId: createTelegramPendingVideoJobId(sequenceNumber),
                  sequenceNumber,
                  chatId,
                  conversationId: resolvedConversation.conversationId,
                  replyToMessageId,
                  textTelegramMessageId: telegramMessageId,
                  deliveryMode: textDelivery.deliveryMode,
                  telegramMessageIds: textDelivery.telegramMessageIds,
                  chunkCount: textDelivery.chunkCount,
                  videoPayload: animationDeliveryPlan.videoPayload,
                  bundle: emotiveAnimation,
                  attemptCount: 0,
                  stage: 'queued',
                  artifactPath: '',
                  artifactReadyAtMs: 0,
                  nextAttemptAtMs: Date.now(),
                  createdAtMs: Date.now(),
                  updatedAtMs: Date.now(),
                }, {
                  maxHistoryMessages: env.maxHistoryMessages,
                });
                log('queued Telegram emotive animation as async follow-up video', {
                  chatId,
                  conversationId: resolvedConversation.conversationId,
                  sequenceNumber,
                  textDeliveryMode: textDelivery.deliveryMode,
                  telegramMessageId,
                  keyframeCount: queuedAnimationReceipt.keyframeCount,
                  durationSeconds: queuedAnimationReceipt.durationSeconds,
                });
              }
            } else {
              delivery = await sendTelegramOutboxMessage(
                chatId,
                telegramMethod,
                normalizedMessagePayload,
                replyToMessageId,
                text,
              );
              sent = delivery.sent;
              telegramMessageId = delivery.telegramMessageId || getPrimaryTelegramMessageId(delivery.telegramMessageIds, sent);
            }

            state = appendChatTranscriptMessage(
              state,
              chatId,
              'assistant',
              text,
              {
                maxHistoryMessages: env.maxHistoryMessages,
                conversationId: resolvedConversation.conversationId,
                telegramMessageId,
              },
            );
            state = setTelegramOutboxCheckpoint(
              state,
              Math.max(state.telegramOutboxOffset, sequenceNumber),
              { maxHistoryMessages: env.maxHistoryMessages },
            );
            state = recordTelegramOutboxDeliveryReceipt(state, {
              sequenceNumber,
              chatId,
              replyToMessageId: delivery.requestedReplyToMessageId,
              deliveryMode: delivery.deliveryMode === 'no_reply' ? 'fallback_no_reply' : delivery.deliveryMode,
              telegramMessageId,
              ...(Array.isArray(delivery.telegramMessageIds) && delivery.telegramMessageIds.length > 1
                ? {
                  telegramMessageIds: delivery.telegramMessageIds,
                  chunkCount: delivery.chunkCount ?? delivery.telegramMessageIds.length,
                }
                : {}),
              deliveredAtMs: Date.now(),
              ...(animationReceipt ? { animation: animationReceipt } : {}),
            }, { maxHistoryMessages: env.maxHistoryMessages });
            if (animationReceipt?.status === 'sent' && renderBundle) {
              const terminalMoment = extractEmotiveAnimationTerminalMoment(renderBundle);
              state = setTelegramEmotiveAnimationState(state, chatId, {
                lastMoment: terminalMoment,
                lastRenderedAtMs: Date.now(),
              }, {
                maxHistoryMessages: env.maxHistoryMessages,
                conversationId: resolvedConversation.conversationId,
              });
            }
            await persistState();
            log('delivered Telegram follow-up message', {
              chatId,
              conversationId: resolvedConversation.conversationId,
              sequenceNumber,
              telegramMethod: deliveredTelegramMethod,
              replyToMessageId,
              deliveryMode: delivery.deliveryMode,
              telegramMessageId,
              textChunkCount: delivery.chunkCount ?? 1,
              animationStatus: animationReceipt?.status,
              animationStage: animationReceipt?.stage,
              fallbackError: delivery.fallbackError || undefined,
            });
            continue;
          }

          const chatId = normalizedItem.chatId;
          const question = normalizedItem.question;
          const options = normalizedItem.options;
          const promptId = `o${sequenceNumber.toString(36)}`;
          const resolvedConversation = resolveTelegramConversationForOutbound(state, chatId, {});
          state = resolvedConversation.state;
          const replyMarkup = {
            inline_keyboard: options.map((label, index) => ([{
              text: label,
              callback_data: buildOptionCallbackData(promptId, index),
            }])),
          };
          const promptDelivery = normalizedItem.kind === 'approval_request'
            ? await sendTelegramPromptMessage(chatId, question, replyMarkup, normalizedItem.replyToMessageId)
            : { sent: await sendTelegramMessage(chatId, question, { reply_markup: replyMarkup }) };
          const result = promptDelivery.sent;

          state = appendChatTranscriptMessage(
            state,
            chatId,
            'assistant',
            question,
            {
              maxHistoryMessages: env.maxHistoryMessages,
              conversationId: resolvedConversation.conversationId,
              telegramMessageId: Number(result?.message_id ?? 0) || 0,
            },
          );
          state = setPendingOptionPrompt(state, promptId, {
            kind: normalizedItem.kind,
            approvalId: normalizedItem.kind === 'approval_request' ? normalizedItem.approvalId : '',
            chatId,
            question,
            options,
            conversationId: resolvedConversation.conversationId,
            telegramMessageId: Number(result?.message_id ?? 0) || 0,
            createdAtMs: Date.now(),
          }, { maxHistoryMessages: env.maxHistoryMessages });
          state = setTelegramOutboxCheckpoint(
            state,
            Math.max(state.telegramOutboxOffset, sequenceNumber),
            { maxHistoryMessages: env.maxHistoryMessages },
          );
          await persistState();
          log(
            normalizedItem.kind === 'approval_request'
              ? 'delivered Telegram approval prompt'
              : 'delivered Telegram ask-with-options prompt',
            {
              chatId,
              conversationId: resolvedConversation.conversationId,
              promptId,
              approvalId: normalizedItem.kind === 'approval_request' ? normalizedItem.approvalId : undefined,
              optionCount: options.length,
              replyToMessageId: normalizedItem.replyToMessageId ?? 0,
              deliveryMode: promptDelivery.deliveryMode ?? 'no_reply',
              telegramMessageId: Number(result?.message_id ?? 0) || 0,
              fallbackError: promptDelivery.fallbackError || undefined,
            },
          );
        } catch (error) {
          if (normalizedItem.skippable && isTelegramTerminalDeliveryErrorMessage(error?.message)) {
            state = setTelegramOutboxCheckpoint(
              state,
              Math.max(state.telegramOutboxOffset, sequenceNumber),
              { maxHistoryMessages: env.maxHistoryMessages },
            );
            await persistState();
            log('skipped terminally undeliverable Telegram outbox item', {
              sequenceNumber,
              kind: normalizedItem.kind,
              chatId: normalizedItem.chatId,
              replyToMessageId: normalizedItem.replyToMessageId ?? 0,
              approvalId: normalizedItem.kind === 'approval_request' ? normalizedItem.approvalId : undefined,
              error: error.message,
            });
            continue;
          }
          throw error;
        }
      }
    } catch (error) {
      log(`Telegram ask-with-options outbox polling error: ${error.message}`);
    }
    await delay(TELEGRAM_BRIDGE_ASK_OUTBOX_POLL_IDLE_MS);
  }
}

async function pollTelegramLoop() {
  for (;;) {
    try {
      const updates = await telegramRequest('getUpdates', {
        offset: state.telegramOffset,
        timeout: env.pollTimeoutSeconds,
        allowed_updates: ['message', 'callback_query'],
      });

      for (const update of updates) {
        state = updateTelegramOffset(state, update.update_id);
        await persistState();

        if (update.message) {
          try {
            await handleTelegramMessage(update.message);
          } catch (error) {
            log(`failed to handle Telegram message ${update.update_id}: ${error.message}`);
            if (update.message?.chat?.id) {
              await sendTelegramMessage(
                update.message.chat.id,
                `Relay error: ${error.message}`,
                update.message?.message_id ? { reply_to_message_id: update.message.message_id } : {},
              ).catch(() => {});
            }
          }
        }
        if (update.callback_query) {
          try {
            await handleTelegramCallbackQuery(update.callback_query);
          } catch (error) {
            log(`failed to handle Telegram callback ${update.update_id}: ${error.message}`);
            await telegramRequest('answerCallbackQuery', {
              callback_query_id: update.callback_query.id,
              text: `Relay error: ${error.message}`,
              show_alert: false,
            }).catch(() => {});
          }
        }
      }
    } catch (error) {
      log(`Telegram polling error: ${error.message}`);
      await delay(2000);
    }
  }
}

async function handleSelfEmitEvent(event) {
  selfEmitLastEventAt = Date.now();
  if (event.event !== 'response.completed') {
    return;
  }
  const response = event.data?.response;
  const responseId = response?.id;
  if (!responseId || state.proactiveResponseIds.includes(responseId)) {
    return;
  }
  const text = extractResponseText(response);
  if (!text) {
    state = appendProactiveId(state, responseId);
    await persistState();
    return;
  }
  const relayedText = formatTelegramMessage('System self-emission', text);
  await broadcastToChats(relayedText);
  for (const chatId of state.chatIds) {
    state = appendChatTranscriptMessage(
      state,
      chatId,
      'assistant',
      relayedText,
      { maxHistoryMessages: env.maxHistoryMessages },
    );
  }
  state = appendProactiveId(state, responseId);
  await persistState();
}

function openSelfEmitStream(streamGeneration) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let buffer = '';
    const request = sseHttpModule.request({
      protocol: vicunaUrl.protocol,
      hostname: vicunaUrl.hostname,
      port: vicunaUrl.port,
      path: `${vicunaUrl.pathname.replace(/\/$/, '')}/v1/responses/stream?after=${env.selfEmitAfter}`,
      method: 'GET',
      headers: {
        Accept: 'text/event-stream',
        ...(env.vicunaApiKey ? { Authorization: `Bearer ${env.vicunaApiKey}` } : {}),
      },
    }, (response) => {
      if (response.statusCode !== 200) {
        const statusCode = response.statusCode ?? 0;
        response.resume();
        settled = true;
        reject(new Error(`self-emit stream failed: ${statusCode}`));
        return;
      }

      selfEmitLastConnectedAt = Date.now();
      selfEmitLastEventAt = Date.now();

      response.setEncoding('utf8');
      response.on('data', (chunk) => {
        buffer += chunk;
        const { complete, remainder } = splitSseBuffer(buffer);
        buffer = remainder;
        if (!complete) {
          return;
        }
        const events = parseSseChunk(complete);
        Promise.all(events.map((event) => handleSelfEmitEvent(event))).catch((error) => {
          if (!settled) {
            settled = true;
            request.destroy(error);
            reject(error);
          }
        });
      });

      response.on('end', () => {
        if (!settled) {
          settled = true;
          resolve({ generation: streamGeneration, reason: 'end' });
        }
      });

      response.on('close', () => {
        if (!settled) {
          settled = true;
          resolve({ generation: streamGeneration, reason: 'close' });
        }
      });

      response.on('error', (error) => {
        if (!settled) {
          settled = true;
          reject(error);
        }
      });
    });

    request.setNoDelay(true);
    request.setTimeout(0);
    request.on('error', (error) => {
      if (!settled) {
        settled = true;
        reject(error);
      }
    });
    request.end();
  });
}

async function runSelfEmitStreamLoop() {
  for (;;) {
    if (selfEmitStreamActive) {
      await delay(TELEGRAM_BRIDGE_SELF_EMIT_ACTIVE_DELAY_MS);
      continue;
    }
    selfEmitStreamActive = true;
    const streamGeneration = ++selfEmitGeneration;
    try {
      log(`self-emit stream connect generation=${streamGeneration}`);
      const result = await openSelfEmitStream(streamGeneration);
      log(`self-emit stream closed generation=${result.generation} reason=${result.reason}`);
      await delay(250);
    } catch (error) {
      log(`self-emit stream error generation=${streamGeneration}: ${error.message}`);
      await delay(TELEGRAM_BRIDGE_SELF_EMIT_ERROR_DELAY_MS);
    } finally {
      selfEmitStreamActive = false;
    }
  }
}

async function watchdogLoop() {
  for (;;) {
    try {
      const response = await fetch(`${env.vicunaBaseUrl}/health`);
      const health = await response.json();
      const liveStreamConnected = Boolean(health?.proactive_mailbox?.live_stream_connected);
      const now = Date.now();
      const staleForMs = now - Math.max(selfEmitLastConnectedAt, selfEmitLastEventAt);
      if (!liveStreamConnected && !selfEmitStreamActive) {
        log('watchdog detected detached self-emit stream, allowing immediate reconnect');
      }
      if (liveStreamConnected && staleForMs > 120000) {
        log(`watchdog observed idle self-emit stream for ${staleForMs} ms`);
      }
    } catch (error) {
      log(`watchdog health check failed: ${error.message}`);
    }
    await delay(TELEGRAM_BRIDGE_WATCHDOG_DELAY_MS);
  }
}

async function main() {
  await ensureTelegramLongPolling();
  await persistState();
  log(`starting bridge with state ${env.statePath}`);
  log(`vicuna base url ${env.vicunaBaseUrl}`);
  log(`emotive render backend ${env.renderBackend}`, env.renderBackend === 'chromium_webgl'
    ? { webglRendererUrl: env.webglRendererUrl }
    : undefined);
  log(`bridge transcript settings history=${env.maxHistoryMessages} max_tokens=${env.maxTokens < 0 ? 'unlimited' : env.maxTokens}`, {
    chatCount: state.chatIds.length,
    sessionCount: Object.keys(state.chatSessions ?? {}).length,
    requestTimeoutMs: env.requestTimeoutMs,
    replayRetainedOutbox: env.replayRetainedOutbox,
    pendingVideoDeliveries: listTelegramPendingVideoDeliveries(state, {
      maxHistoryMessages: env.maxHistoryMessages,
    }).length,
  });
  await Promise.all([
    pollTelegramLoop(),
    pollTelegramAskOutboxLoop(),
    pollTelegramPendingVideoLoop(),
    runSelfEmitStreamLoop(),
    watchdogLoop(),
  ]);
}

await main();
