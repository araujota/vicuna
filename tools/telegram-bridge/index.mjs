import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import Supermemory, { toFile } from 'supermemory';
import {
  appendProactiveId,
  appendChatTranscriptMessage,
  buildTelegramChatCompletionRequest,
  buildTelegramFileUrl,
  bootstrapTelegramOutboxOffset,
  deletePendingOptionPrompt,
  extractChatCompletionText,
  extractResponseText,
  formatTelegramMessage,
  getPendingOptionPrompt,
  getChatTranscript,
  ingestTelegramDocumentMessage,
  isTelegramReplyTargetErrorMessage,
  isTelegramTerminalDeliveryErrorMessage,
  loadState,
  normalizeTelegramOutboxItem,
  parseInteger,
  parseSseChunk,
  recordTelegramOutboxDeliveryReceipt,
  reconcileTelegramOutboxOffset,
  registerChat,
  resolveTelegramConversationForMessage,
  resolveTelegramConversationForOutbound,
  saveState,
  setTelegramOutboxCheckpoint,
  setPendingOptionPrompt,
  shouldBootstrapTelegramOutboxOffset,
  splitSseBuffer,
  updateTelegramOffset,
} from './lib.mjs';

const execFileAsync = promisify(execFile);
const bridgeDir = path.dirname(fileURLToPath(import.meta.url));

const env = {
  telegramBotToken: process.env.TELEGRAM_BOT_TOKEN ?? '',
  vicunaBaseUrl: (process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL ?? 'http://127.0.0.1:8080').replace(/\/+$/, ''),
  model: process.env.TELEGRAM_BRIDGE_MODEL ?? process.env.VICUNA_DEEPSEEK_MODEL ?? 'deepseek-reasoner',
  statePath: process.env.TELEGRAM_BRIDGE_STATE_PATH ?? '/tmp/vicuna-telegram-bridge-state.json',
  pollTimeoutSeconds: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS, 30)),
  maxHistoryMessages: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES, 12)),
  maxTokens: (() => {
    const configured = parseInteger(process.env.TELEGRAM_BRIDGE_MAX_TOKENS, -1);
    return configured < 0 ? -1 : Math.max(32, configured);
  })(),
  maxDocumentChars: Math.max(256, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_DOCUMENT_CHARS, 12000)),
  selfEmitAfter: Math.max(0, parseInteger(process.env.TELEGRAM_BRIDGE_SELF_EMIT_AFTER, 0)),
  replayRetainedOutbox: String(process.env.TELEGRAM_BRIDGE_REPLAY_RETAINED_OUTBOX ?? '').trim() === '1',
  vicunaApiKey: process.env.VICUNA_API_KEY ?? '',
  supermemoryApiKey: process.env.SUPERMEMORY_API_KEY ?? '',
  supermemoryBaseUrl: (process.env.SUPERMEMORY_BASE_URL ?? 'https://api.supermemory.ai').replace(/\/+$/, ''),
  hardMemoryRuntimeIdentity: (process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY ?? 'vicuna').trim() || 'vicuna',
  documentContainerTag: (
    process.env.TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG ??
    `${(process.env.VICUNA_HARD_MEMORY_RUNTIME_IDENTITY ?? 'vicuna').trim() || 'vicuna'}-telegram-documents`
  ).trim() || 'vicuna-telegram-documents',
  doclingPythonBin: process.env.TELEGRAM_BRIDGE_DOCLING_PYTHON_BIN ?? 'python3',
  doclingParserScriptPath:
    process.env.TELEGRAM_BRIDGE_DOCLING_PARSER_SCRIPT_PATH ?? path.join(bridgeDir, 'docling-parse.py'),
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
let supermemoryClient = null;

function log(message, extra = undefined) {
  const prefix = '[telegram-bridge]';
  if (extra === undefined) {
    console.log(`${prefix} ${message}`);
    return;
  }
  console.log(`${prefix} ${message}`, extra);
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

async function postVicunaJson(pathname, payload) {
  const response = await fetch(`${env.vicunaBaseUrl}${pathname}`, {
    method: 'POST',
    headers: vicunaHeaders(),
    body: JSON.stringify(payload ?? {}),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`vicuna ${pathname} failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body;
}

async function telegramRequest(method, payload) {
  const response = await fetch(`${telegramBaseUrl}/${method}`, {
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

function getSupermemoryClient() {
  if (!env.supermemoryApiKey) {
    return null;
  }
  if (!supermemoryClient) {
    supermemoryClient = new Supermemory({
      apiKey: env.supermemoryApiKey,
      baseURL: env.supermemoryBaseUrl,
    });
  }
  return supermemoryClient;
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

async function writeSupermemoryMemories(payload) {
  if (!env.supermemoryApiKey) {
    throw new Error('SUPERMEMORY_API_KEY is required for parsed chunk persistence.');
  }
  const response = await fetch(`${env.supermemoryBaseUrl}/v4/memories`, {
    method: 'POST',
    headers: {
      accept: 'application/json',
      'content-type': 'application/json',
      Authorization: `Bearer ${env.supermemoryApiKey}`,
      'x-supermemory-api-key': env.supermemoryApiKey,
    },
    body: JSON.stringify(payload),
  });
  const body = await response.text();
  if (!response.ok) {
    throw new Error(`Supermemory chunk write failed with HTTP ${response.status}: ${body.slice(0, 400)}`);
  }
  if (!body.trim()) {
    return {};
  }
  try {
    return JSON.parse(body);
  } catch {
    return {};
  }
}

async function sendTelegramMessage(chatId, text, extra = {}) {
  const messageText = String(text ?? '').trim();
  if (!messageText) {
    return;
  }
  return await telegramRequest('sendMessage', {
    chat_id: chatId,
    text: messageText,
    ...extra,
  });
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

async function sendTelegramOutboxMessage(chatId, method, payload, replyToMessageId = 0) {
  const deliveryPayload = cloneTelegramPayload(payload);
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
      deliveryMode: 'fallback_no_reply',
      requestedReplyToMessageId,
      fallbackError,
    };
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

  if (messages.length === 0) {
    throw new Error('Telegram turn forwarding requires a non-empty transcript');
  }

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
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`vicuna chat failed: ${response.status} ${JSON.stringify(body)}`);
  }
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

async function runDeferredTelegramTurn({ chatId, messageId = 0, conversationId = '' }) {
  try {
    const reply = await callVicunaForTelegramMessage(chatId, messageId, {
      deferredDelivery: true,
      conversationId,
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
    }
  } catch (error) {
    log('deferred Telegram turn failed', {
      chatId: String(chatId),
      messageId,
      conversationId,
      error: error.message,
    });
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
      supermemoryClient: getSupermemoryClient(),
      toFileFactory: (buffer, fileName) => toFile(buffer, fileName),
      writeChunkMemories: writeSupermemoryMemories,
      runtimeIdentity: env.hardMemoryRuntimeIdentity,
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
        await delay(1000);
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
            const resolvedConversation = resolveTelegramConversationForOutbound(state, chatId, {
              replyToMessageId,
            });
            state = resolvedConversation.state;
            const delivery = await sendTelegramOutboxMessage(chatId, telegramMethod, telegramPayload, replyToMessageId);
            const sent = delivery.sent;
            const telegramMessageId = getTelegramResultMessageId(sent);
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
              deliveredAtMs: Date.now(),
            }, { maxHistoryMessages: env.maxHistoryMessages });
            await persistState();
            log('delivered Telegram follow-up message', {
              chatId,
              conversationId: resolvedConversation.conversationId,
              sequenceNumber,
              telegramMethod,
              replyToMessageId,
              deliveryMode: delivery.deliveryMode,
              telegramMessageId,
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
    await delay(1000);
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
      await delay(1000);
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
      await delay(1000);
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
    await delay(5000);
  }
}

async function main() {
  await ensureTelegramLongPolling();
  await persistState();
  log(`starting bridge with state ${env.statePath}`);
  log(`vicuna base url ${env.vicunaBaseUrl}`);
  log(`bridge transcript settings history=${env.maxHistoryMessages} max_tokens=${env.maxTokens < 0 ? 'unlimited' : env.maxTokens}`, {
    chatCount: state.chatIds.length,
    sessionCount: Object.keys(state.chatSessions ?? {}).length,
    replayRetainedOutbox: env.replayRetainedOutbox,
  });
  await Promise.all([
    pollTelegramLoop(),
    pollTelegramAskOutboxLoop(),
    runSelfEmitStreamLoop(),
    watchdogLoop(),
  ]);
}

await main();
