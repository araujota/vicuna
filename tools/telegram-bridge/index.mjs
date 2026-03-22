import http from 'node:http';
import https from 'node:https';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import Supermemory, { toFile } from 'supermemory';
import {
  appendProactiveId,
  appendChatTranscriptMessage,
  buildTelegramFileUrl,
  deletePendingOptionPrompt,
  extractChatCompletionText,
  extractResponseText,
  formatTelegramMessage,
  getPendingOptionPrompt,
  getChatTranscript,
  ingestTelegramDocumentMessage,
  loadState,
  parseInteger,
  parseSseChunk,
  registerChat,
  saveState,
  setPendingOptionPrompt,
  splitSseBuffer,
  summarizeChatCompletion,
  updateTelegramOffset,
} from './lib.mjs';

const execFileAsync = promisify(execFile);

const env = {
  telegramBotToken: process.env.TELEGRAM_BOT_TOKEN ?? '',
  vicunaBaseUrl: (process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL ?? 'http://127.0.0.1:8080').replace(/\/+$/, ''),
  model: process.env.TELEGRAM_BRIDGE_MODEL ?? process.env.VICUNA_RUNTIME_MODEL_ALIAS ?? 'vicuna-runtime',
  statePath: process.env.TELEGRAM_BRIDGE_STATE_PATH ?? '/tmp/vicuna-telegram-bridge-state.json',
  pollTimeoutSeconds: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS, 30)),
  maxHistoryMessages: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES, 12)),
  maxTokens: (() => {
    const configured = parseInteger(process.env.TELEGRAM_BRIDGE_MAX_TOKENS, -1);
    return configured < 0 ? -1 : Math.max(32, configured);
  })(),
  maxDocumentChars: Math.max(256, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_DOCUMENT_CHARS, 12000)),
  selfEmitAfter: Math.max(0, parseInteger(process.env.TELEGRAM_BRIDGE_SELF_EMIT_AFTER, 0)),
  vicunaApiKey: process.env.VICUNA_API_KEY ?? '',
  supermemoryApiKey: process.env.SUPERMEMORY_API_KEY ?? '',
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
let pdfParseModulePromise = null;

function log(message, extra = undefined) {
  const prefix = '[telegram-bridge]';
  if (extra === undefined) {
    console.log(`${prefix} ${message}`);
    return;
  }
  console.log(`${prefix} ${message}`, extra);
}

function transcriptSummary(chatId) {
  const transcript = getChatTranscript(state, chatId);
  return {
    chatId: String(chatId),
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

async function extractPdfText(fileBuffer) {
  if (!pdfParseModulePromise) {
    pdfParseModulePromise = import('pdf-parse');
  }
  const { PDFParse } = await pdfParseModulePromise;
  const parser = new PDFParse({ data: fileBuffer });
  try {
    const result = await parser.getText();
    return typeof result?.text === 'string' ? result.text : '';
  } finally {
    await parser.destroy();
  }
}

async function extractWordText(fileBuffer, descriptor) {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'vicuna-telegram-doc-'));
  const inputPath = path.join(tempDir, descriptor.fileName);
  try {
    await writeFile(inputPath, fileBuffer);
    const { stdout } = await execFileAsync('/usr/bin/textutil', [
      '-convert',
      'txt',
      '-stdout',
      '-strip',
      inputPath,
    ], {
      maxBuffer: 16 * 1024 * 1024,
    });
    return stdout;
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      throw new Error('DOC/DOCX extraction requires /usr/bin/textutil on the bridge host.');
    }
    throw error;
  } finally {
    await rm(tempDir, { recursive: true, force: true });
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

async function callVicunaForTelegramMessage(chatId, messageId = 0, extraSystemMessages = []) {
  const transcript = getChatTranscript(state, chatId);
  const historyTurns = Math.max(1, Math.ceil(transcript.length / 2));
  const baseSystemPrompt = 'You are replying to a Telegram user through middleware. Keep responses clear and concise unless the user asks for depth. Maintain continuity across the provided transcript. When the user explicitly asks to search the web or needs fresh current information, prefer the web search tool if the runtime makes it available. Use direct command-execution tools only when appropriate instead of pretending a command ran.';

  async function requestCompletion(retrySystemMessages = []) {
    const messages = [
      {
        role: 'system',
        content: baseSystemPrompt,
      },
      ...extraSystemMessages,
      ...retrySystemMessages,
      ...transcript,
    ];
    const response = await fetch(`${env.vicunaBaseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        ...vicunaHeaders(),
        'X-Vicuna-Telegram-Chat-Id': String(chatId),
        'X-Vicuna-Telegram-Message-Id': String(messageId || 0),
        'X-Vicuna-Telegram-History-Turns': String(historyTurns),
      },
      body: JSON.stringify({
        model: env.model,
        temperature: 0.2,
        messages,
        max_tokens: env.maxTokens,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(`vicuna chat failed: ${response.status} ${JSON.stringify(body)}`);
    }
    return body;
  }

  log('forwarding Telegram transcript to Vicuna', {
    chatId: String(chatId),
    messageCount: transcript.length,
    roles: transcript.map((entry) => entry.role),
    maxTokens: env.maxTokens,
  });

  const firstBody = await requestCompletion();
  let reply = extractChatCompletionText(firstBody);
  if (reply) {
    return reply;
  }

  log('Vicuna returned no relayable assistant text for Telegram request', {
    chatId: String(chatId),
    attempt: 1,
    completion: summarizeChatCompletion(firstBody),
  });

  const retryBody = await requestCompletion([
    {
      role: 'system',
      content: 'Your previous reply for this Telegram turn contained no relayable assistant text. Reply with a short plain-text assistant message for the user and do not emit an empty response.',
    },
  ]);
  reply = extractChatCompletionText(retryBody);
  if (reply) {
    log('Vicuna retry recovered Telegram assistant text', {
      chatId: String(chatId),
      attempt: 2,
      completion: summarizeChatCompletion(retryBody),
    });
    return reply;
  }

  log('Vicuna retry still returned no relayable assistant text', {
    chatId: String(chatId),
    attempt: 2,
    completion: summarizeChatCompletion(retryBody),
  });
  return '';
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
    const ingestion = await ingestTelegramDocumentMessage({
      message,
      maxDocumentChars: env.maxDocumentChars,
      resolveTelegramFile: (fileId) => telegramRequest('getFile', { file_id: fileId }),
      downloadTelegramFile,
      extractPdfText,
      extractWordText,
      supermemoryClient: getSupermemoryClient(),
      toFileFactory: (buffer, fileName) => toFile(buffer, fileName),
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
      ingestion.transcriptText,
      { maxHistoryMessages: env.maxHistoryMessages },
    );
    await persistState();
    log('appended Telegram document user turn', transcriptSummary(message.chat.id));

    const reply = await callVicunaForTelegramMessage(message.chat.id, message.message_id);
    if (reply) {
      state = appendChatTranscriptMessage(
        state,
        message.chat.id,
        'assistant',
        reply,
        { maxHistoryMessages: env.maxHistoryMessages },
      );
      await persistState();
      log('appended Telegram assistant turn', transcriptSummary(message.chat.id));

      await sendTelegramMessage(
        message.chat.id,
        reply,
        { reply_to_message_id: message.message_id },
      );
    } else {
      log('Telegram document turn produced no direct assistant text; assuming tool-delivered or deferred output', {
        chatId: String(message.chat.id),
        messageId: message.message_id,
      });
    }
    return;
  }

  if (!text) {
    await sendTelegramMessage(
      message.chat.id,
      'Only plain text, PDF, DOC, and DOCX messages are supported right now.',
      { reply_to_message_id: message.message_id },
    );
    return;
  }

  state = appendChatTranscriptMessage(
    state,
    message.chat.id,
    'user',
    text,
    { maxHistoryMessages: env.maxHistoryMessages },
  );
  await persistState();
  log('appended Telegram user turn', transcriptSummary(message.chat.id));

  const reply = await callVicunaForTelegramMessage(message.chat.id, message.message_id);
  if (reply) {
    state = appendChatTranscriptMessage(
      state,
      message.chat.id,
      'assistant',
      reply,
      { maxHistoryMessages: env.maxHistoryMessages },
    );
    await persistState();
    log('appended Telegram assistant turn', transcriptSummary(message.chat.id));

    await sendTelegramMessage(
      message.chat.id,
      reply,
      { reply_to_message_id: message.message_id },
    );
  } else {
    log('Telegram turn produced no direct assistant text; assuming tool-delivered or deferred output', {
      chatId: String(message.chat.id),
      messageId: message.message_id,
    });
  }
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

  await telegramRequest('answerCallbackQuery', {
    callback_query_id: callbackQuery.id,
    text: `Selected: ${selectedOption}`,
    show_alert: false,
  });

  if (chatId && messageId) {
    await telegramRequest('editMessageReplyMarkup', {
      chat_id: chatId,
      message_id: messageId,
      reply_markup: { inline_keyboard: [] },
    }).catch(() => {});
  }

  state = deletePendingOptionPrompt(state, parsed.promptId, { maxHistoryMessages: env.maxHistoryMessages });
  state = appendChatTranscriptMessage(
    state,
    chatId,
    'user',
    `Selected option for "${prompt.question}": ${selectedOption}`,
    { maxHistoryMessages: env.maxHistoryMessages },
  );
  await persistState();
  log('appended Telegram option-selection user turn', transcriptSummary(chatId));

  const reply = await callVicunaForTelegramMessage(
    chatId,
    messageId,
    [{
      role: 'system',
      content: 'The latest user turn is an inline-option selection answering an earlier assistant question. Continue from that selected option and the recent transcript.',
    }],
  );

  if (reply) {
    state = appendChatTranscriptMessage(
      state,
      chatId,
      'assistant',
      reply,
      { maxHistoryMessages: env.maxHistoryMessages },
    );
    await persistState();
    log('appended Telegram assistant turn after option selection', transcriptSummary(chatId));

    await sendTelegramMessage(
      chatId,
      reply,
      messageId ? { reply_to_message_id: messageId } : {},
    );
  } else {
    log('Telegram option-selection turn produced no direct assistant text; assuming tool-delivered or deferred output', {
      chatId: String(chatId),
      messageId,
      promptId: parsed.promptId,
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
      const items = Array.isArray(body?.items) ? body.items : [];
      for (const item of items) {
        const sequenceNumber = Number(item?.sequence_number ?? 0) || 0;
        if (item?.kind !== 'ask_with_options') {
          state.telegramOutboxOffset = Math.max(state.telegramOutboxOffset, sequenceNumber);
          await persistState();
          continue;
        }

        const chatId = String(item?.chat_scope ?? '').trim();
        const question = String(item?.question ?? '').trim();
        const options = Array.isArray(item?.options)
          ? item.options.map((value) => String(value ?? '').trim()).filter(Boolean)
          : [];
        if (!chatId || !question || options.length < 2) {
          throw new Error('runtime telegram outbox ask_with_options item was incomplete');
        }

        const promptId = `o${sequenceNumber.toString(36)}`;
        const result = await sendTelegramMessage(chatId, question, {
          reply_markup: {
            inline_keyboard: options.map((label, index) => ([{
              text: label,
              callback_data: buildOptionCallbackData(promptId, index),
            }])),
          },
        });

        state = appendChatTranscriptMessage(
          state,
          chatId,
          'assistant',
          question,
          { maxHistoryMessages: env.maxHistoryMessages },
        );
        state = setPendingOptionPrompt(state, promptId, {
          chatId,
          question,
          options,
          telegramMessageId: Number(result?.message_id ?? 0) || 0,
          createdAtMs: Date.now(),
        }, { maxHistoryMessages: env.maxHistoryMessages });
        state.telegramOutboxOffset = Math.max(state.telegramOutboxOffset, sequenceNumber);
        await persistState();
        log('delivered Telegram ask-with-options prompt', {
          chatId,
          promptId,
          optionCount: options.length,
        });
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
  });
  await Promise.all([
    pollTelegramLoop(),
    pollTelegramAskOutboxLoop(),
    runSelfEmitStreamLoop(),
    watchdogLoop(),
  ]);
}

await main();
