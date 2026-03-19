import http from 'node:http';
import https from 'node:https';
import { setTimeout as delay } from 'node:timers/promises';
import {
  appendProactiveId,
  appendChatTranscriptMessage,
  extractChatCompletionText,
  extractResponseText,
  formatTelegramMessage,
  getChatTranscript,
  loadState,
  parseInteger,
  parseSseChunk,
  registerChat,
  saveState,
  splitSseBuffer,
  updateTelegramOffset,
} from './lib.mjs';

const env = {
  telegramBotToken: process.env.TELEGRAM_BOT_TOKEN ?? '',
  vicunaBaseUrl: (process.env.TELEGRAM_BRIDGE_VICUNA_BASE_URL ?? 'http://127.0.0.1:8080').replace(/\/+$/, ''),
  model: process.env.TELEGRAM_BRIDGE_MODEL ?? 'qwen2.5:7b-instruct-q8_0',
  statePath: process.env.TELEGRAM_BRIDGE_STATE_PATH ?? '/tmp/vicuna-telegram-bridge-state.json',
  pollTimeoutSeconds: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS, 30)),
  maxHistoryMessages: Math.max(1, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES, 12)),
  maxTokens: Math.max(32, parseInteger(process.env.TELEGRAM_BRIDGE_MAX_TOKENS, 200)),
  selfEmitAfter: Math.max(0, parseInteger(process.env.TELEGRAM_BRIDGE_SELF_EMIT_AFTER, 0)),
  vicunaApiKey: process.env.VICUNA_API_KEY ?? '',
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

async function sendTelegramMessage(chatId, text, extra = {}) {
  const messageText = String(text ?? '').trim();
  if (!messageText) {
    return;
  }
  await telegramRequest('sendMessage', {
    chat_id: chatId,
    text: messageText,
    ...extra,
  });
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

async function callVicunaForTelegramMessage(chatId) {
  const transcript = getChatTranscript(state, chatId);
  log('forwarding Telegram transcript to Vicuna', {
    chatId: String(chatId),
    messageCount: transcript.length,
    roles: transcript.map((entry) => entry.role),
    maxTokens: env.maxTokens,
  });
  const messages = [
    {
      role: 'system',
      content: 'You are replying to a Telegram user through middleware. Keep responses clear and concise unless the user asks for depth. Maintain continuity across the provided transcript. When the user explicitly asks to search the web or needs fresh current information, prefer the web search tool if the runtime makes it available. Use direct command-execution tools only when appropriate instead of pretending a command ran.',
    },
    ...transcript,
  ];
  const response = await fetch(`${env.vicunaBaseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: vicunaHeaders(),
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
  return extractChatCompletionText(body);
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
  if (!text) {
    await sendTelegramMessage(
      message.chat.id,
      'Only plain text messages are supported right now.',
      { reply_to_message_id: message.message_id },
    );
    return;
  }

  if (text === '/start') {
    await sendTelegramMessage(
      message.chat.id,
      'Telegram relay connected. Your messages will be forwarded to the local Vicuña runtime, and proactive system emissions will be relayed here.',
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

  const reply = await callVicunaForTelegramMessage(message.chat.id);

  const finalReply = reply || 'The runtime returned an empty response.';
  state = appendChatTranscriptMessage(
    state,
    message.chat.id,
    'assistant',
    finalReply,
    { maxHistoryMessages: env.maxHistoryMessages },
  );
  await persistState();
  log('appended Telegram assistant turn', transcriptSummary(message.chat.id));

  await sendTelegramMessage(
    message.chat.id,
    finalReply,
    { reply_to_message_id: message.message_id },
  );
}

async function pollTelegramLoop() {
  for (;;) {
    try {
      const updates = await telegramRequest('getUpdates', {
        offset: state.telegramOffset,
        timeout: env.pollTimeoutSeconds,
        allowed_updates: ['message'],
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
  log(`bridge transcript settings history=${env.maxHistoryMessages} max_tokens=${env.maxTokens}`, {
    chatCount: state.chatIds.length,
    sessionCount: Object.keys(state.chatSessions ?? {}).length,
  });
  await Promise.all([
    pollTelegramLoop(),
    runSelfEmitStreamLoop(),
    watchdogLoop(),
  ]);
}

await main();
