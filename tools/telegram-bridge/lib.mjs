import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

export const DEFAULT_STATE = {
  schemaVersion: 3,
  telegramOffset: 0,
  telegramOutboxOffset: 0,
  telegramOutboxCheckpointInitialized: false,
  telegramOutboxDeliveryReceipt: null,
  chatIds: [],
  proactiveResponseIds: [],
  nextConversationOrdinal: 1,
  chatSessions: {},
  chatConversationState: {},
  pendingOptionPrompts: {},
};

export const DEFAULT_MAX_HISTORY_MESSAGES = 12;
export const DEFAULT_MAX_DOCUMENT_CHARS = 12000;
export const DEFAULT_MAX_DOCUMENT_CHUNKS = 128;
export const DEFAULT_MAX_PENDING_OPTION_PROMPTS = 32;
export const DEFAULT_MAX_CONVERSATION_MESSAGE_LINKS = 64;
export const DEFAULT_PROVIDER_MAX_TOKENS = 256;
export const TELEGRAM_BRIDGE_STATE_SCHEMA_VERSION = 3;
export const TELEGRAM_BRIDGE_ASK_OUTBOX_POLL_IDLE_MS = 200;
export const TELEGRAM_BRIDGE_SELF_EMIT_ACTIVE_DELAY_MS = 250;
export const TELEGRAM_BRIDGE_SELF_EMIT_ERROR_DELAY_MS = 250;
export const TELEGRAM_BRIDGE_WATCHDOG_DELAY_MS = 1000;
export const TELEGRAM_RELAY_ALLOWED_METHODS = [
  'sendMessage',
  'sendPhoto',
  'sendDocument',
  'sendAudio',
  'sendVoice',
  'sendVideo',
  'sendAnimation',
  'sendSticker',
  'sendMediaGroup',
  'sendLocation',
  'sendVenue',
  'sendContact',
  'sendPoll',
  'sendDice',
];

export function buildTelegramChatCompletionRequest({
  model,
  transcript,
  maxTokens = DEFAULT_PROVIDER_MAX_TOKENS,
  temperature = 0.2,
}) {
  const messages = Array.isArray(transcript) ? structuredClone(transcript) : [];
  const payload = {
    model: String(model ?? '').trim(),
    temperature,
    messages,
  };
  payload.max_tokens = DEFAULT_PROVIDER_MAX_TOKENS;
  return payload;
}

export function hasQueuedTelegramDelivery(body) {
  return body?.vicuna_telegram_delivery?.handled === true;
}

function deriveTelegramSummaryText(method, payload) {
  const text = typeof payload?.text === 'string' ? payload.text.trim() : '';
  if (text) {
    return text;
  }
  const caption = typeof payload?.caption === 'string' ? payload.caption.trim() : '';
  if (caption) {
    return caption;
  }
  if (method === 'sendPoll') {
    return typeof payload?.question === 'string' && payload.question.trim()
      ? payload.question.trim()
      : 'Telegram poll sent.';
  }
  if (method === 'sendVenue') {
    const title = typeof payload?.title === 'string' ? payload.title.trim() : '';
    const address = typeof payload?.address === 'string' ? payload.address.trim() : '';
    if (title && address) {
      return `${title}\n${address}`;
    }
    return title || address || 'Telegram venue sent.';
  }
  if (method === 'sendContact') {
    const firstName = typeof payload?.first_name === 'string' ? payload.first_name.trim() : '';
    const phoneNumber = typeof payload?.phone_number === 'string' ? payload.phone_number.trim() : '';
    if (firstName && phoneNumber) {
      return `${firstName}\n${phoneNumber}`;
    }
    return firstName || phoneNumber || 'Telegram contact sent.';
  }
  if (method === 'sendMediaGroup') {
    const media = Array.isArray(payload?.media) ? payload.media : [];
    for (const item of media) {
      const mediaCaption = typeof item?.caption === 'string' ? item.caption.trim() : '';
      if (mediaCaption) {
        return mediaCaption;
      }
    }
    return 'Telegram media group sent.';
  }
  if (method === 'sendLocation') {
    return 'Telegram location sent.';
  }
  if (method === 'sendDice') {
    return 'Telegram dice sent.';
  }
  if (method === 'sendSticker') {
    return 'Telegram sticker sent.';
  }
  return `Telegram ${method} sent.`;
}

export function isTelegramCarryableAssistantMessage(content) {
  const trimmed = String(content ?? '').trim();
  if (!trimmed) {
    return false;
  }
  const lowered = trimmed.toLowerCase();
  if (!lowered.startsWith('i ran into a problem while working on that request')) {
    return true;
  }
  return !(
    lowered.includes('vicuna chat failed:') ||
    lowered.includes('exceeds the available context size') ||
    lowered.includes('failed to continue authoritative react turn') ||
    lowered.includes('"type":"server_error"') ||
    lowered.includes('"type":"exceed_context_size_error"')
  );
}

const PDF_MIME_TYPES = new Set([
  'application/pdf',
]);

const DOCX_MIME_TYPES = new Set([
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
]);

const CSV_MIME_TYPES = new Set([
  'text/csv',
  'application/csv',
  'application/vnd.ms-excel',
]);

const HTML_MIME_TYPES = new Set([
  'text/html',
  'application/xhtml+xml',
]);

const MARKDOWN_MIME_TYPES = new Set([
  'text/markdown',
  'text/x-markdown',
]);

const PPTX_MIME_TYPES = new Set([
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
]);

const XLSX_MIME_TYPES = new Set([
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
]);

const EXTENSION_TO_LOGICAL_TYPE = new Map([
  ['pdf', 'pdf'],
  ['docx', 'docx'],
  ['pptx', 'pptx'],
  ['xlsx', 'xlsx'],
  ['csv', 'csv'],
  ['md', 'markdown'],
  ['markdown', 'markdown'],
  ['html', 'html'],
  ['htm', 'html'],
  ['xhtml', 'html'],
  ['adoc', 'asciidoc'],
  ['asciidoc', 'asciidoc'],
  ['tex', 'latex'],
  ['ltx', 'latex'],
]);

const DEFAULT_TELEGRAM_DOCUMENT_CONTAINER_TAG = 'vicuna-telegram-documents';

function normalizeMimeType(value) {
  return String(value ?? '').trim().toLowerCase();
}

function normalizeFileExtension(fileName) {
  const ext = path.extname(String(fileName ?? '')).trim().toLowerCase();
  return ext.startsWith('.') ? ext.slice(1) : ext;
}

function sanitizeIdentifierPart(value, fallback = 'unknown') {
  const normalized = String(value ?? '')
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
  return normalized || fallback;
}

function truncateIdentifier(value, maxLength = 100) {
  return String(value ?? '').slice(0, maxLength);
}

export function detectTelegramDocumentLogicalType(document) {
  const mimeType = normalizeMimeType(document?.mime_type);
  const extension = normalizeFileExtension(document?.file_name);

  if (EXTENSION_TO_LOGICAL_TYPE.has(extension)) {
    return EXTENSION_TO_LOGICAL_TYPE.get(extension);
  }
  if (PDF_MIME_TYPES.has(mimeType)) {
    return 'pdf';
  }
  if (DOCX_MIME_TYPES.has(mimeType)) {
    return 'docx';
  }
  if (PPTX_MIME_TYPES.has(mimeType)) {
    return 'pptx';
  }
  if (XLSX_MIME_TYPES.has(mimeType)) {
    return 'xlsx';
  }
  if (CSV_MIME_TYPES.has(mimeType)) {
    return 'csv';
  }
  if (HTML_MIME_TYPES.has(mimeType)) {
    return 'html';
  }
  if (MARKDOWN_MIME_TYPES.has(mimeType)) {
    return 'markdown';
  }
  return 'unsupported';
}

export function buildTelegramDocumentDescriptor(message) {
  const document = message?.document ?? {};
  const chatId = String(message?.chat?.id ?? '');
  const messageId = String(message?.message_id ?? '');
  const fileName = String(document.file_name ?? '').trim() || `telegram-document-${messageId || 'unknown'}`;
  return {
    chatId,
    messageId,
    fileId: String(document.file_id ?? '').trim(),
    fileUniqueId: String(document.file_unique_id ?? '').trim(),
    fileName,
    mimeType: normalizeMimeType(document.mime_type),
    fileSize: Number(document.file_size ?? 0) || 0,
    extension: normalizeFileExtension(fileName),
    logicalType: detectTelegramDocumentLogicalType(document),
  };
}

export function isSupportedTelegramDocument(message) {
  return buildTelegramDocumentDescriptor(message).logicalType !== 'unsupported';
}

export function normalizeDocumentPlainText(text, options = {}) {
  const maxChars = Math.max(1, parseInteger(options.maxChars, DEFAULT_MAX_DOCUMENT_CHARS));
  let normalized = String(text ?? '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\f/g, '\n')
    .replace(/\u0000/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  let truncated = false;
  if (normalized.length > maxChars) {
    normalized = normalized.slice(0, maxChars).trimEnd();
    truncated = true;
  }

  return {
    text: normalized,
    truncated,
    characterCount: normalized.length,
    lineCount: normalized ? normalized.split('\n').length : 0,
  };
}

export function formatTelegramDocumentTranscript(descriptor, normalizedText) {
  const lines = [
    `[Uploaded document: ${descriptor.fileName}]`,
    `Parsed contents of ${descriptor.fileName}`,
  ];
  if (normalizedText.truncated) {
    lines.push('[Notice: parsed contents truncated]');
  }
  lines.push('', normalizedText.text);
  return lines.join('\n').trim();
}

export function buildTelegramDocumentLinkage(descriptor, options = {}) {
  const configuredContainerTag = sanitizeIdentifierPart(
    String(options.containerTag ?? '').trim(),
    DEFAULT_TELEGRAM_DOCUMENT_CONTAINER_TAG,
  );
  const fileUniquePart = sanitizeIdentifierPart(descriptor.fileUniqueId || descriptor.fileId || descriptor.fileName);
  const linkageSeed = [
    descriptor.chatId,
    descriptor.messageId,
    descriptor.fileId,
    descriptor.fileUniqueId,
    descriptor.fileName,
  ].join(':');
  const linkKey = `telegram-doc-${createHash('sha256').update(linkageSeed).digest('hex').slice(0, 24)}`;
  const containerTag = truncateIdentifier(configuredContainerTag, 100);
  const rawCustomId = truncateIdentifier(`telegram-file-${linkKey}`, 100);
  const parsedCustomId = truncateIdentifier(`telegram-parsed-${linkKey}`, 100);
  const chunkKeyPrefix = truncateIdentifier(`telegram-chunk-${linkKey}`, 100);

  const baseMetadata = {
    source: 'telegram_bridge',
    linkKey,
    documentTitle: descriptor.fileName,
    telegramChatId: descriptor.chatId,
    telegramMessageId: descriptor.messageId,
    telegramFileId: descriptor.fileId,
    telegramFileUniqueId: descriptor.fileUniqueId || fileUniquePart,
    telegramFileName: descriptor.fileName,
    telegramMimeType: descriptor.mimeType || 'unknown',
    telegramLogicalType: descriptor.logicalType,
  };

  return {
    linkKey,
    containerTag,
    rawCustomId,
    parsedCustomId,
    chunkKeyPrefix,
    rawMetadata: {
      ...baseMetadata,
      contentKind: 'source_file',
    },
    parsedMetadata: {
      ...baseMetadata,
      contentKind: 'parsed_output',
    },
  };
}

export function buildTelegramFileUrl(botToken, filePath) {
  return `https://api.telegram.org/file/bot${botToken}/${String(filePath ?? '').replace(/^\/+/, '')}`;
}

function readFirstNonEmptyString(...values) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim().length > 0) {
      return value.trim();
    }
  }
  return '';
}

function parseChunkIndex(value, fallback) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

function normalizeParsedChunk(chunk, fallbackIndex) {
  if (!chunk || typeof chunk !== 'object' || Array.isArray(chunk)) {
    return null;
  }
  const contextualText = readFirstNonEmptyString(
    chunk.contextual_text,
    chunk.contextualText,
    chunk.chunk_text,
    chunk.chunkText,
    chunk.content,
    chunk.text,
  );
  if (!contextualText) {
    return null;
  }
  return {
    chunkIndex: parseChunkIndex(chunk.chunk_index ?? chunk.chunkIndex, fallbackIndex),
    contextualText,
    sourceText: readFirstNonEmptyString(
      chunk.source_text,
      chunk.sourceText,
      chunk.text,
      chunk.content,
      contextualText,
    ),
  };
}

export function normalizeDoclingParsedDocument(parsedDocument, descriptor, options = {}) {
  if (!parsedDocument || typeof parsedDocument !== 'object' || Array.isArray(parsedDocument)) {
    throw new Error('Docling parser returned a non-object payload.');
  }

  const transcriptSource = readFirstNonEmptyString(
    parsedDocument.plain_text,
    parsedDocument.plainText,
    parsedDocument.parsed_markdown,
    parsedDocument.parsedMarkdown,
  );
  if (!transcriptSource) {
    throw new Error(`Docling parser returned no parsed contents for ${descriptor.fileName}.`);
  }

  const parsedMarkdown = readFirstNonEmptyString(
    parsedDocument.parsed_markdown,
    parsedDocument.parsedMarkdown,
    transcriptSource,
  );
  const maxChunks = Math.max(1, parseInteger(options.maxChunks, DEFAULT_MAX_DOCUMENT_CHUNKS));
  const rawChunks = Array.isArray(parsedDocument.chunks) ? parsedDocument.chunks : [];
  const chunks = rawChunks
    .map((chunk, index) => normalizeParsedChunk(chunk, index))
    .filter(Boolean)
    .sort((lhs, rhs) => lhs.chunkIndex - rhs.chunkIndex)
    .slice(0, maxChunks);

  return {
    title: readFirstNonEmptyString(parsedDocument.title, descriptor.fileName),
    parsedMarkdown,
    transcript: normalizeDocumentPlainText(transcriptSource, {
      maxChars: options.maxChars,
    }),
    chunks,
    chunkCount: rawChunks.length,
    storedChunkCount: chunks.length,
    doclingVersion: readFirstNonEmptyString(parsedDocument.docling_version, parsedDocument.doclingVersion),
  };
}

export function buildTelegramParsedChunkMemories(options = {}) {
  const {
    descriptor,
    linkage,
    parsedDocument,
    rawDocumentId = '',
    parsedDocumentId = '',
    runtimeIdentity = 'vicuna',
  } = options;

  return (parsedDocument?.chunks ?? []).map((chunk, index) => ({
    content: chunk.contextualText,
    isStatic: true,
    metadata: {
      source: 'telegram_bridge',
      runtimeIdentity,
      kind: 'tool_observation',
      domain: 'strategy',
      key: `${linkage.chunkKeyPrefix}-${String(index).padStart(4, '0')}`,
      title: `${parsedDocument.title} chunk ${index + 1}`,
      tags: ['telegram_document', 'parsed_chunk'],
      importance: 0.6,
      confidence: 1,
      gainBias: 0.1,
      allostaticRelevance: 0,
      linkKey: linkage.linkKey,
      contentKind: 'parsed_chunk',
      documentTitle: parsedDocument.title,
      telegramChatId: descriptor.chatId,
      telegramMessageId: descriptor.messageId,
      telegramFileName: descriptor.fileName,
      telegramLogicalType: descriptor.logicalType,
      rawDocumentId,
      parsedDocumentId,
      chunkIndex: chunk.chunkIndex,
      chunkCount: parsedDocument.chunks.length,
      sourceText: chunk.sourceText,
      doclingVersion: parsedDocument.doclingVersion || undefined,
    },
  }));
}

export async function ingestTelegramDocumentMessage(options) {
  const {
    message,
    maxDocumentChars = DEFAULT_MAX_DOCUMENT_CHARS,
    maxDocumentChunks = DEFAULT_MAX_DOCUMENT_CHUNKS,
    resolveTelegramFile,
    downloadTelegramFile,
    parseDocument,
    supermemoryClient,
    toFileFactory,
    writeChunkMemories,
    runtimeIdentity = 'vicuna',
    documentContainerTag = DEFAULT_TELEGRAM_DOCUMENT_CONTAINER_TAG,
  } = options ?? {};

  const descriptor = buildTelegramDocumentDescriptor(message);
  if (descriptor.logicalType === 'unsupported') {
    return {
      ok: false,
      descriptor,
      userError: 'Only Docling-supported document uploads such as PDF and DOCX are supported right now.',
    };
  }

  if (!supermemoryClient) {
    return {
      ok: false,
      descriptor,
      userError: 'SUPERMEMORY_API_KEY is required for Telegram document ingestion.',
    };
  }

  if (typeof resolveTelegramFile !== 'function' || typeof downloadTelegramFile !== 'function') {
    throw new Error('Telegram document ingestion requires file resolution and download helpers.');
  }
  if (typeof parseDocument !== 'function') {
    throw new Error('Telegram document ingestion requires a Docling parse helper.');
  }
  if (typeof writeChunkMemories !== 'function') {
    throw new Error('Telegram document ingestion requires a chunk-memory writer.');
  }

  let linkage = null;
  let normalized = null;
  let rawDocumentId = '';
  let parsedDocumentId = '';
  try {
    const fileInfo = await resolveTelegramFile(descriptor.fileId);
    const filePath = String(fileInfo?.file_path ?? '').trim();
    if (!filePath) {
      throw new Error('Telegram getFile did not return a file_path.');
    }

    const fileBuffer = await downloadTelegramFile(filePath, descriptor);
    if (!fileBuffer || fileBuffer.length === 0) {
      throw new Error('Telegram document download returned no bytes.');
    }

    const parsedDocument = normalizeDoclingParsedDocument(
      await parseDocument({ fileBuffer, descriptor }),
      descriptor,
      { maxChars: maxDocumentChars, maxChunks: maxDocumentChunks },
    );
    normalized = parsedDocument.transcript;
    if (!normalized.text) {
      return {
        ok: false,
        descriptor,
        userError: `No extractable text was found in ${descriptor.fileName}.`,
      };
    }

    linkage = buildTelegramDocumentLinkage(descriptor, {
      containerTag: documentContainerTag,
    });
    const rawUploadFile = await toFileFactory(fileBuffer, descriptor.fileName);

    const rawUpload = await supermemoryClient.documents.uploadFile({
      file: rawUploadFile,
      metadata: JSON.stringify(linkage.rawMetadata),
      containerTags: JSON.stringify([linkage.containerTag]),
    });
    rawDocumentId = String(rawUpload?.id ?? '').trim();
    if (!rawDocumentId) {
      throw new Error('Supermemory raw upload did not return an id.');
    }

    await supermemoryClient.documents.update(rawDocumentId, {
      containerTag: linkage.containerTag,
      customId: linkage.rawCustomId,
      metadata: linkage.rawMetadata,
    });

    const parsedUpload = await supermemoryClient.documents.add({
      content: parsedDocument.parsedMarkdown,
      containerTag: linkage.containerTag,
      customId: linkage.parsedCustomId,
      metadata: linkage.parsedMetadata,
    });
    parsedDocumentId = String(parsedUpload?.id ?? '').trim();
    if (!parsedDocumentId) {
      throw new Error('Supermemory parsed-output add did not return an id.');
    }

    const chunkMemories = buildTelegramParsedChunkMemories({
      descriptor,
      linkage,
      parsedDocument,
      rawDocumentId,
      parsedDocumentId,
      runtimeIdentity,
    });
    if (chunkMemories.length === 0) {
      throw new Error(`Docling produced no searchable chunks for ${descriptor.fileName}.`);
    }
    await writeChunkMemories({
      containerTag: linkage.containerTag,
      memories: chunkMemories,
    });

    return {
      ok: true,
      descriptor,
      linkage,
      normalized,
      transcriptText: formatTelegramDocumentTranscript(descriptor, normalized),
      rawDocumentId,
      parsedDocumentId,
      storedChunkCount: chunkMemories.length,
    };
  } catch (error) {
    const messageText = error instanceof Error ? error.message : String(error);
    const partialStage = parsedDocumentId
      ? 'parsed output storage'
      : rawDocumentId
        ? 'raw file storage'
        : '';
    return {
      ok: false,
      descriptor,
      linkage,
      normalized,
      rawDocumentId,
      parsedDocumentId,
      partialFailure: Boolean(rawDocumentId || parsedDocumentId),
      userError: partialStage
        ? `Telegram document ingestion partially failed after ${partialStage}: ${messageText}`
        : `Telegram document ingestion failed: ${messageText}`,
    };
  }
}

export function parseInteger(value, fallback) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function uniqueStrings(values) {
  return [...new Set((values ?? []).map((value) => String(value)))];
}

function normalizeTranscriptMessage(raw) {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const role = raw.role === 'assistant' ? 'assistant' : raw.role === 'user' ? 'user' : '';
  const content = typeof raw.content === 'string' ? raw.content.trim() : '';
  if (!role || !content) {
    return null;
  }
  if (role === 'assistant' && !isTelegramCarryableAssistantMessage(content)) {
    return null;
  }
  const conversationId = typeof raw.conversationId === 'string' ? raw.conversationId.trim() : '';
  const telegramMessageId = Math.max(0, Number(raw.telegramMessageId ?? 0) || 0);
  return {
    role,
    content,
    ...(conversationId ? { conversationId } : {}),
    ...(telegramMessageId > 0 ? { telegramMessageId } : {}),
  };
}

export function retainCoherentTranscriptWindow(messages, options = {}) {
  const maxHistoryMessages = Math.max(1, parseInteger(options.maxHistoryMessages, DEFAULT_MAX_HISTORY_MESSAGES));
  const bounded = (messages ?? []).slice(-maxHistoryMessages);
  if (bounded.length <= 1) {
    return bounded;
  }

  let startIndex = 0;
  while (startIndex < bounded.length - 1 && bounded[startIndex].role === 'assistant') {
    startIndex += 1;
  }
  return bounded.slice(startIndex);
}

function normalizeChatSessions(raw, maxHistoryMessages) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {};
  }
  const chatSessions = {};
  for (const [chatId, session] of Object.entries(raw)) {
    const messages = Array.isArray(session?.messages)
      ? retainCoherentTranscriptWindow(
        session.messages.map(normalizeTranscriptMessage).filter(Boolean),
        { maxHistoryMessages },
      )
      : [];
    if (messages.length > 0) {
      chatSessions[String(chatId)] = { messages };
    }
  }
  return chatSessions;
}

function normalizeChatConversationState(raw, maxMessageLinks) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {};
  }
  const chatConversationState = {};
  for (const [chatId, entry] of Object.entries(raw)) {
    const latestConversationId = typeof entry?.latestConversationId === 'string'
      ? entry.latestConversationId.trim()
      : '';
    const messageEntries = [];
    if (entry?.messageToConversation && typeof entry.messageToConversation === 'object' && !Array.isArray(entry.messageToConversation)) {
      for (const [messageId, conversationIdRaw] of Object.entries(entry.messageToConversation)) {
        const telegramMessageId = Math.max(0, Number(messageId) || 0);
        const conversationId = typeof conversationIdRaw === 'string' ? conversationIdRaw.trim() : '';
        if (telegramMessageId > 0 && conversationId) {
          messageEntries.push([String(telegramMessageId), conversationId]);
        }
      }
    }
    messageEntries.sort((lhs, rhs) => Number(lhs[0]) - Number(rhs[0]));
    if (latestConversationId || messageEntries.length > 0) {
      chatConversationState[String(chatId)] = {
        latestConversationId,
        messageToConversation: Object.fromEntries(messageEntries.slice(-maxMessageLinks)),
      };
    }
  }
  return chatConversationState;
}

function normalizePendingOptionPrompts(raw, maxPendingPrompts) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {};
  }
  const entries = [];
  for (const [promptId, prompt] of Object.entries(raw)) {
    const kind = prompt?.kind === 'approval_request' ? 'approval_request' : 'ask_with_options';
    const approvalId = typeof prompt?.approvalId === 'string' ? prompt.approvalId.trim() : '';
    const question = typeof prompt?.question === 'string' ? prompt.question.trim() : '';
    const options = Array.isArray(prompt?.options)
      ? prompt.options
        .map((value) => String(value ?? '').trim())
        .filter(Boolean)
        .slice(0, 6)
      : [];
    const chatId = String(prompt?.chatId ?? '').trim();
    if (!promptId || !question || options.length < 2 || !chatId) {
      continue;
    }
    if (kind === 'approval_request' && !approvalId) {
      continue;
    }
    entries.push([
      String(promptId),
      {
        kind,
        approvalId,
        chatId,
        question,
        options,
        conversationId: typeof prompt?.conversationId === 'string' ? prompt.conversationId.trim() : '',
        telegramMessageId: Math.max(0, Number(prompt?.telegramMessageId ?? 0) || 0),
        createdAtMs: Math.max(0, Number(prompt?.createdAtMs ?? 0) || 0),
      },
    ]);
  }
  entries.sort((lhs, rhs) => lhs[1].createdAtMs - rhs[1].createdAtMs);
  return Object.fromEntries(entries.slice(-maxPendingPrompts));
}

function normalizeTelegramOutboxDeliveryReceipt(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }

  const sequenceNumber = Math.max(0, Number(raw.sequenceNumber ?? 0) || 0);
  const chatId = String(raw.chatId ?? '').trim();
  const replyToMessageId = Math.max(0, Number(raw.replyToMessageId ?? 0) || 0);
  const telegramMessageId = Math.max(0, Number(raw.telegramMessageId ?? 0) || 0);
  const deliveredAtMs = Math.max(0, Number(raw.deliveredAtMs ?? 0) || 0);
  const deliveryModeRaw = String(raw.deliveryMode ?? '').trim().toLowerCase();
  const deliveryMode = deliveryModeRaw === 'fallback_no_reply'
    ? 'fallback_no_reply'
    : deliveryModeRaw === 'reply'
      ? 'reply'
      : '';
  const animation = normalizeTelegramOutboxDeliveryAnimation(raw.animation);
  if (sequenceNumber <= 0 || !chatId || telegramMessageId <= 0 || !deliveryMode) {
    return null;
  }
  return {
    sequenceNumber,
    chatId,
    replyToMessageId,
    deliveryMode,
    telegramMessageId,
    deliveredAtMs,
    ...(animation ? { animation } : {}),
  };
}

function normalizeTelegramOutboxDeliveryAnimation(raw) {
  if (raw == null) {
    return null;
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }
  const requested = Boolean(raw.requested);
  const status = String(raw.status ?? '').trim().toLowerCase();
  const stage = String(raw.stage ?? '').trim().toLowerCase();
  const normalizedStatus = ['sent', 'skipped', 'failed'].includes(status) ? status : '';
  const normalizedStage = ['not_requested', 'render', 'encode', 'upload', 'complete'].includes(stage) ? stage : '';
  if (!normalizedStatus || !normalizedStage) {
    return null;
  }
  const animation = {
    requested,
    status: normalizedStatus,
    stage: normalizedStage,
    keyframeCount: Math.max(0, Number(raw.keyframeCount ?? 0) || 0),
    durationSeconds: Math.max(0, Number(raw.durationSeconds ?? 0) || 0),
  };
  const animationMessageId = Math.max(0, Number(raw.telegramMessageId ?? 0) || 0);
  const failureReason = String(raw.failureReason ?? '').trim();
  if (animationMessageId > 0) {
    animation.telegramMessageId = animationMessageId;
  }
  if (failureReason) {
    animation.failureReason = failureReason;
  }
  return animation;
}

export function normalizeState(raw, options = {}) {
  const state = raw && typeof raw === 'object' ? raw : {};
  const maxHistoryMessages = Math.max(1, parseInteger(options.maxHistoryMessages, DEFAULT_MAX_HISTORY_MESSAGES));
  const maxPendingPrompts = Math.max(1, parseInteger(options.maxPendingOptionPrompts, DEFAULT_MAX_PENDING_OPTION_PROMPTS));
  const maxConversationMessageLinks = Math.max(1, parseInteger(options.maxConversationMessageLinks, DEFAULT_MAX_CONVERSATION_MESSAGE_LINKS));
  const chatSessions = normalizeChatSessions(state.chatSessions, maxHistoryMessages);
  const telegramOutboxDeliveryReceipt = normalizeTelegramOutboxDeliveryReceipt(state.telegramOutboxDeliveryReceipt);
  const telegramOutboxOffset = Math.max(0, parseInteger(state.telegramOutboxOffset, 0));
  const checkpointInitialized = typeof state.telegramOutboxCheckpointInitialized === 'boolean'
    ? state.telegramOutboxCheckpointInitialized
    : telegramOutboxOffset > 0 || Boolean(telegramOutboxDeliveryReceipt);
  return {
    schemaVersion: Math.max(1, parseInteger(state.schemaVersion, TELEGRAM_BRIDGE_STATE_SCHEMA_VERSION)),
    telegramOffset: Math.max(0, parseInteger(state.telegramOffset, 0)),
    telegramOutboxOffset,
    telegramOutboxCheckpointInitialized: checkpointInitialized,
    telegramOutboxDeliveryReceipt,
    chatIds: uniqueStrings([...(state.chatIds ?? []), ...Object.keys(chatSessions)]),
    proactiveResponseIds: uniqueStrings(state.proactiveResponseIds).slice(-256),
    nextConversationOrdinal: Math.max(1, parseInteger(state.nextConversationOrdinal, 1)),
    chatSessions,
    chatConversationState: normalizeChatConversationState(state.chatConversationState, maxConversationMessageLinks),
    pendingOptionPrompts: normalizePendingOptionPrompts(state.pendingOptionPrompts, maxPendingPrompts),
  };
}

export function reconcileTelegramOutboxOffset(currentOffset, outboxState) {
  const offset = Math.max(0, parseInteger(currentOffset, 0));
  const storedItems = Math.max(0, parseInteger(outboxState?.stored_items, 0));
  const newestSequence = Math.max(0, parseInteger(outboxState?.newest_sequence, 0));
  const oldestSequence = Math.max(0, parseInteger(outboxState?.oldest_sequence, 0));

  if (storedItems <= 0) {
    return 0;
  }

  if (storedItems > 0 && newestSequence > 0 && offset > newestSequence) {
    return Math.max(0, oldestSequence > 0 ? oldestSequence - 1 : 0);
  }

  if (oldestSequence > 0 && offset + 1 < oldestSequence) {
    return Math.max(0, oldestSequence - 1);
  }

  return offset;
}

export function normalizeTelegramOutboxItem(item) {
  const sequenceNumber = Math.max(0, parseInteger(item?.sequence_number, 0));
  const kind = String(item?.kind ?? '').trim();
  const base = {
    ok: false,
    skippable: sequenceNumber > 0,
    sequenceNumber,
    kind,
  };

  if (!kind) {
    return {
      ...base,
      error: 'runtime telegram outbox item was missing kind',
    };
  }

  if (kind === 'message') {
    const chatId = String(item?.chat_scope ?? '').trim();
    const telegramMethod = String(item?.telegram_method ?? '').trim() || 'sendMessage';
    const hasStructuredPayload = Boolean(item && typeof item === 'object' && Object.prototype.hasOwnProperty.call(item, 'telegram_payload'));
    const hasEmotiveAnimation = Boolean(item && typeof item === 'object' && Object.prototype.hasOwnProperty.call(item, 'emotive_animation'));
    if (hasStructuredPayload && (!item?.telegram_payload || typeof item.telegram_payload !== 'object' || Array.isArray(item.telegram_payload))) {
      return {
        ...base,
        error: 'runtime telegram outbox message item had invalid telegram_payload',
      };
    }
    if (hasEmotiveAnimation && (!item?.emotive_animation || typeof item.emotive_animation !== 'object' || Array.isArray(item.emotive_animation))) {
      return {
        ...base,
        error: 'runtime telegram outbox message item had invalid emotive_animation',
      };
    }
    const telegramPayload = hasStructuredPayload
      ? item.telegram_payload
      : { text: String(item?.text ?? '').trim() };
    const text = String(item?.text ?? '').trim() || deriveTelegramSummaryText(telegramMethod, telegramPayload);
    const replyToMessageId = Number(item?.reply_to_message_id ?? 0) || 0;
    if (!chatId || !text) {
      return {
        ...base,
        error: 'runtime telegram outbox message item was incomplete',
      };
    }
    return {
      ok: true,
      skippable: true,
      sequenceNumber,
      kind,
      chatId,
      text,
      telegramMethod,
      telegramPayload,
      replyToMessageId,
      emotiveAnimation: hasEmotiveAnimation ? structuredClone(item.emotive_animation) : null,
    };
  }

  if (kind === 'ask_with_options') {
    const chatId = String(item?.chat_scope ?? '').trim();
    const question = String(item?.question ?? '').trim();
    const options = Array.isArray(item?.options)
      ? item.options.map((value) => String(value ?? '').trim()).filter(Boolean)
      : [];
    if (!chatId || !question || options.length < 2) {
      return {
        ...base,
        error: 'runtime telegram outbox ask_with_options item was incomplete',
      };
    }
    return {
      ok: true,
      skippable: true,
      sequenceNumber,
      kind,
      chatId,
      question,
      options,
    };
  }

  if (kind === 'approval_request') {
    const approvalId = String(item?.approval_id ?? '').trim();
    const chatId = String(item?.chat_scope ?? '').trim();
    const question = String(item?.question ?? '').trim();
    const options = Array.isArray(item?.options)
      ? item.options.map((value) => String(value ?? '').trim()).filter(Boolean)
      : [];
    const replyToMessageId = Number(item?.reply_to_message_id ?? 0) || 0;
    if (!approvalId || !chatId || !question || options.length < 2 || !options.includes('disallow')) {
      return {
        ...base,
        error: 'runtime telegram outbox approval_request item was incomplete',
      };
    }
    return {
      ok: true,
      skippable: true,
      sequenceNumber,
      kind,
      approvalId,
      chatId,
      question,
      options,
      replyToMessageId,
    };
  }

  return {
    ...base,
    error: `runtime telegram outbox item kind '${kind}' is unsupported`,
  };
}

export async function loadState(statePath, options = {}) {
  try {
    const contents = await readFile(statePath, 'utf8');
    return normalizeState(JSON.parse(contents), options);
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return normalizeState(DEFAULT_STATE, options);
    }
    throw error;
  }
}

export async function saveState(statePath, state, options = {}) {
  await mkdir(path.dirname(statePath), { recursive: true });
  await writeFile(statePath, `${JSON.stringify(normalizeState(state, options), null, 2)}\n`, 'utf8');
}

export function setTelegramOutboxCheckpoint(state, nextOffset, options = {}) {
  const normalizedState = normalizeState(state, options);
  return {
    ...normalizedState,
    telegramOutboxOffset: Math.max(0, Number(nextOffset ?? 0) || 0),
    telegramOutboxCheckpointInitialized: true,
  };
}

export function recordTelegramOutboxDeliveryReceipt(state, receipt, options = {}) {
  const normalizedState = normalizeState(state, options);
  const normalizedReceipt = normalizeTelegramOutboxDeliveryReceipt(receipt);
  if (!normalizedReceipt) {
    return normalizedState;
  }
  return {
    ...normalizedState,
    telegramOutboxCheckpointInitialized: true,
    telegramOutboxDeliveryReceipt: normalizedReceipt,
  };
}

export function shouldBootstrapTelegramOutboxOffset(state, outboxState, options = {}) {
  const normalizedState = normalizeState(state, options);
  if (normalizedState.telegramOutboxCheckpointInitialized) {
    return false;
  }
  const storedItems = Math.max(0, parseInteger(outboxState?.stored_items, 0));
  const newestSequence = Math.max(0, parseInteger(outboxState?.newest_sequence, 0));
  return storedItems > 0 && newestSequence > 0;
}

export function bootstrapTelegramOutboxOffset(state, outboxState, options = {}) {
  const normalizedState = normalizeState(state, options);
  const newestSequence = Math.max(0, parseInteger(outboxState?.newest_sequence, 0));
  return {
    ...normalizedState,
    telegramOutboxOffset: newestSequence,
    telegramOutboxCheckpointInitialized: true,
  };
}

export function isTelegramReplyTargetErrorMessage(message) {
  const normalized = String(message ?? '').trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return normalized.includes('reply message not found') ||
    normalized.includes('message to be replied not found') ||
    normalized.includes('message_id_invalid') ||
    normalized.includes('message thread not found') ||
    normalized.includes('replied message not found');
}

export function isTelegramTerminalDeliveryErrorMessage(message) {
  const normalized = String(message ?? '').trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return normalized.includes('chat not found') ||
    normalized.includes('bot was blocked by the user') ||
    normalized.includes('user is deactivated') ||
    normalized.includes('bot was kicked from the group chat') ||
    normalized.includes('bot was kicked from the supergroup chat') ||
    normalized.includes('group chat was deleted') ||
    normalized.includes('bot is not a member of the channel chat');
}

export function appendProactiveId(state, responseId) {
  const proactiveResponseIds = uniqueStrings([
    ...state.proactiveResponseIds,
    responseId,
  ]).slice(-256);
  return {
    ...state,
    proactiveResponseIds,
  };
}

export function registerChat(state, chatId) {
  const chatKey = String(chatId);
  return {
    ...state,
    chatIds: uniqueStrings([...state.chatIds, chatKey]),
  };
}

export function updateTelegramOffset(state, updateId) {
  const nextOffset = Math.max(state.telegramOffset, Number(updateId) + 1);
  return {
    ...state,
    telegramOffset: nextOffset,
  };
}

function withLatestConversationId(state, chatId, conversationId) {
  const chatKey = String(chatId).trim();
  const trimmedConversationId = String(conversationId ?? '').trim();
  if (!chatKey || !trimmedConversationId) {
    return state;
  }
  return {
    ...state,
    chatConversationState: {
      ...(state.chatConversationState ?? {}),
      [chatKey]: {
        latestConversationId: trimmedConversationId,
        messageToConversation: {
          ...(state?.chatConversationState?.[chatKey]?.messageToConversation ?? {}),
        },
      },
    },
  };
}

export function getChatTranscript(state, chatId, options = {}) {
  const chatKey = String(chatId);
  const messages = Array.isArray(state?.chatSessions?.[chatKey]?.messages)
    ? state.chatSessions[chatKey].messages
    : [];
  const conversationId = typeof options.conversationId === 'string'
    ? options.conversationId.trim()
    : '';
  if (!conversationId) {
    return messages;
  }
  return messages.filter((entry) => String(entry?.conversationId ?? '').trim() === conversationId);
}

export function getTelegramRequestDeltaMessages(state, chatId, options = {}) {
  const transcript = getChatTranscript(state, chatId, options);
  for (let index = transcript.length - 1; index >= 0; index -= 1) {
    const entry = transcript[index];
    if (entry?.role === 'user' && typeof entry.content === 'string' && entry.content.trim()) {
      return [{
        role: 'user',
        content: entry.content,
      }];
    }
  }
  return [];
}

export function appendChatTranscriptMessage(state, chatId, role, content, options = {}) {
  const conversationId = typeof options.conversationId === 'string' ? options.conversationId.trim() : '';
  const telegramMessageId = Math.max(0, Number(options.telegramMessageId ?? 0) || 0);
  const normalized = normalizeTranscriptMessage({
    role,
    content,
    ...(conversationId ? { conversationId } : {}),
    ...(telegramMessageId > 0 ? { telegramMessageId } : {}),
  });
  if (!normalized) {
    return state;
  }
  const chatKey = String(chatId);
  const maxHistoryMessages = Math.max(1, parseInteger(options.maxHistoryMessages, DEFAULT_MAX_HISTORY_MESSAGES));
  const maxConversationMessageLinks = Math.max(1, parseInteger(options.maxConversationMessageLinks, DEFAULT_MAX_CONVERSATION_MESSAGE_LINKS));
  const normalizedState = normalizeState(state, options);
  const nextMessages = retainCoherentTranscriptWindow([
    ...getChatTranscript(normalizedState, chatKey),
    normalized,
  ], { maxHistoryMessages });

  const chatConversationEntry = {
    latestConversationId: String(normalizedState?.chatConversationState?.[chatKey]?.latestConversationId ?? '').trim(),
    messageToConversation: {
      ...(normalizedState?.chatConversationState?.[chatKey]?.messageToConversation ?? {}),
    },
  };
  if (conversationId) {
    chatConversationEntry.latestConversationId = conversationId;
  }
  if (conversationId && telegramMessageId > 0) {
    chatConversationEntry.messageToConversation[String(telegramMessageId)] = conversationId;
    const messageEntries = Object.entries(chatConversationEntry.messageToConversation)
      .filter(([messageId, value]) => (Number(messageId) || 0) > 0 && String(value ?? '').trim())
      .sort((lhs, rhs) => Number(lhs[0]) - Number(rhs[0]));
    chatConversationEntry.messageToConversation = Object.fromEntries(messageEntries.slice(-maxConversationMessageLinks));
  }

  return {
    ...registerChat(normalizedState, chatKey),
    chatSessions: {
      ...(normalizedState.chatSessions ?? {}),
      [chatKey]: {
        messages: nextMessages,
      },
    },
    chatConversationState: {
      ...(normalizedState.chatConversationState ?? {}),
      [chatKey]: chatConversationEntry,
    },
  };
}

export function setPendingOptionPrompt(state, promptId, prompt, options = {}) {
  const maxPendingPrompts = Math.max(1, parseInteger(options.maxPendingOptionPrompts, DEFAULT_MAX_PENDING_OPTION_PROMPTS));
  const normalizedState = normalizeState(state, options);
  const nextPrompts = {
    ...(normalizedState.pendingOptionPrompts ?? {}),
    [String(promptId)]: {
      kind: prompt?.kind === 'approval_request' ? 'approval_request' : 'ask_with_options',
      approvalId: String(prompt?.approvalId ?? '').trim(),
      chatId: String(prompt?.chatId ?? '').trim(),
      question: String(prompt?.question ?? '').trim(),
      options: Array.isArray(prompt?.options) ? prompt.options.map((value) => String(value ?? '').trim()).filter(Boolean).slice(0, 6) : [],
      conversationId: String(prompt?.conversationId ?? '').trim(),
      telegramMessageId: Math.max(0, Number(prompt?.telegramMessageId ?? 0) || 0),
      createdAtMs: Math.max(0, Number(prompt?.createdAtMs ?? Date.now()) || Date.now()),
    },
  };
  return {
    ...normalizedState,
    pendingOptionPrompts: normalizePendingOptionPrompts(nextPrompts, maxPendingPrompts),
  };
}

export function getPendingOptionPrompt(state, promptId) {
  return state?.pendingOptionPrompts?.[String(promptId)] ?? null;
}

export function deletePendingOptionPrompt(state, promptId, options = {}) {
  const normalizedState = normalizeState(state, options);
  const nextPrompts = { ...(normalizedState.pendingOptionPrompts ?? {}) };
  delete nextPrompts[String(promptId)];
  return {
    ...normalizedState,
    pendingOptionPrompts: nextPrompts,
  };
}

export function getLatestConversationId(state, chatId) {
  return String(state?.chatConversationState?.[String(chatId)]?.latestConversationId ?? '').trim();
}

export function getConversationForTelegramMessage(state, chatId, telegramMessageId) {
  const messageId = Math.max(0, Number(telegramMessageId ?? 0) || 0);
  if (messageId <= 0) {
    return '';
  }
  return String(
    state?.chatConversationState?.[String(chatId)]?.messageToConversation?.[String(messageId)] ?? '',
  ).trim();
}

export function createTelegramConversation(state, chatId, options = {}) {
  const chatKey = String(chatId);
  const normalizedState = normalizeState(state, options);
  const ordinal = Math.max(1, Number(normalizedState.nextConversationOrdinal) || 1);
  const conversationId = `tc${ordinal.toString(36)}`;
  return {
    state: {
      ...registerChat(normalizedState, chatKey),
      nextConversationOrdinal: ordinal + 1,
      chatConversationState: {
        ...(normalizedState.chatConversationState ?? {}),
        [chatKey]: {
          latestConversationId: conversationId,
          messageToConversation: {
            ...(normalizedState?.chatConversationState?.[chatKey]?.messageToConversation ?? {}),
          },
        },
      },
    },
    conversationId,
    reason: 'created',
  };
}

export function resolveTelegramConversationForMessage(state, message, options = {}) {
  const chatId = String(message?.chat?.id ?? '').trim();
  if (!chatId) {
    return { state, conversationId: '', reason: 'missing_chat' };
  }

  const normalizedState = normalizeState(state, options);
  const explicitReplyConversationId = getConversationForTelegramMessage(
    normalizedState,
    chatId,
    Number(message?.reply_to_message?.message_id ?? 0) || 0,
  );
  if (explicitReplyConversationId) {
    return {
      state: withLatestConversationId(normalizedState, chatId, explicitReplyConversationId),
      conversationId: explicitReplyConversationId,
      reason: 'reply_to_message',
    };
  }

  const latestConversationId = getLatestConversationId(normalizedState, chatId);
  if (latestConversationId) {
    return {
      state: withLatestConversationId(normalizedState, chatId, latestConversationId),
      conversationId: latestConversationId,
      reason: 'latest',
    };
  }

  return createTelegramConversation(normalizedState, chatId, options);
}

export function resolveTelegramConversationForOutbound(state, chatId, options = {}) {
  const chatKey = String(chatId).trim();
  if (!chatKey) {
    return { state, conversationId: '', reason: 'missing_chat' };
  }

  const normalizedState = normalizeState(state, options);
  const preferredConversationId = typeof options.preferredConversationId === 'string'
    ? options.preferredConversationId.trim()
    : '';
  if (preferredConversationId) {
    return {
      state: withLatestConversationId(normalizedState, chatKey, preferredConversationId),
      conversationId: preferredConversationId,
      reason: 'preferred',
    };
  }

  const replyConversationId = getConversationForTelegramMessage(
    normalizedState,
    chatKey,
    Number(options.replyToMessageId ?? 0) || 0,
  );
  if (replyConversationId) {
    return {
      state: withLatestConversationId(normalizedState, chatKey, replyConversationId),
      conversationId: replyConversationId,
      reason: 'reply_to_message',
    };
  }

  const latestConversationId = getLatestConversationId(normalizedState, chatKey);
  if (latestConversationId) {
    return {
      state: normalizedState,
      conversationId: latestConversationId,
      reason: 'latest',
    };
  }

  return createTelegramConversation(normalizedState, chatKey, options);
}

export function sanitizeAssistantRelayText(text) {
  const input = String(text ?? '');
  if (!input.trim()) {
    return '';
  }

  const patterns = [
    /<vicuna_tool_call\b[\s\S]*?<\/vicuna_tool_call>/gi,
    /<minimax:tool_call\b[\s\S]*?<\/minimax:tool_call>/gi,
    /<tool_call\b[\s\S]*?<\/tool_call>/gi,
  ];

  let sanitized = input;
  for (const pattern of patterns) {
    sanitized = sanitized.replace(pattern, '');
  }

  sanitized = sanitized
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n');

  return sanitized.trim();
}

export function extractResponseText(response) {
  if (!response || typeof response !== 'object') {
    return '';
  }
  const output = Array.isArray(response.output) ? response.output : [];
  const parts = [];
  for (const item of output) {
    if (!item || item.role !== 'assistant' || !Array.isArray(item.content)) {
      continue;
    }
    for (const content of item.content) {
      if (!content || typeof content !== 'object') {
        continue;
      }
      if (typeof content.text === 'string' && content.text.trim() !== '') {
        parts.push(content.text);
      }
    }
  }
  return sanitizeAssistantRelayText(parts.join('\n\n'));
}

export function extractChatCompletionText(body) {
  const content = body?.choices?.[0]?.message?.content;
  if (typeof content === 'string') {
    return sanitizeAssistantRelayText(content);
  }
  if (Array.isArray(content)) {
    return sanitizeAssistantRelayText(content
      .map((part) => (typeof part?.text === 'string' ? part.text : ''))
      .join('\n')
    );
  }
  return '';
}

function cloneJsonValue(value) {
  if (value === undefined) {
    return undefined;
  }
  return JSON.parse(JSON.stringify(value));
}

export function extractChatCompletionToolCalls(body) {
  const toolCalls = body?.choices?.[0]?.message?.tool_calls;
  if (!Array.isArray(toolCalls)) {
    return [];
  }
  return toolCalls
    .filter((toolCall) => toolCall && typeof toolCall === 'object' && toolCall.type === 'function')
    .filter((toolCall) => typeof toolCall?.function?.name === 'string' && toolCall.function.name.trim() !== '')
    .map((toolCall) => cloneJsonValue(toolCall));
}

export function extractAssistantToolReplayMessage(body) {
  const message = body?.choices?.[0]?.message;
  if (!message || typeof message !== 'object') {
    return null;
  }
  const replay = {
    role: typeof message.role === 'string' && message.role.trim() ? message.role : 'assistant',
  };
  if (typeof message.content === 'string') {
    replay.content = message.content;
  } else if (Array.isArray(message.content)) {
    replay.content = cloneJsonValue(message.content);
  } else if (!Array.isArray(message.tool_calls) || message.tool_calls.length === 0) {
    replay.content = '';
  }
  if (typeof message.reasoning_content === 'string') {
    replay.reasoning_content = message.reasoning_content;
  }
  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    replay.tool_calls = cloneJsonValue(message.tool_calls);
  }
  return replay;
}

export function stringifyToolObservation(observation) {
  if (typeof observation === 'string') {
    return observation;
  }
  return JSON.stringify(observation ?? {});
}

export function buildChatCompletionToolResultMessage(toolCall, observation) {
  return {
    role: 'tool',
    tool_call_id: String(toolCall?.id ?? ''),
    name: String(toolCall?.function?.name ?? ''),
    content: stringifyToolObservation(observation),
  };
}

export function summarizeChatCompletion(body) {
  const choice = body?.choices?.[0];
  const message = choice?.message;
  const content = message?.content;
  return {
    finishReason: choice?.finish_reason ?? null,
    messageRole: typeof message?.role === 'string' ? message.role : '',
    contentType: Array.isArray(content) ? 'array' : typeof content,
    textLength: extractChatCompletionText(body).length,
    contentPartCount: Array.isArray(content) ? content.length : 0,
    toolCallCount: Array.isArray(message?.tool_calls) ? message.tool_calls.length : 0,
    reasoningLength:
      typeof message?.reasoning_content === 'string'
        ? message.reasoning_content.length
        : 0,
  };
}

export function parseSseChunk(chunk) {
  const events = [];
  const normalized = chunk.replace(/\r\n/g, '\n');
  for (const block of normalized.split('\n\n')) {
    const trimmed = block.trim();
    if (!trimmed) {
      continue;
    }
    let eventName = 'message';
    const dataLines = [];
    for (const line of trimmed.split('\n')) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length === 0) {
      continue;
    }
    events.push({
      event: eventName,
      data: JSON.parse(dataLines.join('\n')),
    });
  }
  return events;
}

export function splitSseBuffer(buffer) {
  const normalized = buffer.replace(/\r\n/g, '\n');
  const boundary = normalized.lastIndexOf('\n\n');
  if (boundary === -1) {
    return { complete: '', remainder: buffer };
  }
  return {
    complete: normalized.slice(0, boundary + 2),
    remainder: normalized.slice(boundary + 2),
  };
}

export function formatTelegramMessage(prefix, text) {
  const body = String(text ?? '').trim();
  if (!body) {
    return '';
  }
  return prefix ? `${prefix}\n\n${body}` : body;
}
