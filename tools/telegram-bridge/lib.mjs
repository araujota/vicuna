import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

export const DEFAULT_STATE = {
  schemaVersion: 3,
  telegramOffset: 0,
  telegramOutboxOffset: 0,
  telegramOutboxCheckpointInitialized: false,
  telegramOutboxDeliveryReceipt: null,
  telegramEmotiveAnimationState: {},
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
export const TELEGRAM_VIDEO_CAPTION_LIMIT = 1024;
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

function telegramHtmlEscape(text) {
  return String(text ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function sanitizeTelegramCodeLanguage(language) {
  return String(language ?? '')
    .trim()
    .replace(/[^A-Za-z0-9._+-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 32);
}

function isMarkdownDividerLine(value) {
  const trimmed = String(value ?? '').trim();
  if (trimmed.length < 3) {
    return false;
  }
  const marker = trimmed[0];
  if (!['-', '*', '_'].includes(marker)) {
    return false;
  }
  return [...trimmed].every((character) => character === marker);
}

function hasMarkdownishFormatting(text) {
  const value = String(text ?? '');
  return (
    /(^|\n)#{1,6}\s+\S/.test(value) ||
    /(^|\n)([-*_])\2\2+\s*($|\n)/.test(value) ||
    /(^|\n)>\s?/.test(value) ||
    /(^|\n)\|.+\|\s*\n\|(?:\s*:?-{3,}:?\s*\|){1,}\s*($|\n)/.test(value) ||
    /```/.test(value) ||
    /\*\*\S[\s\S]*?\S\*\*/.test(value) ||
    /(^|[^\*])\*\S(?:[\s\S]*?\S)?\*(?!\*)/.test(value) ||
    /`[^`\n]+`/.test(value)
  );
}

function hasTelegramHtmlFormatting(text) {
  const value = String(text ?? '');
  return /<(?:\/?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|blockquote|tg-spoiler)|a\s+href=|span\s+class="tg-spoiler")(?=[\s>])[^>]*>/i.test(value);
}

function hasHtmlishFormatting(text) {
  const value = String(text ?? '');
  return /<\/?[A-Za-z][^>]*>/.test(value);
}

function stripTelegramHtmlTags(text) {
  return decodeBasicHtmlEntities(String(text ?? '').replace(/<[^>]+>/g, ''));
}

function decodeBasicHtmlEntities(text) {
  return String(text ?? '')
    .replaceAll('&quot;', '"')
    .replaceAll('&gt;', '>')
    .replaceAll('&lt;', '<')
    .replaceAll('&amp;', '&');
}

function normalizeTelegramHtmlishToSupportedHtml(text) {
  const protectedTags = [];
  const protect = (value) => {
    const token = `__VICUNA_TG_HTML_${protectedTags.length}__`;
    protectedTags.push(value);
    return token;
  };

  let html = decodeBasicHtmlEntities(text)
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/<\s*span\s+class=(["'])tg-spoiler\1\s*>([\s\S]*?)<\s*\/span\s*>/gi, '<tg-spoiler>$2</tg-spoiler>')
    .replace(/<\s*br\s*\/?\s*>/gi, '\n')
    .replace(/<\s*\/p\s*>/gi, '\n\n')
    .replace(/<\s*p(?:\s[^>]*)?>/gi, '')
    .replace(/<\s*\/div\s*>/gi, '\n')
    .replace(/<\s*div(?:\s[^>]*)?>/gi, '')
    .replace(/<\s*li(?:\s[^>]*)?>/gi, '• ')
    .replace(/<\s*\/li\s*>/gi, '\n')
    .replace(/<\s*\/(?:ul|ol)\s*>/gi, '\n')
    .replace(/<\s*(?:ul|ol)(?:\s[^>]*)?>/gi, '')
    .replace(/<\s*h([1-6])(?:\s[^>]*)?>/gi, '<b>')
    .replace(/<\s*\/h([1-6])\s*>/gi, '</b>\n\n')
    .replace(/<\s*strong(?=[\s>])[^>]*>/gi, '<b>')
    .replace(/<\s*\/strong\s*>/gi, '</b>')
    .replace(/<\s*em(?=[\s>])[^>]*>/gi, '<i>')
    .replace(/<\s*\/em\s*>/gi, '</i>')
    .replace(/<\s*ins(?=[\s>])[^>]*>/gi, '<u>')
    .replace(/<\s*\/ins\s*>/gi, '</u>')
    .replace(/<\s*(?:strike|del)(?=[\s>])[^>]*>/gi, '<s>')
    .replace(/<\s*\/(?:strike|del)\s*>/gi, '</s>')
    .replace(/<\s*blockquote(?:\s+expandable)?(?=[\s>])[^>]*>/gi, '<blockquote>');

  html = html.replace(/<\s*code\b([^>]*)>/gi, (_match, attributes) => {
    const languageMatch = /class\s*=\s*(["'])language-([^"']+)\1/i.exec(attributes);
    if (!languageMatch) {
      return '<code>';
    }
    const language = sanitizeTelegramCodeLanguage(languageMatch[2]);
    return language ? `<code class="language-${language}">` : '<code>';
  });
  html = html.replace(/<\s*a\b([^>]*)>/gi, (_match, attributes) => {
    const hrefMatch = /href\s*=\s*(["'])(.*?)\1/i.exec(attributes);
    if (!hrefMatch) {
      return '';
    }
    const href = telegramHtmlEscape(decodeBasicHtmlEntities(hrefMatch[2]));
    return `<a href="${href}">`;
  });

  html = html.replace(/<(?:\/?(?:b|i|u|s|code|pre|blockquote|tg-spoiler)|a\s+href="[^"]*"|\/a|code\s+class="language-[^"]+")>/gi, (match) => protect(match));
  html = html.replace(/<[^>]+>/g, '');
  html = telegramHtmlEscape(html);
  for (const [index, tag] of protectedTags.entries()) {
    html = html.replaceAll(`__VICUNA_TG_HTML_${index}__`, tag);
  }
  return html
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+\n/g, '\n')
    .trim();
}

function canonicalizeTelegramTextToHtml(text, parseMode = '') {
  const sourceText = String(text ?? '').trim();
  if (!sourceText) {
    return '';
  }

  const normalizedParseMode = String(parseMode ?? '').trim().toUpperCase();
  let rendered = sourceText;
  if (normalizedParseMode === 'MARKDOWN' || normalizedParseMode === 'MARKDOWNV2' || hasMarkdownishFormatting(rendered)) {
    rendered = convertMarkdownishToTelegramHtml(rendered);
  }

  if (normalizedParseMode === 'HTML' || hasTelegramHtmlFormatting(rendered) || hasHtmlishFormatting(rendered)) {
    return normalizeTelegramHtmlishToSupportedHtml(rendered);
  }

  return telegramHtmlEscape(rendered);
}

function convertMarkdownInlineToTelegramHtml(text) {
  const value = String(text ?? '');
  let html = '';
  for (let index = 0; index < value.length;) {
    if (value.slice(index, index + 3) === '***') {
      const closing = value.indexOf('***', index + 3);
      if (closing !== -1 && closing > index + 3) {
        html += `<b><i>${convertMarkdownInlineToTelegramHtml(value.slice(index + 3, closing))}</i></b>`;
        index = closing + 3;
        continue;
      }
    }
    if (value.slice(index, index + 2) === '**') {
      const closing = value.indexOf('**', index + 2);
      if (closing !== -1 && closing > index + 2) {
        html += `<b>${convertMarkdownInlineToTelegramHtml(value.slice(index + 2, closing))}</b>`;
        index = closing + 2;
        continue;
      }
    }
    if (value.slice(index, index + 2) === '__') {
      const closing = value.indexOf('__', index + 2);
      if (closing !== -1 && closing > index + 2) {
        html += `<u>${convertMarkdownInlineToTelegramHtml(value.slice(index + 2, closing))}</u>`;
        index = closing + 2;
        continue;
      }
    }
    if (value.slice(index, index + 2) === '~~') {
      const closing = value.indexOf('~~', index + 2);
      if (closing !== -1 && closing > index + 2) {
        html += `<s>${convertMarkdownInlineToTelegramHtml(value.slice(index + 2, closing))}</s>`;
        index = closing + 2;
        continue;
      }
    }
    if (value[index] === '*') {
      const closing = value.indexOf('*', index + 1);
      if (closing !== -1 && closing > index + 1) {
        html += `<i>${convertMarkdownInlineToTelegramHtml(value.slice(index + 1, closing))}</i>`;
        index = closing + 1;
        continue;
      }
    }
    if (value[index] === '`') {
      const closing = value.indexOf('`', index + 1);
      if (closing !== -1 && closing > index + 1) {
        html += `<code>${telegramHtmlEscape(value.slice(index + 1, closing))}</code>`;
        index = closing + 1;
        continue;
      }
    }
    html += telegramHtmlEscape(value[index]);
    index += 1;
  }
  return html;
}

function renderMarkdownParagraphLineToTelegramHtml(line) {
  const trimmed = String(line ?? '').trim();
  if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
    return `• ${convertMarkdownInlineToTelegramHtml(trimmed.slice(2))}`;
  }
  return convertMarkdownInlineToTelegramHtml(trimmed);
}

function parseMarkdownTableCells(line) {
  const trimmed = String(line ?? '').trim();
  if (!trimmed.includes('|')) {
    return [];
  }
  let inner = trimmed;
  if (inner.startsWith('|')) {
    inner = inner.slice(1);
  }
  if (inner.endsWith('|')) {
    inner = inner.slice(0, -1);
  }
  const cells = inner.split('|').map((cell) => cell.trim());
  return cells.length >= 2 ? cells : [];
}

function isMarkdownTableSeparatorRow(line) {
  const cells = parseMarkdownTableCells(line);
  return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function markdownTableCellToPlainText(cell) {
  return stripTelegramHtmlTags(convertMarkdownInlineToTelegramHtml(String(cell ?? '').trim()));
}

function renderMarkdownTableToTelegramHtml(headerCells, bodyRows) {
  const plainRows = [headerCells, ...bodyRows].map((row) => row.map(markdownTableCellToPlainText));
  const widths = plainRows[0].map((_, index) => (
    plainRows.reduce((maxWidth, row) => Math.max(maxWidth, String(row[index] ?? '').length), 0)
  ));
  const border = `+${widths.map((width) => '-'.repeat(width + 2)).join('+')}+`;
  const renderRow = (row) => `| ${row.map((cell, index) => String(cell ?? '').padEnd(widths[index], ' ')).join(' | ')} |`;
  const lines = [border, renderRow(plainRows[0]), border];
  for (const row of plainRows.slice(1)) {
    lines.push(renderRow(row));
  }
  lines.push(border);
  return `<pre>${telegramHtmlEscape(lines.join('\n'))}</pre>`;
}

function convertMarkdownishToTelegramHtml(markdown) {
  const blocks = [];
  const paragraphLines = [];

  const flushParagraph = () => {
    if (paragraphLines.length === 0) {
      return;
    }
    blocks.push(paragraphLines.map(renderMarkdownParagraphLineToTelegramHtml).join('\n'));
    paragraphLines.length = 0;
  };

  const lines = String(markdown ?? '').replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      continue;
    }

    if (trimmed.startsWith('```')) {
      flushParagraph();
      const language = sanitizeTelegramCodeLanguage(trimmed.slice(3));
      const codeLines = [];
      let closed = false;
      for (index += 1; index < lines.length; index += 1) {
        if (lines[index].trim().startsWith('```')) {
          closed = true;
          break;
        }
        codeLines.push(lines[index]);
      }
      const escapedCode = telegramHtmlEscape(codeLines.join('\n'));
      blocks.push(language
        ? `<pre><code class="language-${language}">${escapedCode}</code></pre>`
        : `<pre>${escapedCode}</pre>`);
      if (!closed) {
        break;
      }
      continue;
    }

    const tableHeaderCells = parseMarkdownTableCells(line);
    if (tableHeaderCells.length >= 2 && index + 1 < lines.length && isMarkdownTableSeparatorRow(lines[index + 1])) {
      flushParagraph();
      const bodyRows = [];
      index += 1;
      while (index + 1 < lines.length) {
        const nextCells = parseMarkdownTableCells(lines[index + 1]);
        if (nextCells.length !== tableHeaderCells.length || isMarkdownTableSeparatorRow(lines[index + 1])) {
          break;
        }
        bodyRows.push(nextCells);
        index += 1;
      }
      blocks.push(renderMarkdownTableToTelegramHtml(tableHeaderCells, bodyRows));
      continue;
    }

    if (isMarkdownDividerLine(trimmed)) {
      flushParagraph();
      blocks.push('━━━━━━━━━━━━');
      continue;
    }

    const headingMatch = /^(#{1,6})\s+(.+)$/.exec(trimmed);
    if (headingMatch) {
      flushParagraph();
      blocks.push(`<b>${convertMarkdownInlineToTelegramHtml(headingMatch[2].trim())}</b>`);
      continue;
    }

    if (trimmed === '>' || trimmed.startsWith('> ')) {
      flushParagraph();
      const quoteLines = [];
      while (index < lines.length) {
        const current = lines[index].trim();
        if (!(current === '>' || current.startsWith('> '))) {
          index -= 1;
          break;
        }
        quoteLines.push(convertMarkdownInlineToTelegramHtml(current.length > 1 ? current.slice(1).trim() : ''));
        index += 1;
      }
      blocks.push(`<blockquote>${quoteLines.join('\n')}</blockquote>`);
      continue;
    }

    paragraphLines.push(line);
  }

  flushParagraph();
  return blocks.join('\n\n');
}

function getTelegramTextFieldForMethod(method) {
  if (method === 'sendMessage') {
    return 'text';
  }
  if ([
    'sendPhoto',
    'sendDocument',
    'sendAudio',
    'sendVoice',
    'sendVideo',
    'sendAnimation',
  ].includes(method)) {
    return 'caption';
  }
  return '';
}

export function normalizeTelegramRichTextPayload({
  telegramMethod,
  telegramPayload,
  text = '',
  field = '',
} = {}) {
  const method = String(telegramMethod ?? '').trim() || 'sendMessage';
  const targetField = field || getTelegramTextFieldForMethod(method);
  const payload = telegramPayload && typeof telegramPayload === 'object' && !Array.isArray(telegramPayload)
    ? cloneJsonValue(telegramPayload)
    : {};
  const sourceText = sanitizeAssistantRelayText(String(
    (typeof text === 'string' && text.trim())
      ? text
      : (typeof payload?.[targetField] === 'string' ? payload[targetField] : ''),
  )).trim();

  if (!targetField || !sourceText) {
    return {
      payload,
      method,
      field: targetField,
      sourceText,
      renderedText: sourceText,
      textLength: sourceText.length,
      normalized: false,
    };
  }

  payload[targetField] = sourceText;

  const hasEntities = targetField === 'text'
    ? Array.isArray(payload.entities) && payload.entities.length > 0
    : Array.isArray(payload.caption_entities) && payload.caption_entities.length > 0;
  const parseMode = typeof payload.parse_mode === 'string' ? payload.parse_mode.trim() : '';
  if (hasEntities) {
    const renderedText = String(payload[targetField] ?? sourceText).trim();
    const textLength = parseMode.toUpperCase() === 'HTML'
      ? stripTelegramHtmlTags(renderedText).length
      : sourceText.length;
    return {
      payload,
      method,
      field: targetField,
      sourceText,
      renderedText,
      textLength,
      normalized: false,
    };
  }
  const renderedText = canonicalizeTelegramTextToHtml(sourceText, parseMode);
  payload[targetField] = renderedText;
  payload.parse_mode = 'HTML';
  return {
    payload,
    method,
    field: targetField,
    sourceText,
    renderedText,
    textLength: stripTelegramHtmlTags(renderedText).length,
    normalized: renderedText !== sourceText || parseMode.toUpperCase() !== 'HTML',
  };
}

export function buildTelegramSpoilerVideoPayload({
  telegramMethod,
  telegramPayload,
  text,
}) {
  const method = String(telegramMethod ?? '').trim() || 'sendMessage';
  if (method !== 'sendMessage') {
    return {
      ok: false,
      reason: 'unsupported_method',
      method,
    };
  }

  const payload = telegramPayload && typeof telegramPayload === 'object' && !Array.isArray(telegramPayload)
    ? telegramPayload
    : {};
  const normalizedTextPayload = normalizeTelegramRichTextPayload({
    telegramMethod: method,
    telegramPayload: {
      ...payload,
      text: String(text ?? '').trim() || deriveTelegramSummaryText(method, payload),
    },
    text: String(text ?? '').trim() || deriveTelegramSummaryText(method, payload),
  });
  const caption = String(normalizedTextPayload.renderedText ?? '').trim();
  if (!caption) {
    return {
      ok: false,
      reason: 'missing_caption',
      method,
    };
  }

  if (normalizedTextPayload.textLength > TELEGRAM_VIDEO_CAPTION_LIMIT) {
    return {
      ok: false,
      reason: 'caption_too_long',
      method,
      captionLength: normalizedTextPayload.textLength,
      captionLimit: TELEGRAM_VIDEO_CAPTION_LIMIT,
    };
  }

  const nextPayload = {
    caption,
    show_caption_above_media: true,
    has_spoiler: true,
    supports_streaming: true,
  };
  const passthroughKeys = [
    'business_connection_id',
    'message_thread_id',
    'direct_messages_topic_id',
    'disable_notification',
    'protect_content',
    'allow_paid_broadcast',
    'suggested_post_parameters',
    'message_effect_id',
    'reply_markup',
    'reply_to_message_id',
    'reply_parameters',
  ];
  for (const key of passthroughKeys) {
    if (Object.prototype.hasOwnProperty.call(payload, key)) {
      nextPayload[key] = cloneJsonValue(payload[key]);
    }
  }

  if (Array.isArray(normalizedTextPayload.payload.entities) && normalizedTextPayload.payload.entities.length > 0) {
    nextPayload.caption_entities = cloneJsonValue(normalizedTextPayload.payload.entities);
  } else if (Array.isArray(normalizedTextPayload.payload.caption_entities) && normalizedTextPayload.payload.caption_entities.length > 0) {
    nextPayload.caption_entities = cloneJsonValue(normalizedTextPayload.payload.caption_entities);
  } else if (typeof normalizedTextPayload.payload.parse_mode === 'string' && normalizedTextPayload.payload.parse_mode.trim()) {
    nextPayload.parse_mode = normalizedTextPayload.payload.parse_mode.trim();
  }

  return {
    ok: true,
    method: 'sendVideo',
    captionLength: normalizedTextPayload.textLength,
    payload: nextPayload,
  };
}

export function buildTelegramAnimationDeliveryPlan({
  telegramMethod,
  telegramPayload,
  text,
}) {
  const method = String(telegramMethod ?? '').trim() || 'sendMessage';
  const sourceText = String(text ?? '').trim()
    || (
      telegramPayload && typeof telegramPayload === 'object' && !Array.isArray(telegramPayload)
        ? String(telegramPayload.text ?? '').trim()
        : ''
    );
  const normalizedMessagePayload = normalizeTelegramRichTextPayload({
    telegramMethod: method,
    telegramPayload,
    text: sourceText,
  });
  const spoilerVideoPayload = buildTelegramSpoilerVideoPayload({
    telegramMethod: method,
    telegramPayload,
    text: sourceText,
  });

  if (spoilerVideoPayload.ok) {
    return {
      ok: true,
      strategy: 'primary_spoiler_video',
      messagePayload: normalizedMessagePayload.payload,
      videoPayload: spoilerVideoPayload.payload,
      captionLength: spoilerVideoPayload.captionLength,
      captionLimit: TELEGRAM_VIDEO_CAPTION_LIMIT,
    };
  }

  if (spoilerVideoPayload.reason === 'caption_too_long') {
    const followupVideoPayload = {
      has_spoiler: true,
      supports_streaming: true,
    };
    const passthroughKeys = [
      'business_connection_id',
      'message_thread_id',
      'direct_messages_topic_id',
      'disable_notification',
      'protect_content',
      'allow_paid_broadcast',
      'suggested_post_parameters',
      'message_effect_id',
    ];
    for (const key of passthroughKeys) {
      if (Object.prototype.hasOwnProperty.call(normalizedMessagePayload.payload, key)) {
        followupVideoPayload[key] = cloneJsonValue(normalizedMessagePayload.payload[key]);
      }
    }
    return {
      ok: true,
      strategy: 'split_text_and_video',
      messagePayload: normalizedMessagePayload.payload,
      videoPayload: followupVideoPayload,
      captionLength: spoilerVideoPayload.captionLength,
      captionLimit: spoilerVideoPayload.captionLimit,
    };
  }

  return {
    ok: false,
    reason: spoilerVideoPayload.reason,
    method,
    messagePayload: normalizedMessagePayload.payload,
  };
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

export function buildTelegramParsedDocumentChunks(options = {}) {
  const {
    descriptor,
    linkage,
    parsedDocument,
  } = options;

  return (parsedDocument?.chunks ?? []).map((chunk, index) => ({
    chunk_index: chunk.chunkIndex,
    contextual_text: chunk.contextualText,
    source_text: chunk.sourceText,
    document_title: parsedDocument.title,
    link_key: linkage.linkKey,
    key: `${linkage.chunkKeyPrefix}-${String(index).padStart(4, '0')}`,
    telegram_chat_id: descriptor.chatId,
    telegram_message_id: descriptor.messageId,
    telegram_file_name: descriptor.fileName,
    telegram_logical_type: descriptor.logicalType,
    docling_version: parsedDocument.doclingVersion || undefined,
  }));
}

export function buildTelegramParsedChunkMemories(options = {}) {
  return buildTelegramParsedDocumentChunks(options).map((chunk, index) => ({
    content: chunk.contextual_text,
    isStatic: true,
    metadata: {
      source: 'telegram_bridge',
      key: chunk.key,
      title: `${chunk.document_title} chunk ${index + 1}`,
      tags: ['telegram_document', 'parsed_chunk'],
      linkKey: chunk.link_key,
      contentKind: 'parsed_chunk',
      documentTitle: chunk.document_title,
      telegramChatId: chunk.telegram_chat_id,
      telegramMessageId: chunk.telegram_message_id,
      telegramFileName: chunk.telegram_file_name,
      telegramLogicalType: chunk.telegram_logical_type,
      chunkIndex: chunk.chunk_index,
      chunkCount: (options.parsedDocument?.chunks ?? []).length,
      sourceText: chunk.source_text,
      doclingVersion: chunk.docling_version,
    },
  }));
}

function sanitizeStoredFileName(fileName) {
  const baseName = path.basename(String(fileName ?? '').trim() || 'telegram-document');
  const ext = path.extname(baseName);
  const stem = ext ? baseName.slice(0, -ext.length) : baseName;
  const safeStem = sanitizeIdentifierPart(stem, 'telegram-document');
  const safeExt = ext.replace(/[^A-Za-z0-9.]/g, '').toLowerCase();
  return `${safeStem}${safeExt}`;
}

async function persistTelegramDocumentBundle({
  docsRoot,
  descriptor,
  linkage,
  fileBuffer,
  parsedDocument,
  parseError,
}) {
  const bundleId = linkage.linkKey;
  const bundleDir = path.join(docsRoot, bundleId);
  const sourceDir = path.join(bundleDir, 'source');
  const sourceFileName = sanitizeStoredFileName(descriptor.fileName);
  const sourcePath = path.join(sourceDir, sourceFileName);
  const parsedMarkdownPath = path.join(bundleDir, 'parsed.md');
  const chunksPath = path.join(bundleDir, 'chunks.json');
  const metadataPath = path.join(bundleDir, 'metadata.json');

  await mkdir(sourceDir, { recursive: true });
  await writeFile(sourcePath, fileBuffer);

  const metadata = {
    bundle_id: bundleId,
    link_key: linkage.linkKey,
    document_title: descriptor.fileName,
    source_path: sourcePath,
    parsed_markdown_path: parsedDocument ? parsedMarkdownPath : null,
    chunks_path: parsedDocument ? chunksPath : null,
    metadata_path: metadataPath,
    content_kind: parsedDocument ? 'parsed_output' : 'parse_failed',
    parse_status: parsedDocument ? 'parsed' : 'parse_failed',
    parse_error: parseError || null,
    telegram_chat_id: descriptor.chatId,
    telegram_message_id: descriptor.messageId,
    telegram_file_id: descriptor.fileId,
    telegram_file_unique_id: descriptor.fileUniqueId,
    telegram_file_name: descriptor.fileName,
    telegram_mime_type: descriptor.mimeType || 'unknown',
    telegram_logical_type: descriptor.logicalType,
    docling_version: parsedDocument?.doclingVersion || null,
    stored_chunk_count: parsedDocument?.chunks?.length ?? 0,
    updated_at: new Date().toISOString(),
  };

  if (parsedDocument) {
    await writeFile(parsedMarkdownPath, `${parsedDocument.parsedMarkdown.trim()}\n`);
    await writeFile(
      chunksPath,
      `${JSON.stringify(buildTelegramParsedDocumentChunks({ descriptor, linkage, parsedDocument }), null, 2)}\n`,
    );
  }
  await writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

  return {
    bundleId,
    sourcePath,
    parsedMarkdownPath: parsedDocument ? parsedMarkdownPath : null,
    chunksPath: parsedDocument ? chunksPath : null,
  };
}

export async function ingestTelegramDocumentMessage(options) {
  const {
    message,
    maxDocumentChars = DEFAULT_MAX_DOCUMENT_CHARS,
    maxDocumentChunks = DEFAULT_MAX_DOCUMENT_CHUNKS,
    resolveTelegramFile,
    downloadTelegramFile,
    parseDocument,
    docsRoot,
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

  if (typeof resolveTelegramFile !== 'function' || typeof downloadTelegramFile !== 'function') {
    throw new Error('Telegram document ingestion requires file resolution and download helpers.');
  }
  if (typeof parseDocument !== 'function') {
    throw new Error('Telegram document ingestion requires a Docling parse helper.');
  }
  if (typeof docsRoot !== 'string' || docsRoot.trim().length === 0) {
    throw new Error('Telegram document ingestion requires a local docs root.');
  }

  let linkage = null;
  let normalized = null;
  let bundle = null;
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

    linkage = buildTelegramDocumentLinkage(descriptor, {
      containerTag: documentContainerTag,
    });

    let parsedDocument;
    try {
      parsedDocument = normalizeDoclingParsedDocument(
        await parseDocument({ fileBuffer, descriptor }),
        descriptor,
        { maxChars: maxDocumentChars, maxChunks: maxDocumentChunks },
      );
    } catch (error) {
      bundle = await persistTelegramDocumentBundle({
        docsRoot: docsRoot.trim(),
        descriptor,
        linkage,
        fileBuffer,
        parseError: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
    normalized = parsedDocument.transcript;
    if (!normalized.text) {
      return {
        ok: false,
        descriptor,
        userError: `No extractable text was found in ${descriptor.fileName}.`,
      };
    }
    const chunks = buildTelegramParsedDocumentChunks({
      descriptor,
      linkage,
      parsedDocument,
    });
    bundle = await persistTelegramDocumentBundle({
      docsRoot: docsRoot.trim(),
      descriptor,
      linkage,
      fileBuffer,
      parsedDocument,
    });
    if (chunks.length === 0) {
      throw new Error(`Docling produced no searchable chunks for ${descriptor.fileName}.`);
    }

    return {
      ok: true,
      descriptor,
      linkage,
      normalized,
      transcriptText: formatTelegramDocumentTranscript(descriptor, normalized),
      bundleId: bundle.bundleId,
      sourcePath: bundle.sourcePath,
      parsedMarkdownPath: bundle.parsedMarkdownPath,
      storedChunkCount: chunks.length,
    };
  } catch (error) {
    const messageText = error instanceof Error ? error.message : String(error);
    const partialStage = bundle ? 'local source storage' : '';
    return {
      ok: false,
      descriptor,
      linkage,
      normalized,
      bundleId: bundle?.bundleId,
      sourcePath: bundle?.sourcePath,
      partialFailure: Boolean(bundle),
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

function normalizeTelegramEmotiveMoment(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }
  const moment = {};
  for (const [key, value] of Object.entries(raw)) {
    const id = String(key ?? '').trim();
    if (!id) {
      continue;
    }
    moment[id] = Math.max(0, Math.min(1, Number(value) || 0));
  }
  return Object.keys(moment).length > 0 ? moment : null;
}

export function buildTelegramEmotiveAnimationScopeKey(chatId, conversationId = '') {
  const normalizedChatId = String(chatId ?? '').trim();
  const normalizedConversationId = String(conversationId ?? '').trim();
  return normalizedConversationId
    ? `${normalizedChatId}::${normalizedConversationId}`
    : normalizedChatId;
}

function normalizeTelegramEmotiveAnimationState(raw) {
  const source = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const normalized = {};
  for (const value of Object.values(source)) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      continue;
    }
    const chatId = String(value.chatId ?? '').trim();
    const conversationId = String(value.conversationId ?? '').trim();
    const lastMoment = normalizeTelegramEmotiveMoment(value.lastMoment);
    if (!chatId || !lastMoment) {
      continue;
    }
    const key = buildTelegramEmotiveAnimationScopeKey(chatId, conversationId);
    normalized[key] = {
      chatId,
      ...(conversationId ? { conversationId } : {}),
      lastMoment,
      lastRenderedAtMs: Math.max(0, parseInteger(value.lastRenderedAtMs, 0)),
    };
  }
  return normalized;
}

export function normalizeState(raw, options = {}) {
  const state = raw && typeof raw === 'object' ? raw : {};
  const maxHistoryMessages = Math.max(1, parseInteger(options.maxHistoryMessages, DEFAULT_MAX_HISTORY_MESSAGES));
  const maxPendingPrompts = Math.max(1, parseInteger(options.maxPendingOptionPrompts, DEFAULT_MAX_PENDING_OPTION_PROMPTS));
  const maxConversationMessageLinks = Math.max(1, parseInteger(options.maxConversationMessageLinks, DEFAULT_MAX_CONVERSATION_MESSAGE_LINKS));
  const chatSessions = normalizeChatSessions(state.chatSessions, maxHistoryMessages);
  const telegramOutboxDeliveryReceipt = normalizeTelegramOutboxDeliveryReceipt(state.telegramOutboxDeliveryReceipt);
  const telegramEmotiveAnimationState = normalizeTelegramEmotiveAnimationState(state.telegramEmotiveAnimationState);
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
    telegramEmotiveAnimationState,
    chatIds: uniqueStrings([...(state.chatIds ?? []), ...Object.keys(chatSessions)]),
    proactiveResponseIds: uniqueStrings(state.proactiveResponseIds).slice(-256),
    nextConversationOrdinal: Math.max(1, parseInteger(state.nextConversationOrdinal, 1)),
    chatSessions,
    chatConversationState: normalizeChatConversationState(state.chatConversationState, maxConversationMessageLinks),
    pendingOptionPrompts: normalizePendingOptionPrompts(state.pendingOptionPrompts, maxPendingPrompts),
  };
}

export function getTelegramEmotiveAnimationState(state, chatId, { conversationId = '' } = {}) {
  const normalizedState = normalizeState(state);
  const key = buildTelegramEmotiveAnimationScopeKey(chatId, conversationId);
  return normalizedState.telegramEmotiveAnimationState[key] ?? null;
}

export function setTelegramEmotiveAnimationState(state, chatId, animationState, {
  conversationId = '',
  ...options
} = {}) {
  const normalizedState = normalizeState(state, options);
  const key = buildTelegramEmotiveAnimationScopeKey(chatId, conversationId);
  const nextAnimationState = {
    ...normalizedState.telegramEmotiveAnimationState,
  };
  const lastMoment = normalizeTelegramEmotiveMoment(animationState?.lastMoment);
  if (!lastMoment) {
    delete nextAnimationState[key];
  } else {
    nextAnimationState[key] = {
      chatId: String(chatId ?? '').trim(),
      ...(conversationId ? { conversationId: String(conversationId).trim() } : {}),
      lastMoment,
      lastRenderedAtMs: Math.max(0, parseInteger(animationState?.lastRenderedAtMs, Date.now())),
    };
  }
  return normalizeState({
    ...normalizedState,
    telegramEmotiveAnimationState: nextAnimationState,
  }, options);
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
    /<｜DSML｜function_calls>[\s\S]*?(?:<\/｜DSML｜function_calls>|$)/g,
    /<｜DSML｜invoke\b[\s\S]*?(?:<\/｜DSML｜invoke>|$)/g,
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
