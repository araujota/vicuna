import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  buildEmotiveAnimationRenderPlan,
  extractEmotiveAnimationTerminalMoment,
  normalizeEmotiveAnimationBundle,
  prependEmotiveAnimationStartMoment,
  renderEmotiveAnimationFrames,
} from './emotive-animation-render.mjs';
import {
  buildTelegramEmotiveAnimationBundle,
} from './emotive-animation-bundle.mjs';
import {
  ANCHOR_MAX_RADIUS,
  ANCHOR_MIN_RADIUS,
  buildFrameGeometry,
  buildOuterHullTopology,
  computeBundleRadiusNormalizationRange,
} from './emotive-animation-scene.mjs';
import {
  buildWebglRenderTimeoutMs,
  getWebglRendererHealth,
  renderEmotiveAnimationViaWebglService,
} from './emotive-webgl-renderer-client.mjs';
import { handleBridgeControlCommand } from './control-commands.mjs';

import {
  appendProactiveId,
  appendChatTranscriptMessage,
  buildTelegramAnimationDeliveryPlan,
  buildTelegramBridgeRequestTimeoutMs,
  buildTelegramPlainTextFallbackPayload,
  buildTelegramChatCompletionRequest,
  buildTelegramMessageChunkPlan,
  bootstrapTelegramOutboxOffset,
  buildTelegramDocumentDescriptor,
  buildTelegramDocumentLinkage,
  buildTelegramParsedChunkMemories,
  buildTelegramSpoilerVideoPayload,
  buildChatCompletionToolResultMessage,
  buildTelegramFileUrl,
  createTelegramConversation,
  deleteTelegramPendingVideoDelivery,
  detectTelegramDocumentLogicalType,
  enqueueTelegramPendingVideoDelivery,
  extractAssistantToolReplayMessage,
  extractChatCompletionText,
  extractChatCompletionToolCalls,
  extractResponseText,
  getConversationForTelegramMessage,
  getTelegramEmotiveAnimationState,
  getTelegramPendingVideoDelivery,
  getChatTranscript,
  getLatestConversationId,
  getTelegramRequestDeltaMessages,
  hasQueuedTelegramDelivery,
  ingestTelegramDocumentMessage,
  isTelegramCarryableAssistantMessage,
  isTelegramReplyTargetErrorMessage,
  isTelegramTerminalDeliveryErrorMessage,
  loadState,
  normalizeDoclingParsedDocument,
  normalizeDocumentPlainText,
  normalizeTelegramRichTextPayload,
  normalizeTelegramOutboxItem,
  normalizeState,
  parseSseChunk,
  recordTelegramOutboxDeliveryReceipt,
  reconcileTelegramOutboxOffset,
  retainCoherentTranscriptWindow,
  resolveTelegramConversationForMessage,
  resolveTelegramConversationForOutbound,
  saveState,
  setTelegramEmotiveAnimationState,
  sanitizeAssistantRelayText,
  setTelegramOutboxCheckpoint,
  shouldSuppressDeferredTelegramFailureMessage,
  shouldBootstrapTelegramOutboxOffset,
  splitSseBuffer,
  summarizeChatCompletion,
  TELEGRAM_BRIDGE_ASK_OUTBOX_POLL_IDLE_MS,
  TELEGRAM_BRIDGE_SELF_EMIT_ACTIVE_DELAY_MS,
  TELEGRAM_BRIDGE_SELF_EMIT_ERROR_DELAY_MS,
  TELEGRAM_MESSAGE_SAFE_RENDERED_LIMIT,
  TELEGRAM_MESSAGE_TEXT_LIMIT,
  TELEGRAM_VIDEO_CAPTION_LIMIT,
  TELEGRAM_BRIDGE_WATCHDOG_DELAY_MS,
  updateTelegramPendingVideoDelivery,
  updateTelegramOffset,
} from './lib.mjs';

function assertBalancedTelegramSupportedTags(text) {
  const tokenPattern = /<(\/)?(b|i|u|s|code|pre|blockquote|tg-spoiler|a)(?:\s+[^>]*)?>/gi;
  const stack = [];
  let match;
  while ((match = tokenPattern.exec(String(text ?? ''))) !== null) {
    const isClosingTag = Boolean(match[1]);
    const tagName = String(match[2] ?? '').toLowerCase();
    if (!isClosingTag) {
      stack.push(tagName);
      continue;
    }
    assert.equal(stack.length > 0, true, `unexpected closing tag </${tagName}> in ${text}`);
    assert.equal(stack.pop(), tagName, `mismatched closing tag </${tagName}> in ${text}`);
  }
  assert.deepEqual(stack, [], `unclosed Telegram HTML tags remained in ${text}`);
}

function sampleEmotiveAnimationBundle() {
  return {
    bundle_version: 3,
    trace_id: 'emo_test',
    generation_start_block_index: 3,
    raw_keyframe_count: 4,
    distinct_keyframe_count: 2,
    fps: 30,
    viewport_width: 720,
    viewport_height: 720,
    rotation_period_seconds: 12,
    duration_seconds: 1.8,
    dimensions: [
      { id: 'confidence', label: 'Confidence', direction_index: 0, direction_xyz: [0.577, 0.577, 0.577] },
      { id: 'caution', label: 'Caution', direction_index: 1, direction_xyz: [-0.577, -0.577, 0.577] },
      { id: 'curiosity', label: 'Curiosity', direction_index: 2, direction_xyz: [-0.577, 0.577, -0.577] },
      { id: 'satisfaction', label: 'Satisfaction', direction_index: 3, direction_xyz: [0.577, -0.577, -0.577] },
    ],
    keyframes: [
      {
        ordinal: 0,
        trace_block_index: 3,
        source_kind: 'assistant_reasoning',
        hold_keyframe_count: 2,
        trace_block_span: [3, 4],
        start_offset_ms: 0,
        end_offset_ms: 350,
        hold_duration_ms: 350,
        moment: { confidence: 0.2, caution: 0.7, curiosity: 0.35, satisfaction: 0.4 },
        dominant_dimensions: ['caution'],
      },
      {
        ordinal: 1,
        trace_block_index: 5,
        source_kind: 'assistant_content',
        hold_keyframe_count: 2,
        trace_block_span: [5, 6],
        start_offset_ms: 1200,
        end_offset_ms: 1800,
        hold_duration_ms: 600,
        moment: { confidence: 0.85, caution: 0.25, curiosity: 0.65, satisfaction: 0.75 },
        dominant_dimensions: ['confidence'],
      },
    ],
    segments: [
      {
        segment_index: 0,
        from_keyframe_ordinal: 0,
        to_keyframe_ordinal: 1,
        start_offset_ms: 350,
        duration_ms: 850,
      },
    ],
  };
}

function sampleRawEmotiveTrace() {
  return {
    trace_id: 'emo_trace_raw',
    turn_start_block_index: 1,
    live_generation_start_block_index: 2,
    blocks: [
      {
        block_index: 0,
        source: { kind: 'system_message' },
        timestamp_ms: 1000,
        moment: {
          epistemic_pressure: 0.1,
          confidence: 0.2,
        },
        vad: {
          dominant_dimensions: ['confidence'],
        },
      },
      {
        block_index: 1,
        source: { kind: 'user_message' },
        timestamp_ms: 1500,
        moment: {
          epistemic_pressure: 0.3,
          confidence: 0.7,
          caution: 0.2,
        },
        vad: {
          dominant_dimensions: ['confidence'],
        },
      },
      {
        block_index: 2,
        source: { kind: 'assistant_response' },
        timestamp_ms: 1700,
        moment: {
          epistemic_pressure: 0.3,
          confidence: 0.7,
          caution: 0.2,
        },
        vad: {
          dominant_dimensions: ['confidence', 'caution'],
        },
      },
      {
        block_index: 3,
        source: { kind: 'assistant_response' },
        timestamp_ms: 2300,
        moment: {
          epistemic_pressure: 0.6,
          confidence: 0.4,
          caution: 0.5,
        },
        vad: {
          dominant_dimensions: ['caution'],
        },
      },
    ],
  };
}

function lowSignalEmotiveAnimationBundle() {
  return {
    bundle_version: 2,
    trace_id: 'emo_low_signal',
    seconds_per_keyframe: 0.5,
    fps: 30,
    viewport_width: 720,
    viewport_height: 720,
    dimensions: sampleEmotiveAnimationBundle().dimensions,
    keyframes: [{
      ordinal: 0,
      trace_block_index: 0,
      source_kind: 'assistant_content',
      hold_keyframe_count: 1,
      moment: {
        confidence: 0.8,
        caution: 0,
        curiosity: 0,
        satisfaction: 0.12,
      },
    }, {
      ordinal: 1,
      trace_block_index: 1,
      source_kind: 'runtime_event',
      hold_keyframe_count: 1,
      moment: {
        confidence: 0.7,
        caution: 0,
        curiosity: 0,
        satisfaction: 0.105,
      },
    }],
  };
}

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

test('extractChatCompletionText strips DSML tool-call markup', () => {
  const text = extractChatCompletionText({
    choices: [
      {
        message: {
          content: "Checking.\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"web_search\">\n<｜DSML｜parameter name=\"query\" string=\"true\">Penelope's Vegan Taqueria Logan Square</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
        },
      },
    ],
  });

  assert.equal(text, 'Checking.');
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

test('handleBridgeControlCommand executes host mode helper and replies with summary', async () => {
  const sent = [];
  let execArgs = null;
  const handled = await handleBridgeControlCommand({
    repoRoot: '/tmp/vicuna',
    text: '/togglemode',
    message: {
      chat: { id: 42 },
      message_id: 9,
    },
    sendTelegramMessage: async (...args) => {
      sent.push(args);
    },
    execFileAsyncImpl: async (file, args) => {
      execArgs = { file, args };
      return {
        stdout: JSON.stringify({
          previous_mode: 'standard',
          current_mode: 'experimental',
          pod_action: 'started',
          pod_id: 'pod-123',
          tunnel_local_port: 18080,
          runtime_service: 'vicuna-runtime.service',
        }),
      };
    },
  });

  assert.equal(handled, true);
  assert.deepEqual(execArgs, {
    file: 'sudo',
    args: ['-n', '/tmp/vicuna/tools/ops/host-inference-mode.sh', 'toggle', '--json'],
  });
  assert.equal(sent.length, 1);
  assert.match(sent[0][1], /Inference mode switched: standard -> experimental\./);
  assert.match(sent[0][1], /Relay tunnel listening on 127\.0\.0\.1:18080\./);
});

test('tool-call extraction and assistant replay preserve reasoning_content exactly', () => {
  const body = {
    choices: [
      {
        finish_reason: 'tool_calls',
        message: {
          role: 'assistant',
          content: '',
          reasoning_content: 'Think first.\nThen call the runtime tool.',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: {
                name: 'runtime_health_check',
                arguments: '{"target":"runpod"}',
              },
            },
          ],
        },
      },
    ],
  };

  const toolCalls = extractChatCompletionToolCalls(body);
  assert.equal(toolCalls.length, 1);
  assert.equal(toolCalls[0].function.name, 'runtime_health_check');

  const replay = extractAssistantToolReplayMessage(body);
  assert.deepEqual(replay, {
    role: 'assistant',
    content: '',
    reasoning_content: 'Think first.\nThen call the runtime tool.',
    tool_calls: [
      {
        id: 'call_1',
        type: 'function',
        function: {
          name: 'runtime_health_check',
          arguments: '{"target":"runpod"}',
        },
      },
    ],
  });
});

test('buildChatCompletionToolResultMessage serializes tool observations as tool messages', () => {
  const toolCall = {
    id: 'call_1',
    type: 'function',
    function: {
      name: 'runtime_health_check',
      arguments: '{"target":"runpod","detailed":true}',
    },
  };

  assert.deepEqual(buildChatCompletionToolResultMessage(toolCall, {
    ok: true,
    status: 'healthy',
  }), {
    role: 'tool',
    tool_call_id: 'call_1',
    name: 'runtime_health_check',
    content: '{"ok":true,"status":"healthy"}',
  });
});

test('buildTelegramChatCompletionRequest forwards only the carried transcript and token cap', () => {
  const transcript = [
    { role: 'user', content: 'Earlier user turn.' },
    { role: 'assistant', content: 'Earlier assistant turn.' },
    { role: 'user', content: 'Check the runtime please.' },
  ];

  const payload = buildTelegramChatCompletionRequest({
    model: 'deepseek-reasoner',
    transcript,
    maxTokens: 4096,
  });

  assert.deepEqual(payload, {
    model: 'deepseek-reasoner',
    temperature: 0.2,
    messages: transcript,
    max_tokens: 4096,
  });
  assert.equal('tools' in payload, false);
});

test('buildTelegramBridgeRequestTimeoutMs defaults to a substantially longer request lifetime', () => {
  assert.equal(buildTelegramBridgeRequestTimeoutMs(undefined), 1800000);
  assert.equal(buildTelegramBridgeRequestTimeoutMs('3600000'), 3600000);
  assert.equal(buildTelegramBridgeRequestTimeoutMs('25'), 1000);
});

test('shouldSuppressDeferredTelegramFailureMessage suppresses misleading deferred transport failures once runtime accepted the request', () => {
  assert.equal(shouldSuppressDeferredTelegramFailureMessage({
    classification: 'transport',
    latestTraceEvent: 'relay_started',
  }), true);
  assert.equal(shouldSuppressDeferredTelegramFailureMessage({
    classification: 'timeout',
    latestTraceEvent: 'request_completed',
  }), true);
  assert.equal(shouldSuppressDeferredTelegramFailureMessage({
    classification: 'runtime',
    latestTraceEvent: 'relay_started',
  }), false);
  assert.equal(shouldSuppressDeferredTelegramFailureMessage({
    classification: 'transport',
    latestTraceEvent: 'request_failed',
  }), false);
});

test('bridge polling constants are tightened for lower tail latency', () => {
  assert.equal(TELEGRAM_BRIDGE_ASK_OUTBOX_POLL_IDLE_MS, 200);
  assert.equal(TELEGRAM_BRIDGE_SELF_EMIT_ACTIVE_DELAY_MS, 250);
  assert.equal(TELEGRAM_BRIDGE_SELF_EMIT_ERROR_DELAY_MS, 250);
  assert.equal(TELEGRAM_BRIDGE_WATCHDOG_DELAY_MS, 1000);
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

test('sanitizeAssistantRelayText strips DSML blocks even when they run to end-of-string', () => {
  const text = sanitizeAssistantRelayText(
    'prefix\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="web_search">\n<｜DSML｜parameter name="query" string="true">vegan restaurants</｜DSML｜parameter>\n</｜DSML｜invoke>',
  );

  assert.equal(text, 'prefix');
});

test('detectTelegramDocumentLogicalType supports Docling-backed formats and rejects legacy doc', () => {
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'application/pdf',
    file_name: 'report.bin',
  }), 'pdf');
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'application/octet-stream',
    file_name: 'report.docx',
  }), 'docx');
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'text/markdown',
    file_name: 'notes.md',
  }), 'markdown');
  assert.equal(detectTelegramDocumentLogicalType({
    mime_type: 'application/msword',
    file_name: 'report.doc',
  }), 'unsupported');
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
  }, {
    containerTag: 'shared-documents',
  });

  assert.match(linkage.linkKey, /^telegram-doc-/);
  assert.equal(linkage.containerTag, 'shared-documents');
  assert.equal(linkage.rawMetadata.linkKey, linkage.parsedMetadata.linkKey);
  assert.equal(linkage.rawMetadata.contentKind, 'source_file');
  assert.equal(linkage.parsedMetadata.contentKind, 'parsed_output');
  assert.match(linkage.chunkKeyPrefix, /^telegram-chunk-/);
});

test('buildTelegramFileUrl creates telegram file endpoint', () => {
  assert.equal(
    buildTelegramFileUrl('token123', '/documents/file.pdf'),
    'https://api.telegram.org/file/bottoken123/documents/file.pdf',
  );
});

test('hasQueuedTelegramDelivery detects runtime-owned Telegram delivery metadata', () => {
  assert.equal(hasQueuedTelegramDelivery({
    vicuna_telegram_delivery: {
      handled: true,
      queued: true,
    },
  }), true);
  assert.equal(hasQueuedTelegramDelivery({
    vicuna_telegram_delivery: {
      handled: false,
    },
  }), false);
  assert.equal(hasQueuedTelegramDelivery({}), false);
});

test('normalizeDoclingParsedDocument keeps parsed markdown and contextualized chunks', () => {
  const parsed = normalizeDoclingParsedDocument({
    title: 'report.pdf',
    parsed_markdown: '# Heading\n\nAlpha',
    plain_text: 'Heading\n\nAlpha',
    docling_version: '2.0.0',
    chunks: [
      {
        chunk_index: 1,
        source_text: 'Alpha',
        contextual_text: 'Heading\nAlpha',
      },
      {
        chunk_index: 0,
        source_text: 'Heading',
        contextual_text: 'Heading',
      },
    ],
  }, {
    fileName: 'report.pdf',
  }, {
    maxChars: 50,
    maxChunks: 8,
  });

  assert.equal(parsed.title, 'report.pdf');
  assert.equal(parsed.parsedMarkdown, '# Heading\n\nAlpha');
  assert.equal(parsed.transcript.text, 'Heading\n\nAlpha');
  assert.equal(parsed.chunks.length, 2);
  assert.equal(parsed.chunks[0].chunkIndex, 0);
  assert.equal(parsed.chunks[1].contextualText, 'Heading\nAlpha');
});

test('buildTelegramParsedChunkMemories preserves searchable parsed chunk metadata', () => {
  const linkage = buildTelegramDocumentLinkage({
    chatId: '42',
    messageId: '9',
    fileId: 'file_1',
    fileUniqueId: 'uniq_1',
    fileName: 'memo.docx',
    mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    logicalType: 'docx',
  });
  const memories = buildTelegramParsedChunkMemories({
    descriptor: {
      chatId: '42',
      messageId: '9',
      fileName: 'memo.docx',
      logicalType: 'docx',
    },
    linkage,
    parsedDocument: {
      title: 'memo.docx',
      doclingVersion: '2.0.0',
      chunks: [
        { chunkIndex: 0, contextualText: 'Intro chunk', sourceText: 'Intro' },
      ],
    },
  });

  assert.equal(memories.length, 1);
  assert.equal(memories[0].content, 'Intro chunk');
  assert.equal(memories[0].metadata.contentKind, 'parsed_chunk');
  assert.equal(memories[0].metadata.documentTitle, 'memo.docx');
  assert.equal(memories[0].metadata.linkKey, linkage.linkKey);
});

test('ingestTelegramDocumentMessage persists local source and parsed bundles with shared linkage', async () => {
  const docsRoot = await mkdtemp(path.join(os.tmpdir(), 'vicuna-doc-bundle-'));
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
    parseDocument: async () => ({
      title: 'report.pdf',
      parsed_markdown: '# Report\n\nAlpha\n\nBeta',
      plain_text: 'Alpha\n\nBeta',
      docling_version: '2.0.0',
      chunks: [
        {
          chunk_index: 0,
          source_text: 'Alpha',
          contextual_text: 'Report\nAlpha',
        },
        {
          chunk_index: 1,
          source_text: 'Beta',
          contextual_text: 'Report\nBeta',
        },
      ],
    }),
    docsRoot,
    documentContainerTag: 'shared-documents',
  });

  assert.equal(result.ok, true);
  assert.match(result.bundleId, /^telegram-doc-/);
  assert.equal(result.storedChunkCount, 2);
  assert.match(result.transcriptText, /Parsed contents of report\.pdf/);
  assert.match(result.transcriptText, /Alpha\n\nBeta/);
  const metadata = JSON.parse(await readFile(path.join(docsRoot, result.bundleId, 'metadata.json'), 'utf8'));
  const chunks = JSON.parse(await readFile(path.join(docsRoot, result.bundleId, 'chunks.json'), 'utf8'));
  assert.equal(metadata.parse_status, 'parsed');
  assert.equal(metadata.source_path, result.sourcePath);
  assert.equal(chunks[0].document_title, 'report.pdf');
  assert.equal(chunks[0].link_key, result.bundleId);
  await rm(docsRoot, { recursive: true, force: true });
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
  });

  assert.equal(result.ok, false);
  assert.equal(result.userError, 'Only Docling-supported document uploads such as PDF and DOCX are supported right now.');
});

test('ingestTelegramDocumentMessage requires a local docs root for document processing', async () => {
  await assert.rejects(
    ingestTelegramDocumentMessage({
      message: {
        chat: { id: 77 },
        message_id: 10,
        document: {
          file_id: 'file_1',
          file_name: 'report.pdf',
          mime_type: 'application/pdf',
        },
      },
      resolveTelegramFile: async () => ({ file_path: 'docs/report.pdf' }),
      downloadTelegramFile: async () => Buffer.from('pdf bytes'),
      parseDocument: async () => ({
        title: 'report.pdf',
        parsed_markdown: '# Report\n\nAlpha',
        plain_text: 'Alpha',
        chunks: [{ chunk_index: 0, source_text: 'Alpha', contextual_text: 'Report\nAlpha' }],
      }),
    }),
    /local docs root/,
  );
});

test('ingestTelegramDocumentMessage preserves the source file when Docling parsing fails', async () => {
  const docsRoot = await mkdtemp(path.join(os.tmpdir(), 'vicuna-doc-fail-'));
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
    parseDocument: async () => {
      throw new Error('Docling is not available in the configured Python environment.');
    },
    docsRoot,
  });

  assert.equal(result.ok, false);
  assert.equal(result.partialFailure, true);
  assert.match(result.userError, /local source storage/);
  assert.match(result.userError, /Docling/);
  const metadata = JSON.parse(await readFile(path.join(docsRoot, result.bundleId, 'metadata.json'), 'utf8'));
  assert.equal(metadata.parse_status, 'parse_failed');
  assert.equal(typeof result.sourcePath, 'string');
  await rm(docsRoot, { recursive: true, force: true });
});

test('ingestTelegramDocumentMessage rejects parsed outputs with no searchable chunks', async () => {
  const docsRoot = await mkdtemp(path.join(os.tmpdir(), 'vicuna-doc-empty-'));
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
    parseDocument: async () => ({
      title: 'report.pdf',
      parsed_markdown: '# Report\n\nAlpha',
      plain_text: 'Alpha',
      chunks: [],
    }),
    docsRoot,
  });

  assert.equal(result.ok, false);
  assert.equal(result.partialFailure, true);
  assert.match(result.userError, /no searchable chunks/);
  await rm(docsRoot, { recursive: true, force: true });
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
    animation: {
      requested: true,
      status: 'sent',
      stage: 'complete',
      keyframeCount: 2,
      durationSeconds: 1,
      telegramMessageId: 103,
    },
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
    animation: {
      requested: true,
      status: 'sent',
      stage: 'complete',
      keyframeCount: 2,
      durationSeconds: 1,
      telegramMessageId: 103,
    },
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

test('recordTelegramOutboxDeliveryReceipt preserves chunked delivery metadata', () => {
  const state = recordTelegramOutboxDeliveryReceipt(normalizeState({}), {
    sequenceNumber: 22,
    chatId: '7502424413',
    replyToMessageId: 592,
    deliveryMode: 'chunked_reply',
    telegramMessageId: 105,
    telegramMessageIds: [103, 104, 105],
    chunkCount: 3,
    deliveredAtMs: 123999,
  });

  assert.deepEqual(state.telegramOutboxDeliveryReceipt, {
    sequenceNumber: 22,
    chatId: '7502424413',
    replyToMessageId: 592,
    deliveryMode: 'chunked_reply',
    telegramMessageId: 105,
    telegramMessageIds: [103, 104, 105],
    chunkCount: 3,
    deliveredAtMs: 123999,
  });
});

test('pending video deliveries persist bounded retry metadata', () => {
  let state = normalizeState({});
  state = enqueueTelegramPendingVideoDelivery(state, {
    jobId: 'tv_1',
    sequenceNumber: 22,
    chatId: '7502424413',
    conversationId: 'tc1',
    replyToMessageId: 608,
    textTelegramMessageId: 610,
    deliveryMode: 'chunked_reply',
    telegramMessageIds: [608, 609, 610],
    chunkCount: 3,
    videoPayload: { has_spoiler: true, supports_streaming: true },
    bundle: sampleEmotiveAnimationBundle(),
    attemptCount: 1,
    stage: 'queued',
    nextAttemptAtMs: 123000,
    createdAtMs: 122000,
    updatedAtMs: 122500,
  });

  assert.deepEqual(getTelegramPendingVideoDelivery(state, 'tv_1'), {
    jobId: 'tv_1',
    sequenceNumber: 22,
    chatId: '7502424413',
    conversationId: 'tc1',
    replyToMessageId: 608,
    textTelegramMessageId: 610,
    deliveryMode: 'chunked_reply',
    telegramMessageIds: [608, 609, 610],
    chunkCount: 3,
    videoPayload: { has_spoiler: true, supports_streaming: true },
    bundle: sampleEmotiveAnimationBundle(),
    attemptCount: 1,
    stage: 'queued',
    nextAttemptAtMs: 123000,
    createdAtMs: 122000,
    updatedAtMs: 122500,
  });

  state = updateTelegramPendingVideoDelivery(state, 'tv_1', {
    attemptCount: 2,
    nextAttemptAtMs: 124000,
    lastError: 'fetch failed',
    updatedAtMs: 123500,
  });
  assert.equal(getTelegramPendingVideoDelivery(state, 'tv_1').attemptCount, 2);
  assert.equal(getTelegramPendingVideoDelivery(state, 'tv_1').lastError, 'fetch failed');

  state = deleteTelegramPendingVideoDelivery(state, 'tv_1');
  assert.equal(getTelegramPendingVideoDelivery(state, 'tv_1'), null);
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
    telegram_method: 'sendMessage',
    telegram_payload: {
      text: 'done',
    },
    reply_to_message_id: 91,
  }), {
    ok: true,
    skippable: true,
    sequenceNumber: 12,
    kind: 'message',
    chatId: '7502424413',
    text: 'done',
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: 'done',
    },
    replyToMessageId: 91,
    emotiveAnimation: null,
  });
});

test('normalizeTelegramOutboxItem derives structured message summaries and preserves payloads', () => {
  assert.deepEqual(normalizeTelegramOutboxItem({
    sequence_number: 15,
    kind: 'message',
    chat_scope: '7502424413',
    telegram_method: 'sendPhoto',
    telegram_payload: {
      photo: 'https://example.com/report.png',
      caption: '<b>Ready</b>',
      parse_mode: 'HTML',
    },
  }), {
    ok: true,
    skippable: true,
    sequenceNumber: 15,
    kind: 'message',
    chatId: '7502424413',
    text: '<b>Ready</b>',
    telegramMethod: 'sendPhoto',
    telegramPayload: {
      photo: 'https://example.com/report.png',
      caption: '<b>Ready</b>',
      parse_mode: 'HTML',
    },
    replyToMessageId: 0,
    emotiveAnimation: null,
  });
});

test('normalizeTelegramOutboxItem preserves emotive animation bundles for message follow-ups', () => {
  const item = normalizeTelegramOutboxItem({
    sequence_number: 33,
    kind: 'message',
    chat_scope: '7502424413',
    text: 'done',
    telegram_method: 'sendMessage',
    telegram_payload: {
      text: 'done',
    },
    emotive_animation: sampleEmotiveAnimationBundle(),
  });

  assert.equal(item.ok, true);
  assert.equal(item.emotiveAnimation.trace_id, 'emo_test');
  assert.equal(item.emotiveAnimation.keyframes.length, 2);
});

test('buildTelegramSpoilerVideoPayload derives a spoilered sendVideo caption payload', () => {
  assert.deepEqual(buildTelegramSpoilerVideoPayload({
    telegramMethod: 'sendMessage',
    text: '<b>Hello!</b>',
    telegramPayload: {
      text: '<b>Hello!</b>',
      parse_mode: 'HTML',
      reply_markup: {
        inline_keyboard: [[{ text: 'Open', url: 'https://example.com' }]],
      },
      disable_notification: true,
      disable_web_page_preview: true,
    },
  }), {
    ok: true,
    method: 'sendVideo',
    captionLength: 6,
    payload: {
      caption: '<b>Hello!</b>',
      parse_mode: 'HTML',
      reply_markup: {
        inline_keyboard: [[{ text: 'Open', url: 'https://example.com' }]],
      },
      disable_notification: true,
      show_caption_above_media: true,
      has_spoiler: true,
      supports_streaming: true,
    },
  });
});

test('buildTelegramEmotiveAnimationBundle preserves full-turn timing from raw trace blocks', () => {
  const bundle = buildTelegramEmotiveAnimationBundle(sampleRawEmotiveTrace());

  assert.equal(bundle.trace_id, 'emo_trace_raw');
  assert.equal(bundle.generation_start_block_index, 1);
  assert.equal(bundle.turn_start_block_index, 1);
  assert.equal(bundle.live_generation_start_block_index, 2);
  assert.equal(bundle.raw_keyframe_count, 3);
  assert.equal(bundle.distinct_keyframe_count, 2);
  assert.equal(bundle.keyframes.length, 2);
  assert.equal(bundle.keyframes[0].hold_keyframe_count, 2);
  assert.equal(bundle.keyframes[0].start_offset_ms, 0);
  assert.equal(bundle.keyframes[0].end_offset_ms, 200);
  assert.deepEqual(bundle.keyframes[0].trace_block_span, [1, 2]);
  assert.deepEqual(bundle.keyframes[0].dominant_dimensions, ['confidence', 'caution']);
  assert.equal(bundle.keyframes[1].start_offset_ms, 800);
  assert.equal(bundle.segments.length, 1);
  assert.equal(bundle.segments[0].start_offset_ms, 200);
  assert.equal(bundle.segments[0].duration_ms, 600);
  assert.equal(bundle.duration_seconds, 0.8);
});

test('buildTelegramEmotiveAnimationBundle clamps mixed timestamp epochs to monotonic offsets', () => {
  const trace = sampleRawEmotiveTrace();
  trace.blocks[1].timestamp_ms = 1775157640000;
  trace.blocks[2].timestamp_ms = 3169201721;
  trace.blocks[3].timestamp_ms = 1775157640762;

  const bundle = buildTelegramEmotiveAnimationBundle(trace);

  assert.equal(bundle.keyframes[0].start_offset_ms, 0);
  assert.equal(bundle.keyframes[0].end_offset_ms, 0);
  assert.equal(bundle.keyframes[1].start_offset_ms, 500);
  assert.equal(bundle.duration_seconds, 0.5);
});

test('normalizeTelegramRichTextPayload converts markdownish sendMessage text into Telegram HTML', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: '### Status\n\n**Ready**\n\n---\n\n> steady\n\n`code`',
    },
  });

  assert.equal(normalized.normalized, true);
  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(
    normalized.payload.text,
    '<b>Status</b>\n\n<b>Ready</b>\n\n━━━━━━━━━━━━\n\n<blockquote>steady</blockquote>\n\n<code>code</code>',
  );
});

test('normalizeTelegramRichTextPayload promotes provider-authored Telegram HTML to parse_mode HTML', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: '✅ <b>Added</b> <i>successfully</i>',
    },
  });

  assert.equal(normalized.normalized, true);
  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(normalized.payload.text, '✅ <b>Added</b> <i>successfully</i>');
  assert.equal(normalized.textLength, '✅ Added successfully'.length);
});

test('normalizeTelegramRichTextPayload canonicalizes htmlish paragraphs and headings into Telegram HTML', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: '<h3>Status</h3><p><strong>Ready</strong></p><p>Next<br>Step</p>',
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(
    normalized.payload.text,
    '<b>Status</b>\n\n<b>Ready</b>\n\nNext\nStep',
  );
  assert.equal(normalized.textLength, 'Status\n\nReady\n\nNext\nStep'.length);
});

test('normalizeTelegramRichTextPayload canonicalizes markdown parse_mode into HTML', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      parse_mode: 'MarkdownV2',
      text: '### Status\n\n**Ready**',
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(normalized.payload.text, '<b>Status</b>\n\n<b>Ready</b>');
  assert.equal(normalized.textLength, 'Status\n\nReady'.length);
});

test('normalizeTelegramRichTextPayload canonicalizes captions for media methods', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendPhoto',
    telegramPayload: {
      caption: '<p><b>Ready</b> &amp; steady</p>',
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(normalized.payload.caption, '<b>Ready</b> &amp; steady');
  assert.equal(normalized.textLength, 'Ready & steady'.length);
});

test('normalizeTelegramRichTextPayload rewrites markdown pipe tables into rich text sections', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: '| Name | Score |\n| --- | --- |\n| Calm | 0.42 |\n| Trust | 0.80 |',
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.match(normalized.payload.text, /<b>Name:<\/b> Calm/);
  assert.match(normalized.payload.text, /<b>Score:<\/b> 0\.42/);
  assert.ok(!normalized.payload.text.includes('| --- | --- |'));
});

test('normalizeTelegramRichTextPayload flattens HTML tables into rich text sections', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: [
        '<table>',
        '<thead><tr><th>Name</th><th>Score</th><th>Notes</th></tr></thead>',
        '<tbody><tr><td>Calm</td><td>0.42</td><td><b>Readable</b> narrative output</td></tr></tbody>',
        '</table>',
      ].join(''),
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.match(normalized.payload.text, /<b>Name:<\/b> Calm/);
  assert.match(normalized.payload.text, /<b>Score:<\/b> 0\.42/);
  assert.match(normalized.payload.text, /<b>Notes:<\/b> Readable narrative output/);
  assert.ok(!normalized.payload.text.includes('<table>'));
  assert.ok(!normalized.payload.text.includes('<tr>'));
});

test('normalizeTelegramRichTextPayload balances dangling provider-authored html tags', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      parse_mode: 'HTML',
      text: '<b>Status update\n\nStill working',
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(normalized.payload.text, '<b>Status update\n\nStill working</b>');
  assertBalancedTelegramSupportedTags(normalized.payload.text);
});

test('normalizeTelegramRichTextPayload repairs mismatched nested html tags', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      parse_mode: 'HTML',
      text: '<b>Bold <i>italic</b> tail',
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(normalized.payload.text, '<b>Bold <i>italic</i></b> tail');
  assertBalancedTelegramSupportedTags(normalized.payload.text);
});

test('buildTelegramPlainTextFallbackPayload strips telegram html and parse mode', () => {
  const fallback = buildTelegramPlainTextFallbackPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: '<b>Bold</b> and <i>italic',
      parse_mode: 'HTML',
      reply_parameters: { message_id: 42 },
    },
  });

  assert.equal(fallback.payload.text, 'Bold and italic');
  assert.equal(Object.hasOwn(fallback.payload, 'parse_mode'), false);
  assert.deepEqual(fallback.payload.reply_parameters, { message_id: 42 });
});

test('normalizeTelegramRichTextPayload strips DSML markup before canonicalizing HTML', () => {
  const normalized = normalizeTelegramRichTextPayload({
    telegramMethod: 'sendMessage',
    telegramPayload: {
      text: '### Status\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="web_search">\n<｜DSML｜parameter name="query" string="true">vegan</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>\n\n**Ready**',
    },
  });

  assert.equal(normalized.payload.parse_mode, 'HTML');
  assert.equal(normalized.payload.text, '<b>Status</b>\n\n<b>Ready</b>');
});

test('buildTelegramSpoilerVideoPayload maps entities and rejects captions over Telegram limits', () => {
  const withEntities = buildTelegramSpoilerVideoPayload({
    telegramMethod: 'sendMessage',
    text: 'Hello',
    telegramPayload: {
      text: 'Hello',
      entities: [{ type: 'bold', offset: 0, length: 5 }],
    },
  });
  assert.equal(withEntities.ok, true);
  assert.deepEqual(withEntities.payload.caption_entities, [{ type: 'bold', offset: 0, length: 5 }]);
  assert.equal(Object.prototype.hasOwnProperty.call(withEntities.payload, 'parse_mode'), false);

  assert.deepEqual(buildTelegramSpoilerVideoPayload({
    telegramMethod: 'sendMessage',
    text: 'x'.repeat(TELEGRAM_VIDEO_CAPTION_LIMIT + 1),
    telegramPayload: {
      text: 'x'.repeat(TELEGRAM_VIDEO_CAPTION_LIMIT + 1),
    },
  }), {
    ok: false,
    reason: 'caption_too_long',
    method: 'sendMessage',
    captionLength: TELEGRAM_VIDEO_CAPTION_LIMIT + 1,
    captionLimit: TELEGRAM_VIDEO_CAPTION_LIMIT,
  });
});

test('buildTelegramSpoilerVideoPayload counts visible text length after markdown normalization', () => {
  const payload = buildTelegramSpoilerVideoPayload({
    telegramMethod: 'sendMessage',
    text: '**Bold**',
    telegramPayload: {
      text: '**Bold**',
    },
  });

  assert.equal(payload.ok, true);
  assert.equal(payload.captionLength, 4);
  assert.equal(payload.payload.caption, '<b>Bold</b>');
  assert.equal(payload.payload.parse_mode, 'HTML');
});

test('buildTelegramSpoilerVideoPayload preserves provider-authored HTML captions', () => {
  const payload = buildTelegramSpoilerVideoPayload({
    telegramMethod: 'sendMessage',
    text: '✅ <b>Added</b> <i>successfully</i>',
    telegramPayload: {
      text: '✅ <b>Added</b> <i>successfully</i>',
    },
  });

  assert.equal(payload.ok, true);
  assert.equal(payload.captionLength, '✅ Added successfully'.length);
  assert.equal(payload.payload.caption, '✅ <b>Added</b> <i>successfully</i>');
  assert.equal(payload.payload.parse_mode, 'HTML');
});

test('buildTelegramAnimationDeliveryPlan always uses split text and standalone video', () => {
  const sourceText = '### Status\n\nShort reply';
  const plan = buildTelegramAnimationDeliveryPlan({
    telegramMethod: 'sendMessage',
    text: sourceText,
    telegramPayload: {
      text: sourceText,
      disable_notification: true,
      reply_markup: {
        inline_keyboard: [[{ text: 'Open', url: 'https://example.com' }]],
      },
    },
  });

  assert.equal(plan.ok, true);
  assert.equal(plan.strategy, 'split_text_and_video');
  assert.equal(plan.messagePayload.parse_mode, 'HTML');
  assert.equal(plan.messagePayload.disable_notification, true);
  assert.deepEqual(plan.messagePayload.reply_markup, {
    inline_keyboard: [[{ text: 'Open', url: 'https://example.com' }]],
  });
  assert.deepEqual(plan.videoPayload, {
    has_spoiler: true,
    supports_streaming: true,
    disable_notification: true,
  });
  assert.equal(plan.captionLength, 0);
  assert.equal(plan.captionLimit, TELEGRAM_VIDEO_CAPTION_LIMIT);
});

test('buildTelegramMessageChunkPlan splits long rich text into Telegram-safe chunks', () => {
  const paragraph = Array.from(
    { length: 40 },
    (_, index) => `### Finding ${index + 1}\n\nThis supporting explanation stays detailed enough to force chunking while preserving readable markdown structure, explicit source bullets, careful operator-facing delivery verification, and a long enough paragraph body to exceed Telegram single-message limits once repeated across many findings in a single answer.`,
  ).join('\n\n');
  const sourceText = `${paragraph}\n\nFinal operator note.`;
  const plan = buildTelegramMessageChunkPlan({
    telegramPayload: {
      text: sourceText,
    },
    text: sourceText,
  });

  assert.ok(plan.chunkCount > 1);
  for (const chunk of plan.chunks) {
    assert.ok(chunk.textLength <= TELEGRAM_MESSAGE_TEXT_LIMIT);
    assert.ok(chunk.renderedLength <= TELEGRAM_MESSAGE_SAFE_RENDERED_LIMIT);
    assert.equal(chunk.payload.parse_mode, 'HTML');
    assert.ok(String(chunk.payload.text ?? '').trim().length > 0);
  }
});

test('buildTelegramMessageChunkPlan strips raw markdown table separators from chunked delivery', () => {
  const row = '| Calm | 0.42 | This explanation is intentionally long so the overall answer forces Telegram chunking when the table is rewritten into supported rich text. |';
  const sourceText = `| Name | Score | Notes |\n| --- | --- | --- |\n${Array.from({ length: 24 }, () => row).join('\n')}`;
  const plan = buildTelegramMessageChunkPlan({
    telegramPayload: {
      text: sourceText,
    },
    text: sourceText,
  });

  assert.ok(plan.chunkCount > 1);
  for (const chunk of plan.chunks) {
    const rendered = String(chunk.payload.text ?? '');
    assert.ok(!rendered.includes('| --- | --- | --- |'));
    assert.ok(!rendered.includes('| Calm | 0.42 |'));
  }
});

test('buildTelegramMessageChunkPlan strips HTML table markup from chunked delivery', () => {
  const row = '<tr><td>Calm</td><td>0.42</td><td>This explanation is intentionally long so the overall answer forces Telegram chunking when the HTML table is flattened into supported rich text.</td></tr>';
  const sourceText = `<table><thead><tr><th>Name</th><th>Score</th><th>Notes</th></tr></thead><tbody>${Array.from({ length: 24 }, () => row).join('')}</tbody></table>`;
  const plan = buildTelegramMessageChunkPlan({
    telegramPayload: {
      text: sourceText,
      parse_mode: 'HTML',
    },
    text: sourceText,
  });

  assert.ok(plan.chunkCount > 1);
  for (const chunk of plan.chunks) {
    const rendered = String(chunk.payload.text ?? '');
    assert.ok(!rendered.includes('<table>'));
    assert.ok(!rendered.includes('<tr>'));
    assert.ok(!rendered.includes('<td>'));
    assert.match(rendered, /<b>Name:<\/b> Calm/);
  }
});

test('buildTelegramMessageChunkPlan balances malformed html across chunks', () => {
  const sourceText = `<b>${'Detailed literature note. '.repeat(300)}`;
  const plan = buildTelegramMessageChunkPlan({
    telegramPayload: {
      text: sourceText,
      parse_mode: 'HTML',
    },
    text: sourceText,
  });

  assert.ok(plan.chunkCount > 1);
  for (const chunk of plan.chunks) {
    assertBalancedTelegramSupportedTags(chunk.payload.text);
    assert.equal(chunk.payload.parse_mode, 'HTML');
  }
});

test('webgl renderer page keeps the saturated heatmap shader controls', async () => {
  const page = await readFile(
    new URL('./emotive-webgl-renderer-page.html', import.meta.url),
    'utf8',
  );

  assert.match(page, /uSaturationBoost:\s*\{\s*value:\s*2\.3\s*\}/);
  assert.match(page, /vec3 c0 = vec3\(0\.82,\s*0\.00,\s*1\.00\)/);
  assert.match(page, /vec3 c5 = vec3\(1\.00,\s*0\.00,\s*0\.00\)/);
  assert.match(page, /renderer\.toneMappingExposure = 0\.70/);
});

test('normalizeEmotiveAnimationBundle and render plan keep timing explicit', () => {
  const bundle = normalizeEmotiveAnimationBundle(sampleEmotiveAnimationBundle());
  const plan = buildEmotiveAnimationRenderPlan(sampleEmotiveAnimationBundle());

  assert.equal(bundle.traceId, 'emo_test');
  assert.equal(bundle.dimensions.length, 4);
  assert.equal(bundle.keyframes.length, 2);
  assert.equal(bundle.keyframes[0].holdKeyframeCount, 2);
  assert.equal(bundle.keyframes[0].startOffsetMs, 0);
  assert.equal(bundle.keyframes[0].endOffsetMs, 350);
  assert.equal(bundle.rawKeyframeCount, 4);
  assert.equal(bundle.distinctKeyframeCount, 2);
  assert.equal(bundle.segments.length, 1);
  assert.equal(bundle.durationSeconds, 1.8);
  assert.equal(plan.keyframeCount, 2);
  assert.equal(plan.rawKeyframeCount, 4);
  assert.equal(plan.totalFrames, 54);
  assert.equal(plan.durationSeconds, 1.8);
});

test('computeBundleRadiusNormalizationRange anchors the heatmap to the explicit support range', () => {
  const range = computeBundleRadiusNormalizationRange(sampleEmotiveAnimationBundle());

  assert.equal(range.minRadius, ANCHOR_MIN_RADIUS);
  assert.equal(range.maxRadius, ANCHOR_MAX_RADIUS);
  assert.equal(range.maxMomentValue, 0.85);
});

test('outer hull shoulder moat profile remains broad across the solved topology', () => {
  const bundle = normalizeEmotiveAnimationBundle(sampleEmotiveAnimationBundle());
  const topology = buildOuterHullTopology(bundle);

  for (let anchorIndex = 0; anchorIndex < bundle.dimensions.length; anchorIndex += 1) {
    const shoulderCoverage = topology.vertexBindings.filter(
      (binding) => binding.shoulderProfile[anchorIndex] > 0.1,
    ).length;
    const moatCoverage = topology.vertexBindings.filter(
      (binding) => binding.shoulderRingProfile[anchorIndex] > 0.1,
    ).length;
    const moatMin = Math.min(...topology.vertexBindings.map(
      (binding) => binding.shoulderRingProfile[anchorIndex],
    ));

    assert.ok(shoulderCoverage > 0);
    assert.ok(moatCoverage / shoulderCoverage > 0.9);
    assert.ok(moatMin > 0.05);
  }
});

test('low-signal bundles still solve to a visible membrane above the minimum hull radius', () => {
  const bundle = normalizeEmotiveAnimationBundle(lowSignalEmotiveAnimationBundle());
  const topology = buildOuterHullTopology(bundle);
  const { worldVertices } = buildFrameGeometry(bundle, topology, 0);
  const radii = worldVertices.map((vertex) => vertex.length());

  assert.ok(Math.min(...radii) >= ANCHOR_MIN_RADIUS);
  assert.ok(Math.max(...radii) > 1.05);
  assert.ok(Math.max(...radii) - Math.min(...radii) > 0.35);
});

test('strong register values stay materially expressed in the solved membrane and disturb the full topology', () => {
  const bundle = normalizeEmotiveAnimationBundle(sampleEmotiveAnimationBundle());
  const topology = buildOuterHullTopology(bundle);
  const neverAboveFloor = new Array(topology.vertexCount).fill(true);
  let globalMax = -Infinity;

  for (let frameIndex = 0; frameIndex < bundle.totalFrames; frameIndex += 1) {
    const { worldVertices } = buildFrameGeometry(bundle, topology, frameIndex);
    for (let vertexIndex = 0; vertexIndex < worldVertices.length; vertexIndex += 1) {
      const radius = worldVertices[vertexIndex].length();
      globalMax = Math.max(globalMax, radius);
      if (radius > (ANCHOR_MIN_RADIUS + 0.01)) {
        neverAboveFloor[vertexIndex] = false;
      }
    }
  }

  assert.ok(globalMax > 1.12);
  assert.ok(neverAboveFloor.filter(Boolean).length <= 4);
});

test('prependEmotiveAnimationStartMoment seeds the next clip from the prior terminal pose', () => {
  const priorMoment = {
    confidence: 0.91,
    caution: 0.12,
    curiosity: 0.66,
    satisfaction: 0.73,
  };
  const bundle = prependEmotiveAnimationStartMoment(sampleEmotiveAnimationBundle(), priorMoment);

  assert.equal(bundle.keyframes.length, 3);
  assert.equal(bundle.rawKeyframeCount, 5);
  assert.equal(bundle.keyframes[0].sourceKind, 'prior_delivery_terminal');
  assert.equal(bundle.keyframes[0].holdKeyframeCount, 1);
  assert.equal(bundle.keyframes[0].startOffsetMs, 0);
  assert.equal(bundle.keyframes[0].endOffsetMs, 0);
  assert.deepEqual(bundle.keyframes[0].moment, priorMoment);
  assert.equal(bundle.durationSeconds, 1.8);
  assert.equal(bundle.totalFrames, 54);
  assert.equal(bundle.segments.length, 2);
  assert.equal(bundle.keyframes[1].ordinal, 1);
});

test('telegram emotive animation state persists the last delivered terminal moment per conversation', () => {
  let state = setTelegramEmotiveAnimationState({}, 'chat-1', {
    lastMoment: {
      confidence: 0.4,
      caution: 0.8,
    },
    lastRenderedAtMs: 1234,
  }, {
    conversationId: 'tc1',
  });

  assert.deepEqual(getTelegramEmotiveAnimationState(state, 'chat-1', { conversationId: 'tc1' }), {
    chatId: 'chat-1',
    conversationId: 'tc1',
    lastMoment: {
      confidence: 0.4,
      caution: 0.8,
    },
    lastRenderedAtMs: 1234,
  });

  state = setTelegramEmotiveAnimationState(state, 'chat-1', null, {
    conversationId: 'tc1',
  });
  assert.equal(getTelegramEmotiveAnimationState(state, 'chat-1', { conversationId: 'tc1' }), null);
});

test('extractEmotiveAnimationTerminalMoment returns the final normalized keyframe moment', () => {
  assert.deepEqual(extractEmotiveAnimationTerminalMoment(sampleEmotiveAnimationBundle()), {
    confidence: 0.85,
    caution: 0.25,
    curiosity: 0.65,
    satisfaction: 0.75,
  });
});

test('renderEmotiveAnimationFrames writes one PNG per planned frame', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'vicuna-emotive-render-test-'));
  try {
    const result = await renderEmotiveAnimationFrames(sampleEmotiveAnimationBundle(), dir);
    const files = await readdir(dir);

    assert.equal(result.totalFrames, 54);
    assert.equal(files.length, 54);
    assert.equal(files[0], 'frame-0000.png');
    assert.equal(files.at(-1), 'frame-0053.png');
    assert.equal(result.topology.anchorCount, 4);
    assert.ok(result.topology.vertexCount > 700);
    assert.ok(result.topology.triangleCount > 1500);
    assert.equal(result.scene.background, '#000000');
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
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

test('renderEmotiveAnimationViaWebglService posts a render job and returns the service payload', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async (url, init) => {
      assert.equal(String(url), 'http://127.0.0.1:8091/render');
      assert.equal(init?.method, 'POST');
      const parsed = JSON.parse(String(init?.body));
      assert.equal(parsed.outputPath, '/tmp/out.mp4');
      assert.equal(parsed.bundle.bundle_version, 3);
      return {
        ok: true,
        async json() {
          return {
            outputPath: '/tmp/out.mp4',
            backend: 'chromium_webgl',
            keyframeCount: 2,
          };
        },
      };
    };

    const result = await renderEmotiveAnimationViaWebglService(sampleEmotiveAnimationBundle(), {
      serviceUrl: 'http://127.0.0.1:8091',
      outputPath: '/tmp/out.mp4',
    });
    assert.equal(result.outputPath, '/tmp/out.mp4');
    assert.equal(result.backend, 'chromium_webgl');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('renderEmotiveAnimationViaWebglService retries one transient transport failure', async () => {
  const originalFetch = globalThis.fetch;
  let attempts = 0;
  try {
    globalThis.fetch = async () => {
      attempts += 1;
      if (attempts === 1) {
        throw new Error('fetch failed');
      }
      return {
        ok: true,
        async json() {
          return {
            outputPath: '/tmp/out.mp4',
            backend: 'chromium_webgl',
            keyframeCount: 2,
          };
        },
      };
    };

    const result = await renderEmotiveAnimationViaWebglService(sampleEmotiveAnimationBundle(), {
      serviceUrl: 'http://127.0.0.1:8091',
      outputPath: '/tmp/out.mp4',
      timeoutMs: 2000,
      maxAttempts: 2,
    });
    assert.equal(attempts, 2);
    assert.equal(result.outputPath, '/tmp/out.mp4');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('renderEmotiveAnimationViaWebglService surfaces renderer_not_ready without burning local retries', async () => {
  const originalFetch = globalThis.fetch;
  let attempts = 0;
  try {
    globalThis.fetch = async () => {
      attempts += 1;
      return {
        ok: false,
        status: 503,
        async json() {
          return {
            error: 'renderer_not_ready',
          };
        },
      };
    };

    await assert.rejects(
      renderEmotiveAnimationViaWebglService(sampleEmotiveAnimationBundle(), {
        serviceUrl: 'http://127.0.0.1:8091',
        outputPath: '/tmp/out.mp4',
        maxAttempts: 3,
      }),
      (error) => {
        assert.equal(error.code, 'renderer_not_ready');
        assert.equal(error.stage, 'render');
        return true;
      },
    );
    assert.equal(attempts, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('buildWebglRenderTimeoutMs scales with bundle size within explicit bounds', () => {
  const shortTimeout = buildWebglRenderTimeoutMs(sampleEmotiveAnimationBundle(), {
    minTimeoutMs: 45000,
    maxTimeoutMs: 180000,
  });
  const largerBundle = {
    ...sampleEmotiveAnimationBundle(),
    duration_seconds: 22,
    distinct_keyframe_count: 37,
    keyframes: Array.from({ length: 37 }, (_, index) => ({
      ordinal: index,
      trace_block_index: index + 1,
      source_kind: 'assistant_content',
      hold_keyframe_count: 1,
      trace_block_span: [index + 1, index + 1],
      moment: {
        confidence: 0.5,
        runtime_trust: 0.8,
      },
      dominant_dimensions: ['confidence'],
    })),
  };
  const largeTimeout = buildWebglRenderTimeoutMs(largerBundle, {
    minTimeoutMs: 45000,
    maxTimeoutMs: 180000,
  });

  assert.ok(shortTimeout >= 45000);
  assert.ok(largeTimeout > shortTimeout);
  assert.ok(largeTimeout <= 180000);
});

test('getWebglRendererHealth reads the local service health payload', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async (url) => {
      assert.equal(String(url), 'http://127.0.0.1:8091/health');
      return {
        ok: true,
        async json() {
          return {
            status: 'ok',
            backend: 'chromium_webgl',
            gpu: {
              renderer: 'NVIDIA RTX',
              software: false,
            },
          };
        },
      };
    };

    const health = await getWebglRendererHealth('http://127.0.0.1:8091');
    assert.equal(health.status, 'ok');
    assert.equal(health.gpu.renderer, 'NVIDIA RTX');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('pending video deliveries preserve stage and persisted artifact metadata', () => {
  const state = enqueueTelegramPendingVideoDelivery({}, {
    jobId: 'job-1',
    chatId: '7502424413',
    conversationId: 'tc1',
    sequenceNumber: 2,
    replyToMessageId: 609,
    textTelegramMessageId: 611,
    deliveryMode: 'reply',
    videoPayload: {
      has_spoiler: true,
      supports_streaming: true,
    },
    bundle: sampleEmotiveAnimationBundle(),
    attemptCount: 3,
    stage: 'upload',
    artifactPath: '/var/lib/vicuna/telegram-video-spool/job-1.mp4',
    artifactReadyAtMs: 1234,
    nextAttemptAtMs: 5678,
    createdAtMs: 1000,
    updatedAtMs: 2000,
    lastError: 'telegram sendVideo transport failed: fetch failed',
  });

  assert.deepEqual(getTelegramPendingVideoDelivery(state, 'job-1'), {
    jobId: 'job-1',
    chatId: '7502424413',
    conversationId: 'tc1',
    sequenceNumber: 2,
    replyToMessageId: 609,
    textTelegramMessageId: 611,
    deliveryMode: 'reply',
    videoPayload: {
      has_spoiler: true,
      supports_streaming: true,
    },
    bundle: sampleEmotiveAnimationBundle(),
    attemptCount: 3,
    stage: 'upload',
    artifactPath: '/var/lib/vicuna/telegram-video-spool/job-1.mp4',
    artifactReadyAtMs: 1234,
    nextAttemptAtMs: 5678,
    createdAtMs: 1000,
    updatedAtMs: 2000,
    lastError: 'telegram sendVideo transport failed: fetch failed',
  });
});
