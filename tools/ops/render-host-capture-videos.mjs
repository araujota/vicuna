import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  buildTelegramEmotiveAnimationBundle,
} from '../telegram-bridge/emotive-animation-bundle.mjs';
import {
  getWebglRendererHealth,
  renderEmotiveAnimationViaWebglService,
} from '../telegram-bridge/emotive-webgl-renderer-client.mjs';

const env = {
  sourcePath: process.env.VICUNA_HOST_CAPTURE_RENDER_SOURCE_PATH ?? '/var/lib/vicuna/experimental-capture/live/emotive_traces.jsonl',
  videoDir: process.env.VICUNA_HOST_CAPTURE_RENDER_VIDEO_DIR ?? '/var/lib/vicuna/experimental-capture/live/videos',
  statePath: process.env.VICUNA_HOST_CAPTURE_RENDER_STATE_PATH ?? '/var/lib/vicuna/experimental-capture/render-state.json',
  rendererUrl: (process.env.VICUNA_WEBGL_RENDERER_URL ?? 'http://127.0.0.1:8091').replace(/\/+$/, ''),
  timeoutMs: Math.max(1000, Number(process.env.VICUNA_HOST_CAPTURE_RENDER_TIMEOUT_MS ?? 240000) || 240000),
};

function log(message, extra = undefined) {
  const payload = {
    schema_version: 'vicuna.service_event.v1',
    timestamp_ms: Date.now(),
    service: 'host-capture-render',
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

function parseInteger(value, fallback = 0) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.trunc(parsed);
}

function slugify(value, fallback) {
  const normalized = String(value ?? '')
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return normalized || fallback;
}

async function readState() {
  try {
    const text = await readFile(env.statePath, 'utf8');
    const parsed = JSON.parse(text);
    return {
      line_count: Math.max(0, parseInteger(parsed?.line_count, 0)),
    };
  } catch {
    return { line_count: 0 };
  }
}

async function writeState(state) {
  const next = {
    line_count: Math.max(0, parseInteger(state?.line_count, 0)),
    updated_at: new Date().toISOString(),
  };
  await mkdir(path.dirname(env.statePath), { recursive: true });
  const tempPath = `${env.statePath}.tmp`;
  await writeFile(tempPath, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
  await writeFile(env.statePath, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
}

async function sourceLines() {
  try {
    const text = await readFile(env.sourcePath, 'utf8');
    return text.split(/\r?\n/).filter((line) => line.length > 0);
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return [];
    }
    throw error;
  }
}

function extractTraceRow(line) {
  const row = JSON.parse(line);
  const trace = row?.emotive_trace && typeof row.emotive_trace === 'object' && !Array.isArray(row.emotive_trace)
    ? row.emotive_trace
    : row?.trace && typeof row.trace === 'object' && !Array.isArray(row.trace)
      ? row.trace
      : null;
  return { row, trace };
}

async function fileExists(targetPath) {
  try {
    await stat(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function processRow(line, lineNumber) {
  const { row, trace } = extractTraceRow(line);
  if (!trace) {
    log('skipping capture row without emotive trace', {
      lineNumber,
      reason: 'missing_trace',
    });
    return { status: 'skipped' };
  }

  const bundle = row?.emotive_animation && typeof row.emotive_animation === 'object' && !Array.isArray(row.emotive_animation)
    ? row.emotive_animation
    : buildTelegramEmotiveAnimationBundle(trace);
  if (!bundle) {
    log('skipping capture row without renderable bundle', {
      lineNumber,
      traceId: String(trace?.trace_id ?? ''),
      reason: 'missing_bundle',
    });
    return { status: 'skipped' };
  }

  const baseName = slugify(trace?.trace_id || row?.trace_id || row?.request_id, `trace_line_${lineNumber}`);
  const outputPath = path.join(env.videoDir, `${baseName}.mp4`);
  if (await fileExists(outputPath)) {
    log('skipping already-rendered emotive video', {
      lineNumber,
      traceId: String(trace?.trace_id ?? ''),
      outputPath,
    });
    return { status: 'already_exists', outputPath };
  }

  const requestId = `host_capture_render_${baseName}_${lineNumber}`;
  const result = await renderEmotiveAnimationViaWebglService(bundle, {
    serviceUrl: env.rendererUrl,
    outputPath,
    requestId,
    timeoutMs: env.timeoutMs,
  });
  log('rendered emotive capture video', {
    lineNumber,
    traceId: String(trace?.trace_id ?? ''),
    requestId,
    outputPath: String(result?.outputPath ?? outputPath),
    keyframeCount: Number(result?.keyframeCount ?? bundle?.keyframes?.length ?? 0) || 0,
    durationSeconds: Number(result?.durationSeconds ?? bundle?.duration_seconds ?? 0) || 0,
  });
  return { status: 'rendered', outputPath: String(result?.outputPath ?? outputPath) };
}

async function main() {
  await mkdir(env.videoDir, { recursive: true });
  const health = await getWebglRendererHealth(env.rendererUrl);
  log('renderer health checked', {
    status: String(health?.status ?? ''),
    rendererUrl: env.rendererUrl,
    backend: String(health?.backend ?? ''),
  });

  const lines = await sourceLines();
  const state = await readState();
  let lastProcessed = Math.max(0, parseInteger(state.line_count, 0));
  if (lines.length < lastProcessed) {
    log('source trace file shrank; resetting render offset', {
      previousLineCount: lastProcessed,
      currentLineCount: lines.length,
    });
    lastProcessed = 0;
  }

  if (lines.length === lastProcessed) {
    log('no new emotive trace rows to render', {
      lineCount: lines.length,
    });
    if (lines.length !== state.line_count) {
      await writeState({ line_count: lines.length });
    }
    return;
  }

  for (let index = lastProcessed; index < lines.length; index += 1) {
    await processRow(lines[index], index + 1);
    await writeState({ line_count: index + 1 });
  }
}

main().catch((error) => {
  log('host capture render failed', {
    error: String(error?.message ?? error),
    stage: String(error?.stage ?? ''),
  });
  process.exitCode = 1;
});
