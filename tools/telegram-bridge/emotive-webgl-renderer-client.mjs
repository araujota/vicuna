import { setTimeout as delay } from 'node:timers/promises';
import { buildEmotiveAnimationRenderPlan } from './emotive-animation-render.mjs';

export async function getWebglRendererHealth(serviceUrl) {
  const url = new URL('/health', serviceUrl);
  const response = await fetch(url, {
    method: 'GET',
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`webgl renderer health failed: ${response.status} ${JSON.stringify(body)}`);
  }
  return body;
}

export function buildWebglRenderTimeoutMs(bundle, {
  minTimeoutMs = 120000,
  maxTimeoutMs = 240000,
  perSecondMs = 2500,
  perKeyframeMs = 750,
} = {}) {
  const floorMs = Math.max(1000, Number(minTimeoutMs) || 120000);
  const ceilingMs = Math.max(floorMs, Number(maxTimeoutMs) || 240000);
  const renderPlan = buildEmotiveAnimationRenderPlan(bundle);
  const durationSeconds = Math.max(0, Number(renderPlan?.durationSeconds ?? bundle?.duration_seconds ?? 0) || 0);
  const keyframeCount = Math.max(
    0,
    Number(renderPlan?.keyframeCount ?? bundle?.distinct_keyframe_count ?? bundle?.keyframes?.length ?? 0) || 0,
  );
  const derivedMs = floorMs + (durationSeconds * perSecondMs) + (keyframeCount * perKeyframeMs);
  return Math.max(floorMs, Math.min(ceilingMs, Math.round(derivedMs)));
}

export async function renderEmotiveAnimationViaWebglService(bundle, {
  serviceUrl,
  outputPath = '',
  requestId = '',
  timeoutMs = 120000,
  maxAttempts = 3,
} = {}) {
  const url = new URL('/render', serviceUrl);
  const requestBody = JSON.stringify({
    bundle,
    outputPath,
    requestId,
  });
  const boundedAttempts = Math.max(1, Math.min(5, Number(maxAttempts) || 3));
  const boundedTimeoutMs = buildWebglRenderTimeoutMs(bundle, {
    minTimeoutMs: timeoutMs,
  });
  let lastError = null;
  for (let attempt = 1; attempt <= boundedAttempts; attempt += 1) {
    let response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: requestBody,
        signal: AbortSignal.timeout(boundedTimeoutMs),
      });
    } catch (error) {
      const aborted = error?.name === 'TimeoutError' || error?.name === 'AbortError';
      lastError = new Error(
        aborted
          ? `webgl render request timed out after ${boundedTimeoutMs} ms`
          : `webgl render transport failed: ${String(error?.message ?? error)}`,
      );
      lastError.stage = 'render';
      if (attempt >= boundedAttempts) {
        throw lastError;
      }
      await delay(250 * attempt);
      continue;
    }

    let body = {};
    try {
      body = await response.json();
    } catch {
      body = {};
    }
    if (response.ok) {
      return body;
    }

    if (response.status === 503 && String(body?.error ?? '').trim() === 'renderer_not_ready') {
      lastError = new Error('renderer_not_ready');
      lastError.code = 'renderer_not_ready';
      lastError.stage = 'render';
      throw lastError;
    }

    const retryable = response.status >= 500 && response.status < 600;
    lastError = new Error(body?.error || `webgl renderer failed: ${response.status}`);
    lastError.stage = String(body?.error_stage ?? '').trim() || 'render';
    if (!retryable || attempt >= boundedAttempts) {
      throw lastError;
    }
    await delay(250 * attempt);
  }
  throw lastError ?? new Error('webgl renderer failed without a usable response');
}
