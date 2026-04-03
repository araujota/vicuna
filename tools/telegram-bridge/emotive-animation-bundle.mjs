const TELEGRAM_EMOTIVE_DIMENSIONS = [
  ['epistemic_pressure', 'Epistemic Pressure'],
  ['confidence', 'Confidence'],
  ['contradiction_pressure', 'Contradiction Pressure'],
  ['planning_clarity', 'Planning Clarity'],
  ['curiosity', 'Curiosity'],
  ['caution', 'Caution'],
  ['frustration', 'Frustration'],
  ['satisfaction', 'Satisfaction'],
  ['momentum', 'Momentum'],
  ['stall', 'Stall'],
  ['semantic_novelty', 'Semantic Novelty'],
  ['user_alignment', 'User Alignment'],
  ['runtime_trust', 'Runtime Trust'],
  ['runtime_failure_pressure', 'Runtime Failure Pressure'],
];

function trimString(value) {
  return String(value ?? '').trim();
}

function parseInteger(value, fallback = 0) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.trunc(parsed);
}

function parseFloatNumber(value, fallback = 0) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return parsed;
}

function momentEquals(lhs, rhs) {
  if (!lhs || !rhs || typeof lhs !== 'object' || typeof rhs !== 'object') {
    return false;
  }
  return JSON.stringify(lhs) === JSON.stringify(rhs);
}

function appendUniqueStringValues(target, values) {
  const seen = new Set(
    Array.isArray(target)
      ? target.map((value) => trimString(value)).filter(Boolean)
      : [],
  );
  const result = Array.isArray(target) ? [...target] : [];
  for (const value of Array.isArray(values) ? values : []) {
    const text = trimString(value);
    if (!text || seen.has(text)) {
      continue;
    }
    result.push(text);
    seen.add(text);
  }
  return result;
}

function dimensionDirection(index, total) {
  if (total <= 0) {
    return [0, 1, 0];
  }
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - ((2 * (index + 0.5)) / total);
  const radius = Math.sqrt(Math.max(0, 1 - (y * y)));
  const theta = goldenAngle * index;
  return [
    Math.cos(theta) * radius,
    y,
    Math.sin(theta) * radius,
  ];
}

export function resolveTelegramEmotiveAnimationStartBlockIndex(trace) {
  const blocks = Array.isArray(trace?.blocks) ? trace.blocks : [];
  if (blocks.length === 0) {
    return 0;
  }

  const totalBlocks = blocks.length;
  const liveStart = Math.min(
    Math.max(0, parseInteger(trace?.live_generation_start_block_index, 0)),
    totalBlocks,
  );
  const turnStart = Math.min(
    Math.max(0, parseInteger(trace?.turn_start_block_index, 0)),
    liveStart,
  );
  if (turnStart > 0 || parseInteger(trace?.turn_start_block_index, -1) === 0) {
    if (liveStart > 0 || turnStart > 0) {
      return turnStart;
    }
  }

  for (let index = liveStart - 1; index >= 0; index -= 1) {
    const kind = trimString(blocks[index]?.source?.kind);
    if (kind === 'user_message') {
      return index;
    }
  }
  return 0;
}

export function buildTelegramEmotiveMoment(moment) {
  if (!moment || typeof moment !== 'object' || Array.isArray(moment)) {
    return null;
  }
  const result = {};
  for (const [id] of TELEGRAM_EMOTIVE_DIMENSIONS) {
    result[id] = parseFloatNumber(moment[id], 0);
  }
  return result;
}

export function buildTelegramEmotiveAnimationBundle(trace) {
  if (!trace || typeof trace !== 'object' || Array.isArray(trace)) {
    return null;
  }
  const blocks = Array.isArray(trace.blocks) ? trace.blocks : [];
  if (blocks.length === 0) {
    return null;
  }

  const totalBlocks = blocks.length;
  const turnStart = Math.min(
    resolveTelegramEmotiveAnimationStartBlockIndex(trace),
    totalBlocks,
  );
  const requestedStart = Math.max(0, parseInteger(trace.live_generation_start_block_index, 0));
  const liveStart = Math.min(requestedStart, totalBlocks);
  const animationStart = Math.min(turnStart, liveStart);

  const dimensions = TELEGRAM_EMOTIVE_DIMENSIONS.map(([id, label], index) => ({
    id,
    label,
    direction_index: index,
    direction_xyz: dimensionDirection(index, TELEGRAM_EMOTIVE_DIMENSIONS.length),
  }));

  const keyframes = [];
  const segments = [];
  let rawKeyframeCount = 0;
  const fallbackStepMs = 500;
  const maxReasonableTimestampDeltaMs = 60 * 1000;
  let previousOffsetMs = 0;
  let previousRawTimestampMs = 0;
  let havePreviousOffset = false;
  let havePreviousRawTimestamp = false;

  for (let blockIndex = animationStart; blockIndex < totalBlocks; blockIndex += 1) {
    const block = blocks[blockIndex];
    if (!block || typeof block !== 'object' || Array.isArray(block)) {
      continue;
    }
    const moment = buildTelegramEmotiveMoment(block.moment);
    if (!moment) {
      continue;
    }

    const sourceBlockIndex = parseInteger(block.block_index, blockIndex);
    const dominantDimensions = Array.isArray(block?.vad?.dominant_dimensions)
      ? block.vad.dominant_dimensions
      : [];
    const hasTimestampMs = Number.isInteger(block.timestamp_ms);
    const rawTimestampMs = hasTimestampMs ? parseInteger(block.timestamp_ms, 0) : 0;
    let blockOffsetMs = 0;
    if (hasTimestampMs) {
      if (!havePreviousOffset || !havePreviousRawTimestamp) {
        blockOffsetMs = 0;
      } else {
        let deltaMs = rawTimestampMs - previousRawTimestampMs;
        if (deltaMs < 0) {
          deltaMs = 0;
        } else if (deltaMs > maxReasonableTimestampDeltaMs) {
          deltaMs = fallbackStepMs;
        }
        blockOffsetMs = previousOffsetMs + deltaMs;
      }
      previousRawTimestampMs = rawTimestampMs;
      havePreviousRawTimestamp = true;
    } else if (havePreviousOffset) {
      blockOffsetMs = previousOffsetMs + fallbackStepMs;
    }

    rawKeyframeCount += 1;
    previousOffsetMs = blockOffsetMs;
    havePreviousOffset = true;

    const previous = keyframes[keyframes.length - 1];
    if (previous && momentEquals(previous.moment, moment)) {
      previous.hold_keyframe_count += 1;
      previous.end_offset_ms = Math.max(previous.end_offset_ms, blockOffsetMs);
      previous.hold_duration_ms = Math.max(0, previous.end_offset_ms - previous.start_offset_ms);
      if (!Array.isArray(previous.trace_block_span) || previous.trace_block_span.length !== 2) {
        previous.trace_block_span = [previous.trace_block_index, sourceBlockIndex];
      } else {
        previous.trace_block_span[1] = sourceBlockIndex;
      }
      previous.dominant_dimensions = appendUniqueStringValues(previous.dominant_dimensions, dominantDimensions);
      continue;
    }

    keyframes.push({
      ordinal: keyframes.length,
      trace_block_index: sourceBlockIndex,
      source_kind: trimString(block?.source?.kind) || 'runtime_event',
      hold_keyframe_count: 1,
      trace_block_span: [sourceBlockIndex, sourceBlockIndex],
      start_offset_ms: blockOffsetMs,
      end_offset_ms: blockOffsetMs,
      hold_duration_ms: 0,
      moment,
      dominant_dimensions: Array.isArray(dominantDimensions) ? [...dominantDimensions] : [],
    });
  }

  if (keyframes.length === 0) {
    return null;
  }

  for (let index = 0; index + 1 < keyframes.length; index += 1) {
    const current = keyframes[index];
    const next = keyframes[index + 1];
    const transitionStartMs = parseInteger(current.end_offset_ms, parseInteger(current.start_offset_ms, 0));
    const transitionEndMs = Math.max(transitionStartMs, parseInteger(next.start_offset_ms, transitionStartMs));
    segments.push({
      segment_index: segments.length,
      from_keyframe_ordinal: parseInteger(current.ordinal, index),
      to_keyframe_ordinal: parseInteger(next.ordinal, index + 1),
      start_offset_ms: transitionStartMs,
      duration_ms: Math.max(0, transitionEndMs - transitionStartMs),
    });
  }

  const finalKeyframe = keyframes[keyframes.length - 1];
  const totalDurationMs = Math.max(
    0,
    parseInteger(finalKeyframe.end_offset_ms, parseInteger(finalKeyframe.start_offset_ms, 0)),
  );

  return {
    bundle_version: 3,
    trace_id: trimString(trace.trace_id),
    generation_start_block_index: animationStart,
    turn_start_block_index: turnStart,
    live_generation_start_block_index: liveStart,
    raw_keyframe_count: rawKeyframeCount,
    distinct_keyframe_count: keyframes.length,
    fps: 30,
    viewport_width: 720,
    viewport_height: 720,
    rotation_period_seconds: 12,
    duration_seconds: totalDurationMs / 1000,
    dimensions,
    keyframes,
    segments,
  };
}
