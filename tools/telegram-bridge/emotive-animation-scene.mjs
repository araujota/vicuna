import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Matrix4,
  Vector3,
} from 'three';
import { ConvexHull } from 'three/addons/math/ConvexHull.js';

export const ANCHOR_BASE_RADIUS = 1.01;
export const ANCHOR_MIN_RADIUS = ANCHOR_BASE_RADIUS / 3;
export const ANCHOR_MAX_RADIUS = (ANCHOR_BASE_RADIUS * 5) / 3;
export const ANCHOR_DEFORMATION_AMPLITUDE = ANCHOR_MAX_RADIUS - ANCHOR_BASE_RADIUS;
const OUTER_HULL_TARGET_TRIANGLES = 9000;
const OUTER_HULL_DIRECTION_RELAX_PASSES = 5;
const OUTER_HULL_DIRECTION_RELAXATION_ALPHA = 0.34;
const OUTER_HULL_INFLUENCE_SHARPNESS = 2.2;
const OUTER_HULL_LOCAL_ANCHOR_COUNT = 10;
const OUTER_HULL_BROAD_INFLUENCE_SHARPNESS = 0.36;
const OUTER_HULL_BROAD_ANCHOR_COUNT = 14;
const OUTER_HULL_CAP_INNER_ANGLE = 0.09;
const OUTER_HULL_CAP_OUTER_ANGLE = 0.28;
const OUTER_HULL_SHOULDER_INNER_ANGLE = 0.14;
const OUTER_HULL_SHOULDER_OUTER_ANGLE = 2.28;
const OUTER_HULL_SHOULDER_RING_CENTER = 0.72;
const OUTER_HULL_SHOULDER_RING_WIDTH = 0.88;
const OUTER_HULL_MOUND_INNER_ANGLE = 0.06;
const OUTER_HULL_MOUND_OUTER_ANGLE = 0.34;
const OUTER_HULL_BASE_SAG = 0.12;
const OUTER_HULL_VARIANCE_SAG = 0.1;
const OUTER_HULL_RELAX_PASSES = 16;
const OUTER_HULL_RELAXATION_ALPHA = 0.28;
export const OUTER_HULL_MIN_RADIUS = ANCHOR_MIN_RADIUS;
export const BACKGROUND_COLOR = '#000000';
export const SURFACE_INNER = new Color('#7f2cff');
export const SURFACE_MID = new Color('#1affc7');
export const SURFACE_OUTER = new Color('#ff5b1f');
export const LIGHT_DIRECTION = new Vector3(0.45, 0.65, 1.0).normalize();

export function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

export function lerp(lhs, rhs, alpha) {
  return lhs + ((rhs - lhs) * alpha);
}

function smoothstep(edge0, edge1, value) {
  if (edge0 === edge1) {
    return value >= edge1 ? 1 : 0;
  }
  const t = clamp01((value - edge0) / (edge1 - edge0));
  return t * t * (3 - (2 * t));
}

export function easeInOutSine(alpha) {
  return 0.5 - (0.5 * Math.cos(Math.PI * clamp01(alpha)));
}

export function normalizeVector3(values) {
  let vector = null;
  if (values && typeof values.clone === 'function') {
    vector = values.clone();
  } else if (Array.isArray(values) && values.length === 3) {
    vector = new Vector3(
      Number(values[0]) || 0,
      Number(values[1]) || 0,
      Number(values[2]) || 0,
    );
  } else if (values && typeof values === 'object') {
    vector = new Vector3(
      Number(values.x) || 0,
      Number(values.y) || 0,
      Number(values.z) || 0,
    );
  }
  if (!vector) {
    return null;
  }
  if (vector.lengthSq() <= 1e-9) {
    return null;
  }
  return vector.normalize();
}

export function normalizeEmotiveAnimationBundle(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }

  const dimensions = Array.isArray(raw.dimensions)
    ? raw.dimensions.map((dimension, index) => {
      const id = String(dimension?.id ?? '').trim();
      const label = String(dimension?.label ?? '').trim();
      const direction = normalizeVector3(dimension?.direction_xyz ?? dimension?.direction);
      if (!id || !label || !direction) {
        return null;
      }
      return {
        id,
        label,
        directionIndex: Math.max(0, Number(dimension?.direction_index ?? index) || index),
        direction,
      };
    }).filter(Boolean)
    : [];

  if (dimensions.length === 0) {
    return null;
  }

  const dimensionIds = new Set(dimensions.map((dimension) => dimension.id));
  const keyframes = Array.isArray(raw.keyframes)
    ? raw.keyframes.map((keyframe, index) => {
      const momentRaw = keyframe?.moment && typeof keyframe.moment === 'object' && !Array.isArray(keyframe.moment)
        ? keyframe.moment
        : null;
      if (!momentRaw) {
        return null;
      }
      const moment = {};
      for (const dimension of dimensions) {
        moment[dimension.id] = clamp01(momentRaw[dimension.id]);
      }
      const dominantDimensions = Array.isArray(keyframe?.dominant_dimensions)
        ? keyframe.dominant_dimensions.map((value) => String(value ?? '').trim()).filter((value) => dimensionIds.has(value))
        : Array.isArray(keyframe?.dominantDimensions)
          ? keyframe.dominantDimensions.map((value) => String(value ?? '').trim()).filter((value) => dimensionIds.has(value))
        : [];
      return {
        ordinal: Math.max(0, Number(keyframe?.ordinal ?? index) || index),
        traceBlockIndex: Math.max(0, Number(keyframe?.trace_block_index ?? keyframe?.traceBlockIndex ?? index) || index),
        sourceKind: String(keyframe?.source_kind ?? keyframe?.sourceKind ?? '').trim() || 'runtime_event',
        holdKeyframeCount: Math.max(1, Math.round(Number(keyframe?.hold_keyframe_count ?? keyframe?.holdKeyframeCount ?? 1) || 1)),
        traceBlockSpan: Array.isArray(keyframe?.trace_block_span) && keyframe.trace_block_span.length === 2
          ? [
            Math.max(0, Number(keyframe.trace_block_span[0] ?? keyframe?.trace_block_index ?? index) || index),
            Math.max(0, Number(keyframe.trace_block_span[1] ?? keyframe?.trace_block_index ?? index) || index),
          ]
          : Array.isArray(keyframe?.traceBlockSpan) && keyframe.traceBlockSpan.length === 2
            ? [
              Math.max(0, Number(keyframe.traceBlockSpan[0] ?? keyframe?.traceBlockIndex ?? index) || index),
              Math.max(0, Number(keyframe.traceBlockSpan[1] ?? keyframe?.traceBlockIndex ?? index) || index),
            ]
          : [
            Math.max(0, Number(keyframe?.trace_block_index ?? keyframe?.traceBlockIndex ?? index) || index),
            Math.max(0, Number(keyframe?.trace_block_index ?? keyframe?.traceBlockIndex ?? index) || index),
          ],
        moment,
        dominantDimensions,
      };
    }).filter(Boolean)
    : [];

  if (keyframes.length === 0) {
    return null;
  }

  const secondsPerKeyframe = Math.max(0.1, Number(raw.seconds_per_keyframe ?? raw.secondsPerKeyframe ?? 0.5) || 0.5);
  const fps = Math.max(1, Math.round(Number(raw.fps ?? 24) || 24));
  const viewportWidth = Math.max(256, Math.round(Number(raw.viewport_width ?? raw.viewportWidth ?? 720) || 720));
  const viewportHeight = Math.max(256, Math.round(Number(raw.viewport_height ?? raw.viewportHeight ?? 720) || 720));
  const rotationPeriodSeconds = Math.max(1, Number(raw.rotation_period_seconds ?? raw.rotationPeriodSeconds ?? 12) || 12);
  const timelineSlots = [];
  for (const keyframe of keyframes) {
    for (let slot = 0; slot < keyframe.holdKeyframeCount; slot += 1) {
      timelineSlots.push(keyframe.moment);
    }
  }
  const rawKeyframeCount = Math.max(1, timelineSlots.length);
  const distinctKeyframeCount = Math.max(1, keyframes.length);
  const durationSeconds = secondsPerKeyframe * rawKeyframeCount;
  const totalFrames = Math.max(1, Math.round(durationSeconds * fps));

  return {
    bundleVersion: Math.max(1, Number(raw.bundle_version ?? raw.bundleVersion ?? 1) || 1),
    traceId: String(raw.trace_id ?? raw.traceId ?? '').trim(),
    generationStartBlockIndex: Math.max(0, Number(raw.generation_start_block_index ?? raw.generationStartBlockIndex ?? 0) || 0),
    rawKeyframeCount,
    distinctKeyframeCount,
    secondsPerKeyframe,
    fps,
    viewportWidth,
    viewportHeight,
    rotationPeriodSeconds,
    durationSeconds,
    totalFrames,
    timelineSlots,
    dimensions,
    keyframes,
  };
}

export function coerceNormalizedBundle(rawBundle) {
  if (
    rawBundle
    && typeof rawBundle === 'object'
    && !Array.isArray(rawBundle)
    && Array.isArray(rawBundle.dimensions)
    && Array.isArray(rawBundle.keyframes)
    && Array.isArray(rawBundle.timelineSlots)
    && Number(rawBundle.totalFrames ?? 0) > 0
    && Number(rawBundle.durationSeconds ?? 0) > 0
    && Number(rawBundle.secondsPerKeyframe ?? 0) > 0
    && rawBundle.dimensions.every((dimension) => dimension?.direction && typeof dimension.direction.clone === 'function')
  ) {
    return rawBundle;
  }
  return normalizeEmotiveAnimationBundle(rawBundle);
}

export function buildEmotiveAnimationRenderPlan(rawBundle) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    return null;
  }
  return {
    traceId: bundle.traceId,
    keyframeCount: bundle.distinctKeyframeCount,
    rawKeyframeCount: bundle.rawKeyframeCount,
    totalFrames: bundle.totalFrames,
    durationSeconds: bundle.durationSeconds,
    fps: bundle.fps,
    viewportWidth: bundle.viewportWidth,
    viewportHeight: bundle.viewportHeight,
  };
}

export function supportedAnchorRadiusForMagnitude(magnitude) {
  return lerp(ANCHOR_MIN_RADIUS, ANCHOR_MAX_RADIUS, clamp01(magnitude));
}

export function computeBundleRadiusNormalizationRange(rawBundle, topology = null) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    return null;
  }

  let maxMomentValue = 0;
  for (const keyframe of bundle.keyframes) {
    for (const dimension of bundle.dimensions) {
      maxMomentValue = Math.max(maxMomentValue, clamp01(keyframe.moment?.[dimension.id]));
    }
  }

  return {
    minRadius: ANCHOR_MIN_RADIUS,
    maxRadius: ANCHOR_MAX_RADIUS,
    maxMomentValue,
  };
}

export function extractEmotiveAnimationTerminalMoment(rawBundle) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle || bundle.keyframes.length === 0) {
    return null;
  }
  return JSON.parse(JSON.stringify(bundle.keyframes[bundle.keyframes.length - 1].moment ?? null));
}

export function prependEmotiveAnimationStartMoment(rawBundle, startMoment) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    return null;
  }
  if (!startMoment || typeof startMoment !== 'object' || Array.isArray(startMoment)) {
    return bundle;
  }

  const moment = {};
  let changed = false;
  for (const dimension of bundle.dimensions) {
    moment[dimension.id] = clamp01(startMoment[dimension.id]);
    if (Math.abs(moment[dimension.id] - (bundle.keyframes[0]?.moment?.[dimension.id] ?? 0)) > 1e-6) {
      changed = true;
    }
  }
  if (!changed) {
    return bundle;
  }

  const keyframes = [
    {
      ordinal: 0,
      traceBlockIndex: Math.max(0, bundle.generationStartBlockIndex - 1),
      sourceKind: 'prior_delivery_terminal',
      holdKeyframeCount: 1,
      traceBlockSpan: [
        Math.max(0, bundle.generationStartBlockIndex - 1),
        Math.max(0, bundle.generationStartBlockIndex - 1),
      ],
      moment,
      dominantDimensions: [],
    },
    ...bundle.keyframes.map((keyframe, index) => ({
      ...keyframe,
      ordinal: index + 1,
    })),
  ];
  const timelineSlots = [{ ...moment }, ...bundle.timelineSlots];
  const rawKeyframeCount = bundle.rawKeyframeCount + 1;
  const durationSeconds = bundle.secondsPerKeyframe * rawKeyframeCount;
  return {
    ...bundle,
    rawKeyframeCount,
    distinctKeyframeCount: keyframes.length,
    durationSeconds,
    totalFrames: Math.max(1, Math.round(durationSeconds * bundle.fps)),
    timelineSlots,
    keyframes,
  };
}

export function computeFrameState(bundle, frameIndex) {
  const duration = bundle.durationSeconds;
  const t = bundle.totalFrames <= 1
    ? 0
    : (frameIndex / Math.max(1, bundle.totalFrames - 1)) * duration;
  const segmentIndex = Math.min(
    bundle.timelineSlots.length - 1,
    Math.max(0, Math.floor(t / bundle.secondsPerKeyframe)),
  );
  const nextSegmentIndex = Math.min(bundle.timelineSlots.length - 1, segmentIndex + 1);
  const segmentStart = segmentIndex * bundle.secondsPerKeyframe;
  const localAlpha = nextSegmentIndex === segmentIndex
    ? 0
    : easeInOutSine((t - segmentStart) / bundle.secondsPerKeyframe);

  const magnitudes = {};
  for (const dimension of bundle.dimensions) {
    const current = bundle.timelineSlots[segmentIndex]?.[dimension.id] ?? 0;
    const next = bundle.timelineSlots[nextSegmentIndex]?.[dimension.id] ?? current;
    magnitudes[dimension.id] = lerp(current, next, localAlpha);
  }

  return {
    t,
    magnitudes,
    rotationY: (2 * Math.PI * t) / bundle.rotationPeriodSeconds,
    rotationX: 0.18 * Math.sin((2 * Math.PI * t) / (bundle.rotationPeriodSeconds * 1.7)),
  };
}

function buildAnchorTriangles(dimensions) {
  const anchorPoints = dimensions.map((dimension) => dimension.direction.clone());
  const pointIndex = new Map(anchorPoints.map((point, index) => [point, index]));
  const hull = new ConvexHull().setFromPoints(anchorPoints);
  const triangles = [];
  const seen = new Set();

  for (const face of hull.faces) {
    const polygon = [];
    let edge = face.edge;
    do {
      polygon.push(pointIndex.get(edge.head().point));
      edge = edge.next;
    } while (edge !== face.edge);

    for (let index = 1; index + 1 < polygon.length; index += 1) {
      const triangle = [polygon[0], polygon[index], polygon[index + 1]];
      const a = anchorPoints[triangle[0]];
      const b = anchorPoints[triangle[1]];
      const c = anchorPoints[triangle[2]];
      const centroid = a.clone().add(b).add(c);
      const outward = b.clone().sub(a).cross(c.clone().sub(a)).dot(centroid) >= 0;
      if (!outward) {
        [triangle[1], triangle[2]] = [triangle[2], triangle[1]];
      }
      const key = triangle
        .slice()
        .sort((lhs, rhs) => lhs - rhs)
        .join(':');
      if (!seen.has(key)) {
        seen.add(key);
        triangles.push(triangle);
      }
    }
  }

  return triangles;
}

function buildAdjacency(vertexCount, indices) {
  const adjacencySets = Array.from({ length: vertexCount }, () => new Set());
  for (let index = 0; index < indices.length; index += 3) {
    const a = indices[index];
    const b = indices[index + 1];
    const c = indices[index + 2];
    adjacencySets[a].add(b);
    adjacencySets[a].add(c);
    adjacencySets[b].add(a);
    adjacencySets[b].add(c);
    adjacencySets[c].add(a);
    adjacencySets[c].add(b);
  }
  return adjacencySets.map((neighbors) => Array.from(neighbors));
}

function buildDirectionalInfluence(direction, dimensions) {
  const dots = new Float32Array(dimensions.length);
  let maxDot = -1;
  let strongestAnchorIndex = 0;
  for (let index = 0; index < dimensions.length; index += 1) {
    const dot = Math.max(-1, Math.min(1, direction.dot(dimensions[index].direction)));
    dots[index] = dot;
    if (dot > maxDot) {
      maxDot = dot;
      strongestAnchorIndex = index;
    }
  }

  const buildKernelWeights = (sharpness, keepCount) => {
    const weights = new Float32Array(dimensions.length);
    const rankedWeights = [];
    let weightSum = 0;
    for (let index = 0; index < dimensions.length; index += 1) {
      const weight = Math.exp(sharpness * (dots[index] - 1));
      weights[index] = weight;
      weightSum += weight;
      rankedWeights.push({ index, weight });
    }
    rankedWeights.sort((lhs, rhs) => rhs.weight - lhs.weight);
    const keep = new Set(
      rankedWeights
        .slice(0, Math.max(1, Math.min(keepCount, rankedWeights.length)))
        .map((entry) => entry.index),
    );
    weightSum = 0;
    for (let index = 0; index < weights.length; index += 1) {
      if (!keep.has(index)) {
        weights[index] = 0;
        continue;
      }
      weightSum += weights[index];
    }
    if (weightSum > 0) {
      for (let index = 0; index < weights.length; index += 1) {
        weights[index] /= weightSum;
      }
    }
    return weights;
  };

  const weights = buildKernelWeights(OUTER_HULL_INFLUENCE_SHARPNESS, OUTER_HULL_LOCAL_ANCHOR_COUNT);
  const broadWeights = buildKernelWeights(OUTER_HULL_BROAD_INFLUENCE_SHARPNESS, OUTER_HULL_BROAD_ANCHOR_COUNT);
  let normalizedMaxWeight = 0;
  for (let index = 0; index < weights.length; index += 1) {
    normalizedMaxWeight = Math.max(normalizedMaxWeight, weights[index]);
  }
  const nearestAnchorDistanceNorm = clamp01(Math.acos(Math.max(-1, Math.min(1, maxDot))) / Math.PI);
  const strongestAngle = Math.acos(Math.max(-1, Math.min(1, maxDot)));
  return {
    weights,
    broadWeights,
    maxWeight: normalizedMaxWeight,
    nearestAnchorDistanceNorm,
    strongestAnchorIndex,
    strongestAngle,
  };
}

function buildAnchorProfileFields(direction, dimensions) {
  const shoulderProfile = new Float32Array(dimensions.length);
  const shoulderRingProfile = new Float32Array(dimensions.length);
  const moundProfile = new Float32Array(dimensions.length);
  for (let index = 0; index < dimensions.length; index += 1) {
    const dot = Math.max(-1, Math.min(1, direction.dot(dimensions[index].direction)));
    const angle = Math.acos(dot);
    const shoulderOuterAngle = Math.max(0.0001, OUTER_HULL_SHOULDER_OUTER_ANGLE);
    const broadShoulder = Math.exp(-Math.pow(
      angle / shoulderOuterAngle,
      1.18,
    ));
    const shoulder = 0.22 + (0.78 * broadShoulder);
    const coreShoulder = Math.exp(-Math.pow(
      angle / Math.max(0.0001, shoulderOuterAngle * OUTER_HULL_SHOULDER_RING_CENTER),
      2.15,
    ));
    const ring = Math.max(
      0,
      broadShoulder - ((0.48 + (0.12 * (1 - OUTER_HULL_SHOULDER_RING_CENTER))) * coreShoulder),
    );
    const mound = 1 - smoothstep(
      OUTER_HULL_MOUND_INNER_ANGLE,
      OUTER_HULL_MOUND_OUTER_ANGLE,
      angle,
    );
    shoulderProfile[index] = Math.pow(shoulder, 0.46) * (0.72 + (0.18 * shoulder));
    shoulderRingProfile[index] = (0.34 + (0.66 * Math.pow(ring, 0.6))) * (0.9 + (0.1 * shoulder));
    moundProfile[index] = mound * mound * (0.18 + (0.06 * shoulder));
  }
  return {
    shoulderProfile,
    shoulderRingProfile,
    moundProfile,
  };
}

function chooseOuterHullSubdivisionLevels(baseTriangleCount) {
  let levels = 0;
  let triangleCount = Math.max(1, baseTriangleCount);
  while (triangleCount < OUTER_HULL_TARGET_TRIANGLES) {
    levels += 1;
    triangleCount *= 4;
  }
  return levels;
}

function subdivideOuterHull(anchorDirections, anchorTriangles, levels) {
  const directions = anchorDirections.map((direction) => direction.clone());
  const anchorIndexByVertex = anchorDirections.map((_, index) => index);
  let triangles = anchorTriangles.map((triangle) => triangle.slice());

  for (let level = 0; level < levels; level += 1) {
    const midpointCache = new Map();
    const nextTriangles = [];

    const midpointIndex = (lhs, rhs) => {
      const key = lhs < rhs ? `${lhs}:${rhs}` : `${rhs}:${lhs}`;
      const existing = midpointCache.get(key);
      if (existing !== undefined) {
        return existing;
      }
      const direction = directions[lhs].clone().add(directions[rhs]).normalize();
      const index = directions.length;
      directions.push(direction);
      anchorIndexByVertex.push(-1);
      midpointCache.set(key, index);
      return index;
    };

    for (const [a, b, c] of triangles) {
      const ab = midpointIndex(a, b);
      const bc = midpointIndex(b, c);
      const ca = midpointIndex(c, a);
      nextTriangles.push(
        [a, ab, ca],
        [ab, b, bc],
        [ca, bc, c],
        [ab, bc, ca],
      );
    }

    triangles = nextTriangles;
  }

  return {
    directions,
    anchorIndexByVertex,
    triangles,
  };
}

function relaxOuterHullDirections(anchorDirections, directions, anchorIndexByVertex, adjacency) {
  let current = directions.map((direction) => direction.clone());
  let next = directions.map(() => new Vector3());

  for (let pass = 0; pass < OUTER_HULL_DIRECTION_RELAX_PASSES; pass += 1) {
    for (let index = 0; index < current.length; index += 1) {
      const anchorIndex = anchorIndexByVertex[index];
      if (anchorIndex >= 0) {
        next[index].copy(anchorDirections[anchorIndex]);
        continue;
      }

      const neighbors = adjacency[index];
      if (!neighbors || neighbors.length === 0) {
        next[index].copy(current[index]);
        continue;
      }

      const averaged = new Vector3();
      for (const neighborIndex of neighbors) {
        averaged.add(current[neighborIndex]);
      }
      if (averaged.lengthSq() <= 1e-9) {
        next[index].copy(current[index]);
        continue;
      }

      averaged.normalize();
      next[index]
        .copy(current[index])
        .lerp(averaged, OUTER_HULL_DIRECTION_RELAXATION_ALPHA)
        .normalize();
    }
    [current, next] = [next, current];
  }

  return current;
}

export function buildOuterHullTopology(bundle) {
  const anchorDirections = bundle.dimensions.map((dimension) => dimension.direction.clone());
  const anchorTriangles = buildAnchorTriangles(bundle.dimensions);
  const subdivisionLevels = chooseOuterHullSubdivisionLevels(anchorTriangles.length);
  const subdivided = subdivideOuterHull(anchorDirections, anchorTriangles, subdivisionLevels);
  const indices = subdivided.triangles.flat();
  const adjacency = buildAdjacency(subdivided.directions.length, indices);
  const relaxedDirections = relaxOuterHullDirections(
    anchorDirections,
    subdivided.directions,
    subdivided.anchorIndexByVertex,
    adjacency,
  );
  const vertexBindings = relaxedDirections.map((direction, index) => {
    const influence = buildDirectionalInfluence(direction, bundle.dimensions);
    const profiles = buildAnchorProfileFields(direction, bundle.dimensions);
    const awayFromAnchor = 1 - influence.maxWeight;
    return {
      anchorIndex: subdivided.anchorIndexByVertex[index],
      direction,
      influenceWeights: influence.weights,
      broadInfluenceWeights: influence.broadWeights,
      shoulderProfile: profiles.shoulderProfile,
      shoulderRingProfile: profiles.shoulderRingProfile,
      moundProfile: profiles.moundProfile,
      maxInfluence: influence.maxWeight,
      nearestAnchorDistanceNorm: influence.nearestAnchorDistanceNorm,
      capAnchorIndex: influence.strongestAnchorIndex,
      capBlend: subdivided.anchorIndexByVertex[index] >= 0
        ? 1
        : 1 - smoothstep(OUTER_HULL_CAP_INNER_ANGLE, OUTER_HULL_CAP_OUTER_ANGLE, influence.strongestAngle),
      shoulderBlend: subdivided.anchorIndexByVertex[index] >= 0
        ? 1
        : 1 - smoothstep(OUTER_HULL_SHOULDER_INNER_ANGLE, OUTER_HULL_SHOULDER_OUTER_ANGLE, influence.strongestAngle),
      valleyPotential:
        smoothstep(0.02, 0.26, influence.nearestAnchorDistanceNorm)
        * smoothstep(0.04, 0.68, awayFromAnchor),
    };
  });

  const geometry = new BufferGeometry();
  const positionAttribute = new Float32BufferAttribute(new Float32Array(vertexBindings.length * 3), 3);
  geometry.setIndex(indices);
  geometry.setAttribute('position', positionAttribute);

  const anchorVertexIndices = vertexBindings
    .map((binding, index) => (binding.anchorIndex >= 0 ? [binding.anchorIndex, index] : null))
    .filter(Boolean)
    .reduce((map, [anchorIndex, vertexIndex]) => map.set(anchorIndex, vertexIndex), new Map());

  return {
    geometry,
    positionAttribute,
    vertexBindings,
    indices,
    adjacency,
    anchorVertexIndices,
    anchorCount: bundle.dimensions.length,
    triangleCount: indices.length / 3,
    vertexCount: vertexBindings.length,
    subdivisions: subdivisionLevels,
  };
}

export function colorForNormalizedDistance(alpha) {
  const clamped = clamp01(alpha);
  if (clamped <= 0.2) {
    return SURFACE_INNER.clone().lerp(new Color('#124dff'), clamped / 0.2);
  }
  if (clamped <= 0.4) {
    return new Color('#124dff').lerp(SURFACE_MID, (clamped - 0.2) / 0.2);
  }
  if (clamped <= 0.6) {
    return SURFACE_MID.clone().lerp(new Color('#ffe53b'), (clamped - 0.4) / 0.2);
  }
  if (clamped <= 0.8) {
    return new Color('#ffe53b').lerp(SURFACE_OUTER, (clamped - 0.6) / 0.2);
  }
  return SURFACE_OUTER.clone().lerp(new Color('#f5261e'), (clamped - 0.8) / 0.2);
}

function computeRadiusField(bundle, topology, frameState) {
  const anchorRadii = bundle.dimensions.map((dimension) => supportedAnchorRadiusForMagnitude(
    frameState.magnitudes[dimension.id] ?? 0,
  ));
  const anchorLifts = anchorRadii.map((radius) => radius - ANCHOR_BASE_RADIUS);
  const meanAnchorRadius = anchorRadii.reduce((sum, radius) => sum + radius, 0) / Math.max(1, anchorRadii.length);
  const peakAnchorRadius = anchorRadii.reduce((max, radius) => Math.max(max, radius), ANCHOR_BASE_RADIUS);
  const meanAnchorLift = clamp01((meanAnchorRadius - ANCHOR_BASE_RADIUS) / Math.max(1e-6, ANCHOR_DEFORMATION_AMPLITUDE));
  const peakAnchorLift = clamp01((peakAnchorRadius - ANCHOR_BASE_RADIUS) / Math.max(1e-6, ANCHOR_DEFORMATION_AMPLITUDE));
  const lowerBounds = new Float32Array(topology.vertexCount);
  const upperBounds = new Float32Array(topology.vertexCount);
  const initialRadii = new Float32Array(topology.vertexCount);

  for (let index = 0; index < topology.vertexBindings.length; index += 1) {
    const binding = topology.vertexBindings[index];
    let weightedRadius = 0;
    let localLift = 0;
    let broadLift = 0;
    let shoulderLift = 0;
    let shoulderWeightSum = 0;
    let shoulderSquaredWeightSum = 0;
    let shoulderRingLift = 0;
    let shoulderRingWeightSum = 0;
    let shoulderRingSquaredWeightSum = 0;
    let moundLift = 0;
    let moundWeightSum = 0;
    let moundSquaredWeightSum = 0;
    for (let anchorIndex = 0; anchorIndex < binding.influenceWeights.length; anchorIndex += 1) {
      const influence = binding.influenceWeights[anchorIndex];
      weightedRadius += influence * anchorRadii[anchorIndex];
      const lift = anchorLifts[anchorIndex];
      localLift += influence * lift;
      const broadInfluence = binding.broadInfluenceWeights[anchorIndex];
      broadLift += broadInfluence * lift;
      const shoulderProfile = binding.shoulderProfile[anchorIndex];
      shoulderWeightSum += shoulderProfile;
      shoulderSquaredWeightSum += shoulderProfile * shoulderProfile;
      shoulderLift += shoulderProfile * lift;
      const shoulderRingProfile = binding.shoulderRingProfile[anchorIndex];
      shoulderRingWeightSum += shoulderRingProfile;
      shoulderRingSquaredWeightSum += shoulderRingProfile * shoulderRingProfile;
      shoulderRingLift += shoulderRingProfile * lift;
      const moundProfile = binding.moundProfile[anchorIndex];
      moundWeightSum += moundProfile;
      moundSquaredWeightSum += moundProfile * moundProfile;
      moundLift += moundProfile * lift;
    }
    let variance = 0;
    for (let anchorIndex = 0; anchorIndex < binding.influenceWeights.length; anchorIndex += 1) {
      const delta = anchorRadii[anchorIndex] - weightedRadius;
      variance += binding.influenceWeights[anchorIndex] * delta * delta;
    }

    const varianceScale = clamp01(Math.sqrt(variance) / Math.max(1e-6, ANCHOR_DEFORMATION_AMPLITUDE));
    const valleyStrength = binding.valleyPotential;
    const localSupport = Math.pow(binding.maxInfluence, 0.72);
    const capBlend = clamp01(binding.capBlend ?? 0);
    const shoulderBlend = clamp01(binding.shoulderBlend ?? 0);
    const strongestRadius = anchorRadii[binding.capAnchorIndex ?? 0];
    const strongestLift = strongestRadius - ANCHOR_BASE_RADIUS;
    const normalizedShoulderLift = shoulderWeightSum > 1e-6
      ? shoulderLift / shoulderWeightSum
      : 0;
    const normalizedShoulderRingLift = shoulderRingWeightSum > 1e-6
      ? shoulderRingLift / shoulderRingWeightSum
      : 0;
    const normalizedMoundLift = moundWeightSum > 1e-6
      ? moundLift / moundWeightSum
      : 0;
    const shoulderEffectiveSupport = shoulderSquaredWeightSum > 1e-6
      ? (shoulderWeightSum * shoulderWeightSum) / shoulderSquaredWeightSum
      : 1;
    const moundEffectiveSupport = moundSquaredWeightSum > 1e-6
      ? (moundWeightSum * moundWeightSum) / moundSquaredWeightSum
      : 1;
    const shoulderOverlapBlend = clamp01((shoulderEffectiveSupport - 1) / 3.5);
    const shoulderRingEffectiveSupport = shoulderRingSquaredWeightSum > 1e-6
      ? (shoulderRingWeightSum * shoulderRingWeightSum) / shoulderRingSquaredWeightSum
      : 1;
    const shoulderRingOverlapBlend = clamp01((shoulderRingEffectiveSupport - 1) / 3.5);
    const moundOverlapBlend = clamp01((moundEffectiveSupport - 1) / 2.5);
    const broadMembraneRadius = Math.max(
      ANCHOR_BASE_RADIUS
        + (0.07 * meanAnchorLift)
        + (0.06 * peakAnchorLift)
        + (0.05 * broadLift)
        - (0.06 * valleyStrength),
      OUTER_HULL_MIN_RADIUS + 0.17 + (0.1 * meanAnchorLift),
    );
    const sag = (OUTER_HULL_BASE_SAG + (OUTER_HULL_VARIANCE_SAG * varianceScale))
      * (0.72 + (0.72 * valleyStrength));
    const shoulderOutward = (broadLift * (0.42 + (0.08 * (1 - valleyStrength))))
      + (normalizedShoulderLift * (0.58 + (0.12 * shoulderBlend)) * (0.48 + (0.3 * shoulderOverlapBlend)))
      + (localLift * (0.16 + (0.05 * localSupport)))
      + (strongestLift * shoulderBlend * (0.06 + (0.03 * shoulderOverlapBlend)));
    const shoulderMoat = normalizedShoulderRingLift
      * (0.78 + (0.28 * valleyStrength))
      * (1.02 + (0.32 * shoulderRingOverlapBlend));
    const moundOutward = (normalizedMoundLift * (0.1 + (0.05 * localSupport)) * (0.1 + (0.18 * moundOverlapBlend)))
      + (strongestLift * capBlend * (0.025 + (0.015 * moundOverlapBlend)));
    const targetRadius = broadMembraneRadius + shoulderOutward - shoulderMoat + moundOutward - sag;
    const lowerBound = Math.max(
      OUTER_HULL_MIN_RADIUS,
      ANCHOR_BASE_RADIUS
        - (0.44 + (0.24 * valleyStrength) + (0.14 * varianceScale))
        - (0.24 * normalizedShoulderRingLift),
    );
    const upperBound = Math.min(
      ANCHOR_MAX_RADIUS,
      Math.max(
        broadMembraneRadius + 0.54 + (0.18 * localSupport) + (0.1 * capBlend),
        weightedRadius + 0.22 + (0.14 * capBlend),
      ),
    );
    lowerBounds[index] = lowerBound;
    upperBounds[index] = upperBound;
    initialRadii[index] = Math.max(lowerBound, Math.min(upperBound, targetRadius));
  }

  let current = Float32Array.from(initialRadii);
  let next = new Float32Array(initialRadii.length);
  for (let pass = 0; pass < OUTER_HULL_RELAX_PASSES; pass += 1) {
    for (let index = 0; index < topology.vertexBindings.length; index += 1) {
      const binding = topology.vertexBindings[index];
      const neighbors = topology.adjacency[index];
      let neighborSum = 0;
      for (const neighborIndex of neighbors) {
        neighborSum += current[neighborIndex];
      }
      const neighborAverage = neighbors.length > 0 ? neighborSum / neighbors.length : current[index];
      const relaxed = lerp(current[index], neighborAverage, OUTER_HULL_RELAXATION_ALPHA);
      const capBlend = clamp01(binding.capBlend ?? 0);
      const restored = lerp(
        relaxed,
        initialRadii[index],
        0.06 + (0.03 * binding.maxInfluence) + (0.035 * capBlend),
      );
      next[index] = Math.max(lowerBounds[index], Math.min(upperBounds[index], restored));
    }
    [current, next] = [next, current];
  }

  return {
    anchorRadii,
    smoothedRadii: current,
  };
}

export function solveWorldVertices(bundle, topology, frameState, rotation) {
  const radiusField = computeRadiusField(bundle, topology, frameState);
  const worldVertices = topology.vertexBindings.map((binding, index) => (
    binding.direction
      .clone()
      .multiplyScalar(radiusField.smoothedRadii[index])
      .applyMatrix4(rotation)
  ));

  return {
    radiusField,
    worldVertices,
  };
}

export function updateTopologyNormals(topology, worldVertices) {
  const positions = topology.positionAttribute.array;
  for (let index = 0; index < worldVertices.length; index += 1) {
    const offset = index * 3;
    const vertex = worldVertices[index];
    positions[offset] = vertex.x;
    positions[offset + 1] = vertex.y;
    positions[offset + 2] = vertex.z;
  }
  topology.positionAttribute.needsUpdate = true;
  topology.geometry.computeVertexNormals();

  const normalsAttribute = topology.geometry.getAttribute('normal');
  return Array.from({ length: normalsAttribute.count }, (_, index) => (
    new Vector3().fromBufferAttribute(normalsAttribute, index).normalize()
  ));
}

export function buildFrameGeometry(bundle, topology, frameIndex) {
  const frameState = computeFrameState(bundle, frameIndex);
  const rotation = new Matrix4()
    .makeRotationY(frameState.rotationY)
    .multiply(new Matrix4().makeRotationX(frameState.rotationX));
  const { radiusField, worldVertices } = solveWorldVertices(bundle, topology, frameState, rotation);
  const vertexNormals = updateTopologyNormals(topology, worldVertices);
  return {
    frameState,
    rotation,
    radiusField,
    worldVertices,
    vertexNormals,
  };
}

export function projectPoint(vector, camera, width, height) {
  const projected = vector.clone().project(camera);
  return {
    x: (projected.x * 0.5 + 0.5) * width,
    y: (-projected.y * 0.5 + 0.5) * height,
    z: projected.z,
  };
}
