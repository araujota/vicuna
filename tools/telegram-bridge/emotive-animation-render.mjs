import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { createCanvas } from '@napi-rs/canvas';
import {
  Color,
  IcosahedronGeometry,
  Matrix4,
  PerspectiveCamera,
  Vector3,
} from 'three';

const execFileAsync = promisify(execFile);

const BASE_RADIUS = 1.25;
const DEFORMATION_AMPLITUDE = 0.75;
const LABEL_FONT = '600 18px sans-serif';
const LABEL_COLOR = '#102A43';
const BACKGROUND_START = '#F8F4EC';
const BACKGROUND_END = '#DDE9F4';
const SURFACE_LOW = new Color('#8FB8D8');
const SURFACE_HIGH = new Color('#E07A5F');
const WIREFRAME_COLOR = 'rgba(16, 42, 67, 0.18)';
const LIGHT_DIRECTION = new Vector3(0.45, 0.65, 1.0).normalize();

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function lerp(lhs, rhs, alpha) {
  return lhs + ((rhs - lhs) * alpha);
}

function easeInOutSine(alpha) {
  return 0.5 - (0.5 * Math.cos(Math.PI * clamp01(alpha)));
}

function normalizeVector3(values) {
  if (!Array.isArray(values) || values.length !== 3) {
    return null;
  }
  const vector = new Vector3(
    Number(values[0]) || 0,
    Number(values[1]) || 0,
    Number(values[2]) || 0,
  );
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
      const direction = normalizeVector3(dimension?.direction_xyz);
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
        : [];
      return {
        ordinal: Math.max(0, Number(keyframe?.ordinal ?? index) || index),
        traceBlockIndex: Math.max(0, Number(keyframe?.trace_block_index ?? index) || index),
        sourceKind: String(keyframe?.source_kind ?? '').trim() || 'runtime_event',
        moment,
        dominantDimensions,
      };
    }).filter(Boolean)
    : [];

  if (keyframes.length === 0) {
    return null;
  }

  const secondsPerKeyframe = Math.max(0.1, Number(raw.seconds_per_keyframe ?? 0.5) || 0.5);
  const fps = Math.max(1, Math.round(Number(raw.fps ?? 24) || 24));
  const viewportWidth = Math.max(256, Math.round(Number(raw.viewport_width ?? 720) || 720));
  const viewportHeight = Math.max(256, Math.round(Number(raw.viewport_height ?? 720) || 720));
  const rotationPeriodSeconds = Math.max(1, Number(raw.rotation_period_seconds ?? 12) || 12);
  const durationSeconds = secondsPerKeyframe * keyframes.length;
  const totalFrames = Math.max(1, Math.round(durationSeconds * fps));

  return {
    bundleVersion: Math.max(1, Number(raw.bundle_version ?? 1) || 1),
    traceId: String(raw.trace_id ?? '').trim(),
    generationStartBlockIndex: Math.max(0, Number(raw.generation_start_block_index ?? 0) || 0),
    secondsPerKeyframe,
    fps,
    viewportWidth,
    viewportHeight,
    rotationPeriodSeconds,
    durationSeconds,
    totalFrames,
    dimensions,
    keyframes,
  };
}

function coerceNormalizedBundle(rawBundle) {
  if (
    rawBundle
    && typeof rawBundle === 'object'
    && !Array.isArray(rawBundle)
    && Array.isArray(rawBundle.dimensions)
    && Array.isArray(rawBundle.keyframes)
    && Number(rawBundle.totalFrames ?? 0) > 0
    && Number(rawBundle.durationSeconds ?? 0) > 0
    && Number(rawBundle.secondsPerKeyframe ?? 0) > 0
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
    keyframeCount: bundle.keyframes.length,
    totalFrames: bundle.totalFrames,
    durationSeconds: bundle.durationSeconds,
    fps: bundle.fps,
    viewportWidth: bundle.viewportWidth,
    viewportHeight: bundle.viewportHeight,
  };
}

function computeFrameState(bundle, frameIndex) {
  const duration = bundle.durationSeconds;
  const t = bundle.totalFrames <= 1
    ? 0
    : (frameIndex / Math.max(1, bundle.totalFrames - 1)) * duration;
  const segmentIndex = Math.min(
    bundle.keyframes.length - 1,
    Math.max(0, Math.floor(t / bundle.secondsPerKeyframe)),
  );
  const nextSegmentIndex = Math.min(bundle.keyframes.length - 1, segmentIndex + 1);
  const segmentStart = segmentIndex * bundle.secondsPerKeyframe;
  const localAlpha = nextSegmentIndex === segmentIndex
    ? 0
    : easeInOutSine((t - segmentStart) / bundle.secondsPerKeyframe);

  const magnitudes = {};
  for (const dimension of bundle.dimensions) {
    const current = bundle.keyframes[segmentIndex].moment[dimension.id] ?? 0;
    const next = bundle.keyframes[nextSegmentIndex].moment[dimension.id] ?? current;
    magnitudes[dimension.id] = lerp(current, next, localAlpha);
  }

  return {
    t,
    magnitudes,
    rotationY: (2 * Math.PI * t) / bundle.rotationPeriodSeconds,
    rotationX: 0.18 * Math.sin((2 * Math.PI * t) / (bundle.rotationPeriodSeconds * 1.7)),
  };
}

function buildGeometryCache(bundle) {
  const geometry = new IcosahedronGeometry(1, 3);
  const positions = geometry.getAttribute('position');
  const indices = geometry.index
    ? Array.from(geometry.index.array)
    : Array.from({ length: positions.count }, (_, index) => index);
  const baseVertices = [];
  const weights = [];

  for (let index = 0; index < positions.count; index += 1) {
    const base = new Vector3().fromBufferAttribute(positions, index).normalize();
    baseVertices.push(base);

    const vertexWeights = bundle.dimensions.map((dimension) => {
      const influence = Math.max(0, base.dot(dimension.direction) + 0.35);
      return influence * influence;
    });
    const weightSum = vertexWeights.reduce((sum, value) => sum + value, 0);
    weights.push(weightSum > 0
      ? vertexWeights.map((value) => value / weightSum)
      : vertexWeights.map(() => 1 / bundle.dimensions.length));
  }

  geometry.dispose();
  return { baseVertices, indices, weights };
}

function drawBackground(ctx, width, height) {
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, BACKGROUND_START);
  gradient.addColorStop(1, BACKGROUND_END);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  const glow = ctx.createRadialGradient(width * 0.25, height * 0.18, 20, width * 0.25, height * 0.18, width * 0.75);
  glow.addColorStop(0, 'rgba(255, 255, 255, 0.72)');
  glow.addColorStop(1, 'rgba(255, 255, 255, 0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);
}

function colorForMagnitude(magnitude) {
  return SURFACE_LOW.clone().lerp(SURFACE_HIGH, clamp01(magnitude));
}

function projectPoint(vector, camera, width, height) {
  const projected = vector.clone().project(camera);
  return {
    x: (projected.x * 0.5 + 0.5) * width,
    y: (-projected.y * 0.5 + 0.5) * height,
    z: projected.z,
  };
}

function renderFrame(canvas, bundle, geometryCache, frameIndex) {
  const ctx = canvas.getContext('2d');
  const width = bundle.viewportWidth;
  const height = bundle.viewportHeight;
  const frameState = computeFrameState(bundle, frameIndex);
  drawBackground(ctx, width, height);

  const camera = new PerspectiveCamera(34, width / height, 0.1, 100);
  camera.position.set(0, 0, 5.1);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  const rotation = new Matrix4()
    .makeRotationY(frameState.rotationY)
    .multiply(new Matrix4().makeRotationX(frameState.rotationX));

  const worldVertices = geometryCache.baseVertices.map((baseVertex, vertexIndex) => {
    let weightedMagnitude = 0;
    const vertexWeights = geometryCache.weights[vertexIndex];
    for (let weightIndex = 0; weightIndex < vertexWeights.length; weightIndex += 1) {
      weightedMagnitude += vertexWeights[weightIndex] * frameState.magnitudes[bundle.dimensions[weightIndex].id];
    }
    return baseVertex
      .clone()
      .multiplyScalar(BASE_RADIUS + (DEFORMATION_AMPLITUDE * weightedMagnitude))
      .applyMatrix4(rotation);
  });

  const projectedVertices = worldVertices.map((vertex) => projectPoint(vertex, camera, width, height));
  const triangles = [];

  for (let index = 0; index < geometryCache.indices.length; index += 3) {
    const ai = geometryCache.indices[index];
    const bi = geometryCache.indices[index + 1];
    const ci = geometryCache.indices[index + 2];
    const a = worldVertices[ai];
    const b = worldVertices[bi];
    const c = worldVertices[ci];
    const viewA = a.clone().applyMatrix4(camera.matrixWorldInverse);
    const viewB = b.clone().applyMatrix4(camera.matrixWorldInverse);
    const viewC = c.clone().applyMatrix4(camera.matrixWorldInverse);
    const normal = viewC.clone().sub(viewA).cross(viewB.clone().sub(viewA)).normalize();
    if (normal.z <= 0) {
      continue;
    }
    const averageDepth = (viewA.z + viewB.z + viewC.z) / 3;
    const averageMagnitude = (
      projectedVertices[ai].z +
      projectedVertices[bi].z +
      projectedVertices[ci].z
    );
    triangles.push({
      ai,
      bi,
      ci,
      averageDepth,
      normal,
      averageMagnitude,
    });
  }

  triangles.sort((lhs, rhs) => lhs.averageDepth - rhs.averageDepth);
  ctx.lineWidth = 1;
  ctx.strokeStyle = WIREFRAME_COLOR;

  for (const triangle of triangles) {
    const pa = projectedVertices[triangle.ai];
    const pb = projectedVertices[triangle.bi];
    const pc = projectedVertices[triangle.ci];
    const brightness = Math.max(0.2, triangle.normal.dot(LIGHT_DIRECTION));
    const faceMagnitude = clamp01(1 - ((triangle.averageMagnitude / 3) + 1) * 0.5);
    const fill = colorForMagnitude(faceMagnitude).multiplyScalar(0.7 + (0.45 * brightness));

    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.lineTo(pc.x, pc.y);
    ctx.closePath();
    ctx.fillStyle = `rgba(${Math.round(fill.r * 255)}, ${Math.round(fill.g * 255)}, ${Math.round(fill.b * 255)}, 0.94)`;
    ctx.fill();
    ctx.stroke();
  }

  ctx.font = LABEL_FONT;
  ctx.textBaseline = 'middle';
  for (const dimension of bundle.dimensions) {
    const magnitude = frameState.magnitudes[dimension.id] ?? 0;
    const point = dimension.direction
      .clone()
      .multiplyScalar(BASE_RADIUS + (DEFORMATION_AMPLITUDE * magnitude))
      .applyMatrix4(rotation);
    const viewPoint = point.clone().applyMatrix4(camera.matrixWorldInverse);
    const projected = projectPoint(point, camera, width, height);
    const side = projected.x >= width / 2 ? 1 : -1;
    const labelOffset = 26;
    const textX = projected.x + (side * labelOffset);
    const textY = projected.y;
    const alpha = Math.max(0.28, Math.min(1, 1.18 - Math.max(0, viewPoint.z + 1.8) * 0.18));

    ctx.strokeStyle = `rgba(16, 42, 67, ${0.18 * alpha})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(projected.x, projected.y);
    ctx.lineTo(textX - (side * 6), textY);
    ctx.stroke();

    ctx.fillStyle = `rgba(255, 255, 255, ${0.82 * alpha})`;
    const textWidth = ctx.measureText(dimension.label).width;
    const boxWidth = textWidth + 14;
    const boxX = side > 0 ? textX - 4 : textX - boxWidth + 4;
    ctx.fillRect(boxX, textY - 12, boxWidth, 24);

    ctx.fillStyle = `rgba(16, 42, 67, ${alpha})`;
    ctx.textAlign = side > 0 ? 'left' : 'right';
    ctx.fillText(dimension.label, textX, textY);

    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 3.5 + (magnitude * 2), 0, Math.PI * 2);
    ctx.fillStyle = `rgba(224, 122, 95, ${0.75 * alpha})`;
    ctx.fill();
  }

  return frameState;
}

export async function renderEmotiveAnimationFrames(rawBundle, frameDir) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    throw new Error('emotive animation bundle was invalid');
  }

  await mkdir(frameDir, { recursive: true });
  const geometryCache = buildGeometryCache(bundle);
  const canvas = createCanvas(bundle.viewportWidth, bundle.viewportHeight);

  for (let frameIndex = 0; frameIndex < bundle.totalFrames; frameIndex += 1) {
    renderFrame(canvas, bundle, geometryCache, frameIndex);
    const framePath = path.join(frameDir, `frame-${String(frameIndex).padStart(4, '0')}.png`);
    await writeFile(framePath, canvas.toBuffer('image/png'));
  }

  return {
    ...buildEmotiveAnimationRenderPlan(bundle),
    bundle,
    frameDir,
  };
}

export async function renderEmotiveAnimationMp4(rawBundle, options = {}) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    throw new Error('emotive animation bundle was invalid');
  }

  const ffmpegBin = String(options.ffmpegBin ?? 'ffmpeg').trim() || 'ffmpeg';
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'vicuna-emotive-animation-'));
  const frameDir = path.join(tempDir, 'frames');
  const outputPath = options.outputPath
    ? path.resolve(String(options.outputPath))
    : path.join(tempDir, 'emotive-animation.mp4');

  try {
    let renderResult;
    try {
      renderResult = await renderEmotiveAnimationFrames(bundle, frameDir);
    } catch (error) {
      error.stage = 'render';
      throw error;
    }
    try {
      await execFileAsync(ffmpegBin, [
        '-y',
        '-framerate',
        String(bundle.fps),
        '-i',
        path.join(frameDir, 'frame-%04d.png'),
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        outputPath,
      ], {
        maxBuffer: 32 * 1024 * 1024,
      });
    } catch (error) {
      error.stage = 'encode';
      throw error;
    }

    return {
      ...renderResult,
      outputPath,
    };
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}
