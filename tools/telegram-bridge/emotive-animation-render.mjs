import os from 'node:os';
import path from 'node:path';
import { execFile, spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { createCanvas } from '@napi-rs/canvas';
import {
  PerspectiveCamera,
  Color,
} from 'three';
import {
  BACKGROUND_COLOR,
  computeBundleRadiusNormalizationRange,
  LIGHT_DIRECTION,
  buildEmotiveAnimationRenderPlan,
  buildFrameGeometry,
  buildOuterHullTopology,
  clamp01,
  coerceNormalizedBundle,
  colorForNormalizedDistance,
  extractEmotiveAnimationTerminalMoment,
  prependEmotiveAnimationStartMoment,
  projectPoint,
  normalizeEmotiveAnimationBundle,
  supportedAnchorRadiusForMagnitude,
} from './emotive-animation-scene.mjs';
export {
  buildEmotiveAnimationRenderPlan,
  extractEmotiveAnimationTerminalMoment,
  prependEmotiveAnimationStartMoment,
  normalizeEmotiveAnimationBundle,
} from './emotive-animation-scene.mjs';

const execFileAsync = promisify(execFile);
const NVENC_OPTION_CACHE = new Map();

const LABEL_FONT = '600 15px "Helvetica Neue", sans-serif';
const LABEL_LINE_COLOR = 'rgba(94, 255, 242, 0.36)';
const LABEL_GLOW_COLOR = 'rgba(0, 216, 255, 0.85)';

function rgbaFromColor(color, alpha) {
  return `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, ${alpha})`;
}
function drawBackground(ctx, width, height) {
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, width, height);

  const halo = ctx.createRadialGradient(width * 0.5, height * 0.45, 0, width * 0.5, height * 0.45, width * 0.42);
  halo.addColorStop(0, 'rgba(0, 175, 255, 0.09)');
  halo.addColorStop(0.55, 'rgba(0, 92, 255, 0.04)');
  halo.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, width, height);
}

async function resolveNvencEncoderArgs(ffmpegBin, videoEncoder) {
  const cacheKey = `${ffmpegBin}::${videoEncoder}`;
  if (NVENC_OPTION_CACHE.has(cacheKey)) {
    return NVENC_OPTION_CACHE.get(cacheKey);
  }

  let helpText = '';
  try {
    const { stdout = '', stderr = '' } = await execFileAsync(ffmpegBin, [
      '-hide_banner',
      '-h',
      `encoder=${videoEncoder}`,
    ], {
      maxBuffer: 4 * 1024 * 1024,
    });
    helpText = `${stdout}\n${stderr}`;
  } catch (error) {
    const stdout = typeof error?.stdout === 'string' ? error.stdout : '';
    const stderr = typeof error?.stderr === 'string' ? error.stderr : '';
    helpText = `${stdout}\n${stderr}`;
  }

  const args = [];
  const hasOption = (name) => new RegExp(`(^|\\n)\\s*-${name}(\\s|$)`, 'm').test(helpText);

  if (hasOption('preset')) {
    args.push('-preset', 'p6');
  }
  if (hasOption('tune')) {
    args.push('-tune', 'hq');
  }
  if (hasOption('rc')) {
    args.push('-rc', 'vbr');
  }
  if (hasOption('cq')) {
    args.push('-cq', '21');
  }
  if (hasOption('b')) {
    args.push('-b:v', '0');
  }
  if (hasOption('profile')) {
    args.push('-profile:v', 'high');
  }

  NVENC_OPTION_CACHE.set(cacheKey, args);
  return args;
}

async function writeChunk(stream, chunk) {
  if (stream.destroyed || !stream.writable) {
    throw new Error('encoder stdin is not writable');
  }
  if (stream.write(chunk)) {
    return;
  }
  await once(stream, 'drain');
}

async function startStreamingEncoder(ffmpegBin, videoEncoder, bundle, outputPath) {
  const ffmpegArgs = [
    '-y',
    '-f',
    'image2pipe',
    '-vcodec',
    'png',
    '-framerate',
    String(bundle.fps),
    '-i',
    '-',
    '-c:v',
    videoEncoder,
  ];
  if (videoEncoder === 'h264_nvenc') {
    ffmpegArgs.push(...await resolveNvencEncoderArgs(ffmpegBin, videoEncoder));
  }
  ffmpegArgs.push(
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+faststart',
    outputPath,
  );
  const child = spawn(ffmpegBin, ffmpegArgs, {
    stdio: ['pipe', 'ignore', 'pipe'],
  });
  let stderr = '';
  child.stderr?.on('data', (chunk) => {
    stderr += String(chunk);
    if (stderr.length > 16384) {
      stderr = stderr.slice(-16384);
    }
  });
  const completion = new Promise((resolve, reject) => {
    child.once('error', (error) => {
      error.stage = error.stage || 'encode';
      reject(error);
    });
    child.once('close', (code, signal) => {
      if (code === 0) {
        resolve({ stderr });
        return;
      }
      const error = new Error(
        `ffmpeg exited with code ${code ?? 'null'}${signal ? ` signal ${signal}` : ''}: ${stderr.trim() || 'no stderr'}`,
      );
      error.stage = 'finalize';
      reject(error);
    });
  });
  return {
    child,
    completion,
  };
}

function renderFrame(canvas, surfaceCanvas, bundle, topology, frameIndex) {
  const ctx = canvas.getContext('2d');
  const surfaceCtx = surfaceCanvas.getContext('2d');
  const width = bundle.viewportWidth;
  const height = bundle.viewportHeight;
  drawBackground(ctx, width, height);
  surfaceCtx.clearRect(0, 0, width, height);

  const camera = new PerspectiveCamera(34, width / height, 0.1, 100);
  camera.position.set(0, 0, 5.1);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  const { frameState, rotation, worldVertices, vertexNormals } = buildFrameGeometry(bundle, topology, frameIndex);
  const projectedVertices = worldVertices.map((vertex) => projectPoint(vertex, camera, width, height));
  const radiusRange = computeBundleRadiusNormalizationRange(bundle, topology);
  const minRadius = radiusRange?.minRadius ?? 0;
  const maxRadius = radiusRange?.maxRadius ?? 1;
  const radiusSpan = Math.max(1e-6, maxRadius - minRadius);
  const vertexLights = vertexNormals.map((normal) => {
    const diffuse = Math.max(0, normal.dot(LIGHT_DIRECTION));
    const rim = Math.pow(1 - Math.max(0, normal.z), 2);
    return 0.92 + (0.05 * diffuse) + (0.03 * rim);
  });
  const triangles = [];

  for (let index = 0; index < topology.indices.length; index += 3) {
    const ai = topology.indices[index];
    const bi = topology.indices[index + 1];
    const ci = topology.indices[index + 2];
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
    const averageRadius = (a.length() + b.length() + c.length()) / 3;
    triangles.push({
      ai,
      bi,
      ci,
      averageDepth,
      averageRadius,
      averageLight: (vertexLights[ai] + vertexLights[bi] + vertexLights[ci]) / 3,
    });
  }

  triangles.sort((lhs, rhs) => lhs.averageDepth - rhs.averageDepth);
  surfaceCtx.lineJoin = 'round';
  surfaceCtx.lineCap = 'round';

  for (const triangle of triangles) {
    const pa = projectedVertices[triangle.ai];
    const pb = projectedVertices[triangle.bi];
    const pc = projectedVertices[triangle.ci];
    const distanceAlpha = clamp01((triangle.averageRadius - minRadius) / radiusSpan);
    const fill = colorForNormalizedDistance(distanceAlpha).multiplyScalar(triangle.averageLight);

    surfaceCtx.beginPath();
    surfaceCtx.moveTo(pa.x, pa.y);
    surfaceCtx.lineTo(pb.x, pb.y);
    surfaceCtx.lineTo(pc.x, pc.y);
    surfaceCtx.closePath();
    const fillStyle = `rgba(${Math.round(fill.r * 255)}, ${Math.round(fill.g * 255)}, ${Math.round(fill.b * 255)}, 1)`;
    surfaceCtx.fillStyle = fillStyle;
    surfaceCtx.fill();
    surfaceCtx.strokeStyle = fillStyle;
    surfaceCtx.lineWidth = 0.5;
    surfaceCtx.stroke();
  }

  ctx.save();
  ctx.filter = 'blur(8px)';
  ctx.globalAlpha = 0.34;
  ctx.drawImage(surfaceCanvas, 0, 0);
  ctx.restore();
  ctx.drawImage(surfaceCanvas, 0, 0);

  ctx.font = LABEL_FONT;
  ctx.textBaseline = 'middle';
  for (const dimension of bundle.dimensions) {
    const magnitude = frameState.magnitudes[dimension.id] ?? 0;
    const point = dimension.direction
      .clone()
      .multiplyScalar(supportedAnchorRadiusForMagnitude(magnitude))
      .applyMatrix4(rotation);
    const viewPoint = point.clone().applyMatrix4(camera.matrixWorldInverse);
    const projected = projectPoint(point, camera, width, height);
    const side = projected.x >= width / 2 ? 1 : -1;
    const labelOffset = 26;
    const textX = projected.x + (side * labelOffset);
    const textY = projected.y;
    const alpha = Math.max(0.28, Math.min(1, 1.18 - Math.max(0, viewPoint.z + 1.8) * 0.18));

    ctx.save();
    ctx.strokeStyle = LABEL_LINE_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(projected.x, projected.y);
    ctx.lineTo(textX - (side * 6), textY);
    ctx.stroke();
    ctx.shadowBlur = 14;
    ctx.shadowColor = LABEL_GLOW_COLOR;
    ctx.fillStyle = `rgba(230, 251, 255, ${alpha})`;
    ctx.textAlign = side > 0 ? 'left' : 'right';
    ctx.fillText(dimension.label, textX, textY);
    ctx.restore();

    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 3.5 + (magnitude * 2), 0, Math.PI * 2);
    ctx.fillStyle = rgbaFromColor(new Color('#7dfcff'), 0.35 + (0.45 * alpha));
    ctx.fill();
  }

  return {
    ...frameState,
    scene: {
      background: BACKGROUND_COLOR,
      anchorCount: topology.anchorCount,
      vertexCount: topology.vertexCount,
      triangleCount: topology.triangleCount,
    },
  };
}

export async function renderEmotiveAnimationFrames(rawBundle, frameDir) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    throw new Error('emotive animation bundle was invalid');
  }

  await mkdir(frameDir, { recursive: true });
  const topology = buildOuterHullTopology(bundle);
  const canvas = createCanvas(bundle.viewportWidth, bundle.viewportHeight);
  const surfaceCanvas = createCanvas(bundle.viewportWidth, bundle.viewportHeight);
  let lastFrameState = null;

  for (let frameIndex = 0; frameIndex < bundle.totalFrames; frameIndex += 1) {
    lastFrameState = renderFrame(canvas, surfaceCanvas, bundle, topology, frameIndex);
    const framePath = path.join(frameDir, `frame-${String(frameIndex).padStart(4, '0')}.png`);
    await writeFile(framePath, canvas.toBuffer('image/png'));
  }

  return {
    ...buildEmotiveAnimationRenderPlan(bundle),
    bundle,
    frameDir,
    topology: {
      anchorCount: topology.anchorCount,
      vertexCount: topology.vertexCount,
      triangleCount: topology.triangleCount,
      subdivisions: topology.subdivisions,
    },
    scene: lastFrameState?.scene ?? {
      background: BACKGROUND_COLOR,
      anchorCount: topology.anchorCount,
      vertexCount: topology.vertexCount,
      triangleCount: topology.triangleCount,
    },
  };
}

export async function renderEmotiveAnimationMp4(rawBundle, options = {}) {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    throw new Error('emotive animation bundle was invalid');
  }

  const ffmpegBin = String(options.ffmpegBin ?? 'ffmpeg').trim() || 'ffmpeg';
  const videoEncoder = String(options.videoEncoder ?? 'h264_nvenc').trim() || 'h264_nvenc';
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'vicuna-emotive-animation-'));
  const outputPath = options.outputPath
    ? path.resolve(String(options.outputPath))
    : path.join(tempDir, 'emotive-animation.mp4');

  try {
    const topology = buildOuterHullTopology(bundle);
    const canvas = createCanvas(bundle.viewportWidth, bundle.viewportHeight);
    const surfaceCanvas = createCanvas(bundle.viewportWidth, bundle.viewportHeight);
    let lastFrameState = null;
    let encoder = null;
    try {
      encoder = await startStreamingEncoder(ffmpegBin, videoEncoder, bundle, outputPath);
      for (let frameIndex = 0; frameIndex < bundle.totalFrames; frameIndex += 1) {
        lastFrameState = renderFrame(canvas, surfaceCanvas, bundle, topology, frameIndex);
        await writeChunk(encoder.child.stdin, canvas.toBuffer('image/png'));
      }
      encoder.child.stdin.end();
      await encoder.completion;
    } catch (error) {
      error.stage = error.stage || 'render';
      encoder?.child?.stdin?.destroy?.();
      encoder?.child?.kill?.('SIGKILL');
      throw error;
    }

    return {
      ...buildEmotiveAnimationRenderPlan(bundle),
      bundle,
      topology: {
        anchorCount: topology.anchorCount,
        vertexCount: topology.vertexCount,
        triangleCount: topology.triangleCount,
        subdivisions: topology.subdivisions,
      },
      scene: lastFrameState?.scene ?? {
        background: BACKGROUND_COLOR,
        anchorCount: topology.anchorCount,
        vertexCount: topology.vertexCount,
        triangleCount: topology.triangleCount,
      },
      pipeline: 'streamed_image2pipe_png',
      outputPath,
    };
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}
