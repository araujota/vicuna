import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';
import {
  buildEmotiveAnimationRenderPlan,
  coerceNormalizedBundle,
} from './emotive-animation-scene.mjs';

const execFileAsync = promisify(execFile);
const serviceDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(serviceDir, '..', '..');
const NVENC_OPTION_CACHE = new Map();
const DEFAULT_PORT = 8091;
const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_MAX_CONCURRENT_RENDERS = 1;
const DEFAULT_GPU_MEMORY_BUDGET_MB = 1024;
const DEFAULT_TIMEOUT_MS = 300000;
const DEFAULT_STARTUP_TIMEOUT_MS = 15000;
const DEFAULT_TEMP_ROOT = path.join(repoRoot, 'var', 'webgl-renderer');

const env = {
  host: (process.env.VICUNA_WEBGL_RENDERER_HOST ?? DEFAULT_HOST).trim() || DEFAULT_HOST,
  port: Math.max(1, Number(process.env.VICUNA_WEBGL_RENDERER_PORT ?? DEFAULT_PORT) || DEFAULT_PORT),
  chromiumBin: '',
  ffmpegBin: '',
  videoEncoder: (process.env.VICUNA_WEBGL_RENDERER_FFMPEG_VIDEO_ENCODER ?? process.env.TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER ?? 'h264_nvenc').trim() || 'h264_nvenc',
  maxConcurrentRenders: Math.max(1, Math.round(Number(process.env.VICUNA_WEBGL_RENDERER_MAX_CONCURRENT_RENDERS ?? DEFAULT_MAX_CONCURRENT_RENDERS) || DEFAULT_MAX_CONCURRENT_RENDERS)),
  gpuMemoryBudgetMb: Math.max(128, Math.round(Number(process.env.VICUNA_WEBGL_RENDERER_GPU_MEMORY_BUDGET_MB ?? DEFAULT_GPU_MEMORY_BUDGET_MB) || DEFAULT_GPU_MEMORY_BUDGET_MB)),
  mandatoryGpu: String(process.env.VICUNA_WEBGL_RENDERER_MANDATORY_GPU ?? '1').trim() !== '0',
  timeoutMs: Math.max(1000, Math.round(Number(process.env.VICUNA_WEBGL_RENDERER_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS)),
  startupTimeoutMs: Math.max(
    1000,
    Math.min(
      Math.round(Number(process.env.VICUNA_WEBGL_RENDERER_STARTUP_TIMEOUT_MS ?? DEFAULT_STARTUP_TIMEOUT_MS) || DEFAULT_STARTUP_TIMEOUT_MS),
      Math.round(Number(process.env.VICUNA_WEBGL_RENDERER_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS),
    ),
  ),
  tempRoot: process.env.VICUNA_WEBGL_RENDERER_TEMP_ROOT ?? DEFAULT_TEMP_ROOT,
};

function log(message, extra = undefined) {
  const prefix = '[vicuna-webgl-renderer]';
  if (extra === undefined) {
    console.log(`${prefix} ${message}`);
    return;
  }
  console.log(`${prefix} ${message}`, extra);
}

function fail(message) {
  throw new Error(message);
}

function resolveBinary(envVarName, candidates, friendlyName) {
  const configured = String(process.env[envVarName] ?? '').trim();
  if (configured) {
    return configured;
  }
  for (const candidate of candidates) {
    if (candidate) {
      return candidate;
    }
  }
  fail(`could not resolve ${friendlyName}; set ${envVarName}`);
}

async function commandExists(command, args = ['--version']) {
  try {
    await execFileAsync(command, args, { maxBuffer: 1024 * 1024 });
    return true;
  } catch {
    return false;
  }
}

async function resolveChromiumBin() {
  const configured = String(process.env.VICUNA_WEBGL_RENDERER_CHROMIUM_BIN ?? process.env.TELEGRAM_BRIDGE_CHROMIUM_BIN ?? '').trim();
  const candidates = [
    configured,
    '/snap/bin/chromium',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/google-chrome-stable',
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (await commandExists(candidate)) {
      return candidate;
    }
  }
  fail('could not resolve Chromium; set VICUNA_WEBGL_RENDERER_CHROMIUM_BIN');
}

async function resolveFfmpegBin() {
  const configured = String(process.env.VICUNA_WEBGL_RENDERER_FFMPEG_BIN ?? process.env.TELEGRAM_BRIDGE_FFMPEG_BIN ?? '').trim();
  const candidates = [
    configured,
    '/usr/bin/ffmpeg',
    '/opt/homebrew/bin/ffmpeg',
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (await commandExists(candidate, ['-version'])) {
      return candidate;
    }
  }
  fail('could not resolve ffmpeg; set VICUNA_WEBGL_RENDERER_FFMPEG_BIN');
}

function buildChromiumArgs() {
  return [
    '--headless=new',
    '--allow-file-access-from-files',
    '--enable-gpu',
    '--use-angle=vulkan',
    '--enable-features=Vulkan',
    '--disable-vulkan-surface',
    '--ignore-gpu-blocklist',
    '--disable-software-rasterizer',
    '--disable-dev-shm-usage',
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--enable-unsafe-webgpu',
    '--window-size=720,720',
    '--hide-scrollbars',
    '--mute-audio',
    '--no-first-run',
    '--no-default-browser-check',
  ];
}

async function buildPageHtml() {
  const templatePath = path.join(serviceDir, 'emotive-webgl-renderer-page.html');
  const template = await readFile(templatePath, 'utf8');
  const threeModuleUrl = pathToFileURL(path.join(repoRoot, 'node_modules', 'three', 'build', 'three.module.js')).href;
  const threeAddonsBaseUrl = pathToFileURL(path.join(repoRoot, 'node_modules', 'three', 'examples', 'jsm')).href.replace(/\/?$/, '/');
  const sceneModuleUrl = pathToFileURL(path.join(serviceDir, 'emotive-animation-scene.mjs')).href;
  return template
    .replaceAll('__THREE_MODULE_URL__', threeModuleUrl)
    .replaceAll('__THREE_ADDONS_BASE_URL__', threeAddonsBaseUrl)
    .replaceAll('__SCENE_MODULE_URL__', sceneModuleUrl);
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
    helpText = `${error?.stdout ?? ''}\n${error?.stderr ?? ''}`;
  }

  const hasOption = (name) => new RegExp(`(^|\\n)\\s*-${name}(\\s|$)`, 'm').test(helpText);
  const args = [];
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

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  if (chunks.length === 0) {
    return {};
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function writeJson(res, statusCode, payload) {
  res.writeHead(statusCode, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(payload));
}

let browser = null;
let page = null;
let pagePath = '';
let browserProfilePath = '';
let browserVersion = '';
let rendererReady = false;
let lastStartupError = '';
let startupInFlight = null;
let startupRetryTimer = null;
let shuttingDown = false;
let gpuInfo = {
  vendor: '',
  renderer: '',
  software: true,
};
let activeRenders = 0;
let renderQueue = Promise.resolve();

async function startBrowser() {
  env.chromiumBin = await resolveChromiumBin();
  env.ffmpegBin = await resolveFfmpegBin();

  await mkdir(env.tempRoot, { recursive: true });
  browserProfilePath = path.join(env.tempRoot, 'chrome-profile');
  await rm(browserProfilePath, { recursive: true, force: true }).catch(() => {});
  const tempDir = await mkdtemp(path.join(env.tempRoot, 'vicuna-webgl-page-'));
  pagePath = path.join(tempDir, 'renderer.html');
  await writeFile(pagePath, await buildPageHtml());

  const launchArgs = buildChromiumArgs();
  browser = await puppeteer.launch({
    executablePath: env.chromiumBin,
    headless: true,
    protocolTimeout: env.startupTimeoutMs,
    userDataDir: browserProfilePath,
    args: launchArgs,
  });
  browserVersion = await browser.version();
  page = await browser.newPage();
  page.on('console', (message) => {
    const text = message.text();
    log(`page console [${message.type()}] ${text}`);
  });
  page.on('pageerror', (error) => {
    log('page error', {
      message: error?.message ?? String(error),
      stack: error?.stack ?? '',
    });
  });
  page.on('requestfailed', (request) => {
    log('page request failed', {
      url: request.url(),
      method: request.method(),
      failure: request.failure()?.errorText ?? 'unknown',
    });
  });
  await page.goto(pathToFileURL(pagePath).href, { waitUntil: 'load', timeout: env.timeoutMs });
  await page.waitForFunction(() => Boolean(window.vicunaWebglRenderer), { timeout: env.startupTimeoutMs });
  const health = await page.evaluate(() => window.vicunaWebglRenderer.ping());
  gpuInfo = health?.gpu ?? gpuInfo;
  rendererReady = Boolean(health?.ready);
  if (env.mandatoryGpu && (!rendererReady || gpuInfo.software)) {
    fail(`Chromium WebGL renderer is not using GPU hardware: ${JSON.stringify(gpuInfo)}`);
  }
  log('renderer browser ready', {
    chromiumBin: env.chromiumBin,
    ffmpegBin: env.ffmpegBin,
    videoEncoder: env.videoEncoder,
    browserVersion,
    gpuInfo,
    launchFlags: launchArgs,
  });
  lastStartupError = '';
}

async function stopBrowser() {
  if (browser) {
    await browser.close().catch(() => {});
  }
  browser = null;
  page = null;
  if (pagePath) {
    await rm(path.dirname(pagePath), { recursive: true, force: true }).catch(() => {});
    pagePath = '';
  }
  if (browserProfilePath) {
    await rm(browserProfilePath, { recursive: true, force: true }).catch(() => {});
    browserProfilePath = '';
  }
}

function scheduleBrowserStart(delayMs = 0) {
  if (shuttingDown || startupInFlight) {
    return startupInFlight;
  }
  if (startupRetryTimer) {
    clearTimeout(startupRetryTimer);
    startupRetryTimer = null;
  }
  const launch = async () => {
    rendererReady = false;
    browserVersion = '';
    gpuInfo = {
      vendor: '',
      renderer: '',
      software: true,
    };
    try {
      await stopBrowser();
      await startBrowser();
    } catch (error) {
      lastStartupError = String(error?.message ?? error);
      rendererReady = false;
      log('renderer startup failed', {
        error: lastStartupError,
        retryInMs: 2000,
      });
      await stopBrowser();
      if (!shuttingDown) {
        startupRetryTimer = setTimeout(() => {
          startupRetryTimer = null;
          scheduleBrowserStart();
        }, 2000);
      }
    } finally {
      startupInFlight = null;
    }
  };
  if (delayMs > 0) {
    startupInFlight = new Promise((resolve) => {
      startupRetryTimer = setTimeout(() => {
        startupRetryTimer = null;
        resolve(launch());
      }, delayMs);
    });
    return startupInFlight;
  }
  startupInFlight = launch();
  return startupInFlight;
}

async function renderBundle(rawBundle, outputPath = '') {
  const bundle = coerceNormalizedBundle(rawBundle);
  if (!bundle) {
    fail('emotive animation bundle was invalid');
  }
  if (!page) {
    fail('renderer browser is not ready');
  }

  const tempDir = await mkdtemp(path.join(env.tempRoot, 'vicuna-webgl-render-'));
  const frameDir = path.join(tempDir, 'frames');
  const finalOutputPath = outputPath
    ? path.resolve(String(outputPath))
    : path.join(tempDir, 'emotive-animation.mp4');

  try {
    await page.setViewport({
      width: bundle.viewportWidth,
      height: bundle.viewportHeight,
      deviceScaleFactor: 1,
    });
    await mkdir(frameDir, { recursive: true });
    const renderPlan = await page.evaluate((nextBundle) => window.vicunaWebglRenderer.prepare(nextBundle), bundle);
    const root = await page.$('#capture-root');
    if (!root) {
      fail('renderer page capture root was missing');
    }
    for (let frameIndex = 0; frameIndex < bundle.totalFrames; frameIndex += 1) {
      await page.evaluate((index) => window.vicunaWebglRenderer.renderFrame(index), frameIndex);
      await root.screenshot({ path: path.join(frameDir, `frame-${String(frameIndex).padStart(4, '0')}.png`) });
    }

    const ffmpegArgs = [
      '-y',
      '-framerate',
      String(bundle.fps),
      '-i',
      path.join(frameDir, 'frame-%04d.png'),
      '-c:v',
      env.videoEncoder,
    ];
    if (env.videoEncoder === 'h264_nvenc') {
      ffmpegArgs.push(...await resolveNvencEncoderArgs(env.ffmpegBin, env.videoEncoder));
    }
    ffmpegArgs.push(
      '-pix_fmt',
      'yuv420p',
      '-movflags',
      '+faststart',
      finalOutputPath,
    );
    await execFileAsync(env.ffmpegBin, ffmpegArgs, {
      maxBuffer: 32 * 1024 * 1024,
    });

    return {
      ...renderPlan,
      outputPath: finalOutputPath,
      tempDir,
      backend: 'chromium_webgl',
      chromiumBin: env.chromiumBin,
      ffmpegBin: env.ffmpegBin,
      videoEncoder: env.videoEncoder,
      gpu: gpuInfo,
      launchFlags: buildChromiumArgs(),
    };
  } catch (error) {
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

function enqueueRender(task) {
  const run = async () => {
    activeRenders += 1;
    try {
      return await task();
    } finally {
      activeRenders = Math.max(0, activeRenders - 1);
    }
  };
  const next = renderQueue.then(run, run);
  renderQueue = next.catch(() => {});
  return next;
}

async function refreshGpuInfoIfIdle() {
  if (!page || activeRenders > 0) {
    return;
  }
  try {
    const health = await page.evaluate(() => window.vicunaWebglRenderer.ping());
    if (health?.gpu) {
      gpuInfo = health.gpu;
    }
    rendererReady = Boolean(health?.ready);
  } catch (error) {
    log('gpu info refresh failed', {
      error: String(error?.message ?? error),
    });
  }
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === 'GET' && req.url === '/health') {
      await refreshGpuInfoIfIdle();
      writeJson(res, 200, {
        status: rendererReady && (!env.mandatoryGpu || !gpuInfo.software) ? 'ok' : 'degraded',
        backend: 'chromium_webgl',
        chromiumBin: env.chromiumBin,
        ffmpegBin: env.ffmpegBin,
        videoEncoder: env.videoEncoder,
        launchFlags: buildChromiumArgs(),
        browser: {
          ready: rendererReady,
          version: browserVersion,
          startupInFlight: Boolean(startupInFlight),
          startupError: lastStartupError,
          startupTimeoutMs: env.startupTimeoutMs,
        },
        gpu: gpuInfo,
        limits: {
          maxConcurrentRenders: env.maxConcurrentRenders,
          gpuMemoryBudgetMb: env.gpuMemoryBudgetMb,
          activeRenders,
        },
      });
      return;
    }

    if (req.method === 'POST' && req.url === '/render') {
      if (!rendererReady || !page) {
        writeJson(res, 503, {
          error: 'renderer_not_ready',
          startupError: lastStartupError,
        });
        return;
      }
      const body = await readJsonBody(req);
      const bundle = body?.bundle ?? body;
      const result = await enqueueRender(() => renderBundle(bundle, body?.outputPath ?? ''));
      writeJson(res, 200, result);
      return;
    }

    writeJson(res, 404, { error: 'not_found' });
  } catch (error) {
    writeJson(res, 500, {
      error: String(error?.message ?? error),
    });
  }
});

server.listen(env.port, env.host, () => {
  log(`listening on http://${env.host}:${env.port}`);
  scheduleBrowserStart();
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, async () => {
    shuttingDown = true;
    if (startupRetryTimer) {
      clearTimeout(startupRetryTimer);
      startupRetryTimer = null;
    }
    server.close();
    await stopBrowser();
    process.exit(0);
  });
}
