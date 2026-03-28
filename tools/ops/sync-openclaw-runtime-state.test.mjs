import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { access, cp, mkdtemp, rm } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

test('sync-openclaw-runtime-state bootstraps a missing harness dist entrypoint', { timeout: 180000 }, async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'vicuna-openclaw-sync-'));
  const tempRepo = path.join(tempRoot, 'repo');
  const tempSecrets = path.join(tempRoot, 'openclaw-tool-secrets.json');
  const tempCatalog = path.join(tempRoot, 'openclaw-catalog.json');
  const tempEnvFile = path.join(tempRoot, 'missing-vicuna.env');

  try {
    await cp(path.join(REPO_ROOT, 'tools', 'ops'), path.join(tempRepo, 'tools', 'ops'), { recursive: true });
    await cp(path.join(REPO_ROOT, 'tools', 'openclaw-harness'), path.join(tempRepo, 'tools', 'openclaw-harness'), { recursive: true });
    await rm(path.join(tempRepo, 'tools', 'openclaw-harness', 'dist'), { recursive: true, force: true });
    await rm(path.join(tempRepo, 'tools', 'openclaw-harness', 'node_modules'), { recursive: true, force: true });

    await execFileAsync('bash', [path.join(tempRepo, 'tools', 'ops', 'sync-openclaw-runtime-state.sh')], {
      cwd: tempRepo,
      env: {
        ...process.env,
        REPO_ROOT: tempRepo,
        VICUNA_SYSTEM_ENV_FILE: tempEnvFile,
        VICUNA_OPENCLAW_NODE_BIN: process.execPath,
        VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH: tempSecrets,
        VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH: tempCatalog,
      },
      maxBuffer: 32 * 1024 * 1024,
    });

    const builtEntrypoint = path.join(tempRepo, 'tools', 'openclaw-harness', 'dist', 'index.js');
    await access(builtEntrypoint, fsConstants.R_OK);
    await access(tempSecrets, fsConstants.R_OK);
  } finally {
    await rm(tempRoot, { recursive: true, force: true });
  }
});
