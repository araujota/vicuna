import { execFile } from 'node:child_process';
import path from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

function parseJsonOutput(stdout) {
  const text = String(stdout ?? '').trim();
  if (!text) {
    throw new Error('empty helper response');
  }
  return JSON.parse(text);
}

function buildToggleReply(payload) {
  const currentMode = String(payload?.current_mode ?? '').trim() || 'unknown';
  const previousMode = String(payload?.previous_mode ?? '').trim() || 'unknown';
  const runtimeService = String(payload?.runtime_service ?? '').trim();
  const podAction = String(payload?.pod_action ?? '').trim();
  const podId = String(payload?.pod_id ?? '').trim();
  const tunnelPort = Number(payload?.tunnel_local_port ?? 0) || 0;

  const lines = [
    `Inference mode switched: ${previousMode} -> ${currentMode}.`,
  ];
  if (podAction) {
    lines.push(`Pod action: ${podAction}${podId ? ` (${podId})` : ''}.`);
  }
  if (currentMode === 'experimental' && tunnelPort > 0) {
    lines.push(`Relay tunnel listening on 127.0.0.1:${tunnelPort}.`);
  }
  if (runtimeService) {
    lines.push(`Runtime service: ${runtimeService}.`);
  }
  return lines.join('\n');
}

export async function handleBridgeControlCommand({
  repoRoot,
  text,
  message,
  sendTelegramMessage,
  execFileAsyncImpl = execFileAsync,
}) {
  const normalized = String(text ?? '').trim();
  if (normalized !== '/togglemode') {
    return false;
  }
  if (!message?.chat?.id || typeof sendTelegramMessage !== 'function') {
    throw new Error('togglemode requires a chat id and sendTelegramMessage callback');
  }

  const helperPath = path.join(repoRoot, 'tools/ops/host-inference-mode.sh');
  try {
    const { stdout } = await execFileAsyncImpl('sudo', ['-n', helperPath, 'toggle', '--json'], {
      env: {
        ...process.env,
        REPO_ROOT: repoRoot,
      },
    });
    const payload = parseJsonOutput(stdout);
    await sendTelegramMessage(
      message.chat.id,
      buildToggleReply(payload),
      { reply_to_message_id: message.message_id },
    );
  } catch (error) {
    const reason = String(error?.message ?? error).trim() || 'unknown error';
    await sendTelegramMessage(
      message.chat.id,
      `I could not toggle inference mode: ${reason}`,
      { reply_to_message_id: message.message_id },
    );
  }
  return true;
}
