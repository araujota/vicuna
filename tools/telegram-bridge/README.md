# Telegram Bridge

This bridge connects a Telegram bot to the local Vicuña server.

It does two jobs in one process:

- polls Telegram for inbound text messages and forwards them to
  `/v1/chat/completions`
- subscribes to `/v1/responses/stream` and relays proactive self-emits to every
  registered Telegram chat
- polls `/v1/telegram/outbox` for runtime-owned `ask_with_options` prompts and
  approval-gate prompts, then delivers them with Telegram
  `reply_markup.inline_keyboard`

## Environment

Load the repo env file first:

```bash
source ./.envrc
```

The bridge requires Node `>=20.16.0`. The managed launcher resolves a suitable
Node binary automatically, preferring:

- `TELEGRAM_BRIDGE_NODE_BIN` when explicitly set
- the current `node` on `PATH` if it is new enough
- the user's `nvm` Node 20 installation when available

Required variables:

- `TELEGRAM_BOT_TOKEN`

Optional variables:

- `TELEGRAM_BRIDGE_VICUNA_BASE_URL` default: `http://127.0.0.1:8080`
- `TELEGRAM_BRIDGE_MODEL` default: `vicuna-runtime`
- `TELEGRAM_BRIDGE_STATE_PATH` default: `/tmp/vicuna-telegram-bridge-state.json`
- `TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS` default: `30`
- `TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES` default: `12`
- `TELEGRAM_BRIDGE_MAX_TOKENS` default: `-1` (unlimited)
- `TELEGRAM_BRIDGE_MAX_DOCUMENT_CHARS` default: `12000`
- `TELEGRAM_BRIDGE_REPLAY_RETAINED_OUTBOX` default: `0`
  - when `0`, a fresh or reset bridge state fast-forwards to the newest
    retained runtime outbox sequence instead of replaying historical follow-ups
  - set to `1` only for an explicit operator replay of retained outbox items
- `SUPERMEMORY_API_KEY` required for PDF/DOC/DOCX ingestion and linked
  Supermemory persistence
- `VICUNA_API_KEY` if the server runs with bearer auth enabled

## Document Ingestion

Supported inbound Telegram document formats:

- PDF
- DOC
- DOCX

For each supported Telegram document, the bridge:

1. downloads the raw file through Telegram `getFile`
2. extracts plain text only
3. stores the raw file in Supermemory
4. stores the extracted text in Supermemory as a second linked document
5. appends the normalized extracted text to the Telegram chat transcript before
   forwarding the turn to Vicuña

Extraction policy:

- PDF uses the local Node dependency `pdf-parse` and loads it lazily only when
  a PDF message arrives
- DOC and DOCX use the host's `/usr/bin/textutil` converter

If `SUPERMEMORY_API_KEY` is missing, supported document ingestion is rejected.
If `/usr/bin/textutil` is missing, DOC and DOCX ingestion fails with a direct
host requirement error.

## Start

```bash
npm run telegram-bridge:start
```

## Managed Service

For resilient host operation, use the repo-owned user service units:

- [vicuna-runtime.service](/home/tyler-araujo/Projects/vicuna/tools/ops/systemd/vicuna-runtime.service)
- [vicuna-telegram-bridge.service](/home/tyler-araujo/Projects/vicuna/tools/ops/systemd/vicuna-telegram-bridge.service)

The launchers they use are:

- [run-vicuna-runtime.sh](/home/tyler-araujo/Projects/vicuna/tools/ops/run-vicuna-runtime.sh)
- [run-telegram-bridge.sh](/home/tyler-araujo/Projects/vicuna/tools/ops/run-telegram-bridge.sh)

The managed runtime launcher now owns an explicit reasoning GGUF instead of an
Ollama blob path. By default it expects:

- `VICUNA_RUNTIME_MODEL_PATH`:
  `/home/tyler-araujo/Projects/vicuna/models/runtime/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf`
- `VICUNA_RUNTIME_MODEL_URL`:
  [bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF)
- `VICUNA_RUNTIME_MODEL_CHAT_TEMPLATE_FILE`:
  [deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja](/Users/tyleraraujo/vicuna/models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja)
- `VICUNA_RUNTIME_MODEL_REASONING_FORMAT`: `deepseek`

If the model file is missing, the launcher fetches it through
[fetch-runtime-model.sh](/Users/tyleraraujo/vicuna/tools/ops/fetch-runtime-model.sh)
before starting `llama-server`.

The runtime launcher now enables the repo-owned cognitive bash tool path by
default for managed operation. Override these if needed:

- `VICUNA_BASH_TOOL_ENABLED` default: `1`
- `VICUNA_BASH_TOOL_PATH` default: `$(command -v bash)` from the launcher host
- `VICUNA_BASH_TOOL_WORKDIR` default: repo root
- `VICUNA_BASH_TOOL_TIMEOUT_MS` default: `15000`
- `VICUNA_BASH_TOOL_MAX_STDOUT_BYTES` default: `16384`
- `VICUNA_BASH_TOOL_MAX_STDERR_BYTES` default: `8192`
- `VICUNA_BASH_TOOL_LOGIN_SHELL` default: `0`
- `VICUNA_BASH_TOOL_INHERIT_ENV` default: `1`
- `VICUNA_BASH_TOOL_MAX_CHILD_PROCESSES` default: `4096`

`VICUNA_BASH_TOOL_LOGIN_SHELL=0` keeps the runtime on `bash -c` by default so
bounded tool commands do not depend on host login-shell startup hooks such as
`/etc/profile.d/*`. Set it back to `1` only when a specific command truly
requires login-shell initialization.

The managed default for `VICUNA_BASH_TOOL_MAX_CHILD_PROCESSES` is intentionally
much higher than the runtime's old low safety floor because Linux applies
`RLIMIT_NPROC` per user, not per command. Values that are too small can prevent
even simple allowed commands from forking on a busy workstation.

## Behavior

- `/start` registers the chat for proactive relay and returns a confirmation
- plain text user messages are sent to the local Vicuña runtime and the
  assistant follow-up is sent back to the same Telegram chat once the deferred
  turn finishes
- the bridge now tracks a per-chat Telegram conversation anchor so replies to
  runtime-authored assistant messages, plus plain next messages when one latest
  active conversation is clear, stay attached to the same bounded continuity
  transcript instead of starting an unrelated follow-up thread
- runtime-owned `ask_with_options` tool calls are delivered through a dedicated
  Telegram outbox surface, not through fallback assistant text
- runtime-owned mutation approvals are delivered through the same outbox as
  `approval_request` items; the bridge submits inline-button decisions back to
  `/v1/telegram/approval` as structured approval events instead of rewriting
  them into synthetic transcript text
- before forwarding a fresh Telegram user message, the bridge calls the runtime
  interruption surface so pending DMN-owned approval waits for that chat are
  superseded cleanly rather than blocking the new foreground turn
- inline option clicks arrive as Telegram `callback_query` updates; ordinary
  `ask_with_options` prompts are still rewritten into the bounded transcript,
  while approval prompts are resolved through the runtime-owned approval object
  and do not create a synthetic user turn
- plain-text answers sent in response to an `ask_with_options` prompt are still
  treated as normal user messages rather than structured option selections, but
  they remain attached to the same Telegram conversation anchor as the prompt
- forwarded chat requests now include Telegram chat metadata headers so the
  runtime can maintain its own bounded last-`N` turn dialogue object instead of
  depending only on bridge-local transcript state
- supported PDF, DOC, and DOCX messages are converted into plain text before
  they enter the local transcript and are also persisted to Supermemory as both
  a raw file record and an extracted-text record linked by shared metadata
- each Telegram chat keeps its own bounded persisted transcript keyed by
  Telegram `chat_id`, so follow-up turns reuse recent context after restarts
- pending inline-option prompts are also kept in bounded persisted bridge state
  so callback selections can be resolved safely after bridge restart
- the persisted bridge state now also records an explicit runtime outbox
  checkpoint plus the last delivered runtime outbox receipt, including the
  exact Telegram `message_id` returned for that follow-up
- bounded transcript trimming drops any leading assistant-only orphan created by
  raw history clipping so the runtime keeps seeing a coherent conversation that
  still includes the latest user turn
- each forwarded Telegram turn now logs transcript length and role sequence to
  the bridge journal for continuity debugging
- normal Telegram text and document turns no longer send a bridge-authored
  acknowledgement before the real work starts; the first bridge-authored reply
  for a deferred turn is now the substantive final follow-up or failure message
- proactive runtime self-emits are consumed from `/v1/responses/stream` and
  sent to all registered chats while also being recorded into each chat's local
  transcript window
- those proactive emits are also represented inside the runtime as broadcast
  Telegram dialogue turns, so DMN-origin or bridge-origin follow-ups can reuse
  recent user-facing continuity without treating the SSE mailbox as dialogue
  memory
- the bridge intentionally reconnects the proactive SSE stream when the server
  closes an idle stream; it always reconnects with `after=0` and deduplicates
  by retained `response_id`, so retained self-emits are replay-safe across
  reconnects
- repeated Telegram `409 Conflict: terminated by other getUpdates request`
  errors mean another bot poller is still running with the same token, either
  on this host or somewhere else; clear the extra poller before expecting
  stable message ingress
- runtime-owned follow-up delivery now logs the runtime outbox sequence,
  requested reply target, fallback mode, and returned Telegram `message_id`
- if Telegram rejects a reply anchor, the bridge retries once without the reply
  anchor so the user still sees the final message
- fresh or transcript-reset bridge state does not replay the retained runtime
  outbox backlog by default; explicit replay requires
  `TELEGRAM_BRIDGE_REPLAY_RETAINED_OUTBOX=1`
- if the bridge appears to answer an older Telegram topic, inspect the bridge
  state file and journal together: if `telegramOffset` is still increasing and
  new `appended Telegram user turn` entries appear, the defect is in transcript
  shaping rather than Telegram update replay
- intentionally empty completion text is now allowed on Telegram turns because
  the user-visible payload may already have been delivered by a tool such as
  `ask_with_options`

## Reset Guidance

For a clean transcript reset without replaying old runtime follow-ups:

1. stop `vicuna-telegram-bridge.service`
2. back up `/var/lib/vicuna/telegram-bridge-state.json`
3. preserve:
   - `telegramOffset`
   - `telegramOutboxOffset`
   - `telegramOutboxCheckpointInitialized`
   - `telegramOutboxDeliveryReceipt`
4. clear:
   - `chatSessions`
   - `chatConversationState`
   - `pendingOptionPrompts`
5. restart the bridge

Do not clear the outbox checkpoint fields unless you explicitly intend to
replay retained runtime outbox items.

## Test

```bash
npm run test:telegram-bridge
```
