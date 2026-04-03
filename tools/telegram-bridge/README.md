# Telegram Bridge

This bridge connects a Telegram bot to the local Vicuña server.

It does two jobs in one process:

- polls Telegram for inbound text messages and forwards them to
  `/v1/chat/completions`
- subscribes to `/v1/responses/stream` and relays proactive self-emits to every
  registered Telegram chat
- polls `/v1/telegram/outbox` for runtime-owned follow-up messages and other
  retained compatibility items
- keeps those retained outbox/self-emit polling loops intentionally tight so a
  completed server turn is surfaced with less trailing delay

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
- `TELEGRAM_BRIDGE_MODEL` default: `deepseek-chat`
- `TELEGRAM_BRIDGE_STATE_PATH` default: `/tmp/vicuna-telegram-bridge-state.json`
- `TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS` default: `30`
- `TELEGRAM_BRIDGE_REQUEST_TIMEOUT_MS` default: `1800000`
- `TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES` default: `12`
- `TELEGRAM_BRIDGE_MAX_TOKENS` default: `1024`
- `TELEGRAM_BRIDGE_WEBGL_RENDER_TIMEOUT_MS` default: `120000`
- `TELEGRAM_BRIDGE_WEBGL_RENDER_MAX_ATTEMPTS` default: `3`
- `TELEGRAM_BRIDGE_VIDEO_SPOOL_DIR` default: sibling `telegram-video-spool/` beside `TELEGRAM_BRIDGE_STATE_PATH`
- `TELEGRAM_BRIDGE_VIDEO_RETRY_BASE_MS` default: `1000`
- `TELEGRAM_BRIDGE_VIDEO_MAX_ATTEMPTS` default: `0` (`0` means retryable video jobs stay durable until success)
- `TELEGRAM_BRIDGE_VIDEO_POLL_IDLE_MS` default: `500`
- `TELEGRAM_BRIDGE_MAX_DOCUMENT_CHARS` default: `12000`
- `TELEGRAM_BRIDGE_DOCLING_PYTHON_BIN` default: `python3`
- `TELEGRAM_BRIDGE_DOCLING_PARSER_SCRIPT_PATH` default: repo-local `tools/telegram-bridge/docling-parse.py`
- `TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG` default: `vicuna-telegram-documents`
- `VICUNA_DOCS_DIR` default: `"$VICUNA_HOST_SHELL_ROOT/docs"` or `/home/vicuna/home/docs`
- `TELEGRAM_BRIDGE_REPLAY_RETAINED_OUTBOX` default: `0`
  - when `0`, a fresh or reset bridge state fast-forwards to the newest
    retained runtime outbox sequence instead of replaying historical follow-ups
  - set to `1` only for an explicit operator replay of retained outbox items
- `VICUNA_API_KEY` if the server runs with bearer auth enabled

## Document Ingestion

Supported inbound Telegram document formats:

- PDF
- DOCX
- Markdown
- HTML/XHTML
- CSV
- XLSX
- PPTX
- AsciiDoc
- LaTeX

For each supported Telegram document, the bridge:

1. downloads the raw file through Telegram `getFile`
2. parses the file on the host through Docling
3. stores the raw file under the local Vicuña docs root
4. stores the parsed full-document output beside it as `parsed.md`
5. stores context-enriched parsed chunks beside it as `chunks.json`
6. appends the normalized parsed text to the Telegram chat transcript before
   forwarding the turn to Vicuña

Extraction policy:

- parsing runs through the host-side Python helper
  [docling-parse.py](/Users/tyleraraujo/vicuna/tools/telegram-bridge/docling-parse.py)
- that helper requires a Python environment on the bridge host with Docling and
  its chunking dependencies installed
- the same-turn user transcript now includes the exact label
  `Parsed contents of <filename>`

If `VICUNA_DOCS_DIR` is missing, supported document ingestion is rejected. If
the configured Python interpreter cannot import Docling, document ingestion
fails with a direct host requirement error after preserving the downloaded
source file in the local docs bundle.

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

The managed runtime launcher now starts the provider-backed server directly.
The key runtime variables are:

- `VICUNA_DEEPSEEK_API_KEY`
- `VICUNA_DEEPSEEK_BASE_URL` default: `https://api.deepseek.com`
- `VICUNA_DEEPSEEK_MODEL` default: `deepseek-reasoner`
- `VICUNA_DEEPSEEK_TIMEOUT_MS` default: `60000`

## Agentic Training Harness

For repeated training-data collection runs that should still land in the live
Telegram chat, use the host-side harness:

- launcher:
  [run-telegram-agentic-harness.sh](/Users/tyleraraujo/vicuna/tools/ops/run-telegram-agentic-harness.sh)
- sample input:
  [telegram-agentic-training-sample.json](/Users/tyleraraujo/vicuna/tools/ops/examples/telegram-agentic-training-sample.json)

Behavior:

- posts a visible prompt message into the configured Telegram chat
- forwards the actual runtime request through the same Telegram-scoped host
  `/v1/chat/completions` seam the bridge normally uses
- waits on bridge state for text delivery and queued-or-terminal emotive-video
  status so later async video retries do not block the batch
- verifies that host-side `transitions.jsonl`, `decode_traces.jsonl`, and
  `emotive_traces.jsonl` advanced before continuing to the next turn

Usage:

```bash
tools/ops/run-telegram-agentic-harness.sh \
  /absolute/path/to/requests.json \
  /absolute/path/to/report.json
```

Important limitation:

- the visible prompt message in Telegram is bot-authored because the Telegram
  Bot API cannot spoof the user account
- the runtime request is still simulated as a user turn through the
  Telegram-scoped host headers and the harness-maintained transcript

## Behavior

- `/start` registers the chat for proactive relay and returns a confirmation
- plain text user messages are sent to the local Vicuña runtime and the
  assistant follow-up is sent back to the same Telegram chat once the deferred
  turn finishes
- every forwarded Telegram turn now sends only the bounded transcript plus
  Telegram routing headers
- the server owns runtime tool catalog loading, staged family -> method ->
  payload orchestration, runtime tool execution, and the final explicit
  Telegram delivery decision
- the bridge now tracks a per-chat Telegram conversation anchor so replies to
  runtime-authored assistant messages, plus plain next messages when one latest
  active conversation is clear, stay attached to the same bounded continuity
  transcript instead of starting an unrelated follow-up thread
- runtime-owned follow-up messages delivered through server-owned explicit
  Telegram delivery methods now arrive through the dedicated Telegram outbox
  surface, not through fallback assistant text
- if a bridge-scoped Telegram turn still comes back as plain assistant text, the
  runtime now normalizes that into a `sendMessage` outbox item and returns
  `vicuna_telegram_delivery` metadata so the bridge does not drop or duplicate
  the reply while waiting for outbox delivery
- those runtime-owned follow-up messages may now be simple plain text, rich
  formatted text, photos, documents, polls, or dice sends with typed method
  contracts, so formatting, media sends, and reply markup survive the bridge
- runtime-owned mutation approvals are delivered through the same outbox as
  `approval_request` items; the bridge submits inline-button decisions back to
  `/v1/telegram/approval` as structured approval events instead of rewriting
  them into synthetic transcript text
- before forwarding a fresh Telegram user message, the bridge calls the runtime
  interruption surface so pending runtime-owned approval waits for that chat are
  superseded cleanly rather than blocking the new foreground turn
- inline approval clicks still arrive as Telegram `callback_query` updates and
  are resolved through the runtime-owned approval object instead of becoming a
  synthetic user turn
- forwarded chat requests now include Telegram chat metadata headers so the
  runtime can maintain its own bounded last-`N` turn dialogue object instead of
  depending only on bridge-local transcript state
- supported Docling-backed uploads are parsed on the host and are persisted to
  the local docs root as a source file, a parsed markdown artifact, and
  searchable parsed chunks linked by shared metadata
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
  the structured bridge service log under `/var/log/vicuna/telegram-bridge/`
  for continuity debugging
- each deferred bridge-owned request now carries `X-Client-Request-Id` through
  to the runtime and emits structured JSON log events such as
  `vicuna_request_started`, `vicuna_request_finished`, and
  `deferred_turn_failed`, so host-side bridge/runtime traces can be correlated
- normal Telegram text and document turns no longer send a bridge-authored
  acknowledgement before the real work starts; the first bridge-authored reply
  for a deferred turn is now the substantive final follow-up or failure message
- proactive runtime self-emits are consumed from `/v1/responses/stream` and
  sent to all registered chats while also being recorded into each chat's local
  transcript window
- those proactive emits are also represented inside the runtime as broadcast
  Telegram dialogue turns, so proactive or bridge-origin follow-ups can reuse
  recent user-facing continuity without treating the SSE mailbox as dialogue
  memory
- the bridge intentionally reconnects the proactive SSE stream when the server
  closes an idle stream; it always reconnects with `after=0` and deduplicates
  by retained `response_id`, so retained self-emits are replay-safe across
  reconnects
- retained outbox polling and self-emit reconnect delays are intentionally
  shorter now; the Telegram Bot API `getUpdates` long-poll timeout is unchanged
- repeated Telegram `409 Conflict: terminated by other getUpdates request`
  errors mean another bot poller is still running with the same token, either
  on this host or somewhere else; clear the extra poller before expecting
  stable message ingress
- runtime-owned follow-up delivery now logs the runtime outbox sequence,
  requested reply target, fallback mode, and returned Telegram `message_id`
- if Telegram rejects a reply anchor, the bridge retries once without the reply
  anchor so the user still sees the final message
- Telegram rich-text normalization now rebalances the supported HTML subset
  before delivery, so malformed `<b>` or similar inline tags cannot strand a
  completed outbox reply behind a `can't parse entities` retry loop
- if Telegram still rejects a normalized rich-text payload, the bridge retries
  once as plain text so the reply is delivered instead of remaining stranded in
  the retained outbox
- bridge-scoped provider requests now honor the configured
  `TELEGRAM_BRIDGE_MAX_TOKENS` cap instead of forcing `256`, which prevents
  long-form literature and research answers from being clipped prematurely
- text delivery and emotive video delivery are now decoupled: bridge text is
  delivered and checkpointed first, and any follow-up emotive video is queued
  as a persisted background job instead of blocking the user-visible message
- the bridge no longer uses caption-first video delivery for rich replies; text
  always remains a `sendMessage`, and the emotive animation is always a
  separate `sendVideo` follow-up
- the async video worker now persists rendered MP4 artifacts under the bridge
  state root, so a successful render can be reused across upload retries and
  bridge restarts instead of forcing a rerender on every transient upload fault
- emotive animation timing now follows the actual elapsed time between trace
  moments, so rendered videos match the real turn duration instead of using a
  fixed synthetic slot size
- the WebGL renderer now streams frame buffers directly into `ffmpeg` while the
  animation is being produced, instead of waiting for a full frame directory to
  exist first
- the renderer page caches compatible topology and scene scaffolding across
  requests, and `/health` now exposes cache diagnostics alongside GPU health
- the WebGL render client now sizes render timeout from animation duration and
  keyframe count, which prevents larger clips from failing solely because they
  exceed a static transport deadline
- `renderer_not_ready`, render transport failures, and render timeouts now stay
  queued by default; only terminal Telegram payload failures fail the follow-up
  video job closed
- malformed Telegram delivery handling is now explicitly layered:
  - malformed outbox items are rejected or skipped before delivery
  - oversized text is chunked into multiple `sendMessage` deliveries
  - malformed or unbalanced Telegram HTML is normalized and balanced
  - Telegram entity-parse failures fall back to plain text
  - invalid reply targets fall back to no-reply delivery
  - terminal chat failures are logged and skipped
  - follow-up video jobs move explicitly through `queued`, `render`, `upload`,
    `complete`, and terminal `failed`
  - retryable renderer and upload transport failures remain durable in the
    queue, while only terminal Telegram payload failures are removed
- fresh or transcript-reset bridge state does not replay the retained runtime
  outbox backlog by default; explicit replay requires
  `TELEGRAM_BRIDGE_REPLAY_RETAINED_OUTBOX=1`
- runtime outbox items that fail with terminal Telegram delivery errors such as
  `chat not found` or `bot was blocked by the user` are now logged and skipped,
  so one dead chat cannot pin every later follow-up behind it forever
- if the bridge appears to answer an older Telegram topic, inspect the bridge
  state file and journal together: if `telegramOffset` is still increasing and
  new `appended Telegram user turn` entries appear, the defect is in transcript
  shaping rather than Telegram update replay
- intentionally empty completion text is now allowed on Telegram turns because
  the user-visible payload may already have been delivered by an explicit
  Telegram delivery method
- the bridge should remain transport/state middleware only; future work should
  not reintroduce bridge-owned prompt construction, live-tool injection, or a
  second tool-continuation loop

Future tool-family additions must therefore satisfy both sides of the contract:

- the runtime catalog entry must carry explicit family metadata, method
  metadata, and a fully described typed contract
- the server-owned Telegram runtime path must be able to fetch that tool from
  the installed runtime catalog without bridge-authored request shaping

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

If a stale retained outbox item is already blocking current delivery, the safe
operator repair is to preserve `telegramOffset`, inspect the newest retained
runtime outbox sequence, and fast-forward `telegramOutboxOffset` only to that
known safe frontier before restarting the bridge.

## Test

```bash
npm run test:telegram-bridge
```
