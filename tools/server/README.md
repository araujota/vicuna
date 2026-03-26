# Vicuña Provider Server

`llama-server` is the retained product surface in this repository.

It is a DeepSeek-backed HTTP server with an optional local
`Qwen3-Embedding-0.6B-GGUF` backend for emotive-trace enrichment.

## Metacognitive control loop

All provider passes, including foreground turns, cognitive replay, and
ongoing-task execution, run through the same bounded control path:

- preserves replayed assistant `reasoning_content` for state reconstruction
- recomputes the current emotive moment and VAD state from replayed request
  history, not just the newest user text
- retrieves at most one matching heuristic from persisted bad-path memory
- converts the live state plus any retrieved heuristic into an explicit control
  policy that scores direct, reflective, tool-light, tool-heavy, and
  background-defer modes
- injects additive system guidance for policy, heuristic, and VAD without
  mutating the user-visible task payload
- persists the turn-local `final_policy` and `heuristic_retrieval` objects in
  `vicuna_emotive_trace`

By default the server targets `deepseek-chat` and injects
`thinking: {"type":"enabled"}` on outbound DeepSeek requests. If you send a
provider-native `thinking` object, the server forwards it unchanged instead of
rewriting it.

If the caller does not provide an explicit output-token cap, the server derives
one from the metacognitive reasoning depth:

- `none -> 256`
- `short -> 512`
- `medium -> 1024`
- `deep -> 2048`

Every outbound DeepSeek turn also uses `temperature: 0.2`. The runtime stamps
that value onto direct, staged, bridge-scoped, and background turns instead of
leaving temperature implicit or caller-controlled.

The DeepSeek adapter reuses one persistent configured HTTP client per provider
authority instead of rebuilding transport state on every turn. Inspect the
live transport counters at `/health -> provider -> transport`.

The runtime also retains a bounded structured request-trace registry. Each
foreground or background pass emits labeled JSON events with a shared
`request_id` across runtime handling, staged selection, provider traffic,
runtime tool execution, and Telegram outbox queueing. Inspect the summary at
`/health -> request_traces` and the retained event stream at
`GET /v1/debug/request-traces`.
Completed provider events retain the exact returned `reasoning_content` and
visible `content`, and runtime-guidance events retain the exact additive VAD or
heuristic text that was injected or skipped.

## Flattened runtime tools

The active runtime tool surface is flattened and provider-visible. The
installed runtime catalog exposes:

- `media_read`
- `media_download`
- `media_delete`
- `hard_memory_read`
- `hard_memory_write`
- `web_search`
- `ongoing_task_create`
- `ongoing_task_delete`

These flattened capabilities map onto the existing runtime backends and keep
tool policy explicit in CPU-side control code. The default path no longer
requires a family -> method -> payload staging waterfall.

Legacy staged family or method selection remains available only as a fallback
for compatibility testing behind `VICUNA_ENABLE_STAGED_TOOL_FALLBACK=1`.

For retained bridge-scoped Telegram turns, the server also caches the loaded
runtime tool catalog in memory. Inspect those bounded cache counters at
`/health -> bridge_runtime`.

Bridge-scoped turns no longer depend on a provider-visible Telegram tool
family. The server parses assistant rich-plan text, validates optional delivery
metadata, enqueues the outbox item itself, and returns additive
`vicuna_telegram_delivery` metadata so the bridge waits for outbox delivery
instead of relaying assistant text directly.

Tool-continuation guidance remains additive and keeps:

- replayed assistant `reasoning_content` unchanged
- request-scoped VAD guidance unchanged
- heuristic guidance unchanged
- staged follow-up turns receive renewed additive VAD guidance even without a
  classic assistant/tool continuation span

## Rich-plan bridge delivery

Bridge-scoped responses may be returned as plain text or as a rich-plan body:

- front matter delimited by `---`
- optional `format`, `title`, `disable_web_page_preview`, `delivery_hint`
- optional `reply_markup`
- message body after the closing delimiter

The runtime parses this structure into a Telegram `sendMessage` outbox payload.
Invalid metadata is stripped server-side, the message body is preserved, and
the trace records the strip event.

Bridge-scoped message items may also carry an additive `emotive_animation`
bundle:

- every post-seed emotive recomputation is exported as one keyframe candidate
- `vicuna_emotive_trace.live_generation_start_block_index` marks the explicit
  boundary between seeded transcript replay and live reply generation
- queued Telegram message items retain `text` as the canonical reply body and
  add `emotive_animation` only as render metadata for the bridge
- the retained Telegram bridge renders that bundle into a fixed-topology MP4,
  uploads it as a follow-up `sendVideo`, and keeps text delivery even when
  render, encode, or upload fails
- the bridge host must provide `ffmpeg` or the animation path falls back to
  text-only delivery with an inspectable failure reason

Tool metadata policy:

- runtime tools should still provide a clear family, method, and contract layer
  when available, even though the default provider surface is now flat
- request tools may optionally provide:
  - `x-vicuna-family-id`
  - `x-vicuna-family-name`
  - `x-vicuna-family-description`
  - `x-vicuna-method-name`
  - `x-vicuna-method-description`
- if those fields are absent, the server derives family/method names from the
  function name, but explicit metadata is preferred
- future tools and bridge/runtime integrations should provide those explicit
  layers at the tool-definition source rather than trying to patch them in only
  inside the server
- bridge-scoped runtime turns fetch those explicit layers from the installed
  runtime catalog inside the server, not from bridge-authored request shaping

## Cognitive replay

When a foreground trace produces a significant negative emotive delta, the
runtime stores one bounded cognitive replay entry. After the server has been
idle long enough, a background worker replays the stored episode through the
same provider/emotive path with a fixed replay prompt.

Replay resolution is explicit:

- replay-mode traces are marked `cognitive_replay` and cannot admit new replay entries
- replay success is scored from assistant-generated replay blocks only
- seeded replay prompts and trailing runtime bookkeeping do not count toward resolve/defer validation
- each resolved replay runs one follow-up compression pass that receives labeled `Bad Path` and `Better Path`
- successful compression persists one hard-memory record containing the full bad path, full better path, and one structured heuristic

Inspect replay state with `GET /v1/emotive/cognitive-replay`.

## Heuristic memory

Resolved replay episodes are compressed into one reusable heuristic object and
stored in bounded hard memory. Live request assembly then:

- embeds the latest thought/request trace
- scores it against stored bad-path objects
- reranks with structural and emotive similarity
- injects only the matching heuristic as brief critical guidance when the
  bounded threshold is met
- derives bounded control biases from the matched heuristic so prior failures
  can shift routing and reasoning policy without bypassing live recomputation

Inspect persisted heuristic records and the latest retrieval decision with
`GET /v1/emotive/heuristics`.

## Pre-idle ongoing tasks

After the replay queue is exhausted for the current idle cycle, the same
background worker runs one explicit ongoing-task stage before true idle:

- polls the hard-memory-backed ongoing-task registry and current system time
- runs one due-decision pass that considers both cadence fields and original task wording
- launches at most one selected task by sending the exact stored `task_text`
  through the normal provider/emotive path
- advances the task's `last_done_at` timestamp only after that background run succeeds
- suppresses new cognitive replay admissions during ongoing-task decision and execution traces

Inspect worker state with `GET /v1/emotive/ongoing-tasks` or under
`/health -> emotive_runtime -> ongoing_tasks`.

## Supported routes

- `GET /health`
- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/emotive/trace/latest`
- `GET /v1/emotive/cognitive-replay`
- `GET /v1/emotive/heuristics`
- `GET /v1/emotive/ongoing-tasks`
- `GET /v1/debug/request-traces`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/stream`
- `GET /v1/telegram/outbox`
- `POST /v1/telegram/outbox`
- `POST /v1/telegram/approval`
- `POST /v1/telegram/interruption`

Default-surface aliases remain for `/models`, `/completions`,
`/chat/completions`, and `/responses`.

## Removed

- local chat inference as the main serving path
- hidden provider-side tool-selection policy
- legacy WebUI variants, themes, and benchmark helpers
- slot/router/KV orchestration as product features

The Telegram bridge endpoints are intentionally retained as a narrow transport
surface for external dialogue delivery. The bridge no longer owns Telegram
prompt construction, runtime tool injection, or runtime tool continuation.

## Telegram outbox contract

`POST /v1/telegram/outbox` still accepts retained `kind=message` writes, but
message items may now carry either:

- plain text only, which the server normalizes to `telegram_method=sendMessage`
- an explicit `telegram_method` plus `telegram_payload` for richer Bot API
  sends such as formatted text, media, and `reply_markup`

The server keeps a normalized summary `text` field on each queued item so the
bridge can preserve transcript continuity and readable delivery logs.
For bridge-scoped Telegram turns, the runtime derives the final outbox payload
from plain assistant text or parsed rich-plan metadata and injects chat scope,
reply anchor, dedupe, and urgency metadata itself.
When present, `emotive_animation` is already fully materialized server-side;
the bridge must not reconstruct keyframes by polling other runtime endpoints.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build --target llama-server -j8
```

## Run

```bash
export VICUNA_DEEPSEEK_API_KEY="your-key"
export VICUNA_DEEPSEEK_MODEL="deepseek-chat"
export VICUNA_DEEPSEEK_BASE_URL="https://api.deepseek.com"
export VICUNA_SYSTEM_ENV_FILE="/etc/vicuna/vicuna.env"
export VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH="/etc/vicuna/openclaw-tool-secrets.json"
export VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH="/var/lib/vicuna/openclaw-catalog.json"
export VICUNA_EMOTIVE_EMBED_MODEL="/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf"
export VICUNA_EMOTIVE_EMBED_POOLING="last"
export VICUNA_OPENCLAW_NODE_BIN="node"
export VICUNA_OPENCLAW_ENTRY_PATH="/absolute/path/to/tools/openclaw-harness/dist/index.js"
export VICUNA_ONGOING_TASKS_ENABLED="true"
export VICUNA_ONGOING_TASKS_BASE_URL="https://api.supermemory.ai"
export VICUNA_ONGOING_TASKS_AUTH_TOKEN="your-supermemory-key"

./build/bin/llama-server --host 127.0.0.1 --port 8080 --api-surface openai --no-webui
```

For rebuild-safe host deployments, the runtime and bridge scripts now source
`VICUNA_SYSTEM_ENV_FILE` or `/etc/vicuna/vicuna.env` automatically. That host
env file should hold the durable values for:

- `TELEGRAM_BOT_TOKEN`
- `VICUNA_DEEPSEEK_API_KEY`
- `RADARR_API_KEY`
- `SONARR_API_KEY`
- `CHAPTARR_API_KEY`
- `TAVILY_API_KEY`
- `SUPERMEMORY_API_KEY`
- `SUPERMEMORY_BASE_URL`
- `VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH`
- `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH`

At startup, `tools/ops/sync-openclaw-runtime-state.sh` rehydrates the OpenClaw
tool secrets and runtime catalog from that env file into the stable secrets and
catalog paths outside the checkout. That keeps media-tool and Tavily
credentials alive across rebuilds and branch changes.

When probing runtime tools manually on the host, export the same synced paths
the running service uses or the harness may fall back to the checkout-local
`.cache` path and report false missing-key errors:

```bash
VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH=/home/tyler-araujo/.config/vicuna/openclaw-tool-secrets.json \
VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=/home/tyler-araujo/.local/state/vicuna/openclaw-catalog.json \
/home/tyler-araujo/.nvm/versions/node/v20.19.5/bin/node \
  /home/tyler-araujo/Projects/vicuna-codex-130-strict-selector-tools/tools/openclaw-harness/dist/index.js \
  invoke-runtime --tool-name=sonarr_list_downloaded_series --arguments-base64=e30=
```

Ongoing-task env vars:

- `VICUNA_ONGOING_TASKS_ENABLED`
- `VICUNA_ONGOING_TASKS_BASE_URL`
- `VICUNA_ONGOING_TASKS_AUTH_TOKEN`
- `VICUNA_ONGOING_TASKS_CONTAINER_TAG`
- `VICUNA_ONGOING_TASKS_RUNTIME_IDENTITY`
- `VICUNA_ONGOING_TASKS_REGISTRY_KEY`
- `VICUNA_ONGOING_TASKS_REGISTRY_TITLE`
- `VICUNA_ONGOING_TASKS_QUERY_THRESHOLD`
- `VICUNA_ONGOING_TASKS_POLL_MS`
- `VICUNA_ONGOING_TASKS_TIMEOUT_MS`

Bridge-scoped runtime env vars:

- `VICUNA_OPENCLAW_NODE_BIN`
- `VICUNA_OPENCLAW_ENTRY_PATH`
- `VICUNA_TELEGRAM_RUNTIME_MAX_ROUNDS`
- `VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON` for test/mocked catalog injection
- `VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON` for test/mocked observations

## Validate control policy and bridge delivery

Run the focused provider tests:

```bash
LLAMA_SERVER_BIN_PATH=./build/bin/llama-server pytest tools/server/tests/unit/test_deepseek_provider.py -q -k "tool or interleaved or control_policy or bridge"
```

Legacy staged fallback coverage remains available behind
`VICUNA_ENABLE_STAGED_TOOL_FALLBACK=1`.

Run the cognitive replay coverage:

```bash
LLAMA_SERVER_BIN_PATH=./build/bin/llama-server pytest tools/server/tests/unit/test_deepseek_provider.py -q -k "cognitive_replay"
```

Run the heuristic-memory coverage:

```bash
LLAMA_SERVER_BIN_PATH=./build/bin/llama-server pytest tools/server/tests/unit/test_deepseek_provider.py -q -k "heuristic"
```

Run the ongoing-task idle-stage coverage:

```bash
LLAMA_SERVER_BIN_PATH=./build/bin/llama-server pytest tools/server/tests/unit/test_deepseek_provider.py -q -k "ongoing_task"
```
