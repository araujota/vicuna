# Vicuña Provider Server

`llama-server` is the retained product surface in this repository.

It is a DeepSeek-backed HTTP server with an optional local
`Qwen3-Embedding-0.6B-GGUF` backend for emotive-trace enrichment.

## Interleaved tool continuations

When a request resumes an active DeepSeek tool loop, the server now:

- preserves the assistant tool-call step's `reasoning_content` for replay
- reconstructs the current emotive/VAD state from the replayed request history
- injects one additive VAD guidance `system` message after the newest tool-result span
- runs bounded heuristic retrieval in parallel against persisted bad-path memory
- injects at most one brief critical-guidance heuristic when the current trace matches a stored bad path
- keeps the original tool payload unchanged

By default the server targets `deepseek-chat` and injects
`thinking: {"type":"enabled"}` on outbound DeepSeek requests. If you send a
provider-native `thinking` object, the server forwards it unchanged instead of
rewriting it.

All outbound DeepSeek turns, including staged family/method/payload turns and
background/internal provider passes, are capped at `max_tokens: 256`. The
server enforces that ceiling even if the caller supplies a different output
token field, and it does so without disabling reasoning traces.
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

## Staged tool loop

When an incoming OpenAI-compatible request includes `tools` with automatic tool
selection, the server no longer exposes the flat tool list directly to the
provider. Instead it runs an explicit staged loop:

- family selection: the provider sees only high-level tool families and must
  return `{"family":"..."}` in JSON
- method selection: the provider sees only methods for the chosen family and
  must return `{"method":"..."}`, `{"method":"back"}`, or
  `{"method":"complete"}`
- payload construction: the provider sees the chosen method's typed contract
  and must return `{"action":"submit","payload":{...}}` or
  `{"action":"back"}`

After a valid payload is produced, the server emits one normal OpenAI tool call
back to the caller. After the caller returns a real tool result on the next
request, the staged loop begins again from family selection.

The server now assumes callers inject the real direct tool definitions they
want exposed for that turn, with one explicit exception: bridge-scoped
Telegram requests. For those requests the server loads the installed runtime
tool catalog itself, appends explicit Telegram delivery methods, executes any
selected runtime tools internally, and continues the staged loop until it can
queue final Telegram delivery.

For retained bridge-scoped Telegram turns, the server also caches the loaded
runtime tool catalog plus the derived staged family/method/payload prompt
bundle in memory. Inspect those bounded cache counters at
`/health -> bridge_runtime`.

Bridge-scoped Telegram turns are the one built-in exception: when a request
arrives with Telegram bridge headers and resolves to an explicit Telegram
delivery method, the server enqueues the Telegram outbox item itself, clears
the outward tool call, and returns additive `vicuna_telegram_delivery` metadata
so the bridge can wait for outbox delivery instead of trying to relay
assistant text directly.

The staged prompts are additive and keep:

- replayed assistant `reasoning_content` unchanged
- request-scoped VAD guidance unchanged
- heuristic guidance unchanged

Tool metadata policy:

- every staged-exposed tool should provide a clear family, method, and contract
  layer
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
- bridge-scoped Telegram turns fetch those explicit layers from the installed
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
export VICUNA_EMOTIVE_EMBED_MODEL="/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf"
export VICUNA_EMOTIVE_EMBED_POOLING="last"
export VICUNA_OPENCLAW_NODE_BIN="node"
export VICUNA_OPENCLAW_ENTRY_PATH="/absolute/path/to/tools/openclaw-harness/dist/index.js"
export VICUNA_ONGOING_TASKS_ENABLED="true"
export VICUNA_ONGOING_TASKS_BASE_URL="https://api.supermemory.ai"
export VICUNA_ONGOING_TASKS_AUTH_TOKEN="your-supermemory-key"

./build/bin/llama-server --host 127.0.0.1 --port 8080 --api-surface openai --no-webui
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

Bridge-scoped Telegram runtime-tool env vars:

- `VICUNA_OPENCLAW_NODE_BIN`
- `VICUNA_OPENCLAW_ENTRY_PATH`
- `VICUNA_TELEGRAM_RUNTIME_MAX_ROUNDS`
- `VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON` for test/mocked catalog injection
- `VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON` for test/mocked observations

## Validate staged tool-loop and VAD guidance

Run the focused provider tests:

```bash
LLAMA_SERVER_BIN_PATH=./build/bin/llama-server pytest tools/server/tests/unit/test_deepseek_provider.py -q -k "tool or interleaved or staged"
```

The staged family/method selectors retry one corrective JSON turn if DeepSeek
returns empty or malformed JSON, which matches DeepSeek's documented occasional
JSON-mode empty-content behavior.

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
