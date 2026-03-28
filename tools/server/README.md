# Vicuña Provider Server

`llama-server` is the retained product surface in this repository.

It is a DeepSeek-backed HTTP server with an optional local
`Qwen3-Embedding-0.6B-GGUF` backend for emotive-trace enrichment.

## Metacognitive control loop

All provider passes, including foreground turns, cognitive replay, and
cron-triggered recurring execution, run through the same bounded control path:

- preserves replayed assistant `reasoning_content` for state reconstruction
- recomputes the current emotive moment and VAD state from replayed request
  history, not just the newest user text
- retrieves at most one matching heuristic from persisted bad-path memory
- converts the live state plus any retrieved heuristic into an explicit control
  policy that scores direct, reflective, tool-light, tool-heavy, and
  background-defer modes
- converts that same policy into bounded DeepSeek request controls for
  `thinking`, token budget, prefix, stop, sampling, repetition, and tool
  choice before the provider call is sent
- injects additive system guidance for policy, heuristic, and VAD without
  mutating the user-visible task payload
- persists the turn-local `final_policy` and `heuristic_retrieval` objects in
  `vicuna_emotive_trace`

The server always targets `deepseek-chat`. The CPU-side metacognitive policy
now owns the outbound `thinking` switch turn by turn instead of leaving it
fixed on. When `thinking` remains enabled, DeepSeek-compatible shaping
suppresses sampling and repetition parameters that the provider documents as
ineffective in thinking mode. When `thinking` is disabled, the server can
apply bounded sampling, repetition, prefix, and stop profiles directly.

If the caller does not provide an explicit output-token cap, the server derives
one from the metacognitive reasoning depth:

- `none -> 256`
- `short -> 512`
- `medium -> 1024`
- `deep -> 2048`

Provider shaping is profile-based rather than raw caller passthrough:

- `sampling_profile=deterministic` applies `temperature: 0.0`
- `sampling_profile=balanced` applies `temperature: 0.2`
- `sampling_profile=creative` applies `top_p: 0.95`
- `repetition_profile=anti_stall_soft` applies modest
  `frequency_penalty/presence_penalty`
- `repetition_profile=anti_stall_hard` applies stronger repetition penalties
- `prefix_profile` and `stop_profile` are only applied when DeepSeek beta
  prefix completion is valid for the turn

Applied controls are retained in `final_policy.applied_provider_controls` and
exported through the policy-transition surface.

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

## RL runtime surface

The server now exposes a bounded RL-oriented governance surface alongside the
native control policy.

- each completed provider turn can append one typed `policy_transition`
- the transition captures the pre-decision observation, executed action,
  reward ledger, next observation, termination metadata, rollout metadata, and
  the concrete provider controls applied after compatibility checks
- `capture` mode records transitions only
- `shadow` mode also requests a bounded candidate action from
  `VICUNA_POLICY_CANDIDATE_URL` and records disagreements without changing live
  execution
- `eval_only` requests candidate proposals but always keeps execution native
- `canary_live` can execute candidate actions for a deterministic canary slice
  after confidence and native safety checks
- `/v1/policy/status` exposes mode, stored-transition count, shadow counters,
  candidate alias/version, canary share, rollback state, the active
  desired-state reward model, and the current export window size
- `/v1/policy/transitions` exports the bounded in-memory transition window for
  offline training or evaluation tooling
- `/health -> policy_runtime` exposes coarse runtime health for operators

Native CPU-side control remains authoritative in every mode. Live canary
execution still fails closed to native execution on low confidence, bounds
violations, proposal failures, or rollback.

The next lifecycle stage is now implemented offline under
`tools/policy-learning/`: operators can persist the exported transition window
into a dataset, build a masked training corpus, train deterministic candidate
artifacts, register them with alias metadata, and run a nightly offline batch
without changing the live server.

Reward semantics are now explicit and inspectable. Each captured transition
stores the active desired-state reward model plus a typed reward breakdown
covering:

- a weighted closeness score over all 14 emotive-moment dimensions
- a weighted closeness score over all 3 VAD axes
- a potential-style progress term from pre-turn state to post-turn state
- a terminal-alignment term plus bounded completion, latency, token, tool, and
  candidate-failure terms

Use `VICUNA_POLICY_REWARD_CONFIG_PATH=/absolute/path/to/reward-profile.json`
to override the default target profile. Invalid override payloads fail startup.

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
- `host_shell`

These flattened capabilities map onto the existing runtime backends and keep
tool policy explicit in CPU-side control code. The default path no longer
requires a family -> method -> payload staging waterfall.

`hard_memory_read` and `hard_memory_write` use a local markdown-backed hard
memory store. Host deployments should provision
`VICUNA_HARD_MEMORY_DIR=/home/vicuna/home/memories`; local/dev runs fall back
to a repo-local host-shell cache directory.

`ongoing_task_create` and `ongoing_task_delete` install and remove `vicuna`
user cron entries through the runtime harness. Host installs should provision
`VICUNA_ONGOING_TASKS_DIR=/var/lib/vicuna/ongoing-tasks`; the retained runner
posts the stored task text back to `/v1/chat/completions` as a `system`
message.

`host_shell` is intentionally a last-resort fallback, not a general substitute
for the specialized runtime tools. It executes as the runtime service user from
an explicit workspace root and returns a structured observation envelope rather
than a raw shell transcript. Host installs should provision and export
`VICUNA_HOST_SHELL_ROOT=/home/vicuna/home`; local/dev runs fall back to a
repo-local `.cache/vicuna/host-shell-home` workspace when that host path is not
present.

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
Bridge delivery defaults to plain text when `format` is omitted. Explicit
`format: html` remains the supported rich-text path and maps to Telegram
`parse_mode: HTML`. Rich-plan `format: markdown` is retained in
`vicuna_rich_response` metadata for inspection but is now normalized into
Telegram-supported HTML at the server boundary, and the bridge also applies the
same normalization to plain `sendMessage` payloads that still contain authored
Markdown-style decorators, so headings, emphasis, blockquotes, code blocks, and
divider lines survive delivery without relying on `MarkdownV2`.

Bridge-scoped message items may also carry an additive `emotive_animation`
bundle:

- every post-seed emotive recomputation is exported, but consecutive identical
  moments are compacted server-side into one keyframe with
  `hold_keyframe_count`
- `vicuna_emotive_trace.live_generation_start_block_index` marks the explicit
  boundary between seeded transcript replay and live reply generation
- queued Telegram message items retain `text` as the canonical reply body and
  add `emotive_animation` only as render metadata for the bridge
- the retained Telegram bridge renders that bundle into a fixed-topology neon
  membrane scene with one anchor per emotive dimension, a black background,
  inward-sagging triangulated surface, and starts rendering in parallel with
  delivery planning; the live hull solve is anchored around the requested
  `r/3 -> 5r/3` register radii so strong emotive values produce visibly large
  outward lift while broad positive and negative shoulder fields disturb the
  whole spherical topology; if the formatted reply fits under Telegram's caption
  limit, it sends one spoilered `sendVideo` message whose caption reuses the
  reply text, and if the caption would overflow it sends the formatted text
  first and follows with the video as a separate reply as soon as rendering
  finishes; render, encode, or upload failure still falls back to text-only
  delivery
- the bridge persists the terminal emotive moment per Telegram conversation
  and prepends that prior end-state as the next clip's start pose, so only new
  keyframes are animated on each delivery
- bundle version `2` uses `seconds_per_keyframe = 0.5`, `fps = 30`,
  `raw_keyframe_count`, and `distinct_keyframe_count`; the bridge renders holds
  for repeated identical states instead of inventing extra motion
- the bridge host must provide `ffmpeg`; the live service now resolves and
  pins `TELEGRAM_BRIDGE_FFMPEG_BIN` explicitly, and the default encode path
  targets `h264_nvenc` for VRAM-backed conversion with an inspectable failure
  reason if GPU encode is unavailable
- when `TELEGRAM_BRIDGE_RENDER_BACKEND=chromium_webgl`, the bridge sends
  emotive render jobs to the local `VICUNA_WEBGL_RENDERER_URL` service instead
  of the in-process CPU canvas path; that service owns the headless Chromium
  browser, reports GPU readiness at `/health`, and requires an explicit
  Chromium binary plus bounded concurrency or memory settings

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

## Recurring host cron tasks

The server no longer owns a pre-idle ongoing-task polling stage. Recurring work
is now scheduled by host cron:

- the runtime tool layer installs one `vicuna`-owned cron entry per recurring
  task
- the retained host wrapper loads the canonical env, acquires a per-task lock,
  and posts the stored task wording back to the server as a `system` message
- the server therefore sees scheduled work as a normal provider pass instead of
  a special idle-worker mode

## Manual skill and memory discovery

Foreground and bridge-scoped provider passes now inject bounded file-name
indexes before the DeepSeek call:

- `SKILLS:` lists visible `*.md` files from `VICUNA_SKILLS_DIR`, defaulting to
  `/home/vicuna/home/skills` on the host
- `MEMORIES:` lists visible `*.md` files from `VICUNA_HARD_MEMORY_DIR`,
  defaulting to `/home/vicuna/home/memories` on the host
- the runtime advertises names only; full file bodies are never auto-inlined
- the model must call `skill_read` or `hard_memory_read` explicitly to load one
  file body
- `skill_create` remains visible but prompt-gated: it may only be used when the
  user directly asks to create or update a skill in the current conversation

After a completed user-facing response, the runtime also runs one bounded local
memory-admission pass over the visible session transcript and writes a markdown
memory file only when it finds a durable preference, fact, verified failure
pattern, reusable fix, or project convention.

## Supported routes

- `GET /health`
- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/emotive/trace/latest`
- `GET /v1/emotive/cognitive-replay`
- `GET /v1/emotive/heuristics`
- `GET /v1/policy/status`
- `GET /v1/policy/transitions`
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
The bridge now treats Telegram HTML as the canonical delivery format for all
outgoing user-facing message text and captions unless explicit Telegram
entities are already present; plain text, markdown-like decorators, and
broader htmlish tags are reduced into Telegram-supported HTML before delivery.
Markdown pipe tables are normalized into aligned `<pre>` grids because
Telegram's supported HTML subset does not provide native table elements.
DeepSeek bridge-scoped replies that place DSML
`<｜DSML｜function_calls>` markup in assistant `content` are recovered back into
structured runtime tool calls server-side, and the bridge strips any surviving
DSML blocks from outgoing text/caption payloads as a fail-closed delivery
guard.
If a bridge-scoped request uses up its runtime-tool round budget without
settling, the runtime forces one tool-free synthesis pass and then degrades to
an explicit clarification reply instead of returning HTTP 500.
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
export VICUNA_HARD_MEMORY_DIR="/home/vicuna/home/memories"
export VICUNA_HEURISTIC_MEMORY_PATH="/home/vicuna/home/heuristics/vicuna-heuristic-memory.json"
export VICUNA_ONGOING_TASKS_DIR="/var/lib/vicuna/ongoing-tasks"
export VICUNA_ONGOING_TASKS_TMPDIR="/var/lib/vicuna/ongoing-tasks/tmp"
export VICUNA_ONGOING_TASKS_RUNNER_SCRIPT="/absolute/path/to/tools/ops/run-ongoing-task-cron.sh"
export VICUNA_DOCS_DIR="/home/vicuna/home/docs"
export VICUNA_EMOTIVE_EMBED_MODEL="/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf"
export VICUNA_EMOTIVE_EMBED_POOLING="last"
export VICUNA_OPENCLAW_NODE_BIN="node"
export VICUNA_OPENCLAW_ENTRY_PATH="/absolute/path/to/tools/openclaw-harness/dist/index.js"
export VICUNA_POLICY_MODE="capture"
export VICUNA_POLICY_MAX_TRANSITIONS="128"
export VICUNA_POLICY_CANDIDATE_URL="http://127.0.0.1:18081"
export VICUNA_POLICY_TIMEOUT_MS="500"

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
- `VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH`
- `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH`
- `VICUNA_HARD_MEMORY_DIR`
- `VICUNA_ONGOING_TASKS_DIR`
- `VICUNA_DOCS_DIR`

At startup, `tools/ops/sync-openclaw-runtime-state.sh` rehydrates the OpenClaw
tool secrets and runtime catalog from that env file into the stable secrets and
catalog paths outside the checkout. If the checkout is fresh and
`tools/openclaw-harness/dist/index.js` is missing, the helper now bootstraps
the harness with the same Node runtime selected for the services before
syncing the catalog. That keeps media-tool and Tavily credentials alive across
rebuilds and branch changes and prevents request-time Telegram 500s caused by
missing harness build output.

For an explicit rebuild-safe host refresh, run:

```bash
tools/ops/rebuild-vicuna-runtime.sh
```

That rebuild helper now materializes the OpenClaw runtime harness before it
starts `vicuna-runtime.service`.

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

Policy-runtime env vars:

- `VICUNA_POLICY_MODE`
- `VICUNA_POLICY_MAX_TRANSITIONS`
- `VICUNA_POLICY_CANDIDATE_URL`
- `VICUNA_POLICY_TIMEOUT_MS`
- `VICUNA_POLICY_CANARY_STEPS`
- `VICUNA_POLICY_CANARY_MIN_REQUESTS_PER_STEP`
- `VICUNA_POLICY_LIVE_CONFIDENCE_THRESHOLD`
- `VICUNA_POLICY_ROLLBACK_MAX_CANDIDATE_FAILURE_RATE`
- `VICUNA_POLICY_ROLLBACK_MAX_INVALID_ACTION_RATE`
- `VICUNA_POLICY_ROLLBACK_MAX_FALLBACK_RATE`

Offline policy-learning commands:

- `python3 tools/policy-learning/cli.py export --server http://127.0.0.1:8080 --dataset-dir .cache/vicuna/policy-datasets/local-v1 --dataset-id vicuna-governance-local-v1`
- `python3 tools/policy-learning/cli.py build-training-set --dataset-dir .cache/vicuna/policy-datasets/local-v1`
- `python3 tools/policy-learning/cli.py train --dataset-dir .cache/vicuna/policy-datasets/local-v1 --model-name vicuna-governance --registry-dir .cache/vicuna/policy-registry`
- `python3 tools/policy-learning/cli.py evaluate --dataset-dir .cache/vicuna/policy-datasets/local-v1 --candidate-command "python3 tools/policy-learning/registry_policy_adapter.py --artifact .cache/vicuna/policy-runs/<training-run-id>/artifact.json"`
- `python3 tools/policy-learning/cli.py register --registry-dir .cache/vicuna/policy-registry --model-name vicuna-governance --artifact-path .cache/vicuna/policy-runs/<training-run-id>/artifact.json --training-run-manifest-path .cache/vicuna/policy-runs/<training-run-id>/training_run_manifest.json --evaluation-report-path .cache/vicuna/policy-datasets/local-v1/reports/offline_eval_<policy-version>.json`
- `python3 tools/policy-learning/cli.py nightly-batch --server http://127.0.0.1:8080 --dataset-dir .cache/vicuna/policy-datasets/nightly --dataset-id vicuna-governance-nightly-v1 --registry-dir .cache/vicuna/policy-registry --model-name vicuna-governance`
- `python3 tools/policy-learning/cli.py serve-registry --host 127.0.0.1 --port 18081 --registry-dir .cache/vicuna/policy-registry --model-name vicuna-governance --default-alias candidate --fallback-alias champion`

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

Run the startup-helper coverage for missing harness build output:

```bash
node --test tools/ops/sync-openclaw-runtime-state.test.mjs
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

Run the recurring-task surface coverage:

```bash
LLAMA_SERVER_BIN_PATH=./build/bin/llama-server pytest tools/server/tests/unit/test_deepseek_provider.py -q -k "ongoing_task"
```

Run the RL runtime surface coverage:

```bash
LLAMA_SERVER_BIN_PATH=./build/bin/llama-server pytest tools/server/tests/unit/test_deepseek_provider.py -q -k "policy_transition_capture or shadow_policy"
```

Run the offline policy-learning coverage:

```bash
pytest -q tools/policy-learning/tests
```
