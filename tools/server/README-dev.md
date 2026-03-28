# Provider Server Development Notes

Current retained components:

- `server.cpp`: route table and provider request flow
- `server-http.cpp`: HTTP transport and middleware
- `server-common.cpp`: JSON/error/SSE helpers
- `server-deepseek.cpp`: DeepSeek request and response mapping
- `server-emotive-runtime.cpp`: block-wise emotive moment, VAD projection, and
  metacognitive control policy
- `server-embedding-backend.cpp`: optional `Qwen3-Embedding-0.6B-GGUF` backend

Current direction:

- provider-first request handling only
- internal streaming for block-wise emotive capture
- flattened runtime tool exposure for media, hard memory, manual skill reads,
  web search, and recurring cron-task lifecycle operations, plus a last-resort
  bounded `host_shell` fallback
- every provider pass injects bounded `SKILLS:` and `MEMORIES:` file indexes so
  the model can choose explicit `skill_read` or `hard_memory_read` lookups
- active tool continuations preserve assistant `reasoning_content`
- request-scoped policy, heuristic, and VAD guidance is injected additively
  after the newest tool-result span
- heuristic retrieval runs in parallel with request-scoped guidance assembly
- matching heuristics are injected as one additive critical-guidance `system`
  message and may also bias the control policy
- a bounded RL runtime surface captures typed governance transitions for the
  native controller, supports status/export plus safe shadow comparison, and
  can execute registry-backed candidate actions for a bounded canary slice
- an offline policy-learning pipeline under `tools/policy-learning/` consumes
  those transitions for dataset export, masked training-contract generation,
  deterministic offline training, registry management, nightly batch
  orchestration, and registry-backed proposal serving
- Telegram bridge compatibility remains intentionally narrow
- no local chat inference product surface

The only justified retained dependency on the native `llama` library is the
optional local `Qwen3-Embedding-0.6B-GGUF` backend used by the emotive runtime.

Local embedding policy:

- only `general.architecture=qwen3` GGUF models are accepted
- the recommended local model is `Qwen3-Embedding-0.6B-GGUF`
- use `VICUNA_EMOTIVE_EMBED_POOLING=last`

Retained bridge endpoints:

- `/v1/responses/stream`
- `/v1/telegram/outbox`
- `/v1/telegram/approval`
- `/v1/telegram/interruption`

Telegram outbox policy:

- retained `kind=message` items may now include explicit `telegram_method` and
  `telegram_payload` fields instead of only plain text
- plain text writes are still normalized to `sendMessage`
- the server validates an allowlist of outbound Telegram send methods before
  queueing a message item
- queued items keep a normalized summary `text` field for bridge transcript and
  delivery logging purposes
- bridge-scoped requests parse rich-plan assistant text into Telegram delivery
  payloads, including optional formatting and `reply_markup`
- bridge-scoped rich-plan delivery defaults to plain text when `format` is
  omitted; explicit `html` remains the only rich parse mode emitted by the
  server fallback
- rich-plan `markdown` remains accepted for compatibility metadata and is now
  normalized into Telegram-supported HTML instead of being downgraded to plain
  text or emitted as `MarkdownV2`
- bridge-scoped plain assistant text is normalized into a `sendMessage` outbox
  item as the compatibility backstop, and the bridge reapplies that same
  Markdown-style normalization before Telegram delivery when plain message
  payloads still contain authored decorators such as `**`, `###`, or divider
  lines
- Telegram HTML is now the bridge's canonical user-facing text/caption format
  whenever explicit `entities` or `caption_entities` are absent; even payloads
  that arrive with missing or mismatched parse metadata are reduced into
  Telegram-supported HTML so raw tags do not leak into chat
- markdown pipe tables are normalized into aligned `<pre>` grids because
  Telegram HTML does not support table tags
- DeepSeek bridge-scoped replies that place DSML
  `<｜DSML｜function_calls>` markup in assistant `content` are now recovered
  back into structured runtime tool calls server-side, and the bridge strips
  any surviving DSML blocks from outgoing text/caption payloads as a fail-closed
  delivery guard
- if a bridge-scoped request exhausts its runtime-tool round budget without
  reaching a direct reply, the server now runs one explicit tool-free synthesis
  pass and falls back to a local clarification reply instead of returning HTTP
  500
- bridge-scoped traces now persist `live_generation_start_block_index` so the
  bridge can render only the keyframes generated for the current user-facing
  reply
- queued bridge-scoped `kind=message` items may include an additive
  `emotive_animation` bundle with stable dimension labels, Fibonacci-sphere
  directions, live-only keyframes, exact consecutive compaction, and hold
  metadata
- the bridge owns deterministic render, `ffmpeg` encode, and `sendVideo`
  primary delivery for that bundle when the source item is a `sendMessage`;
  it starts rendering in parallel with delivery planning, uses a single
  spoilered caption-above-media video when the formatted reply fits within
  Telegram's caption limit, switches to formatted text plus a separate reply
  video when the caption would overflow, and falls back to text-only delivery
  only on render, encode, or upload failure while keeping the current
  black-stage neon membrane scene with one fixed anchor per emotive dimension,
  smooth inward sag between anchors, strong `r/3 -> 5r/3` register-radius fidelity,
  and broad overlapping positive/negative shoulder fields that keep the full
  topology disturbed
- bridge state now carries the last successfully rendered terminal emotive
  moment per Telegram conversation and prepends it to the next bundle, so each
  clip starts from the prior end pose instead of replaying older keyframes
- bundle version `2` uses `0.5s` per raw emotive slot and `30fps`; repeated
  identical slots become holds rather than interpolated fake motion
- the live bridge should resolve `TELEGRAM_BRIDGE_FFMPEG_BIN` explicitly; the
  default encoder for that path is `h264_nvenc`, override with
  `TELEGRAM_BRIDGE_FFMPEG_VIDEO_ENCODER` only when a non-NVIDIA host is
  intentional
- the bridge may now target `TELEGRAM_BRIDGE_RENDER_BACKEND=chromium_webgl`
  and forward render jobs to `VICUNA_WEBGL_RENDERER_URL`; that localhost
  service owns the headless Chromium browser, must resolve its own Chromium and
  ffmpeg binaries explicitly, and should fail closed when only software WebGL
  rendering is available in mandatory-GPU mode
- invalid rich-plan metadata is stripped server-side while preserving the
  message body and trace visibility
- bridge-scoped requests own their runtime context inside the server; the
  bridge remains transport and state middleware only

Interleaved-thinking policy:

- replay assistant `reasoning_content` only for active assistant messages that also carry `tool_calls`
- seed the emotive runtime from ordered request history, not just user text
- derive the continuation VAD and metacognitive policy from that request-scoped
  replay state
- inject the policy, heuristic, and VAD sentences as separate `system`
  messages so tool payloads stay unchanged
- preserve `reasoning_content` exactly while adding any VAD or heuristic guidance messages
- keep DeepSeek serving fixed to `deepseek-chat`
- let the CPU-side control policy choose `thinking={"type":"enabled"}` or
  `thinking={"type":"disabled"}` per turn
- derive the default outbound token budget from the control-policy reasoning
  depth unless the caller already supplied an explicit cap
- when `thinking` is disabled, map bounded sampling profiles onto `temperature`
  or `top_p` and bounded repetition profiles onto `frequency_penalty` and
  `presence_penalty`
- when `thinking` is enabled, suppress those sampling and repetition fields so
  request shaping matches the DeepSeek docs
- use bounded prefix and stop profiles for synthesis or replan turns when
  beta prefix-completion preconditions are satisfied
- reuse one configured `cpp-httplib` client per DeepSeek authority and expose
  its build/reuse counters at `/health -> provider -> transport`
- retain a bounded structured request-trace registry with labeled JSON events
  across runtime handling, provider traffic, staged selection, runtime tool
  execution, Telegram delivery, and background replay/task flows
- inspect request-trace summary counters at `/health -> request_traces`
- inspect retained request-trace events at `GET /v1/debug/request-traces`
- retain exact `reasoning_content` and visible `content` strings on completed
  provider events so failed staged turns can be inspected verbatim
- emit an explicit `runtime_guidance/guidance_evaluated` trace event that
  records injected policy/VAD/heuristic guidance text or the skip reason
- emit an explicit `control_policy/policy_computed` trace event for each
  provider pass
- persist `final_policy` and `heuristic_retrieval` into each retained emotive
  trace
- retain `applied_provider_controls` in the final policy and policy-transition
  export so offline analysis can compare requested versus actually applied
  request shaping

RL runtime policy surface:

- capture one typed transition per completed provider pass when
  `VICUNA_POLICY_MODE` is `capture`, `shadow`, `eval_only`, `native_only`, or
  `canary_live`
- keep the executed action native unless `canary_live` samples the request and
  the candidate passes confidence and native safety validation
- include the richer provider-control heads in the action schema and training
  contract so offline learners can reason about `thinking`, prefix, stop,
  sampling, repetition, and tool-choice profiles directly
- expose bounded counters and runtime health at `/v1/policy/status` and
  `/health -> policy_runtime`
- expose the retained transition window at `GET /v1/policy/transitions`
- compute reward around one explicit desired-state target over all 14 emotive
  dimensions plus VAD, retain the typed reward model and reward breakdown on
  every transition, and fail startup on invalid
  `VICUNA_POLICY_REWARD_CONFIG_PATH` overrides
- fail closed to native execution on candidate lookup timeout, HTTP failure, or
  invalid proposal payload
- advance canary share through explicit configured steps and roll back to
  native-only execution when failure, invalid-action, or fallback rates exceed
  thresholds

Offline policy-learning pipeline:

- keep it outside the request path and free of new mandatory trainer
  dependencies
- treat `/v1/policy/transitions` plus `/v1/policy/status` as the source of
  truth for durable dataset export
- validate candidate policies offline against masks before trusting shadow or
  future rollout work
- keep trained artifacts and alias promotion explicit in a local file-backed
  registry rather than hidden service state
- treat the nightly batch as offline-only orchestration; it may refresh the
  `candidate` alias but live serving still consumes it only through the
  explicit rollout controller
- expose trained aliases through `tools/policy-learning/policy_registry_server.py`
  rather than embedding registry state into the C++ server

Flattened runtime-tool policy:

- expose the flattened runtime tool catalog directly to the provider
- keep backend dispatch explicit in CPU-side runtime code
- map flattened provider-visible tools onto the existing runtime wrappers
- support `media_read`, `media_download`, `media_delete`,
  `hard_memory_read`, `hard_memory_write`, `web_search`,
  `ongoing_task_create`, `ongoing_task_delete`, and `host_shell`
- allow metacognitive policy to decide whether direct tool use should be light,
  heavy, sequential, or parallel
- expect live callers to inject the authoritative direct tool definitions for
  that turn unless the request is the retained bridge-scoped surface
- do not reintroduce bridge-owned prompt construction, live-tool catalogs, or
  second-pass tool continuation loops
- the retained Telegram bridge should forward transcript plus routing headers
  only; the server is the sole owner of Telegram prompt and tool policy
- cache the retained bridge-scoped runtime tool catalog in memory; inspect its
  counters at
  `/health -> bridge_runtime`
- source one durable host env file from `VICUNA_SYSTEM_ENV_FILE` or
  `/etc/vicuna/vicuna.env` before startup
- keep `hard_memory_read` and `hard_memory_write` on the local markdown store
  rooted at `VICUNA_HARD_MEMORY_DIR`, defaulting to
  `/home/vicuna/home/memories` for host installs and
  `.cache/vicuna/host-shell-home/memories` for local/dev runs
- keep `skill_read` and `skill_create` on the local markdown store rooted at
  `VICUNA_SKILLS_DIR`, defaulting to `/home/vicuna/home/skills` for host
  installs and `.cache/vicuna/host-shell-home/skills` for local/dev runs
- persist replay-derived heuristics at
  `VICUNA_HEURISTIC_MEMORY_PATH=/home/vicuna/home/heuristics/vicuna-heuristic-memory.json`
  by default on host installs
- rehydrate stable OpenClaw secrets and runtime-catalog files outside the
  checkout with `tools/ops/sync-openclaw-runtime-state.sh`
- if `tools/openclaw-harness/dist/index.js` is missing on a fresh checkout,
  require `tools/ops/sync-openclaw-runtime-state.sh` to bootstrap the harness
  or fail startup explicitly before Telegram traffic begins
- keep Radarr, Sonarr, Chaptarr, Tavily, and the cron/docs local backends in
  that host env path instead of repo-local `.envrc` only
- when probing runtime tools manually, export
  `VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH` and
  `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH` to match the running service or
  the harness may fall back to the checkout-local `.cache` path and report
  false missing-key errors
- retain `VICUNA_ENABLE_STAGED_TOOL_FALLBACK` only for compatibility coverage;
  the staged selector path is no longer the default runtime
- treat `host_shell` as a last-resort fallback only; prefer specialized tools
  for media, web, memory, and task work, and keep its workspace rooted at
  `VICUNA_HOST_SHELL_ROOT` so host-side file mutations stay confined and
  inspectable

Runtime-tool catalog validation:

- run `node --test tools/ops/sync-openclaw-runtime-state.test.mjs` to verify a
  clean checkout can materialize `tools/openclaw-harness/dist/index.js`
- run `tools/ops/rebuild-vicuna-runtime.sh` for the supported rebuild path that
  now prepares the OpenClaw harness before restarting the runtime service

Cognitive replay policy:

- admit bounded replay entries only from non-replay traces with explicit
  negative-mass, VAD-drop, persistence, and control-failure gates
- start replay only after foreground idleness and only one replay job at a time
- run replay through the existing provider/emotive path so interleaved VAD/tool handling still applies
- mark replay traces as `cognitive_replay` and suppress recursive replay admission
- resolve or defer entries by comparing assistant-generated replay blocks against the original episode, excluding seeded replay prompts and trailing runtime events
- after a replay resolves, run one labeled bad-path vs better-path compression request before moving to the next entry
- persist the resulting `(bad path, better path, heuristic)` object to bounded disk-backed hard memory
- inspect registry and latest replay result state at `GET /v1/emotive/cognitive-replay`

Heuristic-memory policy:

- store searchable bad-path thought/message objects with embeddings inside each persisted record
- use exact cosine or lexical fallback over the bounded record set instead of ANN indexing
- rerank semantic candidates with structural-tag overlap and emotive-signature similarity
- inject only the matched heuristic, never the full stored replay narrative
- derive bounded control biases from the matched heuristic so prior failures can
  shift routing, reasoning depth, tool aggression, or stop thresholds without
  bypassing live recomputation
- inspect persisted records and the latest retrieval decision at `GET /v1/emotive/heuristics`

Recurring cron-task policy:

- install recurring work as `vicuna`-owned cron entries instead of a server
  idle-stage registry
- keep stable task metadata under `VICUNA_ONGOING_TASKS_DIR` only for explicit
  create/delete/edit/execute bookkeeping
- the retained cron runner must load the canonical env, take a per-task lock,
  and post the stored task text back to the normal provider surface as a
  `system` message
- the server should not expose `/v1/emotive/ongoing-tasks` or a second
  background due-decision worker
