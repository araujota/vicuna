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
- flattened runtime tool exposure for media, hard memory, web search, and
  ongoing-task lifecycle operations
- active tool continuations preserve assistant `reasoning_content`
- request-scoped policy, heuristic, and VAD guidance is injected additively
  after the newest tool-result span
- heuristic retrieval runs in parallel with request-scoped guidance assembly
- matching heuristics are injected as one additive critical-guidance `system`
  message and may also bias the control policy
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
- bridge-scoped plain assistant text is normalized into a `sendMessage` outbox
  item as the compatibility backstop
- bridge-scoped traces now persist `live_generation_start_block_index` so the
  bridge can render only the keyframes generated for the current user-facing
  reply
- queued bridge-scoped `kind=message` items may include an additive
  `emotive_animation` bundle with stable dimension labels, Fibonacci-sphere
  directions, and live-only keyframes
- the bridge owns deterministic render, `ffmpeg` encode, and `sendVideo`
  follow-up delivery for that bundle; text delivery remains canonical and must
  survive animation failures
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
- default DeepSeek provider requests to `deepseek-chat` with `thinking={"type":"enabled"}`
- pass through DeepSeek's top-level `thinking` field unchanged when callers provide it
- derive the default outbound token budget from the control-policy reasoning
  depth unless the caller already supplied an explicit cap
- force every outbound DeepSeek request, including staged and background turns, to use `temperature: 0.2`
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

Flattened runtime-tool policy:

- expose the flattened runtime tool catalog directly to the provider
- keep backend dispatch explicit in CPU-side runtime code
- map flattened provider-visible tools onto the existing runtime wrappers
- support `media_read`, `media_download`, `media_delete`,
  `hard_memory_read`, `hard_memory_write`, `web_search`,
  `ongoing_task_create`, and `ongoing_task_delete`
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
- rehydrate stable OpenClaw secrets and runtime-catalog files outside the
  checkout with `tools/ops/sync-openclaw-runtime-state.sh`
- keep Radarr, Sonarr, Chaptarr, Tavily, and Supermemory configuration in that
  host env path instead of repo-local `.envrc` only
- when probing runtime tools manually, export
  `VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH` and
  `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH` to match the running service or
  the harness may fall back to the checkout-local `.cache` path and report
  false missing-key errors
- retain `VICUNA_ENABLE_STAGED_TOOL_FALLBACK` only for compatibility coverage;
  the staged selector path is no longer the default runtime

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

Pre-idle ongoing-task policy:

- once replay work is exhausted for an idle cycle, poll the ongoing-task hard-memory registry before allowing true idle
- load current system time plus active task summaries and run one explicit due-decision prompt
- keep ongoing-task decision and execution on the normal provider/emotive path so additive VAD and heuristic guidance still apply
- launch execution with the exact stored `task_text`, not a rewritten paraphrase
- suppress replay admission for both `ongoing_task_decision` and `ongoing_task_execution` traces
- advance `last_done_at` only after the selected background task completes successfully
- inspect worker state at `GET /v1/emotive/ongoing-tasks` and under `/health -> emotive_runtime -> ongoing_tasks`
