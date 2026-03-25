# Provider Server Development Notes

Current retained components:

- `server.cpp`: route table and provider request flow
- `server-http.cpp`: HTTP transport and middleware
- `server-common.cpp`: JSON/error/SSE helpers
- `server-deepseek.cpp`: DeepSeek request and response mapping
- `server-emotive-runtime.cpp`: block-wise emotive moment and VAD projection
- `server-embedding-backend.cpp`: optional `Qwen3-Embedding-0.6B-GGUF` backend

Current direction:

- provider-first request handling only
- internal streaming for block-wise emotive capture
- staged family -> method -> payload tool orchestration for auto-tool requests
- active tool continuations preserve assistant `reasoning_content`
- request-scoped VAD guidance is injected additively after the newest tool-result span
- heuristic retrieval runs in parallel with request-scoped guidance assembly
- matching heuristics are injected as one additive critical-guidance `system` message
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
- bridge-scoped Telegram requests may now resolve `telegram_relay` internally:
  the server queues the outbox item itself, returns `vicuna_telegram_delivery`,
  and suppresses outward tool calls so the bridge only delivers from outbox
- bridge-scoped plain assistant text is normalized into a `sendMessage` outbox
  item as a compatibility backstop instead of being allowed to drop

Interleaved-thinking policy:

- replay assistant `reasoning_content` only for active assistant messages that also carry `tool_calls`
- seed the emotive runtime from ordered request history, not just user text
- derive the continuation VAD from that request-scoped replay state
- inject the VAD sentence as a separate `system` message so tool payloads stay unchanged
- preserve `reasoning_content` exactly while adding any VAD or heuristic guidance messages
- pass through DeepSeek's top-level `thinking` field when callers provide it
- force every outbound DeepSeek request, including staged and background turns, to use `max_tokens: 1024`
- ignore caller-supplied `max_tokens`, `max_completion_tokens`, and `max_output_tokens` values that differ from the fixed runtime cap

Staged tool-loop policy:

- intercept OpenAI-compatible requests that include `tools` and auto tool choice
- normalize those flat tools into family, method, and typed-contract layers
- ask the provider for one JSON family choice, then one JSON method choice, then one JSON payload
- allow `back` from method and payload stages and `complete` from method selection
- after payload validation, emit one normal OpenAI tool call back to the caller
- when `complete` is chosen, run one final direct-response turn without tools
- if DeepSeek JSON mode returns empty or malformed family/method output, retry that staged selector once with the validation error injected back into the prompt
- prefer explicit request-tool metadata:
  - `x-vicuna-family-id`
  - `x-vicuna-family-name`
  - `x-vicuna-family-description`
  - `x-vicuna-method-name`
  - `x-vicuna-method-description`
- require typed field descriptions throughout the method contract so payload prompts stay inspectable
- expect live callers to inject the authoritative direct tool definitions for
  that turn; do not reintroduce a second hidden live-tool catalog inside the
  server
- the Telegram bridge now follows that rule by injecting the full installed
  OpenClaw runtime catalog directly and executing returned tool calls itself

Cognitive replay policy:

- admit bounded replay entries only from non-replay traces with explicit negative-mass, VAD-drop, and persistence gates
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
- inspect persisted records and the latest retrieval decision at `GET /v1/emotive/heuristics`

Pre-idle ongoing-task policy:

- once replay work is exhausted for an idle cycle, poll the ongoing-task hard-memory registry before allowing true idle
- load current system time plus active task summaries and run one explicit due-decision prompt
- keep ongoing-task decision and execution on the normal provider/emotive path so additive VAD and heuristic guidance still apply
- launch execution with the exact stored `task_text`, not a rewritten paraphrase
- suppress replay admission for both `ongoing_task_decision` and `ongoing_task_execution` traces
- advance `last_done_at` only after the selected background task completes successfully
- inspect worker state at `GET /v1/emotive/ongoing-tasks` and under `/health -> emotive_runtime -> ongoing_tasks`
