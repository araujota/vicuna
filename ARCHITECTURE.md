# Vicuña Architecture

## Current Product

Vicuña now ships one primary runtime surface:

- a provider-first HTTP server backed by DeepSeek
- a bounded emotive-moment runtime that processes reasoning and content
  block-by-block
- an immediate VAD projection layer that turns the emotive moment into a
  reusable tone/style control surface
- an optional local GGUF embedding backend used only to enrich emotive
  estimation when local VRAM is available
- a bounded cognitive replay and heuristic-memory layer that converts negative
  traces into reusable control heuristics
- an explicit staged tool controller that expands auto-tool requests into
  family, method, and payload checkpoints before final tool-call emission
- a pre-idle ongoing-task stage that replays recurring user directives from a
  hard-memory-backed registry before true idle
- a retained Telegram bridge transport that consumes the narrow compatibility
  endpoints exposed by the provider server

## Explicit Non-Goals

The active architecture no longer includes:

- local chat inference as the main serving path
- slot scheduling, KV-cache orchestration, or router mode as product features
- hidden provider-owned tool selection policy
- DMN, self-state, hard-memory, or Active LoRA control loops as active product
  surfaces
- upstream `llama.cpp` example, benchmark, conversion, and distribution assets

## Request Flow

1. The server accepts an OpenAI-compatible request.
2. If the request includes automatic tools, the server normalizes them into
   family, method, and payload layers and runs explicit staged JSON turns.
3. Otherwise the request is normalized directly into a DeepSeek
   `/chat/completions` provider call.
4. Provider output is streamed internally even when the external response is
   non-streaming.
5. Reasoning and assistant text are captured block-by-block.
6. Each block updates the rich emotive moment.
7. The VAD projection is recomputed immediately from the full emotive vector.
8. If the request resumes a tool continuation, one additive VAD guidance
   message is inserted after the newest tool-result span.
9. In parallel, the latest trace window is compared against persisted bad-path
   memory and may inject one brief heuristic constraint.
10. The staged controller either emits one final validated tool call, moves
    back one stage, or finishes the loop and asks for a direct final response.
11. The final response is returned with `vicuna_emotive_trace`.
12. After foreground idleness, background work runs in order: cognitive replay
    first, then one ongoing-task due-decision stage, then true idle if nothing
    is selected.

## Bridge Surface

The server retains a narrow bridge compatibility surface:

- `GET /v1/responses/stream`
- `GET /v1/telegram/outbox`
- `POST /v1/telegram/outbox`
- `POST /v1/telegram/approval`
- `POST /v1/telegram/interruption`

Those endpoints exist only to support the retained Telegram bridge. They are
not a general return to the old local orchestration stack.

The bridge is transport/state middleware only:

- it forwards the bounded transcript plus Telegram routing headers
- it persists transcript, option-prompt, and outbox-delivery state
- it handles Telegram Bot API polling, callback transport, document ingestion,
  and outbox delivery
- it does not own Telegram prompt construction, runtime tool injection, or
  runtime tool continuation

## Emotive Runtime

The emotive runtime is intentionally bounded and inspectable.

- input is segmented into retained blocks
- each block stores source metadata, text, the emotive moment, the delta from
  the previous block, and the projected VAD/style guidance
- traces are kept only for a bounded recent history window
- the estimator can operate in lexical-only mode or with local embedding
  enrichment
- significant negative foreground traces can admit bounded cognitive replay
  entries
- resolved replay episodes are compressed into persisted heuristic-memory
  records for future retrieval
- background ongoing-task decision/execution traces keep the same emotive/VAD
  path but suppress replay admission so the idle loop stays bounded

## Staged Tool Controller

The staged controller keeps tool policy explicit in CPU-side code:

- tools are grouped into high-level families
- each family exposes named methods
- each method exposes one typed payload contract with field descriptions
- the provider sees only one stage at a time and must answer in strict JSON
- `back` and `complete` are controller transitions, not real tools
- final tool execution remains a normal tool call or runtime action after
  server-side validation
- bridge-scoped Telegram turns are a built-in server-owned variant: the server
  loads the installed runtime catalog, injects Telegram guidance plus
  `telegram_relay`, and continues runtime tool execution internally until final
  Telegram delivery or a direct final answer

## VAD Projection

The VAD surface is derived from the entire emotive-moment vector rather than a
small handpicked subset.

- all emotive dimensions contribute
- cross-terms capture important joint states
- EMA smoothing keeps the output stable across adjacent blocks
- the result includes trend, labels, dominant dimensions, and a prompt-ready
  style guide

## Build Boundary

The provider-first server remains C++17 and still links against the local
`llama` library only because the optional embedding backend uses the native
model/context APIs. That local embedding path is the only justified retained
dependency on the old native inference stack.

## Heuristic Replay

The replay subsystem is explicit and inspectable rather than latent:

- foreground traces with sharp negative deltas admit replay entries
- idle background replay searches for a better path without allowing recursive
  replay admission
- each resolved replay is compressed into one structured heuristic plus the
  original bad path and better path
- live requests retrieve against stored bad-path objects using bounded exact
  similarity and inject only the matched heuristic

## Pre-idle Ongoing Tasks

The idle worker has a second bounded stage after replay exhaustion:

- it polls the ongoing-task registry over hard-memory HTTP using the existing
  task contract
- it supplies current system time, cadence fields, due-state fields, and raw
  task wording to one due-decision prompt
- it launches at most one selected task by sending the exact stored task text
  through the normal provider/emotive request path
- it writes the updated `last_done_at` timestamp back to the registry only
  after that run succeeds
- it exposes typed state and the latest decision via `/health` and
  `GET /v1/emotive/ongoing-tasks`
