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
- a retained Telegram bridge transport that consumes the narrow compatibility
  endpoints exposed by the provider server

## Explicit Non-Goals

The active architecture no longer includes:

- local chat inference as the main serving path
- slot scheduling, KV-cache orchestration, or router mode as product features
- OpenClaw runtime catalogs or harness-driven tool orchestration
- DMN, self-state, hard-memory, or Active LoRA control loops as active product
  surfaces
- upstream `llama.cpp` example, benchmark, conversion, and distribution assets

## Request Flow

1. The server accepts an OpenAI-compatible request.
2. The request is normalized into a DeepSeek `/chat/completions` provider call.
3. Provider output is streamed internally even when the external response is
   non-streaming.
4. Reasoning and assistant text are captured block-by-block.
5. Each block updates the rich emotive moment.
6. The VAD projection is recomputed immediately from the full emotive vector.
7. The final response is returned with `vicuna_emotive_trace`.

## Bridge Surface

The server retains a narrow bridge compatibility surface:

- `GET /v1/responses/stream`
- `GET /v1/telegram/outbox`
- `POST /v1/telegram/outbox`
- `POST /v1/telegram/approval`
- `POST /v1/telegram/interruption`

Those endpoints exist only to support the retained Telegram bridge. They are
not a general return to the old local orchestration stack.

## Emotive Runtime

The emotive runtime is intentionally bounded and inspectable.

- input is segmented into retained blocks
- each block stores source metadata, text, the emotive moment, the delta from
  the previous block, and the projected VAD/style guidance
- traces are kept only for a bounded recent history window
- the estimator can operate in lexical-only mode or with local embedding
  enrichment

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
