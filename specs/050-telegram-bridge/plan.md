# Implementation Plan: Telegram Bridge Middleware

**Branch**: `049-host-build-bringup` | **Date**: 2026-03-17 | **Spec**: [spec.md](/home/tyler-araujo/Projects/vicuna/specs/050-telegram-bridge/spec.md)

## Summary

Add a small Node-based bridge service that long-polls Telegram for inbound
messages, forwards user text into the running Vicuña OpenAI-compatible API, and
subscribes to `/v1/responses/stream` so proactive self-emits are relayed to
Telegram through the same middleware process. Extend that bridge with bounded
per-chat transcript persistence keyed by Telegram `chat_id`, and make the
repo-owned runtime launcher expose the cognitive bash tool explicitly so
command-execution requests can work under managed startup.

## Technical Context

**Language/Version**: Node.js 20 ESM  
**Primary Dependencies**: Built-in `fetch`, built-in `node:test`, existing repo
Node runtime  
**Target Platform**: Local Ubuntu host running the GPU-enabled
`llama-server` on `127.0.0.1:8080`  
**State Storage**: Local JSON file under `/tmp` by default, including bounded
per-chat transcripts keyed by Telegram `chat_id`  
**Validation**: Unit tests for bridge helpers and transcript state, runtime
smoke test against local Vicuña server, live Telegram bridge process startup

## Constitution Check

- **Runtime Policy**: The bridge observes runtime policy via the server’s
  existing HTTP API; no hidden runtime policy is added in C++.
- **Typed State**: The bridge consumes server-produced OpenAI Responses events
  without changing typed self-state or hard-memory representations; bridge-side
  chat state remains an explicit JSON transcript keyed by chat ID.
- **Bounded Memory**: Bridge persistence is bounded to cursors, subscribers, a
  capped dedupe list for proactive response IDs, and capped per-chat
  transcripts.
- **Validation**: Local tests and a live process startup are required.
- **Documentation & Scope**: Operator docs and env configuration are updated.

## Implementation Phases

### Phase 1: Bridge Contract

- Confirm the existing Vicuña endpoints used for user requests and proactive
  self-emits.
- Define the bridge state model, including bounded per-chat transcript storage,
  Telegram polling flow, and self-emission relay strategy.
- Confirm how managed runtime startup exposes cognitive bash tool configuration.

### Phase 2: Bridge Implementation

- Implement Telegram long polling, inbound message handling, and outbound reply
  delivery.
- Persist and reuse bounded chat-completions transcripts per Telegram chat ID.
- Implement a resilient SSE consumer for `/v1/responses/stream`.
- Persist Telegram offsets, known chats, proactive dedupe state, and transcript
  state.
- Update the repo-owned runtime launcher to set explicit bash-tool environment
  variables for managed operation.

### Phase 3: Validation And Docs

- Add helper-level tests for transcript persistence/bounding, event parsing, and
  text extraction.
- Document required env vars, startup commands, and managed bash-tool
  configuration.
- Run the bridge against the live GPU-enabled Vicuña server.

## Risks

- Telegram network access may require elevated host execution in this session.
- The proactive stream allows only one live subscriber, so the bridge must own
  that subscription exclusively.
- Duplicate proactive delivery could occur if the bridge restarts without a
  dedupe record.
- Long Telegram conversations could grow without bound unless transcript
  trimming is explicit and deterministic.
- Runtime command execution can still fail if the loaded model never chooses the
  bash tool, even after launcher configuration is corrected.
