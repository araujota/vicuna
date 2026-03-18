# Implementation Plan: Telegram Bridge Middleware

**Branch**: `049-host-build-bringup` | **Date**: 2026-03-17 | **Spec**: [spec.md](/home/tyler-araujo/Projects/vicuna/specs/050-telegram-bridge/spec.md)

## Summary

Add a small Node-based bridge service that long-polls Telegram for inbound
messages, forwards user text into the running Vicuña OpenAI-compatible API, and
subscribes to `/v1/responses/stream` so proactive self-emits are relayed to
Telegram through the same middleware process.

## Technical Context

**Language/Version**: Node.js 20 ESM  
**Primary Dependencies**: Built-in `fetch`, built-in `node:test`, existing repo
Node runtime  
**Target Platform**: Local Ubuntu host running the GPU-enabled
`llama-server` on `127.0.0.1:8080`  
**State Storage**: Local JSON file under `/tmp` by default  
**Validation**: Unit tests for bridge helpers, runtime smoke test against local
Vicuña server, live Telegram bridge process startup

## Constitution Check

- **Runtime Policy**: The bridge observes runtime policy via the server’s
  existing HTTP API; no hidden runtime policy is added in C++.
- **Typed State**: The bridge consumes server-produced OpenAI Responses events
  without changing typed self-state or hard-memory representations.
- **Bounded Memory**: Bridge persistence is bounded to cursors, subscribers, and
  a capped dedupe list for proactive response IDs.
- **Validation**: Local tests and a live process startup are required.
- **Documentation & Scope**: Operator docs and env configuration are updated.

## Implementation Phases

### Phase 1: Bridge Contract

- Confirm the existing Vicuña endpoints used for user requests and proactive
  self-emits.
- Define the bridge state model, Telegram polling flow, and self-emission relay
  strategy.

### Phase 2: Bridge Implementation

- Implement Telegram long polling, inbound message handling, and outbound reply
  delivery.
- Implement a resilient SSE consumer for `/v1/responses/stream`.
- Persist Telegram offsets, known chats, and proactive dedupe state.

### Phase 3: Validation And Docs

- Add helper-level tests for event parsing and text extraction.
- Document required env vars and startup commands.
- Run the bridge against the live GPU-enabled Vicuña server.

## Risks

- Telegram network access may require elevated host execution in this session.
- The proactive stream allows only one live subscriber, so the bridge must own
  that subscription exclusively.
- Duplicate proactive delivery could occur if the bridge restarts without a
  dedupe record.
