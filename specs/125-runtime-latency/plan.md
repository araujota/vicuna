# Implementation Plan: Runtime Latency Hot Path Reduction

**Branch**: `125-runtime-latency` | **Date**: 2026-03-25 | **Spec**: [/Users/tyleraraujo/vicuna/specs/125-runtime-latency/spec.md](/Users/tyleraraujo/vicuna/specs/125-runtime-latency/spec.md)
**Input**: Feature specification from `/specs/125-runtime-latency/spec.md`

## Summary

Reduce avoidable latency in the provider-first runtime without changing the
staged-turn architecture by reusing a persistent DeepSeek HTTP client,
introducing explicit in-memory caches for the bridge-scoped Telegram runtime
tool catalog and its derived staged metadata, tightening the bridge's retained
internal polling loops, and fixing the provider sampling policy at
`temperature: 0.2` for every outbound turn.

## Technical Context

**Language/Version**: C++17 for runtime code, Node.js ESM for the Telegram bridge, Python 3 for unit tests  
**Primary Dependencies**: `cpp-httplib`, `nlohmann::json`, Node fetch/http modules, pytest, node:test  
**Storage**: In-memory bounded caches only; no new persisted storage  
**Testing**: `pytest` provider-mode unit tests and `node --test` bridge tests  
**Target Platform**: Linux/macOS provider-first server with retained Telegram bridge  
**Project Type**: native HTTP service plus retained transport bridge  
**Performance Goals**: remove repeated provider client construction, eliminate repeated bridge catalog subprocess cost on unchanged input, and reduce bridge tail latency from polling sleeps  
**Constraints**: preserve explicit CPU-side policy, keep staged round trips intact, avoid hidden ANN/pooling infrastructure, keep cache invalidation inspectable, and keep provider sampling policy explicit  
**Scale/Scope**: one provider adapter, one server-owned Telegram tool path, one retained bridge loop, and associated docs/tests

## Constitution Check

- **Runtime Policy**: Pass. Client reuse, cache keys, and polling policy remain
  explicit in `server-deepseek.*`, `server.cpp`, and bridge code.
- **Typed State**: Pass. New typed cache/client holders will be added instead of
  opaque globals or ad hoc JSON reuse.
- **Bounded Memory**: Pass. The design keeps at most one configured DeepSeek
  client and one current Telegram tool/metadata cache entry in memory.
- **Validation**: Pass. Provider tests, bridge tests, and build validation will
  prove the changed hot paths without altering core request semantics.
- **Documentation & Scope**: Pass. Runtime and bridge docs will be updated in
  the same change; no new external service dependency is introduced.

## Project Structure

### Documentation (this feature)

```text
specs/125-runtime-latency/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

### Source Code (repository root)

```text
tools/
├── server/
│   ├── server.cpp
│   ├── server-deepseek.cpp
│   ├── server-deepseek.h
│   ├── server-runtime.cpp
│   ├── README.md
│   ├── README-dev.md
│   └── tests/unit/test_deepseek_provider.py
└── telegram-bridge/
    ├── index.mjs
    ├── bridge.test.mjs
    └── README.md
```

**Structure Decision**: Keep the work inside the provider adapter, the
server-owned Telegram runtime path, and the retained bridge loops. No new
service or standalone cache layer is justified.

## Phase 0: Research

- Confirm `cpp-httplib` client reuse expectations and keep-alive support from
  upstream documentation.
- Confirm current runtime hotspots from local code inspection:
  per-request `server_http_client(...)`, per-turn Telegram runtime catalog
  subprocess, and staged metadata rebuilds.
- Confirm current bridge polling and reconnect delays from local bridge code.

## Phase 1: Design

- Add an explicit persistent DeepSeek client holder keyed by the configured base
  URL and guarded for reuse.
- Add an explicit Telegram runtime tool cache keyed by the authoritative tool
  payload, plus a sibling cache for the derived staged catalog.
- Refactor bridge-scoped prompt assembly to consume the cached staged metadata
  instead of rebuilding the catalog structure on every request.
- Tighten bridge outbox polling, self-emit reconnect, and watchdog intervals
  while keeping Telegram long-poll semantics unchanged.
- Make the DeepSeek request builder stamp `temperature: 0.2` on every outbound
  provider body regardless of caller surface.

## Complexity Tracking

No constitution violations or justified complexity exceptions.
