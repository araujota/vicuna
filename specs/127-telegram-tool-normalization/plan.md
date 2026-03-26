# Implementation Plan: Telegram Tool Normalization

**Branch**: `127-telegram-tool-normalization` | **Date**: 2026-03-26 | **Spec**: [/Users/tyleraraujo/vicuna/specs/127-telegram-tool-normalization/spec.md](/Users/tyleraraujo/vicuna/specs/127-telegram-tool-normalization/spec.md)
**Input**: Feature specification from `/specs/127-telegram-tool-normalization/spec.md`

## Summary

Normalize the retained Telegram bridge path so Telegram appears as one ordinary
staged family with explicit methods and typed payload contracts. Remove the
bridge-only Telegram system prompt, replace generic `telegram_relay` with a
small method set (`send_plain_text`, `send_formatted_text`, `send_photo`,
`send_document`, `send_poll`, `send_dice`), and translate those tool calls
internally into the existing Telegram outbox contract.

## Technical Context

**Language/Version**: C++17 runtime, Python 3 tests  
**Primary Dependencies**: `nlohmann::json`, existing DeepSeek adapter/runtime, pytest  
**Storage**: Existing in-memory Telegram outbox only; no new persistence  
**Testing**: `pytest` provider-mode unit tests plus runtime build validation  
**Target Platform**: Linux/macOS provider-first runtime with retained Telegram bridge  
**Project Type**: native HTTP service  
**Performance Goals**: reduce bridge-scoped family-stage prompt burden without reducing staged checkpoints  
**Constraints**: preserve staged family -> method -> payload narrowing, preserve internal Telegram outbox delivery, keep behavior explicit in CPU-side code, avoid reintroducing prompt-special Telegram logic  
**Scale/Scope**: server-owned Telegram tool family and provider tests only

## Constitution Check

- **Runtime Policy**: Pass. Telegram delivery behavior remains explicit in
  `server.cpp` through typed tool metadata and internal outbox translation.
- **Typed State**: Pass. The Telegram family will gain explicit method
  contracts and a deterministic mapping from tool-call name to Telegram outbox
  item fields.
- **Bounded Memory**: Pass. No new unbounded state is introduced.
- **Validation**: Pass. Provider tests will cover family prompt shape, method
  narrowing, and outbox translation.
- **Documentation & Scope**: Pass. Runtime docs and architecture notes will be
  updated; no new services or libraries are required.

## Project Structure

### Documentation (this feature)

```text
specs/127-telegram-tool-normalization/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

### Source Code (repository root)

```text
tools/
└── server/
    ├── server.cpp
    ├── README.md
    ├── README-dev.md
    └── tests/unit/test_deepseek_provider.py
```

**Structure Decision**: The change is runtime-local. The bridge remains thin
and does not need new logic for normalized Telegram methods.

## Phase 0: Research

- Confirm the current bridge-scoped Telegram prompt shape and failure mode from
  request traces.
- Review Telegram Bot API method groupings and existing metadata-driven tool
  patterns.

## Phase 1: Design

- Define the explicit Telegram staged methods and their payload contracts.
- Remove the bridge-only Telegram system prompt from request assembly.
- Replace generic relay parsing with per-method Telegram outbox translation.
- Update provider tests to assert the new family/method/payload shape.

## Complexity Tracking

No constitution violations or justified complexity exceptions.
