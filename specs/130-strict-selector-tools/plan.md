# Implementation Plan: Strict Selector Tools And Waterfall VAD

**Branch**: `codex/130-strict-selector-tools` | **Date**: 2026-03-26 | **Spec**: [/Users/tyleraraujo/vicuna/specs/130-strict-selector-tools/spec.md](/Users/tyleraraujo/vicuna/specs/130-strict-selector-tools/spec.md)
**Input**: Feature specification from `/specs/130-strict-selector-tools/spec.md`

## Summary

Replace staged freeform JSON selector turns with DeepSeek beta strict tool
calls using enum-constrained arguments, keep the family -> method -> payload
waterfall explicit in `server.cpp`, and fix additive VAD injection so each
staged step after prior reasoning receives renewed guidance.

## Technical Context

**Language/Version**: C++17 runtime, Python 3 pytest  
**Primary Dependencies**: existing DeepSeek adapter, `nlohmann::json`, request-trace subsystem  
**Storage**: in-memory staged prompt cache and request-trace ring buffer  
**Testing**: `tools/server/tests/unit/test_deepseek_provider.py`, runtime build  
**Target Platform**: provider-first runtime on Linux/macOS and deployed Linux host  
**Project Type**: native HTTP service with staged provider orchestration  
**Performance Goals**: improve selector termination reliability without removing reasoning mode; preserve cached prompt/tool assembly where possible  
**Constraints**: keep the staged controller explicit and inspectable; preserve reasoning capture; stay within DeepSeek strict-schema subset; do not collapse the waterfall into one hidden provider step  
**Scale/Scope**: `server.cpp`, `server-deepseek.cpp`, tests, and operator/dev docs

## Research Notes

- DeepSeek strict tool mode supports `enum` and works with thinking mode, which
  is a better fit for selector turns than freeform JSON mode.
- DeepSeek JSON mode explicitly warns that it may return empty content, which
  matches the observed failures in the live Telegram flow.
- The current VAD injection path depends on a classic assistant/tool-result
  span and therefore skips staged turns even though the builder state updates.

## Constitution Check

- **Runtime Policy**: Pass. The CPU-side staged controller remains explicit and
  becomes more inspectable because each selector stage is represented by a
  concrete tool schema.
- **Typed State**: Pass. Selector decisions move from hand-parsed text to
  strongly typed tool-call arguments.
- **Bounded Memory/Performance**: Pass. Prompt/tool caches remain bounded and
  the request-trace ring buffer remains bounded.
- **Validation**: Pass. Provider tests will exercise strict tool payloads,
  retries, and per-stage VAD injection.
- **Documentation & Scope**: Pass. Docs will describe beta strict-tool routing
  and staged VAD behavior.

## Project Structure

### Documentation (this feature)

```text
specs/130-strict-selector-tools/
├── spec.md
├── research.md
├── plan.md
└── tasks.md
```

### Source Code

```text
tools/server/
├── server.cpp
├── server-deepseek.cpp
├── README.md
├── README-dev.md
└── tests/unit/test_deepseek_provider.py
```

## Implementation Strategy

1. Introduce server-owned strict selector tool schemas and prompt-bundle cache
   helpers for family, method, and payload stages.
2. Update the staged loop to send those selector tools through the provider
   using DeepSeek beta strict tool mode and parse the resulting tool-call
   arguments rather than `content`.
3. Keep staged retries and sentinels explicit while making prompt prefixes
   stable across repeated attempts.
4. Add staged-turn VAD eligibility so every step after previous reasoning can
   inject renewed additive guidance and trace it explicitly.
5. Update provider tests and docs to cover strict selector tools, beta routing,
   and per-stage VAD propagation.

## Complexity Tracking

No constitution violations or justified complexity exceptions.
