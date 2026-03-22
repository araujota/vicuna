# Implementation Plan: Retry Grounding Surgery

**Branch**: `[070-truth-runtime-refactor]` | **Date**: 2026-03-22 | **Spec**: [/Users/tyleraraujo/vicuna/specs/074-retry-grounding-surgery/spec.md](/Users/tyleraraujo/vicuna/specs/074-retry-grounding-surgery/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/074-retry-grounding-surgery/spec.md`

## Summary

Patch the authoritative ReAct server loop so fresh mutable Telegram turns do not trust stale assistant dialogue as grounding, then add bounded retry escalation that forces one tool call after repeated rejected non-tool retries. Keep the current continuation loop, resumed tool-result flow, and CPU-side non-selection guarantees intact.

## Technical Context

**Language/Version**: C++17  
**Primary Dependencies**: Existing Vicuña server runtime, authoritative ReAct prompt builder, OpenClaw tool fabric, Telegram dialogue history surfaces  
**Storage**: In-memory runtime state plus persisted runtime snapshot JSON  
**Testing**: Native C++ regression tests plus host rebuild and runtime journal trace  
**Target Platform**: Linux host runtime and local native build environment  
**Project Type**: Native inference runtime and server  
**Performance Goals**: Eliminate long retry spirals on mutable Telegram turns without adding new heavyweight passes  
**Constraints**: Keep runtime policy explicit and inspectable, preserve ReAct ownership of action/tool choice, avoid broad prompt rewrites  
**Scale/Scope**: Server grounding heuristic, retry escalation, targeted tests, docs, and host validation

## Constitution Check

- **Runtime Policy**: Pass. Grounding trust and retry escalation stay explicit in CPU-side control code.
- **Typed State**: Pass. The change uses existing `server_task` retry and origin fields; no hidden side-channel state is introduced.
- **Bounded Memory**: Pass. Telegram dialogue stays bounded and is merely reclassified for trust on fresh mutable turns.
- **Validation**: Pass. Add targeted regression tests and verify on the host with a Telegram-shaped live-fact request.
- **Documentation & Scope**: Pass. Update architecture and server docs with the new trusted-grounding hierarchy and escalation rule.

## Implementation Phases

### Phase 1: Policy Surfaces

- Add helpers for identifying mutable active retry escalation and trusted grounding sources.
- Split grounding evaluation so Telegram assistant dialogue can be excluded for fresh mutable turns without affecting resumed tool-result turns.

### Phase 2: Runtime Wiring

- Update the grounding candidate logic in `server-context.cpp`.
- Escalate tool choice from `auto` to `required` after repeated rejected non-tool retries on mutable active turns.
- Strengthen retry feedback for the escalated state.

### Phase 3: Validation and Documentation

- Extend targeted tests for trusted grounding and escalation.
- Update `ARCHITECTURE.md` and `tools/server/README-dev.md`.
- Rebuild on the host and verify the problematic weather-style Telegram path uses a tool early instead of looping.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Retry escalation threshold | Needed to keep unbounded continuation from turning into minute-long local spin loops | Prompt-only wording already failed repeatedly on the live host |
