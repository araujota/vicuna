# Implementation Plan: Authoritative ReAct Continuation

**Branch**: `[073-authoritative-react-continuation]` | **Date**: 2026-03-22 | **Spec**: [/Users/tyleraraujo/vicuna/specs/073-authoritative-react-continuation/spec.md](/Users/tyleraraujo/vicuna/specs/073-authoritative-react-continuation/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/073-authoritative-react-continuation/spec.md`

## Summary

Replace the current “first valid answer ends the turn” behavior with an explicit authoritative ReAct continuation policy. The server will keep active and DMN turns in the same authoritative loop until a valid terminal condition is reached, require a tool-first first step for clearly mutable/live user requests, reject procedural non-answers, and preserve direct answers only when the request is stable or already grounded in canonical context.

## Technical Context

**Language/Version**: C++17  
**Primary Dependencies**: Existing Vicuña runtime, `llama-server`, OpenClaw tool fabric, common chat-template/tool-choice machinery  
**Storage**: In-memory runtime state plus persisted runtime snapshot JSON  
**Testing**: Native C++ test targets under `tests/`, host rebuild and journal validation  
**Target Platform**: Linux host runtime plus local native build environment  
**Project Type**: Native inference runtime and server  
**Performance Goals**: Preserve current single-turn latency for stable direct answers while allowing unbounded continuation for unresolved tool-grounded turns  
**Constraints**: Keep runtime policy explicit in CPU control code, do not reintroduce CPU-side tool selection, preserve bounded context and existing tool resume flow  
**Scale/Scope**: Active Telegram-facing ReAct path plus shared authoritative control surfaces used by DMN

## Constitution Check

- **Runtime Policy**: Pass. The design keeps continuation, grounding checks, and retry behavior in explicit CPU-side server control code instead of implicit prompt hope.
- **Typed State**: `server_task` will gain explicit foreground text and unbounded continuation defaults; existing authoritative turn structs remain the typed source of truth.
- **Bounded Memory**: No new unbounded memory surface is introduced. The loop remains unbounded in steps, but state stays on the existing task/trace/context surfaces.
- **Validation**: Add native regression tests for grounding detection and procedural-answer rejection, plus host rebuild and Telegram weather trace validation.
- **Documentation & Scope**: Update `ARCHITECTURE.md`, `tools/server/README-dev.md`, and the new 073 spec artifacts.

## Research Summary

- Local code audit shows `tools/server/server-context.cpp` only retries on parse or validation failure and resumes the loop after tool results; a first accepted `answer` ends the active turn immediately.
- The current prompt builder already supports multi-step tool follow-up after observations, so the gap is terminal validation and first-step tool forcing, not missing resume plumbing.
- Official OpenAI function-calling guidance documents the standard multi-step pattern: send tool results back to the model and continue until a final response is produced, with `tool_choice: "required"` available when a tool call must happen.
- Anthropic’s stop-reason guidance similarly treats tool use as a loop that continues after tool execution instead of treating the first natural-language span as final.
- Hugging Face `smolagents` uses an explicit step loop that continues until a final-answer condition is met rather than treating the first assistant text as terminal.

## Project Structure

### Documentation (this feature)

```text
specs/073-authoritative-react-continuation/
├── plan.md
├── research.md
├── tasks.md
```

### Source Code (repository root)

```text
tools/server/
├── server-common.cpp
├── server-common.h
├── server-context.cpp
└── server-task.h

tests/
├── test-server-common.cpp
└── test-openclaw-tool-fabric.cpp
```

**Structure Decision**: Keep all runtime policy changes in the existing server control layer. Use `server-common.*` for pure policy helpers that can be unit tested, and `server-context.cpp` for authoritative loop continuation and prompt wiring.

## Implementation Phases

### Phase 0: Research and Policy Definition

- Capture the current failure mode from host logs and local code audit.
- Document the gap between current behavior and ReAct/tool-loop research.

### Phase 1: Explicit Continuation Surfaces

- Persist foreground turn text on `server_task`.
- Add explicit helper functions for:
  - mutable/live request detection
  - procedural meta-answer rejection
  - first-step tool-grounding requirement

### Phase 2: Authoritative Loop Continuation

- Switch first active step tool choice to `required` when the latest turn clearly needs fresh tool grounding.
- Add semantic terminal validation after parse but before final response dispatch.
- Convert unsupported terminal answers into same-turn retries with feedback.
- Remove the small fixed retry cap and default to effectively unbounded continuation.

### Phase 3: Validation and Documentation

- Extend native tests for the new helper policy.
- Add focused authoritative ReAct regression coverage where possible.
- Update architecture and operator docs.
- Rebuild and validate on the host with a live weather-style Telegram request.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Explicit semantic validator after parse | Needed to distinguish substantive terminal answers from procedural non-answers | Prompt-only steering already failed in production and still accepted unsupported `answer` actions |
