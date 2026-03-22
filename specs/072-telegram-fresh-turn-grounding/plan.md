# Implementation Plan: Telegram Fresh-Turn Grounding

**Branch**: `[070-truth-runtime-refactor]` | **Date**: 2026-03-22 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/072-telegram-fresh-turn-grounding/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/072-telegram-fresh-turn-grounding/spec.md`

## Summary

Fix the stale Telegram-turn regression by grounding active foreground extraction in the raw incoming request body and by reordering Telegram-scoped canonical ReAct prompt assembly so bounded Telegram dialogue remains the terminal conversational context. Validate with targeted server-unit coverage and live host verification.

## Technical Context

**Language/Version**: C++17, Node 20  
**Primary Dependencies**: existing `llama-server`, `common_chat_msg`, runtime Telegram bridge  
**Storage**: in-memory runtime state plus persisted runtime/bridge JSON state  
**Testing**: native C++ test targets plus bridge Node tests  
**Target Platform**: local macOS dev machine and Linux host runtime  
**Project Type**: native inference server + Telegram bridge  
**Performance Goals**: no material regression in active-turn prompt assembly or Telegram request handling  
**Constraints**: keep runtime policy explicit in CPU control code, preserve authoritative ReAct ownership, preserve bounded dialogue history  
**Scale/Scope**: narrow serving bug fix affecting Telegram-scoped active turns

## Constitution Check

- **Runtime Policy**: The fix stays in explicit CPU-side request handling and prompt assembly.
- **Typed State**: No new runtime state types are required; only request-source selection and prompt-ordering policy change.
- **Bounded Memory**: Existing bounded shared-context and bounded Telegram-dialogue surfaces remain intact.
- **Validation**: Add targeted regression tests, run bridge tests, and verify behavior on the host runtime.
- **Documentation & Scope**: Update architecture/spec artifacts if prompt-grounding policy changes.

## Project Structure

### Documentation

```text
specs/072-telegram-fresh-turn-grounding/
├── spec.md
├── plan.md
├── research.md
└── tasks.md
```

### Source Code

```text
tools/server/
├── server-common.cpp
├── server-common.h
└── server-context.cpp

tests/
├── test-openclaw-tool-fabric.cpp
└── test-server-common.cpp   # new targeted regression test
```

**Structure Decision**: Keep the fix in existing server request/prompt assembly code and add one focused native regression target for request-body foreground extraction.

## Phase Outline

1. Add research/spec artifacts and capture the stale-turn failure path.
2. Patch request foreground extraction to prefer raw `messages` from the incoming HTTP body.
3. Patch Telegram-scoped canonical ReAct message ordering so latest Telegram dialogue remains terminal.
4. Add/update targeted regression tests and rerun validation.
5. Rebuild the host runtime and verify fresh Telegram-scoped prompts no longer replay the old Tesla answer.
