# Implementation Plan: Strict Live Memory Runtime

**Branch**: `003-strict-memory-runtime` | **Date**: 2026-03-10 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/spec.md`

## Summary

Replace the current implicit adapter handling with an explicit ordered serving stack that preserves runtime memory layers during live decode, then enforce strict KV coherence by replaying the retained suffix whenever eviction causes an accepted Active LoRA write.

## Technical Context

**Language/Version**: C++17, C public API surface  
**Primary Dependencies**: existing `llama.cpp`/ggml runtime, current LoRA graph path, server batching/checkpoint pipeline, Active/Past LoRA manager  
**Storage**: in-memory runtime state plus documentation artifacts; no new persistent database  
**Testing**: CTest-based C++ tests under `tests/`, plus targeted server/runtime regression coverage  
**Target Platform**: macOS, Linux, and Windows builds supported by the current fork  
**Project Type**: native inference/runtime library with CLI/server frontends  
**Performance Goals**: preserve memory adapters in every live decode, bound replay work to the retained suffix only, and avoid unnecessary replay when Active weights did not change  
**Constraints**: keep composition and replay policy inspectable in CPU-side C++, preserve base-model immutability, avoid hidden backend-side policy, and respect the memory cascade ordering in `ARCHITECTURE.md`  
**Scale/Scope**: adapter composition refactor in `src/`, strict replay hook in `tools/server/`, targeted tests, and architecture/doc updates

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Memory Cascade First`: Pass. The design makes runtime memory adapters part of the actual live serving stack and keeps eviction flowing into Active LoRA before replay.
- `Typed Persistent Self-State`: Pass. Serving composition state, replay scheduling state, and replay outcomes remain explicit, typed, and inspectable.
- `Pressure-Driven Dual Loops`: Pass for current scope. The change affects the active serving loop and preserves explicit pressure-driven rollover/replay behavior without introducing hidden background actions.
- `Inspectable Policy, Dense Backend Math`: Pass. Adapter composition order, request/runtime layer separation, and strict replay scheduling stay in CPU-side C++.
- `Documentation, Tests, and Auditability`: Pass. The plan includes targeted tests, runtime logs for composition/replay decisions, and architecture updates.

## Project Structure

### Documentation (this feature)

```text
specs/003-strict-memory-runtime/
├── architecture-guideline.md
├── self-state-spec.md
├── self-state-data-model.md
├── self-state-tasks.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── memory-runtime-api.md
│   └── self-state-api.md
└── tasks.md
```

### Source Code (repository root)

```text
include/
└── llama.h

src/
├── llama-active-lora.cpp
├── llama-active-lora.h
├── llama-adapter.cpp
├── llama-adapter.h
├── llama-context.cpp
├── llama-context.h
├── llama-graph.cpp
└── llama-graph.h

tools/server/
└── server-context.cpp

tests/
├── CMakeLists.txt
├── test-active-lora.cpp
└── test-past-lora.cpp
```

**Structure Decision**: Refactor adapter composition at the `llama_context` and graph boundary so the effective serving stack is explicit and deterministic. Keep strict replay logic in `tools/server/server-context.cpp`, because generation-time context shift is currently owned by the server loop.

## Phase 0: Research Outcomes

- The most robust near-term design is to separate request adapters from runtime memory adapters and rebuild one ordered effective stack for graph construction rather than mutating one shared unordered map.
- Since LoRA application in the current graph path is additive, “active on top” should be implemented as explicit precedence and inspectable ordering, not as a serial non-commutative neural pipeline.
- Strict KV coherence should be implemented as retained-suffix replay after accepted Active LoRA writes; updating adapter weights without replay leaves the surviving KV state semantically stale.
- Replay should be skipped for redundant-span ingest or zero retained suffix and those skip cases should be explicit in runtime logs and stats.

## Phase 1: Design

### Runtime Composition Model

- Split adapter ownership into:
  - request-managed adapters set by request/server task logic
  - runtime-managed adapters owned by the memory system
- Build an explicit ordered serving stack from those sources.
- Preserve memory-layer precedence as `past_oldest -> ... -> past_newest -> active`.
- Keep request adapters separate from memory layers so task-level adapter changes cannot erase memory state.

### Graph Integration Model

- Replace unordered adapter iteration with an ordered stack type passed into `llm_graph_context`.
- Keep the math additive on top of base weights.
- Make the stack ordering explicit for inspection and future policy changes even though the current additive math is effectively commutative.

### Strict Replay Model

- During generation-time context shift:
  - identify evicted span
  - evict from KV
  - ingest into Active LoRA
  - if Active weights changed, invalidate stale retained suffix KV state
  - reset slot-local replay state
  - replay only the retained suffix under the updated serving stack
  - resume sampling only after replay completes
- Reset or invalidate stale checkpoints created under the pre-update stack.

### Replay Scheduling Policy

- Schedule replay only when:
  - Active LoRA ingestion was accepted as a real weight change
  - and the slot still has retained suffix tokens
- Skip replay when:
  - the span is redundant
  - no retained suffix exists
  - Active memory is disabled
- Log the reason in all three cases: replay run, replay skip, replay failure.

## Phase 2: Implementation Strategy

1. Refactor adapter ownership in `llama_context` so request adapters and runtime memory adapters are separate.
2. Introduce an explicit ordered adapter-stack type and use it in graph construction.
3. Update Active/Past memory attachment code to register layers into the runtime portion of the stack with explicit roles.
4. Add strict retained-suffix replay state and replay execution to the server context-shift path.
5. Add regression tests for stack preservation, replay scheduling, and replay skip behavior.

## Validation Strategy

- Runtime tests for effective stack rebuild and preservation of memory layers across request adapter changes.
- Context-shift tests for accepted-update replay, redundant-update replay skip, and zero-suffix replay skip.
- Regression checks that request adapters still work, Active/Past scales remain applied, and generation resumes after replay.
- Documentation validation against `ARCHITECTURE.md` and feature quickstart behavior.

## Follow-On Self-State Guidance

This feature directory now also carries [`architecture-guideline.md`](/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/architecture-guideline.md), which turns whitepaper Sections 5-8 into an implementation guideline for:

- typed persistent self-state
- register-bank algebra
- predict/admit/update recomputation
- feature extraction and verifier heads
- and future constrained self-modification of updater logic

The current implementation slice is specified in:

- [`self-state-spec.md`](/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/self-state-spec.md)
- [`self-state-data-model.md`](/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/self-state-data-model.md)
- [`self-state-tasks.md`](/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/self-state-tasks.md)
- [`self-state-api.md`](/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/contracts/self-state-api.md)

This slice is intentionally limited to typed state storage, datetime/event surfaces, analytic register updates, and inspection APIs. Learned heads, replay-time self-modification, and sparse reactivation maps remain follow-on work.
