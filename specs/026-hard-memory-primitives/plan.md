# Implementation Plan: Hard Memory Primitives

**Branch**: `026-hard-memory-primitives` | **Date**: 2026-03-13 | **Spec**: [/Users/tyleraraujo/vicuna/specs/026-hard-memory-primitives/spec.md](/Users/tyleraraujo/vicuna/specs/026-hard-memory-primitives/spec.md)
**Input**: Feature specification from `/specs/026-hard-memory-primitives/spec.md`

## Summary

Expand Vicuña’s Supermemory integration from a single generic perturbation-event
archive path into a typed hard-memory primitive system that captures event
fragments, loop trajectories, outcomes, tool observations, user-model
fragments, and self-model fragments. Query hits will parse this metadata back
into typed surfaces, drive richer self-model promotion, and emit a fixed-width
retrieval summary that can cooperate with functional LoRA gain control without
introducing an unbounded side-channel.

## Technical Context

**Language/Version**: C++17  
**Primary Dependencies**: `cpp-httplib`, `nlohmann/json`, existing Vicuña runtime types  
**Storage**: Supermemory remote API via `/v4/memories` and `/v4/profile`  
**Testing**: CMake unit/integration-style native tests in `tests/test-self-state.cpp`, `tests/test-cognitive-loop.cpp`, `tests/test-active-lora.cpp`  
**Target Platform**: CPU-first native runtime on macOS/Linux  
**Project Type**: Native inference/runtime library with server integration  
**Performance Goals**: Maintain bounded archive batches and fixed-width retrieval summaries; no unbounded per-step network payload growth  
**Constraints**: Explicit CPU-side policy, bounded metadata sizes, no new default tools, backward-compatible self-state/gating integration  
**Scale/Scope**: Hard-memory structs, archive/query code, self-state promotion, cognitive-loop archival hooks, docs, and targeted tests

## Constitution Check

- **Runtime Policy**: Yes. All archival admission, primitive construction,
  retrieval summarization, and promotion scoring remain explicit in CPU-side
  runtime code.
- **Typed State**: `include/llama.h`, `src/llama-hard-memory.*`,
  `src/llama-self-state.*`, `src/llama-active-lora.*`, and cognitive-loop trace
  paths will gain typed primitive, batch, hit-metadata, and summary structs.
- **Bounded Memory**: The design uses explicit primitive-count caps, bounded
  strings, fixed-width summaries, and existing config thresholds. Base model
  immutability remains unchanged.
- **Validation**: Targeted native tests will cover archival batch construction,
  query metadata parsing, self-model promotion, and gating input compatibility.
- **Documentation & Scope**: `README.md`, `ARCHITECTURE.md`, `Vicuña_WP.md`, and
  `tools/server/README-dev.md` will be updated. No new third-party dependency is
  needed.

## Project Structure

### Documentation (this feature)

```text
specs/026-hard-memory-primitives/
├── plan.md
├── research.md
├── data-model.md
├── implementation-approach.md
├── quickstart.md
└── tasks.md
```

### Source Code (repository root)

```text
include/
└── llama.h

src/
├── llama-hard-memory.cpp
├── llama-hard-memory.h
├── llama-self-state.cpp
├── llama-self-state.h
├── llama-active-lora.cpp
├── llama-active-lora.h
├── llama-cognitive-loop.cpp
└── llama-context.h

tests/
├── test-self-state.cpp
├── test-cognitive-loop.cpp
└── test-active-lora.cpp
```

**Structure Decision**: Keep the implementation in the existing runtime memory,
self-state, and loop files because the feature extends explicit CPU-side runtime
policy rather than introducing a new subsystem boundary.

## Implementation Phases

### Phase 0: Research and Type Design

- Define primitive kinds, domains, tags, batch budgets, and retrieval-summary
  dimensions.
- Map existing runtime traces to specific primitive emitters.

### Phase 1: Hard-Memory Primitive Core

- Extend public structs/enums for primitive metadata, archive batches, and hit
  metadata.
- Refactor `llama_hard_memory` to archive multi-primitive batches and parse
  query metadata.

### Phase 2: Runtime Emitters And Retrieval Cooperation

- Upgrade self-state archival to emit typed primitive batches.
- Add cognitive-loop archival for trajectories, outcomes, tool observations, and
  user/self-model fragments.
- Feed retrieval summaries into self-model promotion and functional-gating
  observation.

### Phase 3: Validation And Documentation

- Add/extend targeted tests.
- Update architecture, README, working paper, and server-development docs.

## Complexity Tracking

No constitution violations are expected.
