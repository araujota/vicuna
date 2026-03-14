# Implementation Plan: Self-Model Extension Registry

**Branch**: current branch | **Date**: 2026-03-13 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/025-self-model-extension-registry/spec.md)  
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/025-self-model-extension-registry/spec.md`

## Summary

Implement a bounded self-model extension registry that preserves the authored
core, supports counterfactual promotion of hard-memory representations, supports
tool-authored scalar parameters with optional desirable-state metadata, and
feeds the functional gating path through a fixed extension summary rather than
an unbounded raw list.

## Technical Context

**Language/Version**: C++17 and C public API  
**Primary Dependencies**: `include/llama.h`, `src/llama-self-state.cpp`,
`src/llama-self-state.h`, `src/llama-active-lora.cpp`,
`src/llama-cognitive-loop.cpp`, `src/llama-context.*`,
`src/llama-hard-memory.cpp`  
**Storage**: bounded runtime state inside `llama_self_state`  
**Testing**: `tests/test-self-state.cpp`, `tests/test-active-lora.cpp`,
`tests/test-cognitive-loop.cpp`  
**Target Platform**: CPU-first `libllama` runtime and `llama-server`
integrators  
**Constraints**: explicit CPU-side policy, typed bounded state, no destructive
branch/worktree cleanup, preserve existing self-model core and functional-gating
semantics  
**Scope**: runtime code, public API, tests, architecture docs, README guidance,
server developer guidance

## Constitution Check

- **Explicit Runtime Policy**: Pass. Extension effects, promotion logic, and
  gain/allostasis flags stay in inspectable CPU-side control code.
- **Typed Mathematical Surfaces**: Pass. The design adds typed extension
  structs, bounded slots, and fixed summaries rather than free-form blobs.
- **Immutable Base Model / Bounded Runtime Memory**: Pass. The change only adds
  bounded runtime-managed self-model state.
- **Validation Before Merge**: Pass. Targeted tests will cover extension CRUD,
  hard-memory promotion, and extension-aware summaries.
- **Minimal Documented Change Sets**: Pass. The implementation stays inside
  existing self-state/tool/gating surfaces and updates docs in the same change.

## Project Structure

### Documentation

```text
specs/025-self-model-extension-registry/
├── spec.md
├── plan.md
├── tasks.md
├── research.md
├── data-model.md
├── implementation-approach.md
└── quickstart.md
```

### Source Code

```text
include/
└── llama.h

src/
├── llama-self-state.cpp
├── llama-self-state.h
├── llama-active-lora.cpp
├── llama-context.cpp
├── llama-context.h
└── llama-hard-memory.cpp

tests/
├── test-self-state.cpp
├── test-active-lora.cpp
└── test-cognitive-loop.cpp
```

### Docs

```text
README.md
ARCHITECTURE.md
Vicuña_WP.md
tools/server/README-dev.md
```

## Design Approach

### Phase 1: Research And Contracts

1. Reconcile this feature with the existing expanded self-model work in
   `specs/015-self-model-expansion/`.
2. Research comparable state-extension patterns from Letta, LangGraph, and
   Voyager, plus active-inference/homeostatic-control literature.
3. Define the extension entities, flags, bounded capacities, and summary
   surfaces.

### Phase 2: Runtime Self-State Extension Layer

1. Add public API enums, structs, and methods for extension CRUD and inspection.
2. Add bounded extension storage and validation to `llama_self_state`.
3. Add extension summary computation and extension-aware self-model updates.
4. Add hard-memory counterfactual promotion into the registry.

### Phase 3: Control-Path Integration

1. Feed a fixed extension summary into the functional gating MLP input.
2. Keep allostatic distance backward-compatible by separating gain influence
   from allostatic participation.
3. Preserve existing favorable-state and loop logic while allowing tool-authored
   desirable-state parameters to surface as bounded extension pressure.

### Phase 4: Validation And Documentation

1. Add self-state and hard-memory tests for extension lifecycle and promotion.
2. Add targeted functional-gating regression coverage for extension summaries.
3. Document how tools should add self-model state accurately and safely.

## Validation Strategy

- Unit/integration coverage for:
  - extension upsert/get/remove behavior
  - validation and clamping
  - hard-memory promotion traces
  - extension summary values
  - allostatic participation rules
  - extension-aware gating stability
- Manual validation by reading public APIs and docs to confirm tool extension
  guidance is accurate and concrete.

## Research Summary

- The repo already has a typed authored self-model core and a fixed gain-control
  path, making a bounded extension layer the right insertion point.
- Letta’s memory-block model, LangGraph’s injected state/store tools, and
  Voyager’s persistent skill library all favor explicit inspectable state
  contracts over ad hoc hidden memory.
- Active-inference and homeostatic-control work support a hybrid of fixed prior
  structure plus learned/adaptive state rather than replacing the prior with a
  fully unconstrained latent surface.
