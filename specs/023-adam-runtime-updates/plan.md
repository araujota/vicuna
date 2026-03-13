# Implementation Plan: Expand Adam-Based Optimization For Self-State-Driven Runtime Updates

**Branch**: `023-adam-runtime-updates` | **Date**: 2026-03-13 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/023-adam-runtime-updates/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/023-adam-runtime-updates/spec.md`

## Summary

Extend Adam from the new functional gating MLP into the two additional
self-state-driven parameter-update paths that are strong candidates in the
runtime: the shared runtime LoRA writer and the temporal write-bias controller.
Keep the counterfactual ablation ladder explicit and non-Adam because it ranks
discrete interventions rather than optimizing differentiable parameters.

## Technical Context

**Language/Version**: C11 and C++17 in the existing Vicuña runtime
**Primary Dependencies**: existing active LoRA manager, cognitive-loop runtime, self-state runtime, ggml-backed LoRA infrastructure
**Storage**: in-memory typed runtime state only
**Testing**: targeted native tests under `/Users/tyleraraujo/vicuna/tests/` plus focused `ctest` runs
**Target Platform**: CPU-first validation on the existing native runtime targets
**Project Type**: native inference/runtime library
**Performance Goals**: keep added optimizer bookkeeping bounded and negligible compared with decode and runtime write cost
**Constraints**: no new dependencies, explicit CPU-side policy, typed state, bounded optimizer memory, preserve serving-stack semantics
**Scale/Scope**: runtime-core change across Active LoRA writes, functional family writes, temporal bias updates, tests, and docs

## Constitution Check

### Runtime Policy

Pass target:

- keep counterfactual candidate ranking explicit in CPU policy
- keep optimizer use explicit and limited to real parameter-update paths
- expose new optimizer state through typed stats or traces

### Typed State

New or extended typed surfaces required:

- runtime LoRA optimizer state owned by the active LoRA manager
- temporal bias optimizer state owned by the active LoRA manager
- public stats or bias structs exposing optimizer advancement

### Bounded Memory

Pass target:

- one bounded moment state per runtime-mutable LoRA tensor pair
- fixed-size scalar Adam state for temporal reward and dampening biases
- no replay buffers or unbounded optimizer history

### Validation

Required:

- Active LoRA tests proving optimizer-backed runtime writes advance explicit
  step state
- cognitive-loop tests proving temporal bias optimizer state advances on DMN
  temporal self-improvement
- regression tests for existing Active LoRA and cognitive-loop flows

### Documentation & Scope

Files expected to change:

- `/Users/tyleraraujo/vicuna/include/llama.h`
- `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`
- `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`
- `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`
- `/Users/tyleraraujo/vicuna/Vicuña_WP.md`

## Project Structure

### Documentation (this feature)

```text
/Users/tyleraraujo/vicuna/specs/023-adam-runtime-updates/
├── spec.md
├── research.md
├── plan.md
├── data-model.md
├── implementation-approach.md
├── quickstart.md
└── tasks.md
```

### Source Code (repository root)

```text
/Users/tyleraraujo/vicuna/include/
└── llama.h

/Users/tyleraraujo/vicuna/src/
└── llama-active-lora.cpp

/Users/tyleraraujo/vicuna/tests/
├── test-active-lora.cpp
└── test-cognitive-loop.cpp
```

**Structure Decision**: Concentrate the implementation in the active LoRA
manager because it already owns the runtime LoRA writer, functional family
writer, and temporal encoding bias controller.

## Workstreams

### 1. Runtime LoRA Adamization

- introduce typed per-weight optimizer state for runtime-mutable LoRA tensors
- reinterpret existing self-state deltas as pseudo-gradients
- preserve weight decay and gain normalization after optimizer-backed updates

### 2. Temporal Bias Adamization

- introduce typed scalar Adam state for reward and dampening biases
- replace threshold-jump updates with Adam-conditioned bounded updates
- preserve existing effective write-scale clamp semantics

### 3. Observability And Validation

- expose optimizer step counts and update norms in public stats/bias surfaces
- extend tests to prove optimizer advancement on both paths
- document the explicit non-adoption decision for counterfactual ranking

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| Extra optimizer state for runtime-mutable LoRA tensors | Adam requires persistent moments per parameter to condition noisy online updates | Reusing raw additive writes would not answer the research question or improve conditioning |
| Public optimizer observability fields | The constitution requires inspectable runtime policy and tests need to prove the new optimizer paths are real | Keeping the optimizer entirely internal would make the change opaque and hard to validate |
