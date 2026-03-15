# Implementation Plan: Functional LoRA Bootstrap Perturbation

**Branch**: `[030-functional-lora-bootstrap-perturbation]` | **Date**: 2026-03-13 | **Spec**: [/Users/tyleraraujo/vicuna/specs/030-functional-lora-bootstrap-perturbation/spec.md](/Users/tyleraraujo/vicuna/specs/030-functional-lora-bootstrap-perturbation/spec.md)
**Input**: Feature specification from `/specs/030-functional-lora-bootstrap-perturbation/spec.md`

## Summary

Keep learned functional LoRA adapters as zero-initialized no-ops, but add a second per-family bootstrap adapter initialized with tiny random weights and driven by a sampled signed perturbation scale. The bootstrap perturbation decays from an initial magnitude toward a nonzero floor as each family accumulates activation count, and it is exposed separately from gating exploration in typed config, state, and traces.

## Technical Context

**Language/Version**: C++17  
**Primary Dependencies**: existing Vicuña runtime, C++ standard library  
**Storage**: in-memory runtime state in `llama_active_lora_manager` and public typed API structs  
**Testing**: CMake C++ tests (`test-active-lora`, `test-cognitive-loop`)  
**Target Platform**: CPU-first native runtime across existing supported platforms  
**Project Type**: native inference runtime / library  
**Performance Goals**: O(1) extra work per family activation and no change to base-model immutability  
**Constraints**: explicit CPU-side policy, bounded noise, preserve current gain clip semantics, keep learned/bootstrapped state separate  
**Scale/Scope**: one extra runtime adapter and a few typed fields per functional family

## Constitution Check

- **Runtime Policy**: Pass. Bootstrap decay, sampling, clipping, and attachment remain explicit in CPU-side control code.
- **Typed State**: Pass. Public structs in `include/llama.h` will gain explicit bootstrap config and trace fields.
- **Bounded Memory**: Pass. Each family gets one additional fixed-size runtime adapter; no base-model weights are mutated.
- **Validation**: Requires targeted tests for initialization semantics, perturbation sampling, decay-to-floor behavior, and trace exposure.
- **Documentation & Scope**: Requires updates to headers plus operator and architecture docs. No new dependency is needed.

## Project Structure

### Documentation (this feature)

```text
specs/030-functional-lora-bootstrap-perturbation/
├── spec.md
├── research.md
├── plan.md
└── tasks.md
```

### Source Code (repository root)

```text
include/
└── llama.h

src/
├── llama-active-lora.cpp
└── llama-adapter.h

tests/
├── test-active-lora.cpp
└── test-cognitive-loop.cpp
```

**Structure Decision**: Implement the feature entirely in the existing functional-LoRA runtime control path. No new subsystem is needed because activation, ablation, and tracing already centralize in `llama-active-lora.cpp`.

## Architecture Decision

### Decision

Represent early accidental discovery with a separate bootstrap adapter per functional family, not by changing learned-adapter initialization.

### Why

- It preserves the current no-op learned-adapter contract.
- It makes the stochastic effect real at the adapter level, not just at the gain level.
- It keeps observability and future optimizer work coherent.

### Consequences

- Each family now has two LoRA artifacts: learned and bootstrap.
- Activation logic must manage two scales instead of one.
- Public trace/state surfaces gain a few new fields.

## Proposed Runtime Flow

1. Initialize learned functional adapters as zeroed no-ops.
2. Initialize bootstrap adapters with tiny random weights.
3. On gain prediction, compute per-family bootstrap std from activation count.
4. On activation, if the family is active, sample a bounded signed bootstrap perturbation.
5. Attach learned adapter with routed gain and bootstrap adapter with perturbation scale.
6. Increment family activation count and expose resulting bootstrap fields in trace/state.
7. Leave Adam updates untouched for the learned adapter only.

## Validation Strategy

- Unit-style initialization checks in `test-active-lora.cpp`.
- Activation-trace checks for nonzero bootstrap perturbation in `test-cognitive-loop.cpp`.
- Repeated-activation checks for monotonic decay to a positive floor.
- Manual focused validation with `ctest` on the affected test binaries.

## Recommended Implementation Order

1. Add typed bootstrap config and state fields to `include/llama.h`.
2. Add bootstrap adapters, decay helpers, and sampling logic to `src/llama-active-lora.cpp`.
3. Thread sampled bootstrap perturbation through functional activation and family state.
4. Add targeted tests.
5. Update docs.

## Complexity Tracking

No constitution violations are expected. The main risk is letting bootstrap perturbation become hidden or conflated with learned state; the implementation should keep those surfaces separate in both code and traces.
