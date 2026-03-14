# Implementation Plan: User Model Expansion and Counterfactual User Simulation

**Branch**: `023-adam-runtime-updates` | **Date**: 2026-03-13 | **Spec**: `/Users/tyleraraujo/vicuna/specs/028-user-model-simulation-lora/spec.md`
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/028-user-model-simulation-lora/spec.md`

## Summary

Expand the current user model into a typed, bounded profile surface across hard
memory and self-state, then add a dedicated user-personality runtime LoRA plus
DMN-only counterfactual reply simulation. The implementation reuses the current
Adam-backed runtime LoRA writer and inspectable serving-stack plumbing, adds a
scoped user-simulation serving override that temporarily ablates temporal
memory layers, and records typed traces for simulated message, reply, and
self-state effect.

## Technical Context

**Language/Version**: C++17 and C public API in `include/llama.h`  
**Primary Dependencies**: ggml, existing runtime LoRA and cognitive-loop code,
Supermemory HTTP integration  
**Storage**: In-memory runtime state plus Supermemory archival retrieval  
**Testing**: CMake targets with focused native tests in `tests/`  
**Target Platform**: CPU-first native inference runtime  
**Project Type**: Native inference library and server runtime  
**Performance Goals**: Keep user simulation bounded to small decode budgets and
explicitly skip on low-readiness paths  
**Constraints**: Immutable base model, bounded runtime adapters, explicit
CPU-side serving policy, no unbounded profile growth  
**Scale/Scope**: Changes span public API, active-lora manager, self-state,
hard-memory, cognitive loop, tests, and docs

## Constitution Check

- **Runtime Policy**: Pass. Serving override, user-personality adapter
  activation, temporal ablation, and simulation gating remain explicit in CPU
  control code.
- **Typed State**: Pass. The design adds typed user preference summaries,
  user-simulation traces, adapter roles, and fixed-size profile fields instead
  of free-form hidden state.
- **Bounded Memory**: Pass. The user-personality LoRA has explicit rank and
  update policy, does not roll into temporal buckets, and the simulation path
  uses bounded token budgets and traces.
- **Validation**: Pass. Focused tests will cover hard-memory user-model
  expansion, user-personality training policy, serving-stack override behavior,
  and counterfactual simulated reply traces.
- **Documentation & Scope**: Pass. This change updates headers, architecture
  docs, operator docs, and Spec Kit artifacts without adding new dependencies.

## Project Structure

### Documentation

```text
specs/028-user-model-simulation-lora/
├── spec.md
├── research.md
├── plan.md
├── tasks.md
├── data-model.md
├── implementation-approach.md
└── quickstart.md
```

### Source Code

```text
include/llama.h
src/llama-active-lora.h
src/llama-active-lora.cpp
src/llama-adapter.cpp
src/llama-context.h
src/llama-context.cpp
src/llama-self-state.h
src/llama-self-state.cpp
src/llama-hard-memory.h
src/llama-hard-memory.cpp
src/llama-cognitive-loop.h
src/llama-cognitive-loop.cpp
tests/test-active-lora.cpp
tests/test-self-state.cpp
tests/test-cognitive-loop.cpp
README.md
ARCHITECTURE.md
Vicuña_WP.md
tools/server/README-dev.md
```

**Structure Decision**: Implement directly in the existing runtime surfaces.
The feature is a cross-cutting inference and memory change, so it belongs in
the native runtime control path rather than in a separate service layer.

## Architecture

### 1. Typed User Model Expansion

Add typed user-preference and rhetorical summary surfaces to the public API and
self-state model info. Keep them distinct from existing social and outcome
profiles so the system can reason about both:

- relationship state
- task-outcome state
- stable user preference and rhetorical style state

Also enrich `USER_MODEL` hard-memory primitive content and retrieval summaries
so hard memory can preserve richer durable user-profile fragments.

### 2. Dedicated User Personality Runtime Adapter

Extend the runtime adapter manager with a dedicated user-personality adapter:

- created at initialization like other runtime adapters
- fixed rank and fixed scale policy
- continuously updated in place
- trained only from evicted user-authored spans
- separate stats and optimizer state
- never part of the temporal bucket condensation pipeline

### 3. Serving Stack Override for User Simulation

Add an explicit scoped override in `llama_context` that can:

- snapshot currently attached runtime layers
- temporarily detach or ablate selected classes of layers
- attach only the user-personality adapter
- restore the original stack even on failure

This is needed for DMN simulated-user decode passes and keeps policy
inspectable.

### 4. DMN Counterfactual User Simulation

Extend the counterfactual ladder and DMN compare path with a bounded
message-variant simulation mode:

1. generate or select a candidate outbound message under counterfactual
   conditions
2. swap to the user-simulation serving stack
3. synthesize a bounded simulated user reply
4. restore the normal serving stack
5. apply the simulated reply to a shadow or counterfactual self-state channel
6. score the resulting user-response-induced self-state delta

This simulation becomes additional evidence inside DMN counterfactual
comparison rather than a third loop.

## Validation

- `tests/test-self-state.cpp`
  - user-model summary expansion
  - user-model hard-memory archival enrichment
  - counterfactual simulated user reply remains on counterfactual channel
- `tests/test-active-lora.cpp`
  - user-personality adapter initialization and update policy
  - non-user spans do not update the user-personality substrate
- `tests/test-cognitive-loop.cpp`
  - DMN counterfactual simulation trace
  - temporal-LoRA ablation during user simulation
  - normal stack restoration after simulation

Validation commands:

```sh
cmake --build /Users/tyleraraujo/vicuna/build-codex --target test-active-lora test-self-state test-cognitive-loop -j4
ctest --test-dir /Users/tyleraraujo/vicuna/build-codex --output-on-failure -R 'test-active-lora|test-self-state|test-cognitive-loop'
```
