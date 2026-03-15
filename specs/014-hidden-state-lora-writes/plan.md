# Implementation Plan: Hidden-State-Derived Active LoRA Embeddings And Feature-Derived Write Rules

**Branch**: `014-hidden-state-lora-writes` | **Date**: 2026-03-11 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/spec.md`

## Summary

Produce an implementation architecture for replacing the current Active LoRA
hash/token-pool embedder and token-modulo write rule with:

- hidden-state-derived span embeddings from the serving model family
- a feature-derived write-direction path tied to typed self-state
- explicit rank and magnitude policy informed by AdaLoRA and DoRA
- explicit quantization boundaries informed by QLoRA

while preserving current serving stack ordering, temporal bucket behavior,
remediation semantics, and counterfactual LoRA-ablation parity.

## Technical Context

**Language/Version**: Markdown design artifacts for a C++17 / C runtime  
**Primary Dependencies**: current `llama.cpp`/ggml runtime, `src/llama-active-lora.cpp`, `src/llama-context.cpp`, existing self-state and cognitive-loop surfaces  
**Storage**: design artifacts under `specs/014-hidden-state-lora-writes/`  
**Testing**: evidence-backed architecture review plus future task breakdown for targeted C++ tests  
**Target Platform**: native inference/runtime library and `llama-server`  
**Project Type**: runtime-memory architecture and implementation planning  
**Performance Goals**: improve semantic quality of Active LoRA writes without regressing serving-order invariants or exploding write-time/runtime overhead  
**Constraints**: no new git branch for the Spec Kit workflow; preserve serving-layer order and temporal bucket semantics; preserve LoRA-ablation parity; keep runtime policy explicit in CPU-side control code  
**Scale/Scope**: design only for now, but detailed enough to guide follow-on implementation in `include/`, `src/`, `tests/`, `docs/`, and `tools/server/`

## Constitution Check

- **Runtime Policy**: Pass. The design keeps embedder selection, hidden-state
  extraction mode, layer allocation, gain control, and fallback behavior
  explicit in CPU-side control code.
- **Typed State**: Pass. The design will introduce explicit params, feature
  structs, traces, and rank-allocation state instead of hidden heuristics.
- **Bounded Memory**: Pass. Base weights remain immutable; Active/Past LoRAs
  stay bounded; any shadow or auxiliary context must be budgeted explicitly.
- **Validation**: Pass. The design includes future targeted tests for embedder
  parity, writer parity, serving-order parity, and ablation parity.
- **Documentation & Scope**: Pass. The output is a narrowly scoped architecture
  package plus future implementation tasks.

## Project Structure

### Documentation (this feature)

```text
specs/014-hidden-state-lora-writes/
├── spec.md
├── plan.md
├── tasks.md
├── research.md
├── data-model.md
└── implementation-approach.md
```

### Source Code (implementation targets)

```text
include/
└── llama.h

src/
├── llama-active-lora.cpp
├── llama-active-lora.h
├── llama-context.cpp
├── llama-context.h
├── llama-cognitive-loop.cpp
└── llama-adapter.cpp

tests/
├── test-active-lora.cpp
├── test-past-lora.cpp
├── test-serving-lora-stack.cpp
└── test-cognitive-loop.cpp

tools/server/
└── server-context.cpp
```

**Structure Decision**: Keep the design deliverable fully under
`specs/014-hidden-state-lora-writes/` while scoping future implementation to
the existing Active LoRA manager, context embedding surfaces, serving stack
tests, and cognitive-loop parity tests.

## Design Approach

### Phase 0: Research

1. Re-read current Active LoRA, past-LoRA, remediation, and serving-stack code.
2. Identify current embedder/write bottlenecks and current parity invariants.
3. Research public PEFT methods and memory systems:
   - LoRA / AdaLoRA / DoRA / QLoRA
   - hidden-state or latent-memory systems such as LongMem and MemoryLLM
4. Translate those ideas into runtime-memory design choices rather than
   offline-finetuning defaults.

### Phase 1: Architecture

1. Define a default hidden-state embedder path:
   - minimum-invasive path using current embedding outputs
   - optional later path using selected late-layer activation taps
2. Define a feature-derived write rule:
   - content embedding
   - typed self-state features
   - per-layer sensitivity/allocation
   - direction and gain update rules
3. Define adoption boundaries for:
   - AdaLoRA-like adaptive rank budgeting
   - DoRA-like magnitude/direction handling
   - QLoRA-like quantization support
4. Define parity constraints:
   - serving-layer order
   - temporal bucket ordering
   - LoRA ablation family
   - remediation/update path parity

### Phase 2: Future Implementation Planning

1. Add explicit data-model and API proposals.
2. Produce a future implementation task list with targeted tests and docs.

## Validation Strategy

- Verify the architecture cites concrete local file references for every parity
  or implementation constraint.
- Verify the design includes direct references to external primary papers or
  official repositories for every AdaLoRA/DoRA/QLoRA decision.
- Verify the design distinguishes:
  - what is implementable immediately with current Vicuña surfaces
  - what requires deeper internal model taps
  - what should remain out of scope for runtime memory

## Research Summary

- The current bottleneck is semantic quality of the writer, not rank alone.
- The current code already has a partial DoRA-like direction/gain split.
- Existing embedding APIs in the local runtime can support a minimally invasive
  hidden-state embedder path.
- AdaLoRA ideas are relevant for periodic layer-budget reallocation, but its
  training-step schedule should not be copied directly into the runtime.
- QLoRA is relevant mainly for optional auxiliary or shadow contexts and
  calibration workflows, not as the primary live serving-memory path.
