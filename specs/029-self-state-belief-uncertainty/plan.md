# Implementation Plan: Partial-Observation Self-State Belief Gradient

**Branch**: `[029-self-state-belief-uncertainty]` | **Date**: 2026-03-13 | **Spec**: [/Users/tyleraraujo/vicuna/specs/029-self-state-belief-uncertainty/spec.md](/Users/tyleraraujo/vicuna/specs/029-self-state-belief-uncertainty/spec.md)
**Input**: Feature specification from `/specs/029-self-state-belief-uncertainty/spec.md`

## Summary

Augment Vicuña’s explicit self-state gradient with a bounded belief layer that models uncertainty over incompletely rendered cares. The new layer will treat the current self-state as observation, maintain a small typed posterior-like summary over residual hidden pressure, and expose only fixed-width belief features to the existing gating MLP. This preserves explicit allostasis and typed inspectability while enabling more cautious, exploratory, or information-seeking gain control under self-model uncertainty.

## Technical Context

**Language/Version**: C++17  
**Primary Dependencies**: existing `llama.cpp`/Vicuña runtime, C++ standard library  
**Storage**: in-memory runtime state inside `llama_context`; optional trace exposure through existing public structs  
**Testing**: CMake + existing C++ unit/integration tests (`test-self-state`, `test-active-lora`, `test-cognitive-loop`)  
**Target Platform**: CPU-first native runtime across existing supported build targets  
**Project Type**: native inference runtime / library  
**Performance Goals**: preserve fixed-width gating features and O(1) belief updates per settled transaction  
**Constraints**: explicit CPU-side control policy, bounded memory, no raw latent text into gate, no base-model mutation  
**Scale/Scope**: one new belief-state family per `llama_context`, shared by active and DMN flows

## Constitution Check

- **Runtime Policy**: Pass. Belief update rules, clipping, decay, promotion thresholds, and gain influence stay explicit in CPU-side control code.
- **Typed State**: Pass. New typed structs belong in `include/llama.h`, `src/llama-self-state.h`, and `src/llama-active-lora.h` with bounded numeric invariants.
- **Bounded Memory**: Pass. Fixed-size belief slots and summaries preserve base-model immutability and bounded runtime memory.
- **Validation**: Requires targeted tests for belief update math, gating input compatibility, clipping, and promotion-candidate surfacing.
- **Documentation & Scope**: Requires updates to `ARCHITECTURE.md`, `README.md`, and likely `Vicuña_WP.md` if implemented. No new dependency is required.

## Project Structure

### Documentation (this feature)

```text
specs/029-self-state-belief-uncertainty/
├── plan.md
├── research.md
├── data-model.md
└── tasks.md
```

### Source Code (repository root)

```text
include/
└── llama.h

src/
├── llama-self-state.h
├── llama-self-state.cpp
├── llama-active-lora.h
├── llama-active-lora.cpp
└── llama-cognitive-loop.cpp

tests/
├── test-self-state.cpp
├── test-active-lora.cpp
└── test-cognitive-loop.cpp
```

**Structure Decision**: Extend the existing self-state and functional-gating runtime surfaces rather than adding a new subsystem. The belief layer lives next to self-state because it models partial observability over self-state, and only its fixed summary is exported to active LoRA gain control.

## Architecture Decision

### Decision

Implement a bounded belief filter over incompletely modeled cares, not a full latent self-model.

### Why

- It matches POMDP and active-inference style reasoning under partial observability.
- It preserves Vicuña’s typed explicit control ontology.
- It avoids turning the self-state into opaque hidden memory.

### Consequences

- The controller can act differently when it suspects missing cares.
- The runtime becomes more robust to incompleteness, but slightly more complex.
- Promotion from latent residue to explicit self-model state becomes an explicit future pathway.

## Proposed Runtime Flow

1. Build explicit observed self-state gradient as today.
2. Build belief evidence from:
   - predicted versus realized self-state change,
   - tool/user outcome mismatch,
   - counterfactual miss,
   - hard-memory retrieval residue.
3. Update bounded belief slots and belief summary with clip + decay.
4. Concatenate belief summary onto the gating feature vector.
5. Predict gains as today with Adam-updated gate.
6. Optionally surface promotion candidates when repeated residue stabilizes.

## Validation Strategy

- Unit tests for belief update clipping, decay, and confidence transitions.
- Unit tests for belief-summary serialization and slot budgeting.
- Functional tests proving gain predictions change through new fixed features only.
- Replay-style tests where explicit state is constant but residual evidence differs.
- Documentation validation by aligning architecture docs and public stats with the new ontology.

## Recommended Implementation Order

1. Add typed belief structs and config to public and internal headers.
2. Implement belief evidence extraction and bounded filtering in self-state.
3. Append belief summary fields to the gating MLP input builder.
4. Expose traces and stats.
5. Add tests.
6. Update documentation.

## Complexity Tracking

No constitution violations are currently expected. The main risk is conceptual scope creep if the belief layer grows from a residual controller into a second hidden self-model. Keep it narrow.
