# Implementation Plan: Self-Model Expansion For Efficient Goal Pursuit, User Satisfaction, And Self-Repair

**Branch**: `015-self-model-expansion` | **Date**: 2026-03-11 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/spec.md)  
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/spec.md`

## Summary

Produce an implementation architecture for expanding Vicuña's explicit
self-model so it can represent:

- richer goal-progress and blocker state
- user satisfaction and interaction-contract risk
- epistemic condition and self-estimate confidence
- efficiency, loop-cost, and repetition risk
- favorable-state divergence and self-repair trajectory
- explicit motivation for safe self-improvement

while preserving the current prewrite/postwrite pipeline, favorable-state
machinery, counterfactual traces, remediation flow, and foreground/background
loop behavior.

## Technical Context

**Language/Version**: Markdown design artifacts for a C++17 / C runtime  
**Primary Dependencies**: `include/llama.h`, `src/llama-self-state.cpp`, `src/llama-cognitive-loop.cpp`, existing favorable/counterfactual/remediation traces  
**Storage**: design artifacts under `specs/015-self-model-expansion/`  
**Testing**: evidence-backed architecture review plus future targeted C++ tests for parity and new profile behavior  
**Target Platform**: native runtime library and `llama-server`  
**Project Type**: runtime architecture and typed self-state expansion  
**Performance Goals**: improve user-aligned and self-repair-aligned behavior with fewer loop steps and less wasted inference by giving the runtime better internal state estimates  
**Constraints**: no new git branch for Spec Kit workflow; preserve inspectable CPU-side policy; preserve typed bounded state; preserve prewrite/postwrite and cognitive-loop parity  
**Scale/Scope**: design only for this request, but detailed enough to guide follow-on changes in `include/`, `src/`, `tests/`, and documentation

## Constitution Check

- **Runtime Policy**: Pass. The design keeps motivation, forecast, and recovery
  policy explicit in CPU-side control code.
- **Typed State**: Pass. The expansion is organized into typed profiles and
  horizon slices instead of opaque latent memory.
- **Boundedness**: Pass. Every proposed profile and trace remains bounded and
  exportable.
- **Parity**: Pass. The plan explicitly preserves prewrite/postwrite ordering,
  favorable-state logic, counterfactual traces, remediation semantics, and loop
  behavior.
- **Documentation**: Pass. The output is a durable Spec Kit package with
  research, data model, and implementation architecture.

## Project Structure

### Documentation (this feature)

```text
specs/015-self-model-expansion/
├── spec.md
├── plan.md
├── tasks.md
├── research.md
├── data-model.md
└── implementation-approach.md
```

### Source Code (future implementation targets)

```text
include/
└── llama.h

src/
├── llama-self-state.cpp
├── llama-self-state.h
├── llama-cognitive-loop.cpp
├── llama-context.cpp
└── llama-context.h

tests/
├── test-self-state.cpp
├── test-cognitive-loop.cpp
├── test-active-lora.cpp
└── test-serving-lora-stack.cpp
```

**Structure Decision**: Keep the research and architecture package fully under
`specs/015-self-model-expansion/`, then scope future implementation to the
existing self-state, cognitive loop, and trace export surfaces.

## Design Approach

### Phase 0: Evidence Gathering

1. Re-read the audit findings for persistent self-representation and register
   bank limitations.
2. Inspect current self-state structs, updater features, favorable-state
   dimensions, and loop traces in `include/llama.h`,
   `src/llama-self-state.cpp`, and `src/llama-cognitive-loop.cpp`.
3. Research public agent-state implementations and primary literature relevant
   to explicit self-models, user satisfaction estimation, uncertainty
   calibration, self-correction, and homeostatic control.

### Phase 1: Self-Model Architecture

1. Separate the self-model into layers:
   - fast control registers
   - typed profiles
   - multi-timescale horizon slices
   - forecast and prediction-error traces
2. Define the new profile families:
   - goal progress
   - user outcome
   - epistemic control
   - efficiency
   - recovery and homeostasis
   - strategic mode
   - self-improvement governance
3. Define how those profiles connect back into current favorable-state,
   counterfactual, remediation, Active Loop, and DMN logic.

### Phase 2: Incremental Adoption Plan

1. Define the public API changes and bounded structs needed in `include/llama.h`.
2. Define the profile update order and how it composes with current prewrite and
   postwrite feature extraction.
3. Define rollout stages so implementation can preserve parity and remain
   debuggable.

## Validation Strategy

- Verify that each proposed self-model family has:
  - a clear semantic meaning
  - an explicit bounded representation
  - an inspectable update path
  - a reason it improves user satisfaction, goal progress, or recovery
- Verify that the design preserves:
  - current register bank usage for fast loop control
  - favorable-state and governance trace compatibility
  - import/export and replay compatibility expectations
- Verify that the design is supported by both local repository evidence and
  primary external sources or mature public implementations.

## Research Summary

- The current self-model is operational, but mainly tracks local pressure rather
  than structured progress, forecasts, or recovery trajectories.
- Mature public agent systems represent persistent execution metadata, memory
  composition, budget state, and operating context explicitly rather than as one
  flat register bank.
- Current research supports explicit user-satisfaction estimation, interpretable
  uncertainty signals, self-correction traces, and homeostatic or
  active-inference-inspired control variables.
- The strongest next move is not "more arbitrary scalars", but a layered
  explicit self-model with grouped profiles, horizons, forecasts, and
  prediction-error traces.
