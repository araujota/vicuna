# Implementation Plan: Frozen Past-LoRA Condensation Stack

**Branch**: `002-past-lora-condensation` | **Date**: 2026-03-10 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/002-past-lora-condensation/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/002-past-lora-condensation/spec.md`

## Summary

Extend the current Active LoRA memory path into a full temporal memory cascade by adding five frozen past-memory buckets, explicit condensation jobs between adjacent stages, deterministic time-decayed inference scaling, and DoRA-style direction-versus-gain parameterization for both live writes and condensed artifacts.

## Technical Context

**Language/Version**: C++17, C public API surface  
**Primary Dependencies**: existing `llama.cpp`/ggml runtime, current LoRA graph path, backend device memory APIs  
**Storage**: in-memory runtime state plus documentation artifacts; no new persistent database  
**Testing**: CTest-based C++ tests under `tests/`, model-backed runtime tests with explicit tick timestamps  
**Target Platform**: macOS, Linux, and Windows builds supported by the current fork  
**Project Type**: native inference/runtime library with CLI/server frontends  
**Performance Goals**: keep every bucket within configured RAM/VRAM proportions, keep condensation bounded and infrequent, and avoid base-weight mutation  
**Constraints**: preserve frozen past-bucket auditability, keep scheduling and policy in CPU-side C++, avoid hidden background mutation, and keep inference integration on the existing LoRA stack path  
**Scale/Scope**: runtime manager extensions in `src/`, public API additions in `include/`, first server tick integration, targeted tests, and architecture-doc updates

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Memory Cascade First`: Pass. The design explicitly extends the cascade beyond Active LoRA into ordered frozen past buckets with decayed influence.
- `Typed Persistent Self-State`: Pass. Bucket state, due jobs, gain limits, decay scales, and condensation history remain explicit and inspectable.
- `Pressure-Driven Dual Loops`: Pass for current scope. The implementation adds explicit due-job and rollover pressure surfaces without introducing hidden background behavior.
- `Inspectable Policy, Dense Backend Math`: Pass. Scheduling, gain control, decay, and condensation policy stay in CPU-side C++; inference continues to use ggml LoRA application.
- `Documentation, Tests, and Auditability`: Pass. The plan includes stats, audit records, targeted tests, and architecture updates for the frozen stack.

## Project Structure

### Documentation (this feature)

```text
specs/002-past-lora-condensation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── past-lora-api.md
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
└── llama.cpp

tools/server/
└── server-context.cpp

tests/
├── CMakeLists.txt
├── test-active-lora.cpp
└── test-past-lora.cpp
```

**Structure Decision**: Extend the current Active LoRA manager so one owner controls the editable Active stage, frozen past buckets, and condensation scheduler. Reuse runtime adapter allocation in `src/llama-adapter.*`, keep inference application on the existing LoRA graph path, and add the first periodic tick integration in `tools/server/server-context.cpp`.

## Phase 0: Research Outcomes

- A DoRA-style direction-versus-gain split best matches the user’s memory-update constraint and can be implemented without touching base model weights.
- Frozen-bucket condensation should use additive low-rank composition plus bounded low-rank recompression instead of raw factor averaging.
- Deterministic decay scales and explicit ticks are a better fit for Vicuña’s inspectability requirements than hidden background threads or stochastic merge defaults.
- One snapshot per bucket is sufficient for a first implementation as long as metadata and APIs leave room for richer bucket histories later.

## Phase 1: Design

### Runtime Model

- Keep Active LoRA as the only continuously editable stage.
- Add five frozen bucket states: `past_week`, `past_month`, `past_quarter`, `past_year`, `all_time`.
- Represent every memory adapter weight as normalized direction factors plus a bounded gain scalar.
- Track explicit condensation jobs for each adjacent stage transition.

### Scheduling Model

- `active -> past_week` becomes due when Active is rollover-ready.
- Older bucket jobs become due when `now_us >= due_at_us` for that stage.
- A tick recomputes effective decay scales for all populated buckets and runs due jobs youngest-to-oldest.

### Condensation Model

- Read younger and target bucket direction/gain state.
- Build a bounded merged low-rank representation.
- Recompress to the target rank.
- Normalize direction factors.
- Clip or decay gain according to policy.
- Atomically replace the target bucket snapshot and record the condensation event.

## Phase 2: Implementation Strategy

1. Extend the public API with past-stack params, stats, and tick entry points.
2. Extend adapter internals to carry explicit gain state alongside LoRA factors.
3. Refactor the Active manager to own bucket configs, bucket artifacts, and condensation jobs.
4. Apply effective scales for all populated buckets through the existing LoRA map.
5. Add server tick integration and targeted model-backed tests.

## Validation Strategy

- Unit-style runtime tests for rollover, bucket freezing, due-job execution, and immutable versions.
- Model-backed tests for decayed scale visibility and ordered condensation across multiple ticks.
- Regression coverage for direction normalization and bounded gain across both Active updates and condensed snapshots.
- Documentation review against `ARCHITECTURE.md` and `Vicuña_WP.md`.
