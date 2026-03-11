# Implementation Plan: Sliding-Window Active LoRA Memory

**Branch**: `001-active-lora-memory` | **Date**: 2026-03-10 | **Spec**: [spec.md](/Users/tyleraraujo/vicuna/specs/001-active-lora-memory/spec.md)
**Input**: Feature specification from `/specs/001-active-lora-memory/spec.md`

## Summary

Implement a bounded Active LoRA memory stage that receives context evicted from the live window, sizes itself from live host/device memory ratios, and updates a runtime-generated mutable LoRA directly through CPU-side writer policy. Keep the writer policy in CPU-side C++ with a pluggable span-embedding interface, expose inspectable stats and rollover readiness, and hook the first frontend ingestion path into server context-shift eviction.

## Technical Context

**Language/Version**: C++17, C public API surface  
**Primary Dependencies**: existing `llama.cpp`/ggml runtime, backend device memory APIs  
**Storage**: in-memory runtime state plus documentation artifacts; no new persistent database  
**Testing**: CTest-based C++ tests under `tests/`, plus existing model-backed test patterns when needed  
**Target Platform**: macOS, Linux, and Windows builds supported by the current `llama.cpp` fork  
**Project Type**: native inference/runtime library with CLI/server frontends  
**Performance Goals**: keep Active LoRA allocation within configured RAM/VRAM ratios, avoid mutating base weights, and keep write overhead bounded without allocating extra unbudgeted runtime state  
**Constraints**: preserve memory-cascade ordering, keep policy in CPU-side C++, avoid unbounded adapter growth, keep embedding strategy swappable, and avoid destroying the serving KV state during writes  
**Scale/Scope**: core runtime changes in `src/` and `include/`, first integration in `tools/server/`, targeted tests and architecture docs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Memory Cascade First`: Pass. The design writes evicted spans into a mutable Active LoRA and exposes rollover readiness for a future frozen past-LoRA stage.
- `Typed Persistent Self-State`: Pass. Budget state, selected embedder, update counters, and rollover readiness remain explicit and inspectable in CPU-side state.
- `Pressure-Driven Dual Loops`: Pass for current scope. This feature does not add DMN behavior and does not move policy into time-based triggers.
- `Inspectable Policy, Dense Backend Math`: Pass. Budgeting, admission, embedder selection, and write orchestration stay in CPU-side C++; inference still uses ggml math paths.
- `Documentation, Tests, and Auditability`: Pass. The plan includes architecture-doc updates, targeted tests, and update records for admitted/skipped spans.

## Project Structure

### Documentation (this feature)

```text
specs/001-active-lora-memory/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── active-lora-api.md
└── tasks.md
```

### Source Code (repository root)

```text
include/
└── llama.h

src/
├── CMakeLists.txt
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
└── test-active-lora.cpp
```

**Structure Decision**: Keep the core feature in `src/` next to adapter/context internals, expose a minimal C API in `include/llama.h`, add the first automatic ingestion hook in `tools/server/server-context.cpp`, and keep targeted regression tests in `tests/`.

## Phase 0: Research Outcomes

- Runtime Active LoRA should be an in-memory mutable adapter rather than a file-loaded GGUF-only artifact.
- Rank planning should derive from live host/device free memory and per-rank target costs, and must reject plans that exceed a zero-budget host or device tier.
- Embedder choice belongs behind an interface boundary and should influence admission/tracing rather than be hard-wired to one model family.
- The first implementation should avoid allocating an extra shadow context because that memory would sit outside the adapter budget invariant; a direct writer is acceptable while a fuller training backend matures.

## Phase 1: Design

### Data Model

- `ActiveLoRAConfig`
- `ActiveLoRABudget`
- `ActiveLoRATarget`
- `EvictedSpan`
- `SpanEmbedding`
- `ActiveLoRAUpdateRecord`
- `ActiveLoRAState`

### Public Interface

- Add `llama_active_lora_params`, `llama_active_lora_stats`, default-params helper, init/ingest/stats APIs.
- Keep the API focused on configuration, ingestion, and inspection; hide shadow-context and optimizer internals.

### Internal Components

- `llama_active_lora_manager`: owns config, budget plan, runtime adapter, embedder, shadow context, optimizer state, and update records.
- Runtime adapter allocation helpers in `llama-adapter.*`: allocate mutable zero-initialized LoRA A/B tensors on buffer types compatible with selected base tensors.
- Context integration in `llama-context.*`: manager lifetime, public API plumbing, and adapter attachment to the inference stack.
- Server hook in `tools/server/server-context.cpp`: feed discarded tokens from context shift into `llama_active_lora_ingest()`.

### Validation Strategy

- Unit-test budget planning, rank selection, embedder swap behavior, and rollover-state transitions without requiring a full large model.
- Add a targeted regression test for update-record ordering and API stats behavior.
- Build and run the affected C++ test target after implementation.

## Post-Design Constitution Check

- `Memory Cascade First`: Still passes. Evicted spans enter Active LoRA through an explicit ingestion API and server context-shift hook.
- `Typed Persistent Self-State`: Still passes. The manager exposes explicit stats and update records rather than hiding state in prompt text.
- `Inspectable Policy, Dense Backend Math`: Still passes. Rank planning, embedder selection, and rollover thresholds remain inspectable in C++ state.
- `Documentation, Tests, and Auditability`: Still passes. The design includes traceable update records, tests, and doc updates as first-class deliverables.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Additional shadow context per Active LoRA | Required to train the adapter without destroying the live serving KV cache | Reusing the serving context would corrupt active inference state |
