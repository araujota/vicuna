# Tasks: Hard Memory Primitives

**Input**: Design documents from `/specs/026-hard-memory-primitives/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Tests**: Required. This feature changes runtime memory behavior and must ship
with targeted automated tests.

## Phase 1: Setup

- [ ] T001 Create feature artifacts for `/Users/tyleraraujo/vicuna/specs/026-hard-memory-primitives/`

---

## Phase 2: Foundational

- [ ] T002 Extend hard-memory public enums/structs in `/Users/tyleraraujo/vicuna/include/llama.h` for primitive kinds, archive batches, query hit metadata, and retrieval summaries
- [ ] T003 Extend internal hard-memory interfaces in `/Users/tyleraraujo/vicuna/src/llama-hard-memory.h`, `/Users/tyleraraujo/vicuna/src/llama-context.h`, and related implementation files for typed archive/query support
- [ ] T004 Implement primitive validation, bounded metadata helpers, multi-record archive batching, and typed query parsing in `/Users/tyleraraujo/vicuna/src/llama-hard-memory.cpp`

**Checkpoint**: Hard-memory primitive core exists and is inspectable.

---

## Phase 3: User Story 1 - Typed Hard-Memory Capture (Priority: P1) 🎯 MVP

**Goal**: Archive runtime artifacts as typed memory primitives instead of a
single generic perturbation blob.

**Independent Test**: Drive self-state and loop activity and verify typed batch
archive traces with multiple primitive kinds.

### Tests for User Story 1

- [ ] T005 [P] [US1] Extend hard-memory archival coverage in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`
- [ ] T006 [P] [US1] Extend cognitive-loop archival coverage in `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`

### Implementation for User Story 1

- [ ] T007 [US1] Upgrade self-state postwrite archival to build typed primitive batches in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] T008 [US1] Add cognitive-loop primitive emitters for trajectories, outcomes, tool observations, and user/self-model fragments in `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`
- [ ] T009 [US1] Preserve bounded thresholds, batch limits, and archive traces across `/Users/tyleraraujo/vicuna/src/llama-hard-memory.cpp` and `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`

**Checkpoint**: Typed archival batches are emitted from runtime activity.

---

## Phase 4: User Story 2 - Retrieval That Cooperates With Self-State And LoRA Bias (Priority: P1)

**Goal**: Parse typed memory metadata from query hits and use it to improve
self-model promotion and LoRA-related bias summaries.

**Independent Test**: Run a hard-memory query and verify typed hit metadata,
retrieval summary fields, and extension/gating cooperation.

### Tests for User Story 2

- [ ] T010 [P] [US2] Add typed query parsing and promotion assertions in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`
- [ ] T011 [P] [US2] Add retrieval-summary/gating compatibility assertions in `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp` and `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`

### Implementation for User Story 2

- [ ] T012 [US2] Add retrieval-summary state and typed hit metadata propagation in `/Users/tyleraraujo/vicuna/src/llama-hard-memory.cpp` and `/Users/tyleraraujo/vicuna/src/llama-hard-memory.h`
- [ ] T013 [US2] Upgrade hard-memory-to-self-model promotion logic in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] T014 [US2] Feed hard-memory retrieval summaries into functional-gating observations in `/Users/tyleraraujo/vicuna/src/llama-active-lora.h`, `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`, and `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`

**Checkpoint**: Retrieval is typed and cooperates with self-state and LoRA bias.

---

## Phase 5: User Story 3 - Extensible Tool And Memory Primitive Contract (Priority: P2)

**Goal**: Make the primitive model safe and extensible for future tools.

**Independent Test**: A maintainer can inspect docs and API types and add
custom primitives without changing runtime internals.

### Tests for User Story 3

- [ ] T015 [P] [US3] Add public API validation coverage for custom primitive writes in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

### Implementation for User Story 3

- [ ] T016 [US3] Expose public archive/query helper surfaces and validation defaults in `/Users/tyleraraujo/vicuna/include/llama.h` and `/Users/tyleraraujo/vicuna/src/llama-hard-memory.cpp`
- [ ] T017 [US3] Document primitive vocabulary and extension guidance in `/Users/tyleraraujo/vicuna/README.md`, `/Users/tyleraraujo/vicuna/tools/server/README-dev.md`, `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`, and `/Users/tyleraraujo/vicuna/Vicuña_WP.md`

**Checkpoint**: Primitive contract is documented and externally usable.

---

## Phase 6: Polish & Validation

- [ ] T018 Run targeted build/test validation for `test-self-state`, `test-cognitive-loop`, and `test-active-lora`
- [ ] T019 Inspect archive/query payloads and traces for boundedness and metadata correctness

## Dependencies & Execution Order

- T002-T004 block the story phases.
- US1 should land before US2 because retrieval logic depends on the new typed
  primitive model.
- US3 documentation and public-surface polish can land after the runtime core,
  but before final validation.
