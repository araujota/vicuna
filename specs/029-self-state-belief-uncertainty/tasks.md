# Tasks: Partial-Observation Self-State Belief Gradient

**Input**: Design documents from `/specs/029-self-state-belief-uncertainty/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Tests**: Targeted automated tests are REQUIRED because this changes inference behavior and functional gain control.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Create or confirm `specs/029-self-state-belief-uncertainty/` artifacts align with the final scope in `/Users/tyleraraujo/vicuna/specs/029-self-state-belief-uncertainty/`.

---

## Phase 2: Foundational (Blocking Prerequisites)

- [ ] T002 Add public belief-layer config and trace structs to `/Users/tyleraraujo/vicuna/include/llama.h`.
- [ ] T003 Add internal belief-layer state types and invariants to `/Users/tyleraraujo/vicuna/src/llama-self-state.h`.
- [ ] T004 Define fixed-width belief-summary features for functional gain routing in `/Users/tyleraraujo/vicuna/src/llama-active-lora.h`.

**Checkpoint**: The typed runtime surface exists and user-story work can begin.

---

## Phase 3: User Story 1 - Preserve Explicit Self-State While Admitting Partial Observability (Priority: P1) 🎯 MVP

**Goal**: Treat explicit self-state as observation while adding a bounded belief filter for incompletely modeled cares.

**Independent Test**: Keep explicit self-state fixed, vary evidence tuples, and verify bounded belief summaries update without changing explicit register values.

### Tests For User Story 1

- [ ] T005 [P] [US1] Add belief-update unit tests to `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`.
- [ ] T006 [P] [US1] Add clipping and slot-budget tests to `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`.

### Implementation For User Story 1

- [ ] T007 [US1] Implement belief evidence extraction in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`.
- [ ] T008 [US1] Implement bounded belief-slot filtering and summary synthesis in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`.
- [ ] T009 [US1] Implement promotion-candidate surfacing hooks in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`.

**Checkpoint**: Explicit self-state remains primary; belief summary exists and is testable.

---

## Phase 4: User Story 2 - Let Functional Gain Respond To Uncertainty Over Missing Cares (Priority: P2)

**Goal**: Feed the new belief summary into functional gain prediction without breaking the current gradient semantics.

**Independent Test**: Replay identical explicit gradients with different belief summaries and verify bounded gain differences.

### Tests For User Story 2

- [ ] T010 [P] [US2] Add gating-input compatibility tests to `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`.
- [ ] T011 [P] [US2] Add active/DMN replay tests for belief-driven gain changes to `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.

### Implementation For User Story 2

- [ ] T012 [US2] Extend the gating input builder to append belief-summary features in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T013 [US2] Update gain trace surfaces to expose belief influence in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T014 [US2] Thread belief-summary capture through active and DMN loop invocation points in `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`.

**Checkpoint**: The gate responds to partial-observability summaries while remaining explicit and bounded.

---

## Phase 5: User Story 3 - Keep The Belief Layer Inspectable, Bounded, And Promotable (Priority: P3)

**Goal**: Make uncertainty over missing cares auditable and safely promotable into explicit self-model state.

**Independent Test**: Inspect trace output after repeated residual evidence and verify promotion candidates appear without automatic ontology mutation.

### Tests For User Story 3

- [ ] T015 [P] [US3] Add promotion-candidate stability tests to `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`.
- [ ] T016 [P] [US3] Add trace-surface coverage tests to `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.

### Implementation For User Story 3

- [ ] T017 [US3] Add public belief and promotion stats to `/Users/tyleraraujo/vicuna/include/llama.h`.
- [ ] T018 [US3] Expose belief-layer observability through runtime traces in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp` and `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T019 [US3] Document the ontology and operating model in `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`, `/Users/tyleraraujo/vicuna/README.md`, and `/Users/tyleraraujo/vicuna/Vicuña_WP.md`.

**Checkpoint**: Belief-state uncertainty is inspectable, bounded, and promotion-ready.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T020 Run targeted validation commands for `test-self-state`, `test-active-lora`, and `test-cognitive-loop`.
- [ ] T021 Review belief update math and comments for clarity in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`.
- [ ] T022 Confirm public headers and docs match the final runtime behavior.

## Dependencies & Execution Order

- T002-T004 block all implementation work.
- US1 must land before US2 because the gate consumes the belief summary produced by self-state.
- US3 depends on US1 and US2 because observability must reflect the actual integrated runtime.
- Final validation follows all completed implementation tasks.
