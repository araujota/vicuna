# Tasks: Functional LoRA Bootstrap Perturbation

**Input**: Design documents from `/specs/030-functional-lora-bootstrap-perturbation/`
**Prerequisites**: plan.md, spec.md, research.md

**Tests**: Targeted automated tests are REQUIRED because this changes inference behavior in the functional LoRA runtime path.

**Organization**: Tasks are grouped by user story to enable independent implementation and validation.

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Confirm the scope and contracts in `/Users/tyleraraujo/vicuna/specs/030-functional-lora-bootstrap-perturbation/spec.md`, `/Users/tyleraraujo/vicuna/specs/030-functional-lora-bootstrap-perturbation/research.md`, and `/Users/tyleraraujo/vicuna/specs/030-functional-lora-bootstrap-perturbation/plan.md`.

---

## Phase 2: Foundational (Blocking Prerequisites)

- [ ] T002 Add bootstrap perturbation config and trace fields to `/Users/tyleraraujo/vicuna/include/llama.h`.
- [ ] T003 Add bootstrap runtime adapter ownership and helper policy to `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.

**Checkpoint**: The typed surface and runtime scaffolding exist for implementation.

---

## Phase 3: User Story 1 - Keep New Functional LoRAs Near No-Op While Letting Them Accidentally Help (Priority: P1) 🎯 MVP

**Goal**: Preserve zero-initialized learned functional adapters while giving each active family a tiny explicit bootstrap perturbation path.

**Independent Test**: Inspect a fresh context and an activated family to verify the learned adapter remains a no-op while bootstrap perturbation is available and traceable.

### Tests For User Story 1

- [ ] T004 [P] [US1] Extend initialization assertions in `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`.
- [ ] T005 [P] [US1] Add activation-trace assertions for sampled bootstrap perturbation in `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.

### Implementation For User Story 1

- [ ] T006 [US1] Add per-family bootstrap adapters and tiny random initialization in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T007 [US1] Attach bootstrap adapters only during family activation in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.

**Checkpoint**: Functional families remain learned no-ops at birth but can accidentally perturb behavior when invoked.

---

## Phase 4: User Story 2 - Decay Exploration With Use But Never To Zero (Priority: P2)

**Goal**: Make bootstrap perturbation strongest early and progressively smaller as family usage accumulates, while preserving a nonzero floor.

**Independent Test**: Repeatedly activate a family and verify the current bootstrap standard deviation declines toward, but never reaches below, the configured minimum.

### Tests For User Story 2

- [ ] T008 [P] [US2] Add bootstrap decay-to-floor assertions in `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`.

### Implementation For User Story 2

- [ ] T009 [US2] Add explicit bootstrap decay policy and usage counters in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T010 [US2] Expose bootstrap std and sampled perturbation in family state and activation trace in `/Users/tyleraraujo/vicuna/include/llama.h` and `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.

**Checkpoint**: Bootstrap exploration decays with use and remains minimally available forever.

---

## Phase 5: User Story 3 - Keep Bootstrap Exploration Inspectable And Compatible With Runtime Learning (Priority: P3)

**Goal**: Make bootstrap perturbation explicit, bounded, and clearly distinct from the learned adapter path.

**Independent Test**: Read config/state/trace surfaces and verify developers can distinguish gate exploration from bootstrap perturbation.

### Tests For User Story 3

- [ ] T011 [P] [US3] Add observability assertions for bootstrap fields in `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`.

### Implementation For User Story 3

- [ ] T012 [US3] Update operator and architecture docs in `/Users/tyleraraujo/vicuna/README.md` and `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`.

**Checkpoint**: Bootstrap perturbation is explicit, documented, and production-legible.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T013 Run targeted validation commands for `test-active-lora` and `test-cognitive-loop`.
- [ ] T014 Review code comments and field naming for clarity and consistency with the spec.

## Dependencies & Execution Order

- T002-T003 block all runtime implementation work.
- US1 lands before US2 because decay policy depends on the bootstrap substrate existing.
- US3 depends on US1-US2 so docs and observability reflect the integrated final behavior.
- Final validation follows all implementation changes.
