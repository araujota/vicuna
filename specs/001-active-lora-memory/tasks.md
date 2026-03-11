# Tasks: Sliding-Window Active LoRA Memory

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/001-active-lora-memory/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/active-lora-api.md`, `quickstart.md`

## Phase 1: Setup

- [ ] T001 Add the new Active LoRA runtime sources to `/Users/tyleraraujo/vicuna/src/CMakeLists.txt`
- [ ] T002 Create the Active LoRA manager declarations in `/Users/tyleraraujo/vicuna/src/llama-active-lora.h`
- [ ] T003 Create the Active LoRA manager implementation scaffold in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`

## Phase 2: Foundational

- [ ] T004 Extend the public Active LoRA configuration and stats API in `/Users/tyleraraujo/vicuna/include/llama.h`
- [ ] T005 Wire the public API entry points through `/Users/tyleraraujo/vicuna/src/llama.cpp`
- [ ] T006 Extend adapter internals for mutable runtime LoRA allocation in `/Users/tyleraraujo/vicuna/src/llama-adapter.h`
- [ ] T007 Implement mutable runtime LoRA allocation helpers in `/Users/tyleraraujo/vicuna/src/llama-adapter.cpp`
- [ ] T008 Integrate Active LoRA manager lifetime and plumbing into `/Users/tyleraraujo/vicuna/src/llama-context.h`
- [ ] T009 Integrate Active LoRA initialization, ingestion, and stats handling into `/Users/tyleraraujo/vicuna/src/llama-context.cpp`

## Phase 3: User Story 1 - Budgeted Active LoRA Memory (Priority: P1)

**Goal**: Build the fixed-budget Active LoRA manager, compute rank from live RAM/VRAM budgets, and write evicted spans into a mutable adapter through a shadow training context.

**Independent Test**: Initialize Active LoRA on a context, ingest an evicted token span, and verify that a non-zero-rank mutable LoRA updates while reported budget usage stays within configured proportions.

- [ ] T010 [US1] Implement budget planning, target selection, and rank computation in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T011 [US1] Implement the shadow training context and evicted-span writer path in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T012 [US1] Attach the Active LoRA adapter to inference and training contexts in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`

## Phase 4: User Story 2 - Swappable Embedding Strategy (Priority: P2)

**Goal**: Make span embedding strategy a replaceable policy surface that feeds admission and update metadata without changing the surrounding manager API.

**Independent Test**: Run the same ingestion flow with two embedding strategies and verify that stats/reporting reflect the chosen strategy while the write path remains unchanged.

- [ ] T013 [US2] Implement the embedding-strategy interface and default hash-based strategy in `/Users/tyleraraujo/vicuna/src/llama-active-lora.h`
- [ ] T014 [US2] Implement embedding-based admission, strategy selection, and update metadata in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T015 [US2] Surface embedding strategy selection through the public config in `/Users/tyleraraujo/vicuna/include/llama.h`

## Phase 5: User Story 3 - Inspectable Rollover and Audit Trail (Priority: P3)

**Goal**: Expose update records, rollover readiness, and the first automatic ingestion hook from server context shift.

**Independent Test**: Force repeated context-shift evictions in the server path and verify ordered update records plus rollover-ready state once the configured boundary is crossed.

- [ ] T016 [US3] Add update-record and rollover-state tracking to `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T017 [US3] Feed discarded context-shift tokens into Active LoRA ingestion from `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T018 [P] Add targeted Active LoRA regression tests in `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`
- [ ] T019 [P] Register the new Active LoRA test target in `/Users/tyleraraujo/vicuna/tests/CMakeLists.txt`
- [ ] T020 Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` to document budgeted Active LoRA memory, swappable embedders, and rollover semantics
- [ ] T021 Update `/Users/tyleraraujo/vicuna/Vicuña_WP.md` to reflect the implemented Active LoRA write path and budget model

## Dependencies

- Setup must finish before foundational work.
- Foundational work must finish before User Story 1.
- User Story 1 must finish before User Story 2 because the embedder plugs into the Active LoRA manager.
- User Story 1 must finish before User Story 3 because rollover and server ingestion depend on a functioning manager.
- Polish can run after the relevant story work lands.

## Parallel Opportunities

- T018 and T019 can run in parallel once the core runtime API is stable.
- Documentation updates T020 and T021 can run in parallel after the implementation semantics are finalized.

## Implementation Strategy

- Deliver MVP as Phases 1 through 3: a working budgeted Active LoRA manager with explicit API and evicted-span writes.
- Add strategy swapping in Phase 4 without changing the core ingestion contract.
- Add server auto-ingestion plus rollover auditability in Phase 5.
- Finish with tests and doc updates in Phase 6.
