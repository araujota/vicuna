# Tasks: Frozen Past-LoRA Condensation Stack

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/002-past-lora-condensation/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/past-lora-api.md`, `quickstart.md`

## Phase 1: Setup

- [ ] T001 Extend the existing memory-stage sources for frozen bucket support in `/Users/tyleraraujo/vicuna/src/llama-active-lora.h`
- [ ] T002 Extend the existing memory-stage implementation scaffold for bucket condensation in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T003 Register any new test source in `/Users/tyleraraujo/vicuna/tests/CMakeLists.txt`

## Phase 2: Foundational

- [ ] T004 Extend the public past-stack configuration and stats API in `/Users/tyleraraujo/vicuna/include/llama.h`
- [ ] T005 Wire the public API entry points through `/Users/tyleraraujo/vicuna/src/llama.cpp`
- [ ] T006 Extend adapter internals with explicit gain state in `/Users/tyleraraujo/vicuna/src/llama-adapter.h`
- [ ] T007 Implement gain-aware runtime adapter helpers in `/Users/tyleraraujo/vicuna/src/llama-adapter.cpp`
- [ ] T008 Integrate past-stack lifetime and tick plumbing into `/Users/tyleraraujo/vicuna/src/llama-context.h`
- [ ] T009 Integrate past-stack initialization, ticking, and stats handling into `/Users/tyleraraujo/vicuna/src/llama-context.cpp`

## Phase 3: User Story 1 - Frozen Temporal Memory Stack (Priority: P1)

**Goal**: Add five frozen temporal buckets, explicit bucket configs, and concurrent decayed inference application.

**Independent Test**: Force an Active rollover, run a tick, and verify that `past_week` becomes populated, remains frozen between non-due ticks, and applies a visible effective scale.

- [ ] T010 [US1] Implement temporal bucket state, budget planning, and frozen snapshot metadata in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T011 [US1] Implement frozen bucket allocation and atomic snapshot replacement in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T012 [US1] Apply per-bucket decayed inference scales through the runtime adapter map in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`

## Phase 4: User Story 2 - Directional-Gain Memory Updates (Priority: P2)

**Goal**: Enforce normalized direction plus bounded gain for both Active writes and frozen artifacts.

**Independent Test**: Ingest multiple spans and verify that Active gains stay bounded, then condense into a frozen bucket and verify the frozen artifact reports normalized direction state plus bounded gain statistics.

- [ ] T013 [US2] Implement gain-aware Active update normalization in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T014 [US2] Implement direction-plus-gain condensation recompression in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T015 [US2] Surface gain and normalization stats through `/Users/tyleraraujo/vicuna/include/llama.h`

## Phase 5: User Story 3 - Scheduled Condensation Across Time Buckets (Priority: P3)

**Goal**: Add explicit condensation jobs and runtime ticks for `active -> past_week -> past_month -> past_quarter -> past_year -> all_time`.

**Independent Test**: Advance explicit timestamps across multiple ticks and verify that jobs run in order, bucket versions change only on due ticks, and older buckets populate over time.

- [ ] T016 [US3] Implement condensation job scheduling and due-state tracking in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T017 [US3] Hook periodic past-stack ticks into `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T018 [US3] Add model-backed condensation coverage in `/Users/tyleraraujo/vicuna/tests/test-past-lora.cpp`

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T019 [P] Extend `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp` to cover gain-bounded Active updates
- [ ] T020 [P] Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` to document the frozen temporal stack, decay scales, and condensation semantics
- [ ] T021 [P] Update `/Users/tyleraraujo/vicuna/Vicuña_WP.md` to reflect the implemented past-stack buckets, periodic jobs, and direction-versus-gain representation

## Dependencies

- Setup must finish before foundational work.
- Foundational work must finish before User Story 1.
- User Story 1 must finish before User Story 2 because gain-aware frozen artifacts depend on bucket state.
- User Story 2 must finish before User Story 3 because the scheduler operates on the direction-plus-gain representation.
- Polish can run after the relevant story work lands.

## Parallel Opportunities

- T019 and T018 can run in parallel once the public stats surface stabilizes.
- T020 and T021 can run in parallel after the implementation semantics are finalized.

## Implementation Strategy

- Deliver MVP as Phases 1 through 3: a frozen `past_week` bucket plus visible decay and bucket stats.
- Add direction-versus-gain enforcement in Phase 4 without changing the external bucket lifecycle.
- Add multi-bucket scheduled condensation in Phase 5.
- Finish with tests and documentation in Phase 6.
