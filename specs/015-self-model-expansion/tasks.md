# Tasks: Self-Model Expansion For Efficient Goal Pursuit, User Satisfaction, And Self-Repair

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, implementation-approach.md

**Tests**: Future implementation should ship with targeted automated tests for
self-state profile updates, multi-horizon summaries, favorable-state parity,
Active Loop parity, import/export parity, and trace inspection.

## Phase 1: Setup

- [x] T001 Review `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/audit-report.md`,
  `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`, and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md`.
- [x] T002 Inspect the current self-state, favorable-state, and cognitive-loop
  surfaces in `/Users/tyleraraujo/vicuna/include/llama.h`,
  `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`, and
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`.

## Phase 2: External Research

- [x] T003 Use GitHub codebase research to inspect explicit agent-state and
  memory schemas in public agent systems.
- [x] T004 Use web research to gather primary sources on active inference,
  generative agents, self-correction, user satisfaction estimation, and
  uncertainty calibration.

## Phase 3: User Story 1 - Richer Self-State Taxonomy (Priority: P1)

**Goal**: Define what additional self-state should be represented and why.

**Independent Test**: A maintainer can identify the proposed profile families,
their semantics, and their benefit to user satisfaction, goal completion, or
self-repair.

- [x] T005 [US1] Summarize current local self-model limitations and strengths in
  `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/research.md`.
- [x] T006 [US1] Define the expanded self-model taxonomy and typed entities in
  `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/data-model.md`.
- [x] T007 [US1] Write the architectural rationale and grouping rules in
  `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/implementation-approach.md`.

## Phase 4: User Story 2 - Multi-Timescale And Forecast State (Priority: P1)

**Goal**: Define horizon slices, trend state, and forecast state that can
support lower-step behavior and faster recovery.

**Independent Test**: A maintainer can see how instantaneous state, EMA trends,
baselines, and prediction errors should be represented and updated.

- [x] T008 [US2] Record external evidence for multi-timescale and reflective
  state in `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/research.md`.
- [x] T009 [US2] Define horizon slices, forecast traces, and prediction-error
  structs in `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/data-model.md`.
- [x] T010 [US2] Write the update-order and rollout architecture in
  `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/implementation-approach.md`.

## Phase 5: User Story 3 - Explicit Motivation For Safe Self-Improvement (Priority: P2)

**Goal**: Define bounded state that can later motivate self-repair and
self-improvement without hiding policy.

**Independent Test**: A maintainer can identify explicit update-worthiness,
expected-gain, and evidence-deficit surfaces and how they relate to governance.

- [x] T011 [US3] Capture the motivation-related research and public patterns in
  `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/research.md`.
- [x] T012 [US3] Define self-improvement governance entities in
  `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/data-model.md`.
- [x] T013 [US3] Write the governance integration and rollout guidance in
  `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/implementation-approach.md`.

## Phase 6: Polish

- [x] T014 Verify that each major recommendation is backed by local file
  references or primary-source external citations.
- [x] T015 Ensure the design preserves current prewrite/postwrite, favorable,
  counterfactual, remediation, and loop parity.
- [x] T016 Produce a future implementation sequence and test plan.

## Phase 7: Future Implementation Follow-Through

- [ ] T017 Add new public API structs and enums in
  `/Users/tyleraraujo/vicuna/include/llama.h` for typed self profiles, horizon
  slices, and forecast traces.
- [ ] T018 Extend `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp` and
  `/Users/tyleraraujo/vicuna/src/llama-self-state.h` to compute and persist the
  new profiles while preserving current register parity.
- [ ] T019 Extend `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp` to
  consume the new efficiency, recovery, and user-outcome surfaces without
  regressing the current Active Loop or DMN decision traces.
- [ ] T020 Add parity-sensitive regression coverage in
  `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp` and
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.
- [ ] T021 Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md` once implementation lands.
