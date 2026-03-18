# Tasks: PR #4 CI Remediation

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/`  
**Prerequisites**: plan.md, spec.md

**Tests**: Targeted automated tests and CI-equivalent baseline checks are REQUIRED because this work changes server behavior and merge-gating quality signals.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Record the failing CI scope and align the remediation with repository workflow.

- [x] T001 Create the remediation spec, plan, and task artifacts in `/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/`
- [x] T002 [P] Capture failing PR #4 checks and automated review findings that will drive the remediation in `/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/spec.md` and `/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/plan.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the runtime and quality guardrails before story-specific fixes land.

- [x] T003 Inspect the repository CI workflows and baseline scripts in `/Users/tyleraraujo/vicuna/.github/workflows/vicuna-ci.yml`, `/Users/tyleraraujo/vicuna/.github/scripts/check_lizard_baseline.py`, and `/Users/tyleraraujo/vicuna/.github/scripts/check_clang_tidy_baseline.py`
- [x] T004 Identify the exact runtime, quality, and review hot spots in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`, `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`, `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`, `/Users/tyleraraujo/vicuna/src/llama-context.cpp`, and `/Users/tyleraraujo/vicuna/tests/test-serving-lora-stack.cpp`

**Checkpoint**: The remediation targets and validation gates are explicit.

---

## Phase 3: User Story 1 - Restore Runtime Snapshot Persistence (Priority: P1) 🎯 MVP

**Goal**: Make runtime snapshot persistence succeed when optional archive state is absent.

**Independent Test**: `unit/test_basic.py::test_runtime_snapshot_survives_restart` passes and produces a runtime snapshot file.

### Tests for User Story 1

- [x] T005 [P] [US1] Update or confirm persistence-backed assertions in `/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_basic.py`

### Implementation for User Story 1

- [x] T006 [US1] Guard optional functional/process-functional archive persistence paths in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [x] T007 [US1] Adjust any supporting runtime accessors needed for absent optional state in `/Users/tyleraraujo/vicuna/src/llama-context.cpp` and `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`

**Checkpoint**: Runtime persistence works without requiring optional LoRA archive initialization.

---

## Phase 4: User Story 2 - Clear PR #4 Quality Gates (Priority: P1)

**Goal**: Remove new clang-tidy and lizard regressions introduced by PR #4.

**Independent Test**: The repository baseline scripts report no new or changed warnings for the remediated branch.

### Tests for User Story 2

- [x] T008 [P] [US2] Re-run the lizard baseline script from `/Users/tyleraraujo/vicuna/.github/scripts/check_lizard_baseline.py`
- [ ] T009 [P] [US2] Re-run the clang-tidy baseline script from `/Users/tyleraraujo/vicuna/.github/scripts/check_clang_tidy_baseline.py`

### Implementation for User Story 2

- [x] T010 [US2] Refactor `llama_cognitive_loop::active_loop_process` and `llama_cognitive_loop::dmn_tick` in `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp` to reduce complexity and remove shadowing or nested-conditional regressions
- [x] T011 [US2] Refactor discovered-state and expanded-model helpers in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp` to reduce complexity and remove nested-conditional regressions
- [x] T012 [US2] Clean up touched test include usage and related baseline regressions in `/Users/tyleraraujo/vicuna/tests/test-serving-lora-stack.cpp`

**Checkpoint**: The changed runtime and test paths no longer regress repository quality baselines.

---

## Phase 5: User Story 3 - Resolve Valid PR Review Findings (Priority: P2)

**Goal**: Fix the automated review findings that are technically valid and document the review state accurately.

**Independent Test**: The reviewed code paths no longer contain the flagged valid patterns and behavior remains unchanged.

### Tests for User Story 3

- [x] T013 [P] [US3] Rebuild or run targeted validation for the touched runtime and API paths

### Implementation for User Story 3

- [x] T014 [US3] Replace large-object-by-value process-functional import surfaces in `/Users/tyleraraujo/vicuna/include/llama.h`, `/Users/tyleraraujo/vicuna/src/llama-context.cpp`, `/Users/tyleraraujo/vicuna/src/llama-context.h`, and call sites
- [x] T015 [US3] Add focused clarifying structure or comments for valid large-function findings in `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp` and `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] T016 [US3] Record the absence of Copilot review comments and the disposition of automated review findings in the final remediation summary

**Checkpoint**: Valid automated review findings are fixed, and review status is accurately summarized.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and artifact alignment.

- [x] T017 [P] Update task completion state in `/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/tasks.md`
- [x] T018 [P] Run the focused Python server tests and targeted native verification commands for the touched files
- [ ] T019 [P] Run the quality baseline scripts with the remediated branch state

## Dependencies & Execution Order

- Setup and Foundational tasks must complete before code edits.
- User Story 1 should land before final Python verification because it fixes the concrete runtime defect.
- User Story 2 should land before the final quality baseline verification.
- User Story 3 can proceed alongside User Story 2 once the core hot spots are identified.

## Parallel Opportunities

- T002 and T003 can run in parallel after T001.
- T008 and T009 can run in parallel during verification.
- T013 can run in parallel with T017 once implementation is complete.

## Implementation Strategy

### MVP First

1. Finish setup and foundational inspection.
2. Fix runtime snapshot persistence.
3. Run the focused Python test that was failing.
4. Refactor the changed runtime paths to clear quality gates.
5. Apply valid review-finding cleanup and re-run verification.
