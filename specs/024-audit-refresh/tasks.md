# Tasks: Refreshed Functionality Audit Of The Exotic-Intelligence Runtime

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/`
**Prerequisites**: plan.md, spec.md

**Tests**: No new runtime tests are required because this is a documentation and
research deliverable. Validation is evidence quality: repository references,
commit references, and primary-source citations.

## Phase 1: Setup

- [ ] T001 Create `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/` and add `spec.md`, `plan.md`, and `tasks.md`

## Phase 2: Foundational Research

- [ ] T002 Recover the original audit baseline from `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/audit-report.md`
- [ ] T003 Inspect current runtime evidence in `/Users/tyleraraujo/vicuna/include/llama.h`, `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`, `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`, `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`, and `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T004 Research GitHub commit history for changes since the original audit, including commits `69e222cbe9f893145d0c91d59618131921193f40`, `07437946cad0c5360b3d264cc9605cb635d2e733`, and `9858ba91c94d23fdb3112a61bf6512c3b625ddf2`
- [ ] T005 Research external primary sources for active inference/allostasis, stateful agent runtimes, durable agent memory, and bounded self-improvement

## Phase 3: User Story 1 - Recover The Original Audit Baseline (Priority: P1)

**Goal**: Restate the first audit’s verdict and gaps clearly enough to serve as
the comparison baseline for the refresh.

**Independent Test**: A reader can identify the original verdict and priority
gaps directly from the refresh materials.

- [ ] T006 [US1] Write original-baseline findings into `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/research.md`

## Phase 4: User Story 2 - Reassess Current Runtime Status (Priority: P1)

**Goal**: Produce a current audit with explicit judgments on functionality,
elegance, generalizability, expandability, and current RSI status.

**Independent Test**: A reader can understand Vicuña’s current status without
opening the source tree.

- [ ] T007 [US2] Write current-runtime evidence and gap-delta notes into `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/research.md`
- [ ] T008 [US2] Write the refreshed audit report in `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/audit-report.md`

## Phase 5: User Story 3 - Compare To External Best Practice (Priority: P2)

**Goal**: Tie the refreshed verdict and recommendations to current public
theses and implementations for this class of architecture.

**Independent Test**: Every major recommendation is externally justified.

- [ ] T009 [US3] Add external SOTA comparisons and next-step recommendations to `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/research.md`
- [ ] T010 [US3] Add externally justified recommendations and final verdict language to `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/audit-report.md`

## Phase 6: Polish

- [ ] T011 Verify that every major local claim in `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/audit-report.md` has a repository reference
- [ ] T012 Verify that every major external comparison in `/Users/tyleraraujo/vicuna/specs/024-audit-refresh/audit-report.md` has a primary-source or official-project citation
