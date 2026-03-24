# Tasks: Authoritative ReAct Action Contract Guarantee

**Input**: Design documents from `/specs/103-react-action-contract-guarantee/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md

**Tests**: Targeted automated tests are REQUIRED because this change alters authoritative runtime behavior and Telegram-visible outcomes.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the feature artifacts and affected paths before editing runtime behavior.

- [ ] T001 Review `/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/spec.md`, `/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/plan.md`, and `/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/research.md` against the current runtime implementation.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared parser and prompt-preparation changes that all stories depend on.

- [ ] T002 Update staged assistant prefill ownership in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` so fixed action bytes are host-owned for all staged phases.
- [ ] T003 Update staged grammar or parser handling in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` and `/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp` so the model only generates variable tails while strict phase validation remains intact.
- [ ] T004 Tighten retry and fallback classification in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` so malformed control payloads remain internal and control-shaped JSON is not accepted as visible prose.

**Checkpoint**: The runtime owns the staged action contract and has one explicit malformed-control classification path.

---

## Phase 3: User Story 1 - Staged tool turns always carry the exact action contract (Priority: P1) 🎯 MVP

**Goal**: Make the exact staged action contract runtime-owned and keep the parser aligned with that contract.

**Independent Test**: Run targeted runtime tests proving the parser resolves the exact staged action value even when the model omits the fixed action field.

### Tests for User Story 1

- [ ] T005 [P] [US1] Add regression coverage in `/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp` and `/Users/tyleraraujo/vicuna/tests/test-server-common.cpp` for runtime-owned staged action normalization across `select_tool_family`, `select_method`, `emit_arguments`, `decide_after_tool`, and `emit_response`.

### Implementation for User Story 1

- [ ] T006 [US1] Refactor staged parsing in `/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp` and `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` so the exact phase action is normalized when omitted and conflicting actions remain invalid.
- [ ] T007 [US1] Add shared control-payload helpers in `/Users/tyleraraujo/vicuna/tools/server/server-common.cpp` and `/Users/tyleraraujo/vicuna/tools/server/server-common.h` for staged action normalization and control-shaped JSON detection.

**Checkpoint**: Staged phases no longer rely on the model to perfectly regenerate fixed action text.

---

## Phase 4: User Story 2 - Missing or malformed action labels stay inside retry (Priority: P1)

**Goal**: Keep malformed staged control inside retry and rewind, never in Telegram-visible content.

**Independent Test**: Reproduce a missing-action staged payload and verify the runtime retries or rewinds without publishing the malformed artifact.

### Tests for User Story 2

- [ ] T008 [P] [US2] Add regression coverage in `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp` for missing-action staged payload retry and stage rewind behavior.

### Implementation for User Story 2

- [ ] T009 [US2] Update `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` retry handling so malformed staged control remains internal and terminal user-visible failures stay generic.
- [ ] T010 [US2] Update `/Users/tyleraraujo/vicuna/tools/server/README-dev.md` to document retry-only handling for malformed staged control.

**Checkpoint**: Malformed staged payloads stay inside retry and no internal controller text is exposed to users.

---

## Phase 5: User Story 3 - Non-staged fallback cannot mistake control JSON for a user reply (Priority: P2)

**Goal**: Prevent bare control-shaped JSON from being treated as terminal visible prose.

**Independent Test**: Feed control-shaped JSON into the non-staged fallback path and verify it retries instead of surfacing the artifact.

### Tests for User Story 3

- [ ] T011 [P] [US3] Add regression coverage in `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp` for rejection of control-shaped JSON in non-staged fallback while preserving ordinary prose fallback.

### Implementation for User Story 3

- [ ] T012 [US3] Tighten `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` fallback classification so control-shaped visible JSON is rejected as malformed control rather than accepted as visible terminal text.

**Checkpoint**: Raw controller artifacts can no longer leak through visible-tail fallback.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T013 [P] Update `/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/quickstart.md` with the exact validation commands used.
- [ ] T014 Run `cmake --build build --target test-cognitive-loop -j8` from `/Users/tyleraraujo/vicuna`.
- [ ] T015 Run `./build/bin/test-cognitive-loop` from `/Users/tyleraraujo/vicuna`.
- [ ] T016 Capture the final local validation results in `/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/plan.md` or `/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/quickstart.md` if commands change during implementation.

## Dependencies & Execution Order

- T001 must complete before implementation.
- T002 through T004 block all user stories.
- T005 can be written before or alongside T006 and T007.
- T008 can be written before or alongside T009 and T010.
- T011 can be written before or alongside T012.
- T014 through T016 run after implementation is complete.

## Implementation Strategy

1. Complete the foundational runtime hardening in `server-context.cpp` and `server-openclaw-fabric.cpp`.
2. Land the P1 guarantee that staged phases own their fixed action tokens.
3. Land the P1 retry-only handling so malformed staged control never becomes user-visible.
4. Land the P2 fallback rejection for control-shaped JSON.
5. Run the targeted runtime tests and update docs.
