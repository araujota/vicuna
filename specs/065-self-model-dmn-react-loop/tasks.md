# Tasks: Self-Model-Translated DMN ReAct Loop

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Targeted native tests are required for self-model revision gating,
translation determinism, DMN episode supersession, and Telegram relay tool
integration.

## Phase 1: Setup

- [ ] T001 Create typed feature scaffolding in `include/llama.h`,
  `src/llama-cognitive-loop.h`, and `src/llama-self-state.h` for self-model
  revisions, translation inputs, prompt revisions, and DMN episode lineage.
- [ ] T002 Update `ARCHITECTURE.md` to add the target-state design sections for
  hidden self-model translation, shared DMN/active runner semantics, and
  Telegram relay tool behavior before implementation diverges from docs.

## Phase 2: Foundational

- [ ] T003 Implement `SelfModelRevision` detection and materiality policy in
  `src/llama-self-state.cpp`, `src/llama-self-state.h`, `src/llama-context.cpp`,
  and `src/llama-context.h`.
- [ ] T004 Implement bounded `SelfModelTranslationInput` assembly and explicit
  `ReportableConceptFrame` selection in `src/llama-cognitive-loop.cpp` and
  `src/llama-cognitive-loop.h`.
- [ ] T005 Implement `DmnPromptRevision` realization, lineage tracking, and
  supersession traces in `src/llama-cognitive-loop.cpp` and `include/llama.h`.
- [ ] T006 Update server inspection/export surfaces in `tools/server/server-context.cpp`
  to expose prompt-revision and DMN-episode traces.

**Checkpoint**: Hidden-to-reportable translation exists and is inspectable
before DMN execution policy is switched over.

## Phase 3: User Story 1 - Hidden self-model becomes a bounded reportable motivation prompt (Priority: P1)

**Goal**: Compile typed self-model changes into deterministic bounded natural
language prompt revisions through an explicit concept layer.

**Independent Test**: Change self-state in targeted tests and verify prompt
revision output and lineage are deterministic under explicit materiality policy.

### Tests for User Story 1

- [ ] T007 [P] [US1] Add revision-materiality and extension-whitelist tests in
  `tests/test-self-state.cpp`.
- [ ] T008 [P] [US1] Add prompt-realization determinism tests in
  `tests/test-cognitive-loop.cpp`.

### Implementation for User Story 1

- [ ] T009 [US1] Implement concept-selection and macro-ordering policy in
  `src/llama-cognitive-loop.cpp`.
- [ ] T010 [US1] Implement bounded natural-language realization for
  `DmnPromptRevision` in `src/llama-cognitive-loop.cpp`.
- [ ] T011 [US1] Add typed C/API inspection fields in `include/llama.h` for
  prompt revisions and translation traces.

## Phase 4: User Story 2 - DMN reuses the active planner/tool runner as an internal ReAct loop (Priority: P1)

**Goal**: Start DMN episodes from prompt revisions and run them through the
shared planner/tool substrate instead of DMN winner-action scoring.

**Independent Test**: Run DMN episodes that plan, invoke tools, integrate
results, and supersede on newer prompt revisions without using the old DMN
action stack.

### Tests for User Story 2

- [ ] T012 [P] [US2] Add DMN prompt-revision episode-start and supersession
  tests in `tests/test-cognitive-loop.cpp`.
- [ ] T013 [P] [US2] Add regression tests proving the old endogenous-seed path
  no longer governs DMN behavior in `tests/test-cognitive-loop.cpp`.

### Implementation for User Story 2

- [ ] T014 [US2] Refactor `src/llama-cognitive-loop.cpp` so DMN episodes enter
  through the shared planner/tool runner.
- [ ] T015 [US2] Remove or repurpose obsolete DMN winner-action code paths in
  `src/llama-cognitive-loop.cpp` and `src/llama-cognitive-loop.h`, including
  `assemble_seed(...)`, `fill_dmn_candidate(...)`, and the dedicated
  `LLAMA_DMN_ACTION_*` winner-selection flow where no longer needed.
- [ ] T016 [US2] Update DMN trace reporting in `tools/server/server-context.cpp`
  to show prompt-revision lineage and shared-runner state rather than
  winner-action summaries.

## Phase 5: User Story 3 - DMN can reach the user through Telegram as a first-class tool (Priority: P2)

**Goal**: Represent DMN-origin user outreach as a tool request/result while
keeping active engagement accounting separate.

**Independent Test**: A DMN episode selects Telegram relay, delivers or fails,
and continues background cognition with the result integrated as a tool
observation.

### Tests for User Story 3

- [ ] T017 [P] [US3] Add runtime tests for DMN-origin relay accounting in
  `tests/test-cognitive-loop.cpp`.
- [ ] T018 [P] [US3] Add bridge/server integration coverage for DMN relay
  request/result handling in the appropriate server or bridge test surface.

### Implementation for User Story 3

- [ ] T019 [US3] Add DMN Telegram relay tool definitions and request/result
  structs in `include/llama.h` and `src/llama-cognitive-loop.h`.
- [ ] T020 [US3] Integrate the relay tool into runtime tool selection and
  result handling in `src/llama-cognitive-loop.cpp` and
  `tools/server/server-context.cpp`.
- [ ] T021 [US3] Update `tools/telegram-bridge/README.md` and any affected
  bridge/server code to document and surface DMN-origin relay behavior.

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T022 Update `ARCHITECTURE.md` with the final hidden-self-model versus
  reportable-DMN distinction and the removal of the old DMN heuristic stack.
- [ ] T023 Add or update operator docs in `tools/server/README-dev.md` and
  `tools/telegram-bridge/README.md` for prompt revisions, supersession, and DMN
  relay validation.
- [ ] T024 Run targeted validation from
  `/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/quickstart.md`
  and capture any remaining cleanup tasks before merge.
