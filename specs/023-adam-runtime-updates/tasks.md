# Tasks: Expand Adam-Based Optimization For Self-State-Driven Runtime Updates

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/023-adam-runtime-updates/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`

**Tests**: Targeted automated tests are REQUIRED because this changes runtime
learning behavior and online control state.

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Review the existing runtime LoRA writer, temporal bias controller,
  and counterfactual ladder in
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` and
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp` against the
  feature research.

## Phase 2: Foundational (Blocking Prerequisites)

- [ ] T002 Add typed public observability fields for optimizer-backed runtime
  writes and temporal bias updates in `/Users/tyleraraujo/vicuna/include/llama.h`.
- [ ] T003 Implement bounded Adam state ownership for runtime LoRA tensor pairs
  in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T004 Implement bounded Adam state ownership for temporal reward and
  dampening biases in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.

**Checkpoint**: The runtime has explicit typed optimizer state for all selected
candidate paths.

## Phase 3: User Story 1 - Runtime LoRA Writes Use Adam (Priority: P1) 🎯 MVP

**Goal**: Move self-state-driven runtime LoRA tensor mutation onto Adam.

**Independent Test**: Trigger Active LoRA and functional-family writes and
verify optimizer-backed steps advance.

### Tests for User Story 1

- [ ] T005 [P] [US1] Extend `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`
  to verify Active LoRA stats expose optimizer steps and non-zero update norms
  after runtime writes.
- [ ] T006 [P] [US1] Extend `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`
  to verify functional-family updates still settle through the optimizer-backed
  runtime writer.

### Implementation for User Story 1

- [ ] T007 [US1] Replace direct additive `A`/`B` tensor writes with Adam-backed
  updates in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T008 [US1] Preserve explicit weight decay and gain normalization in
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T009 [US1] Surface optimizer-backed runtime write stats in
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` and
  `/Users/tyleraraujo/vicuna/include/llama.h`.

## Phase 4: User Story 2 - Temporal Write-Bias Control Uses Adam (Priority: P2)

**Goal**: Move temporal reward and dampening bias adaptation onto Adam while
keeping bounded write-scale semantics.

**Independent Test**: Trigger DMN temporal self-improvement and verify temporal
  bias optimizer state advances.

### Tests for User Story 2

- [ ] T010 [P] [US2] Extend `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`
  to verify temporal bias state exposes Adam advancement and bounded write
  scale.
- [ ] T011 [P] [US2] Extend `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`
  to verify default temporal bias optimizer state starts empty.

### Implementation for User Story 2

- [ ] T012 [US2] Replace heuristic temporal bias increments with scalar Adam
  updates in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.
- [ ] T013 [US2] Surface temporal bias optimizer observability in
  `/Users/tyleraraujo/vicuna/include/llama.h` and
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`.

## Phase 5: User Story 3 - Discrete Counterfactual Ranking Remains Explicit (Priority: P2)

**Goal**: Document and test the explicit non-adoption decision for the
counterfactual ladder.

**Independent Test**: Inspect docs and confirm the counterfactual ladder still
uses explicit ranking logic without optimizer state.

### Tests for User Story 3

- [ ] T014 [P] [US3] Add or extend assertions in
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp` that the
  counterfactual ladder remains explicit ranking behavior.

### Implementation for User Story 3

- [ ] T015 [US3] Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` to document
  where Adam is used and where it is intentionally not used.
- [ ] T016 [US3] Update `/Users/tyleraraujo/vicuna/Vicuña_WP.md` to reflect the
  same optimizer boundary.

## Phase 6: Validation

- [ ] T017 Run targeted validation from
  `/Users/tyleraraujo/vicuna/specs/023-adam-runtime-updates/quickstart.md`
  and fix any regressions before closeout.

## Phase 7: PR Code-Scanning Cleanup

- [x] T018 Fix the touched Python wildcard import in
  `/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_basic.py`.
- [x] T019 Replace raw float equality in request LoRA stack matching and widen
  embedding-copy arithmetic in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`.
- [x] T020 Replace the by-value bash-tool configure C API with a pointer-based
  entry point in `/Users/tyleraraujo/vicuna/include/llama.h`,
  `/Users/tyleraraujo/vicuna/src/llama-bash-tool.cpp`,
  `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`, and
  `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`.
- [x] T021 Remove low-signal CodeQL findings from touched native files by
  simplifying the one-case switch in
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` and removing the
  commented-out formula in `/Users/tyleraraujo/vicuna/src/llama-sampler.cpp`.
