# Tasks: Ubiquitous Staged Tool Family/Method/Payload Orchestration

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/119-staged-tool-loop/`  
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/staged-tool-loop.md`

## Phase 1: Setup

**Purpose**: Establish the typed controller surface and documentation targets.

- [ ] T001 Add staged tool-loop state and normalized catalog declarations in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T002 Add staged-loop contract and metadata guidance to `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`

---

## Phase 2: Foundations

**Purpose**: Build reusable metadata normalization and prompt assembly before wiring execution paths.

- [ ] T003 Implement family/method/contract normalization from shared capability metadata in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T004 Implement reusable prompt builders for family selection, method selection, and payload construction in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T005 Implement strict staged JSON parsers and validators in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`

**Checkpoint**: The runtime can build and validate staged selections from normalized metadata without yet replacing the main execution flow.

---

## Phase 3: User Story 1 - Stage Every Tool Decision Through Family, Method, and Payload Selection (Priority: P1)

**Goal**: All tool-backed requests execute through the staged controller and restart from family selection after each tool result.

**Independent Test**: Send one tool-backed request and assert the provider sees family, method, and payload stages in order before execution, then sees family selection again after the observation.

- [ ] T006 [US1] Wire the staged controller into the main provider/tool execution path in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T007 [US1] Preserve additive VAD and heuristic guidance across staged prompts in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T008 [US1] Ensure DeepSeek request assembly preserves `reasoning_content` across staged continuations in `/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp`
- [ ] T009 [US1] Add provider tests for family→method→payload execution and restart-after-tool-result in `/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py`

**Checkpoint**: Foreground execution uses the staged controller end to end.

---

## Phase 4: User Story 2 - Support Back Navigation and Explicit Completion Without Losing ReAct Behavior (Priority: P2)

**Goal**: The provider can back out of mistaken branches and explicitly finish the active loop.

**Independent Test**: Force `back` from method and payload stages and `complete` from method selection, asserting the controller transitions correctly and performs no invalid execution.

- [ ] T010 [US2] Implement `back` transitions for method and payload stages in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T011 [US2] Implement explicit completion handling and handoff to replay/idle flow in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T012 [US2] Add provider tests for navigation and completion in `/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py`

**Checkpoint**: The controller supports recovery and clean termination.

---

## Phase 5: User Story 3 - Make the Staged Controller Extensible and Ubiquitous Across Active and Background Modes (Priority: P3)

**Goal**: Shared catalog metadata and the same controller power foreground and autonomous active loops.

**Independent Test**: Assert that ongoing-task execution uses the staged controller and that docs require family/method/contract metadata for future tools.

- [ ] T013 [US3] Route ongoing-task and other autonomous active flows through the staged controller in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T014 [US3] Preserve replay-admission suppression for staged background-active flows in `/Users/tyleraraujo/vicuna/tools/server/server.cpp`
- [ ] T015 [US3] Add provider tests for staged ongoing-task execution and suppression semantics in `/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py`
- [ ] T016 [US3] Document future metadata requirements and staged-loop behavior in `/Users/tyleraraujo/vicuna/tools/server/README.md`
- [ ] T017 [US3] Document developer-facing staged controller details in `/Users/tyleraraujo/vicuna/tools/server/README-dev.md`

**Checkpoint**: The staged controller is shared, explicit, and documented.

---

## Phase 6: Finish

**Purpose**: Cross-cutting validation and cleanup.

- [ ] T018 Run focused staged-loop provider tests in `/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py`
- [ ] T019 Run the full provider test file in `/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py`
- [ ] T020 Rebuild `llama-server` and record validation commands in the final summary
