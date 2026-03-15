# Tasks: Strict Live Memory Runtime

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/memory-runtime-api.md`, `quickstart.md`

## Phase 1: Setup

- [ ] T001 Create the strict-serving feature artifacts in `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/`
- [ ] T002 Extend test registration as needed in `/Users/tyleraraujo/vicuna/tests/CMakeLists.txt`

## Phase 2: Foundational

- [ ] T003 Introduce explicit ordered adapter-stack types in `/Users/tyleraraujo/vicuna/src/llama-adapter.h`
- [ ] T004 Implement ordered adapter-stack helpers in `/Users/tyleraraujo/vicuna/src/llama-adapter.cpp`
- [ ] T005 Refactor adapter ownership into request-managed and runtime-managed sets in `/Users/tyleraraujo/vicuna/src/llama-context.h`
- [ ] T006 Rebuild the effective serving stack in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] T007 Pass the explicit serving stack through graph parameters in `/Users/tyleraraujo/vicuna/src/llama-graph.h`
- [ ] T008 Update graph-side LoRA application to iterate the explicit serving stack in `/Users/tyleraraujo/vicuna/src/llama-graph.cpp`

## Phase 3: User Story 1 - Persistent Memory Stack During Live Inference (Priority: P1)

**Goal**: Preserve Active and Past memory layers in the live serving stack across request adapter changes.

**Independent Test**: Enable Active/Past memory, change request adapters, and verify the effective serving stack still includes memory layers during decode.

- [ ] T009 [US1] Register runtime memory adapters with explicit layer roles in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T010 [US1] Update runtime attachment APIs for explicit memory-layer registration in `/Users/tyleraraujo/vicuna/src/llama-context.h`
- [ ] T011 [US1] Implement request-adapter updates without overwriting runtime memory layers in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] T012 [US1] Add regression coverage for memory-stack preservation in `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`

## Phase 4: User Story 2 - Strict KV Coherence After Active Writes (Priority: P1)

**Goal**: Replay the retained suffix after accepted Active LoRA writes caused by generation-time context shift.

**Independent Test**: Force a context shift that changes Active LoRA, verify replay runs before the next generated token, and verify replay is skipped when no weight change occurred.

- [ ] T013 [US2] Extend Active LoRA ingest results with explicit weight-change reporting in `/Users/tyleraraujo/vicuna/src/llama-active-lora.h`
- [ ] T014 [US2] Implement explicit ingest outcome reporting in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- [ ] T015 [US2] Add slot-local strict replay state to `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T016 [US2] Schedule strict replay from the generation-time context-shift path in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T017 [US2] Execute retained-suffix replay before resumed sampling in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T018 [US2] Invalidate stale checkpoints and reset sampler state at replay boundaries in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T019 [US2] Add regression coverage for replay-run and replay-skip behavior in `/Users/tyleraraujo/vicuna/tests/test-past-lora.cpp`

## Phase 5: User Story 3 - Explicit, Inspectable Adapter Composition Policy (Priority: P2)

**Goal**: Make composition order and replay lifecycle auditable.

**Independent Test**: Inspect logs or stats during serving and verify deterministic layer ordering plus replay lifecycle events.

- [ ] T020 [US3] Add stack-rebuild logging and layer-role traces in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] T021 [US3] Add replay lifecycle logs and skip reasons in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T022 [US3] Extend documentation in `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`
- [ ] T023 [US3] Update the working paper in `/Users/tyleraraujo/vicuna/Vicuña_WP.md`

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T024 [P] Run and stabilize targeted Active/Past LoRA tests in `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp`
- [ ] T025 [P] Run and stabilize strict-serving regressions in `/Users/tyleraraujo/vicuna/tests/test-past-lora.cpp`

## Dependencies

- Foundational tasks must complete before story work.
- User Story 1 must land before strict replay because replay depends on the corrected serving stack.
- User Story 2 must land before observability polish because replay lifecycle logging depends on replay state.
- Documentation updates can finalize after behavior is stable.

## Parallel Opportunities

- T003 and T004 can run in parallel.
- T007 and T008 can run in parallel after the adapter-stack shape is finalized.
- T022 and T023 can run in parallel once implementation semantics stabilize.

## Implementation Strategy

- Deliver the serving-stack fix first so memory actually participates in inference.
- Add strict replay next so retained KV stays coherent after Active updates.
- Finish with logs, tests, and docs once behavior is stable.
