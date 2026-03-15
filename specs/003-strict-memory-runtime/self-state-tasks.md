# Tasks: Typed Persistent Self-State

**Input**: `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/self-state-spec.md`, `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/self-state-data-model.md`, `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/architecture-guideline.md`

## Phase 1: Setup

- [ ] TSS001 Add self-state documentation artifacts in `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/`
- [ ] TSS002 Extend source registration for self-state implementation files in `/Users/tyleraraujo/vicuna/src/CMakeLists.txt`
- [ ] TSS003 Register a targeted self-state test in `/Users/tyleraraujo/vicuna/tests/CMakeLists.txt`

## Phase 2: Foundational

- [ ] TSS004 Add public self-state enums, structs, and API declarations in `/Users/tyleraraujo/vicuna/include/llama.h`
- [ ] TSS005 Add internal self-state data structures and helpers in `/Users/tyleraraujo/vicuna/src/llama-self-state.h`
- [ ] TSS006 Implement typed register definitions, time-surface math, and event-anchor logic in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS007 Attach self-state ownership to `llama_context` in `/Users/tyleraraujo/vicuna/src/llama-context.h`
- [ ] TSS008 Initialize and expose self-state methods from `llama_context` in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`

## Phase 3: User Story 1 - Inspectable Self-State Surface (Priority: P1)

**Goal**: Expose a predefined typed register bank and datetime surface per context.

**Independent Test**: Create a context and verify that registers and datetime fields are populated and bounded.

- [ ] TSS009 [US1] Implement register inspection APIs in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] TSS010 [US1] Implement datetime inspection APIs in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] TSS011 [US1] Add self-state inspection coverage in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 4: User Story 2 - Deterministic Time and Event Updates (Priority: P1)

**Goal**: Make time and event updates deterministic and replayable.

**Independent Test**: Apply explicit time points and note events, then verify derived deltas and bounded analytic updates.

- [ ] TSS012 [US2] Implement explicit time-point update APIs in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] TSS013 [US2] Implement system-clock refresh convenience API in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] TSS014 [US2] Implement user/tool/emit event marker APIs in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] TSS015 [US2] Add deterministic time and event-anchor tests in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 5: User Story 3 - Explicit Channel and Register Provenance (Priority: P2)

**Goal**: Keep register family and provenance metadata inspectable.

**Independent Test**: Query register metadata before and after updates and verify timestamps, source masks, and updater version.

- [ ] TSS016 [US3] Implement channel-state mutation API in `/Users/tyleraraujo/vicuna/src/llama-context.cpp`
- [ ] TSS017 [US3] Stamp provenance metadata during time and event updates in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS018 [US3] Add provenance and channel-state tests in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 6: User Story 4 - Prewrite/Postwrite Feature Builders (Priority: P1)

**Goal**: Build explicit feature vectors and apply bounded analytic register updates.

**Independent Test**: Build deterministic features for token events, apply prewrite/postwrite updates, and verify bounded register movement.

- [ ] TSS019 [US4] Add public event, params, and feature-vector API declarations in `/Users/tyleraraujo/vicuna/include/llama.h`
- [ ] TSS020 [US4] Extend self-state internals with configurable params and prior-event sketches in `/Users/tyleraraujo/vicuna/src/llama-self-state.h`
- [ ] TSS021 [US4] Implement prewrite/postwrite feature builders in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS022 [US4] Implement bounded analytic register updates in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS023 [US4] Expose feature-builder and update APIs from `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS024 [US4] Add deterministic feature-builder tests in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 7: User Story 5 - Learned Contradiction and Uncertainty Heads Behind Flags (Priority: P2)

**Goal**: Allow contradiction and uncertainty scoring to be upgraded through optional callbacks.

**Independent Test**: Configure learned heads behind flags and verify that their scores affect the prewrite pipeline.

- [ ] TSS025 [US5] Add learned-head configuration flags and callbacks in `/Users/tyleraraujo/vicuna/include/llama.h`
- [ ] TSS026 [US5] Implement flag-gated callback invocation and fallback behavior in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS027 [US5] Add learned-head callback regression tests in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 8: Polish

- [ ] TSS028 [P] Update architecture docs with the implemented self-state slice in `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`
- [ ] TSS029 [P] Run and stabilize self-state regression coverage in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 9: Gap Closure Dependencies

- [ ] TSS030 Add typed identity, goal, commitment, and working-memory surfaces in `/Users/tyleraraujo/vicuna/src/llama-self-state.h`
- [ ] TSS031 Implement retrieval-backed feature inputs for identity, goals, commitments, and working memory in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS032 Route `r_goal_relevance` and `r_self_relevance` through retrieval-backed bounded updates in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS033 Admit postwrite events into a typed working-memory store in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS034 Add public APIs for identity/goals/commitments and working-memory counts in `/Users/tyleraraujo/vicuna/include/llama.h`
- [ ] TSS035 Add regression coverage for retrieval-backed self surfaces in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 10: Remaining Whitepaper Closure

- [x] TSS036 Add sparse `r_reactivation_priority[m_i]` storage and update rules in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS037 Introduce typed tool lifecycle state and async job surfaces in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS038 Add memory-handle retrieval inputs and typed handle APIs in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS039 Add deterministic updater replay hooks in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS040 Add in-tree learned contradiction/uncertainty implementations behind flags in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`

## Phase 11: Remaining Sections 5-8 Closure

- [x] TSS041 Add frozen-LoRA bucket handles and a working-memory-to-cluster bridge in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS042 Add trace serialization and import/export support for self-state traces in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS043 Add an updater-version registry and declarative updater-program surface in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS044 Add counterfactual evaluation over frozen traces for candidate updater variants in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS045 Add richer social and relationship state surfaces in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS048 Replace shared-gain register updates with constrained per-register updater rules in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS049 Keep frozen-bucket handles consolidation-driven instead of mutating them on every admitted message in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS050 Add a dedicated counterfactual replay channel and channel-aware replay/evaluation APIs in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [x] TSS051 Add regression coverage that counterfactual-channel replay does not advance primary-channel social or activation state in `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`

## Phase 12: Future Refinement

- [ ] TSS046 Calibrate or retrain the in-tree probe coefficients against held-out traces in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- [ ] TSS047 Expand scalar social state into optional richer relationship facets without breaking prefix persistence in `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
