# Tasks: Self-Model Extension Registry

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/025-self-model-extension-registry/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`,
`implementation-approach.md`

## Phase 1: Setup And Research

- [ ] T001 Review `/Users/tyleraraujo/vicuna/specs/015-self-model-expansion/`,
  `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`, and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md`.
- [ ] T002 Inspect the current self-state, hard-memory, tool, and functional
  gating paths in `/Users/tyleraraujo/vicuna/include/llama.h`,
  `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`,
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`, and
  `/Users/tyleraraujo/vicuna/src/llama-hard-memory.cpp`.
- [ ] T003 Record GitHub and web research in
  `/Users/tyleraraujo/vicuna/specs/025-self-model-extension-registry/research.md`.

## Phase 2: Contracts And Design

- [ ] T004 Define extension entities, flags, bounded capacities, and summaries
  in `/Users/tyleraraujo/vicuna/specs/025-self-model-extension-registry/data-model.md`.
- [ ] T005 Write the runtime architecture and control-path integration plan in
  `/Users/tyleraraujo/vicuna/specs/025-self-model-extension-registry/implementation-approach.md`.
- [ ] T006 Write tool-integration and validation guidance in
  `/Users/tyleraraujo/vicuna/specs/025-self-model-extension-registry/quickstart.md`.

## Phase 3: Public API And Self-State Runtime

- [ ] T007 Add public enums, structs, constants, and API declarations in
  `/Users/tyleraraujo/vicuna/include/llama.h`.
- [ ] T008 Extend `/Users/tyleraraujo/vicuna/src/llama-self-state.h` and
  `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp` with bounded extension
  storage, CRUD, summaries, and extension-aware model updates.
- [ ] T009 Extend `/Users/tyleraraujo/vicuna/src/llama-context.h` and
  `/Users/tyleraraujo/vicuna/src/llama-context.cpp` with wrapper methods for
  the new APIs.

## Phase 4: Hard-Memory Promotion And Control Integration

- [ ] T010 Extend `/Users/tyleraraujo/vicuna/src/llama-hard-memory.cpp` and
  `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp` so hard-memory query
  results can be counterfactually promoted into extension slots.
- [ ] T011 Extend `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` so the
  functional gating MLP consumes a fixed extension summary tail while preserving
  bounded gain behavior.

## Phase 5: Tests

- [ ] T012 Add self-state regression coverage in
  `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp` for extension CRUD,
  hard-memory promotion, and allostatic-flag behavior.
- [ ] T013 Add functional-gating regression coverage in
  `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp` or
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp` for extension-aware
  summaries and stable bounded gains.

## Phase 6: Documentation

- [ ] T014 Update `/Users/tyleraraujo/vicuna/README.md` with operator guidance
  on accurate tool-authored self-model additions.
- [ ] T015 Update `/Users/tyleraraujo/vicuna/tools/server/README-dev.md` with
  extension-tool integration guidance.
- [ ] T016 Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md` so the authored-core plus extension
  registry model is documented.

## Phase 7: Validation

- [ ] T017 Run targeted builds/tests for `test-self-state`, `test-active-lora`,
  and `test-cognitive-loop`.
- [ ] T018 Verify the docs and spec artifacts remain aligned with the delivered
  runtime behavior.
