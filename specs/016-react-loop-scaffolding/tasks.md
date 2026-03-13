# Tasks: Bounded Tool-Loop Scaffolding For Active And DMN

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/016-react-loop-scaffolding/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, implementation-approach.md

**Tests**: Targeted cognitive-loop regression coverage and relevant build/test
runs.

## Phase 1: Setup

- [x] T001 Review current loop/runtime artifacts in
  `/Users/tyleraraujo/vicuna/specs/004-active-engagement-loop/`,
  `/Users/tyleraraujo/vicuna/specs/005-pressure-driven-dmn/`,
  `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`, and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md`.
- [x] T002 Inspect the existing active-loop, DMN, host-state, remediation, and
  trace surfaces in `/Users/tyleraraujo/vicuna/include/llama.h`,
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`, and
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.

## Phase 2: External Research

- [x] T003 Use GitHub codebase research to inspect state-machine or graph-based
  agent loop implementations in LangGraph, Letta, and smolagents.
- [x] T004 Use web research to confirm primary references for ReAct and related
  agent-loop architectures.

## Phase 3: User Story 1 - Shared Long-Running Loop Substrate (Priority: P1)

**Goal**: Define and implement bounded shared loop scaffolding.

- [x] T005 [US1] Document the chosen architecture and rejected alternative in
  `/Users/tyleraraujo/vicuna/specs/016-react-loop-scaffolding/implementation-approach.md`.
- [x] T006 [US1] Define the new loop entities and enum surfaces in
  `/Users/tyleraraujo/vicuna/specs/016-react-loop-scaffolding/data-model.md`.
- [x] T007 [US1] Add public loop/tool scaffolding structs and enums in
  `/Users/tyleraraujo/vicuna/include/llama.h`.
- [x] T008 [US1] Extend `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.h`
  and `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp` with persistent
  tool-registry and episode-state handling.

## Phase 4: User Story 2 - Distinct Active And DMN Policies (Priority: P1)

**Goal**: Keep one substrate but two loop policies.

- [x] T009 [US2] Write active-vs-DMN policy guidance in
  `/Users/tyleraraujo/vicuna/specs/016-react-loop-scaffolding/implementation-approach.md`.
- [x] T010 [US2] Implement foreground episode derivation and tool-plan
  scaffolding around active-loop winner selection in
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`.
- [x] T011 [US2] Implement background episode derivation and tool-plan
  scaffolding around DMN winner selection in
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`.

## Phase 5: User Story 3 - Parity-Preserving Tool Insertion Points (Priority: P2)

**Goal**: Preserve current winner behavior while exposing future-ready tool
insertion seams.

- [x] T012 [US3] Capture parity constraints and registry design in
  `/Users/tyleraraujo/vicuna/specs/016-react-loop-scaffolding/research.md`.
- [x] T013 [US3] Add regression coverage in
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.
- [x] T014 [US3] Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md`.

## Phase 6: Verification

- [x] T015 Build the affected targets.
- [x] T016 Run targeted `ctest` coverage for cognitive-loop behavior.
