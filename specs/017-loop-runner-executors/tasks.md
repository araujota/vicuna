# Tasks: Planner-Executor Runners For Active And DMN

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/017-loop-runner-executors/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, implementation-approach.md

## Phase 1: Setup

- [x] T001 Review the existing loop scaffold and host integration in
  `/Users/tyleraraujo/vicuna/include/llama.h`,
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`, and
  `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`.
- [x] T002 Review the current loop-related specs in
  `/Users/tyleraraujo/vicuna/specs/016-react-loop-scaffolding/`,
  `/Users/tyleraraujo/vicuna/specs/004-active-engagement-loop/`, and
  `/Users/tyleraraujo/vicuna/specs/005-pressure-driven-dmn/`.

## Phase 2: External Research

- [x] T003 Use GitHub research to inspect current runner/controller patterns in
  LangGraph, Letta, smolagents, and OpenHands.
- [x] T004 Use web research to inspect primary references for ReAct and
  planning/execution separation.

## Phase 3: User Story 1 - Event-Driven Foreground Runner (Priority: P1)

- [x] T005 [US1] Document the foreground runner architecture in
  `/Users/tyleraraujo/vicuna/specs/017-loop-runner-executors/implementation-approach.md`.
- [x] T006 [US1] Define runner command and status entities in
  `/Users/tyleraraujo/vicuna/specs/017-loop-runner-executors/data-model.md`.
- [x] T007 [US1] Add public runner command/status APIs in
  `/Users/tyleraraujo/vicuna/include/llama.h`.
- [x] T008 [US1] Implement active runner persistence, pending commands, and
  observation resumption in `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`.

## Phase 4: User Story 2 - Maintenance-Oriented DMN Runner (Priority: P1)

- [x] T009 [US2] Document bounded DMN continuation in
  `/Users/tyleraraujo/vicuna/specs/017-loop-runner-executors/implementation-approach.md`.
- [x] T010 [US2] Implement DMN bounded continuation and command scheduling in
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`.

## Phase 5: User Story 3 - Host-Visible Commands And Status (Priority: P2)

- [x] T011 [US3] Document host polling and acknowledgment semantics in
  `/Users/tyleraraujo/vicuna/specs/017-loop-runner-executors/implementation-approach.md`.
- [x] T012 [US3] Add host-facing queue accessors in
  `/Users/tyleraraujo/vicuna/src/llama-context.h`,
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.h`, and
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`.
- [x] T013 [US3] Update `llama-server` integration in
  `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`.

## Phase 6: Verification

- [x] T014 Add regression coverage in
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.
- [x] T015 Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md`.
- [x] T016 Build affected targets and run targeted `ctest`.
