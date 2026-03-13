# Tasks: Functionality Audit Of The Exotic-Intelligence Runtime

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/`
**Prerequisites**: plan.md, spec.md

**Tests**: Validation is evidence-based for this audit. The deliverable must be
backed by concrete repository references, existing tests, GitHub history, and
primary-source external citations.

## Phase 1: Setup

- [x] T001 Review `/Users/tyleraraujo/vicuna/Vicuña_WP.md`,
  `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`, and existing specs relevant to
  memory, self-state, DMN, governance, and hard memory.

## Phase 2: Foundational

- [x] T002 Inspect the current local implementation across
  `/Users/tyleraraujo/vicuna/include/llama.h`,
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`,
  `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`,
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`,
  `/Users/tyleraraujo/vicuna/src/llama-hard-memory.cpp`, and relevant context
  plumbing.
- [x] T003 Inspect server/runtime integration across
  `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`,
  `/Users/tyleraraujo/vicuna/tools/server/server-common.cpp`, and adjacent
  serving code.
- [x] T004 Inspect existing regression coverage in
  `/Users/tyleraraujo/vicuna/tests/test-self-state.cpp`,
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`,
  `/Users/tyleraraujo/vicuna/tests/test-serving-lora-stack.cpp`, and related
  tests.

## Phase 3: User Story 1 - Establish The Current Functional Surface (Priority: P1)

**Goal**: Determine what the current application actually implements and how it
is wired.

**Independent Test**: Confirm the report can trace every current-functionality
claim to code and tests.

- [x] T005 [US1] Summarize the local functionality inventory and evidence in
  `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/research.md`.
- [x] T006 [US1] Write the current-functionality and expected-behavior sections
  in `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/audit-report.md`.

## Phase 4: User Story 2 - Measure Whitepaper Parity (Priority: P1)

**Goal**: Compare the implemented runtime against the whitepaper’s intended
agent architecture.

**Independent Test**: Confirm each whitepaper pillar has an expected-behavior
description and a parity judgment.

- [x] T007 [US2] Map whitepaper pillars to code and observed behavior in
  `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/research.md`.
- [x] T008 [US2] Write the whitepaper-parity analysis in
  `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/audit-report.md`.

## Phase 5: User Story 3 - Compare Against Current State Of The Art (Priority: P2)

**Goal**: Position Vicuña relative to current public best practice.

**Independent Test**: Confirm each major area includes external citations and a
  state-of-the-art comparison.

- [x] T009 [US3] Research current public primary sources for memory, persistent
  agent state, autonomous loops, tool use, and self-improvement, and record the
  findings in `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/research.md`.
- [x] T010 [US3] Write the state-of-the-art comparison and gap analysis in
  `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/audit-report.md`.

## Phase 6: Polish

- [x] T011 Verify that every major report claim is backed by local file
  references or external citations.
- [x] T012 Review `/Users/tyleraraujo/vicuna/specs/013-functionality-audit/audit-report.md`
  for clear separation between implemented, partial, and speculative behavior.
