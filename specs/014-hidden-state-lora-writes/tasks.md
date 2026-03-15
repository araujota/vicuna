# Tasks: Hidden-State-Derived Active LoRA Embeddings And Feature-Derived Write Rules

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, implementation-approach.md

**Tests**: Future implementation must ship with targeted automated tests for
embedder parity, writer parity, serving-order parity, temporal-bucket parity,
and counterfactual-ablation parity.

## Phase 1: Setup

- [x] T001 Review `/Users/tyleraraujo/vicuna/specs/001-active-lora-memory/`,
  `/Users/tyleraraujo/vicuna/specs/002-past-lora-condensation/`,
  `/Users/tyleraraujo/vicuna/specs/008-lora-remediation-engine/`,
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md`, and
  `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`.

## Phase 2: Foundational

- [x] T002 Inspect the current Active LoRA embedder and write path in
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` and
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.h`.
- [x] T003 Inspect hidden-state and embedding extraction surfaces in
  `/Users/tyleraraujo/vicuna/include/llama.h` and
  `/Users/tyleraraujo/vicuna/src/llama-context.cpp`.
- [x] T004 Inspect serving-stack, past-bucket, remediation, and ablation parity
  constraints in `/Users/tyleraraujo/vicuna/tests/test-serving-lora-stack.cpp`,
  `/Users/tyleraraujo/vicuna/tests/test-past-lora.cpp`, and
  `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`.

## Phase 3: User Story 1 - Hidden-State Embedder Architecture (Priority: P1)

**Goal**: Define the default hidden-state-derived embedder path and its
fallbacks.

**Independent Test**: A maintainer can read the design and implement the
embedder without guessing where hidden states come from or how fallback modes
behave.

- [x] T005 [US1] Summarize local embedder and hidden-state extraction evidence
  in `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/research.md`.
- [x] T006 [US1] Define the hidden-state embedder params, traces, and runtime
  entities in `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/data-model.md`.
- [x] T007 [US1] Write the hidden-state embedder implementation approach in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/implementation-approach.md`.

## Phase 4: User Story 2 - Feature-Derived Write Directions (Priority: P1)

**Goal**: Define the new write-feature pipeline and write-direction rule while
preserving layering and remediation parity.

**Independent Test**: A maintainer can trace how content and self-state become
bounded direction/gain updates without breaking current bucket or remediation
semantics.

- [x] T008 [US2] Capture current writer bottlenecks and parity constraints in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/research.md`.
- [x] T009 [US2] Define write-feature and rank-allocation entities in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/data-model.md`.
- [x] T010 [US2] Write the feature-derived write-direction architecture,
  remediation parity rules, and temporal parity rules in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/implementation-approach.md`.

## Phase 5: User Story 3 - Rank, Magnitude, And Quantization Policy (Priority: P2)

**Goal**: Translate AdaLoRA, DoRA, and QLoRA ideas into Vicuña-specific runtime
choices.

**Independent Test**: A maintainer can see which external PEFT ideas are being
adopted, adapted, or rejected and why.

- [x] T011 [US3] Record the external AdaLoRA, DoRA, QLoRA, LongMem, and
  MemoryLLM research in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/research.md`.
- [x] T012 [US3] Write the adopt/adapt/reject decisions and rollout strategy in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/implementation-approach.md`.

## Phase 6: Polish

- [x] T013 Verify that every major claim in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/implementation-approach.md`
  is backed by local references or primary-source external citations.
- [x] T014 Review the design artifacts for explicit preservation of
  serving-stack order, temporal bucket parity, remediation parity, and
  `LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION` parity.
- [x] T015 Outline future implementation tasks and validation commands in
  `/Users/tyleraraujo/vicuna/specs/014-hidden-state-lora-writes/tasks.md`
  or a follow-on implementation plan update if coding is requested.

## Phase 7: Implementation Follow-Through

- [x] T016 [US1] Implement a hidden-state-derived default Active LoRA embedder
  in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` using an auxiliary
  context of the same base model family and current embedding APIs.
- [x] T017 [US2] Replace the token-modulo writer with a feature-derived writer
  in `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` that combines
  hidden-state content with typed self-state features.
- [x] T018 [US2] Thread event and postwrite-feature context through
  `/Users/tyleraraujo/vicuna/src/llama-context.h`,
  `/Users/tyleraraujo/vicuna/src/llama-context.cpp`, and
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp` so ingest and
  remediation share the same writer core while preserving parity.
- [x] T019 [P] Update regression coverage in
  `/Users/tyleraraujo/vicuna/tests/test-active-lora.cpp` and rerun the
  parity-sensitive Active LoRA, past-LoRA, serving-stack, and cognitive-loop
  tests.
- [x] T020 [P] Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md` to reflect the hidden-state default
  embedder and feature-derived writer.
