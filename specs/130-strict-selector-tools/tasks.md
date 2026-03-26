# Tasks: Strict Selector Tools And Waterfall VAD

**Input**: Design documents from `/specs/130-strict-selector-tools/`
**Prerequisites**: plan.md, spec.md, research.md

**Tests**: Provider tests are required for strict selector tool payloads,
per-stage retries, and staged VAD propagation.

## Phase 1: Setup

- [x] T001 Review the current staged selector loop, DeepSeek provider request assembly, and additive VAD eligibility code in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp) and [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)

## Phase 2: Foundational

- [x] T002 Add strict selector tool schema builders, stable selector prompt prefixes, and cache plumbing in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [x] T003 Add provider support for staged selector strict-tool beta routing in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)
- [x] T004 Add staged-turn VAD eligibility tracking that does not rely solely on classic tool-continuation spans in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 3: User Story 1 - Selector turns terminate with structured outputs (Priority: P1)

**Goal**: family, method, and payload stages use strict selector tools instead
of freeform JSON mode

**Independent Test**: provider tests inspect outbound staged requests and assert
strict tool schemas plus parsed tool-call outputs

### Tests for User Story 1

- [x] T005 [P] [US1] Add provider tests for strict family/method/payload selector tools in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 1

- [x] T006 [US1] Replace staged family/method/payload JSON parsing with strict selector tool-call parsing in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [x] T007 [US1] Route staged selector calls through DeepSeek beta strict-tool mode in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)

## Phase 4: User Story 2 - Waterfall control remains explicit and inspectable (Priority: P1)

**Goal**: preserve explicit family -> method -> payload orchestration with
stable selector prefixes and traceable stage metadata

**Independent Test**: provider tests and request traces confirm separate stage
events, sentinels, and stable prompt assembly

### Tests for User Story 2

- [x] T008 [P] [US2] Extend staged request-trace assertions for strict selector stage metadata in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 2

- [x] T009 [US2] Keep explicit stage prompts and add stable-prefix prompt assembly/caching in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 5: User Story 3 - Renewed VAD guidance appears on each following stage (Priority: P1)

**Goal**: every staged step after prior reasoning receives renewed additive VAD
guidance and logs it

**Independent Test**: provider tests show `vad_injected=true` on following
staged turns and expose the exact injected sentence in traces

### Tests for User Story 3

- [x] T010 [P] [US3] Add per-stage VAD propagation assertions in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 3

- [x] T011 [US3] Update additive guidance assembly to inject renewed VAD on staged follow-up turns in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 6: Polish & Validation

- [x] T012 Update staged-loop documentation for DeepSeek beta strict-tool selector turns and staged VAD propagation in [/Users/tyleraraujo/vicuna/tools/server/README.md](/Users/tyleraraujo/vicuna/tools/server/README.md) and [/Users/tyleraraujo/vicuna/tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md)
- [x] T013 Run `DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build /Users/tyleraraujo/vicuna/build-codex-emotive --target llama-server -j8`
- [x] T014 Run `LLAMA_SERVER_BIN_PATH=/Users/tyleraraujo/vicuna/build-codex-emotive/bin/llama-server /opt/homebrew/bin/pytest /Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py -q`
