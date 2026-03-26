# Tasks: Host Secrets And Verbatim Stage Traces

**Input**: Design documents from `/specs/129-host-secrets-reasoning/`
**Prerequisites**: plan.md, spec.md, research.md

**Tests**: Provider tests are required for verbatim reasoning retention and
runtime-guidance trace visibility. Startup-script and doc updates are required
for durable host secrets.

## Phase 1: Setup

- [ ] T001 Review the active implementation files for env loading, OpenClaw secret paths, additive VAD injection, and request-trace emission in [/Users/tyleraraujo/vicuna/tools/ops/runtime-env.sh](/Users/tyleraraujo/vicuna/tools/ops/runtime-env.sh), [/Users/tyleraraujo/vicuna/tools/ops/run-vicuna-runtime.sh](/Users/tyleraraujo/vicuna/tools/ops/run-vicuna-runtime.sh), [/Users/tyleraraujo/vicuna/tools/ops/run-telegram-bridge.sh](/Users/tyleraraujo/vicuna/tools/ops/run-telegram-bridge.sh), [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp), and [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)

## Phase 2: Foundational

- [ ] T002 Add durable host env loading and OpenClaw secret/catalog path exports in [/Users/tyleraraujo/vicuna/tools/ops/runtime-env.sh](/Users/tyleraraujo/vicuna/tools/ops/runtime-env.sh)
- [ ] T003 Add a rebuild-safe OpenClaw startup sync path for media, Tavily, and memory settings in [/Users/tyleraraujo/vicuna/tools/ops/run-vicuna-runtime.sh](/Users/tyleraraujo/vicuna/tools/ops/run-vicuna-runtime.sh) and [/Users/tyleraraujo/vicuna/tools/ops/run-telegram-bridge.sh](/Users/tyleraraujo/vicuna/tools/ops/run-telegram-bridge.sh)
- [ ] T004 Update the system install env template to include the durable OpenClaw and memory settings in [/Users/tyleraraujo/vicuna/tools/ops/install-vicuna-system-service.sh](/Users/tyleraraujo/vicuna/tools/ops/install-vicuna-system-service.sh)

## Phase 3: User Story 1 - Host secrets survive rebuilds (Priority: P1)

**Goal**: runtime and bridge always pick up the same host-level secrets, and
OpenClaw tools keep their credentials outside the checkout

**Independent Test**: startup behavior and docs show one durable env file plus
one durable secrets/catalog path

### Implementation for User Story 1

- [ ] T005 [US1] Document the durable host env file and OpenClaw secret/catalog paths in [/Users/tyleraraujo/vicuna/tools/server/README.md](/Users/tyleraraujo/vicuna/tools/server/README.md), [/Users/tyleraraujo/vicuna/tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md), and [/Users/tyleraraujo/vicuna/tools/openclaw-harness/README.md](/Users/tyleraraujo/vicuna/tools/openclaw-harness/README.md)

## Phase 4: User Story 2 - Verbatim staged reasoning in request traces (Priority: P1)

**Goal**: request traces preserve exact provider reasoning and content for each
stage

**Independent Test**: provider tests recover exact strings from staged
`provider_request_finished` events

### Tests for User Story 2

- [ ] T006 [P] [US2] Add provider-trace assertions for verbatim `reasoning_content` and `content` retention in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 2

- [ ] T007 [US2] Enrich provider-finished request-trace events with exact reasoning/content text in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)

## Phase 5: User Story 3 - Inspectable VAD guidance injection (Priority: P1)

**Goal**: traces prove whether VAD/heuristic guidance was injected or skipped

**Independent Test**: provider tests can inspect explicit runtime-guidance
events and see the exact VAD text or skip reason

### Tests for User Story 3

- [ ] T008 [P] [US3] Add request-trace assertions for additive guidance events in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 3

- [ ] T009 [US3] Add runtime-guidance trace events and skip reasons around additive VAD/heuristic injection in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 6: Polish & Validation

- [ ] T010 Run `DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build /Users/tyleraraujo/vicuna/build-codex-emotive --target llama-server -j8`
- [ ] T011 Run `LLAMA_SERVER_BIN_PATH=/Users/tyleraraujo/vicuna/build-codex-emotive/bin/llama-server /opt/homebrew/bin/pytest /Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py -q`
- [ ] T012 Deploy the updated runtime/bridge startup scripts to the host, populate the durable env file and durable OpenClaw secrets, rebuild the required surfaces, and verify the live request trace exposes verbatim stage reasoning
