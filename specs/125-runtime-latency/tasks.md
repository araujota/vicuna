# Tasks: Runtime Latency Hot Path Reduction

**Input**: Design documents from `/specs/125-runtime-latency/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Provider-mode and bridge automated tests are required because this
change alters runtime hot paths and retained transport timing.

## Phase 1: Setup

- [x] T001 Confirm active feature paths with `/.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` for `/Users/tyleraraujo/vicuna/specs/125-runtime-latency`

## Phase 2: Foundational

- [x] T002 Add a persistent DeepSeek client holder in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.h](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.h) and [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)
- [x] T003 Add explicit Telegram runtime tool and staged-catalog cache structures in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 3: User Story 1 - Provider Requests Reuse Persistent Transport State (Priority: P1)

**Goal**: remove repeated DeepSeek client construction from staged and ordinary
provider requests

**Independent Test**: provider tests exercise repeated requests without
regressing request mapping or errors

### Tests for User Story 1

- [x] T004 [P] [US1] Add provider coverage for shared DeepSeek client reuse behavior in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 1

- [x] T005 [US1] Rework DeepSeek request execution to reuse one configured client in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)

## Phase 4: User Story 2 - Telegram Turns Reuse Server-Owned Tool Metadata (Priority: P1)

**Goal**: avoid repeated runtime-tools subprocess and repeated staged metadata
rebuilds on unchanged bridge-scoped tool payloads

**Independent Test**: repeated bridge-scoped requests hit cache paths and still
build identical staged prompts

### Tests for User Story 2

- [x] T006 [P] [US2] Add cache hit/miss coverage for Telegram runtime tool loading and staged catalog reuse in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 2

- [x] T007 [US2] Cache runtime tool payload loading in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [x] T008 [US2] Cache derived staged family/method/contract metadata in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 5: User Story 3 - Bridge Polling Detects Ready Work Faster (Priority: P2)

**Goal**: reduce retained bridge tail latency after the server has already
finished work

**Independent Test**: bridge tests and direct code assertions verify the new
polling/reconnect timings while Telegram long-poll behavior stays intact

### Tests for User Story 3

- [x] T009 [P] [US3] Update bridge timing assertions in [/Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs](/Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs)

### Implementation for User Story 3

- [x] T010 [US3] Tighten internal outbox polling, self-emit reconnect, and watchdog intervals in [/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs](/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs)

## Phase 6: User Story 4 - Every Provider Turn Uses One Explicit Temperature (Priority: P1)

**Goal**: keep direct, staged, bridge-scoped, and background DeepSeek turns on
one explicit `temperature: 0.2` policy

**Independent Test**: provider tests assert that outbound requests across
surfaces always carry `temperature: 0.2`

### Tests for User Story 4

- [x] T011 [P] [US4] Add assertions that outbound DeepSeek requests always include `temperature: 0.2` in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 4

- [x] T012 [US4] Make the DeepSeek adapter emit `temperature: 0.2` on every outbound provider request in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)

## Phase 7: User Story 5 - Every Provider Turn Uses One Explicit Max Token Cap (Priority: P1)

**Goal**: keep direct, staged, bridge-scoped, and background DeepSeek turns on
one explicit `max_tokens: 256` policy

**Independent Test**: provider tests and bridge request-shape tests assert that
outbound request paths carry `256` instead of the previous `1024` cap

### Tests for User Story 5

- [x] T013 [P] [US5] Update provider and bridge assertions to require `max_tokens: 256` in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py) and [/Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs](/Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs)

### Implementation for User Story 5

- [x] T014 [US5] Reduce the fixed outbound and staged token caps to `256` in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp), [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp), and [/Users/tyleraraujo/vicuna/tools/telegram-bridge/lib.mjs](/Users/tyleraraujo/vicuna/tools/telegram-bridge/lib.mjs)

## Phase 8: Polish & Cross-Cutting Concerns

- [x] T015 [P] Update runtime and bridge docs in [/Users/tyleraraujo/vicuna/tools/server/README.md](/Users/tyleraraujo/vicuna/tools/server/README.md), [/Users/tyleraraujo/vicuna/tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md), and [/Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md](/Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md)
- [ ] T016 Run `DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build /Users/tyleraraujo/vicuna/build-codex-emotive --target llama-server -j8`
- [ ] T017 Run `LLAMA_SERVER_BIN_PATH=/Users/tyleraraujo/vicuna/build-codex-emotive/bin/llama-server /opt/homebrew/bin/pytest /Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py -q`
- [ ] T018 Run `node --test /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs`
