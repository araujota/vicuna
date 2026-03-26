# Tasks: Request-Scoped Structured Trace Logging

**Input**: Design documents from `/specs/126-request-trace-logging/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Provider-mode and bridge automated tests are required because this
change alters runtime observability and retained bridge failure reporting.

## Phase 1: Setup

- [X] T001 Confirm active feature paths with `/.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` for `/Users/tyleraraujo/vicuna/specs/126-request-trace-logging`

## Phase 2: Foundational

- [X] T002 Add bounded runtime request-trace registry/state helpers in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [X] T003 Add provider trace hook types in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.h](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.h) and [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)

## Phase 3: User Story 1 - Follow one request end to end (Priority: P1)

**Goal**: emit correlated structured events through bridge ingress, staged
selection, provider traffic, runtime tool execution, and Telegram queueing

**Independent Test**: one bridge-scoped tool turn exposes a complete correlated
event chain

### Tests for User Story 1

- [X] T004 [P] [US1] Add provider-mode assertions for structured request-trace events in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)
- [ ] T005 [P] [US1] Add bridge assertions for correlation-id propagation and structured deferred-turn logging in [/Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs](/Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs)

### Implementation for User Story 1

- [X] T006 [US1] Add structured runtime event emission to top-level request handling, staged tool loops, runtime tool execution, and Telegram delivery in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [X] T007 [US1] Add structured provider request start/finish/error emission in [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp)
- [X] T008 [US1] Add bridge request-id generation, propagation, and deferred-turn structured logs in [/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs](/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs)

## Phase 4: User Story 2 - Inspect recent failures after the fact (Priority: P1)

**Goal**: expose a bounded runtime request-trace inspection endpoint

**Independent Test**: request-trace events can be queried by request id over
HTTP after request completion

### Tests for User Story 2

- [X] T009 [P] [US2] Add endpoint coverage for bounded request-trace inspection in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 2

- [X] T010 [US2] Add `/v1/debug/request-traces` and related `/health` summary fields in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 5: Polish & Cross-Cutting Concerns

- [X] T011 [P] Update runtime and bridge docs in [/Users/tyleraraujo/vicuna/tools/server/README.md](/Users/tyleraraujo/vicuna/tools/server/README.md), [/Users/tyleraraujo/vicuna/tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md), and [/Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md](/Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md)
- [X] T012 Run `DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build /Users/tyleraraujo/vicuna/build-codex-emotive --target llama-server -j8`
- [X] T013 Run `LLAMA_SERVER_BIN_PATH=/Users/tyleraraujo/vicuna/build-codex-emotive/bin/llama-server /opt/homebrew/bin/pytest /Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py -q`
- [X] T014 Run `node --test /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs`
