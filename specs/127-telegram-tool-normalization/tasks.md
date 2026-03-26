# Tasks: Telegram Tool Normalization

**Input**: Design documents from `/specs/127-telegram-tool-normalization/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Provider-mode tests are required because this change alters staged
Telegram family/method/payload behavior and internal outbox delivery.

## Phase 1: Setup

- [ ] T001 Confirm active feature paths with `/.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` for `/Users/tyleraraujo/vicuna/specs/127-telegram-tool-normalization`

## Phase 2: Foundational

- [ ] T002 Define explicit Telegram staged methods and typed payload schemas in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [ ] T003 Remove the bridge-scoped Telegram system prompt from [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 3: User Story 1 - Normalize Telegram family selection (Priority: P1)

**Goal**: expose Telegram as one ordinary family at family selection time with
no bridge-only prompt shaping

**Independent Test**: a bridge-scoped request shows Telegram only as a normal
family description in the staged family prompt

### Tests for User Story 1

- [ ] T004 [P] [US1] Update bridge-scoped family-selection assertions in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 1

- [ ] T005 [US1] Remove `build_telegram_bridge_system_prompt` usage and keep bridge request assembly uniform in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 4: User Story 2 - Add explicit Telegram methods and payload contracts (Priority: P1)

**Goal**: replace `telegram_relay` with explicit Telegram methods and
method-specific payload contracts

**Independent Test**: a bridge-scoped request can queue at least one simple
text Telegram delivery and one rich Telegram delivery through explicit Telegram
methods

### Tests for User Story 2

- [ ] T006 [P] [US2] Update Telegram delivery tool-loop coverage for normalized methods in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)
- [ ] T007 [P] [US2] Add provider assertions for method-stage narrowing to the normalized Telegram methods in [/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py](/Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py)

### Implementation for User Story 2

- [ ] T008 [US2] Replace `build_telegram_bridge_relay_tool` with explicit Telegram method tool definitions in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [ ] T009 [US2] Replace `parse_telegram_relay_tool_call` with explicit Telegram method parsing/translation in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)
- [ ] T010 [US2] Update bridge-scoped Telegram execution logic to recognize the normalized Telegram methods in [/Users/tyleraraujo/vicuna/tools/server/server.cpp](/Users/tyleraraujo/vicuna/tools/server/server.cpp)

## Phase 5: Polish & Cross-Cutting Concerns

- [ ] T011 [P] Update Telegram family/method/payload docs in [/Users/tyleraraujo/vicuna/tools/server/README.md](/Users/tyleraraujo/vicuna/tools/server/README.md), [/Users/tyleraraujo/vicuna/tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md), and [/Users/tyleraraujo/vicuna/ARCHITECTURE.md](/Users/tyleraraujo/vicuna/ARCHITECTURE.md)
- [ ] T012 Run `DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build /Users/tyleraraujo/vicuna/build-codex-emotive --target llama-server -j8`
- [ ] T013 Run `LLAMA_SERVER_BIN_PATH=/Users/tyleraraujo/vicuna/build-codex-emotive/bin/llama-server /opt/homebrew/bin/pytest /Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py -q -k "telegram"`
