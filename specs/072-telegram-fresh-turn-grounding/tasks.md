# Tasks: Telegram Fresh-Turn Grounding

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/072-telegram-fresh-turn-grounding/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`

**Tests**: Targeted native regression coverage plus host runtime verification are required.

## Phase 1: Foundations

- [ ] T001 Add a raw-request-aware foreground extraction path in `/Users/tyleraraujo/vicuna/tools/server/server-common.cpp`, `/Users/tyleraraujo/vicuna/tools/server/server-common.h`, and `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T002 Reorder Telegram-scoped canonical ReAct message assembly so bounded Telegram dialogue remains terminal in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`

## Phase 2: Validation

- [ ] T003 Add targeted regression coverage for raw-body foreground extraction in `/Users/tyleraraujo/vicuna/tests/test-server-common.cpp` and wire it into `/Users/tyleraraujo/vicuna/tests/CMakeLists.txt`
- [ ] T004 Update the relevant architecture/spec artifacts for Telegram active-turn grounding in `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and `/Users/tyleraraujo/vicuna/specs/072-telegram-fresh-turn-grounding/`
- [ ] T005 Run local validation, rebuild the host runtime, and verify a fresh Telegram-scoped prompt no longer replays the old Tesla answer
