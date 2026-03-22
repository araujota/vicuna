# Tasks: Telegram Ask-With-Options Tool

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/071-telegram-ask-options/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`

**Tests**: Targeted native runtime tests plus bridge/runtime validation are required.

## Phase 1: Foundations

- [x] T001 Add typed `ask_with_options` tool kinds, request/result structs, and C API surfaces in `/Users/tyleraraujo/vicuna/include/llama.h`, `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.h`, and `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`
- [x] T002 Add the new OpenClaw builtin tool descriptor, schema, XML contract, and active/DMN eligibility in `/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp`
- [x] T003 Add a bounded Telegram ask-options outbox surface and dispatch path in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`, `/Users/tyleraraujo/vicuna/tools/server/server-context.h`, `/Users/tyleraraujo/vicuna/tools/server/server.cpp`, and `/Users/tyleraraujo/vicuna/tools/server/server-http.cpp`

## Phase 2: Bridge Continuation

- [x] T004 Extend bridge state for outbound ask-options cursor and pending option prompts in `/Users/tyleraraujo/vicuna/tools/telegram-bridge/lib.mjs`
- [x] T005 Extend the bridge to poll the runtime ask-options outbox, send Telegram `reply_markup.inline_keyboard`, and persist pending prompt state in `/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs`
- [x] T006 Extend the bridge to consume `callback_query` updates, acknowledge them, disable consumed keyboards, and rewrite the selection into bounded transcript state before resuming the Telegram-origin turn in `/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs`
- [x] T007 Remove the hardcoded empty-response fallback for intentional tool-delivered Telegram output in `/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs`

## Phase 3: Validation and Docs

- [x] T008 Add native regression coverage for the new tool descriptor, XML parsing, and request/result plumbing in `/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp` and `/Users/tyleraraujo/vicuna/tests/test-cognitive-loop.cpp`
- [x] T009 Update operator and architecture docs in `/Users/tyleraraujo/vicuna/tools/server/README-dev.md`, `/Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md`, and `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`
- [ ] T010 Run targeted local validation and rebuild the host runtime and bridge, then verify the new tool is present in the live capability set
