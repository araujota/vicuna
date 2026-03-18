# Tasks: Telegram Bridge Middleware

**Input**: Design documents from `/specs/050-telegram-bridge/`

**Tests**: Required. This task must verify bridge parser behavior and live
startup against the local Vicuña server.

## Phase 1: Bridge Contract

- [x] T001 Record the local, GitHub, and web research for Telegram bridge
  memory and tool wiring in `/Users/tyleraraujo/vicuna/specs/050-telegram-bridge/research.md`
- [x] T002 Update the Telegram bridge spec and plan to cover bounded per-chat
  transcript persistence and managed bash-tool configuration in
  `/Users/tyleraraujo/vicuna/specs/050-telegram-bridge/spec.md` and
  `/Users/tyleraraujo/vicuna/specs/050-telegram-bridge/plan.md`

## Phase 2: Implementation

- [x] T003 Extend `/Users/tyleraraujo/vicuna/tools/telegram-bridge/lib.mjs`
  with normalized chat-id keyed transcript state, bounded trimming helpers, and
  persistence support
- [x] T004 Update `/Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs`
  to reuse persisted transcript state for each Telegram chat while preserving
  proactive self-emit relay behavior
- [x] T005 Update `/Users/tyleraraujo/vicuna/tools/ops/run-vicuna-runtime.sh`
  to export explicit `VICUNA_BASH_TOOL_*` settings for managed runtime startup
- [x] T006 Add operator documentation for transcript persistence and managed
  bash-tool configuration in
  `/Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md`

## Phase 3: Validation

- [x] T007 Add helper-level tests for transcript persistence, transcript
  trimming, stream parsing, and response text extraction in
  `/Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs`
- [x] T008 Run `npm run test:telegram-bridge`
- [x] T009 Smoke-check the managed runtime launcher command path relevant to the
  bridge workflow
