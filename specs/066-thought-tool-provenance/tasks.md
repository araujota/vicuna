# Tasks: Full Thought and Tool Provenance Logging

## Phase 1: Provenance Schema Expansion

- [ ] T001 Add JSON helpers for full plan-step and candidate serialization in `tools/server/server-context.cpp`.
- [ ] T002 Add structured request serializers for bash, hard-memory, Codex, and Telegram relay requests in `tools/server/server-context.cpp`.
- [ ] T003 Add a dedicated `tool_call` provenance append path in `tools/server/server-context.cpp`.

## Phase 2: Active and DMN Narration Capture

- [ ] T004 Include active planner narration, visible output, and raw tool XML in active provenance events when available.
- [ ] T005 Include full DMN prompt narration, concept detail, and plan/candidate detail in DMN provenance events.

## Phase 3: Docs and Tests

- [ ] T006 Update `tools/server/README-dev.md` to document full narration and exact tool-call logging.
- [ ] T007 Add/adjust native assertions in `tests/test-cognitive-loop.cpp` for planner reasoning note capture.
- [ ] T008 Add server provenance assertions in `tools/server/tests/unit/test_basic.py`.

## Phase 4: Verification

- [ ] T009 Rebuild `llama-server` and targeted native tests.
- [ ] T010 Run targeted native and server test suites and confirm old summary-only logging gaps are closed.
