# Tasks: Retry Grounding Surgery

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/074-retry-grounding-surgery/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`

**Tests**: Targeted native regression coverage plus host runtime verification are required.

## Phase 1: Runtime Policy

- [ ] T001 Add explicit helpers for mutable active retry escalation and trusted-grounding classification in `/Users/tyleraraujo/vicuna/tools/server/server-common.h` and `/Users/tyleraraujo/vicuna/tools/server/server-common.cpp`
- [ ] T002 Refine canonical grounding checks in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` so fresh mutable Telegram turns do not trust stale assistant dialogue as grounded context

## Phase 2: Prompt and Continuation Wiring

- [ ] T003 Escalate active mutable retries to `tool_choice=required` after repeated rejected non-tool retries in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T004 Tighten retry feedback and preserve resumed tool-result behavior in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`

## Phase 3: Validation

- [ ] T005 Extend `/Users/tyleraraujo/vicuna/tests/test-server-common.cpp` with regression coverage for retry escalation and mutable request classification
- [ ] T006 Extend `/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp` with regression coverage for authoritative retry defaults used by the new policy
- [ ] T007 Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and `/Users/tyleraraujo/vicuna/tools/server/README-dev.md`
- [ ] T008 Run targeted local validation, rebuild the host runtime, and verify the Telegram weather-style path uses a tool without long retry spirals
