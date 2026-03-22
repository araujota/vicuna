# Tasks: Authoritative ReAct Continuation

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/073-authoritative-react-continuation/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`

**Tests**: Targeted native regression coverage plus host runtime verification are required.

## Phase 1: Foundations

- [ ] T001 Add explicit foreground-turn policy helpers for live-fact detection and procedural-answer rejection in `/Users/tyleraraujo/vicuna/tools/server/server-common.cpp` and `/Users/tyleraraujo/vicuna/tools/server/server-common.h`
- [ ] T002 Persist the latest foreground text on `/Users/tyleraraujo/vicuna/tools/server/server-task.h` and populate it in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`

## Phase 2: Authoritative ReAct Continuation

- [ ] T003 Add first-step tool-choice forcing for clearly mutable/live active turns in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T004 Add explicit semantic terminal validation and same-turn retry feedback for unsupported active answers in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`
- [ ] T005 Replace the small fixed authoritative ReAct retry limit with an effectively unbounded continuation budget in `/Users/tyleraraujo/vicuna/tools/server/server-task.h`

## Phase 3: Validation

- [ ] T006 Extend `/Users/tyleraraujo/vicuna/tests/test-server-common.cpp` with regression coverage for live-fact detection and procedural-answer rejection
- [ ] T007 Extend `/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp` with regression coverage for authoritative ReAct continuation defaults and resume state
- [ ] T008 Update `/Users/tyleraraujo/vicuna/ARCHITECTURE.md` and `/Users/tyleraraujo/vicuna/tools/server/README-dev.md` to document grounded continuation and terminal-answer policy
- [ ] T009 Run targeted local validation, rebuild the host runtime, and verify a Telegram weather-style request takes a tool-grounded continuation path before answering
