# Tasks: Tavily Source-First Quality

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/075-tavily-source-first-quality/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`

## Phase 1: Wrapper Hardening

- [x] T001 Add explicit Tavily request normalization helpers in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/tavily-search.ts`
- [x] T002 Disable provider-generated answers by default and request richer source content in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/tavily-search.ts`

## Phase 2: Runtime Schema and Dispatch

- [x] T003 Expand the `web_search` capability schema in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/catalog.ts`
- [x] T004 Pass through the expanded Tavily parameters in `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp`

## Phase 3: Validation

- [x] T005 Add Node regression tests for Tavily schema and request normalization in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/test/catalog.test.ts` and a new targeted test file if needed
- [x] T006 Update `/Users/tyleraraujo/vicuna/tools/openclaw-harness/README.md` and `/Users/tyleraraujo/vicuna/tools/server/README-dev.md`
