# Tasks: OpenClaw Tool Descriptor Unification

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/077-openclaw-tool-descriptor-unification/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`

## Phase 1: Contract Validation

- [x] T001 Add recursive schema-description validation to `/Users/tyleraraujo/vicuna/common/openclaw-tool-fabric.cpp`
- [x] T002 Add matching schema-description validation to `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/contracts.ts`

## Phase 2: Descriptor Completion

- [x] T003 Add parameter descriptions to builtin tool schemas in `/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp`
- [x] T004 Add parameter descriptions to external tool schemas in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/catalog.ts`

## Phase 3: Verification and Docs

- [x] T005 Extend `/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp`
- [x] T006 Extend `/Users/tyleraraujo/vicuna/tools/openclaw-harness/test/catalog.test.ts`
- [x] T007 Update `/Users/tyleraraujo/vicuna/tools/server/README-dev.md` and `/Users/tyleraraujo/vicuna/tools/openclaw-harness/README.md`
- [ ] T008 Validate locally where possible and redeploy the host runtime if needed
