# Tasks: Radarr and Sonarr OpenClaw Tools

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/078-radarr-sonarr-openclaw/`  
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`

## Phase 1: Setup and Foundational Work

- [ ] T001 Extend `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/config.ts` with Radarr/Sonarr secrets and default LAN base URLs
- [ ] T002 Add `radarr` and `sonarr` runtime catalog descriptors with full parameter descriptions in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/catalog.ts`
- [ ] T003 Update `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/index.ts` and related runtime-catalog code if needed so the new capabilities are emitted through the shared external catalog path

## Phase 2: User Story 1 - Inspect Radarr and Sonarr (Priority: P1)

**Goal**: Make both services visible and usable for read-only inspection through the authoritative OpenClaw surface.

**Independent Test**: Build the runtime catalog and confirm it advertises `radarr` and `sonarr`; run wrapper tests for status, queue, root-folder, quality-profile, list, lookup, and calendar request shaping.

### Tests for User Story 1

- [ ] T004 [P] [US1] Extend `/Users/tyleraraujo/vicuna/tools/openclaw-harness/test/catalog.test.ts` for the new runtime capabilities and their schema descriptions
- [ ] T005 [P] [US1] Add `/Users/tyleraraujo/vicuna/tools/openclaw-harness/test/servarr.test.ts` for config resolution, request shaping, and typed error handling

### Implementation for User Story 1

- [ ] T006 [P] [US1] Add shared Servarr HTTP/config logic in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/servarr.ts`
- [ ] T007 [P] [US1] Add the Radarr CLI wrapper in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/radarr.ts`
- [ ] T008 [P] [US1] Add the Sonarr CLI wrapper in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/sonarr.ts`
- [ ] T009 [P] [US1] Add wrapper launch scripts in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/bin/radarr-api` and `/Users/tyleraraujo/vicuna/tools/openclaw-harness/bin/sonarr-api`
- [ ] T010 [US1] Update `/Users/tyleraraujo/vicuna/tools/server/server-context.cpp` to dispatch the new capabilities through the wrapper scripts

## Phase 3: User Story 2 - Search and Add Media (Priority: P1)

**Goal**: Support lookup and add flows through the same tool surface with upstream-compatible request bodies.

**Independent Test**: Invoke lookup and add payloads locally and verify request-body construction and typed failure on missing required add inputs.

### Tests for User Story 2

- [ ] T011 [P] [US2] Extend `/Users/tyleraraujo/vicuna/tools/openclaw-harness/test/servarr.test.ts` for `add_movie` and `add_series` payload validation and body shaping

### Implementation for User Story 2

- [ ] T012 [US2] Implement Radarr lookup-assisted add flow in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/radarr.ts`
- [ ] T013 [US2] Implement Sonarr lookup-assisted add flow in `/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/sonarr.ts`

## Phase 4: User Story 3 - One Shared Fabric and Host Deployment (Priority: P2)

**Goal**: Keep Radarr and Sonarr in the same OpenClaw system and deploy them live on the host.

**Independent Test**: Rebuild the host and confirm the live capability log and XML tool guidance include `radarr` and `sonarr`.

### Tests for User Story 3

- [ ] T014 [P] [US3] Extend `/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp` if needed for XML guidance or external capability validation

### Implementation for User Story 3

- [ ] T015 [US3] Update `/Users/tyleraraujo/vicuna/tools/openclaw-harness/README.md` and `/Users/tyleraraujo/vicuna/tools/server/README-dev.md`
- [ ] T016 [US3] Build the harness, sync the runtime catalog, rebuild the host runtime, and confirm the live OpenClaw capability set advertises `radarr` and `sonarr`
