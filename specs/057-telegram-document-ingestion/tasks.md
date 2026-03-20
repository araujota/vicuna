# Tasks: Telegram Document Ingestion And Supermemory Linking

**Input**: Design documents from `/Users/tyleraraujo/vicuna/specs/057-telegram-document-ingestion/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Tests**: Targeted Node tests are required because this change alters bridge
behavior, Telegram request handling, and Supermemory persistence.

**Organization**: Tasks are grouped by user story to enable independent
implementation and testing.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare dependencies and bridge documentation surfaces

- [ ] T001 Add the PDF extraction dependency in /Users/tyleraraujo/vicuna/package.json and refresh /Users/tyleraraujo/vicuna/package-lock.json
- [ ] T002 Update operator guidance for Telegram document ingestion in /Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add explicit shared helpers for document detection, metadata
linking, and normalization before wiring the Telegram entrypoint

- [ ] T003 Implement Telegram document type detection, linkage metadata builders, and transcript formatting helpers in /Users/tyleraraujo/vicuna/tools/telegram-bridge/lib.mjs
- [ ] T004 [P] Add unit coverage for helper-level detection, metadata linking, and text normalization in /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs

**Checkpoint**: Helper policy is explicit and test-covered; entrypoint wiring can begin.

---

## Phase 3: User Story 1 - Telegram Documents Become Plain-Text Chat Turns (Priority: P1) 🎯 MVP

**Goal**: Supported document messages are downloaded, converted to plain text,
and forwarded as user turns

**Independent Test**: Simulate supported Telegram document messages and confirm
the bridge produces normalized plain-text transcript content instead of the
plain-text-only rejection

### Tests for User Story 1

- [ ] T005 [P] [US1] Add document-ingestion transcript tests in /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs

### Implementation for User Story 1

- [ ] T006 [US1] Implement Telegram `getFile` lookup and raw file download handling in /Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs
- [ ] T007 [US1] Implement PDF extraction with `pdf-parse` and DOC/DOCX extraction with `textutil` in /Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs
- [ ] T008 [US1] Append normalized document transcript turns and forward them through the existing Vicuna chat flow in /Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs

**Checkpoint**: The bridge can ingest supported documents into plain-text user turns.

---

## Phase 4: User Story 2 - Raw Files And Extracted Text Are Persisted In Supermemory (Priority: P1)

**Goal**: Each supported document persists both the raw file and extracted text
with shared linkage metadata

**Independent Test**: Mock Supermemory calls and verify raw upload, update, and
extracted-text add requests share the expected linkage metadata and container
grouping

### Tests for User Story 2

- [ ] T009 [P] [US2] Add Supermemory persistence and linkage tests in /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs

### Implementation for User Story 2

- [ ] T010 [US2] Add Supermemory client configuration and persistence helpers for raw-file upload plus extracted-text add in /Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs
- [ ] T011 [US2] Wire document persistence failures and partial-failure reporting back to Telegram replies in /Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs

**Checkpoint**: The bridge durably stores both source artifacts and extracted text.

---

## Phase 5: User Story 3 - The Bridge Remains Explicit About Supported Formats And Host Requirements (Priority: P2)

**Goal**: Maintain explicit host/tooling rules and predictable operator-visible
failure behavior

**Independent Test**: Unsupported types and missing-host-tool cases produce
clear bridge errors and do not silently degrade

### Tests for User Story 3

- [ ] T012 [P] [US3] Add unsupported-format and missing-host-tool tests in /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs

### Implementation for User Story 3

- [ ] T013 [US3] Add explicit unsupported-format and missing-`textutil` handling in /Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs
- [ ] T014 [US3] Update bridge docs with host requirements and Supermemory configuration in /Users/tyleraraujo/vicuna/tools/telegram-bridge/README.md

**Checkpoint**: Format support and failure behavior are explicit and documented.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and consistency checks

- [ ] T015 [P] Run targeted bridge validation in /Users/tyleraraujo/vicuna/specs/057-telegram-document-ingestion/quickstart.md
- [ ] T016 [P] Review formatting, error wording, and final test coverage in /Users/tyleraraujo/vicuna/tools/telegram-bridge/bridge.test.mjs and /Users/tyleraraujo/vicuna/tools/telegram-bridge/index.mjs
