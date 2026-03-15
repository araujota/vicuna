# Tasks: CI Trigger Dedupe

**Input**: Design documents from `/specs/038-ci-trigger-dedupe/`
**Prerequisites**: plan.md, spec.md, research.md

**Tests**: Validate by re-reading the edited workflow YAML and confirming the target workflows no longer contain `pull_request` triggers while excluded workflows remain unchanged.

## Phase 1: Setup

- [x] T001 [US1] Confirm the duplicated validation workflows and PR-only exclusions in `/Users/tyleraraujo/vicuna/specs/038-ci-trigger-dedupe/research.md`

## Phase 2: Foundational

- [x] T002 [US1] Remove `pull_request` triggers from validation workflows in `/Users/tyleraraujo/vicuna/.github/workflows/vicuna-ci.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/python-lint.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/python-type-check.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/server-webui.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/check-vendor.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/pre-tokenizer-hashes.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/python-check-requirements.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/editorconfig.yml`, `/Users/tyleraraujo/vicuna/.github/workflows/update-ops-docs.yml`, and `/Users/tyleraraujo/vicuna/.github/workflows/copilot-setup-steps.yml`

## Phase 3: User Story 1 - Eliminate duplicate validation runs on PRs (Priority: P1)

**Goal**: Validation workflows run from `push` only and no longer create duplicate PR check noise.

**Independent Test**: Inspect each targeted workflow and confirm the `pull_request` block is absent while `push` remains.

### Validation

- [x] T003 [US1] Re-scan the targeted workflow files and verify only `push`, `workflow_dispatch`, and `schedule` triggers remain where applicable

## Phase 4: User Story 2 - Preserve non-validation PR automation (Priority: P2)

**Goal**: PR-only automation remains intact.

**Independent Test**: Confirm excluded PR automation workflows still retain their PR triggers.

### Validation

- [x] T004 [US2] Re-check `/Users/tyleraraujo/vicuna/.github/workflows/labeler.yml` and any untouched PR automation workflow to confirm PR triggers were preserved

## Phase 5: User Story 3 - Report full gate failure sets in one run (Priority: P2)

**Goal**: Main quality gates evaluate the full repository and surface all failures in a single run.

**Independent Test**: Inspect `Vicuna CI` and confirm quality jobs no longer scope to changed-file inventories or use fail-fast pytest invocation.

### Implementation

- [x] T005 [US3] Update `/Users/tyleraraujo/vicuna/.github/workflows/vicuna-ci.yml` so `lint-type-quality` runs repository-wide pre-commit, flake8, pyright, clang-format, and lizard checks with aggregated exit reporting
- [x] T006 [US3] Update `/Users/tyleraraujo/vicuna/.github/workflows/vicuna-ci.yml` so `clang-tidy` runs across repository native sources and aggregates failures instead of stopping on the first changed-file violation
- [x] T007 [US3] Remove fail-fast pytest behavior from `/Users/tyleraraujo/vicuna/.github/workflows/vicuna-ci.yml`
- [x] T008 [US3] Add a checked-in repository-wide complexity baseline in `/Users/tyleraraujo/vicuna/.github/ci/lizard-baseline.txt` and compare future `lizard` results against it via `/Users/tyleraraujo/vicuna/.github/scripts/check_lizard_baseline.py`
- [x] T010 [US3] Add a checked-in repository-wide clang-format baseline in `/Users/tyleraraujo/vicuna/.github/ci/clang-format-baseline.txt` and compare future full-tree formatting drift via `/Users/tyleraraujo/vicuna/.github/scripts/check_clang_format_baseline.py`
- [x] T011 [US3] Add a checked-in repository-wide clang-tidy baseline in `/Users/tyleraraujo/vicuna/.github/ci/clang-tidy-baseline.txt` and compare future full-tree diagnostics via `/Users/tyleraraujo/vicuna/.github/scripts/check_clang_tidy_baseline.py`
- [x] T012 [US3] Remove the lint dependency from `/Users/tyleraraujo/vicuna/.github/workflows/vicuna-ci.yml` so `python-server-tests` always reports in the same push run
- [x] T013 [US3] Repair mixed C/C++ sanitizer linking for `/Users/tyleraraujo/vicuna/tests/test-c.c` in `/Users/tyleraraujo/vicuna/tests/CMakeLists.txt`

## Phase 6: Polish

- [x] T009 Update task completion state in `/Users/tyleraraujo/vicuna/specs/038-ci-trigger-dedupe/tasks.md` after validation
