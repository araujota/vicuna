# Feature Specification: CI Trigger Dedupe

**Feature Branch**: `[038-ci-trigger-dedupe]`  
**Created**: 2026-03-14  
**Status**: Draft  
**Input**: User description: "modify the CI pipeline to dedupe these types of errors. we should only run these checks on push going forward, no longer on PR."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Eliminate duplicate validation runs on PRs (Priority: P1)

As a maintainer reviewing a pull request, I want code-validation workflows to stop launching both `push` and `pull_request` runs for the same branch update so that failures appear once and check noise is reduced.

**Why this priority**: The current duplication creates misleading failure counts and wastes CI time on every branch update.

**Independent Test**: Update a branch that matches validation workflow path filters and confirm the workflows now trigger only from `push`, while PR-only automation remains available.

**Acceptance Scenarios**:

1. **Given** a branch update that changes files matched by a validation workflow, **When** the branch is pushed, **Then** the workflow runs from `push`.
2. **Given** an open pull request for that branch, **When** the same branch update lands, **Then** no duplicate `pull_request` validation run is created for the affected workflow.

### User Story 2 - Preserve non-validation PR automation (Priority: P2)

As a maintainer, I want PR-specific automation such as labeling to keep working so that deduping validation does not remove workflows that exist specifically for pull-request lifecycle actions.

**Why this priority**: PR metadata workflows serve a different purpose than validation and should not be disabled accidentally.

**Independent Test**: Inspect the workflow inventory after the change and confirm PR-only automation files still retain PR triggers.

**Acceptance Scenarios**:

1. **Given** a workflow that exists to react to PR metadata events, **When** the CI dedupe change is applied, **Then** its PR trigger remains intact.

### Edge Cases

- Workflows with `workflow_dispatch` or `schedule` triggers must retain those triggers unchanged.
- Path-filtered validation workflows must continue to run on `push` only for the same path sets they use today.
- Trigger dedupe must not change job names, concurrency groups, or branch/path filters beyond removing `pull_request`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST stop running duplicate validation workflows on both `push` and `pull_request` for the same branch update.
- **FR-002**: Validation workflows currently configured with both `push` and `pull_request` MUST keep their existing `push`, `workflow_dispatch`, and `schedule` behavior while removing `pull_request`.
- **FR-003**: PR-specific metadata or governance workflows MUST retain their PR triggers.
- **FR-004**: The implementation MUST be limited to workflow-trigger dedupe and MUST NOT change validation job contents, names, or required toolchains.
- **FR-005**: The repository MUST record the intended workflow scope and operator-facing rationale in Spec Kit artifacts for this change.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For validation workflows touched by this feature, the YAML no longer contains a `pull_request` trigger after implementation.
- **SC-002**: PR check lists no longer show duplicated validation jobs caused by paired `push` and `pull_request` runs for the same branch update.
- **SC-003**: PR-specific automation workflows outside the dedupe scope retain their PR trigger configuration unchanged.
