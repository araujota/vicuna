# Feature Specification: PR #4 CI Remediation

**Feature Branch**: `042-discovered-self-state-consolidation`  
**Created**: 2026-03-17  
**Status**: Draft  
**Input**: User description: "the clang-tidy, lint-type-quality, and python-server-test failed in the CI pipeline for pr #4. pull these failures along with any copilot comments on the PR, assess their validity, and rectify all issues completely."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Restore Runtime Snapshot Persistence (Priority: P1)

As a maintainer, I can rely on runtime snapshot persistence working even when optional runtime subsystems have not been initialized, so server restart flows and persistence-backed tests do not fail on startup or after a request.

**Why this priority**: The failing Python server test is a concrete runtime defect that blocks merge readiness and breaks a shipped operator-facing path.

**Independent Test**: Run the focused Python server test suite and verify `test_runtime_snapshot_survives_restart` creates the snapshot file, restores successfully, and leaves health reporting healthy.

**Acceptance Scenarios**:

1. **Given** runtime persistence is enabled and functional LoRA state was never initialized, **When** the server persists runtime state, **Then** snapshot writing succeeds instead of reporting a functional snapshot archive query failure.
2. **Given** a persisted runtime snapshot, **When** the server restarts, **Then** health output reports runtime persistence as enabled, restore attempted, and healthy.

---

### User Story 2 - Clear PR #4 Quality Gates (Priority: P1)

As a maintainer, I can run the repository quality gates for the changed code and see no new clang-tidy or lizard baseline regressions, so PR #4 can pass CI without weakening repository standards.

**Why this priority**: `clang-tidy` and `lint-type-quality` are required CI gates for merge readiness, and the failures are tied to code introduced on this PR path.

**Independent Test**: Run the repository baseline-check scripts used by CI and verify they report no new or regressed findings for the remediated code.

**Acceptance Scenarios**:

1. **Given** the remediated branch, **When** the clang-tidy baseline script runs with the repository build directory, **Then** it reports no new or regressed diagnostics from PR #4 changes.
2. **Given** the remediated branch, **When** the lizard baseline script runs, **Then** it reports no new or changed warnings for the PR #4 code paths.

---

### User Story 3 - Resolve Valid PR Review Findings (Priority: P2)

As a maintainer, I can review the PR comments from automation, determine which findings are valid, and incorporate the valid fixes into the branch so review noise does not remain unresolved.

**Why this priority**: The PR includes automated inline findings that should be triaged deliberately rather than accepted or ignored blindly.

**Independent Test**: Inspect the affected APIs and functions after remediation and verify the valid findings are addressed without changing intended behavior.

**Acceptance Scenarios**:

1. **Given** a valid variable-shadowing or large-object API issue, **When** the code is updated, **Then** the implementation preserves behavior while removing the flagged pattern.
2. **Given** a valid large-function documentation or structure concern, **When** the code is updated, **Then** the affected function is either clarified or split enough to satisfy the quality gates without hiding policy.

### Edge Cases

- What happens when runtime persistence is enabled before any functional or process-functional adapter banks are initialized?
- What happens when a quality gate failure comes from repository baseline debt outside the PR rather than from changed code?
- How does the remediation preserve explicit CPU-side runtime policy while refactoring large functions for lizard or clang-tidy?
- What happens if the PR has no Copilot comments and only security-bot review findings?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow runtime snapshot persistence to succeed when functional LoRA or process-functional runtime state is absent or uninitialized.
- **FR-002**: System MUST preserve valid runtime snapshot export and import behavior for initialized functional and process-functional archives.
- **FR-003**: System MUST remove new or regressed clang-tidy diagnostics introduced by PR #4 without weakening the repository clang-tidy workflow.
- **FR-004**: System MUST remove new or changed lizard baseline warnings introduced by PR #4 without bypassing the repository complexity gate.
- **FR-005**: System MUST assess PR review findings for validity and implement fixes only for findings that are technically justified.
- **FR-006**: System MUST document in the remediation outcome whether any Copilot comments were found on PR #4.
- **FR-007**: System MUST keep runtime policy explicit in CPU-side control code when refactoring the cognitive loop, self-state, or server persistence paths.
- **FR-008**: System MUST update targeted tests or validation commands alongside behavior changes.

### Key Entities *(include if feature involves data)*

- **Runtime Persistence Snapshot**: The server-side serialized runtime state that includes self-state, proactive mailbox data, and optional functional or process-functional archives.
- **Quality Gate Regression**: A new or increased diagnostic or complexity warning detected by the same baseline-comparison scripts used in CI.
- **Review Finding Triage**: The explicit classification of an automated PR comment as valid and fixable, valid but intentionally deferred, or not actionable.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The focused Python server test suite passes locally, including `unit/test_basic.py::test_runtime_snapshot_survives_restart`.
- **SC-002**: The clang-tidy baseline comparison reports no new or regressed diagnostics for the remediated branch.
- **SC-003**: The lizard baseline comparison reports no new or changed warnings for the remediated branch.
- **SC-004**: All valid automated PR review findings for PR #4 are either fixed in code or explicitly assessed as not requiring a code change.
