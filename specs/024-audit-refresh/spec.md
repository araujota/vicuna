# Feature Specification: Refreshed Functionality Audit Of The Exotic-Intelligence Runtime

**Feature Branch**: `024-audit-refresh`
**Created**: 2026-03-13
**Status**: Draft
**Input**: User description: "pull the first audit report you conducted for this application, then run the same audit policy again, focusing on functionality, elegance, generalizability, expandability, and the cumulative status of vicuna as \"exotic intelligence/inference\" or \"crude RSI\" with explicit mention of which gaps from the original report have been improved upon, and how the system might be improved further based on SOTA theses of this sort of biologically inspired architecture"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Recover The Original Audit Baseline (Priority: P1)

As the project operator, I need the first functionality audit recovered and
restated as the comparison baseline so the refreshed report does not drift away
from the original evaluation criteria or overstate what changed.

**Why this priority**: The new audit is only useful if it clearly anchors to the
original report and its gap claims.

**Independent Test**: Read the refreshed report and verify it cites the original
audit artifact, summarizes its verdict, and lists the original high-priority
gaps in comparable language.

**Acceptance Scenarios**:

1. **Given** the original audit artifact in `specs/013-functionality-audit/`,
   **When** the refreshed audit is completed, **Then** it explicitly identifies
   the original verdict and principal gap areas.
2. **Given** a new claim about improvement, **When** the refreshed report
   presents it, **Then** it ties that claim back to a gap or judgment from the
   first audit.

---

### User Story 2 - Reassess Current Runtime Status (Priority: P1)

As the project architect, I need the current runtime audited with the same
policy as before, but with additional focus on functionality, elegance,
generalizability, expandability, and the current status of Vicuña as an
"exotic intelligence" or "crude RSI" system.

**Why this priority**: The user wants a fresh verdict on the system as it
exists now, not only a change log since the first audit.

**Independent Test**: Read the refreshed report and verify it includes a
current-system verdict plus explicit judgments for functionality, elegance,
generalizability, and expandability, backed by repository evidence.

**Acceptance Scenarios**:

1. **Given** the current local repository, **When** the refreshed audit covers a
   subsystem, **Then** it distinguishes what is implemented, what is wired into
   runtime behavior, and what remains heuristic or shallow.
2. **Given** the new self-state-gradient and Adam-backed control surfaces,
   **When** the refreshed audit discusses recursive self-improvement status,
   **Then** it explains whether those additions materially strengthen the RSI
   claim or remain bounded scaffolding.

---

### User Story 3 - Compare Current Architecture To External Best Practice (Priority: P2)

As a research-driven builder, I need the refreshed audit to compare Vicuña’s
current architecture to current public theses and implementations around
biologically inspired control, persistent memory, active inference/allostasis,
and bounded self-improvement so I can prioritize the next serious improvements.

**Why this priority**: The user explicitly wants further-improvement guidance
based on state-of-the-art ideas for this class of architecture.

**Independent Test**: Inspect the refreshed report and verify that each major
recommendation is tied to a primary research source or an official project
source, not just internal opinion.

**Acceptance Scenarios**:

1. **Given** an external comparison area such as active inference, durable agent
   memory, or eval-driven self-improvement, **When** the report cites it,
   **Then** it uses a primary paper, official project docs, or official repo.
2. **Given** a recommended improvement path, **When** the report proposes it,
   **Then** it explains how it would improve functionality, elegance,
   generalizability, or expandability relative to the current runtime.

### Edge Cases

- What happens when the original audit’s wording no longer matches the current
  architecture because new cognitive-loop or Active LoRA features were added?
- What happens when a capability improved locally but still trails durable
  external-memory or planner/executor systems in public practice?
- What happens when there is no public SOTA consensus for concepts such as
  machine selfhood, allostasis, or recursive self-improvement?
- What happens when a subsystem became more sophisticated mathematically but is
  still bounded by authored CPU-side policy and lacks eval-driven validation?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The refreshed audit MUST identify the original audit artifact and
  extract its verdict, key findings, and priority gaps.
- **FR-002**: The refreshed audit MUST reassess the current implementation using
  the same core policy as the original audit: implemented functionality,
  whitepaper parity, and public-SOTA parity.
- **FR-003**: The refreshed audit MUST add explicit judgments for functionality,
  elegance, generalizability, and expandability.
- **FR-004**: The refreshed audit MUST state the current cumulative status of
  Vicuña as an "exotic intelligence" runtime and as a "crude RSI" system.
- **FR-005**: The refreshed audit MUST explicitly identify which major gaps from
  the original report improved, which remain unresolved, and which new gaps
  emerged from later architectural changes.
- **FR-006**: The refreshed audit MUST use concrete repository references for
  current implementation claims, including current self-state, functional LoRA,
  DMN, governance, and hard-memory surfaces.
- **FR-007**: The refreshed audit MUST use GitHub history or commit evidence to
  support claims about what changed since the first audit.
- **FR-008**: The refreshed audit MUST compare the current design against
  current public theses or implementations for biologically inspired control,
  stateful agents, memory, and bounded self-improvement using primary sources.
- **FR-009**: The refreshed audit MUST propose further improvements that are
  explicitly tied to functionality, elegance, generalizability, or
  expandability gains.
- **FR-010**: The refreshed audit MUST be delivered as a durable repository
  artifact under a new specs directory.

### Key Entities *(include if feature involves data)*

- **Original Audit Baseline**: The first audit report and its supporting spec
  artifacts under `specs/013-functionality-audit/`.
- **Current Runtime Evidence Set**: The current source files, tests, server
  integration points, architecture docs, and GitHub commits used to justify the
  refreshed audit.
- **Gap Delta**: A structured comparison between the original audit’s priority
  gaps and the current runtime status.
- **Refreshed Audit Report**: A markdown report that combines current status,
  delta analysis, external comparison, and forward recommendations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reader can identify the original audit’s verdict and see exactly
  how the refreshed audit differs from it without reading the whole repo.
- **SC-002**: Every substantive claim about current implementation status is
  backed by at least one concrete repository or test reference.
- **SC-003**: Every major external recommendation or comparison is backed by at
  least one primary research source or official project source.
- **SC-004**: The refreshed report clearly distinguishes improved gaps,
  unresolved gaps, and newly introduced risks or limitations.
- **SC-005**: The refreshed report provides a clear bottom-line judgment on
  whether Vicuña is better understood today as exploratory exotic inference,
  bounded self-conditioning, or crude recursive self-improvement.
