# Feature Specification: Functionality Audit Of The Exotic-Intelligence Runtime

**Feature Branch**: `013-functionality-audit`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "conduct a robust functionality audit of this application as it currently stands, with respect to Vicuña_WP.md, the original whitepaper, and with respect to current state of the art implementations of every piece of functionality and high level goal we are attempting to achieve. produce a comprehensive report on the functionality and expected behavior of the application as a crude RSI exotic intelligence that facilitates its own improvement and maintains an internal self-representation that motivates autonomous behavior. present your writeups of each piece of functionality that facilitates this system with reference to their parity with current state of the art understandings of how to achieve each piece and the high-level goals."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish The Current Functional Surface (Priority: P1)

As the project operator, I need a grounded inventory of what the current Vicuña
application actually implements, exposes, and tests so I can distinguish
working runtime behavior from whitepaper intent, aspirational specs, and missing
integration.

**Why this priority**: Any parity or state-of-the-art comparison is invalid if
the audit does not first establish the current implemented surface.

**Independent Test**: Read the report section on implemented functionality and
verify that each claimed capability is backed by concrete repository references
to source files, tests, or server integration points.

**Acceptance Scenarios**:

1. **Given** the local repository and active branch, **When** the audit is
   completed, **Then** it enumerates the implemented memory, self-state,
   cognitive-loop, governance, and hard-memory surfaces with code references.
2. **Given** a capability appears only in specs or the whitepaper, **When** the
   audit discusses it, **Then** it clearly labels that capability as partial,
   absent, or not wired into user-facing runtime behavior.

---

### User Story 2 - Measure Whitepaper Parity (Priority: P1)

As the project architect, I need each major whitepaper commitment translated
into expected runtime behavior and compared against the current implementation
so I can see where the application matches, approximates, or fails the Vicuña
architecture.

**Why this priority**: The whitepaper is the project’s architectural north star,
and the audit must show whether the codebase delivers that design.

**Independent Test**: Read the report section for each whitepaper pillar and
verify that it contains expected behavior, current behavior, and a parity
assessment.

**Acceptance Scenarios**:

1. **Given** a whitepaper pillar such as memory cascade, DMN, or persistent
   self-state, **When** the audit covers it, **Then** it describes the intended
   behavior and the current parity level in the codebase.
2. **Given** a capability is implemented in a crude or heuristic form, **When**
   the audit describes parity, **Then** it states that the system has a partial
   implementation rather than overstating completion.

---

### User Story 3 - Compare Against Current State Of The Art (Priority: P2)

As a research-driven builder, I need the current implementation compared against
current state-of-the-art approaches for long-term memory, persistent agent
state, tool-using autonomy, and self-improvement so I can see whether Vicuña is
leading, trailing, or diverging from the strongest public approaches.

**Why this priority**: The project is trying to build an unconventional agent
architecture, so local parity alone is not enough; the audit must position it
relative to the best public work.

**Independent Test**: Inspect the report sections and verify that each major
functional area includes external primary-source references and an explicit
comparison to current public approaches.

**Acceptance Scenarios**:

1. **Given** a major functional area such as long-term memory or autonomous
   background reasoning, **When** the audit compares it to current practice,
   **Then** it cites relevant primary sources and explains the parity gap.
2. **Given** no stable state-of-the-art consensus exists for a claimed
   capability, **When** the audit covers that area, **Then** it states that the
   area is exploratory and explains what current best practice actually is.

### Edge Cases

- What happens when a capability is implemented in tests or internal APIs but
  not surfaced through `llama-server` or other end-user entry points?
- What happens when local specs promise a capability that the code or tests do
  not yet realize?
- What happens when Vicuña intentionally diverges from public best practice,
  such as using LoRA memory instead of external retrieval memory?
- What happens when external literature does not provide a settled best method
  for “selfhood,” “endogenous thought,” or recursive self-improvement?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The audit MUST identify the current implemented functionality for
  the memory cascade, self-state, active loop, DMN, counterfactual ladder,
  governance, repair signaling, and hard-memory integration.
- **FR-002**: The audit MUST distinguish between code that exists, code that is
  tested, code that is wired into serving behavior, and functionality that
  exists only in specifications or the whitepaper.
- **FR-003**: The audit MUST describe the expected behavior of the application
  as a crude RSI “exotic intelligence” using the whitepaper as the behavioral
  reference frame.
- **FR-004**: The audit MUST compare each major functional area to current
  state-of-the-art public implementations or research using primary sources.
- **FR-005**: The audit MUST explicitly call out where the current system relies
  on heuristic CPU-side control logic rather than learned, adaptive, or
  production-proven mechanisms.
- **FR-006**: The audit MUST include a parity judgment for each major area,
  indicating whether the implementation is absent, scaffolded, partial,
  operational but crude, or near the intended design.
- **FR-007**: The audit MUST include concrete repository references for each
  local implementation claim and external links for each state-of-the-art
  comparison.
- **FR-008**: The audit MUST describe major risks, missing integrations, and
  likely behavioral consequences for the current application.
- **FR-009**: The audit MUST be delivered as a durable repository artifact, not
  only as ephemeral terminal output.

### Key Entities *(include if feature involves data)*

- **Audit Report**: A comprehensive markdown document that inventories
  functionality, expected behavior, parity with the whitepaper, and parity with
  current public state of the art.
- **Functionality Area**: One architectural slice of the system, such as
  Active LoRA memory, frozen temporal memory, self-state, DMN routing, or
  hard-memory retrieval.
- **Parity Assessment**: A structured judgment describing how closely a
  functionality area matches whitepaper intent and current state-of-the-art
  practice.
- **Evidence Set**: The set of local code references, tests, GitHub history,
  and external primary sources used to justify each audit claim.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The delivered report covers all major whitepaper pillars and ties
  each one to current repository evidence.
- **SC-002**: Every substantial local functionality claim in the report has at
  least one concrete file or test reference.
- **SC-003**: Every substantial external comparison in the report has at least
  one primary-source citation or official project reference.
- **SC-004**: A reader can use the report to separate implemented behavior from
  speculative roadmap behavior without needing to reverse-engineer the codebase.

## Assumptions

- The “original whitepaper” for this audit is
  `/Users/tyleraraujo/vicuna/Vicuña_WP.md`.
- A documentation/report artifact counts as the implementation deliverable for
  this work request.
- Current state of the art should be interpreted using public research papers,
  official docs, and leading open implementations available as of March 11,
  2026.
