# Feature Specification: Bounded Tool-Loop Scaffolding For Active And DMN Runtime Control

**Feature Branch**: `016-react-loop-scaffolding`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "with regard to sections 4-5, it seems like the solution is to implement a version of a standard ReAct long-running loop for both the active loop and DMN. we will and should extend the tools the agent has to use to respond to active user enquiries and remediate/simulate in the DMN, for now we need to make sure that the scaffholding for the insertion of these tools is prepared to receive them. research and investigate how to implement these loops or if there is a better option, tailored for the intent of the active loop and DMN respectively, then implement the changes outlined by the results of your research and spec development."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Shared Long-Running Loop Substrate (Priority: P1)

As a Vicuña runtime architect, I need a typed bounded deliberation substrate
for multi-step act/observe cycles, so new tools can be inserted later without
rewriting the active loop or DMN surfaces again.

**Why this priority**: The current loops already score actions explicitly, but
they do not expose durable episode, phase, tool-plan, or observation state.
That is the missing scaffold for future tool growth.

**Independent Test**: A maintainer can inspect public API structs and runtime
code and see explicit loop phase, step budget, tool proposal, and observation
surfaces shared by active and DMN execution.

**Acceptance Scenarios**:

1. **Given** the current `ANSWER/ASK/ACT/WAIT` and DMN action scoring, **When**
   the implementation is reviewed, **Then** it adds a shared bounded loop-step
   model without removing the current candidate-ranking behavior.
2. **Given** future tool additions, **When** new tool kinds are introduced,
   **Then** the runtime already has inspectable registry metadata and proposal
   structs for loop eligibility, latency class, simulation safety, and
   remediation safety.

---

### User Story 2 - Distinct Active And DMN Policies (Priority: P1)

As a Vicuña architect, I need the active loop and DMN to use different
policies on top of the shared substrate, so foreground user response and
background remediation are not forced into the same generic ReAct behavior.

**Why this priority**: Current public agent systems separate foreground and
background work. Vicuña also already distinguishes user-facing and maintenance
control, so the new substrate should preserve that difference.

**Independent Test**: A maintainer can inspect the traces and see different
phase budgets, stop conditions, and tool eligibility for active and DMN loops.

**Acceptance Scenarios**:

1. **Given** an active user event, **When** the foreground loop runs, **Then**
   the trace records a latency-sensitive episode with a small step budget and a
   tool proposal path suitable for direct user support.
2. **Given** a DMN tick, **When** the background loop runs, **Then** the trace
   records maintenance-oriented episode state with explicit remediation,
   simulation, and governance-oriented tool planning surfaces.

---

### User Story 3 - Parity-Preserving Tool Insertion Points (Priority: P2)

As a maintainer, I need the new loop scaffolding to preserve current memory,
ablation, remediation, and action parity, so future tool insertion does not
regress existing favorable-state, counterfactual, or Active/Past LoRA update
behavior.

**Why this priority**: The repo already has strong typed parity requirements.
The scaffold must prepare for tools without disturbing current loop semantics.

**Independent Test**: Existing cognitive-loop tests still pass, and new tests
show that active and DMN traces now surface tool plans and step metadata while
maintaining current winner actions.

**Acceptance Scenarios**:

1. **Given** current Active Loop and DMN routing cases, **When** the new
   scaffold is enabled, **Then** winner actions remain behaviorally consistent
   while additional loop metadata becomes visible.
2. **Given** remediation, governance, and `LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION`,
   **When** the DMN loop runs, **Then** the new scaffold preserves their update
   ordering and only adds bounded planning metadata around them.

### Edge Cases

- What happens when no tool is registered for the winning action path?
- How should the active loop expose an `ACT` winner when the tool plan is still
  provisional and execution has not started?
- How should the DMN represent simulation-safe versus externally mutating tools?
- What happens when governance denies a DMN tool proposal after the loop
  substrate has already assembled a candidate?
- How should the runtime distinguish a terminal user-facing answer from a
  continue-observe loop step?
- How should the scaffold avoid turning a bounded explicit runtime into an
  opaque unconstrained prompt loop?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The implementation MUST add a shared typed loop substrate for
  deliberation phases, step budgets, continuation state, and tool proposals.
- **FR-002**: The implementation MUST preserve the current explicit active-loop
  and DMN candidate scoring logic as the top-level policy surface.
- **FR-003**: The implementation MUST define a public tool-registry metadata
  surface that records at minimum tool kind, active/DMN eligibility, latency
  class, simulation safety, and remediation safety.
- **FR-004**: The implementation MUST define bounded public structs for active
  and DMN loop episode state, including phase, step budget, steps completed,
  continuation allowance, and terminal reason.
- **FR-005**: The implementation MUST define bounded public structs for tool
  proposals and tool observations, even if the runtime initially only fills a
  subset of the fields.
- **FR-006**: The implementation MUST preserve current active and DMN action
  enums and winner selection behavior.
- **FR-007**: The implementation MUST give the active loop a latency-oriented
  small-step policy and the DMN a maintenance/remediation-oriented policy.
- **FR-008**: The implementation MUST preserve current remediation, governance,
  favorable-state, counterfactual, and Active/Past LoRA update ordering.
- **FR-009**: The implementation MUST make future tool insertion possible
  without changing trace struct shapes again.
- **FR-010**: The implementation MUST add regression coverage for the new loop
  scaffolding surfaces.
- **FR-011**: The implementation MUST update architecture-facing docs to
  reflect that Vicuña uses bounded loop state machines, not free-form generic
  ReAct everywhere.

### Key Entities

- **Loop Tool Spec**: Public metadata describing one tool kind's role,
  eligibility, latency, and safety properties.
- **Loop Episode State**: The bounded state of a single foreground or
  background run, including phase, budget, and stop condition.
- **Loop Tool Proposal**: A typed summary of the current intended tool action,
  including kind, plan family, and safety affordances.
- **Loop Observation**: A typed summary of the most recent tool or internal
  observation that can feed the next step.
- **Shared Tool-Loop Substrate**: Common scaffolding used by both active and
  DMN loops while still allowing different control policies.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A maintainer can inspect `include/llama.h` and find explicit
  loop-phase, tool-spec, episode-state, and proposal/observation structs for
  both active and DMN execution.
- **SC-002**: Existing cognitive-loop winner-action tests continue to pass.
- **SC-003**: New regression tests verify that active and DMN traces expose
  bounded multi-step scaffolding without regressing current routing parity.
- **SC-004**: Architecture docs describe the runtime as a bounded explicit
  tool-loop substrate with distinct foreground and DMN policies.

## Assumptions

- This pass prepares the runtime for later tool expansion rather than
  implementing a large new external tool suite immediately.
- The correct design is a shared explicit substrate with two policies, not a
  single generic prompt-driven ReAct loop copied into both active and DMN code.
- Current action scoring and self-state machinery should remain the primary
  policy drivers during this implementation.
