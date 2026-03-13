# Feature Specification: Self-Model Expansion For Efficient Goal Pursuit, User Satisfaction, And Self-Repair

**Feature Branch**: `015-self-model-expansion`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "based on items 2-3 of your audit report, it seems prudent to expand the self-model in robust and useful ways. my intuition about this is that we need to expand the number and complexity of registers. do your research into what else we could represent, and how, that would be beneficial to continually motivating the system for self-improvement toward our goals; achieving user satisfaction and goals, and returning from undesirable to desireable self-state, all increasingly more efficiently(measured in terms of taking fewer steps in the agent loop/less inference). we will build systems that evaluate the system with respect to these goals later"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Richer Self-State Taxonomy (Priority: P1)

As a Vicuña runtime architect, I need the self-model to represent more than
local contradiction, uncertainty, and social pressure, so the runtime can track
goal progress, user outcome risk, epistemic condition, efficiency, and
self-repair readiness in an explicit inspectable form.

**Why this priority**: The audit found that the current self-state is real but
shallow. If the system is meant to motivate self-improvement, it needs richer
internal estimates of what is going well, what is failing, and what matters
next.

**Independent Test**: A maintainer can read the design artifacts and identify a
concrete proposed taxonomy of new self-representation surfaces, including what
each surface means, how it is updated, and why it helps user satisfaction,
goal completion, or recovery.

**Acceptance Scenarios**:

1. **Given** the current register bank in `include/llama.h`, **When** the
   design is reviewed, **Then** it defines additional self-model families for
   goal progress, user outcome, epistemic control, efficiency, recovery, and
   self-improvement governance.
2. **Given** the whitepaper's requirement for a mathematical self-core,
   **When** the design is reviewed, **Then** it expands the self-model with
   typed and bounded state rather than replacing the explicit bank with opaque
   latent state.

---

### User Story 2 - Multi-Timescale And Forecast State (Priority: P1)

As a Vicuña architect, I need the self-model to track not only current scalar
pressure but also short-horizon trends, long-horizon baselines, and explicit
forecast signals, so the runtime can pursue fewer-step behavior and faster
recovery rather than only reacting to the immediate turn.

**Why this priority**: Efficiency-oriented self-improvement requires estimates
of progress velocity, expected remaining work, and recovery trajectory. Single
step scalar pressure is not enough.

**Independent Test**: A maintainer can read the design artifacts and identify
how instantaneous, short-horizon, and persistent self-estimates coexist, how
prediction error is recorded, and how those surfaces feed existing favorable,
counterfactual, and governance logic.

**Acceptance Scenarios**:

1. **Given** the current favorable-state and DMN traces, **When** the design is
   reviewed, **Then** it defines a multi-timescale self-model that preserves
   current loop behavior while adding trend and forecast surfaces.
2. **Given** the user goal of fewer agent-loop steps and less inference,
   **When** the design is reviewed, **Then** it includes explicit expected
   steps remaining, expected inference cost, and loop inefficiency signals.

---

### User Story 3 - Explicit Motivation For Safe Self-Improvement (Priority: P2)

As a runtime architect, I need explicit surfaces that estimate whether a
self-modification, self-repair, or strategic mode shift is worth attempting, so
the system can later be evaluated and optimized for when it should spend effort
on self-improvement versus immediate user-facing action.

**Why this priority**: The user wants a system that continually improves toward
user satisfaction, goal completion, and favorable self-state. That requires an
explicit internal representation of "improvement opportunity" and "recovery
need", not only local response pressure.

**Independent Test**: A maintainer can read the design artifacts and identify a
bounded representation of update worthiness, evidence deficit, reversibility,
and expected gain without changing current layering, ablation, or update parity.

**Acceptance Scenarios**:

1. **Given** current governance and remediation traces, **When** the design is
   reviewed, **Then** it defines self-improvement motivation surfaces that can
   plug into those traces later.
2. **Given** the requirement to keep runtime policy explicit, **When** the
   design is reviewed, **Then** it encodes self-improvement motivation as typed
   bounded state and inspectable CPU-side policy rather than opaque heuristics.

### Edge Cases

- How should the design avoid turning the self-model into an unstructured list
  of dozens of unrelated scalars?
- How should the runtime represent user satisfaction risk when there is no
  labeled evaluator yet and only local interaction evidence is available?
- What happens when short-horizon signals indicate failure but long-horizon
  trends still look healthy, or vice versa?
- How should the design represent uncertainty about its own self-estimates so
  later evaluators can calibrate or override them?
- How should new self-model fields preserve parity with current prewrite,
  postwrite, favorable-state, counterfactual, and remediation update ordering?
- What happens when a proposed self-improvement action could reduce long-term
  loop cost but would likely degrade immediate user satisfaction?
- How should the design distinguish "need to ask a clarifying question" from
  "need to improve internal state" from "need to act with tools"?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The design MUST define an expanded self-model taxonomy that
  preserves the current explicit register bank while adding richer typed
  representations for goal progress, user outcome, epistemic control,
  efficiency, recovery, strategy, and self-improvement governance.
- **FR-002**: The design MUST keep runtime policy explicit and inspectable in
  CPU-side control code rather than delegating self-motivation to hidden latent
  state alone.
- **FR-003**: The design MUST distinguish instantaneous state, short-horizon
  trend state, and long-horizon baseline state.
- **FR-004**: The design MUST define forecast-oriented self-estimates including
  expected steps remaining, expected inference cost remaining, and estimated
  recovery cost to favorable self-state.
- **FR-005**: The design MUST define user-outcome surfaces beyond current trust
  and dissatisfaction, including at minimum satisfaction estimate, frustration
  risk, misunderstanding risk, and preference uncertainty.
- **FR-006**: The design MUST define goal-progress surfaces including at
  minimum progress estimate, blocker severity, dependency readiness, and
  expected gain from the next action.
- **FR-007**: The design MUST define epistemic-control surfaces including at
  minimum answerability, evidence sufficiency, self-estimate confidence, and
  ambiguity concentration.
- **FR-008**: The design MUST define recovery and homeostatic surfaces including
  at minimum favorable-state divergence by family, recovery momentum,
  regulation debt, and unresolved tension load.
- **FR-009**: The design MUST define efficiency-oriented surfaces including at
  minimum loop inefficiency, repetition risk, context-thrash risk, and tool
  round-trip cost estimate.
- **FR-010**: The design MUST define explicit self-improvement motivation
  surfaces including at minimum update worthiness, expected gain, evidence
  deficit, and reversibility or blast-radius estimate.
- **FR-011**: The design MUST preserve parity with the current prewrite and
  postwrite update phases and with current favorable-state, counterfactual, and
  remediation traces.
- **FR-012**: The design MUST state how new self-model surfaces should be
  layered relative to existing bounded scalar registers rather than simply
  appending all new state to a single flat enum.
- **FR-013**: The design MUST recommend how to expose the expanded self-model
  through public API structs, traces, tests, and docs.
- **FR-014**: The design MUST be implementable incrementally without breaking
  current Active Loop, DMN, self-state replay, or self-state import/export
  invariants.
- **FR-015**: The design MUST be delivered as durable repository artifacts
  rather than terminal-only advice.

### Key Entities

- **Fast Control Register Bank**: The existing bounded scalar and categorical
  bank used for low-latency loop decisions.
- **Typed Self Profile**: A grouped state object that represents one coherent
  domain such as goal progress, user outcome, or recovery.
- **Horizon Slice**: A snapshot of a profile at one timescale, such as
  instantaneous, short-horizon EMA, or persistent baseline.
- **Forecast Estimate**: An inspectable prediction about future work or outcome,
  such as expected remaining steps or recovery cost.
- **Prediction Error Trace**: A bounded record comparing prior forecast against
  observed outcome so later evaluators can calibrate the self-model.
- **Motivation Surface**: A bounded estimate of urgency, value density, or
  update worthiness that can eventually drive self-improvement behavior.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The delivered design proposes a richer self-model that materially
  expands beyond the current register bank while preserving explicit bounded
  runtime state.
- **SC-002**: The delivered design defines a multi-timescale representation and
  at least one concrete forecast-oriented path for efficiency and recovery.
- **SC-003**: The delivered design specifies how user satisfaction, goal
  progress, and favorable self-state can all be represented simultaneously
  without collapsing into one scalar reward proxy.
- **SC-004**: A maintainer can use the artifacts to implement the self-model
  expansion incrementally while preserving current loop and update parity.

## Assumptions

- This request asks for research and implementation architecture, not immediate
  runtime code changes.
- Later evaluation systems can remain future work as long as the new self-model
  exposes the right inspectable surfaces now.
- The expanded self-model should support, not replace, the existing favorable
  state, counterfactual, remediation, and Active/DMN runtime machinery.
