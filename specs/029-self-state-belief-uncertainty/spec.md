# Feature Specification: Partial-Observation Self-State Belief Gradient

**Feature Branch**: `[029-self-state-belief-uncertainty]`  
**Created**: 2026-03-13  
**Status**: Draft  
**Input**: User description: "we have a self-malleable agent self-state that contains an expandable variety of information, and we treat it as a gradient that adaptively modifies that gain of invoked functional loras, which are themselves tuned by their own success. i have a potentially interesting question; to what extent is it possible to treat the self-state implicitly as though it is not a complete rendering of all possible things the agent could care about? ie, \"reasoning under uncertainty\", or something similar to a markov process? do the research on if/how we could implement this for our self-state gradient and what the benefits/implications/consequences would be."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preserve Explicit Self-State While Admitting Partial Observability (Priority: P1)

As a Vicuña operator, I need the runtime to treat the self-state as an explicit but incomplete observation surface so that functional gain control can reason about uncertainty over unmodeled cares without collapsing into opaque latent memory.

**Why this priority**: This is the minimum ontology change that answers the request without breaking the current typed self-state, allostatic control, or inspectability guarantees.

**Independent Test**: Feed the runtime identical observed self-state registers under two different uncertainty configurations and verify that the controller keeps the same explicit gradient while producing different bounded uncertainty summaries and different gain predictions.

**Acceptance Scenarios**:

1. **Given** a typed observed self-state register bank and no latent evidence, **When** the belief layer runs, **Then** it MUST preserve the observed gradient unchanged and produce a prior-only uncertainty summary.
2. **Given** repeated unexplained post-action state shifts that are not captured by current explicit registers, **When** the belief layer updates, **Then** it MUST increase bounded latent uncertainty or residual-pressure summaries without inventing unbounded new state.

---

### User Story 2 - Let Functional Gain Respond To Uncertainty Over Missing Cares (Priority: P2)

As a Vicuña operator, I need the gating controller to account for uncertainty over incompletely modeled cares so that the system can behave more conservatively, more information-seeking, or more exploratory when the explicit self-state is likely insufficient.

**Why this priority**: The point of the change is not just modeling uncertainty, but making it operational in the existing self-state-gradient-to-gain path.

**Independent Test**: Replay the same active or DMN transaction with belief uncertainty disabled and enabled, then verify that gain prediction changes only through the new bounded belief summary inputs.

**Acceptance Scenarios**:

1. **Given** high explicit allostatic divergence and high belief confidence, **When** gains are predicted, **Then** the controller SHOULD primarily follow the explicit self-state gradient.
2. **Given** modest explicit divergence but high residual uncertainty, **When** gains are predicted, **Then** the controller SHOULD be able to increase tool-selection, planning, counterfactual, or self-observation pressure in bounded ways.

---

### User Story 3 - Keep The Belief Layer Inspectable, Bounded, And Promotable (Priority: P3)

As a Vicuña developer, I need the uncertainty layer to remain typed, bounded, inspectable, and compatible with later promotion into explicit self-model extensions so that the system does not become a hidden heuristic blob.

**Why this priority**: The architecture only remains coherent if latent uncertainty can be audited and, when justified, turned into authored or discovered explicit state rather than staying permanently opaque.

**Independent Test**: Run a belief-update cycle, inspect typed traces for observed state, latent summaries, posterior confidence, and promotion candidates, and verify that budget limits and clipping hold.

**Acceptance Scenarios**:

1. **Given** belief updates from repeated forecast-error residue, **When** a latent concern repeatedly aligns with a stable hard-memory-backed pattern, **Then** the runtime SHOULD be able to surface it as a promotion candidate for the self-model extension registry rather than silently hiding it.
2. **Given** runaway noise, contradictory evidence, or no useful evidence, **When** the belief layer updates, **Then** the runtime MUST clip, decay, or reset the affected belief summaries and expose the failure through traceable stats.

## Edge Cases

- What happens when explicit self-state and latent residual disagree strongly for many steps?
- How does the runtime avoid hallucinating new cares from one noisy user turn or one bad counterfactual?
- What happens when hidden concern pressure remains high but there is no clear promotion target in hard memory or tool-authored state?
- How does the controller behave when belief uncertainty is high but the active loop must still answer immediately?
- What happens when the belief update path is disabled, unavailable, or fails validation on a backend-sensitive build?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime MUST treat the current typed self-state register and extension surface as the observed portion of agent state rather than as a claim of complete coverage over everything the agent may care about.
- **FR-002**: The runtime MUST maintain a bounded belief layer that summarizes uncertainty over incompletely modeled or currently unrendered cares using fixed-size typed state rather than free-form latent text.
- **FR-003**: The belief layer MUST separate at least:
  - observed explicit self-state gradient,
  - residual unexplained state shift,
  - posterior confidence or entropy,
  - latent allostatic pressure estimate,
  - and promotion candidates for explicit self-model extension.
- **FR-004**: The belief layer MUST update only from explicit bounded evidence sources such as forecast error, counterfactual miss, tool outcome mismatch, user-response mismatch, hard-memory retrieval residue, and unexplained self-state deltas.
- **FR-005**: The functional gating MLP MUST consume only a fixed belief-summary surface in addition to the existing self-state gradient and auxiliary control features; it MUST NOT consume variable-length latent memory directly.
- **FR-006**: The runtime MUST preserve backward compatibility for the existing self-state gradient semantics so that explicit allostatic dimensions remain the primary interpretable control signal.
- **FR-007**: The belief layer MUST keep its influence bounded through explicit clipping, decay, confidence floors, and update-rate limits.
- **FR-008**: The runtime MUST distinguish between:
  - uncertainty over known cares,
  - uncertainty caused by missing observations,
  - and evidence for a genuinely new explicit care dimension.
- **FR-009**: The runtime MUST NOT let latent concern estimates directly redefine the ideal self-state or allostatic objective until they are promoted into typed explicit state through an inspectable promotion path.
- **FR-010**: The runtime SHOULD support conservative or information-seeking gain adjustments when explicit self-state is low-confidence or demonstrably incomplete.
- **FR-011**: The runtime MUST expose belief-layer stats, traces, and promotion-candidate summaries through typed public or cross-module interfaces.
- **FR-012**: The design MUST preserve base-model immutability and use only bounded runtime-managed state.
- **FR-013**: The implementation MUST ship with targeted tests covering confidence decay, residual-pressure accumulation, gain-input compatibility, and promotion-candidate stability.
- **FR-014**: Architecture and operator documentation MUST explain the distinction between observed self-state, belief summary, and explicit self-model promotion.

### Key Entities *(include if feature involves data)*

- **Observed Self-State**: The existing typed register bank, horizon profiles, and extension summary that directly represent explicit modeled cares and allostatic dimensions.
- **Belief Summary**: A bounded typed summary of uncertainty over incompletely modeled cares, including residual pressure, posterior confidence, unexplained-delta mass, and promotion readiness.
- **Latent Concern Slot**: A fixed-size runtime slot tracking a hypothesized hidden concern class using bounded numeric summaries rather than free-form text.
- **Promotion Candidate**: A typed proposal connecting repeated latent residue to a future explicit self-model extension or hard-memory-backed scalar.
- **Belief Update Evidence**: The bounded tuple of pre/post self-state, predicted outcome, realized outcome, retrieved memory support, and counterfactual miss used to update the belief summary.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In targeted tests, the belief layer produces deterministic bounded summaries from fixed evidence and never exceeds declared numeric ranges or slot budgets.
- **SC-002**: In replay tests, enabling the belief layer changes functional gain predictions only through documented fixed-width input features and never through raw variable-length latent content.
- **SC-003**: In synthetic hidden-pressure scenarios, the runtime surfaces increased residual-pressure or uncertainty scores before any explicit self-model promotion occurs.
- **SC-004**: In promotion tests, repeated stable latent residue can be surfaced as a typed promotion candidate without mutating the explicit self-model automatically.
- **SC-005**: Documentation and traces let a developer explain why the controller behaved cautiously, exploratory, or information-seeking under partial observability without reading implementation internals.
