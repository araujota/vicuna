# Feature Specification: Self-Model Extension Registry

**Feature Branch**: `025-self-model-extension-registry`  
**Created**: 2026-03-13  
**Status**: Draft  
**Input**: User description: "the self-model should remain somewhat authored as a sort of genetic core of the system, but the self-model should be directly extensible by the system, primarily through counterfactual simulations of the type, 'would i have behaved better if my self-model contained a representation of this hard memory state'. the system should also support self-model extensibility via the addition of tools that explicitly add things to the self-model. this marriage of 'semi-authored' vs discovery-based self-model expansion seems like the best way forward. all perturbations to the self-model in this way must remain backward-compatible with the self-model gradient's application as a gain modifier. additions to the self-model that are essentially representations of hard-memory values should probably not affect allostasis, but self-model registers/params from tools should have this ability, to essentially come with a 'desirable state'. the only 'tools' the system is shipping with by default are its hard memory query system, and its CLI wrapper, and i do not intend to change this for the bare minimum version. but there should be adequate documentation on extending this tool system."

## User Scenarios & Testing

### User Story 1 - Authored Core With Runtime Extensions (Priority: P1)

As a runtime architect, I need the existing authored self-model to remain the
stable genetic core while allowing bounded runtime extensions, so Vicuña can
discover or accept new self-representations without losing explicit control
policy.

**Why this priority**: This is the ontology change. If the extension layer is
not typed, bounded, and clearly subordinate to the authored core, the self-model
becomes opaque and unsafe.

**Independent Test**: A maintainer can add, inspect, and remove bounded
self-model extensions through public APIs without breaking existing self-model
state or gain-control behavior.

**Acceptance Scenarios**:

1. **Given** an initialized context, **When** a maintainer upserts typed
   self-model extensions, **Then** the authored horizon/profile core remains
   present and the new extensions are exposed through typed inspection APIs.
2. **Given** runtime-discovered or tool-authored extensions, **When** the
   self-model is recomputed, **Then** extension state remains bounded, typed,
   and attributable by source and policy flags.

---

### User Story 2 - Counterfactual Hard-Memory Promotion (Priority: P1)

As Vicuña, I need hard-memory query results to be eligible for counterfactual
promotion into the self-model, so the system can learn that representing some
retrieved memory inside the self-model would improve behavior.

**Why this priority**: This is the main discovery mechanism the user asked for.
Without it, self-model expansion remains manual and tool-driven only.

**Independent Test**: A maintainer can run a hard-memory query and observe that
beneficial retrieved memories become typed self-model extensions through a
bounded counterfactual selection step.

**Acceptance Scenarios**:

1. **Given** a hard-memory query with relevant hits, **When** the runtime
   evaluates whether representing a hit in self-model would help, **Then** the
   best-scoring candidates can be promoted into bounded extension slots.
2. **Given** promoted hard-memory representations, **When** they are applied to
   self-model recomputation, **Then** they influence control surfaces without
   contributing to allostatic divergence by default.

---

### User Story 3 - Tool-Authored Self-Model Parameters (Priority: P1)

As a tool integrator, I need tools to add typed self-model parameters with
optional desirable-state metadata, so tools can write inspectable internal state
that affects gain control and, when appropriate, allostasis.

**Why this priority**: The default shipped tools are limited, but the extension
contract needs to be production-ready now.

**Independent Test**: A maintainer can submit a tool-authored self-model
parameter with a desired state and observe that it contributes to extension
summaries and allostatic pressure according to policy flags.

**Acceptance Scenarios**:

1. **Given** a tool-authored scalar self-model parameter with a desired state,
   **When** it is upserted, **Then** it can contribute to both gain summaries
   and allostatic divergence according to typed flags.
2. **Given** a tool-authored context or memory-like extension without a desired
   state, **When** it is upserted, **Then** it can still influence gain control
   without being treated as an allostatic objective.

### Edge Cases

- What happens when extension capacity is full and a better candidate arrives?
- How should hard-memory-derived extensions decay or be replaced when they stop
  being relevant?
- How should the runtime prevent hard-memory-derived context representations from
  accidentally becoming allostatic objectives?
- What happens when a tool submits invalid desired-state metadata, invalid
  domains, or NaN values?
- How should the gating MLP remain backward-compatible when the number of active
  extensions changes over time?
- How should extension effects remain bounded so tool-authored state cannot
  silently dominate the authored self-model?

## Requirements

### Functional Requirements

- **FR-001**: The runtime MUST preserve the existing authored self-model
  profiles and horizons as the stable core.
- **FR-002**: The runtime MUST add a bounded typed self-model extension
  registry that stores runtime-discovered and tool-authored additions.
- **FR-003**: Each self-model extension MUST record source provenance,
  semantic domain, current value, confidence, salience, and policy flags.
- **FR-004**: Tool-authored scalar self-model extensions MUST support optional
  desirable-state metadata and explicit allostatic participation flags.
- **FR-005**: Hard-memory-derived self-model extensions MUST be discoverable via
  bounded counterfactual evaluation of hard-memory query results.
- **FR-006**: Hard-memory-derived extensions MUST default to affecting gain
  control without contributing to allostatic divergence.
- **FR-007**: The extension registry MUST expose typed inspection APIs for
  counts, per-extension details, and summarized effect surfaces.
- **FR-008**: The self-model recomputation path MUST apply extension effects in
  an explicit CPU-side function that remains bounded and inspectable.
- **FR-009**: The functional-gating input path MUST remain compatible with the
  existing self-state-gradient-as-gain-controller approach by consuming a fixed
  bounded summary of extension effects rather than an unbounded raw list.
- **FR-010**: The runtime MUST reject invalid extension writes and MUST clamp
  values and weights into bounded ranges.
- **FR-011**: The runtime MUST provide an explicit public API path for tool or
  host code to upsert and remove self-model extensions without requiring direct
  struct mutation.
- **FR-012**: The default hard-memory query tool MUST integrate with extension
  discovery in the bare-minimum version.
- **FR-013**: The bare-minimum version MUST document how tool integrations
  should author self-model extensions accurately and safely.
- **FR-014**: The change MUST ship with targeted regression tests for extension
  CRUD, hard-memory promotion, extension summaries, and allostatic-flag
  behavior.
- **FR-015**: The change MUST update architecture and operator/developer docs
  alongside code.

### Key Entities

- **Authored Self-Model Core**: The existing typed register bank plus
  goal/user/epistemic/efficiency/recovery/strategy/self-improvement profiles.
- **Self-Model Extension**: A bounded runtime-added item that augments the
  self-model while remaining inspectable and typed.
- **Extension Summary**: A fixed-size derived summary of extension pressure,
  confidence, activation, and allostatic divergence used by control code.
- **Counterfactual Promotion Trace**: A typed record of whether a hard-memory
  result would have improved self-model state enough to justify promotion.
- **Tool-Authored Parameter Extension**: A typed scalar extension that may
  optionally declare a desirable state and allostatic relevance.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Maintainers can upsert, inspect, and remove self-model extensions
  through stable typed APIs.
- **SC-002**: Hard-memory queries can promote at least one bounded
  counterfactually justified self-model extension in targeted tests.
- **SC-003**: Hard-memory-derived extensions influence extension gain summaries
  without increasing extension allostatic divergence by default.
- **SC-004**: Tool-authored extensions with desirable-state metadata can change
  extension allostatic divergence in targeted tests.
- **SC-005**: Functional-gating and self-model tests pass with extension-aware
  summaries and without regressions to the authored core.

## Assumptions

- The authored self-model remains the primary control ontology; runtime
  extensions are additive and bounded.
- Bare-minimum default tool support means hard-memory discovery is integrated
  directly, while the CLI wrapper receives a general-purpose host API rather
  than a new model-authored schema.
- The first version should prefer explicit lanes/domains and bounded summaries
  over arbitrary tool-defined formulas.
