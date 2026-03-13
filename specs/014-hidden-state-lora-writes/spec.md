# Feature Specification: Hidden-State-Derived Active LoRA Embeddings And Feature-Derived Write Rules

**Feature Branch**: `014-hidden-state-lora-writes`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "research and architect an implementation approach for changing the embedder and write rule. use hidden-state derived embeddings of whatever underlying base model the system is using. investigate feature-derived write directions and explore the ideas related to AdaLora, QLoRA and DoRA that might be relevant here. the goal of the system is to form durable behavior/style biases with respect to the goals of pro-social engagement with the user and desirable self-state, but we will expand on that aspect of it later. ensure that everything related to ablation and layering/updates remains in parity"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Hidden-State Embedder Architecture (Priority: P1)

As a Vicuña runtime architect, I need Active LoRA admission and redundancy
scoring to use hidden-state-derived representations from the same base model
that is currently serving, so the memory writer stops depending on hash or
token-pool heuristics and instead groups evicted spans by the model's own
representation space.

**Why this priority**: The current Active LoRA path is bottlenecked more by the
quality of the write signal than by rank alone. Without a stronger embedder,
increasing adapter capacity is likely to preserve more noise rather than better
behavioral bias.

**Independent Test**: Review the design artifacts and verify that they specify
how to extract hidden-state-derived span embeddings from the serving model,
which APIs or internal hooks are used, what fallback behavior exists, and how
this preserves current Active/Past LoRA layering semantics.

**Acceptance Scenarios**:

1. **Given** the current Active LoRA path, **When** the design is complete,
   **Then** it defines a default hidden-state embedder that derives span
   representations from the same base model family that is serving the request.
2. **Given** a model or backend configuration that cannot expose the preferred
   hidden-state surface cheaply, **When** the design is inspected, **Then** it
   defines an explicit fallback path and its observability and cost tradeoffs.

---

### User Story 2 - Feature-Derived Write Directions (Priority: P1)

As a Vicuña architect, I need the Active LoRA writer to derive update
directions from hidden states plus typed self-state features rather than from
raw token hashes or modulo-mapped token IDs, so the runtime can form durable
behavior and style biases that are aligned with goals, social engagement, and
desirable self-state.

**Why this priority**: The memory writer is supposed to create inference-level
biases, not just compress token identity. That requires a write rule tied to
content, goals, social state, and system condition.

**Independent Test**: Review the design artifacts and verify that they define a
feature pipeline, write-direction construction rule, gain policy, and update
path for both ordinary ingestion and remediation while preserving the current
direction/gain representation.

**Acceptance Scenarios**:

1. **Given** an evicted span and current self-state, **When** the design is
   reviewed, **Then** it defines how hidden-state content signals and typed
   runtime features combine into the write direction.
2. **Given** the current remediation and condensation paths, **When** the
   design is reviewed, **Then** it preserves parity between normal Active LoRA
   writes, remediation writes, and past-bucket recompression semantics.

---

### User Story 3 - Rank, Magnitude, And Quantization Policy (Priority: P2)

As a runtime operator, I need a clear plan for which AdaLoRA, DoRA, and QLoRA
ideas should or should not be adopted in Vicuña's runtime memory path, so the
system can improve expressivity and efficiency without breaking current serving
order, ablation behavior, or temporal bucket behavior.

**Why this priority**: The user explicitly asked whether "more dimensions" or a
different write path is the better move. This design must answer that with a
concrete architecture rather than a vague recommendation.

**Independent Test**: Review the architecture output and verify that it names
the relevant external techniques, translates them into Vicuña-specific design
choices, and explicitly calls out which pieces are adopted, adapted, or
rejected for runtime memory.

**Acceptance Scenarios**:

1. **Given** AdaLoRA, DoRA, and QLoRA are all relevant PEFT techniques,
   **When** the design is reviewed, **Then** it specifies which parts fit
   Vicuña's runtime memory path and which parts remain offline-training-only or
   out of scope.
2. **Given** current serving order and counterfactual LoRA ablation behavior,
   **When** the design is reviewed, **Then** it preserves those invariants and
   explains how the new writer remains compatible with them.

### Edge Cases

- What happens when the serving model can expose only final output embeddings
  and not intermediate late-layer states?
- What happens when the hidden-state extractor would require replaying long
  evicted spans and the added cost exceeds the memory/write budget?
- How should the design behave when the base model is quantized, mixed-precision,
  or running on a backend where hidden-state extraction is expensive?
- How does the new writer preserve the `LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION`
  parity if the write direction becomes feature-derived rather than token-derived?
- How does the design preserve the existing serving stack order
  `request -> all_time -> year -> quarter -> month -> week -> active` and the
  current temporal condensation semantics?
- What happens when hidden-state-derived content suggests one bias but the
  typed self-state surfaces suggest the opposite, such as a strongly negative
  span that should still be steered toward pro-social repair?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The design MUST define a hidden-state-derived embedding strategy
  for Active LoRA admission using the same base model family that is serving
  the runtime.
- **FR-002**: The design MUST preserve the swappable embedder interface so the
  hidden-state embedder becomes a default implementation rather than a hard
  architectural lock-in.
- **FR-003**: The design MUST define a feature-derived write rule that combines
  hidden-state content signals with typed runtime features relevant to goals,
  social engagement, and desirable self-state.
- **FR-004**: The design MUST preserve the existing separation between
  direction and gain, and MUST define how the new writer updates both surfaces.
- **FR-005**: The design MUST preserve base-model immutability and continue to
  express runtime memory through bounded runtime-managed LoRA artifacts only.
- **FR-006**: The design MUST preserve the current serving-layer order and
  bucket roles for Active and past LoRAs.
- **FR-007**: The design MUST preserve parity for counterfactual LoRA ablation,
  remediation writes, and temporal condensation/recompression.
- **FR-008**: The design MUST specify what layer groups the new writer targets,
  how layer sensitivity is scored, and whether any adaptive budget allocation
  is used.
- **FR-009**: The design MUST explicitly evaluate AdaLoRA, DoRA, and QLoRA and
  state which concepts are adopted, adapted, or rejected for this runtime path.
- **FR-010**: The design MUST define observability for embedder identity,
  hidden-state extraction mode, write-feature summaries, per-layer allocation,
  gain statistics, and fallback behavior.
- **FR-011**: The design MUST include a staged rollout path that starts from
  the least invasive hidden-state extractor compatible with the current codebase.
- **FR-012**: The design MUST provide a concrete implementation task breakdown
  covering public API changes, runtime internals, tests, and documentation.
- **FR-013**: The design MUST be delivered as durable repository artifacts
  rather than only terminal output.

### Key Entities

- **Hidden-State Span Embedding**: A normalized span representation derived from
  the serving model's own hidden-state or embedding outputs for an evicted span.
- **Write Feature Vector**: A typed feature bundle combining hidden-state
  content, self-state signals, event metadata, and goal/social context for
  writer decisions.
- **Layer Sensitivity Profile**: The inspectable policy state that decides which
  target layers or layer groups receive more or less low-rank budget.
- **Write Direction Policy**: The rule that converts hidden-state and typed
  features into bounded low-rank direction updates.
- **Magnitude/Gain Policy**: The rule that controls update strength separately
  from update direction.
- **Parity Invariant**: A serving, ablation, layering, or temporal-memory
  behavior that must remain unchanged while the embedder and write rule evolve.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The delivered design identifies a default hidden-state extraction
  path that is implementable against the current Vicuña codebase without
  breaking serving-order or Active/Past LoRA invariants.
- **SC-002**: The delivered design defines at least one concrete feature-derived
  write rule that is materially richer than the current hash/token-modulo path.
- **SC-003**: The delivered design states explicit adopt/adapt/reject decisions
  for AdaLoRA, DoRA, and QLoRA in the context of runtime memory.
- **SC-004**: A maintainer can use the design artifacts to implement the new
  embedder and write rule while preserving current ablation and layering parity
  without reverse-engineering the intent from terminal output.

## Assumptions

- This request asks for research and implementation architecture, not for
  immediate code changes to the runtime.
- The later productization of "pro-social engagement" and "desirable
  self-state" policy can remain partially abstract for now, as long as the
  writer design leaves explicit typed hooks for those signals.
- A staged implementation that uses currently available embedding outputs first
  and deeper late-layer taps later is acceptable if the design makes the phases
  and tradeoffs explicit.
