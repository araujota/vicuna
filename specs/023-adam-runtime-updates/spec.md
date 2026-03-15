# Feature Specification: Expand Adam-Based Optimization For Self-State-Driven Runtime Updates

**Feature Branch**: `023-adam-runtime-updates`
**Created**: 2026-03-13
**Status**: Draft
**Input**: User description: "would Adam be a good optimizer to use for all weight/bias updates computed from a delta in self-state in the way we use it here? i strongly suspect so. do the research and see if it would make sense to insert this optimizer in other places, like the individual tuning mechanisms for functional loras or counterfactual lora ablation tests, and if you identify strong candidate processes to optimize with Adam, insert that functionality"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Runtime LoRA Writes Use Adam (Priority: P1)

As a runtime architect, I need the self-state-driven Active LoRA and
family-specific functional LoRA writes to use Adam instead of direct additive
tensor mutation, so online low-rank updates are better conditioned under noisy,
non-stationary self-state deltas.

**Why this priority**: This is the strongest candidate process in the runtime.
These paths already compute weight deltas from self-state-derived signals and
write them directly into trainable low-rank tensors.

**Independent Test**: A maintainer can trigger Active LoRA and functional LoRA
updates, observe optimizer step counts and update norms advance, and confirm
existing runtime update behavior still succeeds.

**Acceptance Scenarios**:

1. **Given** an Active LoRA or functional LoRA write computes a delta from
   self-state-derived features, **When** the runtime applies the update,
   **Then** the low-rank tensors are updated through Adam-managed first and
   second moments rather than by direct raw addition alone.
2. **Given** repeated noisy updates target the same runtime LoRA weights,
   **When** the optimizer continues stepping, **Then** the runtime preserves
   per-parameter optimizer state and bounded update norms across writes.
3. **Given** no meaningful delta exists for a write, **When** the runtime
   processes the event, **Then** it does not fabricate an optimizer step.

---

### User Story 2 - Temporal Write-Bias Control Uses Adam (Priority: P2)

As a maintainer, I need the temporal write-bias controller to adapt with Adam
instead of fixed heuristic increments, so reward and dampening biases respond
more smoothly to noisy signed and efficiency advantages.

**Why this priority**: This path updates scalar biases directly from
post-transaction self-state deltas and is a credible second candidate, but it
is narrower in scope than the LoRA writer itself.

**Independent Test**: A maintainer can trigger temporal self-improvement and
inspect typed bias state showing Adam step progression while the effective
write scale remains bounded.

**Acceptance Scenarios**:

1. **Given** temporal self-improvement produces positive validated advantage,
   **When** the temporal controller updates, **Then** reward bias is adjusted
   through Adam and the effective write scale remains within its configured
   bounds.
2. **Given** temporal self-improvement produces negative validated advantage,
   **When** the temporal controller updates, **Then** dampening bias is
   adjusted through Adam and the effective write scale remains within its
   configured bounds.
3. **Given** the signal is weak or mixed, **When** the controller updates,
   **Then** Adam state may decay or make only small bounded changes rather than
   applying a large threshold jump.

---

### User Story 3 - Discrete Counterfactual Ranking Remains Explicit (Priority: P2)

As a runtime architect, I need the counterfactual ablation ladder to remain an
explicit discrete ranking policy rather than being forced into Adam, so the
system does not pretend a non-differentiable intervention-ranking process is a
gradient-trained parameter optimizer.

**Why this priority**: The user explicitly asked about counterfactual LoRA
ablation tests. This path must be researched and addressed, even if the right
answer is to leave it non-Adam.

**Independent Test**: A maintainer can inspect the spec, code, and tests and
see that the counterfactual ladder still ranks interventions explicitly while
Adam is only applied to genuine parameter-update paths.

**Acceptance Scenarios**:

1. **Given** the counterfactual ladder evaluates LoRA ablations and other
   interventions, **When** the runtime ranks candidates, **Then** it continues
   using explicit scored policy rather than hidden optimizer state.
2. **Given** Adam is introduced in other runtime learning paths, **When**
   maintainers inspect the architecture docs and tests, **Then** the reasons
   for not applying Adam to discrete counterfactual ranking are explicit.

### Edge Cases

- What happens when the same runtime LoRA tensor is updated intermittently and
  Adam moments must persist across sparse online writes?
- What happens when a functional family settles after a delayed hold window and
  its write signal is much smaller than earlier updates?
- What happens when temporal reward and dampening signals alternate signs over
  short windows?
- What happens when optimizer state exists for a runtime adapter that is
  currently inactive or ablated?
- What happens when the counterfactual ladder proposes LoRA ablation but no
  differentiable parameter update follows from that ranking event?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime MUST evaluate every self-state-driven parameter update
  path and classify whether it is a differentiable weight or bias update, or a
  discrete policy/ranking path.
- **FR-002**: The runtime MUST use Adam for Active LoRA runtime tensor writes
  that are currently driven by self-state-derived deltas.
- **FR-003**: The runtime MUST use the same Adam-backed write path for
  family-specific functional LoRA updates that share the existing runtime LoRA
  mutation mechanism.
- **FR-004**: Adam state for runtime LoRA writes MUST be explicit, typed, and
  bounded in memory.
- **FR-005**: Runtime LoRA optimizer state MUST persist across online writes
  for the same adapter weight tensors.
- **FR-006**: Runtime LoRA writes MUST preserve existing bounded weight decay,
  gain normalization, and serving-stack semantics unless explicitly superseded
  by the new optimizer behavior.
- **FR-007**: The temporal write-bias controller MUST update its trainable
  scalar bias parameters through Adam rather than only through thresholded
  additive heuristics.
- **FR-008**: Temporal reward bias, dampening bias, and effective write scale
  MUST remain explicitly bounded and inspectable after the optimizer change.
- **FR-009**: The implementation MUST expose typed runtime observability for
  the added Adam-controlled paths, including step counts or equivalent proof
  that optimizer state advanced.
- **FR-010**: The implementation MUST NOT apply Adam to counterfactual LoRA
  ablation ranking or other discrete candidate-selection code paths unless a
  real differentiable parameterization is introduced.
- **FR-011**: Architecture docs and research artifacts MUST explicitly state
  why Adam is adopted for runtime LoRA and temporal bias updates but rejected
  for the current counterfactual ablation ladder.
- **FR-012**: Targeted automated tests MUST cover the new optimizer-backed
  runtime LoRA path and the temporal bias path.

### Key Entities

- **Runtime LoRA Adam State**: Per-runtime-adapter optimizer moments and step
  counters for low-rank `A` and `B` tensors updated from self-state deltas.
- **Temporal Bias Adam State**: Typed optimizer state for reward and dampening
  scalar biases used to derive effective write scale.
- **Optimizer Observability Surface**: Public stats or trace fields that expose
  whether Adam-backed updates have advanced.
- **Discrete Counterfactual Ranking Path**: The explicit intervention-ranking
  logic that remains non-Adam because it does not update differentiable
  parameters.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Targeted tests prove Active LoRA and functional LoRA runtime
  writes advance explicit Adam-backed optimizer state.
- **SC-002**: Targeted tests prove temporal write-bias updates advance explicit
  Adam-backed optimizer state while keeping effective write scale bounded.
- **SC-003**: Existing Active LoRA, cognitive-loop, and past-LoRA regression
  tests continue to pass after the optimizer expansion.
- **SC-004**: Architecture and research artifacts explicitly document at least
  one strong non-candidate path where Adam is intentionally not inserted.

## Assumptions

- The current runtime LoRA write deltas are sufficiently gradient-like to
  benefit from adaptive first/second-moment conditioning even though they are
  hand-constructed online signals rather than backpropagated gradients.
- The current counterfactual ablation ladder remains a discrete ranking policy
  and should not be reframed as a differentiable optimizer target in this
  change set.
