# Feature Specification: Frozen Past-LoRA Condensation Stack

**Feature Branch**: `002-past-lora-condensation`  
**Created**: 2026-03-10  
**Status**: Draft  
**Input**: User description: "Extend the memory cascade with frozen past LoRA condensate stages, directional-gain parameterization, and scheduled condensation across All Time, Past Year, Past Quarter, Past Month, and Past Week stacks."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Frozen Temporal Memory Stack (Priority: P1)

As a Vicuña runtime operator, I can enable a fixed set of frozen past-memory LoRA stages for Past Week, Past Month, Past Quarter, Past Year, and All Time, so long-horizon memory survives beyond the editable Active LoRA without being rewritten during normal runtime.

**Why this priority**: This is the architectural core of the next memory stage. Without frozen time buckets in the inference stack, the memory cascade stops at recent experience and cannot express durable long-horizon bias.

**Independent Test**: Can be fully tested by forcing the Active LoRA to roll over, freezing the resulting artifact into Past Week, and verifying that the frozen stage remains present in inference with inspectable time-bucket metadata and no live mutation path.

**Acceptance Scenarios**:

1. **Given** an enabled memory cascade with an initialized Active LoRA, **When** the Active LoRA reaches its rollover boundary, **Then** the runtime freezes it into the Past Week stack and preserves an inspectable artifact record rather than continuing to mutate that frozen unit.
2. **Given** frozen past-memory stages already exist, **When** inference runs, **Then** the runtime applies each stage as a separate LoRA influence with a decayed weight derived from its bucket identity and age rather than rewriting the frozen tensors.
3. **Given** a system restart or repeated inference cycles, **When** no condensation event is triggered, **Then** every frozen past-memory unit remains bitwise stable and auditable across those cycles.

---

### User Story 2 - Directional-Gain Memory Updates (Priority: P2)

As a Vicuña architect, I can represent both Active LoRA writes and past-stage condensation outputs as normalized low-rank directions plus constrained gain components, so memory updates mostly adjust semantic direction while tightly controlling update magnitude.

**Why this priority**: The user explicitly requires that both live updates and backwards compactification obey a magnitude-direction split. Without this, the memory system can drift in uncontrolled ways and the frozen stack becomes hard to reason about.

**Independent Test**: Can be fully tested by applying multiple Active LoRA updates and at least one condensation event, then verifying that direction tensors remain normalized while magnitude or gain surfaces remain explicit, bounded, and inspectable.

**Acceptance Scenarios**:

1. **Given** an evicted span accepted by the Active LoRA writer, **When** the writer updates the adapter, **Then** it records a normalized directional update and a separate magnitude or gain update instead of only unconstrained raw tensor deltas.
2. **Given** a condensation job that merges a younger memory stage into an older frozen bucket, **When** the merge completes, **Then** the produced bucket keeps the same direction-versus-gain separation and enforces bounded gain growth.
3. **Given** a configured magnitude constraint, **When** update pressure is high, **Then** the runtime clips, rescales, or rejects the gain contribution rather than silently allowing unbounded amplification.

---

### User Story 3 - Scheduled Condensation Across Time Buckets (Priority: P3)

As a runtime developer, I can schedule and inspect condensation jobs that progressively compact Active LoRA into Past Week, Past Week into Past Month, Past Month into Past Quarter, Past Quarter into Past Year, and Past Year into All Time, so memory ages through explicit temporal strata instead of staying trapped in one recent stage.

**Why this priority**: The frozen stack only becomes useful if memory can age through it. Scheduled compaction is what turns one frozen rollover into a durable temporal memory hierarchy.

**Independent Test**: Can be fully tested by advancing the condensation scheduler through multiple bucket boundaries, observing the expected handoffs, and verifying that each older bucket receives a bounded condensed artifact with updated decay metadata.

**Acceptance Scenarios**:

1. **Given** a populated Active LoRA and an empty past stack, **When** the Past Week condensation job runs, **Then** the runtime freezes the Active LoRA into Past Week and starts a fresh editable Active LoRA.
2. **Given** populated younger and older buckets, **When** the scheduled Past Week-to-Past Month or Past Month-to-Past Quarter job runs, **Then** the runtime condenses the younger frozen artifact into the older bucket without mutating the younger source artifact in place.
3. **Given** the temporal stack is fully populated, **When** inference runs after multiple condensation cycles, **Then** all five buckets remain separately inspectable and each bucket’s applied influence reflects its configured time-decay policy.

### Edge Cases

- What happens when a configured memory ratio cannot allocate the minimum viable rank for one or more frozen buckets?
- How does the system behave when multiple condensation jobs become due at once, such as after a long idle interval or delayed scheduler tick?
- What happens when a condensation merge would exceed the target bucket budget or its gain ceiling?
- How does the system handle a custom embedder or writer configuration that is valid for Active LoRA writes but incompatible with frozen-bucket condensation?
- What happens when the runtime has no accelerator and every bucket must fit entirely in host memory?
- How does the system behave when a bucket receives no meaningful new signal during a condensation interval and the merge would only add redundant direction?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a frozen past-memory LoRA stack with exactly five named temporal buckets: `past_week`, `past_month`, `past_quarter`, `past_year`, and `all_time`.
- **FR-002**: The system MUST preserve the memory cascade ordering `sliding attention window -> active LoRA -> past_week -> past_month -> past_quarter -> past_year -> all_time`.
- **FR-003**: The system MUST keep each past-memory bucket fixed-size within explicit configurable host-memory and accelerator-memory proportions computed from currently available memory.
- **FR-004**: The system MUST freeze each past-memory artifact after rollover or condensation and MUST NOT rewrite that frozen artifact during ordinary inference-time updates.
- **FR-005**: The system MUST apply all enabled past-memory buckets during inference as separate LoRA influences whose effective strength is decayed by bucket identity and artifact age.
- **FR-006**: The system MUST expose explicit condensation jobs for `active -> past_week`, `past_week -> past_month`, `past_month -> past_quarter`, `past_quarter -> past_year`, and `past_year -> all_time`.
- **FR-007**: The system MUST allow condensation jobs to run because of explicit schedule or pressure state and MUST surface when a job is due, skipped, running, or completed.
- **FR-008**: The system MUST represent Active LoRA updates with a normalized low-rank directional component and a separate bounded magnitude or gain component.
- **FR-009**: The system MUST represent condensed past-memory artifacts with the same direction-versus-gain separation used by the Active LoRA path.
- **FR-010**: The system MUST enforce configurable normalization and gain-bound policies during both live Active LoRA updates and past-bucket condensation.
- **FR-011**: The system MUST prevent base model weights from being modified by either Active LoRA updates or past-bucket condensation.
- **FR-012**: The system MUST preserve inspectable audit metadata for every frozen artifact, including source bucket, target bucket, creation time, age window, budget decision, rank, embedder identity, normalization policy, and gain statistics.
- **FR-013**: The system MUST log or trace why a condensation job ran, skipped, merged, clipped gain, pruned direction, or rejected a merge.
- **FR-014**: The system MUST expose the current temporal stack state, effective decay scales, and pending condensation deadlines through inspectable runtime state rather than only implicit internal tensors.
- **FR-015**: The system MUST keep the embedding strategy used for Active LoRA admission and condensation swappable, including model-specific callback-backed embeddings when needed.
- **FR-016**: The system MUST fail safely when a bucket budget, writer policy, or embedder configuration is incompatible with the requested condensation action.
- **FR-017**: The system MUST provide targeted automated tests covering frozen-bucket immutability, scheduled bucket handoff order, decayed inference weighting, directional normalization, bounded gain behavior, and condensation audit trails.
- **FR-018**: The system MUST update architecture and design documentation to describe the temporal bucket stack, condensation ordering, direction-gain representation, and time-decay semantics.

### Key Entities

- **Temporal Bucket**: One named frozen-memory stage in the ordered stack: `past_week`, `past_month`, `past_quarter`, `past_year`, or `all_time`.
- **Frozen Artifact**: An immutable LoRA memory unit stored inside one temporal bucket together with its provenance, age window, and decay metadata.
- **Condensation Job**: A scheduled or pressure-driven action that merges one younger memory stage into its next older temporal bucket.
- **Directional Component**: The normalized low-rank direction that captures semantic or behavioral change while constraining shape and orientation.
- **Gain Component**: The scalar or vector strength surface that controls how strongly a directional component should affect inference.
- **Decay Policy**: The explicit rule that converts bucket identity, artifact age, and runtime configuration into an inference-time application scale.
- **Condensation Record**: Inspectable metadata describing one rollover or merge event, including source, target, budgets, normalization, gain control, and outcome.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A forced Active LoRA rollover produces a Past Week frozen artifact that remains stable across subsequent inference cycles and is visible through inspectable runtime stats or logs.
- **SC-002**: In automated tests, every temporal bucket applies a distinct non-negative decay-scaled influence during inference once populated.
- **SC-003**: Directional components produced by both live updates and condensation remain normalized within configured tolerance, and gain components remain within configured bounds across all tested rollover paths.
- **SC-004**: Advancing the scheduler through the full chain `active -> past_week -> past_month -> past_quarter -> past_year -> all_time` completes without violating configured per-bucket memory budgets.
- **SC-005**: Automated tests cover the six critical behaviors of this feature: bucket freezing, bucket handoff order, decay weighting, direction normalization, gain bounding, and condensation auditability.

## Assumptions

- The first implementation may keep one frozen artifact per named bucket rather than supporting arbitrary lists of artifacts inside a single bucket, as long as the public state and condensation logic leave room for future expansion.
- The initial decay policy may be deterministic and configuration-driven rather than learned, provided the resulting scales are explicit and inspectable.
- The initial condensation scheduler may run from existing runtime or server ticks rather than a dedicated DMN loop, as long as the job state is explicit and pressure-aware.
- The first release may use bounded merge heuristics and low-rank recompression for condensation as long as the result stays frozen, auditable, and within the bucket budget.
