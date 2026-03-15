# Feature Specification: Typed Persistent Self-State

**Feature Branch**: `003-strict-memory-runtime`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: Whitepaper Sections 5-8 plus `/Users/tyleraraujo/vicuna/specs/003-strict-memory-runtime/architecture-guideline.md`

## User Scenarios & Testing

### User Story 1 - Inspectable Self-State Surface (Priority: P1)

As a runtime developer, I can inspect a context-local self-state with explicit registers and system datetime fields, so continuity state is not reduced to prompt text or hidden adapter state.

**Why this priority**: The self-state needs a concrete runtime surface before richer update logic can exist.

**Independent Test**: Create a context, query the self-state API, and verify that default registers and datetime-derived fields are present and bounded.

**Acceptance Scenarios**:

1. **Given** a newly created context, **When** the self-state is queried, **Then** it exposes predefined registers for epistemic, goal, broadcast, memory, environment, and channel state.
2. **Given** a newly created context, **When** datetime is queried, **Then** it exposes raw wall-clock and monotonic time plus derived local calendar and cyclic features.
3. **Given** a context with no external updates yet, **When** register values are queried, **Then** bounded scalar registers remain within `[0, 1]` and categorical registers expose valid enum-backed values.

### User Story 2 - Deterministic Time and Event Updates (Priority: P1)

As a runtime architect, I can drive the self-state from explicit time points and event markers, so register recomputation remains replayable and suitable for later counterfactual simulation.

**Why this priority**: Replayable time and event inputs are the minimum viable basis for a mathematical self-state.

**Independent Test**: Set explicit time points, note user/tool/emit events, and verify that derived deltas and time-dependent registers update deterministically.

**Acceptance Scenarios**:

1. **Given** an explicit self-state time point, **When** it is applied to the context, **Then** the datetime surface and `r_time_phase` update deterministically from that input instead of from hidden ambient time.
2. **Given** a noted tool event and a later explicit time point, **When** the self-state is queried, **Then** `r_tool_salience` remains bounded and decays from the elapsed time since the tool event.
3. **Given** noted user, tool, and emit events, **When** the datetime surface is queried later, **Then** the elapsed delta fields reflect those anchors.

### User Story 3 - Explicit Channel and Register Provenance (Priority: P2)

As a runtime developer, I can inspect register metadata such as family, bounds, source mask, confidence, and updater version, so future register-update policies can be audited and evolved safely.

**Why this priority**: The register bank must be extensible and inspectable, not just numerically available.

**Independent Test**: Query register metadata and verify that family, confidence, timestamps, and source masks are populated consistently for default and updated registers.

**Acceptance Scenarios**:

1. **Given** any predefined register, **When** it is queried, **Then** the API returns its id, family, current value, confidence, last-update timestamps, source mask, and updater version.
2. **Given** a channel-state update, **When** the channel register is queried, **Then** it reflects the new categorical value and provenance metadata.
3. **Given** a self-state time update, **When** related registers are queried, **Then** their last-update timestamps and source mask show time-driven recomputation.

### User Story 4 - Prewrite/Postwrite Feature Builders (Priority: P1)

As a runtime architect, I can build explicit prewrite and postwrite feature vectors from token events and decoder/tool metadata, so register updates are driven by typed mathematical features instead of ad hoc prompt text.

**Why this priority**: This is the bridge from the raw event stream to the register bank.

**Independent Test**: Build feature vectors for deterministic token events, apply prewrite/postwrite updates, and verify bounded register movement plus stable metadata.

**Acceptance Scenarios**:

1. **Given** a token event and deterministic time state, **When** prewrite features are built, **Then** the runtime returns explicit novelty, topic-shift, decoder, lexical, and recency features.
2. **Given** a token event admitted into working memory, **When** postwrite features are built, **Then** the runtime returns explicit broadcast, inhibition, follow-up, and memory-write feature values.
3. **Given** a built feature vector, **When** the bounded analytic update path runs, **Then** affected registers remain inside their declared bounds.

### User Story 5 - Learned Contradiction and Uncertainty Heads Behind Flags (Priority: P2)

As a runtime developer, I can enable learned contradiction and uncertainty heads behind explicit flags, so the system can upgrade those signals without hard-coding one model family into the runtime.

**Why this priority**: Contradiction and uncertainty should be extensible and swappable.

**Independent Test**: Configure callback-backed learned heads behind flags, build features for a token event, and verify that the resulting head outputs influence the prewrite update path.

**Acceptance Scenarios**:

1. **Given** learned-head flags are disabled, **When** features are built, **Then** contradiction and uncertainty scores fall back to the analytic path.
2. **Given** learned-head flags are enabled with callbacks, **When** features are built, **Then** the callback outputs are incorporated into the resulting feature vector.
3. **Given** learned-head flags are enabled without callbacks, **When** features are built, **Then** the runtime stays functional and falls back explicitly to the analytic path.

### User Story 6 - Retrieval Handles, Tool Lifecycle, and Replay (Priority: P1)

As a runtime architect, I can attach typed memory handles and tool jobs to the self-state, derive sparse reactivation priorities from postwrite updates, and replay recorded traces deterministically, so Sections 5-8 stay extensible and inspectable.

**Why this priority**: Memory handles, tool lifecycle state, and replay are the next dependency layer beneath counterfactual updater work.

**Independent Test**: Upsert a memory handle and tool jobs, drive admitted events through postwrite, then verify handle-based features, typed tool state, sparse reactivation priorities, and deterministic partial/full trace replay.

**Acceptance Scenarios**:

1. **Given** a typed memory handle has been upserted, **When** related prewrite features are built, **Then** the runtime exposes memory-handle similarity and variance fields in the feature vector.
2. **Given** pending and running tool jobs exist, **When** postwrite features are built, **Then** the runtime exposes typed tool readiness and pending-pressure features distinct from `r_tool_salience`.
3. **Given** admitted events have been processed, **When** trace replay runs for a prefix or the full trace, **Then** dynamic self-state is rebuilt deterministically while the stored trace remains intact.
4. **Given** a candidate updater program is supplied, **When** counterfactual evaluation runs on a frozen trace, **Then** the runtime returns replay-derived summary metrics without mutating the live trace.
5. **Given** spans have been evicted from the KV cache, **When** the self-state is queried later, **Then** frozen-bucket handles, social-state scalars, and updater parameters remain available as persistent prefix state.
6. **Given** a stored trace is replayed on the counterfactual channel, **When** the replay completes, **Then** it does not drive primary-channel anchors, social-turn counts, or outward channel activation.

## Edge Cases

- What happens when the caller supplies a non-monotonic time point?
- What happens when the timezone offset is outside a sane range?
- What happens when a register id is queried that is not implemented in the current slice?
- What happens when event markers are applied before any explicit time override?
- What happens when derived time deltas would overflow a 32-bit integer range?
- What happens when a feature builder receives an empty token event?
- What happens when decoder entropy or top-margin metadata is unavailable?
- What happens when a learned head callback returns an invalid score?

## Requirements

### Functional Requirements

- **FR-SS-001**: The system MUST maintain a typed persistent self-state object per `llama_context`.
- **FR-SS-002**: The self-state MUST expose predefined register definitions for the fixed initial register set implemented in this slice.
- **FR-SS-003**: The self-state MUST store raw wall-clock time, raw monotonic time, timezone offset, local calendar fields, and derived cyclic time features.
- **FR-SS-004**: The self-state MUST support applying explicit time points supplied by the caller for deterministic replay and testing.
- **FR-SS-005**: The self-state MUST expose a convenience path that refreshes its time surface from the local system clock.
- **FR-SS-006**: The self-state MUST track anchor times for at least last user event, last tool event, and last system emit event.
- **FR-SS-007**: The self-state MUST recompute derived elapsed-time fields from those anchors whenever time advances.
- **FR-SS-008**: The self-state MUST represent bounded scalar registers and categorical registers with explicit family metadata rather than as one untyped float array.
- **FR-SS-009**: The system MUST expose register inspection APIs that return value plus provenance metadata.
- **FR-SS-010**: The system MUST expose explicit channel-state mutation and event-marker update APIs in CPU-side control code.
- **FR-SS-011**: `r_time_phase` MUST be derived from the explicit time surface instead of being stored as opaque prompt text.
- **FR-SS-012**: `r_tool_salience` MUST be derived from tool-event recency using a bounded analytic rule in this initial slice.
- **FR-SS-013**: Register values returned by the public API MUST remain within the bounds implied by their declared family.
- **FR-SS-014**: Invalid time points, invalid channel states, and unknown register ids MUST fail explicitly rather than being silently accepted.
- **FR-SS-015**: The implementation MUST remain compatible with future constrained self-modification of updater programs by storing updater-version and source metadata.
- **FR-SS-016**: The implementation MUST include targeted automated tests for default state inspection, explicit time updates, event-anchor deltas, and channel-state updates.
- **FR-SS-017**: The runtime MUST expose explicit prewrite and postwrite feature-builder APIs over a typed token event surface.
- **FR-SS-018**: Prewrite features MUST include novelty, topic-shift, recency, token-shape, and optional decoder-stat fields.
- **FR-SS-019**: Postwrite features MUST include broadcast, inhibition, follow-up, and memory-write feature values derived from the admitted event and current self-state.
- **FR-SS-020**: The bounded analytic update path MUST use explicit gains and clamps instead of unconstrained additive updates.
- **FR-SS-021**: The runtime MUST expose configuration for enabling learned contradiction and uncertainty heads independently.
- **FR-SS-022**: Learned heads MUST be optional and callback-backed in the initial implementation slice.
- **FR-SS-023**: Invalid learned-head outputs MUST be rejected or clamped explicitly.
- **FR-SS-024**: The implementation MUST include targeted automated tests for feature building, bounded analytic updates, and flag-gated learned head callbacks.
- **FR-SS-025**: The self-state MUST expose typed identity, goal, and commitment surfaces that remain separate from prompt text.
- **FR-SS-026**: Admitted postwrite events MUST be written into a typed working-memory store with bounded capacity.
- **FR-SS-027**: Prewrite and postwrite feature builders MUST expose retrieval-backed similarity features for working memory, goals, commitments, and identity.
- **FR-SS-028**: `r_goal_relevance` and `r_self_relevance` MUST be updated from retrieval-backed features rather than lexical cues alone.
- **FR-SS-029**: The public API MUST expose countable identity-adjacent surfaces and working-memory size for regression testing and replay inspection.
- **FR-SS-030**: The self-state MUST expose typed memory-handle surfaces with bounded priorities and handle kinds suitable for future cluster and frozen-LoRA namespaces.
- **FR-SS-031**: Prewrite and postwrite feature builders MUST expose memory-handle similarity and dispersion features distinct from working-memory similarity.
- **FR-SS-032**: The self-state MUST expose typed tool-job lifecycle state and derive tool readiness and pending-pressure features from it.
- **FR-SS-033**: Postwrite updates MUST maintain a sparse reactivation-priority surface keyed by memory handle id and kind.
- **FR-SS-034**: The runtime MUST record a deterministic self-state trace and support replaying a requested prefix or the full trace without mutating the stored trace.
- **FR-SS-035**: The runtime MUST expose optional broadcast-policy heads behind flags with explicit fallback to builtin analytic behavior.
- **FR-SS-036**: The implementation MUST include targeted automated tests for memory handles, tool jobs, reactivation priorities, learned broadcast hooks, and trace replay.
- **FR-SS-037**: The runtime MUST support exporting and importing frozen self-state traces for later replay.
- **FR-SS-038**: The runtime MUST expose a constrained updater-program surface that can modify bounded analytic coefficients and per-register update rules without changing the typed state schema.
- **FR-SS-039**: The runtime MUST support counterfactual replay of a candidate updater program and return replay-derived summary metrics without mutating live trace storage.
- **FR-SS-040**: The self-state MUST synchronize typed frozen-LoRA bucket handles from the real Past LoRA stack only at consolidation ticks so reactivation priorities can target persistent memory layers that survive KV eviction.
- **FR-SS-041**: The self-state MUST maintain richer social and relationship state as bounded persistent scalars rather than only deriving social relevance from the current event role.
- **FR-SS-042**: The runtime MUST provide in-tree contradiction, uncertainty, and broadcast probe heads implemented over typed scalar features and enabled behind explicit flags.
- **FR-SS-043**: Frozen-bucket handles, social-state scalars, and updater parameters MUST remain part of the persistent self prefix even after prompt spans are evicted from the KV cache.
- **FR-SS-044**: Updater rules MUST support per-register baselines, asymmetric rise/fall gains, baseline pull, typed feature terms, and bounded scalar cross-register couplings.
- **FR-SS-045**: The runtime MUST represent counterfactual replay as a dedicated event/replay channel rather than overloading the primary interactive channel.
- **FR-SS-046**: Replaying a trace on the counterfactual channel MUST preserve shared typed state mechanics while suppressing primary-channel side effects such as outward channel activation, user/tool/emit anchors, and social-turn accumulation.
- **FR-SS-047**: The public API MUST expose channel-aware replay and counterfactual-evaluation entry points so a future subsystem can trigger counterfactual processing without rewriting the self-state runtime.

### Key Entities

- **Self State**: The typed continuity surface attached to a `llama_context`.
- **Register Definition**: Metadata declaring register id, family, bounds, update semantics, and version.
- **Register Value**: Live value plus provenance and timestamps for a predefined register.
- **Time Point**: Raw wall-clock time, monotonic time, and timezone offset used to update the self-state deterministically.
- **Datetime Surface**: Expanded raw and derived time features exposed from the self-state.
- **Event Anchors**: Last-known monotonic timestamps for user, tool, and emit events.

## Success Criteria

### Measurable Outcomes

- **SC-SS-001**: Automated tests can create a context and inspect a non-empty predefined self-state register set.
- **SC-SS-002**: Automated tests can apply explicit time points and observe deterministic updates to datetime-derived fields and `r_time_phase`.
- **SC-SS-003**: Automated tests can note tool events and later observe bounded, decayed `r_tool_salience`.
- **SC-SS-004**: Automated tests can update channel state and retrieve categorical register metadata that reflects the change.
- **SC-SS-005**: The implementation keeps all self-state policy in CPU-side control code and introduces no hidden backend-side policy.
- **SC-SS-006**: Automated tests can build deterministic prewrite/postwrite feature vectors and apply bounded analytic updates without producing out-of-range register values.
- **SC-SS-007**: Automated tests can enable callback-backed learned heads and observe different contradiction/uncertainty outputs from the analytic fallback path.
- **SC-SS-008**: Automated tests can upsert typed memory handles and observe non-zero handle-similarity features and sparse reactivation priorities.
- **SC-SS-009**: Automated tests can upsert typed tool jobs and observe readiness/pending-pressure features distinct from tool salience.
- **SC-SS-010**: Automated tests can replay a trace prefix and the full trace deterministically while preserving the stored trace length.
- **SC-SS-011**: Automated tests can export a frozen trace, import it into a fresh context, and replay it successfully.
- **SC-SS-012**: Automated tests can evaluate a candidate updater program counterfactually and receive non-empty replay-derived summary metrics.
- **SC-SS-013**: Automated tests can populate a Past LoRA bucket and observe corresponding frozen-bucket handles inside self-state memory-handle surfaces.
- **SC-SS-014**: Automated tests can observe non-zero persistent social-state scalars after user and system/tool interactions.
- **SC-SS-015**: Automated tests can exercise in-tree contradiction, uncertainty, and broadcast probes without supplying external callbacks.
- **SC-SS-016**: Automated tests can set a candidate updater program with non-empty per-register rules, reject an invalid rule set, and counterfactually replay the valid program.
- **SC-SS-017**: Automated tests can replay a stored trace on the counterfactual channel and verify that primary-channel social counts and channel-state activation do not advance.
