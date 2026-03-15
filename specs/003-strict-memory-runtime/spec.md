# Feature Specification: Strict Live Memory Runtime

**Feature Branch**: `003-strict-memory-runtime`  
**Created**: 2026-03-10  
**Status**: Draft  
**Input**: User description: "Research and implement the most effective version of explicit memory-stack serving semantics, composition ordering, and strict KV coherence for the current inference runtime."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Persistent Memory Stack During Live Inference (Priority: P1)

As a Vicuña runtime operator, I can run live request/response inference with the frozen past-memory buckets and the editable Active LoRA always present in the serving adapter stack, so the memory cascade actually conditions generation instead of existing as side state that can be overwritten by per-request adapter changes.

**Why this priority**: This is the architectural baseline. If the serving path can drop runtime memory adapters, the memory cascade is not real during inference.

**Independent Test**: Can be fully tested by enabling Active and Past LoRA memory, applying a request adapter change, and verifying that generation still runs with the full memory stack plus the request adapter rather than replacing memory adapters.

**Acceptance Scenarios**:

1. **Given** a context with initialized Active and Past LoRA memory stages, **When** a live decode batch is assembled for a request, **Then** the runtime composes request adapters and memory adapters into one effective serving stack without dropping the runtime memory layers.
2. **Given** multiple populated past-memory buckets and an editable Active LoRA, **When** inference runs, **Then** the effective serving stack preserves the explicit memory ordering `past_oldest -> ... -> past_newest -> active` and keeps each layer inspectable in logs or stats.
3. **Given** a request changes its external LoRA set between batches, **When** the context updates adapter state, **Then** the runtime preserves the memory layers and only changes the request-adapter portion of the stack.

---

### User Story 2 - Strict KV Coherence After Active Writes (Priority: P1)

As a Vicuña architect, I can require that every Active LoRA update caused by window eviction is followed by strict replay of the retained suffix, so the surviving KV cache reflects the same effective adapter stack that will be used for subsequent generation.

**Why this priority**: The memory cascade becomes semantically inconsistent if the adapter state changes but the retained KV cache still encodes activations from the pre-update stack.

**Independent Test**: Can be fully tested by forcing a context shift during generation, observing Active LoRA ingestion, and verifying that the retained suffix is replayed before the next sampled token is produced.

**Acceptance Scenarios**:

1. **Given** a generating slot that exceeds the live context window, **When** the runtime evicts a span into Active LoRA, **Then** it invalidates the retained suffix KV state that is no longer coherent with the new adapter stack.
2. **Given** a retained suffix after eviction and Active LoRA ingestion, **When** strict coherence is enabled, **Then** the runtime replays that suffix through the updated serving stack before continuing token generation.
3. **Given** checkpoints or cached prompt state created under the pre-update adapter stack, **When** strict replay is triggered, **Then** the runtime invalidates or bypasses stale checkpoint data rather than restoring incoherent KV state.

---

### User Story 3 - Explicit, Inspectable Adapter Composition Policy (Priority: P2)

As a runtime developer, I can inspect and reason about how request adapters, frozen memory buckets, and the Active LoRA are composed for each live decode step, so adapter precedence, scales, and replay decisions are auditable instead of being implicit in container behavior.

**Why this priority**: Ordered, inspectable composition is required to keep policy understandable and to prevent future regressions back to accidental adapter replacement.

**Independent Test**: Can be fully tested by querying logs or runtime state after initialization, request-adapter changes, and context shifts, and verifying that composition order, scales, and replay events are reported consistently.

**Acceptance Scenarios**:

1. **Given** a runtime with request adapters and memory adapters attached, **When** the effective stack is rebuilt, **Then** the runtime records the ordered layers, scales, and layer classes used for the next decode.
2. **Given** a context shift that triggers Active LoRA ingestion, **When** strict replay begins or is skipped, **Then** the runtime records why replay was scheduled, how many tokens were replayed, and which slot was affected.
3. **Given** a runtime memory adapter changes scale because of decay or rollover, **When** the next decode graph is built, **Then** the effective stack reflects the new scale without requiring external request code to know about memory-layer internals.

### Edge Cases

- What happens when Active LoRA ingestion rejects an evicted span as redundant and no adapter weights actually change?
- What happens when strict replay is requested for a slot with zero retained suffix tokens after eviction?
- How does the runtime behave when a request-specific adapter uses aLoRA-style delayed activation while memory adapters are always-on?
- What happens when multiple slots require context shift in one server tick and only one shared decode context exists?
- How does the system behave when the retained suffix replay fails because the replay batch cannot fit or decode returns an error?
- What happens when a past bucket is initialized but currently has zero effective scale?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST preserve the memory cascade in live inference by keeping runtime memory adapters present in the effective serving stack whenever Active or Past LoRA memory is enabled.
- **FR-002**: The system MUST distinguish request-managed adapters from runtime-managed memory adapters so request adapter updates do not overwrite the memory stack.
- **FR-003**: The system MUST construct an explicit ordered serving stack instead of relying on unordered container iteration for adapter composition.
- **FR-004**: The system MUST preserve the logical memory ordering `past_oldest -> ... -> past_newest -> active` inside the serving stack.
- **FR-005**: The system MUST continue to apply request adapters when configured, while preserving the memory stack in the same decode path.
- **FR-006**: The system MUST trigger Active LoRA ingestion from evicted text before continuing generation after a context shift.
- **FR-007**: The system MUST treat an Active LoRA write that changes serving weights as a KV-coherence boundary for the retained suffix.
- **FR-008**: The system MUST invalidate or rebuild retained KV state that is no longer coherent with the updated serving stack before the next generated token is sampled.
- **FR-009**: The system MUST implement strict retained-suffix replay in the server runtime for generation-time context shifts when Active LoRA memory is enabled and an update is accepted.
- **FR-010**: The system MUST skip replay when no Active LoRA weight change occurred or when no retained suffix exists, and MUST log why replay was skipped.
- **FR-011**: The system MUST invalidate stale prompt checkpoints or other reusable state that was captured under the pre-update adapter stack when strict replay is scheduled.
- **FR-012**: The system MUST keep replay scheduling, adapter-layer ordering, and memory-layer scale policy in CPU-side C++ control code rather than burying that policy in backend kernels.
- **FR-013**: The system MUST expose inspectable logs or stats for effective adapter-stack rebuilds, strict replay scheduling, replay completion, replay skip reasons, and replay failures.
- **FR-014**: The system MUST preserve base-model immutability and continue to express memory influence only through runtime-generated LoRA adapters.
- **FR-015**: The system MUST provide targeted automated tests covering memory-stack preservation across request adapter changes, strict replay after Active updates, and replay-skip behavior for redundant or zero-suffix cases.
- **FR-016**: The system MUST update architecture and design documentation to describe explicit serving composition and strict KV coherence semantics.

### Key Entities

- **Serving Adapter Layer**: One ordered entry in the effective live adapter stack, including its adapter handle, scale, role, and precedence.
- **Request Adapter Set**: The request-specific LoRA set configured by client or server task state.
- **Runtime Memory Stack**: The runtime-managed memory adapter set composed from frozen past buckets plus the editable Active LoRA.
- **Effective Serving Stack**: The deterministic ordered composition used by a live decode graph after combining request adapters and runtime memory layers.
- **Strict Replay Event**: A slot-local action that invalidates stale retained KV state and reprocesses the surviving suffix under the updated serving stack.
- **Replay Window**: The retained suffix token range that must be re-decoded after eviction-triggered Active LoRA change.
- **Replay Audit Record**: Inspectable metadata for why replay ran, which slot it affected, how many tokens were replayed, and what outcome occurred.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In automated tests, enabling request adapters while Active and Past LoRA memory are active results in a serving stack that still contains all expected memory layers for every decode batch.
- **SC-002**: In automated tests, a context shift that produces an accepted Active LoRA update causes retained-suffix replay to complete before the next generated token is sampled.
- **SC-003**: In automated tests, redundant-span or zero-suffix context shifts skip replay and report an explicit skip reason instead of silently doing nothing.
- **SC-004**: Runtime logs or trace surfaces show deterministic ordered adapter composition and replay lifecycle events for the tested request/response cycle.
- **SC-005**: The documented request/response sequencing in `ARCHITECTURE.md` matches the implemented serving semantics and strict replay behavior after this feature lands.
