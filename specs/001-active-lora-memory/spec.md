# Feature Specification: Sliding-Window Active LoRA Memory

**Feature Branch**: `001-active-lora-memory`  
**Created**: 2026-03-10  
**Status**: Draft  
**Input**: User description: "Implement sliding-window eviction into a fixed-budget Active LoRA memory stage with swappable embedding strategies and future rollover support"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Budgeted Active LoRA Memory (Priority: P1)

As a Vicuña runtime operator, I can enable an Active LoRA memory stage that consumes context pushed out of the live window and keeps its memory footprint within explicit host-memory and accelerator-memory proportions, so recent experience survives window eviction without exceeding system budgets.

**Why this priority**: This is the core architectural requirement. Without a bounded Active LoRA stage, the memory cascade does not exist and the runtime falls back to pure sliding-window behavior.

**Independent Test**: Can be fully tested by enabling the feature on a model, evicting tokens from the live window, and verifying that the runtime creates and updates an Active LoRA while staying within the configured RAM/VRAM budget surfaces.

**Acceptance Scenarios**:

1. **Given** an enabled Active LoRA memory stage and a running context that exceeds the live window, **When** tokens are pushed out of the sliding window, **Then** the evicted span is admitted into Active LoRA processing and the runtime reports the resulting adapter state and budget usage.
2. **Given** configured memory budget proportions for host memory and accelerator memory, **When** the Active LoRA is allocated or refreshed, **Then** its size is capped by those proportions of currently available memory rather than by a fixed global byte constant.
3. **Given** an existing Active LoRA at capacity, **When** additional eviction pressure arrives, **Then** the runtime preserves fixed-size behavior through bounded updates and rollover policy instead of unbounded adapter growth.

---

### User Story 2 - Swappable Embedding Strategy (Priority: P2)

As a runtime developer, I can choose or replace the embedding strategy used for Active LoRA formation, so the memory pipeline can work across model families that do not share one compatible embedding space.

**Why this priority**: The memory cascade must remain extensible across open-source models. Hard-wiring one embedding model would create a hidden architectural dependency and block future portability.

**Independent Test**: Can be fully tested by instantiating the Active LoRA path with one embedding strategy, switching to another compatible strategy, and verifying that admission and update behavior still execute through the same public runtime interface.

**Acceptance Scenarios**:

1. **Given** multiple embedding strategies registered with the runtime, **When** the Active LoRA feature is initialized, **Then** one strategy is selected explicitly and its identity is visible through configuration or logs.
2. **Given** an embedding strategy that is incompatible with the current model family, **When** Active LoRA formation is requested, **Then** the runtime fails safely with an inspectable error or falls back to a declared compatible strategy rather than silently producing undefined memory updates.

---

### User Story 3 - Inspectable Rollover and Audit Trail (Priority: P3)

As a Vicuña architect, I can inspect how evicted spans were admitted, embedded, written, and rolled through the Active LoRA stage, so future past-LoRA freezing and DMN integration can build on an auditable memory pipeline.

**Why this priority**: The architecture requires explicit state, observable thresholds, and frozen past LoRAs after rollover. Even before the full past stack exists, the Active LoRA path must be traceable.

**Independent Test**: Can be fully tested by forcing multiple eviction cycles and verifying that traces expose admission decisions, selected embedding strategy, budget decisions, update counts, and rollover boundary events.

**Acceptance Scenarios**:

1. **Given** Active LoRA memory is enabled, **When** an evicted span is processed, **Then** the runtime emits traceable metadata describing the span admission, strategy selection, and adapter update outcome.
2. **Given** the Active LoRA reaches its rollover boundary, **When** rollover is triggered, **Then** the runtime records the boundary event and preserves the resulting frozen-state handoff metadata needed for a future past LoRA stack.

### Edge Cases

- What happens when a backend cannot report free or total device memory and only host memory is reliable?
- How does the system handle a configured memory proportion that produces a budget too small to allocate even the minimum viable Active LoRA?
- How does the system handle an eviction span that exceeds the per-update training or write budget?
- What happens when the selected embedding strategy produces vectors with a dimension or format incompatible with the Active LoRA writer?
- How does the system behave when no accelerator is present and all Active LoRA storage must fit in host memory?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide an Active LoRA memory stage that accepts context spans evicted from the sliding attention window.
- **FR-002**: The system MUST size each live or frozen LoRA memory unit from explicit configurable proportions of currently available host memory and accelerator memory, not from fixed byte constants baked into the binary.
- **FR-003**: The system MUST keep Active LoRA memory fixed-size within its computed budget and MUST prevent unbounded growth as more context is evicted.
- **FR-004**: The system MUST preserve the memory cascade ordering `sliding attention window -> active LoRA -> past LoRA stack` and MUST expose a rollover boundary for freezing the Active LoRA into a future past-LoRA unit.
- **FR-005**: The system MUST keep base model weights immutable during Active LoRA updates unless an operator explicitly enters a separate training flow outside this feature.
- **FR-006**: The system MUST expose a swappable embedding-strategy interface for Active LoRA formation, with at least one default strategy available at runtime.
- **FR-007**: The system MUST detect and surface embedding-strategy incompatibility before applying Active LoRA updates.
- **FR-008**: The system MUST keep Active LoRA control state explicit and inspectable, including budget decisions, selected embedding strategy, update counters, admission outcomes, and rollover readiness.
- **FR-009**: The system MUST emit logs or traces explaining why an evicted span was admitted, skipped, truncated, or rolled over.
- **FR-010**: The system MUST provide targeted automated tests covering eviction ordering, fixed-budget sizing, strategy swapping, and rollover-state persistence.
- **FR-011**: The system MUST update architecture documentation to describe the Active LoRA path, budget policy, strategy interface, and rollover semantics.

### Key Entities

- **Evicted Span**: A contiguous block of tokens or text that has been pushed out of the live sliding window and is eligible for durable memory admission.
- **Active LoRA Budget**: The computed host-memory and accelerator-memory allowance that constrains the size of the Active LoRA unit for the current system state.
- **Active LoRA Unit**: The mutable, fixed-size adapter that receives recent evicted experience before rollover.
- **Embedding Strategy**: A replaceable component that converts evicted spans into the vector representation used by the Active LoRA writer.
- **Active LoRA Update Record**: Inspectable metadata describing one eviction-to-adapter write decision, including admission, strategy, budget, and outcome.
- **Rollover Boundary**: The state at which the Active LoRA stops accepting further growth and is prepared to be frozen into a past-LoRA unit.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a supported system, enabling Active LoRA memory and forcing window eviction results in at least one successful Active LoRA update event with corresponding trace output.
- **SC-002**: For every tested memory profile, the allocated Active LoRA storage remains at or below the configured host-memory and accelerator-memory proportions throughout the test run.
- **SC-003**: Switching between supported embedding strategies requires no changes to the surrounding eviction or Active LoRA management call flow.
- **SC-004**: Automated tests cover the four critical behaviors of this feature: eviction ordering, budget enforcement, strategy swap behavior, and rollover metadata persistence.

## Assumptions

- The initial implementation may deliver the Active LoRA stage and rollover handoff metadata before a full past-LoRA retrieval stack is wired into inference weighting.
- The first release of this feature may limit trainable or writable adapter targets to a curated subset of model tensors as long as the budget model and public interfaces support broader future expansion.
- Host-memory budgeting is always required; accelerator-memory budgeting applies when at least one non-CPU backend reports usable device memory.
