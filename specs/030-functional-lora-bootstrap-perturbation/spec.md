# Feature Specification: Functional LoRA Bootstrap Perturbation

**Feature Branch**: `[030-functional-lora-bootstrap-perturbation]`  
**Created**: 2026-03-13  
**Status**: Draft  
**Input**: User description: "when functional loras are first created, they should essentially be no-ops, but should implement a stochastic perturbation that allows the system to \"accidentally\" discover useful biases in the early stages. the magnitude of this stochastic perturbation should decay as more usage builds the LoRA up, but it should never hit zero, just approach a minimum. please confirm this is the correct description of function for the functional loras. if not, make it so."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Keep New Functional LoRAs Near No-Op While Letting Them Accidentally Help (Priority: P1)

As a Vicuña operator, I need newly created functional LoRAs to begin as effectively no-op learned adapters while still carrying a tiny stochastic effect so the runtime can accidentally discover useful early biases before the family has learned meaningful weights.

**Why this priority**: This is the direct requested ontology for functional LoRA birth behavior, and the current gate-only exploration does not satisfy it.

**Independent Test**: Initialize a context, inspect functional family state, and verify that the learned runtime adapter remains zero-initialized while the family exposes nonzero bootstrap-perturbation configuration and a bounded sampled perturbation on activation.

**Acceptance Scenarios**:

1. **Given** a freshly initialized functional LoRA family, **When** no runtime updates have been applied, **Then** the learned adapter MUST remain a no-op and any early nonzero effect MUST come only from an explicit bootstrap perturbation path.
2. **Given** a freshly initialized family that is activated during routing, **When** the activation is applied, **Then** the runtime MUST be able to inject a small stochastic perturbation that can have either sign and is exposed through typed traces.

---

### User Story 2 - Decay Exploration With Use But Never To Zero (Priority: P2)

As a Vicuña operator, I need the functional-LoRA bootstrap perturbation to shrink as a family accumulates usage so that accidental exploration yields to learned behavior, while still never disappearing entirely.

**Why this priority**: The perturbation is only useful if it is strongest when the family is immature and remains minimally available later for continued discovery.

**Independent Test**: Replay multiple family activations and verify that the bootstrap perturbation standard deviation decreases monotonically toward a configured floor and never reaches zero.

**Acceptance Scenarios**:

1. **Given** repeated activations of the same functional family, **When** the bootstrap scale is recomputed, **Then** it MUST decay from its initial magnitude toward a positive minimum using an explicit bounded policy.
2. **Given** a mature family with many activations, **When** it is activated again, **Then** the bootstrap perturbation MUST still remain possible at a small nonzero floor.

---

### User Story 3 - Keep Bootstrap Exploration Inspectable And Compatible With Runtime Learning (Priority: P3)

As a Vicuña developer, I need bootstrap perturbation to remain explicit, bounded, and separable from learned functional weights so that traces, docs, and future optimizer work remain coherent.

**Why this priority**: The repository constitution requires explicit CPU-side policy and typed inspectability for runtime behavior changes.

**Independent Test**: Inspect public config/state/trace surfaces before and after activation and verify that bootstrap config, sampled perturbation, decay state, and usage counters are exposed.

**Acceptance Scenarios**:

1. **Given** a routed functional activation, **When** the trace is captured, **Then** it MUST expose both gate exploration and bootstrap perturbation separately.
2. **Given** later functional adapter updates from success or failure, **When** the family state is read, **Then** the runtime MUST preserve the distinction between learned adapter state and bootstrap perturbation state.

## Edge Cases

- What happens when a family is eligible but not activated because its gain clips to zero?
- How does the runtime avoid the bootstrap perturbation dominating the learned adapter after many successful updates?
- What happens when stochastic bootstrap samples are large relative to the configured gain clip range?
- How does the system behave when a family is heavily used but has received few or no successful updates?
- What happens when functional-stack ablation disables the family while bootstrap perturbation is configured?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Newly created functional LoRA families MUST keep their learned runtime adapter weights initialized as a no-op.
- **FR-002**: The runtime MUST provide a distinct bootstrap perturbation path for each functional LoRA family so early activations can produce a small accidental effect before meaningful learned weights exist.
- **FR-003**: The bootstrap perturbation MUST be stochastic, signed, and bounded.
- **FR-004**: The bootstrap perturbation magnitude MUST decay as family usage increases, using an explicit CPU-side policy.
- **FR-005**: The decay policy MUST approach a positive minimum and MUST NOT decay the perturbation magnitude to exactly zero.
- **FR-006**: Bootstrap perturbation MUST remain separate from the learned functional adapter so the learned adapter can still be described as starting from a no-op state.
- **FR-007**: The runtime MUST expose bootstrap perturbation configuration, sampled perturbation, current perturbation standard deviation, and family usage through typed public state or trace surfaces.
- **FR-008**: The implementation MUST preserve the existing gate-output exploration mechanism unless explicitly superseded; bootstrap perturbation is additive to, not a replacement for, gate exploration.
- **FR-009**: Bootstrap perturbation MUST only affect functional families when that family is actually invoked in the serving stack.
- **FR-010**: The implementation MUST remain compatible with functional ablation, hold windows, counterfactual execution, and user-simulation stack overrides.
- **FR-011**: The implementation MUST preserve bounded runtime memory and base-model immutability.
- **FR-012**: Targeted tests MUST cover initialization, decay-to-floor behavior, trace observability, and activation-time perturbation.
- **FR-013**: Architecture and operator documentation MUST describe that functional families begin as no-op learned adapters with a decaying nonzero bootstrap perturbation path.

### Key Entities *(include if feature involves data)*

- **Functional Family Learned Adapter**: The existing runtime-mutable functional LoRA whose trainable weights begin at zero effect and accumulate learning from runtime updates.
- **Functional Family Bootstrap Adapter**: A separate fixed-size runtime LoRA initialized with tiny random weights and used only as a stochastic early perturbation substrate.
- **Bootstrap Perturbation Policy**: The explicit decay rule that maps family usage to current perturbation standard deviation with a nonzero floor.
- **Functional Family Usage Counter**: The monotonic bounded counter used to track how much a family has been invoked for bootstrap decay purposes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Freshly initialized functional families report no learned-adapter activity but expose nonzero bootstrap perturbation configuration in typed public state.
- **SC-002**: In targeted activation tests, functional families can emit a nonzero bootstrap perturbation while their learned adapter remains zero-initialized.
- **SC-003**: In repeated-activation tests, bootstrap perturbation standard deviation decays monotonically toward a configured positive floor and never reaches zero.
- **SC-004**: In trace tests, developers can distinguish gate exploration from bootstrap perturbation without reading implementation internals.
- **SC-005**: Documentation and headers explain that accidental early discovery comes from an explicit bootstrap perturbation path, not from nonzero learned adapter initialization.
