# Feature Specification: Hard Memory Primitives

**Feature Branch**: `026-hard-memory-primitives`  
**Created**: 2026-03-13  
**Status**: Draft  
**Input**: User description: "regarding section 3 of your new audit: expand hard memory(we are using supermemory) to store a much more robust range of fragments, trajectories, outcomes, user-model/info, etc. research the codebase to analyze what we are generating that should be collected, and research the SOTA/other projects to see what relevant primitives we should prepare for based on the principlese and high-level goals of the application. then architect with spec kit and implement in full and elegant and complete hard memory system that cooperates with Lora memory/bias to greater effect"

## User Scenarios & Testing

### User Story 1 - Typed Hard-Memory Capture (Priority: P1)

As a runtime architect, I need Vicuña to archive typed hard-memory primitives
instead of a single generic perturbation blob, so the persistent store captures
the actual artifacts the runtime already generates.

**Why this priority**: Without a typed capture layer, the memory system cannot
represent trajectories, outcomes, user-model fragments, or tool observations in
an inspectable way.

**Independent Test**: A maintainer can drive active-loop, DMN, and tool events
and observe that the hard-memory archive layer emits bounded typed records with
source metadata and stable policy.

**Acceptance Scenarios**:

1. **Given** an admitted user or tool event with meaningful self-state change,
   **When** the hard-memory archive path runs, **Then** the runtime archives one
   or more typed memory primitives instead of only a single free-form record.
2. **Given** a DMN cycle with counterfactual, governance, or remediation state,
   **When** the cycle settles, **Then** trajectory and outcome primitives can be
   archived with explicit source and domain metadata.

---

### User Story 2 - Retrieval That Cooperates With Self-State And LoRA Bias (Priority: P1)

As Vicuña, I need retrieved hard-memory records to carry enough typed structure
that they can bias self-model extension, gain control, and memory-oriented LoRA
families more effectively than raw text alone.

**Why this priority**: Expanding the store is not enough if retrieval still
collapses everything back into undifferentiated text.

**Independent Test**: A maintainer can query hard memory and observe typed hit
metadata, richer promotion traces, and extension/gating summaries that reflect
retrieved primitive kinds and domains.

**Acceptance Scenarios**:

1. **Given** hard-memory hits representing trajectories, user-model fragments,
   and outcomes, **When** a query result is returned, **Then** the runtime
   parses and exposes typed hit metadata instead of only title/content text.
2. **Given** retrieved typed hits, **When** they are promoted into self-model
   extensions, **Then** the promotion logic uses kind/domain/salience metadata
   and remains backward-compatible with fixed-width gain-control summaries.

---

### User Story 3 - Extensible Tool And Memory Primitive Contract (Priority: P2)

As a maintainer, I need a documented and production-ready primitive contract so
future tools can archive user-model fragments, procedural outcomes, and other
state safely without redesigning hard memory again.

**Why this priority**: The bare-minimum shipped tools are still only hard-memory
query and CLI, but the memory system has to be extensible now.

**Independent Test**: A maintainer can use the public APIs and docs to archive
custom typed memory records and see them survive through archive traces and
query metadata parsing.

**Acceptance Scenarios**:

1. **Given** a host or tool integration, **When** it archives a typed memory
   primitive through the public contract, **Then** the runtime validates,
   clamps, and forwards the primitive with stable metadata.
2. **Given** documentation for memory extension, **When** a maintainer reads the
   README and server docs, **Then** they can tell which primitives exist, which
   should affect allostasis or gain, and how to add new ones safely.

### Edge Cases

- What happens when a memory archive transaction wants to emit more primitives
  than the per-event batch budget allows?
- How are invalid primitive kinds, NaNs, overlong content, or missing keys
  handled before outbound archival?
- What happens when the memory endpoint returns hits without typed metadata or
  with malformed metadata?
- How does the runtime avoid turning user-model or memory-context retrieval into
  allostatic objectives by accident?
- How does the system preserve fixed-width gain-control inputs when the number
  and kinds of stored hard-memory records grow over time?
- What happens when DMN or counterfactual archival is disabled but event
  archival remains enabled?

## Requirements

### Functional Requirements

- **FR-001**: The hard-memory subsystem MUST support a bounded typed primitive
  vocabulary for at least event fragments, trajectories, outcomes,
  tool-observations, user-model fragments, and self-model fragments.
- **FR-002**: The runtime MUST archive typed primitives from existing Vicuña
  state rather than inventing opaque memory text detached from runtime traces.
- **FR-003**: Each archived primitive MUST record stable provenance including
  source role/channel, semantic domain, salience/confidence, and a stable key
  or identifier.
- **FR-004**: The event-driven self-state archive path MUST keep working, but it
  MUST upgrade to emit a bounded primitive batch with metadata instead of only a
  single generic perturbation memory.
- **FR-005**: The cognitive-loop path MUST archive trajectory and outcome
  primitives when active-loop or DMN episodes settle and enough signal is
  present.
- **FR-006**: The hard-memory query result path MUST parse typed hit metadata
  when available and expose it through public structs.
- **FR-007**: Retrieval-to-self-model promotion MUST use typed hit kind/domain
  metadata when scoring or promoting memory-context extensions.
- **FR-008**: The hard-memory query path MUST produce a fixed-size retrieval
  summary that can cooperate with self-state gradient and functional LoRA
  routing without requiring unbounded raw result inspection.
- **FR-009**: The primitive contract MUST remain explicit and CPU-side, with
  validation and clamping before network submission.
- **FR-010**: The design MUST preserve bounded archive behavior through explicit
  caps on primitives-per-transaction, metadata sizes, and supported tags.
- **FR-011**: The public C API MUST expose the typed primitive definitions and
  archive/query traces needed for inspection and testing.
- **FR-012**: The default shipped tool set MUST remain the hard-memory query
  tool and CLI wrapper only, but the system MUST document how future tools can
  author new primitives safely.
- **FR-013**: Documentation MUST explain which runtime artifacts are archived by
  default and how that memory should cooperate with self-model and LoRA bias.
- **FR-014**: The change MUST ship with targeted regression tests for typed
  archival batching, query metadata parsing, retrieval summaries, and
  self-model/LoRA cooperation.
- **FR-015**: Architecture, working-paper, README, and server-development docs
  MUST be updated alongside code.

### Key Entities

- **Hard Memory Primitive**: A bounded typed record sent to or retrieved from
  Supermemory, representing one specific kind of durable artifact.
- **Hard Memory Archive Batch**: A single bounded transaction containing one or
  more primitives emitted from one runtime event or loop settlement.
- **Hard Memory Retrieval Summary**: A fixed-width summary of recent retrieved
  hits used by self-state and functional bias code.
- **Loop Trajectory Primitive**: A memory record describing a settled active or
  DMN episode, including action/tool/remediation context and outcome.
- **User-Model Primitive**: A durable representation of user preference,
  dissatisfaction, trust, autonomy tolerance, or related social state that is
  useful for future control.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Targeted tests can observe at least three distinct hard-memory
  primitive kinds emitted from the runtime.
- **SC-002**: Hard-memory query tests can observe typed metadata parsing on
  returned hits and a non-zero retrieval summary.
- **SC-003**: Retrieved typed hits can change self-model extension summaries and
  remain backward-compatible with the existing fixed-width functional-gating
  input path.
- **SC-004**: Loop settlement tests can verify that trajectory or outcome
  primitives are archived from active-loop or DMN traces when signal thresholds
  are met.
- **SC-005**: The feature passes targeted automated tests for hard memory,
  self-state, cognitive loop, and active-LoRA integration.

## Assumptions

- Supermemory remains the default backing store, and metadata/tags on archived
  memories are the practical mechanism for typed retrieval.
- The first implementation should improve both capture and retrieval structure
  without introducing a new planner or a new default tool.
- Retrieval cooperation with LoRA bias should happen through bounded summaries
  and self-model extensions, not by exposing raw arbitrary memory lists to the
  gating MLP.
