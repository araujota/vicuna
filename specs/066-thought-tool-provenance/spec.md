# Feature Specification: Full Thought and Tool Provenance Logging

**Feature Branch**: `066-thought-tool-provenance (on current 060-service-user-migration worktree)`  
**Created**: 2026-03-20  
**Status**: Draft  
**Input**: User description: "ensure that full narration from active and dmn thought processes is captured by the system's logs, along with exact tool calls"

## User Scenarios & Testing

### User Story 1 - Active loop narration is durably logged (Priority: P1)

As a runtime operator, I need the active loop's planner narration to be present in
the unified provenance logs so I can inspect what the system reasoned before it
answered or invoked a tool.

**Why this priority**: Active reasoning is currently reduced to summary fields in
provenance, which is insufficient for debugging repeated or surprising behavior.

**Independent Test**: Trigger an active loop turn and verify the provenance
event contains the full planner reasoning text and plan detail instead of only a
winner-action summary.

**Acceptance Scenarios**:

1. **Given** an active loop turn that emits a direct answer, **When** the final
   response is logged, **Then** provenance records the planner narration text,
   plan structure, and visible assistant output.
2. **Given** an active loop turn that invokes a tool, **When** the tool call is
   staged, **Then** provenance records the planner narration and the exact tool
   request payload before host execution begins.

### User Story 2 - DMN narration is durably logged (Priority: P1)

As a runtime operator, I need the DMN's translated prompt and current plan state
to be present in provenance so background cognition can be inspected as a real
thought process rather than a terse tick summary.

**Why this priority**: The DMN now runs through a prompt-revision architecture,
but its logged output is still summary-oriented.

**Independent Test**: Trigger a DMN tick and verify provenance records the
rendered prompt revision, concept outline, plan details, and decision context.

**Acceptance Scenarios**:

1. **Given** an admitted DMN tick, **When** provenance is appended, **Then** it
   contains the rendered DMN prompt, concept frames, plan details, and candidate
   reasoning context.
2. **Given** a DMN tick that chooses a tool or relay, **When** the request is
   dispatched, **Then** provenance records the exact structured request payload
   tied to the current DMN trace.

### User Story 3 - Exact tool calls are logged before execution (Priority: P1)

As a maintainer, I need structured request payloads for every runtime tool call
to be recorded before execution so tool behavior can be reconstructed exactly
from logs.

**Why this priority**: Current provenance records tool results, but not the
exact requests that produced them.

**Independent Test**: Dispatch each supported runtime tool family through the
server path and verify a `tool_call` provenance event is appended with the exact
request payload and origin metadata.

**Acceptance Scenarios**:

1. **Given** an active or DMN bash, hard-memory, Codex, or Telegram tool
   request, **When** the server dispatches it, **Then** a structured `tool_call`
   provenance event is appended before host-side execution.
2. **Given** a tool result event later arrives, **When** operators inspect the
   provenance log, **Then** the result can be matched to the earlier request via
   command/job identifiers.

## Requirements

### Functional Requirements

- **FR-001**: The unified provenance repository MUST record the full active-loop
  planner narration when it is available.
- **FR-002**: Active-loop provenance MUST include plan-step detail and candidate
  detail sufficient to reconstruct the decision path.
- **FR-003**: The unified provenance repository MUST record the DMN rendered
  prompt revision and concept-level narration already produced by the
  self-model-to-language translator.
- **FR-004**: DMN provenance MUST include plan-step detail and candidate detail
  sufficient to reconstruct the current background decision path.
- **FR-005**: The server MUST append a dedicated `tool_call` provenance event
  before executing any supported runtime tool request.
- **FR-006**: `tool_call` provenance MUST include exact structured request
  payloads for bash, hard-memory, Codex, and Telegram relay requests.
- **FR-007**: Tool-call provenance MUST include origin metadata and command/job
  identifiers so request and result events can be correlated.
- **FR-008**: Existing `tool_result` provenance behavior MUST be preserved.
- **FR-009**: Operator docs MUST describe where full narration and exact
  tool-call payloads are captured and how to inspect them.
- **FR-010**: Targeted tests MUST verify active narration logging, DMN
  narration logging, and exact tool-call provenance capture.

## Success Criteria

- **SC-001**: Provenance events for active turns expose full planner narration
  text rather than only action summaries.
- **SC-002**: Provenance events for DMN ticks expose the rendered prompt
  revision, concept details, and plan details.
- **SC-003**: Every supported runtime tool dispatch produces a `tool_call`
  provenance event with exact request payloads before the corresponding result.
- **SC-004**: Targeted native and server tests pass with provenance assertions
  for narration and exact tool calls.
