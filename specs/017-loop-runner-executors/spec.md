# Feature Specification: Self-Running Planner-Executors For Active And DMN Loops

**Feature Branch**: `017-loop-runner-executors`  
**Created**: 2026-03-11  
**Status**: Draft  
**Input**: User description: "research and investigate how to build the proper runner for this loop system we are constructing, using your research and speccing flow, then implement the self-running planner-executors for active engagement and DMN loops that are prepared to function in their intended modes"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Event-Driven Foreground Runner (Priority: P1)

As a Vicuña runtime architect, I need the active loop to be backed by a real
bounded planner-executor runner rather than a one-shot trace only, so
foreground interaction can persist episode state, schedule emits or tool work,
and resume coherently on tool observations.

**Why this priority**: The current foreground loop can score an action and emit
trace metadata, but it does not maintain a durable execution command stream or
runner state beyond that decision.

**Independent Test**: A maintainer can inspect runtime state and see a
foreground runner that produces explicit pending commands, updates episode
status, and resumes on tool observations without changing the top-level action
policy.

**Acceptance Scenarios**:

1. **Given** a user event that wins `ACT`, **When** the active runner executes,
   **Then** it schedules a bounded tool command, marks itself waiting on tool,
   and exposes that command through a public API.
2. **Given** a tool-completion event for an active episode, **When** the active
   runner resumes, **Then** it records an observation, advances the episode, and
   produces the correct next emit or finish command.

---

### User Story 2 - Maintenance-Oriented DMN Runner (Priority: P1)

As a Vicuña architect, I need the DMN to run as a bounded maintenance
planner-executor, so it can perform local internal steps, schedule tool or
emit commands, and continue internally within budget instead of stopping after
the first selected action.

**Why this priority**: DMN is intended to be a pressure-driven background
process, not a single scored action. The runner should therefore be able to
execute internal work and re-plan within a bounded tick.

**Independent Test**: A maintainer can inspect DMN traces and runner state and
see bounded internal continuation, pending command generation, and persistent
waiting state for tool-based DMN work.

**Acceptance Scenarios**:

1. **Given** a DMN tick that chooses `INTERNAL_WRITE`, **When** the runner
   executes, **Then** it can continue within the same bounded run and produce a
   follow-on terminal or external command if pressure and governance still
   justify it.
2. **Given** a DMN tick that chooses `INVOKE_TOOL`, **When** the runner
   executes, **Then** it schedules a tool command with remediation context and
   remains resumable for later observation handling.

---

### User Story 3 - Host-Visible Pending Commands And Status (Priority: P2)

As a host integrator, I need a public queue of pending loop commands and runner
status, so `llama-server` and future tool hosts can actually execute the
planner-executor outputs instead of only logging traces.

**Why this priority**: Without a host-visible command surface, the runner
cannot actually orchestrate foreground or DMN work.

**Independent Test**: A maintainer can call public APIs to inspect pending loop
commands, acknowledge execution, and observe runner state transitions.

**Acceptance Scenarios**:

1. **Given** a runner-generated emit or tool command, **When** the host polls
   the runtime, **Then** it can inspect the command kind, origin loop, episode
   identity, and safety/tool metadata.
2. **Given** a completed host command, **When** the runtime is notified,
   **Then** the associated runner state updates and any follow-on planning can
   continue within its bounded policy.

### Edge Cases

- What happens if a loop wants to continue but its step budget is exhausted?
- What happens if a tool command is scheduled for a tool kind that has no host
-side executor yet?
- How should the DMN behave when governance denies an otherwise valid follow-on
  command after an internal write?
- How should the active runner avoid recursively self-triggering after its own
  emit?
- How should the host distinguish actionable pending commands from diagnostic
  trace state only?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime MUST add durable planner-executor runner state for
  active and DMN loops.
- **FR-002**: The runtime MUST expose host-visible pending commands for emits,
  tool invocations, and internal maintenance actions.
- **FR-003**: The active runner MUST resume on relevant tool observations and
  preserve foreground episode continuity.
- **FR-004**: The DMN runner MUST support bounded internal continuation within a
  single admitted run.
- **FR-005**: The runner MUST preserve the existing explicit top-level scoring,
  favorable-state, counterfactual, remediation, governance, and LoRA update
  order.
- **FR-006**: The runtime MUST expose command acknowledgment/completion APIs so
  host integrations can notify the runner when work was executed.
- **FR-007**: The public API MUST distinguish active and DMN command origin and
  provide enough metadata for future host tool execution.
- **FR-008**: The implementation MUST remain bounded: explicit max steps, no
  unconstrained recursive loops, and clear terminal reasons.
- **FR-009**: The implementation MUST preserve parity with current active and
  DMN winner-action behavior where possible.
- **FR-010**: The implementation MUST update tests and architecture docs.

### Key Entities

- **Loop Runner Status**: Persistent state for one active episode or DMN run,
  including current phase, budget, wait state, and terminal status.
- **Loop Command**: Host-visible executable output from a runner, such as emit,
  ask, invoke tool, internal write, or background notify.
- **Loop Command Queue**: Bounded FIFO of pending runner outputs.
- **Command Completion Record**: Host acknowledgment that a command was handled,
  allowing the runner to continue.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A maintainer can inspect public APIs and see persistent runner
  state plus host-visible pending commands.
- **SC-002**: Active-loop tests demonstrate `ACT` scheduling and tool-result
  resumption.
- **SC-003**: DMN tests demonstrate bounded internal continuation and command
  scheduling.
- **SC-004**: The targeted regression suite still passes after runner
  implementation.

## Assumptions

- This pass builds the runner substrate and basic host-facing command flow, not
  a full production external tool orchestration layer.
- The existing `llama-server` integration can remain lightweight as long as the
  runtime surfaces are now complete enough to support future host execution.
