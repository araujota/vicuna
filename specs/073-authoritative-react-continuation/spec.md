# Feature Specification: Authoritative ReAct Continuation

**Feature Branch**: `[073-authoritative-react-continuation]`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "why did it restrict itself to one turn? processes should have unlimited turns until an answer is found or the task is completed. it shouldn't have responded with the \"think\" part, the loop should have continued until it found an answer and responded with that. do some web/codebase research to figure out what's different about our ReAct loop that invites this problem, and fix it so this process can function properly"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Live-Fact Active Turns Continue Until Grounded (Priority: P1)

As a Telegram user, I need active ReAct turns that ask for current or dated facts to keep going until the runtime has a grounded answer, so the system does not stop after one hidden-thought pass with a procedural non-answer.

**Why this priority**: This is the direct production bug. The system currently accepts a first-pass `answer` even when the model has not actually obtained the requested information.

**Independent Test**: Send a Telegram-shaped active request for a live fact such as tomorrow’s weather, verify the first active step cannot terminate as a direct answer without a tool-grounded continuation, and verify the host trace shows tool use before the final visible answer.

**Acceptance Scenarios**:

1. **Given** an active user-origin turn that asks for current, dated, external, or otherwise mutable information, **When** the first authoritative ReAct step is prepared, **Then** the tool surface MUST remain visible and the generation contract MUST require a tool-capable continuation rather than accepting an unsupported direct answer.
2. **Given** an active user-origin turn that emits a procedural visible reply such as “I will use historical data” or “I don’t have real-time access” without a tool call, **When** the runtime validates the step, **Then** it MUST reject that step and continue the same authoritative loop instead of returning it to the user.
3. **Given** an active user-origin turn that completes one or more tool calls, **When** the final answer is eventually emitted, **Then** the user-visible reply MUST be produced only after the loop has admitted the tool observations needed to ground that answer.

---

### User Story 2 - Context-Grounded Stable Replies Can Still End Cleanly (Priority: P1)

As a user, I need stable or already-grounded requests to still complete without unnecessary tool churn, so the runtime remains responsive when the answer is already present in canonical context or does not depend on mutable external state.

**Why this priority**: Fixing the live-fact bug must not regress legitimate context-grounded answers or ordinary continuation.

**Independent Test**: Send a Telegram-shaped active request whose answer is already present in bounded Telegram dialogue or canonical context and verify the runtime can still accept a direct answer without forcing a new tool call.

**Acceptance Scenarios**:

1. **Given** an active user-origin turn whose needed answer is already present in bounded Telegram dialogue or canonical shared context, **When** the model emits a substantive direct answer, **Then** the runtime MAY accept that answer without requiring a new tool call.
2. **Given** an active user-origin turn whose request is stable and non-external, **When** the model emits a substantive direct answer, **Then** the runtime MUST still allow the turn to terminate normally.

---

### User Story 3 - Continuation Policy Is Explicit and Inspectable (Priority: P2)

As an operator, I need the multi-step continuation policy to be explicit in CPU control code, logs, and tests, so the runtime’s termination behavior is inspectable and does not collapse back into opaque prompt hope.

**Why this priority**: The bug exists because the current server path only retries on parse failure, not on unsupported terminal semantics.

**Independent Test**: Inspect the server logs, targeted unit tests, and architecture docs to verify that continuation, required tool-first cases, and terminal-answer rejection are documented and observable.

**Acceptance Scenarios**:

1. **Given** an authoritative ReAct step that is structurally valid but semantically unsupported as a terminal answer, **When** the runtime evaluates it, **Then** the logs MUST record that the step was rejected for continuation rather than silently treating it as complete.
2. **Given** a malformed or unsupported active terminal step, **When** the runtime retries, **Then** the same authoritative turn MUST continue with feedback instead of creating a parallel transcript or CPU-selected action.
3. **Given** the repository tests and docs, **When** this feature ships, **Then** they MUST describe and verify the continuation contract, grounding checks, and unbounded step policy.

### Edge Cases

- The latest user turn asks for live information, but the model emits a disclaimer or an intent statement instead of a tool call.
- The user repeats a question whose answer may already be present in bounded Telegram dialogue; the runtime must not regress into stale replay or unsupported hallucinated recall.
- A post-tool step emits another tool call; the loop must continue without losing the current active episode.
- The model repeatedly emits structurally valid but semantically unsupported terminal answers; the loop must continue without leaking intermediate hidden-thought text to the user.
- DMN and active engagement must preserve the same authoritative continuation semantics even though their terminal actions differ.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The authoritative ReAct server loop MUST continue until it reaches a valid terminal condition rather than terminating after the first structurally parseable assistant step.
- **FR-002**: For active user-origin turns that request current, dated, external, runtime-state, repository-state, or otherwise mutable facts, the first authoritative ReAct step MUST require tool-grounded continuation unless the answer is already grounded in canonical context.
- **FR-003**: The runtime MUST reject procedural or meta-intent visible replies as terminal active answers when they do not actually answer the user’s request or when they defer work without emitting a tool call.
- **FR-004**: A rejected terminal step MUST remain within the same authoritative ReAct turn by retrying with explicit feedback rather than selecting a tool or answer on the model’s behalf.
- **FR-005**: The continuation policy MUST stay explicit in CPU-side control code and MUST NOT reintroduce CPU-side tool or action selection authority.
- **FR-006**: Active and DMN authoritative ReAct loops MUST use an effectively unbounded continuation budget by default instead of a small fixed retry cap.
- **FR-007**: The server task state MUST retain the latest foreground user text needed to apply explicit continuation and grounding policy.
- **FR-008**: The prompt builder MUST be able to switch tool choice from `auto` to `required` for active turns that clearly need fresh tool grounding.
- **FR-009**: Context-grounded direct answers MUST remain allowed for stable requests and for answers already present in canonical shared context or bounded Telegram dialogue.
- **FR-010**: The repository MUST ship targeted automated tests for live-fact grounding detection, procedural-answer rejection, and continuation/retry policy.
- **FR-011**: Operator and architecture documentation MUST explain why a turn continues, when a tool-first step is required, and how the runtime determines valid terminal completion.

### Key Entities *(include if feature involves data)*

- **Foreground Turn Text**: The latest raw user-facing text for the active turn, retained on the task so continuation policy can reason about whether the request needs fresh tool grounding.
- **Continuation Policy**: The explicit CPU-side validation layer that decides whether a parsed authoritative ReAct step is terminal, must continue, or must be retried with feedback.
- **Grounded Terminal Answer**: A visible answer accepted only when it is substantive and supported either by current canonical context or by admitted tool observations for the ongoing turn.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A host-traced Telegram weather-style request no longer terminates on the first pass with `tool_calls=0` and a procedural non-answer.
- **SC-002**: A host-traced Telegram weather-style request shows at least one authoritative tool call before the final user-visible answer when fresh external data is required.
- **SC-003**: A targeted regression test proves that procedural meta-replies such as “I will use historical data” are rejected as terminal active answers.
- **SC-004**: A targeted regression test proves that stable or already-grounded answers can still terminate without a new tool call.
