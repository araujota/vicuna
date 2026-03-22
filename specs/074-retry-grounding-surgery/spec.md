# Feature Specification: Retry Grounding Surgery

**Feature Branch**: `[070-truth-runtime-refactor]`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "so we have a functioning loop there's just this other weird issue. see if you can surgically fix whatever caused this retry error while keeping the functional react loop intact"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fresh Mutable Turns Do Not Trust Stale Telegram Assistant Replies (Priority: P1)

As a Telegram user, I need fresh live-fact turns to ignore stale assistant prose from prior turns as grounding, so the runtime does not loop on unsupported direct answers when a tool call is still needed.

**Why this priority**: This is the direct production failure. A mutable question can inherit bad grounding from old Telegram assistant replies and then spin inside the continuation loop.

**Independent Test**: Replay a Telegram-shaped weather request after prior bad weather answers exist in the same dialogue scope and verify the fresh turn still chooses a tool path instead of treating the old assistant prose as grounded context.

**Acceptance Scenarios**:

1. **Given** a fresh active Telegram turn that asks for mutable external information such as tomorrow's weather, **When** the runtime evaluates whether the exact answer is already grounded, **Then** stale assistant dialogue text from prior turns MUST NOT count as sufficient grounding on its own.
2. **Given** canonical shared-context tool observations from the current active episode or resumed tool chain, **When** the runtime checks grounding, **Then** those admitted observations MAY still satisfy grounding for a final answer.
3. **Given** bounded Telegram dialogue contains earlier assistant answers about the same topic, **When** a new fresh mutable turn begins, **Then** those replies MUST NOT suppress a needed first tool call.

---

### User Story 2 - Repeated Procedural Non-Answers Escalate to Tool Requirement (Priority: P1)

As an operator, I need the active ReAct loop to escalate after repeated invalid direct answers, so the loop converges instead of spinning for dozens of retries.

**Why this priority**: The runtime currently remains alive but can spend more than a minute rejecting disclaimer-style answers without ever forcing the available tool path.

**Independent Test**: Simulate repeated procedural non-answer retries for a mutable active turn and verify the prompt/tool-choice policy escalates to a required tool call before unbounded retries continue indefinitely.

**Acceptance Scenarios**:

1. **Given** an active mutable turn that has already failed terminal validation multiple times without emitting a tool call, **When** the next prompt is prepared, **Then** tool choice MUST escalate from `auto` to `required`.
2. **Given** escalation has triggered, **When** retry feedback is injected into the authoritative prompt, **Then** it MUST explicitly demand one fresh tool call rather than allowing another unsupported direct answer.
3. **Given** a resumed turn already has a fresh tool observation, **When** the next step is prepared, **Then** escalation MUST NOT force an unnecessary extra tool call.

---

### User Story 3 - The Existing ReAct Continuation Loop Remains Intact (Priority: P2)

As a maintainer, I need the fix to preserve the current authoritative ReAct continuation and tool-resume flow, so we correct the retry pathology without reintroducing CPU-side action selection or parallel transcript logic.

**Why this priority**: The loop now works structurally. The fix must stay surgical.

**Independent Test**: Verify local regression tests and a host trace still show think -> tool XML -> tool execution -> resumed answer on a live-fact Telegram turn.

**Acceptance Scenarios**:

1. **Given** a live-fact Telegram turn with no trusted grounding, **When** the model first responds, **Then** the authoritative loop MUST still own tool selection and continuation.
2. **Given** a resumed post-tool turn, **When** the model emits a grounded answer, **Then** the runtime MUST accept it without new CPU-selected direction.
3. **Given** this feature ships, **When** operators inspect code and docs, **Then** the retry-escalation and grounding-trust rules MUST be explicit and inspectable.

### Edge Cases

- The same chat scope already contains a stale assistant answer with overlapping terms and digits.
- The latest mutable request has no current tool observation yet, but the canonical shared context contains older unrelated tool observations.
- The model keeps producing structurally valid `Action: answer` non-answers on retries.
- A resumed tool-result turn must still answer directly if the admitted observation already contains the needed result.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Grounding checks for fresh mutable active turns MUST NOT treat Telegram assistant dialogue alone as sufficient grounding.
- **FR-002**: Grounding checks MUST continue to trust canonical tool observations and same-turn canonical runtime artifacts.
- **FR-003**: The authoritative prompt builder MUST escalate active mutable retries to `tool_choice=required` after a bounded number of failed non-tool retries.
- **FR-004**: Retry feedback for escalated mutable active turns MUST explicitly require exactly one fresh tool call.
- **FR-005**: The fix MUST NOT reintroduce CPU-side tool or action selection authority.
- **FR-006**: Resumed post-tool turns MUST remain eligible to answer directly from admitted tool observations.
- **FR-007**: Automated tests MUST cover stale Telegram grounding rejection and retry escalation.
- **FR-008**: Operator-facing documentation MUST describe the trusted-grounding hierarchy and retry escalation rule.

### Key Entities *(include if feature involves data)*

- **Trusted Grounding Candidate**: A canonical context artifact that may justify a direct answer for the current turn, limited to tool observations and same-turn runtime artifacts for fresh mutable Telegram turns.
- **Mutable Active Retry Escalation**: The explicit rule that forces `tool_choice=required` after repeated unsupported direct-answer retries on a mutable active turn.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A Telegram-shaped mutable live-fact request with stale prior assistant answers no longer loops for dozens of retries before completion.
- **SC-002**: The same request path shows a tool call before the final answer unless a trusted current tool observation already exists.
- **SC-003**: New regression tests prove stale Telegram assistant dialogue does not count as fresh grounding for mutable active turns.
- **SC-004**: New regression tests prove retry escalation forces required tool choice after repeated invalid non-tool retries.
