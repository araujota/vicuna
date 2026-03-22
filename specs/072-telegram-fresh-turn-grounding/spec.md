# Feature Specification: Telegram Fresh-Turn Grounding

**Feature Branch**: `[072-telegram-fresh-turn-grounding]`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "the agent is repeatedly returning the same response to an old question with subsequent messages. investigate the cause of this and fix it completely."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Telegram replies stay grounded in the latest user turn (Priority: P1)

As a Telegram user, I need each new message to be answered from my latest turn rather than from a stale earlier question, so follow-up conversation remains coherent.

**Why this priority**: The bug breaks the primary user-facing loop and makes the runtime appear stuck on old context.

**Independent Test**: Can be tested by sending two different Telegram-shaped chat requests in sequence and verifying the second active turn is seeded from the second request text rather than the earlier one.

**Acceptance Scenarios**:

1. **Given** a Telegram-scoped active conversation with a prior answered question, **When** a new Telegram user message arrives, **Then** the active foreground event MUST be derived from the latest raw incoming message body rather than from the transformed prompt string.
2. **Given** a Telegram-scoped active conversation with shared hidden/tool context from prior turns, **When** the authoritative ReAct prompt is rebuilt, **Then** the latest Telegram dialogue turn MUST remain the terminal conversational turn seen by the model.

---

### User Story 2 - Telegram continuity does not regress under ReAct reconstruction (Priority: P2)

As an operator, I need Telegram dialogue continuity to remain bounded and inspectable without allowing old hidden/tool residues to override the current turn.

**Why this priority**: The bug source sits at the boundary between Telegram dialogue memory and the unified shared context window.

**Independent Test**: Can be tested by rebuilding canonical ReAct messages for a Telegram-scoped task and verifying the message order keeps the latest user turn last while still preserving older hidden/tool context.

**Acceptance Scenarios**:

1. **Given** a Telegram-scoped task with dialogue history plus shared hidden/tool items, **When** canonical ReAct messages are assembled, **Then** shared hidden/tool items MUST appear before the bounded Telegram dialogue slice instead of after it.
2. **Given** a non-Telegram completion request, **When** foreground extraction runs, **Then** existing non-Telegram prompt behavior MUST remain unchanged.

## Edge Cases

- Raw request JSON may be invalid or may not contain `messages`; the runtime must fall back cleanly to the parsed body behavior.
- Telegram-scoped requests with only one user message and no assistant history must still keep that user turn as the final conversational message before assistant generation.
- Shared context may contain recent hidden reasoning or tool results from unrelated chats; those residues must not displace the latest Telegram user turn in prompt order.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The server MUST derive active foreground role and foreground text from the raw incoming request body whenever that raw body contains `messages`.
- **FR-002**: The server MUST fall back to the already-parsed body only when the raw body does not provide usable message content.
- **FR-003**: For Telegram-scoped authoritative ReAct prompt reconstruction, bounded Telegram dialogue messages MUST be appended after any retained shared hidden/tool/internal context so the latest Telegram turn remains terminal.
- **FR-004**: For Telegram-scoped authoritative ReAct prompt reconstruction, shared-context user-visible turns MUST continue to be suppressed when runtime Telegram dialogue is present, avoiding duplicate visible dialogue.
- **FR-005**: The fix MUST preserve current bounded dialogue-memory behavior and MUST NOT reintroduce any CPU-side action or tool selection authority.
- **FR-006**: The repository MUST ship targeted regression coverage for the raw-body foreground extraction path and the stale-turn Telegram regression.
- **FR-007**: Operator and architecture documentation MUST be updated to explain how Telegram-scoped active turns are grounded and ordered under authoritative ReAct reconstruction.

### Key Entities *(include if feature involves data)*

- **Foreground Request Body**: The untransformed HTTP request JSON used to recover the latest user/tool turn before chat-template conversion.
- **Telegram Dialogue Slice**: The bounded runtime-owned per-chat transcript used for Telegram-visible continuity.
- **Canonical ReAct Prompt Order**: The ordered message sequence combining shared hidden/tool/internal context with Telegram dialogue for active or DMN generation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a targeted regression test, a Telegram-scoped second request with different user text is seeded from the second request text rather than the first.
- **SC-002**: In a targeted regression test, Telegram-scoped canonical ReAct ordering leaves the latest Telegram user message as the last conversational message before assistant generation.
- **SC-003**: After host rebuild, the live runtime no longer repeats the prior Tesla answer when given a fresh Telegram-scoped prompt with different content.
