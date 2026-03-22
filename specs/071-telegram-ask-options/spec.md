# Feature Specification: Telegram Ask-With-Options Tool

**Feature Branch**: `071-telegram-ask-options`  
**Created**: 2026-03-21  
**Status**: In Progress  
**Input**: User description: "add an ask-with-options tool for active and DMN use through the Telegram bridge, using reply_markup options and resuming once the user selects an option"

## User Scenarios & Testing

### User Story 1 - Tooled Clarification with Telegram Options (Priority: P1)

As the runtime, I need an `ask_with_options` tool that can be chosen during authoritative ReAct turns in active and DMN execution, so clarification and discrete user choice can be requested through Telegram with inline option buttons.

**Why this priority**: This is the feature itself. Without a first-class tool, clarification-by-choice remains outside the authoritative tool system.

**Independent Test**: Expose the tool in the authoritative tool surface, parse one XML tool call for it, dispatch it through the Telegram transport, and verify the bridge sends a Telegram message with inline reply markup.

**Acceptance Scenarios**:

1. **Given** an active or DMN authoritative turn that needs clarification or a constrained choice, **When** the model emits an `ask_with_options` tool call, **Then** the runtime validates and dispatches that tool call through the same OpenClaw tool path as other tools.
2. **Given** a valid `ask_with_options` request, **When** the bridge delivers it, **Then** the Telegram message contains the question text plus inline keyboard options via `reply_markup`.
3. **Given** a running host with the feature deployed, **When** operator logs or tool catalogs are inspected, **Then** the runtime reports the new tool as available to the authoritative ReAct loop.

### User Story 2 - Selection-Driven Continuation (Priority: P1)

As the runtime, I need the user’s selected Telegram option to re-enter the existing message-aware dialogue flow, so the system can continue from the choice without a separate side-channel workflow.

**Why this priority**: Sending buttons without resumption is incomplete. The choice must become usable dialogue state.

**Independent Test**: Deliver an ask-with-options message, click one option, and verify the bridge records the selected choice into bounded transcript state and triggers a resumed Telegram-origin turn.

**Acceptance Scenarios**:

1. **Given** a pending ask-with-options message, **When** the user clicks an option, **Then** the bridge acknowledges the callback, resolves the selected option against stored prompt state, and removes or disables the keyboard to prevent duplicate selection.
2. **Given** a resolved selection, **When** the bridge forwards the resumed turn to the runtime, **Then** the latest bounded transcript includes the prior assistant question and the selected user answer in a form the ReAct loop can continue from.
3. **Given** a resumed selection turn, **When** the runtime rebuilds Telegram dialogue memory, **Then** the dedicated Telegram dialogue object remains the authoritative user-facing memory surface for that resumed exchange.

### User Story 3 - Bounded Transport and Memory State (Priority: P2)

As an operator, I need the Telegram ask-with-options delivery path to use explicit bounded state for outbound prompts and pending option mappings, so transport remains inspectable and recoverable instead of relying on hidden prompt hacks.

**Why this priority**: The bridge and runtime already have bounded mailbox and transcript state. This new interactive transport should follow the same rule.

**Independent Test**: Inspect runtime and bridge state after sending and selecting an ask-with-options prompt and verify the pending option state is bounded, typed, and cleaned up after use.

**Acceptance Scenarios**:

1. **Given** one or more pending ask-with-options deliveries, **When** state is inspected, **Then** outbound prompt metadata and pending option selections are stored in dedicated bounded objects rather than ad hoc string parsing.
2. **Given** a completed option selection, **When** cleanup runs, **Then** stale pending option state is removed and the transcript remains coherent.
3. **Given** no Telegram scope is available for an ask-with-options dispatch, **When** the tool is invoked, **Then** the runtime fails the tool cleanly instead of silently dropping the request.

### Edge Cases

- A callback arrives after the pending option prompt was already consumed or evicted from bridge state.
- A DMN-origin ask-with-options request exists while more than one Telegram chat scope is available and no unique target can be resolved.
- The active Telegram turn intentionally returns no direct assistant text because user-visible output was delivered through the ask-with-options tool.
- A user taps the same option more than once or taps after the keyboard was already removed.
