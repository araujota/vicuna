# Feature Specification: Telegram Bridge Middleware

**Feature Branch**: `049-host-build-bringup`  
**Created**: 2026-03-17  
**Status**: Draft  
**Input**: User description: "architect and build a bridge server to connect to the telegram bot set up to talk to this system remotely"

## User Scenarios & Testing

### User Story 1 - Remote Telegram Conversation (Priority: P1)

As an operator, I need a Telegram bot to send user messages into the running
Vicuña server so I can talk to the system remotely.

**Why this priority**: Without inbound Telegram relay, there is no remote chat
surface for the live runtime.

**Independent Test**: A Telegram update containing text is converted into a
request to the local Vicuña server and the resulting assistant text is sent
back to the originating Telegram chat.

**Acceptance Scenarios**:

1. **Given** the bridge is running and a Telegram user sends a text message,
   **When** the bridge polls the Bot API, **Then** it forwards that text to the
   configured Vicuña API and replies into the same Telegram chat.
2. **Given** the Telegram user sends `/start`, **When** the bridge receives the
   update, **Then** the chat is registered as a relay target for future system
   emissions.

### User Story 2 - Relay Proactive Self-Emits (Priority: P1)

As an operator, I need unsolicited runtime self-emissions to reach Telegram
through the middleware so the remote bot sees system-originated updates too.

**Why this priority**: The user explicitly requires self-emissions to pass
through the middleware rather than being trapped in the local API.

**Independent Test**: When the runtime publishes an event on
`/v1/responses/stream`, the bridge relays the completed response text to all
registered Telegram chats.

**Acceptance Scenarios**:

1. **Given** the bridge is connected to `/v1/responses/stream`, **When** the
   runtime publishes a proactive response, **Then** the bridge parses the
   `response.completed` event and sends the resulting text to Telegram.
2. **Given** the self-emission stream disconnects, **When** the bridge
   reconnects, **Then** it resumes streaming without manual intervention.

### User Story 3 - Preserve Per-Chat Conversation State (Priority: P1)

As a Telegram user, I need the bridge to remember the recent turns in my chat,
so follow-up questions like "what was my last message?" resolve against my
actual conversation instead of only the latest turn.

**Why this priority**: The current bridge is stateless between Telegram turns,
which breaks basic conversational continuity.

**Independent Test**: After sending at least two user messages in the same
Telegram chat, the bridge forwards enough persisted context for the runtime to
answer a question about the earlier turn correctly.

**Acceptance Scenarios**:

1. **Given** a Telegram chat has already exchanged user and assistant turns,
   **When** the same chat sends a follow-up message, **Then** the bridge
   forwards a bounded transcript for that specific chat ID instead of only the
   newest user message.
2. **Given** two different Telegram chats are using the same bridge,
   **When** one chat asks about its earlier messages, **Then** the bridge MUST
   isolate transcript state by Telegram chat ID and MUST NOT leak turns from
   the other chat.

### User Story 4 - Persist Bridge Cursor, Subscriptions, And Chat State (Priority: P2)

As an operator, I need bridge state to survive process restarts so it does not
replay Telegram updates, forget which chats should receive self-emits, or lose
per-chat conversation continuity.

**Why this priority**: Restart-safe relay behavior reduces duplicate messages
and keeps proactive delivery stable.

**Independent Test**: After a restart, the bridge restores the Telegram update
offset, subscribed chat IDs, and bounded per-chat transcript state from its
local state file.

**Acceptance Scenarios**:

1. **Given** the bridge has already processed Telegram updates, **When** it
   restarts, **Then** it resumes from the stored update offset.
2. **Given** one or more Telegram chats were registered previously, **When**
   the bridge restarts, **Then** proactive self-emits continue to fan out to
   those chats.
3. **Given** a Telegram chat already has persisted turns, **When** the bridge
   restarts and the same chat sends a follow-up, **Then** the bridge restores
   and reuses that chat-specific transcript state.

### User Story 5 - Expose Runtime Command Tool Availability (Priority: P2)

As an operator, I need the managed runtime launch path to expose the cognitive
bash tool explicitly, so Telegram command-execution requests can succeed when
the runtime decides to invoke that tool.

**Why this priority**: The server owns the bounded CLI tool implementation, but
the repo-owned launcher does not currently configure it explicitly.

**Independent Test**: Starting the managed runtime through the repo launcher
logs that the bash tool is enabled with explicit workdir and limits, and a
command-execution request no longer fails solely because the tool path is
disabled.

**Acceptance Scenarios**:

1. **Given** the repo-owned runtime launcher is used, **When** the service
   starts, **Then** it sets explicit bash-tool environment variables instead of
   depending on an untracked external shell configuration.
2. **Given** the bash tool is intentionally disabled or misconfigured,
   **When** the runtime starts, **Then** the operator-facing docs describe the
   exact variables that control command execution availability.

## Edge Cases

- Telegram sends non-text updates such as stickers, photos, or edited messages.
- The Vicuña server is temporarily unavailable while the bridge is polling.
- The proactive SSE stream disconnects or times out.
- The runtime emits a proactive response without text content.
- The bridge restarts after processing a Telegram update but before persisting
  the latest cursor.
- Multiple Telegram chats talk to the same runtime concurrently.
- A single Telegram chat grows long enough that the bridge must trim older
  turns while preserving enough context to stay coherent.
- The runtime is reachable but its bash tool is disabled, so command requests
  need clear operator-visible configuration rather than silent failure.

## Requirements

### Functional Requirements

- **FR-001**: The work MUST create current Spec Kit artifacts for the Telegram
  bridge before implementation.
- **FR-002**: The bridge MUST poll the Telegram Bot API for inbound updates and
  handle plain text messages.
- **FR-003**: The bridge MUST forward inbound Telegram text messages to the
  local Vicuña HTTP API and send the assistant response back to the originating
  Telegram chat.
- **FR-004**: The bridge MUST maintain a live subscription to
  `/v1/responses/stream` so proactive self-emits are relayed through the
  middleware.
- **FR-005**: The bridge MUST relay completed proactive response text to every
  registered Telegram chat.
- **FR-006**: The bridge MUST persist Telegram update offsets, subscribed chat
  IDs, recently relayed proactive response IDs, and bounded per-chat
  conversation transcripts keyed by Telegram chat ID to a local state file.
- **FR-007**: The bridge MUST bound each stored chat transcript explicitly and
  trim older turns deterministically.
- **FR-008**: The bridge MUST isolate conversation state by Telegram chat ID.
- **FR-009**: The bridge MUST expose runtime configuration through explicit
  environment variables rather than hard-coded endpoints.
- **FR-010**: The repo-owned runtime launcher MUST configure the cognitive bash
  tool through explicit environment variables when managed runtime operation is
  used for Telegram command execution.
- **FR-011**: The implementation MUST include tests for transcript
  persistence/bounding plus the stream parsing and text extraction logic used by
  the bridge.
- **FR-012**: The implementation MUST include operator-facing documentation for
  starting the bridge against the local GPU-enabled server, including the bash
  tool configuration path.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Starting the bridge succeeds with only Node 20 and the configured
  environment variables.
- **SC-002**: A Telegram text update results in exactly one relayed Vicuña
  response message in the same chat.
- **SC-003**: A proactive response observed on `/v1/responses/stream` is sent
  to at least one subscribed Telegram chat.
- **SC-004**: After at least one prior turn, asking the same Telegram chat what
  its last message was yields a context-aware answer instead of an amnesia
  failure.
- **SC-005**: The bridge restores its saved Telegram offset, subscriptions, and
  bounded per-chat transcript state after restart.
- **SC-006**: The repo-owned runtime launcher starts with explicit bash-tool
  configuration, so command execution is not blocked solely by missing launcher
  environment.
