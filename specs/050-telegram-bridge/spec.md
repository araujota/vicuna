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

### User Story 3 - Persist Bridge Cursor And Subscriptions (Priority: P2)

As an operator, I need bridge state to survive process restarts so it does not
replay Telegram updates or forget which chats should receive self-emits.

**Why this priority**: Restart-safe relay behavior reduces duplicate messages
and keeps proactive delivery stable.

**Independent Test**: After a restart, the bridge restores the Telegram update
offset and subscribed chat IDs from its local state file.

**Acceptance Scenarios**:

1. **Given** the bridge has already processed Telegram updates, **When** it
   restarts, **Then** it resumes from the stored update offset.
2. **Given** one or more Telegram chats were registered previously, **When**
   the bridge restarts, **Then** proactive self-emits continue to fan out to
   those chats.

## Edge Cases

- Telegram sends non-text updates such as stickers, photos, or edited messages.
- The Vicuña server is temporarily unavailable while the bridge is polling.
- The proactive SSE stream disconnects or times out.
- The runtime emits a proactive response without text content.
- The bridge restarts after processing a Telegram update but before persisting
  the latest cursor.

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
  IDs, and recently relayed proactive response IDs to a local state file.
- **FR-007**: The bridge MUST expose runtime configuration through explicit
  environment variables rather than hard-coded endpoints.
- **FR-008**: The implementation MUST include tests for the stream parsing and
  text extraction logic used by the bridge.
- **FR-009**: The implementation MUST include operator-facing documentation for
  starting the bridge against the local GPU-enabled server.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Starting the bridge succeeds with only Node 20 and the configured
  environment variables.
- **SC-002**: A Telegram text update results in exactly one relayed Vicuña
  response message in the same chat.
- **SC-003**: A proactive response observed on `/v1/responses/stream` is sent
  to at least one subscribed Telegram chat.
- **SC-004**: The bridge restores its saved Telegram offset and subscriptions
  after restart.
