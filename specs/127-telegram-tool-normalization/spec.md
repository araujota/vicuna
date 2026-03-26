# Feature Specification: Telegram Tool Normalization

**Feature Branch**: `127-telegram-tool-normalization`  
**Created**: 2026-03-26  
**Status**: Draft  
**Input**: User description: "implement the normalization of the telegram tool/methods/payloads. keep the methods a reasonable number that we're likely to actually want used. remove the telegram-specific system prompt entirely."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select Telegram through the same staged contract as every other tool family (Priority: P1)

As an operator, I want bridge-scoped Telegram requests to expose Telegram
through ordinary family, method, and payload metadata so the provider sees the
same staged contract shape it sees for Radarr, Sonarr, and other tools.

**Why this priority**: The current bridge path still treats Telegram as a
special-case relay tool plus a custom system prompt, which makes the first
family-selection turn heavier and harder to reason about.

**Independent Test**: A bridge-scoped request can reach the family stage and
the prompt contains Telegram only as a normal family description while the
method stage contains explicit Telegram methods with no bridge-only Telegram
system prompt.

**Acceptance Scenarios**:

1. **Given** a bridge-scoped Telegram request, **When** the server assembles
   the staged family-selection turn, **Then** the prompt lists Telegram only by
   family name and brief description alongside the other families.
2. **Given** the provider selects the Telegram family, **When** the server
   assembles method selection, **Then** only the Telegram methods for that
   family are shown and no bridge-specific Telegram instruction block is added.

---

### User Story 2 - Use explicit Telegram methods with typed payload contracts (Priority: P1)

As an operator, I want the Telegram family to expose a small set of explicit,
useful methods with typed payload contracts so the provider can choose the
right Telegram delivery shape without a generic catch-all relay schema.

**Why this priority**: One generic `telegram_relay` method forces the provider
to infer too much from prose and a broad payload shape. That defeats the point
of the staged method/payload split.

**Independent Test**: A bridge-scoped request can choose one Telegram method,
submit a matching typed payload, and the runtime queues the correct Telegram
outbox item without using a generic request wrapper.

**Acceptance Scenarios**:

1. **Given** the provider chooses a simple Telegram text reply, **When** it
   selects the Telegram method and submits the payload, **Then** the runtime
   queues a `sendMessage` outbox item derived from that method-specific
   contract.
2. **Given** the provider chooses a richer Telegram reply such as formatted
   text, photo, document, poll, or dice, **When** it submits the typed method
   payload, **Then** the runtime validates the payload and queues the matching
   Bot API method with the right payload fields.

---

### User Story 3 - Reduce bridge-stage prompt burden without losing delivery behavior (Priority: P2)

As an operator, I want the bridge-scoped Telegram loop to remove its custom
Telegram prompt burden while preserving internal outbox delivery and direct
user-facing replies.

**Why this priority**: The current traces show family selection spending the
entire output budget on reasoning before any JSON appears. Removing the custom
Telegram prompt and narrowing Telegram through explicit methods is the simplest
way to reduce that burden without changing the staged hypothesis.

**Independent Test**: A bridge-scoped Telegram request still queues delivery
through the outbox, but the family-selection prompt token count is lower than
the old bridge-scoped shape in controlled tests.

**Acceptance Scenarios**:

1. **Given** a bridge-scoped Telegram request, **When** the family-selection
   turn is sent to the provider, **Then** the prompt no longer contains the
   old Telegram bridge system prompt and still preserves outbox delivery
   semantics.

## Edge Cases

- What happens when the provider selects a Telegram method but returns payload
  fields that belong to a different Telegram method?
- What happens when the provider tries to mix a Telegram delivery tool call and
  a runtime tool call in the same bridge round?
- What happens when a Telegram method omits the bridge-scoped routing override
  fields and the runtime must fall back to the current chat/message context?
- What happens when a formatted-text payload requests a parse mode or keyboard
  structure that Telegram will later reject?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The server MUST remove the bridge-scoped Telegram-specific system
  prompt from request assembly.
- **FR-002**: The server MUST expose Telegram inside the staged tool loop as
  one ordinary family with explicit family metadata rather than a special
  prompt-only behavior layer.
- **FR-003**: The Telegram family MUST expose a bounded set of explicit methods
  likely to be used in normal operation: simple text, formatted text, photo,
  document, poll, and dice delivery.
- **FR-004**: Each Telegram method MUST publish a typed payload contract with
  per-field descriptions consistent with the staged payload-construction system.
- **FR-005**: Telegram method payloads MUST use method-specific top-level
  fields rather than the generic `request={method,payload}` wrapper previously
  used by `telegram_relay`.
- **FR-006**: The runtime MUST translate each Telegram method tool call into
  the correct queued Telegram Bot API outbox item with validated payload fields.
- **FR-007**: Bridge-scoped Telegram requests MUST continue to queue delivery
  internally through the Telegram outbox and return `vicuna_telegram_delivery`
  metadata after successful Telegram tool selection.
- **FR-008**: The staged controller MUST continue to narrow correctly by stage:
  family stage sees only family names/descriptions, method stage sees only the
  selected family's methods, and payload stage sees only the selected method's
  typed contract.
- **FR-009**: Tests and docs MUST be updated to describe the normalized
  Telegram family/method/payload behavior and the removal of the bridge-only
  Telegram system prompt.

### Key Entities *(include if feature involves data)*

- **Telegram Method Contract**: One explicit Telegram staged method containing
  method name, description, target Bot API method, and typed payload schema.
- **Telegram Delivery Tool Call**: One selected Telegram method call emitted by
  the provider during bridge-scoped staged execution and translated internally
  to an outbox item.
- **Telegram Delivery Payload**: The validated method-specific payload fields
  used to build the final outbox item for Telegram delivery.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Bridge-scoped family-selection prompts no longer contain the old
  Telegram-specific system prompt and still expose Telegram as a selectable
  family.
- **SC-002**: Automated provider tests cover at least one simple-text Telegram
  delivery and one rich Telegram delivery using the normalized Telegram methods.
- **SC-003**: The staged family-selection prompt for bridge-scoped Telegram
  requests carries fewer prompt messages than the prior bridge-scoped shape.
- **SC-004**: No bridge-scoped Telegram delivery path depends on the generic
  `telegram_relay` request wrapper after the change.
