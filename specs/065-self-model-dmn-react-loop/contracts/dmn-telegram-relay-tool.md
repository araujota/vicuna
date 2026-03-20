# Contract: DMN Telegram Relay Tool

## Purpose

Define a first-class DMN tool for user-directed outreach through the Telegram
bridge without collapsing background cognition into active engagement.

## Request Shape

- `request_id`
- `episode_id`
- `prompt_revision_id`
- `intent_kind`
  - `question`
  - `comment`
  - `conclusion`
- `urgency`
- `routing_mode`
  - `registered_chats`
  - `primary_chat`
  - future explicit routing modes as needed
- `text`
- `dedupe_key`
- `policy_gate_mask`

## Result Shape

- `request_id`
- `delivered`
- `delivery_at_us`
- `bridge_message_id`
- `failure_kind`
- `failure_detail`
- `retry_allowed`

## Policy Rules

- The relay MUST be represented as a tool action and tool observation.
- Relay delivery MUST NOT count as an active engagement completion.
- Relay delivery MUST NOT terminate the DMN loop by itself.
- Relay failure MUST remain available as a tool result for later DMN reasoning.
- Existing chat-registration and anti-spam policy MUST still apply.

## Integration Rules

- The planner/tool runner may select this tool from a DMN episode the same way
  it selects other tools.
- The bridge or server layer must preserve the DMN origin so delivery is
  observable separately from active engagement.
- Tool-result integration should allow a later self-model update to regenerate a
  new prompt revision and continue the DMN loop.

## Non-Goals

- This feature does not replace the Telegram bridge transport itself.
- This feature does not make every DMN message user-visible; it only adds an
  explicit path when user outreach is selected as a tool.
