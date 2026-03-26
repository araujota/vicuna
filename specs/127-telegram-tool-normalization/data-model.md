# Data Model: Telegram Tool Normalization

## Telegram Method Contract

- `tool_name`: internal function/tool-call name such as
  `telegram_send_plain_text`
- `family_id`: always `telegram`
- `family_name`: always `Telegram`
- `family_description`: one brief family description used at family-selection
  time
- `method_name`: staged method name such as `send_plain_text`
- `method_description`: concise method description shown during method
  selection
- `parameters`: typed JSON schema for the method payload
- `telegram_method`: final Bot API method such as `sendMessage`,
  `sendPhoto`, `sendDocument`, `sendPoll`, or `sendDice`

## Telegram Delivery Payload

Shared optional routing/meta fields across Telegram methods:

- `chat_scope`
- `reply_to_message_id`
- `intent`
- `dedupe_key`
- `urgency`

Method-specific payloads:

- `send_plain_text`: `text`
- `send_formatted_text`: `text`, optional `parse_mode`,
  `disable_web_page_preview`, `reply_markup`
- `send_photo`: `photo`, optional `caption`, `parse_mode`, `reply_markup`
- `send_document`: `document`, optional `caption`, `parse_mode`, `reply_markup`
- `send_poll`: `question`, `options`, optional `is_anonymous`,
  `allows_multiple_answers`
- `send_dice`: optional `emoji`

## Telegram Delivery Translation

- One selected Telegram tool call is parsed into one `telegram_outbox_item`
- `telegram_method` is derived from the selected Telegram method contract
- `telegram_payload` is built from the method-specific fields
- `text` summary is derived from the normalized Telegram payload for logging and
  transcript continuity
