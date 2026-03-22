# Data Model: Telegram Ask-With-Options Tool

## Entities

### Telegram Ask-With-Options Request

- `command_id`
- `origin`
- `tool_job_id`
- `command_ready`
- `urgency`
- `chat_scope`
- `dedupe_key`
- `question`
- `option_count`
- `options[]`

#### Invariants

- `question` is required and bounded.
- `option_count` is between 2 and the configured max.
- `chat_scope` is explicit at dispatch time even if it was inferred from the active Telegram task.
- The request is transport-facing only; it does not become the canonical dialogue history by itself.

### Telegram Ask-With-Options Result

- `command_id`
- `tool_job_id`
- `delivered`
- `delivered_at_ms`
- `chat_scope`
- `dedupe_key`
- `error_text`

#### Invariants

- The result reflects handoff to the Telegram bridge outbox, not long-lived user selection state.
- A failed result must not silently disappear.

### Telegram Outbox Event

- `sequence_number`
- `kind`
  - `ask_with_options`
- `chat_scope`
- `dedupe_key`
- `question`
- `options[]`
- `created_at_ms`

#### Invariants

- The outbox is bounded and ordered.
- The bridge consumes items after a monotonic sequence cursor.
- The outbox is transport state, not dialogue memory.

### Pending Option Prompt

- `prompt_id`
- `chat_id`
- `question`
- `options[]`
- `telegram_message_id`
- `callback_prefix`
- `created_at_ms`

#### Invariants

- Pending prompts are bounded in count.
- Each callback token resolves to exactly one pending prompt and one option.
- Completed or stale prompts are removed after use.

### Callback Selection Resume Event

- `chat_id`
- `telegram_message_id`
- `question`
- `selected_option`
- `selected_index`
- `selected_at_ms`

#### Invariants

- The bridge rewrites the selection into bounded transcript state before calling the runtime.
- The runtime continues from transcript plus Telegram dialogue memory, not from a hidden callback-only side channel.
