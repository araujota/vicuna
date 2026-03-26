# Research: Telegram Tool Normalization

## Current problem

- Bridge-scoped Telegram requests still prepend a custom Telegram system prompt
  before entering the staged family/method/payload loop.
- Telegram is modeled as one generic `telegram_relay` method whose broad
  payload schema hides the real delivery shape behind prose and a generic
  `request={method,payload}` wrapper.
- The latest traced failure showed family selection spending the entire
  `256`-token output budget on reasoning and producing no JSON. The request
  trace for `deferred_mn70rvnl_79lbsd` showed two
  `telegram_bridge_family_select` passes, both ending with `finish_reason:
  "length"` and zero JSON content.

## Design implications

- The staged contract only pays off when each family and method boundary is
- explicit. Telegram should follow the same pattern as every other family:
  ordinary family description, explicit methods, typed payload contracts.
- The family stage should remain limited to names and brief descriptions. The
  Telegram bridge-specific instructions are unnecessary once Telegram methods
  themselves describe their delivery semantics.
- A small method set is preferable to a full Bot API surface. The runtime only
  needs the high-frequency user-facing methods that materially change response
  shape.

## Selected Telegram method set

- `send_plain_text`
- `send_formatted_text`
- `send_photo`
- `send_document`
- `send_poll`
- `send_dice`

These cover the common reply patterns we actually want while keeping method
selection bounded and inspectable.

## Telegram Bot API guidance

- Official Telegram Bot API guidance treats `sendMessage` as the primary text
  delivery surface with `parse_mode` and `reply_markup` for rich text and
  inline keyboards.
- Media sends such as `sendPhoto` and `sendDocument` accept captions and the
  same formatting controls for captions.
- `sendPoll` and `sendDice` are separate methods with their own small required
  field sets, which makes them natural candidates for distinct staged methods
  instead of one generic relay wrapper.

## GitHub research

- `openai/openai-agents-python` keeps tool metadata explicit and schema-driven
  in `src/agents/function_schema.py` and `src/agents/tool.py`, which supports
  our decision to move Telegram behavior into method-level contracts instead of
  prompt prose.
- `microsoft/semantic-kernel` keeps function metadata and parameter shape
  explicit in OpenAPI-derived operation models such as
  `python/semantic_kernel/connectors/openapi_plugin/models/rest_api_operation.py`,
  reinforcing the value of small, separately described method contracts over a
  catch-all payload wrapper.

## Decision

1. Remove the bridge-scoped Telegram system prompt entirely.
2. Keep Telegram as one ordinary staged family.
3. Replace `telegram_relay` with a bounded set of explicit Telegram methods and
   method-specific payload schemas.
4. Translate those tool calls internally into the existing outbox item format,
   preserving the retained bridge delivery path.
