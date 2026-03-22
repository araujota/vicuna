# Research: Telegram Ask-With-Options Tool

## Local Findings

- The runtime already has a Telegram-facing tool seam through `telegram_relay`:
  - typed request/result structs in [include/llama.h](/Users/tyleraraujo/vicuna/include/llama.h)
  - cognitive request/result storage in [src/llama-cognitive-loop.cpp](/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp)
  - OpenClaw tool descriptor generation in [tools/server/server-openclaw-fabric.cpp](/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp)
  - server-side dispatch in [tools/server/server-context.cpp](/Users/tyleraraujo/vicuna/tools/server/server-context.cpp)
- The Telegram bridge currently:
  - sends plain outgoing text with `sendMessage`
  - polls only `message` updates
  - maintains a bounded chat transcript in bridge-local state
  - syncs that bounded transcript back into the runtime’s dedicated Telegram dialogue object on every Telegram-origin request
- The existing runtime-owned Telegram dialogue history is already distinct from bridge-local state, which is the correct authoritative memory surface for user-visible continuity.

## External Findings

### Telegram Bot API

- `sendMessage` accepts `reply_markup`, including `InlineKeyboardMarkup`.
- Button clicks arrive as `callback_query` updates.
- Bots should call `answerCallbackQuery` to acknowledge the selection.
- Bots can call `editMessageReplyMarkup` to remove or disable the keyboard after a choice is made.

Primary source: [Telegram Bot API](https://core.telegram.org/bots/api)

### Comparable GitHub Pattern

The [telegraf/telegraf](https://github.com/telegraf/telegraf) API surface follows the same shape:

- `sendMessage(chatId, text, { reply_markup: ... })`
- callback access through `ctx.callbackQuery`
- callback acknowledgement with `answerCbQuery(...)`
- message cleanup with `editMessageReplyMarkup(...)`

This is not copied implementation, only a confirmation that the transport shape aligns with current ecosystem practice.

## Design Implications

1. The new feature should be a new tool, not a prompt pattern layered on top of `telegram_relay`.
2. The bridge must maintain a bounded pending-option map because Telegram callback payloads are compact tokens, not self-describing long texts.
3. The resumed turn should be reconstructed from transcript plus selection metadata instead of trying to keep the original tool call open across an arbitrary user delay.
4. The bridge must tolerate intentionally empty completion text when the user-visible payload was already delivered through the `ask_with_options` tool.
