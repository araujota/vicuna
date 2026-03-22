# Implementation Plan: Telegram Ask-With-Options Tool

**Branch**: `070-truth-runtime-refactor` | **Date**: 2026-03-21 | **Spec**: [/Users/tyleraraujo/vicuna/specs/071-telegram-ask-options/spec.md](/Users/tyleraraujo/vicuna/specs/071-telegram-ask-options/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/071-telegram-ask-options/spec.md`

## Summary

Add a new first-class `ask_with_options` Telegram tool to the authoritative ReAct tool surface. The runtime will validate and dispatch the tool like other OpenClaw tools, the Telegram bridge will deliver the question with inline `reply_markup` options, and callback selections will be translated back into bounded Telegram transcript state so the next Telegram-origin turn continues from the chosen answer. The design keeps transport policy explicit in CPU-side code and bridge code instead of hiding it in prompt conventions.

## Technical Context

**Language/Version**: C++17 runtime/server code plus existing Node.js Telegram bridge  
**Primary Dependencies**: Existing Vicuña authoritative ReAct runtime, OpenClaw tool fabric, Telegram bridge, Telegram Bot API inline keyboard and callback query flow  
**Storage**: Existing runtime-owned Telegram dialogue state plus bounded bridge-local transport state for pending option prompts  
**Testing**: Native C++ tests in `tests/test-openclaw-tool-fabric.cpp` and `tests/test-cognitive-loop.cpp`, plus targeted bridge/runtime validation and host rebuild verification  
**Target Platform**: Existing local/hosted Vicuña runtime with Telegram bridge  
**Constraints**: No new CPU-side action authority, bounded state only, Telegram-visible output must remain message-history aware, and active turns must tolerate tool-delivered visible output without forcing fallback text

## Constitution Check

- **Runtime Policy**: Pass. Tool exposure, outbox dispatch, callback admission, and resumed-turn handling remain explicit in runtime and bridge control code.
- **Typed State**: Pass. The feature introduces explicit request/result structs plus bounded pending option state.
- **Bounded Memory**: Pass. Outbound option prompts and pending callback mappings are bounded and cleaned up after use.
- **Validation**: Pass. Tool contract, cognitive request/result handling, and resumed transcript behavior will ship with targeted tests and runtime verification.
- **Documentation & Scope**: Pass. Server and bridge operator docs remain in scope.

## Research Conclusions

- Local code already has the right seam for a Telegram-facing tool: `telegram_relay` uses cognitive request/result storage, OpenClaw descriptors, and server-side dispatch.
- The current bridge only handles `message` updates and plain text outgoing sends. Inline keyboard selection therefore requires callback-query support plus bounded prompt-state tracking.
- Telegram’s Bot API supports this directly through `sendMessage` with `reply_markup.inline_keyboard`, callback queries carrying `callback_data`, `answerCallbackQuery` for acknowledgement, and optional `editMessageReplyMarkup` to remove stale keyboards. Source: [Telegram Bot API](https://core.telegram.org/bots/api).
- Comparable Node implementations, such as [telegraf/telegraf](https://github.com/telegraf/telegraf), model this as a normal `sendMessage(... reply_markup ...)` plus `callback_query` handling with explicit acknowledgement and message editing.

## Design Decisions

### 1. New Tool, Not Prompt Convention

The feature will add a new OpenClaw builtin tool named `ask_with_options` rather than overloading `telegram_relay` text conventions. This keeps the authoritative tool surface inspectable and lets the model see a separate contract for discrete user choice.

### 2. Dedicated Telegram Outbox

The runtime will expose a dedicated bounded Telegram outbox endpoint for ask-with-options deliveries instead of smuggling reply markup through the proactive self-emit text mailbox. This keeps proactive text delivery and structured Telegram interaction separate.

### 3. Immediate Tool Completion, Deferred User Continuation

`ask_with_options` completes once the runtime successfully hands the prompt to the bridge outbox. The later callback selection resumes as a new Telegram-origin user turn. The continuation is reconstructed from bounded transcript history rather than by holding the original tool command open indefinitely.

### 4. Bridge-Owned Pending Option Map

The bridge will persist a bounded pending-option map keyed by a compact callback token. Each entry stores chat scope, question text, and option labels so callback selections can be resolved safely and rewritten into transcript form.

### 5. Transcript-Driven Continuation

When a callback arrives, the bridge will append a synthetic user transcript message grounded in the prior assistant question, for example `Selected option for "<question>": <label>`, then call the runtime like any other Telegram-origin turn. This reuses the existing bounded transcript and runtime Telegram dialogue sync path.

### 6. Empty Visible-Text Handling

The bridge will stop turning every empty active reply into a user-facing fallback string. For ask-with-options turns, the user-visible output is the tool-delivered Telegram message, not assistant text from the completion response.

## Project Structure

```text
specs/071-telegram-ask-options/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
└── tasks.md
```

## Implementation Phases

1. Add the 071 design artifact set.
2. Add typed tool kinds, request/result structs, and OpenClaw descriptor/schema support.
3. Add runtime outbox dispatch and bridge polling or fetch support.
4. Add bridge callback-query handling, pending option state, and resumed-turn forwarding.
5. Add tests, docs, local validation, host rebuild, and live capability verification.
