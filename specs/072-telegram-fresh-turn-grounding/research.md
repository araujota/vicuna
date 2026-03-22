# Research: Telegram Fresh-Turn Grounding

## Local Code Findings

### 1. Active foreground extraction currently uses the transformed request body

In [tools/server/server-context.cpp](/Users/tyleraraujo/vicuna/tools/server/server-context.cpp), task creation currently does:

- `task.foreground_role = classify_foreground_role(data);`
- `const std::string foreground_text = extract_foreground_message_text(data);`

For OpenAI-chat-style requests, `data` is already the post-template parsed body, not the raw incoming `messages` array. That means the active self-state event can be seeded from transformed prompt text instead of the latest raw Telegram user turn.

### 2. Telegram dialogue sync already uses the raw request body

Also in [tools/server/server-context.cpp](/Users/tyleraraujo/vicuna/tools/server/server-context.cpp), Telegram dialogue sync correctly reads:

- `extract_telegram_transcript_messages(req.body)`
- `sync_telegram_dialogue_history(...)`

So the runtime already has access to the correct raw Telegram messages, but the active foreground extraction path does not use them.

### 3. Canonical Telegram ReAct ordering appends shared hidden/tool context after Telegram dialogue

In `canonical_react_messages(...)` inside [tools/server/server-context.cpp](/Users/tyleraraujo/vicuna/tools/server/server-context.cpp), the current order is:

1. append Telegram dialogue messages
2. append shared context items, suppressing only shared user-visible items

That means older hidden thought/tool/internal items can become the last messages in the prompt after the newest Telegram user turn, which lets stale prior-turn context dominate generation.

## Host Log Findings

From the live host runtime logs already captured during validation:

- The runtime correctly exposes `ask_with_options` in the live capability set.
- A fresh Telegram-scoped probe with new content still returned the prior Tesla answer.
- The active finalize log showed `tools=6` and `tool_calls=0`, but the returned visible answer and hidden reasoning were still Tesla-focused.

This is consistent with a prompt-grounding bug rather than missing tool availability.

## GitHub Research

Using GitHub MCP on `araujota/vicuna`, the recent branch history confirms the regression sits in the newly modified runtime path rather than an older untouched path:

- `3083e0ec2` `Add Telegram ask-with-options tool`
- `bd57ce991` `Fix ask-options clamp typing`
- `23a730e26` `runtime: inject emotive style directive`
- `b6abf71d7` `runtime: remove active preflight authority`

These recent commits all changed the active Telegram/ReAct serving path near the same code that now derives foreground context and rebuilds canonical ReAct prompts.

## Conclusion

The most likely root cause is a combination of:

1. active foreground extraction using the transformed parsed body instead of the raw latest request messages
2. Telegram-scoped canonical ReAct prompt assembly placing older shared hidden/tool context after the newest Telegram dialogue turn

The fix should patch both control points together.
