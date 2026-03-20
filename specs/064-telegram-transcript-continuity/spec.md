# Feature Spec: Telegram Transcript Continuity

## Summary

Fix the Telegram bridge so it preserves coherent chat turns when it trims stored
history for runtime requests. The bridge must continue to accept new Telegram
updates exactly once, but it must stop sending transcript windows that begin
with orphaned assistant replies and bias the runtime toward responding to stale
context instead of the latest user message.

## User Scenarios & Testing

### User Story 1 - Bridge forwards the current user turn (Priority: P1)

When a Telegram chat exceeds the configured history limit, the bridge still
forwards a transcript whose newest user turn is preserved and whose oldest
messages are trimmed without leaving a leading assistant-only orphan.

**Independent Test**: Append more than `TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES`
messages to a chat session and verify the persisted/request transcript begins
with a user turn when trimming would otherwise strand an assistant reply.

**Acceptance Scenarios**:

1. **Given** a stored chat transcript at the history cap, **When** a new user
   message is appended, **Then** the retained transcript still includes that
   newest user message and does not start with an unmatched assistant message.
2. **Given** a stored chat transcript that begins with a valid user turn,
   **When** old history is trimmed, **Then** complete user/assistant turn
   ordering is preserved for the remaining window.

### User Story 2 - Telegram polling does not replay old updates (Priority: P2)

Operators can verify that stale replies are not caused by Telegram update
replay, because the bridge persists monotonic update offsets independently from
the transcript windowing fix.

**Independent Test**: Exercise state normalization and offset advancement while
confirming transcript trimming changes do not alter `telegramOffset` behavior.

**Acceptance Scenarios**:

1. **Given** sequential Telegram update IDs, **When** the bridge persists state,
   **Then** `telegramOffset` remains monotonic and independent from transcript
   trimming.

## Edge Cases

- History trimming after a user turn when the maximum history size is odd.
- Chats with consecutive user turns or consecutive assistant turns due to
  earlier bridge/runtime failures.
- Reloading persisted state from disk after trimming has already occurred.

## Requirements

1. The Telegram bridge MUST keep `telegramOffset` monotonic so Telegram updates
   are not replayed after the fix.
2. The Telegram bridge MUST trim retained chat history without leaving a
   leading assistant message that no longer has its originating user turn.
3. The Telegram bridge MUST preserve the most recent user message in the
   runtime-bound transcript even when history trimming occurs.
4. The Telegram bridge MUST keep transcript shaping policy explicit in bridge
   control code and cover it with targeted automated tests.
5. Operator-facing bridge documentation MUST describe the continuity fix and
   the validation path for diagnosing stale-reply behavior.

## Success Criteria

1. Host reproduction logs show fresh Telegram user turns being accepted once
   while local regression tests confirm trimmed transcripts keep the latest user
   turn and avoid leading assistant orphans.
2. Targeted bridge tests pass for transcript trimming, persisted state reload,
   and offset monotonicity after the change.
