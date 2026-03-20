# Plan: Telegram Transcript Continuity

## Summary

Host logs from `vicuna-telegram-bridge.service` on March 20, 2026 show that new
Telegram user messages arrive exactly once and are persisted into the bridge
state file, while the forwarded transcript window starts with a leading
assistant turn after raw message-count trimming. The runtime then continues a
stale Singapore-time tool thread even when the newest user turn asks about the
stale replies. The fix belongs in the Telegram bridge transcript shaping path,
not in Telegram polling or runtime transport.

## Technical Context

- Language/Version: Node.js 20 ESM
- Primary Dependencies: built-in `fetch`, local helper module in
  `tools/telegram-bridge/lib.mjs`
- Storage: JSON bridge state file
- Testing: `node --test tools/telegram-bridge/bridge.test.mjs`
- Target Platform: Linux host bridge service plus local macOS development
- Project Type: bridge middleware for local runtime
- Performance Goals: retain existing long-poll behavior and bounded transcript
  history while preserving the latest user intent
- Constraints: keep runtime policy explicit in CPU-side bridge code, preserve
  bounded history, avoid new dependencies

## Research Inputs

- Host logs prove `telegramOffset` advances and new messages are appended before
  each stale reply, which rules out Telegram update replay as the primary bug.
- Host bridge state shows trimmed transcripts starting with assistant messages,
  which indicates a bridge-side history-windowing defect.
- GitHub history shows the bridge was introduced in commit
  `cf5588b7a969b30e9c067332d73a32ad6142e0ab` and later extended for document
  ingestion in `59e2f601f8a6fe7b9fbde65ead084b43875c9e9d`; neither change added
  turn-coherent trimming tests.
- Telegram Bot API `getUpdates` documentation says an update is confirmed once
  `getUpdates` is called with an offset higher than its `update_id`, which
  matches the bridge’s current monotonic offset policy and further isolates the
  defect to transcript shaping rather than polling semantics.

## Constitution Check

- Runtime Policy: The change stays in `tools/telegram-bridge/lib.mjs`, where
  transcript shaping policy is explicit and inspectable.
- Typed State: No runtime structs or memory surfaces change; the JSON bridge
  state schema stays stable while transcript normalization rules tighten.
- Bounded Memory: The transcript window remains capped by
  `TELEGRAM_BRIDGE_MAX_HISTORY_MESSAGES`; the fix only chooses a coherent slice.
- Validation: Add targeted Node tests for trimming behavior, state reload, and
  offset monotonicity; run the bridge test suite locally.
- Documentation & Scope: Update Telegram bridge docs with the continuity
  diagnosis and validation guidance.

## Implementation Structure

- Spec artifacts: `specs/064-telegram-transcript-continuity/`
- Code changes: `tools/telegram-bridge/lib.mjs`,
  `tools/telegram-bridge/bridge.test.mjs`,
  `tools/telegram-bridge/README.md`

## Phases

1. Add explicit transcript-window normalization that trims to a coherent
   conversation start and preserves the newest user turn.
2. Reuse that policy for in-memory appends and state reload normalization.
3. Add regression tests that reproduce the leading-assistant history window and
   verify the fixed retained transcript.
4. Update operator docs and run targeted bridge tests.
