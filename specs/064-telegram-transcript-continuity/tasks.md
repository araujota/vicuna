# Tasks: Telegram Transcript Continuity

1. Update `tools/telegram-bridge/lib.mjs` to normalize bounded transcript
   windows into coherent retained conversations while preserving the newest user
   turn.
2. Add regression coverage in `tools/telegram-bridge/bridge.test.mjs` for
   leading-assistant trimming, persisted-state reload, and unchanged offset
   monotonicity.
3. Update `tools/telegram-bridge/README.md` with the stale-reply diagnosis and
   the validation path for operators.
4. Run `node --test tools/telegram-bridge/bridge.test.mjs` and review the host
   evidence against the local fix before concluding.
