# Plan: Telegram Empty Response Guard

1. Inspect the Telegram bridge completion extraction path and confirm the empty
   response fallback trigger.
2. Add a completion-shape summary helper so bridge logs capture why a completion
   had no relayable assistant text.
3. Patch the bridge request path to retry once with an explicit plain-text
   assistant nudge when the first completion has no usable text.
4. Add targeted Telegram bridge tests for completion-shape summarization and
   empty-response extraction cases.
5. Run targeted local verification for the bridge tests and a direct runtime
   reproduction.
