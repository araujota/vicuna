# Quickstart: Authoritative ReAct Action Contract Guarantee

## Targeted Validation

1. Build the targeted runtime test binary:
   - `cmake --build build --target test-cognitive-loop -j8`
2. Run the authoritative ReAct regression coverage:
   - `./build/bin/test-cognitive-loop`
3. If available, run a focused host trace after rebuild:
   - submit the Telegram book-download request again
   - confirm the bridge never delivers raw control JSON
   - confirm runtime logs show retry or staged progression instead of `source=malformed_action_label_fallback` for control-shaped JSON

## Manual Operator Checks

- Inspect runtime logs and staged provenance to confirm the exact required action value is resolved for each phase even when the model omits the fixed `action` field.
- Inspect runtime logs for parse failures and verify malformed staged payloads remain in retry instead of being sent to Telegram.
- Inspect Telegram bridge state after a reproduction attempt and confirm no assistant transcript entry contains raw control JSON such as `{"tool_family_id":"web_search"}`.
