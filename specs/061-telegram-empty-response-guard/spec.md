# Feature Spec: Telegram Empty Response Guard

## Summary

Harden the Telegram bridge so transient or malformed runtime chat completions do
not immediately surface to users as an empty-response fallback. The bridge
should inspect and log the completion shape, retry once for a plain-text
assistant reply, and remain test-covered.

## Requirements

1. The Telegram bridge must detect when a chat completion returns no usable
   assistant text.
2. The bridge must log enough structured completion metadata to distinguish
   empty text, tool-call-only responses, and other non-text completion shapes.
3. The bridge must retry once with an explicit plain-text assistant nudge before
   falling back to the empty-response user message.
4. Add or update tests alongside the behavior change.
5. Verify the bridge behavior with targeted local checks after the patch.
