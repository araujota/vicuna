# Quickstart: Telegram Tool Normalization

## Validate family-stage normalization

1. Start the provider-mode server with mocked Telegram runtime tools enabled.
2. Send one bridge-scoped `/v1/chat/completions` request with Telegram headers.
3. Confirm the first staged request contains:
   - the generic staged core system prompt
   - no Telegram-specific bridge system prompt
   - one `Available families:` list that includes `Telegram`

## Validate method-stage narrowing

1. Mock the provider to select `{"family":"Telegram"}`.
2. Confirm the next staged request contains only Telegram methods:
   - `send_plain_text`
   - `send_formatted_text`
   - `send_photo`
   - `send_document`
   - `send_poll`
   - `send_dice`

## Validate delivery translation

1. Mock the provider to select one Telegram method and submit a matching
   payload.
2. Confirm the runtime returns `vicuna_telegram_delivery` and queues the
   expected `telegram_method` plus `telegram_payload` in `/v1/telegram/outbox`.

## Validation commands

```bash
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build /Users/tyleraraujo/vicuna/build-codex-emotive --target llama-server -j8
LLAMA_SERVER_BIN_PATH=/Users/tyleraraujo/vicuna/build-codex-emotive/bin/llama-server /opt/homebrew/bin/pytest /Users/tyleraraujo/vicuna/tools/server/tests/unit/test_deepseek_provider.py -q -k "telegram"
```
