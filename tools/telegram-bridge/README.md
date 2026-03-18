# Telegram Bridge

This bridge connects a Telegram bot to the local Vicuña server.

It does two jobs in one process:

- polls Telegram for inbound text messages and forwards them to
  `/v1/chat/completions`
- subscribes to `/v1/responses/stream` and relays proactive self-emits to every
  registered Telegram chat

## Environment

Load the repo env file first:

```bash
source ./.envrc
```

Required variables:

- `TELEGRAM_BOT_TOKEN`

Optional variables:

- `TELEGRAM_BRIDGE_VICUNA_BASE_URL` default: `http://127.0.0.1:8080`
- `TELEGRAM_BRIDGE_MODEL` default: `qwen2.5:7b-instruct-q8_0`
- `TELEGRAM_BRIDGE_STATE_PATH` default: `/tmp/vicuna-telegram-bridge-state.json`
- `TELEGRAM_BRIDGE_POLL_TIMEOUT_SECONDS` default: `30`
- `VICUNA_API_KEY` if the server runs with bearer auth enabled

## Start

```bash
npm run telegram-bridge:start
```

## Managed Service

For resilient host operation, use the repo-owned user service units:

- [vicuna-runtime.service](/home/tyler-araujo/Projects/vicuna/tools/ops/systemd/vicuna-runtime.service)
- [vicuna-telegram-bridge.service](/home/tyler-araujo/Projects/vicuna/tools/ops/systemd/vicuna-telegram-bridge.service)

The launchers they use are:

- [run-vicuna-runtime.sh](/home/tyler-araujo/Projects/vicuna/tools/ops/run-vicuna-runtime.sh)
- [run-telegram-bridge.sh](/home/tyler-araujo/Projects/vicuna/tools/ops/run-telegram-bridge.sh)

## Behavior

- `/start` registers the chat for proactive relay and returns a confirmation
- plain text user messages are sent to the local Vicuña runtime and the
  assistant reply is sent back to the same Telegram chat
- proactive runtime self-emits are consumed from `/v1/responses/stream` and
  sent to all registered chats
- the bridge intentionally reconnects the proactive SSE stream when the server
  closes an idle stream; it always reconnects with `after=0` and deduplicates
  by retained `response_id`, so retained self-emits are replay-safe across
  reconnects

## Test

```bash
npm run test:telegram-bridge
```
