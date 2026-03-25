# llama-server Development Notes

The server has been rebuilt as a provider-only HTTP layer.

Current architecture:

- `server.cpp`: route table, bridge compatibility handlers, provider request flow
- `server-http.cpp`: HTTP transport, auth, middleware, static server setup
- `server-common.cpp`: small JSON/error/SSE helpers
- `server-deepseek.cpp`: DeepSeek request/response mapping, including tool-capable chat passthrough
- `server-emotive-runtime.cpp`: block-wise emotive moment estimator and retained trace store
- `server-embedding-backend.cpp`: optional local GGUF embedding backend for trace enrichment

Current VAD policy in `server-emotive-runtime.cpp`:

- every rich moment dimension contributes through explicit projection weights
- block-to-block smoothing uses `VICUNA_EMOTIVE_VAD_ALPHA`
- each block emits VAD trend, labels, dominant dimensions, and a prompt-ready style guide

Removed from the active build:

- `server_context`
- `server_task`
- `server_queue`
- `server_models`
- slot scheduling and KV cache management
- LoRA adapter control paths
- local runtime tool execution paths

Bridge compatibility remains intentionally narrow:

- `/v1/responses/stream` is a keepalive SSE surface for the Telegram bridge
- `/v1/telegram/outbox` now supports both bridge polling and compact provider-only relay writes
- `/v1/telegram/approval` and `/v1/telegram/interruption` remain lightweight compatibility endpoints

DeepSeek provider mode now preserves tool-capable request and response fields
on the provider-only path:

- outbound requests preserve `tools`, `tool_choice`, and `parallel_tool_calls`
- prior assistant `tool_calls` and `tool` role messages are forwarded back to
  DeepSeek for follow-up turns
- inbound provider `message.tool_calls` are normalized into standard
  OpenAI-style `tool_calls`
- `/v1/responses` emits `function_call` output items when the provider selects
  a tool

This keeps the active server surface OpenAI-compatible for external tool
execution without reviving the deleted local server-side tool runtime.

This server should be treated as a provider transport layer, not as a local
inference runtime. The one local-model exception is the optional emotive
embedding backend, which exists only to compute `vicuna_emotive_trace`
extensions and the `/v1/emotive/trace/latest` debug surface.
