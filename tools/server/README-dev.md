# Provider Server Development Notes

Current retained components:

- `server.cpp`: route table and provider request flow
- `server-http.cpp`: HTTP transport and middleware
- `server-common.cpp`: JSON/error/SSE helpers
- `server-deepseek.cpp`: DeepSeek request and response mapping
- `server-emotive-runtime.cpp`: block-wise emotive moment and VAD projection
- `server-embedding-backend.cpp`: optional `Qwen3-Embedding-0.6B-GGUF` backend

Current direction:

- provider-first request handling only
- internal streaming for block-wise emotive capture
- Telegram bridge compatibility remains intentionally narrow
- no OpenClaw tool runtime or catalog sync path
- no local chat inference product surface

The only justified retained dependency on the native `llama` library is the
optional local `Qwen3-Embedding-0.6B-GGUF` backend used by the emotive runtime.

Local embedding policy:

- only `general.architecture=qwen3` GGUF models are accepted
- the recommended local model is `Qwen3-Embedding-0.6B-GGUF`
- use `VICUNA_EMOTIVE_EMBED_POOLING=last`

Retained bridge endpoints:

- `/v1/responses/stream`
- `/v1/telegram/outbox`
- `/v1/telegram/approval`
- `/v1/telegram/interruption`
