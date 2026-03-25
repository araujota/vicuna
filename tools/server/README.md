# Vicuña Provider Server

`llama-server` is the retained product surface in this repository.

It is a DeepSeek-backed HTTP server with an optional local
`Qwen3-Embedding-0.6B-GGUF` backend for emotive-trace enrichment.

## Supported routes

- `GET /health`
- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/emotive/trace/latest`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/stream`
- `GET /v1/telegram/outbox`
- `POST /v1/telegram/outbox`
- `POST /v1/telegram/approval`
- `POST /v1/telegram/interruption`

Default-surface aliases remain for `/models`, `/completions`,
`/chat/completions`, and `/responses`.

## Removed

- local chat inference as the main serving path
- server-side tool runtimes and OpenClaw catalogs
- legacy WebUI variants, themes, and benchmark helpers
- slot/router/KV orchestration as product features

The Telegram bridge endpoints are intentionally retained as a narrow transport
surface for external dialogue delivery.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build --target llama-server -j8
```

## Run

```bash
export VICUNA_DEEPSEEK_API_KEY="your-key"
export VICUNA_DEEPSEEK_MODEL="deepseek-reasoner"
export VICUNA_DEEPSEEK_BASE_URL="https://api.deepseek.com"
export VICUNA_EMOTIVE_EMBED_MODEL="/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf"
export VICUNA_EMOTIVE_EMBED_POOLING="last"

./build/bin/llama-server --host 127.0.0.1 --port 8080 --api-surface openai --no-webui
```
