# Vicuña Provider Server

`llama-server` is now a small DeepSeek-backed HTTP service.

It does not load a local GGUF model, does not manage KV cache state, does not
run the old task/slot/router runtime, and does not expose LoRA control
surfaces. It can optionally load a local embedding GGUF for emotive trace
computation.

## Required environment

```bash
export VICUNA_DEEPSEEK_API_KEY="your-key"
export VICUNA_DEEPSEEK_MODEL="deepseek-reasoner"
export VICUNA_DEEPSEEK_BASE_URL="https://api.deepseek.com"
export VICUNA_DEEPSEEK_TIMEOUT_MS="60000"
```

Optional emotive runtime environment:

```bash
export VICUNA_EMOTIVE_ENABLED="1"
export VICUNA_EMOTIVE_BLOCK_MAX_CHARS="320"
export VICUNA_EMOTIVE_MAX_BLOCKS_PER_TURN="128"
export VICUNA_EMOTIVE_MAX_TURN_HISTORY="8"
export VICUNA_EMOTIVE_DEGRADED_MODE_ALLOWED="1"
export VICUNA_EMOTIVE_EMBED_ENABLED="1"
export VICUNA_EMOTIVE_EMBED_MODEL="/absolute/path/to/Qwen3-Embedding-0.6B.gguf"
export VICUNA_EMOTIVE_EMBED_GPU_LAYERS="999"
export VICUNA_EMOTIVE_EMBED_CTX="4096"
export VICUNA_EMOTIVE_EMBED_POOLING="last"
export VICUNA_EMOTIVE_VAD_ALPHA="0.35"
```

## Build

```bash
cmake -B build
cmake --build build --target llama-server -j8
```

## Run

```bash
./build/bin/llama-server --host 127.0.0.1 --port 8080 --api-surface openai --no-webui
```

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

Default-surface aliases remain for `/models`, `/completions`, `/chat/completions`, and `/responses`.

## Removed

- local GGUF inference
- local GGUF chat inference
- KV-cache-backed slot execution
- router mode
- rerank, infill, tokenize, detokenize
- LoRA adapter management
- Anthropic compatibility
- old server task/context/queue runtime

## Hello validation

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

Expected fields:

- `choices[0].message.content`
- `choices[0].message.reasoning_content`
- `vicuna_emotive_trace.blocks`
- `vicuna_emotive_trace.blocks[*].vad.trend`
- `vicuna_emotive_trace.blocks[*].vad.style_guide`
