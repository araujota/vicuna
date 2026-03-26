# Vicuña

Vicuña is now a provider-first runtime.

The active product in this repository is a small DeepSeek-backed HTTP server in
[`tools/server`](/Users/tyleraraujo/vicuna/tools/server) with:

- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`
- block-wise emotive-moment tracing
- immediate VAD projection and prompt-ready style guidance
- an optional local `Qwen3-Embedding-0.6B-GGUF` backend used only for trace enrichment
- a retained Telegram bridge path for external dialogue delivery

This repository is no longer maintained as a general `llama.cpp` distribution,
local chat runtime, or OpenClaw orchestration stack. Legacy surfaces outside
the provider-first server plus Telegram bridge are being removed.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build --target llama-server -j8
```

## Run

```bash
export VICUNA_DEEPSEEK_API_KEY="your-key"
export VICUNA_DEEPSEEK_MODEL="deepseek-chat"
export VICUNA_DEEPSEEK_BASE_URL="https://api.deepseek.com"

./build/bin/llama-server --host 127.0.0.1 --port 8080 --api-surface openai --no-webui
```

Optional local embedding enrichment:

```bash
export VICUNA_EMOTIVE_ENABLED="1"
export VICUNA_EMOTIVE_EMBED_ENABLED="1"
export VICUNA_EMOTIVE_EMBED_MODEL="/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf"
export VICUNA_EMOTIVE_EMBED_POOLING="last"
```

## References

- [deep-research-report.md](/Users/tyleraraujo/vicuna/deep-research-report.md)
- [ARCHITECTURE.md](/Users/tyleraraujo/vicuna/ARCHITECTURE.md)
- [docs/build.md](/Users/tyleraraujo/vicuna/docs/build.md)
- [tools/server/README.md](/Users/tyleraraujo/vicuna/tools/server/README.md)
