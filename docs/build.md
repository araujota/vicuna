# Build

The supported build target is the provider-first `llama-server`.

## Configure

```bash
cmake -S . -B build -G Ninja
```

## Build

```bash
cmake --build build --target llama-server -j8
```

## Run

```bash
export VICUNA_DEEPSEEK_API_KEY="your-key"
export VICUNA_DEEPSEEK_MODEL="deepseek-reasoner"
export VICUNA_DEEPSEEK_BASE_URL="https://api.deepseek.com"

./build/bin/llama-server --host 127.0.0.1 --port 8080 --api-surface openai --no-webui
```

Optional emotive embedding support:

```bash
export VICUNA_EMOTIVE_ENABLED="1"
export VICUNA_EMOTIVE_EMBED_ENABLED="1"
export VICUNA_EMOTIVE_EMBED_MODEL="/absolute/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf"
export VICUNA_EMOTIVE_EMBED_POOLING="last"
```
