# Vicuña

Vicuña is now a host-runtime repository.

The retained scope is:

- `tools/server`: host HTTP server with a clean split between `standard`
  DeepSeek serving and `experimental` RunPod relay
- `tools/telegram-bridge`: Telegram delivery, outbox polling, and WebGL-backed
  response video delivery
- `tools/openclaw-harness`: host-owned tool harness for hard memory, skills,
  host shell, and Telegram relay surfaces
- `tools/policy-learning`: offline RL and registry infrastructure for the data
  captured from experimental runtime traffic
- `tools/ops`: host ops, RunPod connection scripts, capture sync, and render
  automation

Everything else is being removed from this repository. The remote inference
runtime and any `llama.cpp` fork now live elsewhere.

## Runtime Modes

- `standard`: DeepSeek-only serving with a large token budget and no policy or
  learning logic applied to the request path
- `experimental`: host-owned prompt, tool, bridge, and capture flow relayed to
  the RunPod pod

## Build

```bash
cmake -S . -B build
cmake --build build --target llama-server -j8
```

## Run

```bash
export VICUNA_DEEPSEEK_API_KEY="your-key"
tools/ops/run-vicuna-runtime.sh
```

For the retained host workflow, start from:

- [ARCHITECTURE.md](/Users/tyleraraujo/vicuna/ARCHITECTURE.md)
- [docs/build.md](/Users/tyleraraujo/vicuna/docs/build.md)
- [tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md)
