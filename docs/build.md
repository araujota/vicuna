# Build

## Host Runtime

```bash
cmake -S /Users/tyleraraujo/vicuna -B /Users/tyleraraujo/vicuna/build
cmake --build /Users/tyleraraujo/vicuna/build --target llama-server -j8
```

## Main Host Services

- runtime: `tools/ops/run-vicuna-runtime.sh`
- Telegram bridge: `tools/ops/run-telegram-bridge.sh`
- WebGL renderer: `tools/ops/run-webgl-renderer.sh`

## Runtime Modes

- `VICUNA_HOST_INFERENCE_MODE=standard`
  - use DeepSeek directly
  - no RL/policy shaping in the request path
- `VICUNA_HOST_INFERENCE_MODE=experimental`
  - use the RunPod relay connector
  - persist experimental capture on the host

## RunPod Host Workflow

The retained RunPod scripts are:

- `tools/ops/runpod-ensure-pod.sh`
- `tools/ops/runpod-runtime-common.sh`
- `tools/ops/runpod-runtime-endpoint.sh`
- `tools/ops/runpod-stop-pod.sh`
- `tools/ops/runpod-sync-experimental-capture.sh`
- `tools/ops/run-runpod-mistral-relay.sh`
- `tools/ops/host-inference-mode.sh`

The host-side post-processing workflow is:

- sync JSONL capture from pod to host
- render host videos from synced emotive traces

## Service Installation

Use:

```bash
sudo /Users/tyleraraujo/vicuna/tools/ops/install-vicuna-system-service.sh
```

That installs the retained runtime, bridge, capture-sync, and render services.
