# Vicuña

Vicuña is now a provider-first runtime.

The active product in this repository is a small DeepSeek-backed HTTP server in
[`tools/server`](/Users/tyleraraujo/vicuna/tools/server) with:

- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`
- block-wise emotive-moment tracing
- immediate VAD projection and a metacognitive control policy over routing,
  reasoning depth, tool use, interruption, completion, and DeepSeek request
  shaping
- an optional local `Qwen3-Embedding-0.6B-GGUF` backend used only for trace enrichment
- bounded cognitive replay and heuristic memory that feed forward into later
  turns as additive control biases
- a richer provider control surface that can switch DeepSeek `thinking` on or
  off per turn, apply bounded prefix and stop profiles, and shape sampling,
  repetition, and tool-choice params directly from the CPU-side controller
- a bounded RL runtime surface that captures typed governance transitions,
  exposes policy status and export endpoints, supports safe shadow-policy
  comparison, and can execute registry-backed learned policies through an
  explicit canary and rollback controller
- a desired-state reward model over all 14 emotive-moment dimensions plus all
  3 VAD axes, with typed reward breakdowns retained for every captured
  transition
- an offline policy-learning pipeline that persists those transitions into
  datasets, materializes a masked training contract, trains inspectable
  candidate artifacts, stores them in a local registry, refreshes the
  candidate loop nightly, and serves registry aliases over HTTP for live
  rollout
- a flattened runtime tool surface for media, hard memory, web search, and
  recurring host cron tasks
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

Optional RL runtime controls:

```bash
export VICUNA_POLICY_MODE="canary_live"
export VICUNA_POLICY_MAX_TRANSITIONS="128"
export VICUNA_POLICY_CANDIDATE_URL="http://127.0.0.1:18081"
export VICUNA_POLICY_TIMEOUT_MS="500"
export VICUNA_POLICY_CANARY_STEPS="10,50,100"
export VICUNA_POLICY_CANARY_MIN_REQUESTS_PER_STEP="5"
export VICUNA_POLICY_LIVE_CONFIDENCE_THRESHOLD="0.70"
export VICUNA_POLICY_ROLLBACK_MAX_CANDIDATE_FAILURE_RATE="0.30"
export VICUNA_POLICY_ROLLBACK_MAX_INVALID_ACTION_RATE="0.20"
export VICUNA_POLICY_ROLLBACK_MAX_FALLBACK_RATE="0.40"
export VICUNA_POLICY_REWARD_CONFIG_PATH="/absolute/path/to/reward-profile.json"
```

Inspect the bounded governance surface with `GET /v1/policy/status`,
`GET /v1/policy/transitions`, and `/health -> policy_runtime`.
Those surfaces now include the active desired-state reward model and
per-transition reward breakdowns. The default live profile targets a calm,
aligned, confident, low-stall state with positive valence, moderate-low
arousal, and moderately high dominance.

Offline policy-learning loop:

```bash
python3 tools/policy-learning/cli.py export --server http://127.0.0.1:8080 --dataset-dir .cache/vicuna/policy-datasets/local-v1 --dataset-id vicuna-governance-local-v1
python3 tools/policy-learning/cli.py build-training-set --dataset-dir .cache/vicuna/policy-datasets/local-v1
python3 tools/policy-learning/cli.py train --dataset-dir .cache/vicuna/policy-datasets/local-v1 --model-name vicuna-governance --registry-dir .cache/vicuna/policy-registry
python3 tools/policy-learning/cli.py evaluate --dataset-dir .cache/vicuna/policy-datasets/local-v1 --candidate-command "python3 tools/policy-learning/registry_policy_adapter.py --artifact .cache/vicuna/policy-runs/<training-run-id>/artifact.json"
python3 tools/policy-learning/cli.py register --registry-dir .cache/vicuna/policy-registry --model-name vicuna-governance --artifact-path .cache/vicuna/policy-runs/<training-run-id>/artifact.json --training-run-manifest-path .cache/vicuna/policy-runs/<training-run-id>/training_run_manifest.json --evaluation-report-path .cache/vicuna/policy-datasets/local-v1/reports/offline_eval_<policy-version>.json
python3 tools/policy-learning/cli.py nightly-batch --server http://127.0.0.1:8080 --dataset-dir .cache/vicuna/policy-datasets/nightly --dataset-id vicuna-governance-nightly-v1 --registry-dir .cache/vicuna/policy-registry --model-name vicuna-governance
python3 tools/policy-learning/cli.py serve-registry --host 127.0.0.1 --port 18081 --registry-dir .cache/vicuna/policy-registry --model-name vicuna-governance --default-alias candidate --fallback-alias champion
```

Live rollout is implemented under `specs/149-policy-rollout-surface/`.
Canary execution remains bounded by native masks, confidence thresholds, and
automatic rollback to native-only control.
Direct provider-request shaping is implemented under
`specs/153-policy-api-control/`.

The bridge no longer depends on a provider-visible Telegram tool family.
Bridge-scoped replies are delivered from parsed rich plan text, with optional
formatting and reply markup metadata, while the server-owned outbox remains the
transport boundary.

## References

- [deep-research-report.md](/Users/tyleraraujo/vicuna/deep-research-report.md)
- [ARCHITECTURE.md](/Users/tyleraraujo/vicuna/ARCHITECTURE.md)
- [docs/build.md](/Users/tyleraraujo/vicuna/docs/build.md)
- [tools/policy-learning/README.md](/Users/tyleraraujo/vicuna/tools/policy-learning/README.md)
- [tools/server/README.md](/Users/tyleraraujo/vicuna/tools/server/README.md)
