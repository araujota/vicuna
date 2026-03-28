# Policy Learning Surface

This directory is the landing zone for Vicuña's production RL runtime surface
and its first offline learning pipeline.

The current implementation boundary is intentionally narrow:

- the server captures one typed governance transition per completed request
- operators can inspect bounded runtime state at `GET /v1/policy/status`
- offline tooling can export recent transitions at `GET /v1/policy/transitions`
- shadow mode can request bounded candidate actions without changing native
  execution
- offline tooling can persist those transitions into versioned local datasets
- offline tooling can evaluate candidate policies against those datasets before
  serving
- offline tooling can materialize a first masked training contract for the
  current governance and provider-control heads
- offline tooling can train deterministic candidate artifacts from that
  contract, register them with aliases and lineage, run the loop nightly, and
  serve registry aliases over HTTP for live rollout

The live runtime still does not train in-process, but it can now consume
registry-backed proposals through the separate rollout feature in
`specs/149-policy-rollout-surface/`.

## Runtime controls

- `VICUNA_POLICY_MODE=capture` records transitions only
- `VICUNA_POLICY_MODE=shadow` records transitions and compares a candidate
  policy proposal against the native controller
- `VICUNA_POLICY_MODE=eval_only` requests candidate proposals but always keeps
  execution native
- `VICUNA_POLICY_MODE=canary_live` executes candidate proposals only for the
  active canary slice after confidence and safety checks
- `VICUNA_POLICY_MAX_TRANSITIONS` bounds the in-memory export window
- `VICUNA_POLICY_CANDIDATE_URL` points rollout modes at a proposal service
- `VICUNA_POLICY_TIMEOUT_MS` bounds candidate lookup latency
- `VICUNA_POLICY_CANARY_STEPS`,
  `VICUNA_POLICY_CANARY_MIN_REQUESTS_PER_STEP`,
  `VICUNA_POLICY_LIVE_CONFIDENCE_THRESHOLD`,
  `VICUNA_POLICY_ROLLBACK_MAX_CANDIDATE_FAILURE_RATE`,
  `VICUNA_POLICY_ROLLBACK_MAX_INVALID_ACTION_RATE`, and
  `VICUNA_POLICY_ROLLBACK_MAX_FALLBACK_RATE` bound live rollout

## Offline tools

Use `python3 tools/policy-learning/cli.py` with these subcommands:

- `export` snapshots or follows the runtime export surface into a durable local
  dataset directory
- `build-training-set` converts persisted transitions into
  `vicuna.governance_masked_heads.v2` trainer-facing records
- `train` builds a deterministic masked behavior-cloning artifact plus a
  training run manifest
- `evaluate` replays a candidate policy against the exported dataset and emits
  disagreement, validity, and reward diagnostics
- `register` copies a trained artifact and its reports into the local registry
- `promote` moves a registry alias such as `candidate` or `champion`
- `registry-status` reports alias assignments and version history
- `nightly-batch` runs export, build-training-set, train, evaluate, register,
  and threshold-gated `candidate` promotion in one pass
- `serve-registry` serves a chosen registry alias or fallback alias at
  `POST /v1/policy/propose`

The included `example_candidate_policy.py` provides a deterministic command
adapter for smoke tests and local experimentation. The included
`registry_policy_adapter.py` replays a trained artifact through the same
offline evaluator interface.

## Operator checks

After one request, confirm:

- `/v1/policy/status` reports the active mode, stored transition count, and
  shadow disagreement or failure counters, plus the active desired-state
  reward model
- `/v1/policy/transitions?limit=N` returns typed observations, executed
  actions, applied provider controls, desired-state reward model, reward
  breakdown, and shadow-evaluation metadata when present
- `/health` includes a `policy_runtime` object for coarse service health

Offline loop checks:

- `python3 tools/policy-learning/cli.py export ...` creates `manifest.json`
  plus `data/transitions.jsonl`, including reward-model metadata in the
  dataset manifest
- `python3 tools/policy-learning/cli.py build-training-set ...` creates
  `training/policy_training_manifest_v1.json`
- `python3 tools/policy-learning/cli.py train ...` creates a run-scoped
  `artifact.json` plus `training_run_manifest.json`
- `python3 tools/policy-learning/cli.py register ...` copies those artifacts
  into `<registry-dir>/<model-name>/versions/<n>/`
- `python3 tools/policy-learning/cli.py nightly-batch ...` writes one
  registry-backed batch report under `batch-runs/`
- `python3 tools/policy-learning/cli.py serve-registry ...` exposes
  `/health` plus `/v1/policy/propose` for live rollout

## Research split

- `specs/144-policy-learning/` remains the broad RL research and design track
- `specs/146-rl-runtime-surface/` is the separate implementation spec for the
  first production runtime slice
- `specs/148-policy-learning-pipeline/` is the separate implementation spec
  for the offline export/evaluation/training-contract loop
- `specs/151-offline-trainer-registry/` is the separate implementation spec
  for offline training, registry, and nightly orchestration
- `specs/149-policy-rollout-surface/` is the separate implementation spec for
  live canary, rollback, and learned-policy serving
- `specs/153-policy-api-control/` is the separate implementation spec for the
  richer DeepSeek request-shaping control surface
