# Policy Learning Surface

This directory is the landing zone for Vicuña's production RL runtime surface
and its first offline learning pipeline.

The current implementation boundary is intentionally narrow:

- the server captures one typed governance transition per completed request
- operators can inspect bounded runtime state at `GET /v1/policy/status`
- offline tooling can export recent transitions at `GET /v1/policy/transitions`
- offline tooling can export recent decode-step traces at
  `GET /v1/policy/decode-traces`
- shadow mode can request bounded candidate actions without changing native
  execution
- offline tooling can persist those transitions into versioned local datasets
- offline tooling can evaluate candidate policies against those datasets before
  serving
- offline tooling can materialize a first masked training contract for the
  current governance and provider-control heads
- offline tooling can train deterministic PPO actor-critic candidate artifacts
  from that contract, register them with aliases and lineage, run the loop
  nightly, and serve registry aliases over HTTP for live rollout
- offline tooling can now also train a separate EM/VAD-to-control-vector MLP
  artifact that maps the 14D emotive moment plus 3D VAD state into a runtime
  steering vector with exact `n_embd` output size
- offline tooling can now also build, train, and evaluate a decode-level GRU
  controller artifact from ordered decode traces, then load that artifact back
  into the local runtime as a shadow or canary candidate rail
- retained rollout tooling can move those candidates through live `shadow`,
  `canary_live`, and `champion` promotion based on explicit host-owned
  thresholds

The live runtime still does not train in-process, but all learned rails can
now participate in one continuous online loop:

- request-level PPO still consumes registry-backed proposals through the
  rollout feature in `specs/149-policy-rollout-surface/`
- decode-level GRU and EM/VAD-conditioned cvec generators are hot-loadable
  through the runtime artifact control plane
- the host rollout manager keeps explicit active and candidate slots for those
  rails, samples requests into live execution, and promotes or rolls back them
  with host-owned thresholds

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
- `build-ppo-training-set` converts persisted transitions plus rollout metadata
  into `vicuna.request_level_ppo.v1` trainer-facing records
- `build-cvec-training-set` converts persisted transitions with known control
  vectors or profile-library matches into `vicuna.emvad_to_cvec.v1`
  generator-facing records
- `build-decode-training-set` converts persisted decode traces into
  `vicuna.decode_level_gru.v1` sequential training records
- `train` builds a deterministic PPO actor-critic artifact plus a training run
  manifest
- `train-ppo` is the explicit alias for the same PPO training path
- `train-cvec-generator` builds a deterministic nonlinear MLP artifact that
  outputs an exact-width control vector for a target model embedding size
- `train-decode-controller` builds a recurrent decode-level GRU artifact that
  consumes per-step EM/VAD plus runtime telemetry windows
- `evaluate` replays a candidate policy against the exported dataset and emits
  disagreement, validity, and reward diagnostics
- `evaluate-ppo` evaluates a PPO artifact directly against the PPO training
  corpus
- `evaluate-cvec-generator` replays a generator artifact against its EM/VAD
  corpus and reports weighted MSE and cosine alignment
- `evaluate-decode-controller` replays a decode-controller artifact against its
  sequential decode corpus and reports action-validity and teacher-agreement
  metrics
- `register` copies a trained artifact and its reports into the local registry
- `promote` moves a registry alias such as `candidate` or `champion`
- `registry-status` reports alias assignments and version history
- `nightly-batch` runs export, build-training-set, train, evaluate, register,
  and threshold-gated `candidate` promotion in one pass
- `serve-registry` serves a chosen registry alias or fallback alias at
  `POST /v1/policy/propose`
- `advance-rollout` inspects runtime status plus registry aliases and writes a
  durable rollout decision for `shadow`, `canary_live`, or `champion`
- `deploy-runtime-artifact` resolves a registered `decode_controller` or
  `cvec_generator` artifact and hot-loads it into the live runtime without a
  restart
- `advance-artifact-rollout` inspects runtime-artifact counters plus registry
  aliases and advances one non-request rail through `shadow`,
  `canary_live`, `champion`, or rollback
- `advance-all-rollouts` advances request-level PPO plus decode/cvec artifact
  rollout state in one host-owned pass
- `write-runtime-artifact-env` resolves promoted `decode_controller` and
  `cvec_generator` aliases into runtime env vars such as
  `VICUNA_LOCAL_DECODE_CONTROLLER_ARTIFACT` and
  `VICUNA_LOCAL_CVEC_GENERATOR_ARTIFACT`

The PPO and EM/VAD control-vector generator commands rely on a numpy-enabled
Python interpreter. On this workstation that means `python3.10`; on hosts, set
`VICUNA_POLICY_PYTHON_BIN` to a numpy-capable interpreter. The retained policy
ops wrappers now prefer `python3.10` automatically when it is present.

The included `example_candidate_policy.py` provides a deterministic command
adapter for smoke tests and local experimentation. The included
`registry_policy_adapter.py` replays a trained artifact through the same
offline evaluator interface. The included `cvec_registry_adapter.py` loads a
generator artifact and emits one runtime-usable control vector from a supplied
`moment` and `vad`.

The decode-trace export is now a first-class training surface for the
distilled GRU inner controller. Unlike the request-level PPO corpus, those
traces stay ordered, preserve previous-action carry, and keep teacher versus
executed decode action separation plus short-horizon local outcome proxies.

## Bootstrap strategy

Use the current native controller as the first teacher, not as a competitor to
skip over:

1. `capture`
   - export native-controller transitions and reward ledgers
2. PPO warm start
   - clone the native controller over the masked request-level action space
3. PPO improvement
   - continue from the warm start with the PPO trainer on reward-bearing
     records
4. offline replay evaluation
   - require bounded invalid-action, fallback, disagreement, and reward
     regression rates before live rollout
5. `shadow` and `eval_only`
   - observe live disagreement and confidence without candidate execution
6. `canary_live`
   - execute only the bounded rollout slice
7. `champion`
   - promote only after explicit canary thresholds hold

Train the cvec generator on a parallel ladder:

1. construct trusted target vectors from handwritten profiles, curated
   libraries, or accepted runtime steering traces
2. fit the EM/VAD-to-cvec MLP to reproduce those targets exactly
3. refine with reward-weighted data while keeping norm caps and clipping active
4. require offline cosine/MSE checks before loading the artifact into runtime
5. roll it out behind the same alias, canary, and rollback discipline used for
   other learned artifacts

Train the decode controller on a parallel ladder:

1. export ordered decode traces with EM/VAD, runtime telemetry, teacher
   actions, executed actions, and short-horizon outcomes
2. distill the native decode controller first with the GRU trainer
3. require offline mask-validity, agreement, and outcome checks before any
   live execution
4. hot-load the candidate into the runtime artifact slot and keep it in
   `shadow` until live disagreement and failure counters stay bounded
5. advance only into request-sampled `canary_live`, then `champion`

## Operator checks

After one request, confirm:

- `/v1/policy/status` reports the active mode, stored transition count, and
  shadow disagreement or failure counters, plus the active desired-state
  reward model
- `/v1/policy/transitions?limit=N` returns typed observations, executed
  actions, applied provider controls, desired-state reward model, reward
  breakdown, and shadow-evaluation metadata when present
- `/v1/policy/runtime-artifacts` reports active/candidate slots and live
  rollout counters for `decode_controller` and `cvec_generator`
- `/health` includes a `policy_runtime` object for coarse service health

Offline loop checks:

- `python3 tools/policy-learning/cli.py export ...` creates `manifest.json`
  plus `data/transitions.jsonl`, including reward-model metadata in the
  dataset manifest
- `python3 tools/policy-learning/cli.py build-training-set ...` still creates
  `training/policy_training_manifest_v1.json` for the legacy masked-head BC
  contract
- `python3 tools/policy-learning/cli.py build-ppo-training-set ...` creates
  `training/ppo_training_manifest_v1.json`
- `python3 tools/policy-learning/cli.py build-cvec-training-set ...` creates
  `training/cvec_training_manifest_v1.json`
- `python3 tools/policy-learning/cli.py build-decode-training-set ...` creates
  `training/decode_gru_training_manifest_v1.json`
- `python3 tools/policy-learning/cli.py train ...` creates a run-scoped
  PPO `artifact.json` plus `training_run_manifest.json`
- `python3 tools/policy-learning/cli.py train-ppo ...` is the explicit PPO
  alias for the same artifact path
- `python3 tools/policy-learning/cli.py train-cvec-generator ...` creates a
  run-scoped generator artifact plus training manifest, including
  `target_embedding_dim`, layer range, normalization, and exact MLP weights
- `python3 tools/policy-learning/cli.py train-decode-controller ...` creates a
  run-scoped recurrent decode-controller artifact plus training manifest
- `python3 tools/policy-learning/cli.py register ...` copies those artifacts
  into `<registry-dir>/<model-name>/versions/<n>/`
- `python3 tools/policy-learning/cli.py promote --artifact-kind decode_controller ...`
  writes kind-scoped alias assignments for the decode rail
- `python3 tools/policy-learning/cli.py promote --artifact-kind cvec_generator ...`
  writes kind-scoped alias assignments for the cvec rail
- `python3 tools/policy-learning/cli.py write-runtime-artifact-env ...` writes
  promoted decode/cvec artifact paths into a runtime env file for pod use
- `python3 tools/policy-learning/cli.py nightly-batch ...` writes one
  registry-backed batch report under `batch-runs/`
- `python3 tools/policy-learning/cli.py serve-registry ...` exposes
  `/health` plus `/v1/policy/propose` for live rollout, including PPO rollout
  metadata such as selected log-probability, entropy, and value estimate
- `python3 tools/policy-learning/cli.py advance-rollout ...` writes retained
  rollout controller state under the state root and emits searchable rollout
  decision logs through the active `policy-rollout` service log
- `python3 tools/policy-learning/cli.py deploy-runtime-artifact ...` resolves
  a registry artifact and hot-loads it into the server through
  `/v1/policy/runtime-artifacts`
- `python3 tools/policy-learning/cli.py advance-artifact-rollout ...` evaluates
  one runtime-artifact rail and updates its live slot assignment
- `python3 tools/policy-learning/cli.py advance-all-rollouts ...` advances PPO,
  decode, and cvec rollout state in one host-owned pass

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
- `specs/156-live-policy-promotion/` is the separate implementation spec for
  retained candidate-to-champion rollout automation on live traffic
- `specs/153-policy-api-control/` is the separate implementation spec for the
  richer DeepSeek request-shaping control surface
- `specs/176-emvad-cvec-generator/` is the separate implementation spec for
  the nonlinear EM/VAD-to-control-vector generator layer
