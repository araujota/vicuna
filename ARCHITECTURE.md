# Vicuña Architecture

## Current Product

Vicuña now ships one primary runtime surface:

- a provider-first HTTP server backed by DeepSeek
- a bounded emotive-moment runtime that processes reasoning and content
  block-by-block
- an immediate VAD projection layer that turns the emotive moment into a
  reusable metacognitive control surface
- an optional local GGUF embedding backend used only to enrich emotive
  estimation when local VRAM is available
- a bounded cognitive replay and heuristic-memory layer that converts negative
  traces into reusable control heuristics and bounded policy biases
- a bounded RL runtime surface that captures typed governance transitions from
  the native control policy, exposes offline export/status endpoints, supports
  shadow candidate evaluation, and can execute registry-backed candidate
  policies for a bounded canary slice with automatic rollback
- an offline policy-learning pipeline that persists exported transitions into
  durable datasets, materializes a masked training contract, trains
  deterministic candidate artifacts, stores them in a local registry, and can
  refresh the offline candidate loop nightly
- a flattened runtime tool surface for media, hard memory, manual skill reads,
  web search, and recurring cron-task lifecycle operations
- a retained Telegram bridge transport that consumes the narrow compatibility
  endpoints exposed by the provider server and delivers parsed rich-plan
  responses through the server-owned outbox

## Explicit Non-Goals

The active architecture no longer includes:

- local chat inference as the main serving path
- slot scheduling, KV-cache orchestration, or router mode as product features
- hidden provider-owned tool selection policy
- DMN, self-state, hard-memory, or Active LoRA control loops as active product
  surfaces
- upstream `llama.cpp` example, benchmark, conversion, and distribution assets

## Request Flow

1. The server accepts an OpenAI-compatible request.
2. Request history is replay-seeded into the emotive runtime from system,
   user, assistant, tool, and runtime-event spans.
3. The current emotive moment, VAD state, heuristic retrieval decision, and
   runtime context are combined into one explicit metacognitive policy
   decision.
4. The runtime injects bounded `SKILLS:` and `MEMORIES:` file-name indexes so
   the model can decide whether to read a specific markdown artifact.
5. The policy sets additive guidance plus explicit provider controls before the
   DeepSeek call is made, including `thinking`, token budget, bounded
   sampling or repetition profiles, prefix or stop profiles, and tool-choice
   posture.
6. Tool-bearing requests use the flattened runtime tool surface directly.
   Legacy staged selection remains available only as an opt-in fallback.
7. Provider output is streamed internally even when the external response is
   non-streaming.
8. Reasoning and assistant text are captured block-by-block.
9. Each block updates the rich emotive moment.
10. The VAD projection is recomputed immediately from the full emotive vector.
11. Previously derived heuristics continue to inject additively as bounded
    control biases and guidance, without replacing the live recomputation path.
12. The runtime records one typed governance transition with the pre-decision
    observation, executed action, reward ledger, desired-state reward model,
    next observation, and rollout metadata including candidate confidence,
    canary sampling, and rollback reasoning when present.
13. Offline tooling can snapshot or follow that transition surface into a
    durable dataset, build a trainer-facing masked-head corpus, train a
    candidate artifact, replay it offline, register it behind alias-based
    metadata gates, and serve the chosen alias over HTTP for rollout.
14. The final response is returned with `vicuna_emotive_trace`, including the
    persisted `final_policy` and `heuristic_retrieval` objects for that turn.
15. After a completed user-facing response, the runtime may persist one
    markdown hard-memory note when it sees a durable preference, confirmed
    fact, verified failure pattern, reusable fix, or project convention.
16. After foreground idleness, background work runs cognitive replay only.
    Recurring host work is no longer discovered from an idle-stage registry;
    it is owned by host cron jobs that invoke the normal runtime surface.

## Bridge Surface

The server retains a narrow bridge compatibility surface:

- `GET /v1/responses/stream`
- `GET /v1/telegram/outbox`
- `POST /v1/telegram/outbox`
- `POST /v1/telegram/approval`
- `POST /v1/telegram/interruption`

Those endpoints exist only to support the retained Telegram bridge. They are
not a general return to the old local orchestration stack.

The bridge is transport/state middleware only:

- it forwards the bounded transcript plus Telegram routing headers
- it persists transcript, option-prompt, and outbox-delivery state
- it handles Telegram Bot API polling, callback transport, document ingestion,
  and outbox delivery
- it parses server-returned rich-plan text into formatted Telegram delivery
- it does not own Telegram prompt construction, runtime tool injection, or
  runtime tool continuation

## Emotive Runtime

The emotive runtime is intentionally bounded and inspectable.

- input is segmented into retained blocks
- each block stores source metadata, text, the emotive moment, the delta from
  the previous block, and the projected VAD/control guidance
- traces are kept only for a bounded recent history window
- the estimator can operate in lexical-only mode or with local embedding
  enrichment
- significant negative foreground traces can admit bounded cognitive replay
  entries
- resolved replay episodes are compressed into persisted heuristic-memory
  records for future retrieval
- each trace also stores the turn's final metacognitive policy decision and
  the heuristic retrieval decision that influenced it
- cron-triggered recurring work reaches the same provider/emotive path as any
  other request, but the scheduler itself now lives outside the server

## Metacognitive Control Policy

The control surface is an explicit CPU-side policy over the live belief state:

- inputs include the current emotive moment, VAD state, replay or bridge
  context, and any matched heuristic bias
- the policy scores direct, reflective, tool-light, tool-heavy, and
  background-defer modes
- it derives reasoning depth, tool aggression, tool parallelism,
  interruptability, replan pressure, early-stop permission, forced
  synthesis pressure, `thinking` mode, and bounded provider-control profiles
  for prefix, stop, sampling, repetition, and tool choice
- the resulting policy is injected additively as a separate system guidance
  message so the original task transcript remains inspectable
- the same policy is also applied directly to the outbound DeepSeek request,
  but only through bounded enums owned by CPU-side code rather than free-form
  learned strings or raw unconstrained numeric knobs
- heuristic retrieval remains bounded and additive: stored heuristics bias the
  policy, but they do not replace live emotive or VAD recomputation

## RL Runtime Surface

The production RL surface is an inspectable runtime data plane, not a hidden
trainer loop.

- each completed request can append one bounded `policy_transition`
- observations capture request context, emotive state, heuristic state, and
  native control outputs before execution
- executed actions are derived from the native CPU-side controller unless
  `canary_live` explicitly samples the request, the candidate proposal passes
  confidence and mask checks, and rollback is not active
- each transition also stores the concrete provider controls that were
  actually applied after DeepSeek compatibility guards, including whether
  `thinking` stayed enabled, whether a prefix was used, and which fields were
  suppressed or defaulted
- reward events still expose a flat ledger, but the primary scalar reward now
  comes from desired-state shaping over all 14 emotive registers plus all 3
  VAD axes: weighted before/after closeness, a potential-style progress term,
  and a terminal-alignment term
- the default desired state is explicit CPU-side config, and
  `VICUNA_POLICY_REWARD_CONFIG_PATH` can override it with validated JSON;
  malformed overrides fail startup
- `GET /v1/policy/status` exposes mode, counters, canary step/share, rollback
  state, candidate alias/version, retained-window size, and the active reward
  model
- `GET /v1/policy/transitions` exposes the bounded typed export used by
  offline training or evaluation tooling
- `shadow` mode may ask an external proposal service for a candidate action,
  but safety guards keep execution on the native action and record only the
  comparison result
- `canary_live` can execute the candidate action only after deterministic
  sampling, confidence gating, and native safety validation

## Offline Policy Learning Pipeline

The next lifecycle stage lives outside the server in `tools/policy-learning/`.

- dataset export snapshots or follows `/v1/policy/transitions` into a durable
  filesystem dataset with manifest metadata
- offline evaluation replays candidate policies against exported observations
  and action masks before any serving decision
- training-contract materialization converts the runtime action into factorized
  masked governance heads suitable for future offline PPO-family or supervised
  baselines
- the first trainer is an inspectable masked behavior-cloning baseline that
  emits JSON artifacts plus training manifests instead of opaque model weights
- a local artifact registry stores immutable versions, mutable aliases, and
  promotion history for `candidate` and `champion`
- a retained host nightly batch can export, rebuild the training corpus, train,
  evaluate, register, and threshold-gate the `candidate` alias without touching
  live serving
- a registry-backed HTTP proposal service can resolve `candidate` or
  `champion` aliases into live bounded action proposals
- rollout and canary serving live in the separate
  `149-policy-rollout-surface` package and remain bounded by native fallback

## Flattened Runtime Tools

The active runtime tool surface is flat and provider-visible:

- `media_read`
- `media_download`
- `media_delete`
- `hard_memory_read`
- `hard_memory_write`
- `web_search`
- `ongoing_task_create`
- `ongoing_task_delete`

This removes the family -> method -> payload waterfall from the default path.
Legacy staged selection is retained only as a compatibility fallback behind
`VICUNA_ENABLE_STAGED_TOOL_FALLBACK`.

## VAD Projection

The VAD surface is derived from the entire emotive-moment vector rather than a
small handpicked subset.

- all emotive dimensions contribute
- cross-terms capture important joint states
- EMA smoothing keeps the output stable across adjacent blocks
- the result includes trend, labels, dominant dimensions, and a bounded
  guidance payload used by the metacognitive control policy

## Build Boundary

The provider-first server remains C++17 and still links against the local
`llama` library only because the optional embedding backend uses the native
model/context APIs. That local embedding path is the only justified retained
dependency on the old native inference stack.

## Heuristic Replay

The replay subsystem is explicit and inspectable rather than latent:

- foreground traces with sharp negative deltas admit replay entries
- idle background replay searches for a better path without allowing recursive
  replay admission
- each resolved replay is compressed into one structured heuristic plus the
  original bad path and better path
- live requests retrieve against stored bad-path objects using bounded exact
  similarity and inject only the matched heuristic
- matched heuristics also produce explicit control biases that alter routing,
  reasoning depth, tool policy, and stop or replan thresholds

## Recurring Host Tasks

Recurring user-directed background work is now host-owned rather than
server-idle-owned:

- `ongoing_task_create` and `ongoing_task_delete` manage `vicuna` user cron
  entries through the runtime harness
- each cron entry runs a retained host wrapper that loads the canonical env,
  acquires a per-task lock, and posts the stored task text as a `system`
  message to the live runtime
- stable task metadata remains local under `/var/lib/vicuna/ongoing-tasks`
  only for safe mutation, logging, and inspection; it is no longer a server
  idle-stage registry
- the server no longer exposes `/v1/emotive/ongoing-tasks` and no longer owns
  a due-decision worker

## Local Parsed Documents

Telegram-ingested Docling artifacts are now local filesystem bundles:

- original files, parsed markdown, chunk data, and metadata live under
  `/home/vicuna/home/docs`
- parsed-document retrieval uses explicit local chunk scanning instead of
  Supermemory search
- document ingestion still parses through Docling on the bridge host, but
  persistence is local and inspectable
