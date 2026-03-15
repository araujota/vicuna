# Architecture Guideline: Persistent Self-State and Register Bank

## Purpose

This document translates Sections 5-8 of `Vicuña_WP.md` into an implementation-grade architecture guideline that is compatible with the current `003-strict-memory-runtime` work.

It is intentionally limited to architecture, math-facing data design, and implementation guidance. It does not introduce runtime code.

## Governing Position

The persistent self should be modeled as a typed belief state with explicit invariants:

```text
S_t = {
  R_t,   // register bank
  I_t,   // identity surface
  G_t,   // goal surface
  U_t,   // user/social surface
  T_t,   // tool surface
  Tau_t, // time surface
  C_t,   // commitments and contradictions
  H_t    // memory handles and cluster links
}
```

Why this form:
- It preserves continuity across context eviction.
- It is inspectable and replayable.
- It supports counterfactual simulation of updater changes later.
- It avoids collapsing the self into prompt text or one opaque latent vector.

## Section Analysis

### Section 5: Persistent Self-Representation

Recommended interpretation:
- Keep self-state separate from LoRA memory and working memory.
- Treat the self-core as a control surface, not as narrative prose.
- Factor the "identity embedding" into multiple typed surfaces instead of one vector standing in for everything.

Required additions:
- raw system datetime must live in the self-core
- each state surface needs provenance and last-update metadata
- memory references should be handles, not copied summaries

### Section 6: Register Bank

Recommended interpretation:
- Registers are not a bag of floats.
- Each register must declare its domain, bounds, update rule, and confidence semantics.
- Register families should share update operators where possible.

### Section 7: Register Recomputation

Recommended interpretation:
- The whitepaper's two-stage update is correct.
- It should be implemented as a filter with explicit intermediate states so traces can be replayed under future updater versions.

### Section 8: Feature Extraction

Recommended interpretation:
- Embeddings remain central, but only for geometry-heavy signals.
- Contradiction, uncertainty, and broadcast policy must use specialized heads or estimators.
- Feature outputs should carry both values and quality/confidence metadata.

## Mathematical State Design

### Persistent State Surface

Use a factorized state rather than one monolith:

```text
R_t   = typed control registers
I_t   = stable identity anchors
G_t   = active goal embeddings and scores
U_t   = user relationship and channel state
T_t   = tool lifecycle state
Tau_t = raw and derived time state
C_t   = commitments, unresolved contradictions, promise ledger
H_t   = memory-cluster pointers, frozen-LoRA handles, retrieval priors
```

Each component should have:
- a canonical typed storage form
- a derived feature view for update heads
- last-update timestamp
- provenance for the evidence that changed it
- a schema version

### Register Algebra

Use distinct mathematical families.

`[0, 1]` bounded scalars:
- `r_uncertainty`
- `r_novelty`
- `r_goal_relevance`
- `r_self_relevance`
- `r_social_relevance`
- `r_affordance`
- `r_broadcast_pressure`
- `r_broadcast_inhibition`
- `r_followup_continuation`
- `r_memory_write_priority`
- `r_tool_salience`

`[-1, 1]` signed bounded scalars:
- contradiction residuals
- directional control residuals if later needed

categorical/simplex state:
- `r_channel_state`
- tool readiness categories if modeled probabilistically

sparse keyed maps:
- `r_reactivation_priority[m_i]`
- per-goal or per-commitment salience extensions

raw plus derived time fields:
- system datetime and elapsed-time surfaces

Each register definition should declare:
- `register_id`
- `family`
- `domain`
- `default_value`
- `decay_operator`
- `update_operator`
- `uncertainty_mode`
- `feature_dependencies`
- `hard_bounds`
- `version`

### Register Value Record

Each live register value should carry:

```text
value
confidence_or_variance
last_update_unix_ms
last_update_monotonic_ms
source_mask
updater_version
dirty
```

This is required for later replay and self-modification audits.

## Time Surface

System datetime must be first-class.

Store raw time:
- `wall_clock_unix_ms`
- `monotonic_elapsed_ms`
- `timezone_offset_minutes`
- `local_year`
- `local_month`
- `local_day`
- `local_hour`
- `local_minute`
- `day_of_week`

Store derived features:
- `hour_sin`, `hour_cos`
- `weekday_sin`, `weekday_cos`
- `year_day_sin`, `year_day_cos`
- `delta_since_last_user_ms`
- `delta_since_last_tool_event_ms`
- `delta_since_last_emit_ms`
- `session_age_ms`

Guideline:
- raw fields are the source of truth
- derived fields are cached features
- derived fields are recomputed, not directly edited

## Update Architecture

### Core Rule

Implement register recomputation as:

```text
predict
-> prewrite features
-> provisional register update
-> admission decision
-> working-memory write
-> postwrite features
-> final register update
```

### Predict Stage

Input:
- prior state `S_(t-1)`
- elapsed time
- tool deltas
- channel transitions

Operations:
- apply decay to bounded scalar registers
- roll forward elapsed-time accumulators
- propagate tool state machines
- age memory-cluster replay priorities

Output:
- predicted prior `S_t^pred`

### Prewrite Stage

Input:
- incoming event `o_t`
- predicted prior `S_t^pred`
- retrieved memory neighbors

Compute:
- event embedding
- similarity to working memory, goals, self, and memory clusters
- dispersion among retrieved candidates
- decoder entropy and top-margin if available
- contradiction score
- uncertainty score
- tool delta flags
- time deltas

Update:
- `r_uncertainty`
- `r_contradiction`
- `r_novelty`
- `r_topic_shift`
- `r_goal_relevance`
- `r_self_relevance`
- provisional `r_memory_write_priority`

Output:
- provisional state `S_t^-`
- admission bundle `A_t`

### Admission Stage

Decide separately:
- write to working memory?
- tag as unresolved commitment?
- trigger durable memory write pressure?
- attach tool affordance markers?

This stage should be typed and thresholded, not prompt-based.

### Postwrite Stage

Input:
- admitted working-memory snapshot
- provisional state `S_t^-`
- resulting tool/channel/time state

Update:
- `r_broadcast_pressure`
- `r_broadcast_inhibition`
- `r_followup_continuation`
- final `r_memory_write_priority`
- `r_reactivation_priority[...]`
- `r_tool_salience`
- derived time-state registers

Output:
- final posterior `S_t`

## Feature Extraction Stack

### Base Feature Groups

Geometry features:
- event embedding
- cosine and dot-product similarity to working memory
- similarity to goal embeddings
- similarity to self anchors
- similarity to retrieved memory clusters
- candidate dispersion and cluster entropy

Decoder features:
- token entropy
- top-1/top-2 margin
- answer self-evaluation score
- refusal likelihood if available

Verifier features:
- contradiction score from NLI or prefix-entailment head
- commitment-conflict score against promise ledger
- tool-readiness classifier

Environment features:
- tool started/completed/failed flags
- channel activity state
- system datetime features
- elapsed-time features

### What Should Stay Embedding-Based

Use embedding geometry for:
- novelty
- topic shift
- goal relevance
- self relevance
- memory retrieval
- memory-cluster reactivation

### What Should Not Stay Embedding-Only

Use dedicated heads for:
- contradiction
- calibrated uncertainty
- tool readiness
- broadcast policy

## Updater Programs

The register update logic should be represented as versioned updater programs.

Suggested abstraction:

```text
updater_program {
  version
  feature_schema
  predict_ops[]
  prewrite_ops[]
  postwrite_ops[]
  bounds
  safety_invariants
}
```

Why:
- makes updates replayable on frozen traces
- allows shadow evaluation of alternative programs
- creates a clean boundary for future self-modification

### Self-Modification Constraints

Future self-modification should be limited to:
- coefficients
- thresholds
- feature selection
- operator composition inside a constrained DSL

It should not be allowed to mutate directly:
- register schema
- safety bounds
- provenance requirements
- audit logging requirements

## Extensibility Rules

Add a new register only if all of the following exist:
- semantic definition
- mathematical domain
- update operator
- decay rule or explicit statement that none applies
- provenance source set
- tests or replay traces for expected behavior

Add a new feature extractor only if:
- it declares latency cost
- it declares whether it is online-safe for the active loop
- it exposes confidence or quality metadata when relevant

## Implementation Guidance for This Repository

### Near-Term Integration Order

1. Finish strict live-serving and KV-coherence work from the current `003` plan.
2. Add a typed CPU-side self-state container and trace logger before any learned register heads.
3. Implement time and tool surfaces first, because they are explicit and low-risk.
4. Add prewrite/postwrite feature builders.
5. Add bounded analytic update operators before learned heads.
6. Add learned contradiction and uncertainty heads behind feature flags.
7. Add replay tooling for offline counterfactual evaluation of updater versions.

### Runtime Placement

Keep these in CPU-side control code:
- state container ownership
- register schema
- updater program execution
- provenance logging
- self-modification governance

Keep dense math swappable:
- embedding generation
- contradiction head
- uncertainty head
- retrieval scorer

### Storage Guideline

Do not serialize the self as plain text.

Persist instead:
- typed binary or structured records for the state surfaces
- versioned updater configuration
- event traces needed for replay
- memory-handle references into working memory and LoRA buckets

## Validation Guidance

Minimum validation set for the self-state work:
- state survives prompt-window eviction
- raw system datetime stays correct across idle periods and restarts
- prewrite and postwrite updates produce different, explainable deltas
- contradiction head can raise `r_contradiction` without requiring high cosine distance
- uncertainty head is better calibrated than entropy alone
- replaying the same event trace under the same updater version is deterministic
- replaying under a candidate updater version produces auditable diffs

## Recommended Next Artifact

When implementation begins, add:
- a dedicated self-state `data-model.md`
- a contract for updater-program execution and replay
- tasks split into state container, time/tool surfaces, feature builders, analytic updates, learned heads, and replay governance
