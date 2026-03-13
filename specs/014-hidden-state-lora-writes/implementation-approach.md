# Implementation Approach: Hidden-State-Derived Active LoRA Embeddings And Feature-Derived Write Rules

## Executive Position

The right first move is not "increase rank everywhere." The right first move is
to improve the semantic quality of the writer while preserving the existing
memory-stack invariants.

Today, Active LoRA rank mostly controls slot capacity and overwrite cadence.
The larger problem is that the write path is still driven by low-information
signals:

- 64-dim hash or token-pool embeddings for admission
- token IDs modulo 1024 for one side of the update
- no direct use of the base model's own hidden-state geometry

The proposed architecture changes that in two steps:

1. Make the default embedder use the serving model's own hidden-state-derived
   span representations.
2. Make the write rule derive bounded low-rank directions from content plus
   typed self-state features instead of raw token identity.

## Goals

- Form durable behavior and style biases rather than only token-pattern residue.
- Keep hard memory and LoRA memory complementary rather than competitive.
- Preserve:
  - serving-layer order
  - temporal bucket order and condensation semantics
  - remediation parity
  - counterfactual LoRA-ablation parity
- Keep runtime policy explicit, inspectable, and bounded.

## Non-Goals

- Replacing Supermemory or other hard-memory retrieval.
- Turning the runtime into a full gradient-training loop.
- Introducing opaque learned writer networks with no typed observability.
- Breaking current Active/Past LoRA application order.

## Recommended Architecture

### 1. Split The Current "Embedder" Into Two Distinct Surfaces

The current code uses one embedding both for admission/dedup and for the write.
That is too restrictive.

Use two related but distinct surfaces:

- `admission_embedding`
  - used for redundancy suppression
  - memory-handle similarity
  - cluster assignment
  - update skipping
- `write_features`
  - used for actual direction/gain construction
  - richer than a single pooled span embedding
  - includes self-state and event metadata

This keeps dedup logic simple while allowing the writer to encode goals,
continuation, social posture, repair pressure, and tool context.

### 2. Default Hidden-State Embedder: Minimal-Invasive Phase

Use the current base-model embedding outputs first.

Implementation path:

1. Add a hidden-state embedder mode to Active LoRA params.
2. Build or reuse an auxiliary context of the same base model family with
   embeddings enabled.
3. Feed the evicted span through that auxiliary context.
4. Read per-token embeddings with `llama_get_embeddings_ith(...)` or pooled
   sequence embeddings where appropriate.
5. Pool and normalize those vectors into `admission_embedding`.

Why this is the right first phase:

- the codebase already exposes embedding outputs
- it avoids a broad invasive graph-tap change across model architectures
- it keeps the serving model family identical to the memory model family
- it gives immediate semantic improvement over hash and token-pool modes

Recommended default pooling:

- chunk the evicted span if needed
- mean-pool normalized per-token embeddings within chunk
- attention-weight or salience-weight chunk summaries using:
  - question density
  - contradiction/uncertainty pressure
  - negative user valence
  - goal similarity
- mean-pool chunk summaries into one final `admission_embedding`

### 3. Deeper Hidden-State Path: Late-Layer Tap Phase

If final output embeddings are not expressive enough, add an optional second
phase that captures selected late-layer states, for example the last 2-4 layers
or a small late-layer set such as `[-8, -4, -1]`.

This should be a later step because it is more invasive:

- it touches model-execution internals rather than existing public embedding
  outputs
- it risks architecture-specific branching across model implementations
- it increases implementation complexity sharply

Use this phase only after the first hidden-state embedder is measured.

## Proposed Write Rule

### Principle

The writer should no longer pretend that token identity is the main thing to
preserve. It should instead preserve:

- content stance
- interpersonal posture
- tool and follow-up affordance
- tension or repair trajectory
- goal-relevant bias

That means write directions should be derived from:

- hidden-state content summaries
- typed self-state features
- event-role metadata
- remediation or governance context when present

### Write Feature Construction

Construct `write_features` from four groups:

1. Content features
   - pooled hidden-state span embedding
   - span novelty vs prior Active/past embeddings
   - similarity to goals, commitments, and memory handles

2. Self-state features
   - contradiction
   - uncertainty
   - novelty
   - broadcast pressure / inhibition
   - continuation
   - repair pressure
   - tool readiness / pending backlog
   - social relevance / dissatisfaction / recent user valence

3. Event features
   - role: user / tool / system
   - channel: primary / counterfactual
   - decoder entropy and margin when available
   - admission type: ordinary ingest vs remediation

4. Goal and temporal features
   - active goal centroid projection
   - time phase
   - temporal bucket or expected persistence horizon

### Direction Construction

The recommended rule is an activation-informed low-rank write, not a
token-modulo write.

For each selected layer group:

1. Compute a group-specific write context vector:
   `c_g = LN(P_content h + P_state s + P_event e + P_goal g)`

2. Build the right factor from content subspace, not token IDs:
   - derive `V_g` from top-r principal directions of the chunk/token embedding
     matrix or from repeated projections of `c_g`
   - normalize `V_g`

3. Build the left factor from feature-conditioned steering:
   - derive `U_g` from a group-specific projection of `c_g`
   - optionally mix in a projected goal/self-state residual
   - normalize `U_g`

4. Update gain separately:
   - `gain_delta` depends on confidence, novelty, goal relevance, and whether
     the event is a remediation write
   - clip gain through existing bounded-gain machinery

This preserves the current direction/gain representation while making both
factors semantically grounded.

### Why This Is Better Than A Bigger Rank Alone

Increasing rank today mostly preserves more slots of a crude signal. The
proposed writer instead improves:

- what counts as similar
- which spans are admitted
- which layers get updated
- what behavior the update is trying to bias

Once that exists, rank increases become more valuable.

## Layer Allocation Strategy

### Preserve Current Layering First

Do not change serving-layer order or bucket roles.

Also do not remove the current bias toward late layers as the initial default.
Instead, introduce explicit layer groups, for example:

- `late_attn`
- `late_mlp`
- `mid_attn`
- `mid_mlp`

Start by keeping most budget in late groups and allow measured expansion later.

### AdaLoRA Translation

Adopt:

- sensitivity-aware rank reallocation
- explicit min/max rank per group
- periodic rebudgeting

Do not adopt:

- literal SGD-style `tinit` / `tfinal` / `total_step` scheduling

Vicuña-specific rule:

- keep a fixed total Active rank budget
- allocate that budget across layer groups based on smoothed write energy and
  measured downstream utility
- rebalance only every `N` updates or at rollover boundaries

This keeps the spirit of AdaLoRA while matching a live runtime instead of a
training loop.

### Recommended Sensitivity Signals

- cumulative gain contribution
- counterfactual improvement attributed to group presence
- remediation usefulness
- similarity-to-goal improvements
- reduction in contradiction / dissatisfaction after updates

## Magnitude Policy

### DoRA Translation

Keep and deepen the current direction/gain split.

Recommended choices:

- direction factors remain normalized
- gain stays explicit and bounded
- gain update becomes feature-conditioned rather than only energy-conditioned
- optionally promote gain from one scalar per target to a small bounded vector
  per layer group only if later evidence justifies it

Adopt:

- magnitude/direction separation
- stronger low-rank behavior at modest rank

Do not adopt:

- any change that hides magnitude policy behind opaque training-only logic

This is the most compatible external idea for Vicuña because the codebase
already has explicit gain handling.

## Quantization Policy

### QLoRA Translation

QLoRA is not the core answer to runtime memory quality. It is a support idea.

Adopt:

- optional quantized auxiliary context for hidden-state extraction if memory
  pressure demands it
- NF4-style or equivalent low-memory quantization only for the auxiliary path

Do not adopt:

- paged optimizers as if this were a long-running finetuning loop
- replacing the primary live serving-memory stack with a quantized-training
  mindset

Recommended rule:

- full-precision or current-precision serving path remains authoritative
- optional quantized auxiliary path can be introduced only if hidden-state
  extraction cost becomes a blocker

## Parity Requirements

### Serving-Stack Parity

Must remain:

`request -> all_time -> year -> quarter -> month -> week -> active`

The new embedder and writer must not reorder or rename runtime layers.

### Temporal-Bucket Parity

Past-bucket condensation must continue to operate on bounded direction/gain
artifacts. The embedder change is not permission to rewrite past-bucket
semantics.

### Remediation Parity

`active_lora_remediate(...)` and ordinary ingest must share the same writer core.
The difference should be:

- feature emphasis
- budget scale
- trace labeling

not separate incompatible update logic.

### Ablation Parity

`LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION` must remain meaningful.

That means the new writer must keep:

- per-layer-role visibility
- per-bucket visibility
- enough traceability to know what was active, omitted, or downweighted

If layer-group allocation becomes adaptive, the ablation machinery must be able
to see that allocation.

## Staged Rollout

### Stage 1: Hidden-State Admission Embedder

- Replace default hash/token-pool path with a base-model-derived embedder using
  current embedding APIs.
- Keep existing write rule temporarily.
- Add traces and similarity checks.

Value:
improves admission and dedup quality with limited risk.

### Stage 2: Feature-Derived Writer V2

- Replace token-modulo `B` construction.
- Add typed `write_features`.
- Keep current direction/gain normalization and bucket merge semantics.

Value:
improves actual bias formation.

### Stage 3: Periodic Layer Rebudgeting

- Add AdaLoRA-inspired sensitivity tracking and periodic rank redistribution.
- Keep total budget fixed.

Value:
makes rank work harder without breaking budget invariants.

### Stage 4: Optional Late-Layer Taps And Auxiliary Quantized Context

- Add deeper layer extraction only if Stage 1 embeddings are insufficient.
- Add quantized auxiliary path only if resource pressure justifies it.

Value:
improves representational quality or memory efficiency after the simpler path is
measured.

## Recommended API And Runtime Changes

### Public / Cross-Module Surfaces

- extend `llama_active_lora_params` with:
  - hidden-state embedder mode
  - pooling mode
  - auxiliary-context policy
  - rank-allocation policy
- extend Active LoRA stats/traces with:
  - embedder mode
  - fallback mode
  - selected layer groups
  - write-feature summaries
  - allocation summaries

### Internal Runtime Changes

- add a `hidden_state_embedder` implementation under the current callback
  interface or as a new built-in mode
- add a `write_features_v2` builder
- add per-layer-group allocation state
- refactor `train_on_span(...)` into:
  - embed
  - build features
  - allocate groups
  - construct direction slices
  - apply bounded gain update

## Future Test Plan

The implementation should add targeted tests for:

- hidden-state embedder determinism and fallback behavior
- admission parity versus current interface
- write parity between ingest and remediation
- serving-layer order preservation
- temporal-bucket condensation preservation
- LoRA-ablation parity after writer changes
- rank-allocation stability under repeated updates

## Final Recommendation

Implement the hidden-state embedder first using current embedding APIs from the
base model family, not an external embedding model. Then replace the write rule
with a feature-derived direction builder that uses hidden-state content plus
typed self-state features. Keep direction/gain separation, adapt AdaLoRA only
for periodic budget reallocation, and use QLoRA ideas only for an optional
auxiliary extraction path if resource limits demand it.

That sequence gives the highest leverage improvement while preserving the
current layering, temporal, remediation, and ablation invariants.
