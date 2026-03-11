# Gap Closure Plan: Sections 5-8

## Purpose

This document turns `Vicuña_WP.md` Sections 5-8 into a dependency-ordered implementation plan for the remaining self-model and retrieval/update work.

It follows the existing Spec Kit artifact set in `specs/003-strict-memory-runtime/` and assumes the current branch already contains:

- typed self-state storage
- explicit system datetime
- a first register bank
- prewrite/postwrite feature builders
- bounded analytic updates
- optional callback-backed contradiction and uncertainty heads

## Ground Truth

The target end state is the whitepaper runtime described in:

- Section 5: persistent self-representation
- Section 6: typed register bank
- Section 7: two-stage recomputation with admission and postwrite reconciliation
- Section 8: retrieval-backed feature extraction plus dedicated heads

## Dependency Order

### Phase A: Self-Surface Completion

Deliverables:
- persistent identity surface
- active goal surface
- unresolved commitment surface
- explicit tool-state surface
- social/user relationship surface

Why first:
- Sections 5, 6, and 8 depend on these surfaces existing as typed state before retrieval-backed features can be meaningful.

Current status:
- identity, goals, and commitments now exist as hashed sketch surfaces
- tool state now has typed job status, readiness, and pending-pressure surfaces
- social and relationship state remains shallow

Exit criteria:
- self surfaces are writable, countable, replay-safe, and queryable without prompt inspection

### Phase B: Working-Memory Object Model

Deliverables:
- typed admitted-event records
- fixed-budget working-memory ring
- salience, role, and flag metadata on each item
- deterministic admission semantics

Why second:
- Section 7’s ordering requires a real write target between prewrite and postwrite.
- Section 8’s novelty and topic-shift features depend on recent working-memory similarity.

Current status:
- working-memory admission now stores sketch-backed admitted events
- unresolved-question and tool-affordance metadata now exist
- raw span references remain incomplete

Exit criteria:
- admitted events are persisted into working memory with enough metadata to support retrieval and replay

### Phase C: Retrieval-Backed Feature Inputs

Deliverables:
- similarity to working memory
- similarity to goals
- similarity to identity
- similarity to unresolved commitments
- variance across retrieved working-memory candidates

Why third:
- Section 8 explicitly makes retrieval the input to feature building.
- Section 6.6 requires similarity geometry to participate in register updates.

Current status:
- this slice is now partially implemented with sketch similarity and working-memory variance
- memory-cluster and frozen-LoRA retrieval remain missing

Exit criteria:
- prewrite and postwrite features no longer depend only on the immediately previous event

### Phase D: Register-Bank Completion

Deliverables:
- `r_goal_relevance` driven by actual goal retrieval
- `r_self_relevance` driven by identity and commitment retrieval
- `r_reactivation_priority[m_i]` sparse keyed priorities
- confidence/provenance semantics for retrieval-driven updates

Why fourth:
- register completion depends on the self surfaces and retrieval channels above.

Current status:
- goal relevance and self relevance now update from retrieved surfaces
- sparse reactivation priorities now exist over typed memory handles

Exit criteria:
- every Section 6 register family has a corresponding runtime representation and update path

### Phase E: Memory-Handle and Cluster Layer

Deliverables:
- memory-cluster identifiers
- retrieval priors for frozen LoRA buckets and/or future cluster stores
- per-cluster reactivation priority map
- working-memory-to-cluster bridge

Why fifth:
- Sections 5 and 8 require handles to relevant memory clusters or frozen LoRAs.
- `r_reactivation_priority[m_i]` is meaningless without a real keyed memory-handle space.

Current status:
- typed memory handles now exist with sketch-backed centroids
- postwrite now computes sparse per-handle reactivation priorities
- admitted working-memory updates now bridge into working-memory-cluster handles
- frozen-LoRA bucket handles now synchronize from the real Past LoRA stack

Exit criteria:
- postwrite can increase replay pressure on concrete memory handles rather than an abstract future concept

### Phase F: Tool and Channel Policy Completion

Deliverables:
- tool lifecycle states: pending, busy, completed, failed
- async job handles
- stronger channel and interruption policy
- tool readiness features separate from tool salience

Why sixth:
- Section 5.2 explicitly forbids burying tool/job state in LoRA-only memory.
- Section 8.3 calls out tool readiness as not recoverable from embeddings alone.

Current status:
- tool salience exists
- typed tool jobs now expose pending/running/completed/failed state
- readiness and pending pressure now feed postwrite feature construction
- scalar social familiarity/trust/reciprocity now feed social relevance and broadcast
- deeper relationship modeling remains future work

Exit criteria:
- tool-facing control state is typed, inspectable, and usable in broadcast policy

### Phase G: Learned Heads and Calibration

Deliverables:
- optional learned contradiction head
- optional learned uncertainty head
- optional broadcast-policy head
- explicit fallback and calibration tests

Why seventh:
- learned heads should sit on top of a complete typed feature surface, not substitute for it.

Current status:
- callback seams now exist for contradiction, uncertainty, and broadcast policy
- in-tree linear probe heads now exist for contradiction, uncertainty, and broadcast policy
- callback overrides and analytic fallback paths still exist
- calibration refresh remains future work

Exit criteria:
- learned heads can be enabled behind flags without weakening the analytic fallback path

### Phase H: Replay and Counterfactual Updater Substrate

Deliverables:
- frozen event-trace serialization
- deterministic replayer over self-state transitions
- updater-version registry
- counterfactual evaluator for candidate updater variants

Why last:
- this is the whitepaper’s extensibility layer and depends on stable typed surfaces and reproducible updates first.

Current status:
- updater version metadata exists
- deterministic in-memory trace replay now exists for prefix and full-trace rebuilds
- frozen trace export/import now exists
- declarative updater programs and counterfactual replay summaries now exist
- updater programs now include bounded per-register rules with asymmetric rise/fall gains and baseline pull
- counterfactual replay now has a dedicated execution channel instead of borrowing the primary interactive lane
- trace persistence beyond process memory and richer evaluator policy remain incomplete

Exit criteria:
- candidate updater changes can be run on stored traces without mutating live runtime state

## Production Strategy

### Data Representation

- Keep self-state typed and CPU-owned.
- Use mathematical sketches and bounded scalars as the canonical runtime representation.
- Defer heavier learned encoders to pluggable embedders or heads; do not hide policy inside backend kernels.

### Retrieval Strategy

- Near term: use normalized sketch similarity for identity, goals, commitments, and working memory.
- Mid term: add model-specific callback-backed embedders for higher-fidelity retrieval while preserving the same typed feature contract.
- Long term: add memory-cluster and frozen-LoRA handles as separate retrieval namespaces.

### Update Strategy

- Preserve `predict -> prewrite observe -> admit -> postwrite update`.
- Keep all register writes bounded and provenance-stamped.
- Use constrained per-register updater rules over typed scalar features and bounded scalar cross-register couplings.
- Separate provisional and final memory write logic.
- Reserve learned heads for contradiction, uncertainty, and later broadcast policy.
- Keep frozen-LoRA handles synchronized from consolidation ticks rather than mutating them on every admitted message.

### Validation Strategy

- Unit-test every new self-surface write path.
- Regression-test retrieval-backed feature changes with deterministic tokenized events.
- Add trace-replay tests before introducing self-modifiable updater logic.
- Treat out-of-bounds register updates and non-monotonic time as hard failures.

## Immediate Next Steps

1. Add persistent trace storage beyond process memory and format-compatibility tests for long-lived frozen artifacts.
2. Calibrate the existing rule-based updater programs and in-tree probes on held-out traces before considering a richer DSL.
3. Calibrate or retrain the in-tree probe coefficients on held-out traces.
4. Expand social state beyond scalar familiarity/trust/reciprocity if future requirements need richer relationship structure.
