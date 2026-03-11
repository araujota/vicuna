# Research: Strict Live Memory Runtime and Typed Self-State

## Scope

This feature directory now carries two related research tracks:

1. near-term live-serving correctness for the memory cascade
2. follow-on architecture for the persistent self-state and register bank described in Sections 5-8 of `Vicuña_WP.md`

The second track stays at the architecture and implementation-guideline level. It does not introduce runtime code in this repository.

## Decision: Separate request adapters from runtime memory adapters and rebuild one ordered effective serving stack

**Rationale**:
- The current serving path replaces the context adapter set per batch, which can drop Active and Past memory adapters during live inference.
- Separating request-owned adapters from runtime-owned memory adapters preserves the memory cascade during serving without forcing request code to understand memory internals.
- Rebuilding one explicit effective stack keeps composition policy inspectable and future-proof.

**Alternatives considered**:
- Keep one shared adapter map and try to merge into it opportunistically in the server: rejected because ownership stays ambiguous and future request changes can regress memory preservation again.
- Push memory adapters into request task state: rejected because runtime memory is not request-owned state.

## Decision: Implement "stacking on top" as explicit precedence and deterministic ordering, not serial non-commutative neural composition

**Rationale**:
- The current `llama.cpp` graph path applies LoRAs as additive deltas on the same base weight matmuls.
- In that setting, "top of stack" is best interpreted as an explicit serving-layer policy with inspectable precedence and per-layer scales.
- This preserves the architectural intent while fitting the actual backend math model.

**Alternatives considered**:
- Leave ordering implicit in an unordered container: rejected because it makes the serving policy uninspectable and risks silent behavioral drift.
- Implement true serial adapter composition now: rejected because it would be a much larger graph change than needed to fix current serving correctness.

## Decision: Treat an accepted Active LoRA write as a strict KV-coherence boundary and replay the retained suffix before further sampling

**Rationale**:
- The retained KV cache encodes activations from the pre-update adapter stack.
- If Active LoRA changes and generation continues from the old retained KV state, subsequent tokens are produced under mixed semantics.
- Replaying only the retained suffix is the narrowest strict fix that restores coherence without rebuilding the entire conversation state.

**Alternatives considered**:
- Cheap policy: continue generation and let only future appended tokens see the new Active LoRA. Rejected because it breaks the intended strict semantics of the memory cascade.
- Full-sequence replay from the beginning after every Active write: rejected because it is much more expensive than replaying only the retained suffix.

## Decision: Run strict replay in the server context-shift path, not inside backend kernels or hidden background threads

**Rationale**:
- Generation-time context shift is currently owned by `tools/server/server-context.cpp`.
- Replaying retained suffix tokens is control-plane policy coupled to slot state, checkpoints, and scheduling, not dense backend math.
- Keeping replay there preserves auditability and limits the change surface.

**Alternatives considered**:
- Put replay policy inside the low-level context/memory backend: rejected because the slot/task semantics live above that layer.
- Hide replay in a background thread: rejected because it obscures causality and makes tests harder.

## Decision: Reset slot-local sampler and invalidate stale checkpoints when strict replay is scheduled

**Rationale**:
- Repetition penalties and related sampler state should reflect the compacted retained prompt, not the pre-shift sequence.
- Prompt checkpoints captured under the old adapter stack can no longer be trusted once Active LoRA changes.
- Resetting sampler/checkpoint state at the replay boundary is simpler and safer than trying to prove partial reuse remains coherent.

**Alternatives considered**:
- Keep sampler state unchanged: rejected because it can retain evicted-token influence.
- Reuse old checkpoints after adapter changes: rejected because they encode stale KV state.

## Decision: Skip replay when Active ingestion does not materially change weights

**Rationale**:
- The Active path already rejects highly redundant spans.
- If no weight change occurred, replay work would add latency without improving coherence.
- An explicit skip result keeps the policy auditable.

**Alternatives considered**:
- Replay unconditionally on every context shift: rejected because it wastes work for redundant spans and weakens the point of explicit admission control.

## Decision: Represent the persistent self-core as a typed belief state, not as prompt text and not as one opaque latent vector

**Rationale**:
- POMDP and belief-state work treats the belief state as the sufficient statistic that carries history forward under partial observability.
- Neural belief-state and world-model work shows that robust recurrent state benefits from explicit transition and observation updates, and from carrying uncertainty rather than only point estimates.
- Agent-memory work continues to move toward structured, dynamically organized memory rather than flat summary text.
- For Vicuña specifically, a typed state surface is required for auditability, governance, and eventual self-modification of update policies.

**Recommended shape**:
- explicit register bank for control variables
- separate identity and goal vectors
- explicit tool, channel, and time surfaces
- sparse handles into working-memory clusters and frozen LoRA buckets
- per-register confidence or variance where the source is noisy

**Alternatives considered**:
- Keep the self entirely in prompt summaries: rejected because it is fragile and not operationally inspectable.
- Collapse the self into one learned latent vector: rejected because it is hard to govern and too brittle for later self-modification.

## Decision: Use a hybrid state algebra for registers rather than treating every register as one free scalar

**Rationale**:
- Sections 6-8 list variables with different mathematical roles: bounded pressures, categorical channel state, per-cluster replay priorities, and time features.
- State-space and filtering literature supports matching the estimator to the variable family instead of forcing one update rule everywhere.
- A typed algebra lets the system maintain hard bounds, uncertainty, and provenance for each register family.

**Recommended families**:
- bounded continuous scalars in `[0, 1]` for pressure and relevance values
- signed bounded scalars in `[-1, 1]` for directional residuals
- simplexes for categorical or competing priorities
- sparse maps for `r_reactivation_priority[m_i]`
- immutable raw timestamps plus derived cyclic features for system datetime

**Alternatives considered**:
- One unconstrained float per register: rejected because it makes drift, calibration, and normalization harder.
- Text-only labels for categories like channel state: rejected because downstream control logic needs explicit typed values.

## Decision: Formalize register recomputation as a predict/admit/update filter

**Rationale**:
- The whitepaper's two-stage order already matches the structure of recursive filtering: compare the new observation to prior state, decide admission, then update state after the write.
- Belief-state work and world-model work both rely on explicit transition and observation updates rather than unstructured "recompute everything from scratch" heuristics.
- This keeps the update path stable enough to simulate counterfactual changes later.

**Recommended order**:
- `predict`: apply decay, elapsed-time dynamics, and tool/channel transitions to prior state
- `prewrite observe`: build features against prior state and compute provisional deltas
- `admit`: decide whether the event enters working memory and/or durable memory paths
- `postwrite update`: recompute action, broadcast, and replay policy against the updated working-memory snapshot

**Alternatives considered**:
- Only post-write recomputation: rejected because novelty and contradiction are defined against the prior state.
- Full global recomputation from raw history on each event: rejected because it is too expensive and obscures provenance.

## Decision: Keep embeddings for similarity geometry, but use dedicated heads for contradiction, uncertainty, tool readiness, and broadcast policy

**Rationale**:
- Embeddings remain the right substrate for novelty, topic shift, self relevance, goal relevance, and retrieval.
- Contradiction and calibrated uncertainty require specialized estimators.
- Recent work on self-evaluation and prefix-level entailment supports using dedicated heads or verifiers for uncertainty and inconsistency detection during generation.

**Recommended feature pipeline**:
- event embedding
- retrieved self, goal, memory-cluster, and tool-state neighbors
- similarity and dispersion features
- decoder entropy and top-margin
- contradiction scores from an NLI or prefix-entailment head
- uncertainty scores from a calibration head or `P(IK)`-style estimator
- tool delta flags and elapsed-time features

**Alternatives considered**:
- Embedding-only register updates: rejected because contradiction, calibration, and tool readiness are not reliably recoverable from cosine similarity alone.

## Decision: Implement learned contradiction and uncertainty support as optional callback-backed heads over typed feature vectors

**Rationale**:
- Current uncertainty work has moved toward semantic uncertainty estimators such as semantic-entropy-derived probes and kernelized semantic entropy rather than raw token entropy alone.
- Prefix-level contradiction detection work supports dedicated inconsistency heads operating on structured features instead of embedding similarity by itself.
- A callback-backed interface preserves runtime extensibility without baking one probe architecture or model family into the core inference loop.

**Recommended shape**:
- keep an always-available analytic path using lexical, timing, and decoder features
- expose typed prewrite/postwrite feature vectors
- allow contradiction and uncertainty heads to be enabled independently behind explicit flags
- clamp or reject invalid learned-head outputs before register updates

## Decision: Include raw system datetime in the persistent self-state and derive model-facing time encodings from it

**Rationale**:
- The whitepaper explicitly requires time-state encodings.
- Time2Vec and related temporal-feature work support storing raw time plus derived periodic encodings instead of only text summaries such as "it is evening."
- A typed time surface avoids silently losing absolute time during context shifts.

**Recommended fields**:
- `wall_clock_unix_ms`
- `monotonic_elapsed_ms`
- `timezone_offset_minutes`
- local calendar decomposition
- cyclic encodings for hour-of-day, day-of-week, and day-of-year
- elapsed deltas since key events such as last user message, last tool completion, and last self-initiated emission

## Decision: Represent social and relationship state as bounded persistent scalars before attempting richer symbolic structure

**Rationale**:
- Section 5 requires social state inside the self-core, but the whitepaper also emphasizes typed inspectable state rather than prompt summaries.
- For the first production slice, bounded scalars for familiarity, trust, and reciprocity preserve prefix persistence across KV eviction and remain easy to audit and replay.
- This matches the broader control-system approach used elsewhere in the self model.

**Recommended shape**:
- `familiarity in [0,1]`
- `trust in [0,1]`
- `reciprocity in [0,1]`
- derived `bond_strength in [0,1]`

## Decision: Use in-tree linear probe heads over typed scalar features as the first built-in learned-head implementation

**Rationale**:
- The whitepaper calls for small learned probes, and Sections 6-8 still prefer scalar mathematical state wherever feasible.
- Linear probes over the typed feature basis keep the runtime interpretable and cheap while still moving beyond hand-authored if/else heuristics.
- Semantic-entropy and prefix-level inconsistency work support specialized verifier heads, but the first deployable in-tree form should remain auditable and replay-stable.

## Decision: Future self-modifiable updater logic should be represented as versioned declarative programs with fixed safety invariants

**Rationale**:
- This is an architectural inference from the whitepaper plus model-based planning literature, not a direct claim from one source.
- If the system will eventually simulate counterfactual modifications to its own register-update logic, those update rules must be serializable, inspectable, and replayable on frozen traces.
- Free-form code mutation would make counterfactual evaluation and rollback much harder.

**Recommended constraint**:
- the schema and invariants of the register bank stay human-governed
- self-modification may propose new coefficients, thresholds, feature selections, or updater graphs inside a constrained DSL
- proposals must run on frozen event traces and synthetic rollouts before activation

## Analysis of Whitepaper Sections 5-8

### Section 5: Persistent Self-Representation

Strong:
- correctly separates durable self-state from prompt text and LoRA-only storage
- already identifies tool state, commitments, and time as explicit surfaces

Missing precision:
- "identity embedding" is too coarse by itself; the durable self should be factorized into identity, goals, commitments, relationship state, and control registers
- the section should distinguish raw state from derived features and from memory handles

### Section 6: Register Bank

Strong:
- register families are the right decomposition for pressure, action, memory, and environment control

Missing precision:
- each register needs a domain, bounds, decay rule, uncertainty rule, provenance, and update operator
- `r_channel_state` should be categorical, not just another scalar
- `r_time_phase` should be derived from raw datetime plus elapsed deltas, not treated as one opaque number

### Section 7: Register Recomputation

Strong:
- the two-stage order is the right one

Missing precision:
- the paper should explicitly frame this as `predict -> prewrite observe -> admit -> postwrite update`
- recomputation should preserve provenance so later self-modification can replay prior traces under candidate updater versions

### Section 8: Feature Extraction

Strong:
- the pipeline already separates embeddings, retrieval, feature construction, and update heads

Missing precision:
- contradiction and uncertainty should be called out as dedicated verifier heads
- time features should include raw absolute time plus derived cyclic and elapsed features
- feature builders should output both values and confidence/quality scores

## External Sources Consulted

- Kaelbling, Littman, Cassandra, "Planning and Acting in Partially Observable Stochastic Domains" (1998): https://doi.org/10.1016/S0004-3702(98)00023-X
- Gregor et al., "Neural Predictive Belief Representations" (2018): https://arxiv.org/abs/1811.06407
- Doerr et al., "Probabilistic Recurrent State-Space Models" (2018): https://arxiv.org/abs/1801.10395
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2018): https://arxiv.org/abs/1811.04551
- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023/2024): https://arxiv.org/abs/2301.04104
- Kazemi et al., "Time2Vec: Learning a Vector Representation of Time" (2019): https://arxiv.org/abs/1907.05321
- Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023/2024): https://arxiv.org/abs/2310.08560
- Xu et al., "A-MEM: Agentic Memory for LLM Agents" (NeurIPS 2025): https://arxiv.org/abs/2502.12110
- Kadavath et al., "Language Models (Mostly) Know What They Know" (2022): https://arxiv.org/abs/2207.05221
- Kossen et al., "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs" (2024): https://arxiv.org/abs/2406.15927
- Nikitin et al., "Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities" (2024): https://arxiv.org/abs/2405.20003
- Harary et al., "PrefixNLI: Detecting Factual Inconsistencies as Soon as They Arise" (2025): https://arxiv.org/abs/2511.01359

## GitHub Codebase Research

The following codebases were inspected with the GitHub MCP server as implementation references, not as copy targets:

- `letta-ai/letta`
  - `letta/schemas/memory.py`
  - useful takeaway: keep "core" memory typed and explicitly segmented instead of flattening everything into one opaque summary buffer
- `mem0ai/mem0`
  - `mem0/memory/main.py`
  - `mem0/memory/graph_memory.py`
  - useful takeaway: separate session-scoped memory records from graph-style entity relations, keep retrieval metadata explicit, and preserve update history instead of only storing latest values
- `danijar/dreamerv3`
  - `dreamerv3/agent.py`
  - useful takeaway: keep state transitions explicit and replayable, with a stable latent-state contract between observation, transition, and downstream policy heads
- `langchain-ai/langgraph`
  - `libs/checkpoint/langgraph/checkpoint/base/__init__.py`
  - `libs/checkpoint/langgraph/checkpoint/memory/__init__.py`
  - useful takeaway: treat replayable state as versioned channels with explicit checkpoint metadata, parent linkage, and deterministic write ordering rather than as an opaque blob
- `letta-ai/letta`
  - `letta/schemas/llm_batch_job.py`
  - `letta/services/job_manager.py`
  - useful takeaway: model tool/job lifecycle as typed status transitions with persisted timestamps and safe transition guards so self-state control policy can reason over explicit async execution state

## Follow-on Architecture Decision: Keep updater self-modification constrained to declarative coefficient programs before attempting a richer DSL

**Rationale**:
- The current runtime now supports counterfactual replay over typed traces, which is enough to safely evaluate bounded coefficient changes.
- LangGraph's checkpoint design reinforces that replayable state benefits from explicit schema versioning and deterministic write order before more expressive policy mutation is introduced.
- Letta's typed job schemas reinforce that state-machine evolution should begin from constrained explicit fields and guarded transitions rather than free-form mutation.

**Recommended next step**:
- treat the first self-modifiable updater surface as a versioned coefficient program over typed features
- add richer declarative control-flow constructs only after replay/export compatibility and evaluator metrics stabilize

Implementation implication for Vicuña:
- the self-state runtime should remain a typed control-plane state machine
- retrieval-backed features should be derived from explicit stores and handles
- replayability matters more than maximizing representational cleverness in the first implementation

## Follow-on Architecture Decision: Upgrade updater programs from one shared gain to bounded per-register rules with asymmetric dynamics

**Rationale**:
- DreamerV3's recurrent update core keeps transition dynamics explicit and bounded through gated state updates, which is a useful pattern even though Vicuña's self-state stays CPU-side and scalar-valued.
- Neural predictive belief-state work and filtering literature support transition rules with inertia, asymmetric correction, and explicit prior pull rather than one uniform blend for every latent dimension.
- For Vicuña specifically, per-register rules preserve inspectability and replay while giving future self-modification something richer than a flat coefficient bag to tune.

**Recommended constraint**:
- each bounded scalar register gets an explicit update rule
- each rule may read typed feature ids plus a small bounded set of scalar source registers
- each rule exposes `baseline`, `rise_gain`, `fall_gain`, and `baseline_pull`
- the rule output remains clamped to the register's declared bounds

Implementation implication for Vicuña:
- prewrite and postwrite phases should evaluate rule sets against a snapshot of current scalar registers
- counterfactual replay should compare updater programs at the rule level, not only at the global-weight level

## Follow-on Architecture Decision: Keep frozen-LoRA bridging consolidation-driven

**Rationale**:
- The whitepaper's memory cascade treats Active LoRA as the first durable stage after token eviction and Past LoRA as a slower consolidation hierarchy.
- Letting every admitted message mutate frozen-bucket semantics would blur that boundary and reduce the point of having explicit consolidation intervals.
- The self-model still needs persistent handles into the frozen hierarchy, but those handles should mirror real consolidation state rather than act as a hidden side-channel for per-message memory writes.

Implementation implication for Vicuña:
- admitted events may update working-memory clusters and Active LoRA related state
- frozen-bucket handles should only be refreshed from `past_lora_tick()` and remain part of the persistent prefix between ticks

## Follow-on Architecture Decision: Treat counterfactual replay as a dedicated channel, not as the primary interactive lane

**Rationale**:
- The whitepaper explicitly says the loops do not share the same broadcast policy.
- Dreamer-style systems separate `observe` from `imagine` rollouts; the replay machinery can be shared, but imagined trajectories should not be confused with externally grounded interaction state.
- LangGraph's checkpoint model reinforces that state lanes should be explicit and versioned instead of being implicit side effects on one shared channel.

Implementation implication for Vicuña:
- keep counterfactual replay inside the self-state runtime
- mark replayed events with a dedicated `counterfactual` channel
- preserve shared typed state math while suppressing primary-channel side effects such as outward activation, social-turn accumulation, and user/tool/emit anchors
- expose channel-aware replay APIs so a future subsystem can trigger counterfactual processing without re-implementing replay logic
