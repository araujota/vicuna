# Vicuña Architecture

This document is the high-level architectural source of truth for Vicuña, a fork of `llama.cpp` aimed at building an organism-level agent runtime rather than a chat wrapper.

The project thesis is simple: Vicuña should not wait passively for prompts. Its own evolving state should continuously generate reasons to retrieve memory, think, use tools, and decide whether to speak.

## Executive Thesis

Vicuña is built around four linked ideas:

1. A memory cascade that moves experience from a sliding attention window into an editable active LoRA, then into a stack of frozen past LoRAs with time-decayed influence.
2. A split runtime loop between active engagement and a default-mode-inspired background process.
3. A persistent self-representation that survives context eviction and is not reducible to prompt text.
4. A path to controlled self-improvement through counterfactual simulation and gated self-update proposals.

The resulting design target is continuous self-conditioned cognition rather than reactive completion.

## Architectural Commitments

### Memory Cascade

The memory model is:

```text
sliding attention window
-> active LoRA
-> past LoRA stack
```

The roles are distinct:

- The sliding window is fast, local, and ephemeral.
- The active LoRA is editable, recent, fixed-size, and lossy.
- The past LoRA stack is fixed-size per unit, frozen after rollover, and time-decayed in influence.

Current implementation constraints for the active stage:

- fixed-size means a proportion of currently available host memory and device memory, not a hard-coded byte cap,
- evicted spans enter the stage through an explicit write path that can be logged and audited,
- Active updates are represented as normalized low-rank directions plus bounded gain scalars so semantic direction and update magnitude remain separately inspectable,
- the default span embedder is now hidden-state-derived from an auxiliary context of the same base model family, while the strategy remains swappable through explicit token-pool, hash, or callback-backed alternatives,
- the write rule now derives bounded low-rank directions from hidden-state content plus typed self-state features instead of token-identity modulo heuristics,
- and auxiliary write machinery must fit the same budget discipline rather than hiding extra long-lived memory outside the adapter accounting.

Current implementation constraints for the past stage:

- the first frozen stack is an ordered bucket set of `past_week`, `past_month`, `past_quarter`, `past_year`, and `all_time`,
- bucket snapshots are only replaced during explicit condensation jobs rather than being incrementally rewritten on every Active update,
- condensation preserves the same direction-versus-gain split used by the Active stage,
- and time decay is implemented as explicit per-bucket inference scaling rather than destructive mutation of frozen weights.

Past LoRAs should decay in application weight, not through destructive rewriting. The memory cascade preserves continuity without pretending that infinite text context is memory.

### Live Serving Sequence

The live decode path applies the memory cascade as an explicit serving stack rather than as side state.

The serving order is:

```text
request adapters (if any)
-> all_time
-> past_year
-> past_quarter
-> past_month
-> past_week
-> active
-> functional_tool_selection
-> functional_planning_composition
-> functional_counterfactual
-> functional_memory_compression
-> functional_self_observation
```

This ordering is a control-policy ordering, not a serial neural pipeline. The graph still applies LoRA deltas additively on top of base weights, but the runtime now keeps the effective stack deterministic and inspectable.

The functional bank is intentionally separate from the temporal cascade:

- temporal LoRAs encode time-sliced behavioral residue from lived history
- functional LoRAs encode reusable operation-local bias for tool choice,
  planning/composition, counterfactual comparison, memory compression, and
  self-observation
- each functional family now keeps two explicit runtime adapters:
  a learned adapter that starts as a true no-op and a tiny bootstrap adapter
  that provides signed stochastic perturbation early in the family lifecycle
- bootstrap perturbation decays with family activation count toward a positive
  floor rather than disappearing entirely
- routing still keeps eligibility, hold windows, and ablations in explicit
  CPU-side policy, but the applied gain now comes from a small shared gating
  MLP driven by a typed self-state gradient toward favorable allostasis
- the gating controller is updated online after settled transactions with an
  explicit meta-loss that minimizes post-action distance to ideal self-state,
  using Adam over bounded perturbation-derived credit signals
- the same Adam family is also used for self-state-driven runtime LoRA tensor
  writes and temporal write-bias updates, while discrete counterfactual
  intervention ranking remains explicit CPU policy rather than an optimizer path
- process-functional LoRAs extend that family stack with a bounded
  process/process-step bank keyed by stable semantic execution signatures rather
  than transient plan IDs or tick IDs
- each process entry follows the same runtime pattern as shipped functional
  families: a learned adapter that begins as a no-op plus a tiny bootstrap
  adapter whose perturbation decays toward a small floor
- process-generated functional entries now share the same lifecycle substrate as
  shipped functional families: weekly snapshot capture, archived replay,
  orthogonal/local replay overrides, and differential replay-to-live updates
- only the currently matching process entry is attached, so the serving stack
  grows by two runtime layers at most even if the stored bank contains many
  historical process specializations
- DMN counterfactual simulation scores those process-generated entries through
  the same replay surface, so process-local, process-history, and
  process-orthogonal proposals participate in the same functional-bias ladder

The current request/response cycle is:

1. Request code sets or clears request-managed adapters without overwriting runtime memory layers.
2. The context rebuilds one effective serving stack from request layers plus runtime memory layers and bumps a stack version so decode graphs cannot silently reuse a stale adapter composition.
3. Prompt processing and token generation run through that full effective stack, and KV cache entries are populated normally under that stack.
4. When the slot reaches context-shift pressure, the runtime identifies the evicted span and the retained suffix from the slot prompt buffer.
5. The evicted span is ingested into Active LoRA before generation continues.
6. If Active ingestion produced a real weight change and a retained suffix still exists, the runtime treats that write as a KV-coherence boundary:
   - the stale retained KV state is invalidated,
   - prompt checkpoints captured under the old stack are discarded,
   - the retained suffix is replayed through the updated serving stack,
   - and generation resumes only after replay completes.
7. If Active ingestion was redundant or no retained suffix remains, replay is skipped explicitly and the slot keeps the compacted prompt/KV state.

This is the strict policy for KV coherence. Future tokens should not be sampled from KV that was computed under a pre-update Active LoRA when the serving stack has already changed.

### Split Cognitive Loop

The runtime is split into:

```text
external-input loop
||
default-mode-inspired background loop
```

Both loops share state, but they do not share identical triggers or output policy.

- The active loop is externally triggered and optimized for responsiveness, user relevance, and coherent action.
- The DMN loop is pressure-driven and optimized for reactivation, contradiction surfacing, consolidation, endogenous thought, and strategic follow-up.
- The runtime now exposes a shared bounded tool-loop substrate under both paths:
  explicit phase, terminal reason, tool proposal, observation, and tool-registry
  metadata are public trace surfaces.
- This is intentionally not "generic ReAct everywhere." The active loop and DMN
  use different policies on top of the same bounded scaffold.
- Both loops now also assign explicit functional microphases and route the
  functional LoRA bank through the same public activation substrate.

The shared functional microphase vocabulary currently includes:

- `STATE_INTERPRET`
- `TOOL_CLASS_SELECTION`
- `TOOL_ARGUMENT_PREP`
- `TOOL_RESULT_INTEGRATION`
- `PLAN_DRAFT`
- `PLAN_COMPOSE`
- `PLAN_REVISE`
- `COUNTERFACTUAL_GENERATE`
- `COUNTERFACTUAL_COMPARE`
- `MEMORY_COMPRESSION`
- `MEMORY_AUDIT`
- `SELF_OBSERVE`
- `SELF_FORECAST`
- `POST_ACTION_REFLECTION`

The intended bias families are:

- `functional_tool_selection`: active and DMN tool-class bias under uncertainty
- `functional_planning_composition`: bounded plan drafting, revision, and
  composition bias shared by active and DMN
- `functional_counterfactual`: comparative simulation and alternative ranking
- `functional_memory_compression`: preservation bias for salient evicted spans
- `functional_self_observation`: "what changed in me" interpretation and forecast

Family updates are outcome-weighted rather than generic:

- tool-selection opens on tool commitment and settles only after tool output has
  actually been integrated into the active loop
- planning/composition opens during plan draft or revision and settles against
  the quality of the composed plan as judged by later favorable-state,
  answerability, efficiency, and recovery deltas
- counterfactual settles against favorable-state and efficiency deltas after DMN
  comparison and remediation choice
- memory-compression settles when compression pressure and later audit signals
  indicate whether salient structure was preserved usefully
- self-observation settles against interpretation and recovery deltas after
  reflection-oriented active or DMN phases

Functional gain control now follows a bounded loop:

1. observe the current self-state gradient and allostatic distance,
2. derive a bounded belief summary over incompletely modeled cares from
   forecast error, residual allostatic pressure, novelty, and memory residue,
3. predict per-family gains with the gating MLP from the explicit gradient plus
   the fixed belief-summary tail,
4. apply bounded Gaussian exploration and clip gains into `[0, 2]`,
5. if a family is invoked, apply its routed learned gain plus a separate
   bounded bootstrap perturbation that decays with family usage toward a
   nonzero floor,
6. execute the functional-biased loop step and let registers shift,
7. settle a bounded training tuple and update the gate with Adam.

This is intentionally not a replacement of the explicit self-state. The
register bank and typed self-model remain the observation layer. The belief
layer is a small residual controller for partial observability: it represents
the possibility that the runtime is missing something it should care about,
without letting opaque latent text into the gating path.

The old action/state switches are now subordinate to a shared plan surface:

- active and DMN both compose bounded plans with typed steps
- tool paths are represented explicitly as `INVOKE_TOOL -> OBSERVE_TOOL`
- internal-write paths can append emit or tool follow-up steps in the same plan
- runner status and traces expose `plan_id`, revision count, current step, and
  plan status so the host can inspect composition directly

Outside that gate, runtime LoRA tensor mutation and temporal write-bias control
also use Adam-backed updates because they mutate differentiable parameters from
self-state deltas. The counterfactual ladder does not: it still ranks discrete
interventions explicitly.

### Persistent Self-Core

Vicuña requires a mathematical self-state that survives changes in context and KV cache contents.

The self-core includes:

- register bank,
- identity embedding,
- active goal embeddings,
- social and relationship state,
- tool-state surfaces,
- time-state encodings,
- unresolved commitments,
- handles into memory clusters and frozen LoRAs.

This state is the continuity mechanism of the system. If it disappears when the context window slides, the architecture collapses back into prompt engineering.

Current implementation slice:

- a typed CPU-side self-state container now exists per `llama_context`,
- the initial public surface exposes predefined register definitions and metadata,
- raw system datetime plus derived local/cyclic time features are explicit,
- deterministic caller-supplied time points are supported for replay and testing,
- event anchors for user, tool, and emit events drive the first analytic updates for `r_time_phase`, `r_tool_salience`, and `r_channel_state`,
- explicit prewrite and postwrite feature builders now produce typed feature vectors from token events, decoder stats, and recency state,
- bounded analytic updates now drive the first register recomputation path,
- learned contradiction and uncertainty heads are exposed as optional callback-backed hooks behind explicit flags,
- and the self-state now also maintains a layered typed self-model with:
  - a small expanded fast control bank for `user_satisfaction_risk`, `goal_progress_pressure`, `loop_inefficiency`, `recovery_urgency`, `answerability`, and `preference_uncertainty`,
  - grouped profile families for goal progress, user outcome, epistemic condition, efficiency, recovery, strategy, and self-improvement readiness,
  - explicit `instant`, `short`, and `long` horizon slices,
  - bounded forecast / prediction-error traces for remaining steps, remaining inference cost, expected user-outcome change, and expected recovery change,
  - and a bounded self-model extension registry that sits above the authored core.

The authored profiles remain the stable prior of the system. Runtime extensions
are additive and typed:

- hard-memory query can counterfactually promote `MEMORY_CONTEXT` extensions
  when representing a retrieved memory inside self-model appears useful
- tool or host code can upsert `SCALAR_PARAM` extensions with optional desired
  state
- hard-memory-derived extensions affect gain-control context but do not
  participate in allostasis by default
- tool-authored scalar parameters may participate in allostasis when they
  explicitly declare a desired state

The functional gating MLP does not consume a raw variable-length extension set.
It consumes a fixed summary tail derived from the extension registry, preserving
bounded control and compatibility with the existing self-state-gradient-driven
gain path.

The same principle now applies to belief under incomplete self-modeling. The
runtime keeps a bounded belief summary and a very small number of latent
concern slots. These slots are generic residual buckets, not a second hidden
self-model. They update from prediction-error residue, novelty, and
memory-supported mismatch; only the fixed summary tail reaches the gate. If a
slot remains stable and predictive, it is surfaced as a promotion candidate for
the explicit self-model extension registry rather than becoming durable state
automatically.

## Hard Memory

Hard memory is now a typed persistent substrate rather than a single textual
archive lane. The runtime emits bounded primitives:

- `EVENT_FRAGMENT`
- `TRAJECTORY`
- `OUTCOME`
- `TOOL_OBSERVATION`
- `USER_MODEL`
- `SELF_MODEL_FRAGMENT`

This is the bridge between episodic residue, user-model residue, and durable
control-state residue. The core policy stays explicit and CPU-side:

- self-state postwrite archival emits event, user-model, and self-model
  primitives when the delta exceeds threshold
- active-loop execution emits trajectories, outcomes, and tool observations
- DMN execution emits trajectories, outcomes, and self-model fragments
- retrieval parses primitive metadata back into typed hits and a bounded
  retrieval summary

The retrieval summary is the critical integration surface. Instead of piping raw
retrieved text into the gain controller, the runtime projects retrieval into a
bounded vector of:

- similarity support,
- importance and confidence support,
- gain and allostatic support,
- and domain-specific support for goal progress, user outcome, epistemic state,
  efficiency, recovery, strategy, and self-improvement.

That summary cooperates with the functional LoRA stack in two ways:

- it feeds the gating MLP as part of the self-state-gradient-driven activation
  input
- it can promote selected retrieved memories into bounded self-model extensions,
  which then contribute to the self-model summary consumed by the same gate

This keeps hard memory, self-model state, and LoRA modulation coupled through
typed bounded surfaces rather than prompt text.

### Closed Self-Improvement Synthesis

The self-improvement architecture should be read as one bounded loop, not as a
set of adjacent features:

1. admitted primary feedback updates the typed self-state and may create or
   revise discovered self-model extensions,
2. self-state, extension, belief, and hard-memory residue are compressed into
   bounded summaries rather than fed forward as variable-width prompt text,
3. favorable-state error plus those bounded summaries drive functional gain
   routing,
4. the active or DMN loop executes under the selected functional family and, if
   applicable, the matching process-functional specialization,
5. post-action deltas settle family and process-functional updates, emit durable
   hard-memory residue, and update discovered-state support,
6. DMN counterfactual replay compares local, historical, and orthogonal
   functional hypotheses, including process-functional entries on the same
   lifecycle substrate,
7. validated progress and postwrite support can consolidate discovered state
   into permanent or allostatic self-model structure,
8. runtime snapshots persist the resulting self-state plus functional/process
   archives so the loop survives restart.

The invariants that keep this loop synthesized are:

- no raw variable-length self-model or hard-memory structure reaches the gating
  controller directly,
- process-functional adapters are not a second adaptation regime; they use the
  same snapshot, replay, and differential-update substrate as shipped
  functional families,
- DMN functional counterfactuals improve the same live bias layers that serve
  execution,
- and discovered/permanent/allostatic self-state feeds future routing through
  bounded explicit summaries instead of latent prompt residue.

### Unified Provenance

Release evaluation of semi-biological growth cannot depend on ad hoc debug
strings or last-trace inspection alone. The server now maintains one
append-only structured provenance repository for self-improvement events.

That repository is aligned with the implemented loop boundaries:

- active-loop episodes
- tool-result integrations
- admitted DMN ticks with counterfactual/governance summaries

Each event carries stable session and sequence identity plus bounded summaries
of:

- self-model extension counts and allostatic pressure
- belief/promotion readiness
- functional and process-functional update surfaces
- counterfactual choice and governance outcome

This keeps the observability model unified:

- one local source of truth for longitudinal analysis
- one online metrics/health surface derived from the same data
- and one schema that can later be exported into broader tracing/evaluation
  infrastructure without redesigning the runtime

### Extensibility

The runtime must remain open to:

- new registers,
- new tool states,
- new modalities,
- new evaluators,
- new self-improvement policies,
- new memory compression methods.

Extensibility is a first-order requirement, not an afterthought.

## Register Bank

The register bank is a persistent typed state object that conditions future thought and action.

Initial register families:

- Epistemic pressure: `r_uncertainty`, `r_contradiction`, `r_novelty`, `r_topic_shift`
- Goal and self: `r_goal_relevance`, `r_self_relevance`, `r_social_relevance`, `r_affordance`
- Broadcast control: `r_broadcast_pressure`, `r_broadcast_inhibition`, `r_followup_continuation`
- Memory control: `r_memory_write_priority`, `r_reactivation_priority[...]`
- Environment and system state: `r_time_phase`, `r_tool_salience`, `r_channel_state`
- Expanded routing summaries: `r_user_satisfaction_risk`, `r_goal_progress_pressure`, `r_loop_inefficiency`, `r_recovery_urgency`, `r_answerability`, `r_preference_uncertainty`

Registers are updated from embeddings, retrieval geometry, decoder statistics, learned probes, tool/environment deltas, and decayed prior state. They are not raw prompt text.

The register bank is no longer the whole self-model. It is now the fast routing
surface layered on top of richer typed profile state.

## Update Order

Register recomputation follows a two-stage process.

### Pre-Write Update

Before a new event is written into working memory, the system should:

- compare the event against prior state,
- estimate surprise and conflict,
- decide whether the event deserves attention and memory,
- compute provisional deltas.

### Working-Memory Write

Admitted events are written into active working memory with:

- raw span,
- speaker or source tag,
- summary vector,
- salience markers,
- unresolved-question flags,
- tool-affordance hints.

### Post-Write Reconciliation

After the write, the system should:

- recompute action pressure,
- update broadcast policy,
- refresh replay priorities,
- decide whether follow-up action is warranted.

The ordering rule is:

```text
interpret against old state
-> decide admission
-> write working memory
-> recompute action and replay policy from the new state
```

## Feature Extraction

The intended event-processing path is:

```text
raw event
-> embedding
-> retrieval of relevant memory / self / goal / tool vectors
-> feature builder
-> register update heads
```

Embeddings are well-suited for novelty, topic shift, self relevance, goal relevance, retrieval, and memory cluster activation. They are not sufficient on their own for contradiction detection, calibrated uncertainty, tool readiness, or broadcast policy.

## Active Engagement Loop

The active loop handles user messages and tool returns.

It should:

- parse and absorb external input,
- preserve local coherence,
- maintain trust and responsiveness,
- decide whether to answer, ask, act, or wait,
- feed durable consequences into the memory cascade.

Authoritative call graph:

```text
on_user_message(msg)
  -> encode_event(msg)
  -> embed_event(msg)
  -> retrieve_relevant_state(...)
  -> build_prewrite_features(...)
  -> update_prewrite_registers(...)
  -> gate_working_memory_admission(...)
  -> write_working_memory(...)
  -> summarize_working_memory_delta(...)
  -> update_active_lora_from_evicted_or_booted_text(...)
  -> build_postwrite_features(...)
  -> update_postwrite_registers(...)
  -> refresh_memory_reactivation_priorities(...)
  -> assemble_control_state(...)
  -> generate_silent_candidates(...)
  -> score_candidates(...)
  -> decide_action(...)
      -> maybe_emit_user_message(...)
      -> maybe_invoke_tool(...)
      -> maybe_schedule_followup(...)
      -> maybe_remain_silent(...)
  -> log_episode(...)
```

Tool returns follow the same pre-write, write, and post-write pattern and may justify follow-up output without a new user prompt.

Current implementation slice:

- the core runtime now exposes a typed active-loop trace with scored `answer`,
  `ask`, `act`, and `wait` candidates,
- foreground episodes run through shared self-state and memory surfaces instead
  of bypassing them in host code,
- the foreground path is now backed by a bounded planner-executor runner with a
  persistent episode record, explicit max-step budget, and a host-visible
  pending-command queue for `emit_answer`, `emit_ask`, and `invoke_tool`,
- tool completions resume the same foreground episode rather than starting a
  fresh one, while host completion of a tool command leaves the runner waiting
  for the actual tool observation,
- and `llama-server` classifies foreground user versus tool episodes before
  templating and records real outward emission back into the loop trace.

## DMN Background Loop

The DMN is a default-mode-inspired background process. It is not literal biological terminology.

Its purpose is to:

- reactivate important memory clusters,
- surface unresolved contradictions,
- keep goals alive during idle periods,
- generate candidate thoughts from internal state,
- self-prompt tool use when appropriate,
- self-prompt user interaction when something important crosses threshold,
- consolidate memory.

The DMN should not fire because "time passed." It should fire because pressure exists, such as contradiction, uncertainty, reactivation priority, tool deltas, unfinished goals, counterfactual opportunity, or follow-up pressure.

Authoritative call graph:

```text
dmn_tick()
  -> update_time_phase(...)
  -> update_channel_and_tool_state(...)
  -> select_reactivation_targets(...)
  -> reactivate_memory_vectors(...)
  -> assemble_endogenous_seed(...)
  -> generate_silent_candidates(...)
  -> score_candidates(...)
  -> update_candidate_derived_registers(...)
  -> decide_dmn_action(...)
      -> maybe_write_internal_thought_to_working_memory(...)
      -> maybe_emit_user_message(...)
      -> maybe_invoke_tool(...)
      -> maybe_stay_silent(...)
  -> maybe_compress_working_memory_to_active_lora(...)
  -> maybe_roll_active_lora_into_past_stack(...)
  -> refresh_reactivation_priorities(...)
  -> schedule_next_tick(...)
```

DMN invariants:

- It does not require a new user message to think.
- It may stay silent.
- It may invoke tools.
- It may emit multiple user-facing messages in sequence if continuation remains high and inhibition remains below threshold.
- It must respect channel state and anti-spam controls.

Current implementation slice:

- the core runtime now computes a typed DMN pressure vector from shared
  registers, reactivation priorities, tool state, continuation pressure, and a
  dedicated repair-pressure term derived from dissatisfaction, recent negative
  user valence, trust/reciprocity deficits, failure signals, and social
  relevance,
- DMN admission is pressure-gated even when the host polls from an idle loop,
- the first slice exposes typed `silent`, `internal_write`, `invoke_tool`, and
  `emit` decisions plus burst-count and maintenance markers,
- each admitted or deferred tick now snapshots a typed favorable-self-state
  profile with per-dimension targets, tolerances, weights, and divergence
  ordering for contradiction, uncertainty, tool readiness/backlog, reactivation,
  continuation, broadcast pressure/inhibition, and social trust / reciprocity /
  dissatisfaction,
- the runtime now exposes Supermemory as a typed hard-memory second rail with
  explicit endpoint/auth/container/runtime config, bounded query results, and
  inspectable last-query / last-archive traces,
- primary-channel events that produce above-threshold self-state deltas can now
  be archived into hard memory with message context plus a bounded top-register
  delta summary, while counterfactual-channel events remain excluded by default,
- counterfactual search now follows a hypothesis-first low-risk ladder centered
  on runtime-memory LoRA ablation plus functional-bias replay candidates
  (local, historical, orthogonal), with tool- or retrieval-oriented discrete
  variants kept only as bounded fallbacks before higher-risk sensitivity or
  updater-policy proposals,
- LoRA ablation is treated as a first-class diagnostic over the serving runtime
  memory stack in recency order (`active`, `past_week`, `past_month`,
  `past_quarter`, `past_year`, `all_time`),
- bounded remediation currently targets Active LoRA only, with tool-oriented
  counterfactuals yielding `gather_info` plans that can now be typed as
  first-class bash CLI work or hard-memory query work, with bash requests and
  results carried through explicit bounded structs and executed only in
  `llama-server`, and high-risk updater-policy proposals denied or deferred,
- repair urgency now contributes to the same DMN admission gate through
  updater-program policy (`repair_admission_floor` and
  `repair_admission_weight`) instead of bypassing the scheduler,
- repair emission still uses an explicit second-stage evidence threshold from
  the updater program (`repair_emit_threshold` plus floors / inhibition
  ceiling), so the agent can reason about and propose threshold changes without
  silently applying them,
- governance traces now record proposal family, risk tier, evidence, user
  dissatisfaction, recent negative valence, and optional repair messaging,
- DMN ticks now defer explicitly while foreground runner work is still
  outstanding, which keeps the background process aligned with idle-mode
  semantics instead of competing with a live active episode,
- admitted DMN runs now execute as bounded planner-executors rather than
  single-step routers: `internal_write` may continue locally within budget and
  then yield a tool or background emit command through the same public queue,
- and `llama-server` polls the DMN only from idle states while recording
  foreground deferral when user-facing work is still active.

## Counterfactual Self-Improvement

The implemented DMN counterfactual path is:

```text
reactivation / pressure spike
-> favorable-self-state divergence ranking
-> low-risk counterfactual ladder
-> winner selection with risk-aware tie-breaking
-> bounded remediation plan
-> governance gate
-> Active LoRA update, gather-info tool job, defer, deny, or repair emit
```

The normal DMN counterfactual budget is now spent on temporal and functional
hypothesis testing first:

```text
current self-state pressure
-> temporal runtime-memory ablations
-> local functional perturbation
-> historical weekly functional replay
-> orthogonal attractor-escape replay
-> optional bounded tool/retrieval fallback candidates
```

This keeps DMN compute focused on bias and memory hypotheses rather than asking
the generic question "what if the input were different?" Tool-oriented fallbacks
still exist when tool pressure or retrieval pressure makes them worthwhile.

Functional-bias counterfactual search now runs in a separate control lane from
temporal-memory replay:

- DMN samples functional candidates from three families: local perturbation,
  weekly archived historical replay, and orthogonal attractor-escape replay
- archived functional snapshots are stored once per week, capped at four slots
  per family, and expired after roughly 31 days
- historical and orthogonal functional replay substitute the live functional
  family during evaluation; they do not stack with the current functional LoRA
  the way temporal memory layers stack
- after scoring, the best functional replay candidate produces a signed
  differential update against the live functional LoRA by applying the LoRA
  tensor difference scaled by counterfactual advantage and robustness
- orthogonal replay explicitly projects perturbation directions away from the
  recent dominant functional update direction so the DMN can challenge
  entrenched attractors

Useful meta-registers and governance inputs include:

- `r_curiosity`
- `r_performance_frustration`
- `r_resource_cost`
- `r_repeat_failure`
- `r_expected_gain`
- `r_self_update_risk`

Safe initial improvement surfaces:

- temporal-memory ablation and replay policy,
- functional-bias gain and differential-update policy,
- tool-argument and tool-choice heuristics when tool pressure is high,
- bounded scoring coefficients,
- replay ordering and reactivation refresh policy,
- auxiliary learned probes.

Unsafe early mutation targets:

- base model weights,
- frozen past LoRAs,
- self-state updater formulas, repair-admission / repair-emission policy, and
  consolidation policy without overwhelming counterfactual evidence,
- global safety policy,
- identity-critical priors.

## Implementation Boundaries

Vicuña should start as a fork of `llama.cpp`, but with a strict separation of concerns.

### GPU / Backend Responsibilities

Dense, repeated, latency-sensitive tensor operations should live in GPU or backend kernels:

- LoRA application at inference time,
- active LoRA updates,
- projection or compression from window residue into LoRA form,
- weighted composition of past LoRA influence,
- lightweight high-frequency probe heads.

### CPU / Control Responsibilities

Control logic and typed state management should remain in standard C++:

- register bank maintenance,
- feature assembly,
- working-memory admission,
- Active LoRA budget planning,
- embedding-strategy selection,
- rollover readiness tracking,
- DMN scheduling,
- broadcast and inhibition policy,
- tool orchestration,
- bash CLI request or result policy,
- reactivation scheduling,
- self-improvement proposal management,
- tracing and governance.

The practical rule is:

```text
GPU = heavy tensor math
CPU = policy, state, scheduling, orchestration, governance
```

The `llama.cpp` fork therefore needs hooks for:

- multi-LoRA application,
- runtime weighting of past LoRAs,
- external control-state injection,
- working-memory event callbacks,
- asynchronous side loops for DMN.

## Sanity Checks

Architectural changes should preserve these checks:

- DMN is described as default-mode-inspired, not as a literal biological network.
- The LoRA stack is lossy bias memory, not a perfect factual database.
- Tool state, time state, and inhibition policy remain explicit typed state.
- Pre-write epistemic evaluation happens before memory admission.
- Post-write action policy is recomputed after the write.
- The self-core persists across context eviction.
- Self-improvement begins with configurable policy surfaces, not arbitrary weight mutation.
- Past LoRAs remain frozen after rollover.
- Multi-message DMN output requires explicit continuation and inhibition logic.

## Immediate Roadmap

1. Memory and state skeleton
   Build sliding-window hooks, active LoRA update path, frozen past LoRA stack, register bank, working-memory event model, and tool/time surfaces.
2. Dual-loop runtime
   Build the active engagement loop, DMN scheduler, candidate generation API, candidate scoring, and continuation control.
3. Counterfactual DMN
   Build counterfactual candidates, meta-registers, self-update proposal objects, and governance rules.
4. Optimization and introspection
   Build GPU kernels for LoRA update/composition, register tracing, DMN observability, and audit trails for self-updates.

## Final Statement

Vicuña should be built as an organism-level architecture:

```text
memory cascade
+ persistent self-core
+ active engagement loop
+ DMN background loop
+ counterfactual self-improvement path
+ clean CPU/GPU separation
```

If implemented cleanly, the system can maintain continuity, preserve unresolved tensions, reactivate what matters, decide when to speak, and eventually improve the way it does all of that.
