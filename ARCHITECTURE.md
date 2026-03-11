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
- the span-embedding strategy remains swappable, including callback-backed model-specific embedders, so incompatible model families do not force one embedding space on the whole runtime,
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
```

This ordering is a control-policy ordering, not a serial neural pipeline. The graph still applies LoRA deltas additively on top of base weights, but the runtime now keeps the effective stack deterministic and inspectable.

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
- and learned contradiction and uncertainty heads are exposed as optional callback-backed hooks behind explicit flags.

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

Registers are updated from embeddings, retrieval geometry, decoder statistics, learned probes, tool/environment deltas, and decayed prior state. They are not raw prompt text.

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
- counterfactual search now follows a low-risk-first ladder across message
  variants, tool-argument changes, hard-memory query variants, tool-choice
  changes, timing shifts, and runtime-memory LoRA ablation before it considers
  higher-risk sensitivity or updater-policy proposals,
- LoRA ablation is treated as a first-class diagnostic over the serving runtime
  memory stack in recency order (`active`, `past_week`, `past_month`,
  `past_quarter`, `past_year`, `all_time`),
- bounded remediation currently targets Active LoRA only, with tool-oriented
  counterfactuals yielding `gather_info` plans that can now be typed as generic
  tool work or hard-memory query work, and high-risk updater-policy proposals
  denied or deferred,
- repair urgency now contributes to the same DMN admission gate through
  updater-program policy (`repair_admission_floor` and
  `repair_admission_weight`) instead of bypassing the scheduler,
- repair emission still uses an explicit second-stage evidence threshold from
  the updater program (`repair_emit_threshold` plus floors / inhibition
  ceiling), so the agent can reason about and propose threshold changes without
  silently applying them,
- governance traces now record proposal family, risk tier, evidence, user
  dissatisfaction, recent negative valence, and optional repair messaging,
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

Useful meta-registers and governance inputs include:

- `r_curiosity`
- `r_performance_frustration`
- `r_resource_cost`
- `r_repeat_failure`
- `r_expected_gain`
- `r_self_update_risk`

Safe initial improvement surfaces:

- message-shaping biases captured in Active LoRA,
- tool-argument and tool-choice heuristics,
- timing / continuation policy,
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
