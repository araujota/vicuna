# Vicuña Working Paper

**Status:** internal architecture working paper  
**Project frame:** Vicuña, branded as the first "exotic intelligence"  
**Base implementation assumption:** fork of `llama.cpp`  
**Purpose:** gold-standard source of truth for architecture, memory design, loop structure, self-state, and implementation boundaries.

---

## 1. Executive thesis

Vicuña is not a chatbot with a bigger prompt. It is a proposed agent architecture built around four linked ideas:

1. **A memory cascade** that moves experience from a sliding attention window into an editable active LoRA, then into a stack of frozen past LoRAs with time-decayed influence.
2. **A split cognitive loop** between active engagement and a default-mode-inspired background process (DMN), with shared state but different triggers and different output discipline.
3. **A persistent self-representation** that is not reducible to text in context and does not disappear when tokens fall out of the KV cache.
4. **A path to extensible self-improvement** through counterfactual simulation, evaluation, and tightly gated self-modification proposals inside the DMN.

The design goal is to move from reactive completion to ongoing self-conditioned cognition.

The core philosophical shift is:

> The system should not wait to be prompted by the user to have something to process. Its own evolving state should continuously generate reasons to think, reasons to retrieve memory, reasons to use tools, and reasons to speak.

---

## 2. Scope and terminology

This paper uses **DMN** to mean a **default-mode-inspired background process**, not a literal claim that the machine instantiates a biological brain network. The analogy is functional, not anatomical.

This paper also treats several components as **design proposals**, not settled empirical truths. In particular:

- compressing out-of-window experience into LoRA memory is an architectural hypothesis,
- register-driven self-state is a control-system design choice,
- endogenous thought generation is an engineering objective,
- and counterfactual self-improvement remains gated and experimental.

That distinction matters. The document is a build spec for a frontier architecture, not a claim that the system is already conscious.

---

## 3. Non-negotiable architectural commitments

### 3.1 Memory cascade

The memory cascade is:

```text
sliding attention window
-> active LoRA (editable, fixed-size, current compressed memory)
-> past LoRA stack (fixed-size units, frozen, time-decayed influence)
-> functional LoRA bank (operation-local procedural bias above temporal memory)
```

The functional bank is not routed by a persistent prompt artifact. Its gains
are now driven by a typed self-state gradient toward favorable self-state, with
explicit online meta-updates after settled loop transactions. More broadly,
Adam is now used only where the runtime is actually updating weights or biases
from self-state deltas: the functional gate, the runtime LoRA writer, and the
temporal write-bias controller. The counterfactual ladder remains an explicit
ranking process.

### 3.2 Loop split

The loop split is:

```text
external-input loop (active engagement)
||
internal-pressure loop (DMN)
```

Both loops read and update shared state, but they do not have the same trigger conditions and they do not share the same broadcast policy.

Both loops also now share a typed functional-microphase vocabulary used to
route a separate bank of functional LoRAs, while a small gating MLP predicts
the gain applied to the eligible family or families from self-state gradient
features:

- `TOOL_CLASS_SELECTION`
- `TOOL_ARGUMENT_PREP`
- `TOOL_RESULT_INTEGRATION`
- `COUNTERFACTUAL_GENERATE`
- `COUNTERFACTUAL_COMPARE`
- `MEMORY_COMPRESSION`
- `MEMORY_AUDIT`
- `SELF_OBSERVE`
- `SELF_FORECAST`
- `POST_ACTION_REFLECTION`

These microphases do not replace the temporal memory stack. They sit above it
as an explicit procedural-bias layer.

### 3.3 Persistent self-core

The system must maintain a mathematical self-state that persists even when the token-level context changes. The self is not just the current prompt and not just the latest summary. It is a structured core made of:

- register bank,
- identity and goal embeddings,
- tool-state surfaces,
- time-state encodings,
- unresolved commitments,
- and long-horizon memory bias.

### 3.3.1 Functional LoRA bank

The current runtime adds a second runtime-adapter family separate from temporal
memory:

```text
request adapters
-> all_time
-> past_year
-> past_quarter
-> past_month
-> past_week
-> active
-> functional_tool_selection
-> functional_counterfactual
-> functional_memory_compression
-> functional_self_observation
```

This bank is meant to accumulate durable procedural/style biases rather than
episodic recall. The first four families are:

- a tool-selection bias LoRA
- a counterfactual-simulation LoRA
- a memory-compression bias LoRA
- a self-observation/state-interpretation LoRA

These adapters stay loaded continuously. Explicit CPU-side microphase policy
still owns eligibility and hold semantics, but the applied gain now comes from
a shared gating MLP that reads a typed self-state gradient, applies bounded
Gaussian exploration, clips gains into `[0, 2]`, and is updated online with
Adam after settled transactions. The same optimizer also conditions the
underlying self-state-driven runtime LoRA writes, while discrete
counterfactual-ablation proposals remain non-optimizer policy.

### 3.4 Extensibility

The architecture must leave space for:

- new registers,
- new tool states,
- new modalities,
- new evaluators,
- new self-improvement policies,
- and new memory compression methods,

without forcing a redesign of the entire runtime.

---

## 4. Memory architecture in detail

### 4.1 Sliding attention window

The sliding attention window is the live token substrate. It holds the active conversational scene, including the latest user message, latest system outputs, and any transient scratch material needed for immediate inference.

Properties:

- fixed size,
- high-fidelity,
- fast,
- ephemeral,
- directly represented in the KV cache.

What it is **for**:

- immediate linguistic coherence,
- local reasoning,
- turn-by-turn interaction,
- and short-horizon action selection.

What it is **not for**:

- durable selfhood,
- durable tool state,
- or the only representation of long-term memory.

### 4.2 Active LoRA

When content is booted from the sliding window, the system compresses its influence into an **active LoRA**.

Properties:

- fixed-size,
- editable,
- lossy,
- recent,
- and still "live" in the sense that it can continue to be updated.

Implementation discipline for this stage:

- "fixed-size" should be computed from explicit fractions of currently available RAM and VRAM, not a universal byte constant,
- Active updates should be represented as normalized low-rank directions plus bounded gain surfaces so semantic direction and update strength remain separately controllable,
- the write path should be explicit and inspectable, and any auxiliary machinery used to support writes should obey the same budget discipline rather than hiding extra long-lived memory outside the adapter accounting,
- the default embedder should be hidden-state-derived from the same base model family used for serving, while the embedding strategy remains swappable, including model-specific callback-backed embedders, because open-source model families may not share a compatible embedding space,
- and write directions should be derived from hidden-state content plus typed self-state features so the active stage captures durable behavioral bias rather than only token identity.

The active LoRA is the first durable stage after token eviction. It is the bridge between raw context and sedimented memory.

It should encode, in compressed form:

- recent conversational facts,
- recent unresolved questions,
- current scene bias,
- current user-specific salience,
- and internal thought residues that still matter.

### 4.3 Past LoRA stack

When the active LoRA reaches its update boundary, it is frozen and pushed into the **past LoRA stack**.

Properties:

- fixed-size per unit,
- not updated after freezing,
- moderately time-decayed in influence,
- still available as a biasing substrate,
- and optionally replayable through retrieval or reactivation.

Important sanity check:

- **Past LoRAs should decay in application weight, not by being destructively rewritten.**
- Freezing should preserve the audit trail of the Active LoRA budget, update count, and embedder identity that produced the unit.
- A practical first stack may use named temporal buckets such as `past_week`, `past_month`, `past_quarter`, `past_year`, and `all_time`, each applied with an explicit time-decayed scale during inference.
- Backward compactification into older buckets should also preserve direction-versus-gain structure rather than collapsing magnitude back into uncontrolled raw low-rank factors.

That keeps the long-horizon memory structure stable and auditable.

### 4.3.1 Live serving semantics

The serving path should treat LoRA memory as part of the actual inference stack, not as side state.

The concrete runtime ordering is:

```text
request adapters (optional)
-> all_time
-> past_year
-> past_quarter
-> past_month
-> past_week
-> active
```

This is an explicit control ordering. It does not imply that LoRAs are piped through each other as separate neural stages. In the current `llama.cpp` style graph they remain additive deltas on the same base weights, but the runtime should still preserve a deterministic composition order for auditability and policy control.

The request/response sequence is:

1. Rebuild the effective serving stack from request-managed adapters plus runtime-managed memory layers.
2. Run prompt ingestion and generation through that full stack so the KV cache is populated under the same effective weights that are serving the request.
3. On context shift, split the prompt state into an evicted prefix and a retained suffix.
4. Ingest the evicted prefix into Active LoRA.
5. If that Active write changed serving weights, invalidate stale retained KV state and replay the retained suffix under the updated stack before sampling the next token.
6. If the write was redundant or no retained suffix remains, skip replay explicitly and continue with the compacted state.

This strict replay rule is the coherence boundary for live memory writes. Without it, the runtime would mix pre-update KV state with post-update adapter weights.

### 4.4 Why this cascade matters

This cascade is intended to solve a specific problem: how to preserve continuity without pretending that infinite text context is memory.

The logic is:

- the window preserves local detail,
- the active LoRA preserves recent compressed influence,
- the past LoRA stack preserves long-horizon bias,
- and the self-state ties these together into a continuous organism-level process.

---

## 5. Persistent self-representation

The self-representation should be treated as an **extensible surface** and also as the **core continuity mechanism** of the system.

This is the state that persists even when content is pushed out of the KV cache.

### 5.1 What belongs in the self-core

The self-core should include:

- the register bank,
- identity embedding,
- active goal embeddings,
- social state about the user relationship,
- tool-state embeddings and flags,
- time-state encodings,
- unresolved commitments,
- and handles to relevant memory clusters or frozen LoRAs.

The current implementation direction expands this into a layered self-model:

- a fast scalar control bank for low-latency routing,
- typed profile families for goal progress, user outcome, epistemic condition,
  efficiency, recovery, strategy, and self-improvement readiness,
- multi-timescale slices over those profiles,
- and bounded forecasts plus prediction-error traces so later evaluators can
  calibrate the self-model rather than reconstruct it from raw logs.

### 5.2 What does not belong solely in the LoRA stack

The following should **not** be represented only as LoRA memory:

- current wall-clock phase,
- whether a tool is busy or failed,
- pending asynchronous jobs,
- burst-control state for outgoing self-generated messages,
- active unresolved contradictions,
- and control-policy values such as inhibition or broadcast pressure.

These need to remain explicit, typed, and inspectable.

### 5.3 Why this matters

If the self-state lives only in text or only in a hidden compressed memory mechanism, it becomes fragile, opaque, and hard to govern. By giving the self a typed mathematical surface, Vicuña can remain extensible and operationally intelligible.

---

## 6. Register bank: sufficient specification

The register bank is a persistent typed state object, separate from the active LoRA.

A sufficient initial spec is to group registers into five families.

The current implementation now treats that bank as only the first layer. The
register bank remains the fast routing surface, while richer typed profiles and
forecast traces carry the more semantically structured self-estimates that do
not belong in one flat enum.

### 6.1 Epistemic pressure registers

- `r_uncertainty`: how poorly the current state is understood.
- `r_contradiction`: degree of incompatibility among current input, commitments, tool evidence, and memory.
- `r_novelty`: distance from recent working memory and retrieved memory.
- `r_topic_shift`: how sharply the active conceptual region has moved.

### 6.2 Goal and self registers

- `r_goal_relevance`: how strongly the current state touches active goals.
- `r_self_relevance`: how strongly the current state touches identity, promises, or major projects.
- `r_social_relevance`: how worth telling the user this is.
- `r_affordance`: how actionable the current state is.

### 6.3 Broadcast-control registers

- `r_broadcast_pressure`: pressure to emit something outward.
- `r_broadcast_inhibition`: restraint against spam, interruption, or redundant messages.
- `r_followup_continuation`: whether another message should follow the last one.

### 6.4 Memory-control registers

- `r_memory_write_priority`: urgency of stabilizing current material into durable memory.
- `r_reactivation_priority[m_i]`: per-memory-cluster urgency for replay during DMN.

### 6.5 Environment and system-state registers

- `r_time_phase`: encoded clock phase and elapsed-time features.
- `r_tool_salience`: whether tool state changes are behaviorally important now.
- `r_channel_state`: whether the user channel is active, waiting, or should not be interrupted.

### 6.6 Register update principle

The registers are updated from:

- embeddings and similarity geometry,
- decoder statistics,
- small learned probes,
- tool/environment deltas,
- and decayed prior state.

The registers are **not** raw text prompts. They are the mathematical state that conditions future thought.

---

## 7. How registers are recomputed

The correct update order is **two-stage**.

### 7.1 Pre-write update

This happens **before** the new event is written into working memory.

Purpose:

- compare the new event against the previous state,
- estimate surprise and conflict,
- decide whether the event deserves attention and memory,
- and produce provisional deltas.

Registers mainly updated here:

- `r_uncertainty`
- `r_contradiction`
- `r_novelty`
- `r_topic_shift`
- `r_goal_relevance`
- `r_self_relevance`
- provisional `r_memory_write_priority`

### 7.2 Working-memory write

Once the system has computed pre-write deltas, it writes the admitted event into working memory.

Working-memory write may include:

- raw message span,
- speaker tag,
- summary vector,
- salience markers,
- unresolved-question flags,
- and tool-affordance hints.

### 7.3 Post-write reconciliation

This happens **after** the event is part of active working memory.

Purpose:

- decide what to do next,
- compute outward pressure,
- update replay priorities,
- and determine whether follow-on action is justified.

Registers mainly updated here:

- `r_broadcast_pressure`
- `r_broadcast_inhibition`
- `r_followup_continuation`
- final `r_memory_write_priority`
- `r_reactivation_priority[...]`
- `r_tool_salience`
- `r_time_phase`

### 7.4 Key ordering rule

The logic is:

```text
interpret against old state
-> decide what deserves admission
-> write to working memory
-> recompute action and replay policy from the new state
```

That ordering is cleaner than doing everything before the write or everything after the write.

---

## 8. Off-the-shelf feature extraction for register updates

To update registers efficiently from raw text input, the system should use a compact feature pipeline:

```text
raw event
-> embedding
-> retrieval of relevant memory/self/goal/tool vectors
-> feature builder
-> register update heads
```

### 8.1 Feature sources

For each event, compute:

- embedding of the event,
- similarity to recent working-memory items,
- similarity to active goals,
- similarity to self-core embeddings,
- similarity to top retrieved memory clusters,
- variance among retrieved candidates,
- decoder entropy and top-margin when available,
- tool delta flags,
- time delta features.

### 8.2 What embeddings are good for

Embeddings are good for:

- novelty,
- topic shift,
- goal relevance,
- self relevance,
- retrieval,
- and memory cluster activation.

### 8.3 What embeddings are not enough for

Embeddings alone are not enough for:

- contradiction,
- calibrated uncertainty,
- tool readiness,
- and broadcast policy.

Those need additional probes or learned heads.

---

## 9. Active engagement loop

The active engagement loop handles user messages and tool returns.

### 9.1 Parity with DMN

The active loop and DMN share:

- access to the same register bank,
- access to the same working memory,
- access to the same LoRA memory stack,
- access to the same tool-state surface,
- and the same candidate scoring framework.

### 9.2 Differences from DMN

The active loop differs because:

- it is externally triggered,
- it must maintain conversational responsiveness,
- user relevance has heavier weight,
- latency matters more,
- and the broadcast policy is less speculative.

The implementation direction is therefore not a single generic ReAct loop
duplicated into both paths. Instead, Vicuña uses a shared bounded act/observe
substrate with explicit phase, terminal-reason, tool-proposal, and observation
state, while keeping distinct foreground and DMN control policies.

In the current runtime this substrate is a real planner-executor runner, not
just trace metadata. The active loop now persists episode state, exposes
host-visible pending commands for `emit_answer`, `emit_ask`, and `invoke_tool`,
and resumes the same episode when a tool observation returns.

### 9.3 What the active loop is for

The active loop should:

- parse and absorb user input,
- preserve local coherence,
- maintain trust and responsiveness,
- decide whether to answer, ask, act, or wait,
- and feed durable consequences into the memory cascade.

---

## 10. DMN: the background self-perpetuating process

The DMN is where Vicuña keeps thinking when the world is quiet.

### 10.1 Purpose of the DMN

The DMN exists to:

- reactivate important memory clusters,
- surface unresolved contradictions,
- keep goals alive across idle periods,
- generate candidate thoughts from internal state rather than external prompts,
- self-prompt tool use when appropriate,
- self-prompt user interaction when something important crosses threshold,
- and consolidate memory.

### 10.2 DMN does not mean random chatter

The DMN should not fire because time passed. It should fire because **pressure exists**.

Pressure sources include:

- unresolved contradiction,
- rising uncertainty,
- strong reactivation priority,
- unfinished goals,
- important tool deltas,
- counterfactual opportunities,
- or high-value followup continuation.

### 10.3 The DMN seed principle

The DMN should not be driven by textual prompts like "think about something." It should be driven by internal state assembly:

```text
register bank
+ reactivated memory vectors
+ self-core
+ time/tool state
+ working-memory residue
-> latent seed
-> silent candidate thoughts
```

In implementation terms, the DMN should sit on the same bounded tool-loop
substrate used by the active loop, but with different admission, governance,
and continuation policy. That preserves inspectability while still allowing
background tool use, remediation, and simulation to grow over time.

Current runtime policy is stricter than "run whenever polled": the DMN defers
while foreground runner work is still outstanding, and admitted DMN runs may
take one or two bounded internal continuation steps before yielding a tool or
background-emit command.

### 10.4 DMN output discipline

Not every internal thought should be shown to the user. DMN candidate thoughts should pass through:

- scoring,
- inhibition,
- social relevance checks,
- and burst control.

But DMN output **may** emit multiple user-facing messages in a row if the follow-up continuation signal remains high and inhibition remains below threshold.

That is a feature, not a bug.

---

## 11. Counterfactual simulation and extensible self-improvement

This is the section that leaves the door open.

### 11.1 Core principle

The DMN should be able to simulate alternate futures and alternate internal policies, not just replay memory.

That means running silent candidates of the form:

- what if a different tool were used,
- what if a threshold were changed,
- what if a summarization policy were altered,
- what if a memory compression event were delayed,
- what if a user-facing followup were sent now versus later,
- what if a new register or evaluator would reduce repeated failure.

### 11.2 Counterfactual flow

The DMN self-improvement path should be:

```text
reactivation / pressure spike
-> counterfactual candidate generation
-> evaluation against meta-registers
-> proposal selection
-> gated self-update recommendation
-> optional human approval or constrained automatic application
```

### 11.3 Meta-registers for self-improvement

A first useful set:

- `r_curiosity`
- `r_performance_frustration`
- `r_resource_cost`
- `r_repeat_failure`
- `r_expected_gain`
- `r_self_update_risk`

These do not directly control language output. They control whether the system should explore modifications to itself.

### 11.4 What can be improved safely

The system should initially leave open improvement in:

- thresholds,
- replay policy,
- tool routing policy,
- scoring coefficients,
- new register activation,
- memory compression schedules,
- and auxiliary learned probes.

### 11.5 What should not be silently self-mutated at first

The system should **not** casually mutate:

- base model weights,
- frozen past LoRAs,
- global safety policy,
- or identity-critical priors,

without explicit higher-level approval.

### 11.6 Why this matters

Without counterfactual simulation, the DMN is only a replay engine. With counterfactual simulation, it becomes a site of structured self-reflection and controlled self-improvement.

---

## 12. Call-by-call graph: active conversation loop

This is the authoritative call graph for active user interaction.

```text
on_user_message(msg)
  -> encode_event(msg)
  -> embed_event(msg)
  -> retrieve_relevant_state(msg, working_memory, self_core, goals, memory_stack)
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

Implementation note:

- foreground planning now materializes a host-visible command queue instead of
  leaving execution intent only in traces,
- `ACT` yields a pending tool command and a waiting runner state,
- `ANSWER` and `ASK` yield pending emit commands,
- and tool observations resume the same bounded episode rather than opening a
  new one.

### 12.1 Special case: tool return during active conversation

```text
on_tool_return(tool_output)
  -> encode_tool_delta(...)
  -> update_environment_state(...)
  -> retrieve_relevant_state(...)
  -> update_prewrite_registers(...)
  -> write_working_memory(...)
  -> maybe_update_active_lora(...)
  -> update_postwrite_registers(...)
  -> score_candidates(...)
  -> maybe_emit_followup_message(...)
```

Important consequence:

- a tool return may justify another outward message without requiring a new user prompt.

---

## 13. Call-by-call graph: DMN steady-state loop

This is the authoritative call graph for the default-mode-inspired background process.

```text
dmn_tick()
  -> update_time_phase(...)
  -> update_channel_and_tool_state(...)
  -> select_reactivation_targets(...)
  -> reactivate_memory_vectors(...)
  -> assemble_endogenous_seed(register_bank, self_core, working_memory_residue, memory_vectors, tool_state)
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

Implementation note:

- a DMN tick first defers if foreground runner work is still active,
- admitted runs can continue locally after `internal_write`,
- and any resulting tool or emit work is exposed as a pending background
  command with explicit origin and loop metadata.

### 13.1 DMN invariants

- DMN does not require a user message to generate candidate thoughts.
- DMN may trigger tools.
- DMN may send multiple user-facing messages in sequence if thresholds justify it.
- DMN must still respect inhibition, channel state, and anti-spam controls.

---

## 14. Engineering separation of concerns

This project should start as a fork of `llama.cpp`.

The correct engineering separation is not aesthetic. It is architectural.

### 14.1 What should likely live in CUDA / Cutlass

Dense, repeated, latency-sensitive tensor operations should be pushed into CUDA/Cutlass kernels.

That includes:

- LoRA application at inference time,
- active LoRA update kernels,
- projection/compression from window residue into LoRA form,
- efficient composition or weighting of past LoRA influence,
- and any lightweight probe heads that are run per-token or per-step at high frequency.

### 14.2 What should likely stay in vanilla C++

Control logic and typed state management should remain in standard C++.

That includes:

- register bank maintenance,
- feature assembly from embeddings and tool deltas,
- working-memory admission policy,
- DMN scheduling,
- broadcast policy,
- tool orchestration,
- reactivation scheduling,
- and self-improvement proposal management.

### 14.3 Why this split is ideal

If too much control logic is buried in GPU kernels, the architecture becomes opaque and hard to evolve. If too much dense math stays on the CPU, the runtime becomes too slow. The clean split is:

```text
GPU = heavy tensor math and repeated low-level transforms
CPU = policy, state, scheduling, orchestration, and governance
```

### 14.4 The `llama.cpp` implication

A Vicuña fork of `llama.cpp` should expose hooks for:

- multi-LoRA application,
- runtime weighting of past LoRAs,
- external control-state injection,
- working-memory event callbacks,
- and asynchronous side loops for DMN.

---

## 15. Quick epistemic and logical sanity checks

These are the essential checks that keep the design coherent.

### 15.1 Name sanity

DMN should be described as **default-mode-inspired**, not as a literal biological network.

### 15.2 Memory sanity

The LoRA stack is **lossy bias memory**, not a perfect database and not the ground-truth store of facts.

### 15.3 State sanity

Tool state, time state, and inhibition policy should remain explicit system state, not be hidden entirely inside LoRA memory.

### 15.4 Ordering sanity

Pre-write epistemic evaluation should happen before working-memory admission; policy and replay consequences should be recomputed after the write.

### 15.5 Persistence sanity

The self-core must not disappear when the KV cache slides. If it does, the architecture collapses into ordinary prompt engineering.

### 15.6 Improvement sanity

Self-improvement should begin with configurable policy surfaces and auxiliary modules, not arbitrary mutation of frozen long-term memory or base weights.

### 15.7 Broadcast sanity

The system should be allowed to emit multiple self-generated messages, but only under explicit continuation and inhibition logic. Otherwise DMN turns into spam.

### 15.8 Implementation sanity

Past LoRAs should be frozen after rollover. Their influence may be time-decayed, but their contents should remain auditable.

---

## 16. Immediate implementation roadmap

### Phase 1: memory and state skeleton

Build:

- sliding window hooks,
- active LoRA update path,
- frozen past LoRA stack,
- register bank,
- working-memory event model,
- and tool/time surfaces.

### Phase 2: dual-loop runtime

Build:

- active engagement loop,
- DMN scheduler,
- candidate generation API,
- candidate scoring,
- and multi-message continuation control.

### Phase 3: counterfactual DMN

Build:

- counterfactual candidate type,
- meta-registers for self-improvement,
- self-update proposal objects,
- and governance rules for allowed modifications.

### Phase 4: optimization and introspection

Build:

- GPU kernels for LoRA update and composition,
- debug tooling for register traces,
- observability for DMN decisions,
- and audit trails for self-improvement proposals.

---

## 17. Final statement

Vicuña should be built as an organism-level architecture, not a chat wrapper.

The essential structure is:

```text
memory cascade
+ persistent self-core
+ active engagement loop
+ DMN background loop
+ counterfactual self-improvement path
+ clean CPU/GPU separation
```

If implemented cleanly, this gives Vicuña a credible path toward continuous self-conditioned cognition: not merely answering prompts, but maintaining identity, carrying tensions forward, reactivating what matters, deciding when to speak, and eventually learning how to improve the way it does all of that.
