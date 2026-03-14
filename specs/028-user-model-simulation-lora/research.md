# Research: User Model Expansion and Counterfactual User Simulation

## Current Local State

### Hard Memory User Model

`src/llama-self-state.cpp` currently archives `USER_MODEL` primitives only when
user signal is high. Those fragments mostly contain:

- trust
- dissatisfaction
- reciprocity
- bond strength
- recent user valence

This is useful relationship residue, but it is not a durable user model. It
does not encode formatting habits, directness, rhetorical structure, autonomy
preferences, or clarification behavior.

### Self-State User Model

`src/llama-self-state.cpp` maintains useful user-outcome summaries:

- `social_familiarity`
- `social_trust`
- `social_reciprocity`
- `social_recent_user_valence`
- `social_dissatisfaction`
- `preference_uncertainty`
- `cognitive_load_estimate`
- `autonomy_tolerance_estimate`
- `misunderstanding_risk`
- `trust_repair_need`

These are good control features but they remain outcome-oriented. They do not
form a reusable profile of how the user tends to communicate or what style they
prefer from the system.

### Counterfactual Path

The current counterfactual system is still primarily a bounded candidate ladder
plus trace replay:

- `compute_counterfactual_trace()` ranks candidate families
- `plan_remediation()` chooses a remediation path
- `evaluate_counterfactual()` replays self-state trace with an alternate
  updater program

There is no actual simulated user-response generation.

### Adapter and Serving Stack

The runtime already has the main primitives needed for this feature:

- explicit runtime adapter attach and detach
- inspectable serving-stack layering
- Adam-backed runtime LoRA training
- evicted-span ingestion
- functional family activation and ablation

This makes a dedicated user-personality LoRA a better fit than a new hidden
subsystem.

## External Research

### Letta

Letta keeps in-context memory as explicit typed blocks with labels,
descriptions, and metadata, rather than mixing every memory concern into raw
conversation history.

Relevant files:

- [letta/schemas/memory.py](https://github.com/letta-ai/letta/blob/main/letta/schemas/memory.py)
- [letta/schemas/block.py](https://github.com/letta-ai/letta/blob/main/letta/schemas/block.py)
- [letta/schemas/memory_repo.py](https://github.com/letta-ai/letta/blob/main/letta/schemas/memory_repo.py)

Takeaway: Vicuña should keep user memory typed and separated by role:

- durable user profile fragments in hard memory
- fast bounded user-preference summaries in self-state
- simulation substrate as a dedicated learned adapter

### MemGPT

[MemGPT](https://arxiv.org/abs/2310.08560) argues for an explicit hierarchy
between context and external memory rather than treating all relevant state as
one undifferentiated prompt.

Takeaway: user preference and rhetorical residue should not live only as prompt
text or thin traces. Vicuña should preserve a typed externalized user profile
surface and only summarize the most decision-relevant features into self-state.

### Generative Agents

[Generative Agents](https://arxiv.org/abs/2304.03442) emphasizes memory
retrieval plus higher-level reflection into stable abstractions rather than raw
event accumulation alone.

Takeaway: the user model should not be a dump of user messages. It should
promote repeated user behavior into stable typed summaries such as preference,
clarification style, autonomy tolerance, or formatting tendencies.

### Dialogue Persona and Style Modeling

Persona and style work in dialogue systems consistently separates:

- stable persona or preference features
- short-term dialogue state
- surface-style adaptation

Takeaway: a separate user-personality LoRA used only for simulation is more
coherent than pushing user rhetoric into the system’s own active or temporal
memory stack.

## Design Conclusions

### What High-Leverage User Signals Should Be Added

The strongest additional pieces are:

- directness preference
- verbosity preference
- structure preference
- clarification propensity
- autonomy preference
- disagreement sensitivity
- rhetorical intensity
- preference confidence
- rhetorical confidence

These are high leverage because they can change message planning, explain likely
user responses, and influence whether a candidate response reduces or increases
allostatic divergence.

### Hard Memory vs Self-State Split

Hard memory should preserve:

- stable user preference fragments
- rhetorical-style fragments
- objection or repair patterns
- interaction-policy observations
- user-profile confidence and provenance

Self-state should summarize:

- current best estimate of the above
- uncertainty for those estimates
- likely response favorability risk
- user-simulator readiness

### User Personality LoRA Policy

The user-personality adapter should:

- be a dedicated fixed-size runtime LoRA
- train only from evicted user-authored spans
- reuse the existing Adam-backed runtime update path
- never roll into the temporal bucket stack
- be attached only during DMN counterfactual simulated-user passes

### Counterfactual Serving Policy

During simulated user reply generation the runtime should:

1. save the current effective runtime stack
2. ablate all model-specific temporal LoRAs
3. ablate request-time temporal memory influence
4. attach only the user-personality LoRA on top of the base model
5. generate a bounded simulated user reply
6. restore the normal stack

This must be CPU-visible and typed. It should not depend on hidden backend
state.

### Why Not Put User Simulation Into Active Flow

The request explicitly wants this as a DMN counterfactual mode rather than a
third loop or a default active behavior. That is also the correct boundedness
decision. Simulation is costly and speculative, so it should be invoked only in
counterfactual evaluation where the system is explicitly comparing futures.
