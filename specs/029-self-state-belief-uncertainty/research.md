# Research: Partial-Observation Self-State Belief Gradient

## Question

Can Vicuña treat the self-state as an incomplete observation of what the agent may care about, then use a bounded belief state over missing or unmodeled cares to modify functional LoRA gain selection?

Short answer: yes, but only if the belief layer is implemented as a bounded, typed residual controller. It should augment the current self-state gradient, not replace it.

## Local Findings

### Vicuña Already Has The Right Separation To Support This

Relevant local files:

- `/Users/tyleraraujo/vicuna/ARCHITECTURE.md`
- `/Users/tyleraraujo/vicuna/src/llama-self-state.cpp`
- `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- `/Users/tyleraraujo/vicuna/include/llama.h`

Observed properties:

- The current design already distinguishes hard memory, explicit self-state, self-model extensions, and functional gain control.
- The gating MLP already consumes a fixed-width feature vector derived from the signed self-state gradient, loop metadata, uncertainty, and extension summaries.
- The system already exposes explicit allostatic targets, typed extension summaries, and meta-updated functional gains.
- The architecture already treats explicit self-state as a typed control surface rather than as a raw prompt artifact.

Implication:

- Vicuña does not need a wholesale shift to opaque latent memory.
- It can add a belief layer between observation and gain prediction while preserving the explicit self-state ontology.

### Current Uncertainty Is About Known Variables, Not Missing Variables

Observed properties:

- `r_uncertainty`, `r_preference_uncertainty`, and related features quantify uncertainty about currently modeled conditions.
- Current update paths do not represent the idea that there may be relevant cares that are not yet explicitly rendered.
- The gate therefore acts like a controller over a nearly fully observed state vector, even when the modeled state is clearly incomplete.

Implication:

- The missing concept is not "more uncertainty" in general.
- The missing concept is uncertainty over omitted state dimensions and unexplained post-action residue.

### Existing Self-Model Extension Infrastructure Gives A Promotion Path

Observed properties:

- Vicuña already has an authored self-model core plus typed, bounded extension registry.
- Tool-authored scalar parameters can be allostatic or non-allostatic.
- Hard-memory-derived context extensions are non-allostatic by default.

Implication:

- A belief layer can remain non-authoritative until repeated evidence justifies promotion into explicit self-model state.
- This preserves inspectability and backward compatibility for the existing gain path.

## GitHub Findings

### `letta-ai/letta` Keeps Memory Explicitly Structured Rather Than Hiding State In One Latent Blob

Source:

- `letta-ai/letta` [`letta/schemas/memory.py`](https://github.com/letta-ai/letta/blob/main/letta/schemas/memory.py)

Observed pattern:

- Letta models memory as structured blocks, file-backed memory, archival memory, and recall memory with explicit rendering and limits.
- The system prefers inspectable, typed, separately managed state over one opaque learned state container.

Implication for Vicuña:

- The uncertainty layer should not become a hidden replacement for self-state or hard memory.
- It should remain a small explicit state family alongside them.

### `langchain-ai/langgraph` Shows A Useful Pattern: Shared State With Typed Reducers

Source:

- `langchain-ai/langgraph` [`libs/langgraph/langgraph/graph/state.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)

Observed pattern:

- LangGraph models agent state as a typed shared state with reducer semantics and explicit update channels.
- State growth is controlled by typed schemas and reducers rather than free-form accumulation.

Implication for Vicuña:

- A belief summary should look like another typed state family with explicit update rules, not like unrestricted hidden scratch space.
- Reducer-like aggregation over residual evidence is a closer fit than raw text accumulation.

### `Hunter-Jiang/Progress-Aware-Belief-Update` Supports A Compact Belief Summary Over Full History

Sources:

- `Hunter-Jiang/Progress-Aware-Belief-Update` [`README.md`](https://github.com/Hunter-Jiang/Progress-Aware-Belief-Update/blob/main/README.md)

Observed pattern:

- PABU argues that full action-observation history is inefficient and that a compact belief state with selective retention can outperform full-history conditioning.
- Their framing is task progress, not self-state allostasis, but the key architectural move is the same: summarize what matters rather than replay everything.

Implication for Vicuña:

- Vicuña can justify a small belief state that summarizes unexplained cares or missing pressures instead of trying to make the self-state exhaustive.
- Selective retention and residual tracking are better design primitives than bloating the explicit self-state.

## Web / Paper Findings

### Belief States Are The Standard Way To Act Under Partial Observability

Source:

- Kaelbling, Littman, Cassandra, "Planning and acting in partially observable stochastic domains" [PDF via Brown](https://cs.brown.edu/research/pubs/techreports/reports/CS-96-08.html)

Relevant point:

- A partially observable controller should act on a belief state, not on raw observations alone.

Implication for Vicuña:

- Treat the explicit self-state as observation `o_t`, not full latent state `s_t`.
- Maintain a bounded posterior-like belief summary `b_t` and use `(gradient(o_t), b_t)` for gain control.

### Active Inference Gives The Right Decomposition: Pragmatic Control Plus Epistemic Value

Sources:

- Friston et al., "Active Inference: A Process Theory" [Neural Computation / arXiv mirror](https://arxiv.org/abs/1703.04176)
- Tschantz et al., "Scaling active inference" [arXiv](https://arxiv.org/abs/1910.10684)

Relevant point:

- Under partial observability, good action balances expected improvement in preferred state with reduction of uncertainty.

Implication for Vicuña:

- The belief layer should not only represent hidden concern pressure.
- It should expose whether a loop step should be more exploitative, more cautious, or more information-seeking because the self-model is incomplete.

### Homeostatic Control Under Partial Observability Is A Better Analogy Than Standard Reward Maximization

Sources:

- Keramati and Gutkin, "Homeostatic reinforcement learning" [eLife](https://elifesciences.org/articles/04811)
- Friston et al., "Active inference for learning and development in embodied neuromorphic agents" [MDPI Entropy](https://www.mdpi.com/1099-4300/26/7/582)

Relevant point:

- The controller is not maximizing a single external reward.
- It is trying to remain near preferred internal regimes while learning from incomplete information.

Implication for Vicuña:

- The right objective is not "invent new cares."
- It is "minimize expected allostatic error while accounting for uncertainty about missing internal variables."

### World-Model And Latent-State Work Suggests Using Residual Latents, Not Total Latent Replacement

Sources:

- Ha and Schmidhuber, "World Models" [arXiv](https://arxiv.org/abs/1803.10122)
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels" [arXiv](https://arxiv.org/abs/1811.04551)

Relevant point:

- Latent states are most useful when they summarize what is not directly observed, while still allowing prediction and error correction.

Implication for Vicuña:

- The latent concern system should track residual predictive error and hidden pressure, not replace the whole explicit self-state with a dense latent vector.

## Recommended Architecture

### 1. Keep The Current Self-State As The Observation Layer

Do not demote or hide:

- explicit registers,
- allostatic targets,
- self-model extensions,
- hard-memory summaries.

These remain the inspectable observation surface `o_t`.

### 2. Add A Small Belief Layer Over Missing Cares

Add a bounded belief state `b_t` with fixed-width summaries such as:

- residual unexplained allostatic shift,
- posterior confidence over current self-model sufficiency,
- latent pressure mass,
- novelty-backed hidden-care suspicion,
- promotion readiness,
- uncertainty decomposition:
  - uncertainty over known cares,
  - uncertainty due to missing observations,
  - uncertainty due to likely unmodeled cares.

This can be implemented with:

- a handful of scalar summaries plus
- a very small fixed number of latent concern slots with decay and clipping.

### 3. Update The Belief Layer Like A Filter, Not Like Free-Form Memory

At each settled transaction:

1. observe explicit pre-state and post-state,
2. compare predicted versus realized shift,
3. measure residual unexplained error,
4. attribute some of that residual to missing observation, noise, or possible hidden concern pressure,
5. update `b_t` with bounded decay and confidence rules.

This is POMDP-like and Markov-like enough for runtime control without needing full Bayesian inference.

### 4. Feed Only A Fixed Belief Summary Into The Gating MLP

The controller input should become:

- explicit self-state gradient,
- allostatic distance,
- existing loop and tool features,
- belief summary features.

Do not feed:

- raw hard-memory text,
- variable-length latent histories,
- unconstrained hidden states.

### 5. Use The Belief Layer To Change Control Style, Not To Rewrite Values

The safest first use is to modulate:

- caution,
- exploration,
- tool-seeking under ambiguity,
- self-observation pressure,
- counterfactual depth,
- planning/composition pressure.

The latent belief should not directly redefine desired self-state targets until promoted.

### 6. Promote Stable Hidden Concerns Into Explicit Self-Model State

When latent residue is:

- persistent,
- memory-supported,
- recurrent across sessions or task types,
- and predictive of future allostatic misses,

surface a promotion candidate for the self-model extension registry.

This keeps the belief layer as a scaffold for discovery rather than a permanent shadow ontology.

## Benefits

- Better calibration: the controller stops acting as if the explicit self-state is complete.
- Better caution: the system can avoid overconfident gain shifts when internal model coverage is weak.
- Better exploration: the system can deliberately favor tools, self-observation, or counterfactual probes when hidden pressure is suspected.
- Better extensibility: repeated unknowns can graduate into explicit self-model additions.
- Better biological plausibility: this is closer to active-inference or belief-state control than to a static authored self-profile.

## Risks And Consequences

- More opacity if latent summaries are not surfaced clearly.
- More instability if latent concern pressure can move gains too aggressively.
- Self-delusion risk if noisy residue is overinterpreted as a hidden care.
- Credit-assignment ambiguity because unexplained shift can come from model error, environment noise, or genuinely missing cares.
- Complexity creep if the belief layer becomes a second full self-model rather than a bounded residual controller.

## Conclusion

This is implementable and probably worthwhile, but only in a narrow form:

- explicit self-state remains the primary control surface,
- belief state models incompleteness and residual hidden pressure,
- gain control consumes a fixed belief summary,
- and persistent hidden patterns are promoted into explicit self-model state instead of remaining permanently opaque.

That gives Vicuña a credible "reason under uncertainty about itself" capability without abandoning the typed, inspectable, bounded architecture that currently makes the project coherent.
