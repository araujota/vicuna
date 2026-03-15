# Research Notes: Refreshed Functionality Audit

## Original Audit Baseline

Source baseline:

- `specs/013-functionality-audit/audit-report.md`
- `specs/013-functionality-audit/spec.md`
- `specs/013-functionality-audit/plan.md`

Original top-line judgment:

- Vicuña already implemented a real experimental cognitive runtime rather than a
  paper-only roadmap.
- It qualified as "exotic intelligence" only in a narrow engineering sense.
- It qualified as "crude RSI" only weakly, because self-improvement was bounded,
  heuristic, and not backed by broad evaluation or compounding skill growth.

Original priority gaps:

1. Replace more lexical heuristics with stronger learned or retrieved signals.
2. Make DMN tool invocation a fuller planner/executor path.
3. Add explicit external evaluation around remediation and updater-policy
   changes.
4. Expand hard-memory retrieval and memory-handle use into a first-class
   substrate.
5. Improve repair behavior from canned messages to context-sensitive governed
   outputs.
6. Produce end-to-end demonstrations of multi-step autonomous behavior across
   sessions.

## Local Runtime Evidence

### Architecture And Whitepaper

- `ARCHITECTURE.md`
- `Vicuña_WP.md`

Current docs now explicitly state that:

- functional LoRA gains are produced by a shared gating MLP driven by a typed
  self-state gradient toward favorable allostasis
- functional gain control uses bounded Gaussian exploration and Adam-based
  meta-updates
- runtime LoRA tensor mutation and temporal write-bias control also use Adam
- the counterfactual ladder remains explicit CPU-side policy instead of being
  collapsed into an optimizer

This is a material shift from the original audit’s framing, where most of the
interesting behavior was typed and real but still obviously heuristic.

### Current Public API Surface

Primary file:

- `include/llama.h`

Important exposed surfaces:

- functional activation decisions with predicted gains, sampled noise,
  allostatic distance, and allostatic gradient norm
- functional update info with meta-loss, Adam step, adapter update norm, and
  adapter optimizer step
- temporal self-improvement trace
- expanded self-model surfaces for reactivation, forecast, prediction error,
  governance, counterfactuals, tool-loop traces, and repair state

Interpretation:

- the architecture is not only present internally; much more of it is now
  inspectable through typed runtime APIs
- expandability remains one of Vicuña’s strongest qualities because new traces,
  families, and evaluator surfaces can be added without changing the base-model
  ontology

### Self-State And Self-Model

Primary file:

- `src/llama-self-state.cpp`

Current evidence:

- explicit reactivation priorities
- forecast and prediction-error traces
- counterfactual evaluation on isolated channels
- typed updater program with governance/repair parameters
- hard-memory archive thresholding

Interpretation:

- the self model is still authored, but it is no longer just a flat register
  sheet; it includes fast summaries, horizon slices, and forecast error
  bookkeeping
- this improves functionality and expandability more than it improves true
  generality

### Active Loop, DMN, Governance, And Counterfactuals

Primary file:

- `src/llama-cognitive-loop.cpp`

Current evidence:

- shared bounded tool-loop substrate under active and DMN paths
- explicit functional microphase routing
- low-risk-first counterfactual ladder
- governance trace and repair gating
- temporal self-improvement loop comparing active path results against temporal
  ablation candidates

Interpretation:

- since the first audit, the most important new capability is not generic
  autonomy; it is that the DMN now has a tighter bridge from counterfactual
  evidence to bounded self-modification of temporal bias
- this is still far from broad self-improvement, but it is less ad hoc than the
  earlier remediation-only framing

### Active LoRA, Functional Gating, And Runtime Optimization

Primary file:

- `src/llama-active-lora.cpp`

Current evidence:

- functional allostatic distance objective
- shared functional gating network with stochastic exploration
- online Adam meta-update of the gating MLP
- Adam-based runtime LoRA tensor updates
- Adam-based temporal write-bias updates

Interpretation:

- this is the biggest elegance improvement since the first audit
- the project now has a more coherent ontology for "self-state changes future
  behavior" than it did before: self-state gradient -> gain prediction ->
  bounded action -> state shift -> meta-update
- however, the pseudo-gradients driving runtime LoRA writes are still authored
  and local rather than learned from a generative world model or a broad eval
  harness

### Tests And Validation

Primary files:

- `tests/test-active-lora.cpp`
- `tests/test-cognitive-loop.cpp`
- `tests/test-self-state.cpp`

Current evidence:

- targeted regression coverage exists for Active LoRA, self-state, DMN
  background actions, governance, repair-pressure admission, functional LoRA
  activation, temporal self-improvement, and hard-memory behavior

Interpretation:

- the codebase is unusually well instrumented for a speculative architecture
- the main validation weakness is not lack of unit/integration coverage; it is
  lack of end-to-end agent-eval coverage that would prove the self-modifying
  pathways improve outcomes over time

## GitHub Delta Since The First Audit

Key commits:

- `69e222cbe9f893145d0c91d59618131921193f40`
  - "Implement cognitive runtime runner and self-model upgrades"
  - introduced the first audit bundle and a broad surface expansion of the
    cognitive runtime
- `07437946cad0c5360b3d264cc9605cb635d2e733`
  - "self-state: add temporal self-improvement loop"
  - added DMN temporal self-improvement, temporal bias state, and tests
- `9858ba91c94d23fdb3112a61bf6512c3b625ddf2`
  - "Add Adam to self-state runtime updates"
  - added Adam-backed runtime LoRA writes, temporal bias updates, and explicit
    optimizer telemetry

High-signal delta conclusion:

- the architecture’s weakest pillar in the original audit, self-improvement,
  has improved materially in local mathematical coherence
- it has not improved materially in external validation, compounding skill
  growth, or long-horizon autonomy

## External Research And Official Sources

### Durable Stateful Agent Systems

1. Letta official repository and docs
   - `letta-ai/letta`
   - stateful agents with advanced memory, tools, skills, and subagents
   - best used as the external benchmark for durable memory and operator-facing
     statefulness
2. LangGraph official repository and docs
   - `langchain-ai/langgraph`
   - durable execution, human-in-the-loop, comprehensive memory, and agent
     graphs
   - best used as the benchmark for expandability, execution durability, and
     observability
3. MemGPT paper
   - `https://arxiv.org/abs/2310.08560`
   - explicit memory-tiering and interrupt-driven control flow
4. Generative Agents paper
   - `https://arxiv.org/abs/2304.03442`
   - memory, reflection, and planning loop for believable autonomous behavior

### Self-Improvement And Reflection

1. Reflexion
   - `https://arxiv.org/abs/2303.11366`
   - verbal reflection with explicit episodic memory and external feedback
2. Self-Refine
   - `https://arxiv.org/abs/2303.17651`
   - iterative feedback/refinement without weight mutation
3. Voyager
   - `https://github.com/MineDojo/Voyager`
   - strongest contrastive example for compounding skill growth through a skill
     library, curriculum, and self-verification

### Biologically Inspired / Allostatic / Active-Inference Directions

1. Active Inference for Learning and Development in Embodied Neuromorphic Agents
   - `https://www.mdpi.com/1099-4300/26/7/582`
   - relevant because it frames biologically inspired control around prediction
     error minimization, POMDP structure, and closed-loop adaptation
2. Designing Explainable Artificial Intelligence with Active Inference
   - `https://arxiv.org/abs/2306.04025`
   - relevant because it treats introspection and self-auditing as explicit
     architectural targets rather than side effects
3. Homeostatic reinforcement learning for integrating reward collection and physiological stability
   - `https://elifesciences.org/articles/04811`
   - relevant because it formalizes internal-variable regulation rather than
     treating reward alone as the control target
4. Meta-reinforcement learning in homeostatic regulation
   - `https://2025.ccneuro.org/abstract_pdf/Yoshida_2025_Meta-Reinforcement_Learning_Homeostatic_Regulation.pdf`
   - relevant as a current signal that homeostatic/allostatic control is moving
     toward meta-learned regulation rather than static heuristics

### Brain-Inspired Continual Learning / Metaplasticity

1. Learning to Modulate Random Weights: Neuromodulation-inspired Neural Networks For Efficient Continual Learning
   - `https://arxiv.org/abs/2204.04297`
2. Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks
   - `https://www.nature.com/articles/s41467-022-34938-7`
3. Bayesian continual learning and forgetting in neural networks
   - `https://www.nature.com/articles/s41467-025-64601-w`
4. A brain-inspired algorithm that mitigates catastrophic forgetting of artificial and spiking neural networks with low computational cost
   - `https://pmc.ncbi.nlm.nih.gov/articles/PMC10456855/`

These are relevant because Vicuña’s online adapter mutation and temporal memory
story now clearly sits in continual-learning territory. The strongest external
lesson is that plasticity should be context-dependent and paired with replay,
metaplasticity, or uncertainty-aware stabilization.

### World Models And User/Environment Modeling

1. GenRL: Multimodal-foundation world models for generalization in embodied agents
   - `https://arxiv.org/abs/2406.18043`
2. Language Models Meet World Models: Embodied Experiences Enhance Language Models
   - `https://arxiv.org/abs/2305.10626`

These are relevant because Vicuña still lacks a real generative world/user
model. Its counterfactual and allostatic control loops operate over typed
snapshots and pressure heuristics rather than a learned simulator of user,
environment, and internal state evolution.

## Gap Delta Summary

### Clearly Improved Since The First Audit

1. **Self-improvement ontology**
   - improved from heuristic bounded remediation toward an explicit
     observe/predict/act/meta-update loop
2. **Architectural elegance**
   - improved because functional LoRA gain is no longer framed as a persistent
     prompt/system artifact
3. **Inspectability**
   - improved because more of the optimizer, gating, temporal-bias, and
     counterfactual surfaces are exposed in typed APIs and tests

### Only Partially Improved

1. **Replacing heuristics with learned signals**
   - improved somewhat through the gating MLP and allostatic objective
   - not improved enough at the self-state feature layer, which remains heavily
     authored
2. **DMN autonomy**
   - improved via richer bounded loop scaffolding and temporal self-improvement
   - still not a durable planner/executor runtime

### Largely Unresolved

1. External evaluation around self-modification
2. Hard memory as the primary memory substrate
3. Context-sensitive repair behavior
4. Compounding skill acquisition and long-horizon autonomy
5. Multi-session demonstrations of cumulative improvement
