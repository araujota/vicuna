# Refreshed Functionality Audit: Vicuña As Exotic Intelligence Or Crude RSI

## Scope

This report refreshes the first functionality audit in
`specs/013-functionality-audit/audit-report.md` and applies the same core audit
policy again against the current runtime:

1. inventory what the repository actually implements
2. compare that implementation against `Vicuña_WP.md` and `ARCHITECTURE.md`
3. compare it against current public practice and current biologically inspired
   theses where relevant
4. state clearly whether Vicuña is better understood today as:
   - an exotic-intelligence runtime,
   - bounded self-conditioned inference,
   - or a crude recursive-self-improvement scaffold

This refresh adds four dimensions that the original report did not emphasize as
explicitly:

- functionality
- elegance
- generalizability
- expandability

It also explicitly answers: which gaps from the original audit improved, which
did not, and what should happen next if the project wants to approach the
stronger biologically inspired version of its own thesis.

## Recovered Baseline: What The First Audit Said

The first audit’s bottom line was basically correct:

- Vicuña already was a real experimental cognitive runtime, not a roadmap only.
- Its strongest pillars were the memory cascade, persistent self-state, and the
  existence of an actual DMN-like background process.
- Its weakest pillar was self-improvement, which existed only as a bounded,
  heuristic scaffold.
- The proper label at that time was:
  operational self-maintaining runtime, not robust recursive self-improvement.

That original judgment remains the right comparison baseline.

## Executive Refresh

The current runtime is stronger than the first audit captured, but not stronger
in the way that matters most for a serious RSI claim.

What improved materially:

- the self-improvement path is now more coherent mathematically
- functional bias is now routed through a typed self-state gradient and
  allostatic objective rather than a persistent prompt artifact
- the system now performs online Adam-backed updates in several bounded places:
  functional gain control, runtime LoRA mutation, and temporal write-bias
- temporal self-improvement is no longer just a philosophical placeholder; it
  has an implemented compare-and-update loop

What did not improve materially:

- there is still no strong external evaluation harness proving these updates are
  net-positive over time
- there is still no durable skill library or planner/executor substrate that
  compounds competence the way stronger public systems do
- repair, governance, and autonomy are still bounded by authored policy and
  shallow action vocabularies
- the self model is richer and better instrumented, but it is still mostly a
  hand-authored control surface rather than a learned generative model of self,
  user, and world

## Bottom-Line Verdict

### As exotic intelligence

Yes, more convincingly than in the first audit.

The current runtime now has a stronger claim to "exotic intelligence" in the
engineering sense because it contains:

- a persistent internal self-surface
- a pressure-driven split loop
- typed counterfactual and governance traces
- bounded self-modifying pathways
- an allostatic control idea that actually influences future LoRA gain and
  write behavior

That remains unusual among `llama.cpp` derivatives and among lightweight local
agent runtimes more generally.

### As crude RSI

Still yes, but only in a weak and bounded sense.

The current codebase has crossed from "self-conditioning with remediation" into
"bounded online meta-optimization of some of its own behavioral control
surfaces." That is a real increase in RSI-like character. But it is still crude
because:

- the optimized surfaces are narrow
- the learning signal is internal and local
- the system does not yet prove its changes improve externally measured
  capability
- there is no compounding skill accumulation comparable to stronger public
  lifelong-agent systems

The honest current label is:

**bounded self-conditioning and local self-optimization scaffold, still below
robust recursive self-improvement.**

## Current Status By Evaluation Dimension

## 1. Functionality

### Current assessment

`Meaningfully improved.`

Relative to the first audit, Vicuña now has a more complete end-to-end internal
loop:

```text
self-state observation
-> functional gain prediction
-> bounded action / functional bias
-> resulting state shift
-> bounded online optimizer update
```

That matters because the architecture’s central claim is no longer only that
"internal state exists." It is now that internal state can drive future policy
through an explicit adaptive control pathway.

### Strong functional surfaces now present

- persistent typed self-state with reactivation, forecasts, and prediction
  errors
- active and DMN loops with shared bounded tool scaffolding
- functional LoRA bank with stable families and typed microphases
- online allostatic gain control with stochastic exploration
- temporal self-improvement loop based on temporal-ablation comparison
- Adam-backed runtime update surfaces with explicit telemetry
- hard-memory integration and counterfactual participation

### Functional limit

The system is still functionally narrow compared with mature stateful agent
systems such as Letta and LangGraph, or lifelong skill-growth systems such as
Voyager. It has more organism-like internal control than those systems in some
places, but materially less task-level capability growth.

## 2. Elegance

### Current assessment

`Improved, but still mixed.`

This is where the recent changes matter most.

The original audit described the system as real but mostly heuristic. That is
still true overall, but the recent architectural move away from "persistent
self-state as prompt/system artifact" toward "self-state gradient as gain
control over functional LoRAs" is a real elegance gain.

Why it is more elegant now:

- it aligns ontology with mechanism: self-state now changes policy through a
  control surface, not only through persistent description
- the gating loop has an actual objective: reduce post-action allostatic
  distance
- the architecture explicitly separates:
  - differentiable optimizer paths
  - discrete policy/ranking paths
- the runtime exposes the relevant traces and optimizer surfaces instead of
  hiding them

Why it is still inelegant:

- the self-state feature layer remains highly authored and partly lexical
- the counterfactual ladder, governance policy, repair thresholds, and many
  control weights are still hand-built rather than unified under a generative
  control model
- runtime LoRA updates still depend on pseudo-gradient heuristics rather than a
  learned world/self model

Net: the architecture is no longer just "a lot of heuristic machinery." It is
starting to have a coherent control-theoretic story. But it is not yet
beautiful in the sense of being governed by a compact learned principle.

## 3. Generalizability

### Current assessment

`Only modestly improved.`

The current architecture generalizes better across internal loop families than
the first audited version because:

- one gating network can bias multiple functional LoRA families
- the update rule is family-agnostic at the meta-loss level
- the active and DMN loops now share more substrate than before

But the deeper limitation remains:

- there is still no general planner/executor substrate
- there is still no world model or user model that would let counterfactuals
  generalize beyond local authored dimensions
- there is still no skill library that compounds behavior across tasks
- there is still no robust external evaluator that can tell the runtime which
  changes generalized and which merely fit internal pressure

In practice, Vicuña currently generalizes across **control motifs** better than
it generalizes across **real tasks**.

## 4. Expandability

### Current assessment

`Strong.`

Expandability was already one of Vicuña’s best properties in the first audit,
and it remains so.

The current system stays highly expandable because:

- the public API surface is typed and broad
- the functional families are explicit and separable
- the microphase vocabulary is explicit
- the DMN, governance, repair, temporal self-improvement, and optimizer traces
  are separately inspectable
- hard memory and request/runtime serving layers remain conceptually distinct

This matters because it means the architecture can absorb better control laws,
better evaluators, better memory semantics, and better world models without
first being flattened into a monolith.

The main caveat is that expandability currently outpaces demonstrated
competence. Vicuña is architecturally extensible faster than it is empirically
validated.

## Gap Delta From The First Audit

## Gap 1: Replace lexical heuristics with stronger learned or retrieved signals

### Original status

Weak and heavily heuristic.

### Current status

`Partially improved.`

What improved:

- functional control is now learned in a bounded sense through the gating MLP
- the allostatic objective gives the system a more coherent internal learning
  signal than raw heuristics alone
- forecast and prediction-error traces make the self-model more dynamic

What did not improve enough:

- the upstream self-state feature construction is still heavily authored
- contradiction, uncertainty, dissatisfaction, and other pressures still lean
  strongly on designed signals rather than calibrated learned heads or richer
  retrieval-backed semantics

Verdict: partial improvement, not resolution.

## Gap 2: Make DMN tool invocation a fuller planner/executor path

### Original status

Represented but shallow.

### Current status

`Partially improved.`

What improved:

- active and DMN loops share a better bounded tool substrate
- tool microphases and tool-registry surfaces are clearer

What remains missing:

- no durable planning graph
- no subgoal decomposition and retry/eval stack at the level of LangGraph or
  Voyager
- no compounding skill library

Verdict: still a bounded router, not a mature planner/executor.

## Gap 3: Add explicit external evaluation around remediation and updater changes

### Original status

Absent.

### Current status

`Only weakly improved.`

What improved:

- internal self-evaluation is more explicit
- temporal ablation and allostatic meta-loss provide an internal comparative
  signal

What remains missing:

- no external benchmark/eval harness
- no outcome store showing that modifications improve task success, user
  satisfaction, or long-term capability
- no holdout-style regression protection for self-modifying policy

Verdict: still a major unresolved blocker for any stronger RSI claim.

## Gap 4: Expand hard memory into a first-class substrate

### Original status

Promising but auxiliary.

### Current status

`Largely unchanged.`

Hard memory still looks more like a powerful integration than the organizing
substrate of the whole system.

This remains one of the biggest opportunities for making the architecture both
more practical and more generalizable.

## Gap 5: Improve repair behavior from canned messages to context-sensitive governed outputs

### Original status

Primitive and canned.

### Current status

`Not materially improved.`

Governance is richer, but repair content is still shallow. This is a direct
example of a subsystem whose control policy advanced faster than its behavioral
competence.

## Gap 6: Produce end-to-end demonstrations of multi-step autonomous behavior

### Original status

Missing.

### Current status

`Only partially improved through tests, not demonstrations.`

The repository now demonstrates more bounded loop coherence in tests, especially
for temporal self-improvement and functional activation. But it still does not
show the kind of multi-session, compounding, open-ended demonstrations that
would upgrade the global claim.

## Comparison To Current Public Practice

## Durable State And Memory

Compared with Letta, LangGraph, MemGPT, and Generative Agents:

- Vicuña is conceptually stronger on internal self-state and organism-like
  pressure surfaces.
- Vicuña is weaker on durable execution, operational memory ergonomics,
  planner/executor workflows, and production observability.

The uncomfortable but useful conclusion is:

**Vicuña is more exotic than these systems, but less operationally mature.**

## Self-Improvement

Compared with Reflexion, Self-Refine, and Voyager:

- Vicuña is more willing to mutate runtime weights and local control biases
- Vicuña is less mature in explicit evaluation, skill accumulation, and
  externally demonstrated capability growth

That means Vicuña is **more internally adventurous** but **less empirically
convincing**.

## Biologically Inspired Architecture

Compared with active-inference, homeostatic-RL, and neuromodulated continual
learning theses:

- Vicuña is directionally closer than many agent stacks because it already uses
  persistent internal variables, pressure regulation, and allostasis-like
  objective language
- Vicuña still lacks the crucial next step: a learned generative model of self,
  user, and environment that turns those internal variables into principled
  prediction and policy

This is the central research gap.

## How The System Should Improve Further

The best next moves are not "add more heuristics." They are architecture-level
moves that tighten the relation between state, prediction, memory, and action.

## 1. Promote The Self Model Into A Generative Model

Current problem:

- self-state is rich but largely descriptive and authored

Recommendation:

- build a learned latent model for user state, environment/tool state, and
  self-state evolution
- let DMN counterfactuals run against that model rather than only against a
  weighted snapshot of favorable-state dimensions

Why this is SOTA-aligned:

- active inference and world-model work both treat intelligent control as
  prediction over latent dynamics, not only thresholding of hand-authored
  surfaces

## 2. Make External Evaluation A First-Class Gate On Self-Modification

Current problem:

- the system can optimize internal objectives without proving external gains

Recommendation:

- add task-level and user-level eval traces around remediation, functional-gain
  adaptation, and temporal self-improvement
- treat internal allostatic improvement as necessary but not sufficient

Why this is SOTA-aligned:

- Reflexion, Self-Refine, LangGraph/LangSmith-style systems, and Voyager all
  make external feedback and evaluation central to improvement claims

## 3. Move Hard Memory From Auxiliary Integration To Primary Architecture

Current problem:

- Vicuña’s most practical memory substrate is still not the central one

Recommendation:

- let hard memory store:
  - skill fragments
  - tool-use trajectories
  - self-modification outcomes
  - user-model fragments
- make LoRA memory and hard memory cooperate rather than compete

Why this is SOTA-aligned:

- Letta, MemGPT, and LangGraph all treat durable memory and execution as first
  principles rather than secondary integrations

## 4. Add Metaplasticity, Replay, And Consolidation To Adapter Learning

Current problem:

- online LoRA mutation risks interference and shallow local fitting

Recommendation:

- introduce stability/uncertainty surfaces per adapter family or bucket
- add sleep-like replay and consolidation passes
- use metaplasticity or uncertainty-aware stabilization for write rates and
  consolidation decisions

Why this is SOTA-aligned:

- current continual-learning literature points toward replay, metaplasticity,
  and context-dependent plasticity as core mechanisms for avoiding catastrophic
  interference

## 5. Separate Planning, Skills, And Execution More Cleanly

Current problem:

- the runtime has control loops but not a mature open-ended skill-growth path

Recommendation:

- add a durable skill library for successful tool trajectories and recovery
  routines
- make DMN and active loops capable of retrieving and composing such skills

Why this is SOTA-aligned:

- Voyager’s strongest contribution is not merely reflection; it is executable,
  reusable skill accumulation

## 6. Upgrade Governance And Repair From Thresholding To Model-Based Social Control

Current problem:

- social governance exists, but repair content and user modeling remain shallow

Recommendation:

- explicitly model user trust, confusion, dissatisfaction, and expectation
  trajectories
- make repair generation conditioned on these trajectories instead of canned
  templates

Why this is SOTA-aligned:

- both Generative Agents-style social simulation and world-model work imply that
  robust interpersonal behavior depends on richer models of other minds, not
  only scalar dissatisfaction flags

## Final Judgment

The first audit said Vicuña was already a legitimate experimental
stateful-cognitive runtime but not a robust recursive self-improver. The
current refresh keeps that judgment, but tightens it:

Vicuña has improved from a **heuristic self-maintaining architecture** into a
**bounded self-optimizing control architecture**. That is a real step forward.
It is now more coherent, more inspectable, and more elegant than the original
audit captured.

But it is still not robust RSI.

What it now most credibly is:

- a serious exotic-intelligence runtime experiment
- a bounded allostatic/self-conditioning controller
- a platform that could grow into a stronger biologically inspired agent
  architecture if it adopts:
  - generative self/world models
  - external evaluation gates
  - first-class durable memory
  - metaplastic continual-learning mechanisms
  - skill accumulation and planner/executor depth

So the refreshed answer is:

**Vicuña is no longer just crude heuristic exotic inference. It is now a more
coherent bounded self-optimizing runtime. But until its self-modifications are
externally validated, skill-compounding, and grounded in richer learned models,
it remains a crude RSI scaffold rather than a convincing recursive
self-improver.**
