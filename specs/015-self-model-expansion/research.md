# Research: Self-Model Expansion For Efficient Goal Pursuit, User Satisfaction, And Self-Repair

## Research Questions

1. What does Vicuña already represent explicitly in its self-model?
2. What important internal state is missing if the goals are:
   - user satisfaction and user goal completion
   - return from undesirable to desirable self-state
   - continual pressure toward more efficient behavior
3. How do current public agent runtimes and primary research suggest structuring
   these additional surfaces?
4. What representation strategy preserves current runtime explicitness and
   parity instead of collapsing into opaque heuristics?

## Local Repository Evidence

### Current strengths

- The runtime already exposes a real typed self-surface through public C API
  types in [include/llama.h](/Users/tyleraraujo/vicuna/include/llama.h).
- The current register bank covers uncertainty, contradiction, novelty,
  topic shift, goal relevance, self relevance, social relevance, affordance,
  broadcast pressure, inhibition, follow-up continuation, memory write
  priority, time phase, tool salience, and channel state.
- The current feature vector already includes useful social, tool, decoder, and
  memory-geometric features such as `goal_top_similarity`,
  `tool_pending_pressure`, `social_trust`, `contradiction_score`,
  `uncertainty_score`, and `negative_user_valence`.
- Favorable-state, counterfactual, remediation, and governance traces are
  already explicit and bounded in [include/llama.h](/Users/tyleraraujo/vicuna/include/llama.h)
  and consumed in [src/llama-cognitive-loop.cpp](/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp).

### Current gaps

The audit findings in
[audit-report.md](/Users/tyleraraujo/vicuna/specs/013-functionality-audit/audit-report.md)
remain accurate:

- The self model is explicit but shallow.
- The register bank is mostly immediate-pressure oriented.
- The current feature space is still heavily heuristic.
- There is little explicit representation of:
  - progress toward user goals
  - expected remaining effort
  - recovery trajectory
  - self-estimate confidence
  - preference uncertainty
  - user misunderstanding risk
  - repetitive loop inefficiency
  - motivation to self-modify or self-repair

### Implication

The next expansion should not replace the current bank. The current bank is
still the correct fast-path control surface. The gap is that Vicuña lacks a
second layer of richer state estimates that can explain:

- whether the system is getting closer to a good outcome
- whether it is getting there efficiently
- whether the user is likely becoming more or less satisfied
- whether internal repair is worth spending inference on

## GitHub Codebase Research

### Letta

Files inspected:

- [letta/schemas/agent.py](https://github.com/letta-ai/letta/blob/main/letta/schemas/agent.py)
- [letta/schemas/memory.py](https://github.com/letta-ai/letta/blob/main/letta/schemas/memory.py)

Relevant patterns:

- Persistent agent state is not a flat score bank. It includes identity,
  model config, embedding config, compaction settings, tools, tags, run
  metrics, timezone, file limits, and memory blocks.
- Memory is decomposed into separate renderable surfaces: context-window
  overview, core memory, external memory summary, tool usage rules,
  directories, filesystem memory, and summary memory.

Implication for Vicuña:

- Strong agent systems separate operating context, memory composition, and
  execution metadata into typed surfaces.
- Vicuña should not attempt to express all new internal state through one
  enlarged scalar enum. It should add grouped typed profiles while retaining a
  compact fast control bank.

### OpenHands

File inspected:

- [openhands/controller/state/state_tracker.py](https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/controller/state/state_tracker.py)

Relevant patterns:

- Even in a legacy file, the runtime explicitly tracks iteration limits,
  budget limits, history filtering, metrics snapshots, and persistence.

Implication for Vicuña:

- A system that cares about efficiency should explicitly represent budget,
  step limits, and trajectory-local metrics.
- Vicuña currently lacks a comparable internal surface for expected remaining
  steps, accumulated inefficiency, or context-thrash risk.

### LangGraph

Repository inspected:

- [langgraph README](https://github.com/langchain-ai/langgraph)

Relevant pattern:

- Durable execution, resumability, human-in-the-loop state transitions, and
  persistent runtime state are treated as first-class system properties.

Implication for Vicuña:

- Strong agent runtimes expose execution continuity and state transitions
  clearly. Vicuña should expand self-state in a way that improves introspection
  and later evaluation, not only immediate control.

## External Primary Sources

### Homeostatic and active-inference framing

- [Active Inference and Learning](https://direct.mit.edu/neco/article/29/1/1/8155/Active-Inference-and-Learning)

Usefulness:

- Supports modeling internal control around expected free-energy-like pressures,
  uncertainty reduction, and homeostatic return to preferred state.

Implication for Vicuña:

- Favorable self-state should be decomposed into multiple explicit divergences,
  not only one aggregate.
- The runtime should represent both current divergence and expected recovery
  trajectory.

### Memory, reflection, and long-horizon social simulation

- [Generative Agents](https://arxiv.org/abs/2304.03442)

Usefulness:

- Demonstrates the value of explicit relevance, importance, and reflection
  layers for agent continuity.

Implication for Vicuña:

- Add explicit importance or urgency-like surfaces and reflection-ready summary
  surfaces instead of relying only on raw recent event pressure.

### Self-correction and iterative refinement

- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Self-Refine](https://arxiv.org/abs/2303.17651)
- [Large Language Models have Intrinsic Self-Correction Ability](https://arxiv.org/abs/2406.15673)

Usefulness:

- These works support explicit traces of critique, revision, and outcome-aware
  refinement rather than assuming one-pass control is sufficient.

Implication for Vicuña:

- The self-model should represent predicted gain from revision, recovery
  readiness, and prediction error on prior self-estimates.
- A future evaluator can later optimize these surfaces, but the surfaces should
  exist first.

### Tool use and competence shaping

- [Toolformer](https://arxiv.org/abs/2302.04761)
- [Voyager](https://arxiv.org/abs/2305.16291)

Usefulness:

- Tool use improves when the system models affordance, expected utility, and
  skill acquisition rather than only reacting to a local tool-needed flag.

Implication for Vicuña:

- The self-model should represent actionability and expected payoff of asking,
  answering, tool use, and waiting.
- It should also track competence and novelty pressure in a bounded way where
  relevant.

### User satisfaction estimation

- [Interpretable User Satisfaction Estimation for Conversational Systems with Large Language Models](https://aclanthology.org/2024.acl-long.598/)
- [CAUSE: Counterfactual Assessment of User Satisfaction Estimation in Task-Oriented Dialogue Systems](https://aclanthology.org/2024.findings-acl.871/)
- [User Satisfaction Estimation with Sequential Dialogue Act Modeling in Goal-oriented Conversational Systems](https://arxiv.org/abs/2202.02912)

Usefulness:

- These papers support treating user satisfaction as a structured inferential
  target rather than reducing it to sentiment alone.
- They also suggest that dissatisfaction robustness and dialogue-act dynamics
  matter.

Implication for Vicuña:

- The self-model should represent:
  - estimated user satisfaction
  - frustration risk
  - misunderstanding risk
  - repair necessity
  - preference uncertainty
- These should be kept separate from raw negative valence.

### Uncertainty and calibration

- [Calibrating the Confidence of Large Language Models by Eliciting Fidelity](https://arxiv.org/abs/2404.02655)
- [Revisiting Uncertainty Estimation and Calibration of Large Language Models](https://arxiv.org/abs/2505.23854)
- [SConU: Selective Conformal Uncertainty in Large Language Models](https://arxiv.org/abs/2504.14154)

Usefulness:

- These papers support separating uncertainty from confidence expression and
  highlight the importance of calibrated selective prediction.

Implication for Vicuña:

- The self-model should represent self-estimate confidence and evidence
  sufficiency separately from raw uncertainty.
- It should also record prediction error so future calibration systems have a
  target.

## Synthesis

### What else should be represented?

The strongest additions are not more random scalars. They are grouped explicit
surfaces in seven families:

1. Goal progress
   - progress estimate
   - blocker severity
   - dependency readiness
   - urgency
   - expected next-action gain
2. User outcome
   - estimated satisfaction
   - frustration risk
   - misunderstanding risk
   - trust repair need
   - preference uncertainty
   - user cognitive load estimate
3. Epistemic control
   - answerability
   - evidence sufficiency
   - ambiguity concentration
   - self-estimate confidence
   - tool-need confidence
4. Efficiency
   - expected steps remaining
   - expected inference cost remaining
   - repetition risk
   - context-thrash risk
   - tool round-trip cost
5. Recovery and homeostasis
   - divergence from favorable state by family
   - recovery momentum
   - regulation debt
   - unresolved tension load
   - over-activation and under-activation
6. Strategic mode
   - answer bias
   - ask bias
   - act bias
   - wait bias
   - exploit vs explore
   - compress vs deliberate
7. Self-improvement governance
   - update worthiness
   - expected gain from self-modification
   - evidence deficit
   - reversibility or blast radius
   - observability deficit

### How should it be represented?

The most robust structure is:

- keep the current register bank for low-latency control
- add typed profile structs for coherent domains
- add multiple horizon slices for each profile
- add bounded forecasts and prediction-error traces

This is better than one giant enum because it preserves:

- readability
- debuggability
- import/export stability
- explicit policy
- compatibility with future evaluators

## Recommendation

Expand the self-model in depth and organization, not only count:

- modestly enlarge the fast control register bank
- add profile-level typed surfaces for state that should not live as one-off
  scalars
- add trend and forecast state so the runtime can represent efficiency and
  recovery
- add explicit user-outcome and self-improvement readiness surfaces now, even if
  learned evaluators come later

This is the best path to making the system more continually self-motivating
without giving up the explicit, inspectable, whitepaper-aligned self-core.
