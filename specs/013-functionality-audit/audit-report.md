# Functionality Audit: Vicuña As A Crude RSI "Exotic Intelligence"

## Scope

This report audits the repository as it currently stands against three
reference frames:

1. `Vicuña_WP.md`, treated as the original whitepaper and behavioral target.
2. The repository's actual implementation, tests, and `llama-server`
   integration.
3. Current public state of the art as of March 11, 2026, using primary papers
   and official project documentation.

The core question is not whether Vicuña is philosophically interesting. It is
whether the codebase already behaves like a crude recursive-self-improvement
runtime with persistent self-representation and self-motivated autonomous
behavior, and where it stands relative to stronger public implementations.

## Executive Assessment

Vicuña already implements a real experimental cognitive runtime on top of
`llama.cpp`. It is not just a roadmap. The repository contains:

- a bounded memory cascade with runtime-generated Active LoRA plus frozen
  temporal buckets
- a typed persistent self-state
- a foreground active loop and an idle/background DMN
- counterfactual ranking, bounded remediation, governance, and repair traces
- hard-memory query and archival integration
- partial but genuine `llama-server` runtime wiring

The important qualification is that almost all of the "intelligence" is still
explicitly authored in CPU-side control logic. The system's motivational
structure is real in the narrow engineering sense: registers, pressure signals,
working-memory admission, reactivation, follow-up pressure, repair pressure, and
tool/job state all influence future behavior. But the semantics are still crude
and mostly heuristic. The result is best described as an operational
stateful cognitive runtime, not a robust autonomous recursive self-improver.

## Bottom-Line Verdict

### As an "exotic intelligence"

The application does qualify in a narrow engineering sense. It is unusual among
open `llama.cpp` derivatives because it maintains an explicit internal
self-surface and uses that surface to condition memory, background maintenance,
tool/job state, and emission policy.

### As a crude RSI system

The application qualifies only in a weak and tightly bounded sense.

- It can evaluate alternate updater programs and alternate trajectories through
  replay and typed counterfactual scoring.
- It can apply a bounded form of self-modification through Active LoRA
  remediation.
- It can use dissatisfaction, contradiction, uncertainty, and tool state as
  pressure signals that alter future behavior.

It does not yet qualify as strong RSI in the more serious research sense.

- It does not perform broad autonomous capability improvement.
- It does not run an evidence-heavy eval loop that proves its self-modifications
  are net-positive.
- It does not maintain a mature planner-executor skill library.
- It does not demonstrate robust endogenous agenda formation beyond authored
  pressure heuristics.

The honest label is: operational scaffold for bounded self-conditioning and
maintenance, not robust recursive self-improvement.

## What The Application Currently Is

As implemented today, Vicuña is best understood as an instrumented agent runtime
with four working subsystems:

1. A memory substrate:
   sliding window, Active LoRA, past-LoRA buckets, and optional hard memory.
2. A persistent self substrate:
   register bank, goals, commitments, working memory, reactivation priorities,
   social state, tool state, and traces.
3. A foreground control loop:
   score `answer`, `ask`, `act`, `wait` for each user/tool event.
4. A background maintenance loop:
   compute pressure, rank candidate background actions, perform bounded
   remediation, and decide whether to stay silent, write internally, invoke a
   tool, or emit.

This is a much stronger claim than "the repo contains ideas." It is supported by
the public API in `include/llama.h:1473-1679`, concrete implementations in
`src/llama-active-lora.cpp`, `src/llama-self-state.cpp`,
`src/llama-cognitive-loop.cpp`, regression tests in `tests/`, and runtime
integration in `tools/server/server-context.cpp`.

## Functional Summary Matrix

| Area | Current status | Whitepaper parity | Public-SOTA parity | Short judgment |
| --- | --- | --- | --- | --- |
| Memory cascade | Implemented and server-wired | Operational first slice | Divergent and unproven | Real system, novel but not SOTA-standard |
| Persistent self-core | Implemented and test-covered | Operational but crude | Partial | Strong explicit-state foundation |
| Register recomputation | Implemented | Partial-to-operational | Behind learned/calibrated systems | Heuristic and inspectable |
| Active loop | Implemented and exercised in tests/server | Operational first slice | Partial | Functional foreground router |
| DMN | Implemented and idle-polled in server | Operational first slice | Behind durable agent runtimes | Real background controller, still shallow |
| Counterfactual/self-improvement | Implemented in bounded form | Partial | Below reflection/eval-heavy approaches | Scaffold, not robust RSI |
| Governance and repair | Implemented | Partial-to-operational | Partial | Useful gating, simplistic outputs |
| Hard memory | Implemented and integrated | Supports whitepaper intent | Closest to current practice | Practical external memory hook |
| End-to-end autonomous organism behavior | Very partial | Below intended design | Below SOTA agent stacks | Not yet mature autonomy |

## Pillar-By-Pillar Audit

### 1. Memory Cascade

#### Expected behavior from the whitepaper

The whitepaper defines the cascade as:

`sliding attention window -> Active LoRA -> frozen past-LoRA stack`

It also requires deterministic live-serving ordering and strict replay when an
Active LoRA write changes the effective serving weights
(`Vicuña_WP.md:44-205`).

#### Current implementation

The implementation is real and non-trivial.

- Active LoRA is initialized from explicit host/device budget fractions, not a
  fixed hard-coded memory size (`src/llama-active-lora.cpp:1106-1163`).
- The past memory stack uses named temporal buckets and per-bucket rank/budget
  planning (`src/llama-active-lora.cpp:1166-1216`).
- Active ingestion skips highly redundant spans with cosine similarity above
  `0.995` and supports a bounded remediation update path
  (`src/llama-active-lora.cpp:1219-1269`).
- The serving order is enforced and regression-tested:
  `request -> all_time -> year -> quarter -> month -> week -> active`
  (`tests/test-serving-lora-stack.cpp:99-155`).
- `llama-server` ingests evicted spans on context shift and schedules strict KV
  replay when Active LoRA weights changed
  (`tools/server/server-context.cpp:2155-2184`,
  `tools/server/server-context.cpp:2332-2335`).

#### Expected runtime behavior today

When the context window shifts, evicted text can become live runtime memory.
That memory is not only stored for audit. It changes the active serving stack,
and the server will rebuild retained prompt state if the weights changed. This
is one of the strongest pieces of whitepaper-to-code parity in the repository.

#### Whitepaper parity

`Operational first slice.`

The core cascade is there, the serving order is there, and the replay discipline
is there. The remaining gap is not whether the mechanism exists. The gap is
whether the mechanism is semantically strong enough to represent rich memory.

#### Public-SOTA parity

`Exploratory and divergent, not established SOTA.`

Public best practice for agent memory is still dominated by explicit external
memory, retrieval, and orchestration layers rather than online mutation of the
serving stack:

- [Generative Agents](https://arxiv.org/abs/2304.03442)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [Letta](https://docs.letta.com/)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [Supermemory](https://supermemory.ai/docs)

Vicuña's LoRA-memory path is novel, but it is not a public consensus best
method, and it does not yet have the empirical validation or operational
maturity that external-memory systems have.

#### Main gaps

- Memory quality depends on heuristic embedding/write logic rather than a proven
  learned compression objective.
- Past buckets are temporally structured but still coarse.
- There is no evidence yet that the LoRA path preserves factual precision or
  compositional recall as well as explicit retrieval systems.

### 2. Persistent Self-Representation

#### Expected behavior from the whitepaper

The whitepaper requires a mathematical self-core that persists across context
eviction and includes registers, identity, goals, tool state, time state,
commitments, and memory handles (`Vicuña_WP.md:66-245`).

#### Current implementation

This is one of the strongest implemented pillars.

- The public API exposes typed self-state surfaces for time, registers,
  identity, goals, commitments, working memory, reactivation, tool state,
  social state, traces, and updater programs (`include/llama.h:1511-1653`).
- The implementation keeps bounded working memory, reactivation priorities, tool
  state, and social state in a CPU-side container
  (`src/llama-self-state.h:39-181`,
  `src/llama-self-state.cpp:973-1094`,
  `src/llama-self-state.cpp:1490-1512`,
  `src/llama-self-state.cpp:1859-1960`).
- Trace export/import and replay are implemented and tested
  (`src/llama-self-state.cpp:1145-1397`,
  `tests/test-self-state.cpp:623-666`).

#### Expected runtime behavior today

The application can maintain a persistent internal self-surface even when the
prompt contents change. It can carry explicit notions of unresolved goals,
commitments, current channel state, tool backlog, user dissatisfaction, and
memory reactivation targets. This is a real continuity mechanism, not just a
prompt summary.

#### Whitepaper parity

`Operational but crude.`

The whitepaper asked for an explicit self-core. The code delivers one. What is
still crude is the semantics: many surfaces are hashed sketches, bounded scalar
registers, or heuristic summaries rather than rich learned latent state.

#### Public-SOTA parity

`Partial.`

Vicuña is directionally aligned with current public best practice where explicit
durable state is concerned. Stateful agent systems such as
[Letta](https://docs.letta.com/) and
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) also rely
on explicit long-lived state, memory objects, and persistent execution context.

Where Vicuña trails is in ecosystem maturity:

- fewer durable workflow abstractions
- shallower memory schemas
- weaker planner/executor organization
- less battle-tested persistence and recovery

#### Main gaps

- The self model is explicit but shallow.
- Identity, goals, and commitments are represented as bounded token-derived
  sketches rather than richer world models.
- There is no evidence of long-running multi-session agent operations at the
  level of mature stateful agent platforms.

### 3. Register Bank And Feature Extraction

#### Expected behavior from the whitepaper

The whitepaper expects a typed register bank updated through explicit prewrite
and postwrite phases, using embeddings, decoder signals, tool/environment
signals, and memory geometry (`Vicuña_WP.md:249-424`).

#### Current implementation

The two-stage update pipeline exists.

- Feature extraction combines 32-dimensional event sketches, similarity to
  working memory, memory handles, goals, commitments, identity, lexical probes,
  social/tool state, and decoder statistics
  (`src/llama-self-state.cpp:1623-1799`).
- Prewrite and postwrite feature builders and update application are explicit
  (`src/llama-self-state.cpp:1801-1857`).
- The updater program is inspectable and configurable
  (`src/llama-self-state.cpp:2248-2295`).

#### Expected runtime behavior today

The system derives contradiction, uncertainty, novelty, tool pressure, social
bond strength, broadcast pressure, inhibition, and memory-write pressure from a
mix of lexical and geometric features. Those signals are then used everywhere
else in the runtime.

#### Whitepaper parity

`Partial-to-operational.`

The update order and explicitness match the whitepaper closely. What does not
yet match the more ambitious spirit is the sophistication of the feature model.

#### Public-SOTA parity

`Below stronger calibrated approaches.`

Modern public agent systems typically externalize more logic into planners,
retrievers, or task graphs, and learned research systems often use stronger
embedding/retrieval or evaluator components. Vicuña's register pipeline is
useful and inspectable, but it is still mostly a hand-authored control system.

#### Main gaps

- Heavy reliance on lexical heuristics for contradiction, uncertainty, and user
  valence.
- No strong evidence of calibrated learned heads beyond optional callback hooks.
- Limited semantic depth in the feature space compared with stronger retrieval
  or evaluator pipelines.

### 4. Active Engagement Loop

#### Expected behavior from the whitepaper

The active loop should absorb user/tool events, update shared state, and decide
whether to answer, ask, act, or wait (`Vicuña_WP.md:424-458`).

#### Current implementation

- The foreground loop scores four typed candidates and records a structured
  trace (`src/llama-cognitive-loop.cpp:914-988`).
- Regression tests cover all four actions and tool-follow-up behavior
  (`tests/test-cognitive-loop.cpp:195-375`).
- `llama-server` routes foreground traffic through this loop and classifies
  user vs tool events (`tools/server/server-common.cpp:60-77`,
  `tools/server/server-context.cpp:1241-1257`).

#### Expected runtime behavior today

Foreground traffic no longer bypasses the cognitive runtime. User messages and
tool returns both go through the same self-state and memory surfaces before the
system scores whether it should answer, ask, act, or wait.

#### Whitepaper parity

`Operational first slice.`

This part is working in the intended shape.

#### Public-SOTA parity

`Partial.`

Relative to explicit agent frameworks influenced by
[ReAct](https://arxiv.org/abs/2210.03629) and graph runtimes such as
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), the
active loop is currently much shallower. It is a good typed router, but not a
full planner/executor/retry/evaluator workflow.

#### Main gaps

- No rich action decomposition.
- Limited evidence of tool-argument synthesis or tool-selection optimization.
- No durable foreground planning graph.

### 5. DMN Background Loop

#### Expected behavior from the whitepaper

The DMN should be pressure-driven, not timer-chatter. It should react to
contradiction, uncertainty, reactivation, unfinished goals, tool deltas,
counterfactual opportunities, and continuation pressure, while obeying
broadcast policy (`Vicuña_WP.md:460-518`).

#### Current implementation

- The DMN computes a typed pressure vector and only admits work above a threshold
  (`src/llama-cognitive-loop.cpp:1040-1068`).
- It scores four background actions: `silent`, `internal_write`,
  `invoke_tool`, `emit` (`src/llama-cognitive-loop.cpp:1089-1159`).
- It performs maintenance actions such as past-LoRA tick and reactivation
  refresh (`src/llama-cognitive-loop.cpp:1162-1171`).
- The server polls the DMN only when all slots are idle and otherwise explicitly
  defers it (`tools/server/server-context.cpp:2075-2090`).
- Tests cover silent, internal-write, tool, and emit paths
  (`tests/test-cognitive-loop.cpp:378-660`).

#### Expected runtime behavior today

When the server is idle and internal pressure is high enough, Vicuña can carry
out a background maintenance step without a new user message. This is the
strongest support for the user's framing that the system's internal
self-representation can motivate autonomous behavior.

#### Whitepaper parity

`Operational first slice.`

The DMN exists and is pressure-gated as intended. The remaining gap is depth.
The system has the architecture of a background mind, but not yet the
capabilities of a rich one.

#### Public-SOTA parity

`Below durable agent runtimes.`

The presence of a background loop is distinctive, but current public best
practice for autonomous long-running agents tends to use explicit orchestration,
durable execution graphs, retry policies, tool routing, and external memory:

- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [Letta](https://docs.letta.com/)
- [Voyager](https://arxiv.org/abs/2305.16291)

Vicuña's DMN is more biologically framed and more internal, but materially less
capable than those systems in task-level autonomy.

#### Main gaps

- Limited autonomous action vocabulary.
- No robust background task graph or scheduler beyond the DMN action choice.
- Tool invocation is represented, but not yet operationally rich.

### 6. Counterfactual Simulation And Self-Improvement

#### Expected behavior from the whitepaper

The whitepaper expects the DMN to simulate alternate futures and policy choices,
rank them, and tightly gate self-improvement (`Vicuña_WP.md:519-587`).

#### Current implementation

- The runtime computes a favorable-self-state profile and a low-risk-first
  counterfactual ladder (`src/llama-cognitive-loop.cpp:377-530`,
  `src/llama-cognitive-loop.cpp:1075-1077`).
- The DMN can produce bounded remediation plans, including Active LoRA updates
  or `gather_info` tool work (`src/llama-cognitive-loop.cpp:1127-1182`).
- Self-state trace export/import and counterfactual replay exist
  (`src/llama-self-state.cpp:1145-1397`,
  `tests/test-self-state.cpp:623-666`).
- Tests verify low-risk-first candidate ordering and bounded remediation
  (`tests/test-cognitive-loop.cpp:462-531`).

#### Expected runtime behavior today

The system can compare internal alternatives and make a bounded local
self-modification decision. That is real self-conditioning. But the mechanism is
still closer to heuristic ladder ranking than to rich simulation of alternate
world trajectories.

#### Whitepaper parity

`Partial.`

The scaffolding is real. The strong version of the whitepaper claim is not yet
realized.

#### Public-SOTA parity

`Below reflection-heavy approaches.`

The strongest public self-improvement patterns remain mostly externalized and
evaluation-driven rather than hidden-weight self-mutation:

- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Self-Refine](https://arxiv.org/abs/2303.17651)
- [Voyager](https://arxiv.org/abs/2305.16291)

Those systems improve through reflection traces, program growth, memories,
curricula, and explicit evaluation loops. Vicuña instead performs a more novel
but less validated internal remediation path. That is interesting, but not yet
state of the art in robustness.

#### Main gaps

- Counterfactuals are not grounded in broad external evaluation.
- Self-improvement is narrowly scoped and weakly validated.
- No evidence of compounding skill acquisition comparable to stronger public
  autonomous-agent systems.

### 7. Governance, Safety, And Repair

#### Expected behavior from the whitepaper

Self-modification and background outputs should be explicitly governed, and
repair should be possible but controlled (`Vicuña_WP.md:506-587`).

#### Current implementation

- Governance traces evaluate proposal families and outcomes
  (`src/llama-cognitive-loop.cpp:668-725`,
  `src/llama-cognitive-loop.cpp:1075-1159`).
- Repair pressure contributes to the same DMN admission gate rather than acting
  as a bypass (`src/llama-cognitive-loop.cpp:764-779`).
- Repair emission can be triggered under dissatisfaction/valence/inhibition
  conditions, but the output is a fixed message
  (`src/llama-cognitive-loop.cpp:712-725`,
  `src/llama-cognitive-loop.cpp:1216-1218`).
- Tests verify repair-pressure admission and repair-governance behavior
  (`tests/test-cognitive-loop.cpp:662-890`).

#### Expected runtime behavior today

The system can notice dissatisfaction-like state, feed it into governance, and
sometimes render an apology/repair message. The existence of a separate
governance gate is a real strength because it keeps "self-generated" pressure
from turning into unconditional output.

#### Whitepaper parity

`Partial-to-operational.`

The governance shape is right. The content is still simple.

#### Public-SOTA parity

`Partial.`

Many modern agent stacks have stronger operational guardrails at the workflow
level, but fewer have this exact typed internal-governance structure. Vicuña is
interesting here, but still primitive in actual social/behavioral capability.

#### Main gaps

- Repair text is canned rather than context-sensitive.
- Governance is policyful, but not backed by a rich external evaluation stack.
- User trust management is still shallow.

### 8. Hard Memory Integration

#### Expected behavior from the whitepaper and architecture

The self-state should have explicit durable memory handles and should be able to
use external long-term memory rather than relying only on LoRA-based internal
memory (`Vicuña_WP.md:217-245`, `ARCHITECTURE.md:328-340`).

#### Current implementation

- The API exposes typed hard-memory configuration, query, result, and archive
  traces (`include/llama.h:1638-1653`).
- Self-state archival is thresholded by typed register delta
  (`src/llama-self-state.cpp:2198-2205`).
- The server reads hard-memory configuration from environment variables
  (`tools/server/server-context.cpp:776-819`).
- Tests verify bounded query results, routing metadata, above-threshold archive
  behavior, and below-threshold suppression
  (`tests/test-self-state.cpp:694-835`).

#### Expected runtime behavior today

This is the most practical memory surface in the runtime. Vicuña can query an
external memory service and archive significant self-state perturbations with
typed metadata. This makes the system materially closer to current public
practice than the LoRA-memory path alone.

#### Whitepaper parity

`Operational support for the intended architecture.`

#### Public-SOTA parity

`Closest to current best practice.`

Relative to public systems such as [Supermemory](https://supermemory.ai/docs),
[Letta](https://docs.letta.com/), and memory-centric agent architectures like
[MemGPT](https://arxiv.org/abs/2310.08560), this is the area where Vicuña is
most aligned with current practice.

#### Main gaps

- Hard memory exists as a bounded integration, not yet as the center of the
  whole agent architecture.
- The richer retrieval/planning loops common in mature memory products are not
  yet present.

### 9. Whole-System Autonomous Behavior

#### Expected behavior from the user framing

The user asked whether the application should be understood as a crude RSI
"exotic intelligence" that maintains an internal self-representation that
motivates autonomous behavior.

#### Current implementation

The strongest evidence in favor is:

- internal self-state persists outside prompt text
- internal pressure can trigger DMN activity without a new user message
- dissatisfaction, contradiction, uncertainty, reactivation, and tool deltas
  can alter future behavior
- bounded self-modification and tool-gathering actions exist

The strongest evidence against overclaiming is:

- most semantics are still heuristic
- background actions are few
- autonomous follow-through is limited
- no mature long-horizon planning loop is present
- no broad eval-driven self-improvement loop is present

#### Final parity judgment

`Autonomous self-maintaining runtime: yes, in crude bounded form.`

`Robust recursive self-improver: no.`

That distinction matters. The codebase already contains a meaningful internal
organizing self-model. It does not yet contain the richer capabilities that
would justify strong claims about self-improving intelligence.

## Comparison To Current State Of The Art By Goal

### Goal: Durable memory and continuity

Current public best practice favors explicit external memory, retrieval, and
persistent execution context:

- [Generative Agents](https://arxiv.org/abs/2304.03442)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [Letta](https://docs.letta.com/)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [Supermemory](https://supermemory.ai/docs)

Vicuña is ahead in one unusual way: it experiments with directly memory-bearing
runtime weight deltas. It is behind in the practical maturity, reliability, and
retrieval richness of explicit-memory systems.

### Goal: Persistent self-representation

There is no settled public SOTA for "selfhood" in machines. The practical
consensus is much narrower: keep agent state explicit, typed, durable, and
recoverable. On that narrower standard, Vicuña is aligned. Its weakness is not
absence of self-state. Its weakness is the simplicity of its state semantics.

### Goal: Autonomous tool-using behavior

State of the art favors explicit workflows, planners, retries, and durable
orchestration:

- [ReAct](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [Voyager](https://arxiv.org/abs/2305.16291)

Vicuña's active loop and DMN create a distinctive organism-like control story,
but they do not yet match the task-level competence and workflow depth of these
approaches.

### Goal: Recursive self-improvement

Publicly convincing self-improvement systems are usually bounded, explicit, and
evaluation-heavy:

- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Self-Refine](https://arxiv.org/abs/2303.17651)
- [Voyager](https://arxiv.org/abs/2305.16291)

Vicuña is conceptually ambitious here, but not yet stronger than these systems.
Its internal remediation path is unusual, but not yet validated enough to count
as stronger practice.

## Highest-Signal Findings

1. The repository already implements the whitepaper's main architectural
   surfaces in code. This is not a paper-only project.
2. The memory cascade and self-state pillars have the strongest whitepaper
   parity.
3. The DMN is real and can trigger idle-time autonomous maintenance actions.
4. The strongest practical subsystem relative to public best practice is hard
   memory, not LoRA-based memory.
5. The weakest subsystem relative to public best practice is self-improvement:
   it exists, but only as a bounded heuristic scaffold.
6. The system's autonomy is real but narrow. It is better described as
   self-maintaining than self-improving.

## Priority Gaps

1. Replace more lexical heuristics with stronger learned or retrieved signals.
2. Make DMN tool invocation a full planner/executor path rather than a typed job
   placeholder.
3. Add explicit external evaluation around remediation and updater-policy
   changes.
4. Expand hard-memory retrieval and memory-handle use so external memory becomes
   a first-class substrate, not just an auxiliary integration.
5. Improve repair behavior from canned messages to context-sensitive governed
   outputs.
6. Produce end-to-end demonstrations that show multi-step autonomous behavior
   across sessions rather than only unit and integration slices.

## Final Judgment

The correct high-level writeup is:

Vicuña currently stands as a serious experimental implementation of a
stateful, self-conditioned agent runtime. It already maintains an internal
self-representation, uses that representation to generate motivational
pressures, and can perform bounded autonomous maintenance and bounded
self-modification. That makes it a legitimate crude "exotic intelligence"
runtime in the engineering sense.

It is not yet close to the stronger form of the project vision. Its cognition
is still mostly hand-authored policy. Its memory semantics are still shallow.
Its counterfactuals are still heuristic. Its self-improvement loop is still
narrow and weakly validated. Relative to current public state of the art, it is
most compelling as a novel architecture experiment and least convincing as a
finished autonomous recursive self-improver.
