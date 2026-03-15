# Research Notes: Functionality Audit Of The Exotic-Intelligence Runtime

## Scope

This note records the evidence set used for the audit in
`specs/013-functionality-audit/audit-report.md`.

The audit question is not just whether Vicuña has specs for an "exotic
intelligence" runtime. It is whether the repository, as of March 11, 2026,
already implements a meaningful subset of that runtime, how the behavior maps
to `Vicuña_WP.md`, and how it compares with current public state of the art.

## Local Evidence

### Whitepaper and architecture intent

- `Vicuña_WP.md:10-205` defines the core memory-cascade and live-serving
  semantics: sliding window -> Active LoRA -> past LoRA stack, with explicit
  ordering and strict replay after weight changes.
- `Vicuña_WP.md:211-587` defines persistent self-state, register updates,
  active loop, DMN, counterfactual simulation, governance, and self-improvement
  constraints.
- `ARCHITECTURE.md:18-464` translates the same thesis into an implementation
  roadmap and documents the first implemented slice more concretely.

### Public API surface

- `include/llama.h:1473-1509` exposes Active LoRA, past LoRA, and live serving
  stack inspection.
- `include/llama.h:1511-1653` exposes the typed self-state API: time,
  register bank, identity/goals/commitments, working memory, memory handles,
  reactivation, tool state, social state, trace import/export, updater
  programs, counterfactual replay, and hard-memory query/archive surfaces.
- `include/llama.h:1655-1679` exposes the active loop and DMN entry points.

### Active LoRA and temporal memory implementation

- `src/llama-active-lora.cpp:1106-1163` computes Active LoRA size from current
  host/device memory budgets and creates a runtime adapter.
- `src/llama-active-lora.cpp:1166-1216` builds the frozen temporal bucket stack.
- `src/llama-active-lora.cpp:1219-1269` ingests token spans into Active LoRA,
  skips highly redundant spans above cosine similarity `0.995`, and supports a
  bounded remediation update path.
- `tests/test-active-lora.cpp` verifies initialization, built-in embedders,
  callback-backed embedders, and redundant-span suppression.
- `tests/test-past-lora.cpp` verifies rollover and temporal bucket behavior.
- `tests/test-serving-lora-stack.cpp:99-155` verifies the live runtime serving
  order `request -> all_time -> year -> quarter -> month -> week -> active`
  and checks that request adapters do not wipe runtime memory layers.

### Self-state implementation

- `src/llama-self-state.cpp:1623-1799` shows the first-stage feature builder.
  The current implementation is mainly lexical and rule-based:
  negation terms, uncertainty terms, self terms, error terms, negative-valence
  terms, sketch similarity, goal similarity, commitment similarity, tool-state
  summaries, recency, and decoder statistics.
- `src/llama-self-state.cpp:1801-1857` implements explicit prewrite and
  postwrite updates, working-memory admission, reactivation refresh,
  channel-state updates, social-state updates, and trace append.
- `src/llama-self-state.cpp:2198-2205` archives events to hard memory only when
  the typed self-state delta crosses a configured threshold.
- `src/llama-self-state.cpp:2248-2295` defines the default updater program as
  an explicit CPU-side policy with tunable rule weights and repair thresholds.
- `tests/test-self-state.cpp:623-666` verifies trace export/import and replay,
  including replay on the counterfactual channel without leaking into primary
  channel state.
- `tests/test-self-state.cpp:676-835` verifies bounded hard-memory query and
  delta-threshold archival behavior against a mock server.

### Active loop, DMN, counterfactuals, and governance

- `src/llama-cognitive-loop.cpp:914-988` scores four foreground actions:
  `answer`, `ask`, `act`, `wait`.
- `src/llama-cognitive-loop.cpp:1040-1225` implements pressure-gated DMN
  admission, favorable-self-state computation, counterfactual ladder
  generation, bounded remediation planning, governance gating, internal-write
  behavior, tool-job creation, and multi-burst emit decisions.
- `src/llama-cognitive-loop.cpp:1216-1218` renders repair output using a fixed
  canned message rather than a context-sensitive generation process.
- `tests/test-cognitive-loop.cpp:195-375` verifies the four foreground actions.
- `tests/test-cognitive-loop.cpp:378-405` verifies low-pressure DMN silence.
- `tests/test-cognitive-loop.cpp:407-531` verifies internal-write DMN routing,
  favorable-state profile generation, low-risk-first counterfactual ordering,
  and Active LoRA remediation.
- `tests/test-cognitive-loop.cpp:534-583` verifies DMN tool-oriented
  `gather_info` behavior.
- `tests/test-cognitive-loop.cpp:585-660` verifies high-continuation DMN emit
  behavior.
- `tests/test-cognitive-loop.cpp:662-773` verifies that repair pressure feeds
  the same DMN admission gate instead of bypassing governance.
- `tests/test-cognitive-loop.cpp:775-890` verifies repair-governance traces and
  repair-message rendering under sustained dissatisfaction.

### Server integration

- `tools/server/server-context.cpp:751-819` wires runtime enablement through
  environment variables:
  `VICUNA_ACTIVE_LORA`, `VICUNA_PAST_LORA`,
  `VICUNA_HARD_MEMORY_URL`, `VICUNA_HARD_MEMORY_TOKEN`,
  `VICUNA_HARD_MEMORY_CONTAINER_TAG`, `VICUNA_HARD_MEMORY_RUNTIME_ID`, and
  related thresholds.
- `tools/server/server-common.cpp:60-77` classifies foreground messages as user
  versus tool events before they enter the active loop.
- `tools/server/server-context.cpp:1241-1257` runs
  `llama_active_loop_process(...)` for foreground traffic.
- `tools/server/server-context.cpp:2155-2184` ingests evicted spans into Active
  LoRA during context shift and schedules strict KV replay if runtime memory
  weights changed.
- `tools/server/server-context.cpp:2332-2335` logs replay startup.
- `tools/server/server-context.cpp:2075-2090` polls the DMN only when all slots
  are idle and defers otherwise.

### Git history and repository divergence

- `git log` on the relevant files shows the current feature lineage on the
  Vicuña fork:
  - `d7355c22a` "Add temporal LoRA memory cascade"
  - `41fb439d9` "Implement persistent self-state runtime"
  - `b16493668` "feat: add cognitive runtime and supermemory integration"
- The current branch is `012-supermemory-self-hosting` on
  `https://github.com/araujota/vicuna.git`.
- The upstream remote is `https://github.com/ggerganov/llama.cpp.git`.
- GitHub code search on upstream `llama.cpp` found no matching APIs for the
  Vicuña-specific surfaces such as `llama_dmn_tick`,
  `llama_self_state_get_register`, `llama_active_lora_init`, or
  `llama_hard_memory_query`. These capabilities are fork-local rather than
  inherited upstream.

## External Primary Sources

### Memory, long-term state, and agent memory systems

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
  established the modern "memory stream + retrieval + reflection" framing for
  socially persistent agents.
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
  framed stateful agents as explicit memory hierarchies with controlled paging
  instead of hidden prompt stuffing.
- [MemoryLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624)
  is a direct research point of comparison for in-model or latent memory.
- [Letta docs](https://docs.letta.com/) describe a production-oriented
  stateful-agent platform centered on explicit memory blocks, durable state, and
  tool orchestration.
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview)
  describe durable graph-based agent execution, persistence, and long-running
  workflows.
- [Supermemory docs](https://supermemory.ai/docs) describe a dedicated external
  memory substrate with retrieval, profile/context surfaces, and deployment
  patterns.

### Reasoning/action loops and tool use

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
  remains a strong baseline for explicit action-thought interleaving.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
  is a canonical reference for tool-use augmentation.
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
  is a strong public example of autonomous curriculum, skill accumulation, and
  externalized memory/program growth.

### Reflection, self-improvement, and guarded adaptation

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
  shows a practical pattern for self-improvement via textual reflection rather
  than hidden weight mutation.
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
  is another baseline for bounded iterative self-improvement.

## Synthesis

### What is clearly implemented

- Vicuña is not only a whitepaper or spec bundle. It already has concrete C/C++
  implementation and regression coverage for:
  - bounded Active LoRA memory
  - temporal past-LoRA buckets
  - a typed self-state runtime
  - a foreground active loop
  - an idle/background DMN
  - a favorable-self-state profile
  - a counterfactual ladder
  - bounded remediation
  - governance and repair traces
  - hard-memory query and archival
  - partial `llama-server` integration

### Where the implementation is still crude

- Most "cognitive" behavior is explicit host-side policy, not learned adaptive
  competence.
- Self-state semantics are mostly 32-dimensional hashed sketches plus lexical
  probes and hand-authored weighted rules.
- Counterfactuals are scored ladders and trace replays, not model-based future
  simulation.
- Repair behavior is governed, but the emitted repair message is canned.
- DMN tool behavior creates typed pending jobs and selects tool kinds, but it
  does not yet look like a rich autonomous planner-executor runtime.
- The LoRA-memory approach is novel, but current public best practice for
  durable agent memory is still dominated by explicit external memory and
  orchestration systems.

### Overall framing for the report

- Relative to `Vicuña_WP.md`, the correct label for most pillars is
  "operational first slice" or "operational but crude", not "absent".
- Relative to current public state of the art, Vicuña is strongest where it
  uses explicit typed state and external hard-memory integration, and weakest
  where it relies on heuristic internal policies and runtime self-modification.
- The right characterization of the application today is:
  an experimental stateful cognitive runtime embedded in `llama-server`, not a
  mature autonomous recursive self-improver.
