# Research

## Local Codebase Findings

### Current difference between state switching and proper planning/composition

The current Vicuña runtime already has more structure than a naive action
router, but it still falls short of a true planner/composer.

What exists today:

- active and DMN runners have bounded `max_steps`, pending command ids, and tool
  wait/resume state
- active and DMN both score a small candidate table and pick one winner
- tool proposals and command queue entries are typed and host-visible
- DMN can continue from `internal_write` into tool or emit follow-up logic

What is still missing:

- no explicit plan object shared by active and DMN origins
- no typed sequence of substeps chosen before execution
- no notion of plan status, step status, or plan revision
- no reusable composition stage that decides how answer/tool/observe/emit steps
  relate
- tool observation resumes a runner, but does not revise a durable plan
- candidate scores still jump directly to a winning action, which is a routed
  state switch

Conclusion:

The runtime today is a bounded candidate router with local continuation logic.
It is not yet a proper planning/composition architecture.

## GitHub Codebase Research

### LangGraph

Repository: `langchain-ai/langgraph`

Files inspected:

- `libs/langgraph/langgraph/graph/state.py`
- `libs/prebuilt/langgraph/prebuilt/tool_node.py`

Relevant patterns:

- graph nodes communicate through explicit shared state rather than opaque local
  transitions
- routing can return structured commands instead of only terminal actions
- tool execution is treated as a node/step in the graph, not as an afterthought
- tools receive injected state/runtime/store through typed contracts, which keeps
  the execution surface inspectable

Implication for Vicuña:

Vicuña should not jump directly from candidate scores to a terminal action. It
should materialize a bounded plan state and treat tool use / observation as
typed plan steps.

### Voyager

Repository: `MineDojo/Voyager`

Files inspected:

- `voyager/voyager.py`
- `voyager/agents/skill.py`

Relevant patterns:

- task execution is separated from curriculum selection and skill retrieval
- reusable skills are retrieved and composed into execution context before acting
- tool-like programs are durable objects, not just one-off outputs

Implication for Vicuña:

Even before a full skill library exists, the runtime should introduce a proper
planning/composition surface so active and DMN flows can sequence execution in a
way that later supports skill retrieval and composition.

## Web Research

### ReAct

Source:

- [ReAct paper](https://arxiv.org/abs/2210.03629)

Relevant idea:

- reasoning and acting should interleave, with observations feeding later
  decisions instead of one-shot action selection

Implication:

Vicuña needs explicit observation -> revise -> next-step handling inside its
plan mode, especially around tool use.

### Voyager

Source:

- [Voyager paper](https://arxiv.org/abs/2305.16291)

Relevant idea:

- open-ended capability growth comes from executable skill accumulation and
  retrieval, not only reflection

Implication:

The first prerequisite is a planner/composer that can consume reusable
execution residue.

## Architecture Decision

The correct next step is not a third rail. It is a shared bounded planning mode
inside active and DMN runners:

- active and DMN remain the two origins
- a new shared planner produces bounded step sequences
- lifecycle phases remain as runner bookkeeping only
- semantic control shifts from direct winner-action switching to explicit plan
  execution and revision
- planning/composition gets its own functional LoRA family
