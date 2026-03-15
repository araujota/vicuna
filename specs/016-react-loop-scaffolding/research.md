# Research: Long-Running Agent Loops For Vicuña Active And DMN Control

## Local Runtime Findings

Current Vicuña is already closer to an explicit state machine than to a generic
prompt loop:

- `src/llama-cognitive-loop.cpp` scores bounded active actions:
  `ANSWER`, `ASK`, `ACT`, `WAIT`
- the DMN is a bounded maintenance router with pressure, favorable-state,
  counterfactual, remediation, and governance stages
- there is already typed tool state, memory state, and repair logic

So the local gap is not "add ReAct from scratch." The local gap is:

- no durable public step/phase model
- no explicit tool registry metadata
- no episode/proposal/observation surfaces that future tools can plug into

## External Patterns

### ReAct

The original ReAct pattern is useful as the inner `think -> act -> observe`
cycle, but not as a complete runtime architecture on its own.

Primary source:

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

Useful takeaway:

- explicit alternation between reasoning and action is valuable

Limitation for Vicuña:

- raw ReAct does not specify durable runtime state, governance, foreground vs
  background policies, or safety-oriented tool metadata

### LangGraph

LangGraph’s current agent implementation is more informative for production
structure than plain ReAct. Its prebuilt agent is a graph/state-machine wrapper
around model and tool nodes with explicit continuation logic and remaining-step
budgets.

Primary source:

- [LangGraph `chat_agent_executor.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py)

Useful takeaway:

- robust agent loops are usually state machines with explicit step budgets and
  routing, not raw prompt-only while-loops

### Letta

Letta separates ordinary foreground execution from "sleeptime" or background
work. That matches Vicuña’s split between active loop and DMN.

Primary sources:

- [Letta `agent_loop.py`](https://github.com/letta-ai/letta/blob/main/letta/agents/agent_loop.py)
- [Letta `sleeptime_multi_agent.py`](https://github.com/letta-ai/letta/blob/main/letta/groups/sleeptime_multi_agent.py)

Useful takeaway:

- foreground and background cognition should not share identical loop policy

### smolagents

smolagents provides a clean example of bounded multi-step action loops with
explicit step memory, planning intervals, and tool validation.

Primary source:

- [smolagents `agents.py`](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py)

Useful takeaway:

- future-proof scaffolding needs explicit step state, max-step budgets, and
  tool envelopes even when actual tool suites are still small

## Decision

The best architecture for Vicuña is:

1. keep the existing explicit active and DMN scoring logic
2. add a shared bounded tool-loop substrate
3. expose separate active and DMN policies on top of that substrate

Rejected alternative:

- one generic long-running ReAct loop reused unchanged for active and DMN

Reason for rejection:

- it would erase useful asymmetry already present in the runtime
- it would weaken inspectability by moving policy into generic prompting
- it would make parity with remediation, governance, and ablation harder to
  preserve

## Parity Constraints For Implementation

- preserve current `ANSWER/ASK/ACT/WAIT` and DMN action enums
- preserve current favorable-state, counterfactual, remediation, governance,
  and LoRA update ordering
- add tool/phase/episode metadata around the existing decisions
- keep loop policy explicit in CPU-side code
