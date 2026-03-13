# Research: Proper Loop Runners For Vicuña

## Local Findings

Current Vicuña already has:

- explicit action scoring for active and DMN
- typed self-state and tool state
- bounded loop traces and a tool scaffold

Current missing piece:

- no durable pending-command queue
- no resumable runner state
- no bounded internal re-entry for DMN beyond a single selected action
- no host-facing acknowledgment path for command completion

## GitHub Findings

### LangGraph

Source:

- `langchain-ai/langgraph`
  `libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`

Relevant pattern:

- explicit `remaining_steps`
- conditional routing between model and tool nodes
- interrupt-before / interrupt-after hooks
- tool result validation before continuation

Implication for Vicuña:

- runner needs explicit remaining-budget logic and resumable tool waits

### Letta

Source:

- `letta-ai/letta` `letta/agents/agent_loop.py`

Relevant pattern:

- separate foreground agent and sleeptime/background variants

Implication for Vicuña:

- active and DMN should share runner substrate but not identical continuation
  policy

### smolagents

Source:

- `huggingface/smolagents` `src/smolagents/agents.py`

Relevant pattern:

- explicit step loop
- optional planning interval
- step memory and tool-output processing
- synchronous or streaming execution

Implication for Vicuña:

- a runner should separate planning state from tool execution state and keep
  bounded step bookkeeping

### OpenHands

Source:

- `All-Hands-AI/OpenHands` `openhands/controller/agent_controller.py`

Relevant pattern:

- event-driven controller
- pending action bookkeeping
- should-step gating on user message or observation
- loop recovery / stuck detection

Implication for Vicuña:

- the runtime should move toward event-driven pending-command handling rather
  than one-shot traces only

## Web / Paper Findings

### ReAct

Source:

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

Relevant pattern:

- act/observe alternation is useful

Limitation:

- paper-level ReAct does not provide durable runner state or host integration

### Plan-and-Solve

Source:

- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091)

Relevant pattern:

- separating planning from execution improves reliability for multi-step tasks

Implication for Vicuña:

- DMN especially benefits from an explicit plan-first maintenance pass before
  external action

### ReWOO

Source:

- [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323)

Relevant pattern:

- useful separation between reasoning/planning and observation-dependent
  execution

Implication for Vicuña:

- active and DMN runners should not stall all logic on external waits; they
  should preserve explicit precomputed intent and resume cleanly

## Decision

Vicuña should adopt:

- event-driven planner-executor runners
- bounded pending-command queues
- persistent runner status
- DMN internal re-entry within a fixed budget

It should not adopt:

- a free-form, unconstrained, prompt-only while-loop
