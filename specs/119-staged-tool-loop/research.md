# Research: Ubiquitous Staged Tool Family/Method/Payload Orchestration

## Decision

Implement staged family→method→payload selection as an explicit CPU-side controller using JSON-constrained provider turns. Use native tool execution only for the final validated method call, not for the meta-selection stages.

## Why This Design

### DeepSeek JSON Mode Fits Staged Selections

- DeepSeek’s JSON output guidance recommends explicit JSON instructions and `response_format={"type":"json_object"}` for reliable JSON responses.
- That maps cleanly onto the three staged outputs:
  - family selection JSON
  - method selection JSON
  - payload JSON

Implication: the staged controller should use ordinary chat turns with JSON mode rather than native tool calls for family/method/payload selection.

Source:
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode)

### DeepSeek Thinking/Tool Continuations Require Preserved Reasoning

- DeepSeek tool-use guidance requires `reasoning_content` to be passed back for same-turn tool continuations.
- Existing Vicuña behavior already preserves replayed `reasoning_content` for those continuations.

Implication: staged prompts and VAD/heuristic guidance must be additive messages. They must never rewrite or normalize stored `reasoning_content`.

Sources:
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)

### ReAct Is Preserved by Finer Checkpoints

- ReAct’s core contribution is explicit interleaving of reasoning and acting around observations.
- A staged family→method→payload controller does not remove this pattern. It makes the “act” side more inspectable by splitting one opaque selection into smaller verified checkpoints.

Implication: the staged controller is compatible with ReAct as long as the runtime still grounds on real tool observations and restarts from observation boundaries.

Source:
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

### Tool Metadata Should Stay Hierarchical

- Semantic Kernel models grouped tool metadata as plugin plus function metadata, including descriptions and parameter metadata.
- OpenAI Agents models each tool with explicit name, description, and strict JSON schema.

Implication: Vicuña should normalize from its existing capability catalog rather than inventing a second metadata system. The existing `tool_family_*`, `method_*`, and `input_schema_json` fields already match the needed hierarchy.

Sources:
- [Semantic Kernel `KernelPlugin`](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/functions/kernel_plugin.py)
- [Semantic Kernel `KernelFunctionMetadata`](https://github.com/microsoft/semantic-kernel/blob/main/python/semantic_kernel/functions/kernel_function_metadata.py)
- [OpenAI Agents `tool.py`](https://github.com/openai/openai-agents-python/blob/main/src/agents/tool.py)
- [OpenAI Agents `run.py`](https://github.com/openai/openai-agents-python/blob/main/src/agents/run.py)

### Native Provider Tool Calls Are the Wrong Abstraction for Meta-Selection

- Native tool calls are good for final execution requests with a concrete tool name and payload.
- They are not a good fit for family selection, back navigation, or explicit completion because those are controller-state transitions, not external actions.

Implication: treat family/method/payload selection as server-owned state-machine steps. Reserve actual tool invocation for the execution stage.

### Multi-Tool Agent Work Benefits From Planner/Executor Separation

- Recent agent literature such as OctoTools separates high-level planning from tool execution and uses standardized tool metadata to keep the system extensible.

Implication: the staged controller should be treated as a lightweight planner/executor bridge with explicit runtime state, not as a prompt-only trick.

Source:
- [OctoTools](https://arxiv.org/abs/2502.11271)

## Rejected Alternatives

### Model Every Stage as a Native Tool

Rejected because:
- it hides controller policy inside provider-native tool mechanics
- it weakens explicit runtime validation
- it complicates back navigation and completion
- it reduces the number of useful additive VAD/heuristic injection checkpoints

### Keep Existing Direct Tool Calling and Only Change the System Prompt

Rejected because:
- it does not guarantee the provider touches the runtime more often
- it does not create enforceable family/method/payload checkpoints
- it does not make future tool metadata requirements explicit

### Build a Separate Server-Local Tool Registry

Rejected because:
- the existing catalog already contains family, method, and contract data
- duplicate registries drift
- the user explicitly asked for extensibility, which is better served by one metadata source of truth

## Implementation Guidance

- Normalize from capability catalog entries that include:
  - `tool_family_id`
  - `tool_family_name`
  - `tool_family_description`
  - `method_name`
  - `method_description`
  - `input_schema_json` with descriptions on every field node
- Use strict JSON response parsing at each stage.
- Allow only synthetic controller sentinels for:
  - `back`
  - `complete`
- Keep all VAD and heuristic guidance additive.
- After every real tool result, restart from family selection.
