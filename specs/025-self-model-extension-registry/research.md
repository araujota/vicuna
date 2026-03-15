# Research Notes: Self-Model Extension Registry

## Local Repository Evidence

- `specs/015-self-model-expansion/` already established the authored self-model
  as layered typed profiles plus forecast/prediction-error traces.
- `src/llama-self-state.cpp` currently computes those profiles entirely from the
  fixed authored register/memory/tool surfaces; it has no bounded extension
  registry.
- `src/llama-active-lora.cpp` currently feeds the gating MLP from a fixed
  self-state/favorable snapshot and therefore needs a fixed-size summary, not a
  dynamic raw extension list.
- `src/llama-hard-memory.cpp` already exposes typed query results and is the
  cleanest place to bootstrap automatic discovery of hard-memory-backed
  self-model representations.
- `src/llama-cognitive-loop.cpp` consumes the authored core indirectly through
  favorable profiles and summary registers, so the least disruptive integration
  is to let extensions modify self-model horizons and extension summaries rather
  than rewrite cognitive-loop control policy from scratch.

## GitHub Research

### Letta

- `letta-ai/letta` models persistent agent state with explicit memory blocks and
  strong typed schemas (`README.md`,
  `letta/serialize_schemas/pydantic_agent_schema.py`).
- Relevant lesson: keep core memory/state explicit and typed, and make
  extensions first-class schema objects rather than prompt-only fragments.

### LangGraph

- `langchain-ai/langgraph` `ToolNode` uses injected runtime/state/store
  contracts (`libs/prebuilt/langgraph/prebuilt/tool_node.py`).
- Relevant lesson: tools should receive or write state through explicit typed
  host-controlled channels, not by smuggling control data through natural
  language output.

### Voyager

- `MineDojo/Voyager` keeps a persistent skill manager and retrieval-driven
  action context (`voyager/voyager.py`).
- Relevant lesson: discovered state should be reusable and inspectable rather
  than ephemeral; bounded retrieval-backed additions are a stronger pattern than
  one-shot reflection text.

## Web Research

- Active-inference and explainable-AI work supports fixed prior structure plus
  adaptive inferred state rather than a fully unconstrained latent replacement.
- Homeostatic-control work supports separating internal-variable regulation from
  contextual state that helps policy without itself becoming an optimization
  target.
- This aligns with the requested split:
  - authored self-model as genetic prior
  - discovered/tool-authored state as bounded extensions
  - explicit flags deciding whether an extension affects gain, allostasis, or
    both

Key sources:

- Active inference / explainability:
  `https://arxiv.org/abs/2306.04025`
- Homeostatic reinforcement learning:
  `https://elifesciences.org/articles/04811`
- MemGPT / Letta memory-tier perspective:
  `https://arxiv.org/abs/2310.08560`

## Design Conclusions

1. Keep the authored self-model core unchanged as the always-present prior.
2. Add a bounded extension registry instead of mutating the core enum/register
   bank on the fly.
3. Give every extension explicit source, domain, flags, salience, and
   confidence.
4. Separate:
   - gain influence
   - contextual activation
   - allostatic participation
5. Use hard-memory counterfactual promotion as an explicit shadow-scoring step
   before inserting extensions.
6. Feed functional gating from a fixed summary of extension state, not from a
   variable-length raw set.
