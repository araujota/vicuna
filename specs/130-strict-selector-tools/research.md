# Research: Strict Selector Tools And Waterfall VAD

## Decision

Replace staged freeform JSON selector turns with DeepSeek beta strict tool
calls that use enum-constrained arguments for family and method selection and a
strict schema-derived payload tool for payload submission. Keep the waterfall
controller in CPU-side code and fix VAD injection eligibility so every staged
step after new reasoning can receive renewed additive guidance.

## DeepSeek API Findings

### Strict tool mode is the right primitive for selector turns

DeepSeek's Tool Calls guide documents beta `strict` mode on function tools and
states that it works in both thinking and non-thinking modes. The same guide
documents that strict mode supports `enum`, `object`, `array`, `boolean`,
`number`, `integer`, and `anyOf`, with the requirement that every object field
be `required` and `additionalProperties` be `false`.

Implication:

- Family selection maps naturally to one strict tool like
  `select_family(family: enum[...])`.
- Method selection maps naturally to one strict tool like
  `select_method(method: enum[..., "back", "complete"])`.
- Payload selection can be expressed as one strict tool that mirrors the method
  contract directly, avoiding a second freeform JSON parser.

Source:

- DeepSeek Tool Calls guide:
  [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)

### JSON mode is explicitly known to return empty content

DeepSeek's JSON Output guide warns that JSON mode may occasionally return empty
content and recommends prompt adjustments, while also noting that `max_tokens`
must be set reasonably to avoid truncation.

Implication:

- The current staged selector design is built on the exact feature DeepSeek
  warns can return empty output.
- Moving selector turns to strict tools is stronger than continuing to tune
  freeform JSON prompts.

Source:

- DeepSeek JSON Output guide:
  [JSON Output](https://api-docs.deepseek.com/guides/json_mode)

### Thinking mode still uses one shared output budget

DeepSeek's Thinking Mode guide says `max_tokens` covers the entire generated
output, including the chain-of-thought part. Anthropic compatibility notes also
say `thinking.budget_tokens` is ignored.

Implication:

- We cannot solve the selector problem by splitting reasoning and visible
  budgets.
- The safest mitigation while keeping reasoning enabled is to force the visible
  control outputs through strict tool calls rather than freeform JSON content.

Sources:

- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [Anthropic API](https://api-docs.deepseek.com/guides/anthropic_api)

### Beta features expose prefix completion and beta routing

DeepSeek's Chat Prefix Completion guide requires the `/beta` base URL for beta
features.

Implication:

- Strict selector tool calls should also be routed explicitly through the beta
  endpoint path/config where required instead of the default chat-completions
  URL.

Source:

- [Chat Prefix Completion](https://api-docs.deepseek.com/guides/chat_prefix_completion/)

## GitHub Pattern Findings

### Production agent frameworks prefer structured tool outputs over hand-parsed JSON

OpenAI's `openai-agents-python` chat-completions path converts tool definitions
and `tool_choice` into structured provider request fields rather than depending
on freeform assistant JSON text. The framework treats tools as the primary
structured control surface for intermediate actions.

Implication:

- Our staged controller should keep its CPU-side waterfall, but selector turns
  should look like structured tool choices rather than assistant text that we
  later parse.

Source:

- `openai/openai-agents-python`
  [`src/agents/models/openai_chatcompletions.py`](https://github.com/openai/openai-agents-python/blob/9a96d9e787414510074dfbc84dab960d0d6d5c1d/src/agents/models/openai_chatcompletions.py)

### Community DeepSeek integrations still rely heavily on constrained structured outputs

Community DeepSeek tool/JSON integrations generally describe the application as
producing structured JSON outputs and emphasize contract-driven interaction
loops. They do not rely on a separate low-reasoning mode.

Implication:

- The design space is less about “find the right hidden reasoning knob” and
  more about making intermediate control turns structurally constrained.

Source:

- `Doriandarko/deepseek-engineer`
  [`deepseek-eng.py`](https://github.com/Doriandarko/deepseek-engineer/blob/9aa7a2d3611b1af42b4cde20a00a50df25001fbb/deepseek-eng.py)

## Local Code Findings

- The current staged controller builds one combined system prompt and relies on
  `response_format={"type":"json_object"}` plus freeform `content` parsing for
  family, method, and payload turns.
- VAD injection is currently gated by `find_active_tool_continuation_span()`,
  which is valid for classic assistant-tool-result continuations but false for
  staged selector turns. That is why updated VAD state appears in traces but
  `vad_guidance` stays null.
- The prompt cache already exists and can be extended to hold stable selector
  prefixes plus strict selector tool schemas.

## Chosen Direction

1. Add explicit server-owned strict selector tool schemas for family, method,
   and payload stages.
2. Route those staged selector requests through DeepSeek beta tool mode with
   thinking still enabled.
3. Parse stage decisions from tool-call arguments, not assistant `content`.
4. Keep the existing staged CPU-side loop, retries, and `back`/`complete`
   semantics.
5. Replace the old “active tool continuation span” rule for staged turns with a
   staged-eligibility rule based on the most recent staged assistant reasoning
   result so renewed VAD can be injected on each following stage.
