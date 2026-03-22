# Research: Retry Grounding Surgery

## Local Audit

- `tools/server/server-context.cpp` currently decides whether a fresh mutable active turn must begin with a tool call by combining `foreground_request_requires_fresh_tool_grounding(...)` with `canonical_context_has_grounded_answer_candidate(task)`.
- `canonical_context_has_grounded_answer_candidate(task)` scans `canonical_react_messages(task)`, which currently appends bounded Telegram dialogue verbatim.
- The grounding check only looks for term overlap plus fact-like payload, so stale Telegram assistant text can count as "grounded" even when it is merely a prior conversational answer rather than a current tool-backed observation.
- Host logs for the Chicago weather Telegram turn showed repeated `action=answer` and `tool_calls=0`, with continuation rejection `visible reply described intended work or lack of access instead of completing the turn`, eventually converging only after dozens of retries.
- Host runtime state for chat scope `7502424413` contained prior assistant weather replies, confirming the polluted dialogue surface that matched the grounding heuristic.

## GitHub Research

### Hugging Face `smolagents`

Source: [src/smolagents/memory.py](https://github.com/huggingface/smolagents/blob/edb04feae012c1a96774c2e19066cd24046f8fba/src/smolagents/memory.py)

- `ActionStep` stores `tool_calls` and `observations` as distinct memory artifacts.
- `to_messages()` emits tool calls and observations as separate messages, making tool responses the trusted substrate for later reasoning.
- Error retries are also fed back as explicit tool-response style observations telling the agent to retry differently.

Inference: tool-backed observations should carry more trust than ordinary assistant prose when deciding whether a final answer is grounded.

## Web Research

### ReAct

Source: [ReAct](https://react-lm.github.io/)

- ReAct treats reasoning and acting as an interleaved loop where observations from actions ground subsequent reasoning and answers.

### Anthropic tool-use stop reasons

Source: [Anthropic stop reasons](https://docs.anthropic.com/es/api/handling-stop-reasons)

- Tool use is treated as a continuation loop: model emits tool use, application executes tool, tool result is returned, and the model continues until a final answer is produced.

## Design Conclusions

- Fresh mutable Telegram turns should trust tool observations and same-turn canonical runtime artifacts, not stale assistant dialogue, when deciding whether a direct answer is already grounded.
- Because continuation is intentionally unbounded, the server needs a bounded escalation threshold that upgrades mutable active retries to `tool_choice=required` after repeated rejected non-tool attempts.
- This remains compatible with the current authoritative ReAct architecture because the CPU still does not choose the tool or answer; it only constrains whether the next step must be a tool call.
