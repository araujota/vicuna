# Research: Authoritative ReAct Continuation

## Local Audit

### Current Behavior

`tools/server/server-context.cpp` builds authoritative ReAct prompts, parses a single generated control step, and then:

- retries only when parsing or structural validation fails
- resumes the loop only after a tool result arrives
- treats a structurally valid `answer` as terminal immediately

The critical path is:

- `prepare_react_prompt(...)`
- `parse_authoritative_react_step(...)`
- `send_final_response(...)`

Once `parse_authoritative_react_step(...)` returns a valid active `answer`, `send_final_response(...)` calls `llama_cognitive_active_authoritative_finish(...)` and returns the visible text to the user. There is no semantic check that the answer is actually grounded or complete.

### Observed Host Failure

For the Telegram weather request:

- the model produced hidden reasoning
- the parsed step was `action=answer`
- `tool_calls=0`
- the visible reply was procedural: “To provide an estimate ... I will use historical data...”

That step was accepted as terminal even though it neither answered the question nor performed the promised tool work.

## Difference From Established Tool Loops

### OpenAI Function Calling

OpenAI’s function-calling guide explicitly treats tool use as a multi-step conversation: after tool results are appended to the input, they are sent back to the model “to get a final response.” The same guide documents `tool_choice: "required"` when a tool call must happen. That is a loop, not a one-pass accept-any-answer pattern.

Source:
- [OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling)

### Anthropic Tool Use

Anthropic’s stop-reason guidance distinguishes `tool_use` from `end_turn`, and shows the application executing the tool and returning the result so the model can continue. It also documents `pause_turn` for server tools and instructs the application to continue the conversation rather than treating the partial response as complete.

Source:
- [Anthropic stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)

### ReAct Pattern

ReAct is fundamentally an iterative reason-act-observe loop. The key divergence in the current Vicuña path is not the presence of hidden thought or tool XML, but the acceptance rule: Vicuña currently accepts the first parseable answer-like visible text as terminal even when the task remains unresolved.

Source:
- [ReAct project page](https://react-lm.github.io/)

### Comparable Open-Source Agent Loop

Hugging Face `smolagents` runs a step loop until a final-answer condition is met. The loop continues through action and observation steps, and final termination depends on the tool/action outcome rather than the first non-empty assistant content. This reinforces that explicit continuation policy belongs in the runtime, not only in the prompt.

Source:
- `huggingface/smolagents`, [`src/smolagents/agents.py`](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py)

## Implications For Vicuña

The current authoritative ReAct runtime is already capable of:

- preparing a tool-visible turn
- parsing one tool call
- dispatching a tool
- resuming from tool results

So the missing pieces are not a new planner or a new tool runner. The missing pieces are:

1. first-step forcing for clearly mutable/live requests
2. semantic rejection of procedural non-answers
3. continuation retries beyond a tiny fixed cap

## Recommended Fix

1. Add explicit mutable/live-request detection from the latest foreground turn text.
2. When such a request arrives on an active user-origin turn, set first-step `tool_choice` to `required`.
3. Add a semantic validator after parse that rejects:
   - unsupported direct answers for clearly mutable/live requests before any tool grounding
   - procedural non-answers that narrate intended work instead of completing it
4. Retry the same authoritative turn with feedback instead of terminating.
5. Default the continuation budget to effectively unbounded so the loop can keep working until completion.
