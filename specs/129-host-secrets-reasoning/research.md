# Research: Host Secrets And Verbatim Stage Traces

## Local codebase findings

- The active host user units currently launch
  `/home/tyler-araujo/Projects/.../tools/ops/run-vicuna-runtime.sh` and
  `run-telegram-bridge.sh` directly, without an `EnvironmentFile=` stanza, so
  rebuild-safe secrets cannot rely on the user units alone.
- `tools/ops/install-vicuna-system-service.sh` already writes
  `/etc/vicuna/vicuna.env` for system services, but the current user-service
  path does not consume it.
- `tools/openclaw-harness/src/config.ts` already supports stable non-checkout
  paths through:
  - `VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH`
  - `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH`
- `tools/openclaw-harness/src/index.ts` already exposes durable installers:
  - `install-tavily`
  - `install-radarr`
  - `install-sonarr`
  - `install-chaptarr`
  - `sync-runtime-catalog`
- The current request trace only records `reasoning_chars` and `content_chars`
  in `provider_request_finished`, not the verbatim text.
- Additive VAD injection occurs in `inject_additive_runtime_guidance()` but
  currently has no explicit trace event showing whether VAD was injected or
  skipped.

## Web research

- DeepSeek documents that `max_tokens` covers the whole generated output,
  including chain-of-thought / reasoning. This means the raw
  `reasoning_content` text is part of the actual output stream and can be
  retained exactly once received. Source:
  [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode),
  [Create Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion/).
- DeepSeek JSON mode requires explicit prompt instruction and can still yield
  empty content if the model fails to emit the JSON object before the shared
  output budget is exhausted. Source:
  [JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/).

## GitHub research

- `openai/openai-agents-python` keeps tool-loop state explicit in CPU-side run
  control and retains model-emitted reasoning/message items as first-class
  artifacts rather than only logging aggregate token counts. Relevant files:
  - `openai/openai-agents-python/src/agents/run_internal/run_loop.py`
  - `openai/openai-agents-python/src/agents/items.py`
- `microsoft/semantic-kernel` connector code similarly favors explicit metadata
  and structured transport state rather than implicit hidden state. Relevant
  file:
  - `microsoft/semantic-kernel/python/semantic_kernel/connectors/ai/azure_ai_inference/services/azure_ai_inference_chat_completion.py`

## Implementation implications

- A durable host env file plus stable OpenClaw secrets/catalog paths is the
  least invasive way to make tool credentials survive rebuilds.
- Startup sync should reuse the existing OpenClaw CLI installers instead of
  inventing a second secret format.
- Request traces should retain exact `reasoning_content` and `content` on the
  already-bounded event registry so staged failures can be inspected verbatim.
- Guidance observability should be an explicit trace event near prompt assembly,
  not inferred indirectly from later provider behavior.
