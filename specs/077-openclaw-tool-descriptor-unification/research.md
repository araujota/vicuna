# Research: OpenClaw Tool Descriptor Unification

## Local Audit

- The authoritative ReAct loop already receives tools from `server_openclaw_fabric::build_chat_tools(...)`, which copies each capability's `tool_name`, `description`, and `input_schema_json` into the chat-tool surface.
- Builtin tools are defined in `tools/server/server-openclaw-fabric.cpp`.
- External tools are emitted through the TypeScript OpenClaw harness in `tools/openclaw-harness/src/catalog.ts`.
- Those two sources already share the same `openclaw_tool_capability_descriptor` shape, but most schemas still expose bare parameter types without descriptions.

## Tool-Surface Findings

- `exec` now has parameter descriptions in both the server builtin descriptor and the harness catalog.
- `hard_memory_query`, `hard_memory_write`, `codex`, `telegram_relay`, `ask_with_options`, and `web_search` still expose underspecified parameters.
- The server-side and TypeScript-side capability validators currently check identity fields, but they do not require schema descriptions.

## GitHub and Web Research

- `openai/openai-agents-python` uses schema-bearing tool definitions and relies on structured tool metadata as part of the tool contract, which supports treating parameter descriptions as first-class rather than optional.
- Anthropic's official tool-use guidance says tool descriptions should be detailed and should explain what the tool is for and how it should be used: [Anthropic tool use guide](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use).
- That guidance aligns with Vicuña's architecture: if tool choice belongs to ReAct, then schema semantics must be explicit in the one authoritative tool fabric rather than hidden in side prompts or alternate registries.

## Design Conclusions

- The repo does not need a second tool system; it needs the existing OpenClaw fabric tightened so underspecified schemas are invalid.
- The correct fix is:
  - add recursive schema-description validation in the shared native OpenClaw descriptor validator
  - add the same validation in the TypeScript harness contract checker
  - fill in descriptions for every shipped builtin and external tool parameter
  - document that active and DMN ReAct consume one authoritative tool fabric only
