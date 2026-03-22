# OpenClaw Harness

This package defines the repo-local OpenClaw capability catalog contract that
Vicuña consumes.

Current scope:

- typed capability descriptors for migrated `exec` and `hard_memory_query`
- external runtime-catalog emission for OpenClaw-managed tools
- file-backed OpenClaw tool secrets under `.cache/vicuna`
- Tavily-backed `web_search` wrapper command
- exact selector validation on `tool_surface_id` plus `capability_id`
- simple CLI entrypoint for emitting the catalog and validating invocations
- registry-style catalog construction so more tools can be added without
  changing the call contract
- explicit cognitive eligibility flags so external tools appear in the intended
  active and DMN ReAct loops

Example:

```bash
npm test
node dist/index.js catalog
node dist/index.js install-tavily "$TAVILY_API_KEY"
node dist/index.js sync-runtime-catalog
```

External descriptors must declare the same cognitive eligibility contract that
the runtime enforces for built-ins. At minimum, a live tool should set one or
both of:

- `LLAMA_COG_TOOL_ACTIVE_ELIGIBLE`
- `LLAMA_COG_TOOL_DMN_ELIGIBLE`

If an emitted external catalog entry omits both active and DMN eligibility, the
runtime now rejects that catalog instead of silently loading an unreachable
tool.
