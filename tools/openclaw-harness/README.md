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

Example:

```bash
npm test
node dist/index.js catalog
node dist/index.js install-tavily "$TAVILY_API_KEY"
node dist/index.js sync-runtime-catalog
```
