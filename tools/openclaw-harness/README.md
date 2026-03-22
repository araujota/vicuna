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

The `exec` capability descriptor is meant to be semantically rich, not just
syntactically valid. It should tell the ReAct loop that `exec` is the bounded
shell tool for host-local observation and action, such as filesystem state,
current working directory, repository state, environment state, running
processes, and direct command output.

The same rule applies to every OpenClaw capability in the harness: exposed
parameters must carry descriptions, including nested object fields and array
item surfaces. The harness and the server fabric share the same capability
contract so ReAct sees one authoritative tool system rather than competing
descriptor registries.

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

## Tavily Search Policy

The Tavily-backed `web_search` tool is intentionally source-first.

- the wrapper does not request Tavily's provider-generated `answer` field by
  default
- the wrapper requests richer source evidence using raw-content-compatible
  retrieval and bounded multi-chunk excerpts
- the wrapper maintains a quality floor on `max_results` so a weak single hit
  does not dominate the observation by default
- the runtime-facing schema exposes `time_range`, `include_domains`,
  `exclude_domains`, `country`, `search_depth`, `topic`, and `max_results` so
  ReAct can tune retrieval explicitly when needed

The intended use is: retrieve evidence here, then let the authoritative ReAct
loop synthesize from the returned URLs and excerpts.
