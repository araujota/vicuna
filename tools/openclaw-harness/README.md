# OpenClaw Harness

This package defines the repo-local OpenClaw capability catalog contract that
Vicuña consumes.

Current scope:

- typed capability descriptors for migrated `exec` and `hard_memory_query`
- external runtime-catalog emission for OpenClaw-managed tools
- file-backed OpenClaw tool secrets under `.cache/vicuna`
- Tavily-backed `web_search` wrapper command
- Radarr-backed `radarr` wrapper command for movie-library inspection and explicit download-start flows
- Sonarr-backed `sonarr` wrapper command for series-library inspection and explicit download-start flows
- Chaptarr-backed `chaptarr` wrapper command for ebook-library inspection, Hardcover-backed search, and explicit ebook download-start flows
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

For authoritative ReAct, that contract reaches the model in two forms:

- the chat-tool JSON schema, which preserves the full parameter schema
- the planner-facing staged JSON controller contracts, which expose family
  selection, method selection, and method-local argument schemas one stage at
  a time

Example:

```bash
npm test
node dist/index.js catalog
node dist/index.js install-tavily "$TAVILY_API_KEY"
node dist/index.js install-radarr "$RADARR_API_KEY" "http://10.0.0.218:7878"
node dist/index.js install-sonarr "$SONARR_API_KEY" "http://10.0.0.218:8989"
node dist/index.js install-chaptarr "$CHAPTARR_API_KEY" "http://10.0.0.218:8789"
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

## Media Tools

The harness now emits planner-facing media capability aliases instead of only
one overloaded tool per service. The underlying wrappers are still the same
Radarr, Sonarr, and Chaptarr executables, but the runtime catalog now presents
narrower selectable tools such as:

- `radarr_inspect`, `radarr_queue`, `radarr_root_folders`,
  `radarr_quality_profiles`, `radarr_search`, `radarr_download_movie`,
  `radarr_delete_movie`
- `sonarr_inspect`, `sonarr_queue`, `sonarr_root_folders`,
  `sonarr_quality_profiles`, `sonarr_search`, `sonarr_download_series`,
  `sonarr_delete_series`
- `chaptarr_inspect`, `chaptarr_queue`, `chaptarr_root_folders`,
  `chaptarr_quality_profiles`, `chaptarr_metadata_profiles`,
  `chaptarr_search`, `chaptarr_author_lookup`, `chaptarr_book_lookup`,
  `chaptarr_download_author`, `chaptarr_download_book`,
  `chaptarr_delete_book`

Each alias exposes only the fields needed for that operation. Discovery aliases
such as `radarr_search`, `sonarr_search`, and `chaptarr_book_lookup` are
read-only and do not start downloads. Acquisition aliases such as
`radarr_download_movie`, `sonarr_download_series`, and
`chaptarr_download_book` are the methods intended to start the actual
acquisition workflow. For Chaptarr specifically, both the generic tracked-add
actions (`add_author`, `add_book`) and the download aliases are ebook-only, use
deployment-fixed profile defaults, repair existing tracked authors/books into
ebook-acquisition state, and then trigger Chaptarr's native search commands in
the same wrapper call. They start acquisition/search; they do not guarantee a
completed import. The runtime then
merges fixed/default arguments, such as the downstream `action`, before the
wrapper call is dispatched. This keeps the planner-facing contract smaller for
local models while preserving one authoritative wrapper implementation per
service.

The destructive delete aliases are tracked-item removals rather than soft
unmonitor or untrack operations:

- `radarr_delete_movie` resolves one tracked movie and calls
  `DELETE /api/v3/movie/{id}` with `deleteFiles=true` by default
- `sonarr_delete_series` resolves one tracked series and calls
  `DELETE /api/v3/series/{id}` with `deleteFiles=true` by default
- `chaptarr_delete_book` resolves one tracked book and calls
  `DELETE /api/v1/book/{id}` with `deleteFiles=true` by default

Those delete aliases fail closed on ambiguous title matches instead of
deleting an arbitrary tracked item.

In the current deployment, media add flows use deployment-fixed destination
roots rather than model-generated folder paths:

- Radarr adds into `/movies`
- Sonarr adds into `/tv`
- Chaptarr adds into `/books`

The current deployment also keeps Radarr and Sonarr quality selection on the
host side rather than the model side:

- Radarr add flows inject `qualityProfileId: 6` (`HD - 720p/1080p`) in the
  `POST /api/v3/movie` payload
- Sonarr add flows inject `qualityProfileId: 4` (`HD-1080p`) in the
  `POST /api/v3/series` payload

Those download aliases therefore do not expose `quality_profile_id` as a
model-facing argument.

Chaptarr download flows are intentionally ebook-only. The wrapper uses
Hardcover-backed search results for add-author and add-book so the emitted
write payloads carry V5-compatible identifiers and richer edition metadata
than the older lookup endpoints.

Chaptarr download flows also keep ebook profile selection on the host side rather
than the model side:

- Chaptarr add-author and add-book inject `qualityProfileId: 1` (`E-Book`)
- Chaptarr add-author and add-book inject `metadataProfileId: 2`
  (`Ebook Default`)
- Chaptarr add-author and add-book inject `ebookMetadataProfileId: 2`

Those Chaptarr download aliases therefore do not expose `quality_profile_id` or
`metadata_profile_id` as model-facing arguments.

All media aliases are emitted through the same external OpenClaw runtime
catalog as `web_search`; there is no second tool registry for media
management.

Media inspect, queue, and lookup wrappers intentionally emit compact typed
summaries instead of raw upstream payload dumps. This keeps grounded media
observations inside the runtime bash stdout budget so post-tool response
stages can still see complete title lists and queue state.

Secrets layout:

```json
{
  "tools": {
    "radarr": {
      "base_url": "http://10.0.0.218:7878",
      "api_key": "radarr-api-key"
    },
    "sonarr": {
      "base_url": "http://10.0.0.218:8989",
      "api_key": "sonarr-api-key"
    },
    "chaptarr": {
      "base_url": "http://10.0.0.218:8789",
      "api_key": "chaptarr-api-key"
    }
  }
}
```

If the API keys are missing, the tools still remain visible in the runtime
catalog, but each invocation returns a typed configuration/auth failure payload
instead of silently disappearing from the OpenClaw surface.
