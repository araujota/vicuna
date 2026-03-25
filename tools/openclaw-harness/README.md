# OpenClaw Harness

This package defines the repo-local OpenClaw capability catalog contract that
Vicuña consumes.

Current scope:

- typed capability descriptors for the current provider-only tool surface
- external runtime-catalog emission for OpenClaw-managed tools
- file-backed OpenClaw tool secrets under `.cache/vicuna`
- Tavily-backed `web_search` wrapper command
- Radarr-backed `radarr` wrapper command for movie-library inspection and explicit download-start flows
- Sonarr-backed `sonarr` wrapper command for series-library inspection and explicit download-start flows
- Chaptarr-backed `chaptarr` wrapper command for ebook-library inspection, Hardcover-backed search, and explicit ebook download-start flows
- hard-memory-backed `ongoing_tasks` wrapper command for recurring task CRUD and due polling
- hard-memory-backed `parsed_documents_search_chunks` wrapper command for semantic retrieval over Docling-parsed Telegram uploads
- provider-backed `telegram_relay` wrapper command for direct user-facing follow-up messages
- exact selector validation on `tool_surface_id` plus `capability_id`
- simple CLI entrypoint for emitting the catalog and validating invocations
- registry-style catalog construction so more tools can be added without
  changing the call contract
- explicit cognitive eligibility flags so external tools appear in the intended
  active and DMN ReAct loops

The same rule applies to every OpenClaw capability in the harness: exposed
parameters must carry descriptions, including nested object fields and array
item surfaces. The harness and the server fabric share the same capability
contract so ReAct sees one authoritative tool system rather than competing
descriptor registries.

The current server-side staged tool controller also expects every exposed
capability to support three metadata layers:

- family metadata:
  - `tool_family_id`
  - `tool_family_name`
  - `tool_family_description`
- method metadata:
  - `method_name`
  - `method_description`
- contract metadata:
  - `input_schema_json` with descriptions on every nested field and array item

If a capability omits those layers, it can still exist in the catalog, but it
is a poor fit for staged family -> method -> payload prompting and may be
excluded from that higher-level controller.

For provider-backed execution, that contract primarily reaches the model as
chat-tool JSON schema plus the staged family/method/contract surfaces used by
the retained provider server. The harness keeps the argument surface narrow so
the provider can select the right tool without hidden tool-selection policy.

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
one overloaded tool per service. The runtime catalog presents only the narrowed
current aliases:

- `radarr_list_downloaded_movies`, `radarr_download_movie`,
  `radarr_delete_movies`
- `sonarr_list_downloaded_series`, `sonarr_download_series`,
  `sonarr_delete_series`
- `chaptarr_list_downloaded_books`, `chaptarr_download_book`,
  `chaptarr_delete_books`

Each alias exposes only the fields needed for that operation. List aliases are
read-only and return compact downloaded-item summaries. Acquisition aliases
such as `radarr_download_movie`, `sonarr_download_series`, and
`chaptarr_download_book` are the methods intended to start the actual
acquisition workflow. The runtime merges fixed/default arguments, such as the
downstream `action`, before the wrapper call is dispatched. This keeps the
planner-facing contract smaller while preserving one authoritative wrapper
implementation per service.

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

## Ongoing Tasks

The harness now emits a narrow `ongoing_tasks` family that follows the same
planner-facing alias pattern as the media tools:

- `ongoing_tasks_create`
- `ongoing_tasks_get`
- `ongoing_tasks_get_due`
- `ongoing_tasks_edit`
- `ongoing_tasks_delete`
- `ongoing_tasks_complete`

These aliases all route through one wrapper, but each alias exposes only the
fields needed for that one action. The durable state is stored as one
hard-memory-backed registry document keyed by a stable semantic key. That
keeps create/edit/delete semantics deterministic even though the hard-memory
backend itself is append/update oriented.

The wrapper computes due state locally and returns only compact summaries:

- `task_id`
- `task_text`
- `frequency`
- `last_done_at`
- `next_due_at`
- `due_now`
- `active`

It does not echo raw `/v4/profile` or `/v4/memories` payloads back into the
system loop.

## Telegram Relay

The harness also emits one provider-only `telegram_relay` capability for direct
user-facing follow-up messages.

- it writes one retained outbox item into the current provider server
- the server exposes that item through `GET /v1/telegram/outbox` for the bridge
- plain text is still accepted for simple follow-up messages
- richer calls can send one structured Telegram Bot API request with:
  - `method`
  - `payload`
- the relay keeps `chat_scope` and reply anchoring outside that structured
  payload so routing policy remains explicit in local code
- the wrapper returns only:
  - `sequence_number`
  - `chat_scope`
  - `deduplicated`

The current tool surface intentionally does not include `ask_with_options` or
`codex`, and it still uses an explicit allowlist of outbound Telegram send
methods instead of exposing arbitrary Bot API calls.

## Parsed Documents

The harness now emits one compact parsed-document retrieval capability:

- `parsed_documents_search_chunks`

This wrapper searches only stored parsed-document chunk memories derived from
Telegram-uploaded files. The current contract is intentionally narrow:

- required:
  - `query`
- optional:
  - `limit`
  - `threshold`

The wrapper applies explicit threshold policy in local code:

- short ambiguous queries use a stricter default threshold than longer
  natural-language queries
- weak matches are filtered locally even after the backend search returns
- responses include only:
  - `document_title`
  - `chunk_text`
  - `similarity`
  - `chunk_index`
  - `link_key`

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
      "api_key": "chaptarr-api-key",
      "legal_importer_url": "http://10.0.0.218:5050",
      "legal_importer_wait_ms": 20000,
      "legal_importer_poll_ms": 2000
    },
    "ongoing_tasks": {
      "base_url": "https://api.supermemory.ai",
      "auth_token": "supermemory-api-key",
      "container_tag": "vicuna",
      "runtime_identity": "vicuna",
      "registry_key": "ongoing-tasks-registry",
      "registry_title": "Ongoing task registry",
      "query_threshold": 0
    },
    "parsed_documents": {
      "base_url": "https://api.supermemory.ai",
      "auth_token": "supermemory-api-key",
      "container_tag": "vicuna-telegram-documents",
      "runtime_identity": "vicuna",
      "default_threshold": 0.58,
      "short_query_threshold": 0.68,
      "max_results": 5
    },
    "telegram_relay": {
      "base_url": "http://127.0.0.1:8080",
      "auth_token": "vicuna-bearer-token",
      "default_chat_scope": "7502424413"
    }
  }
}
```

For Chaptarr, `legal_importer_url` is optional. If it is omitted, the harness
derives a default by taking the configured Chaptarr host and substituting port
`5000`, which matches the image's internal listener. The current NAS deployment
overrides that to `http://10.0.0.218:5050` because Synology host port `5000`
was not reusable cleanly. The wait and poll values are also optional and
default to `120000` ms and `5000` ms.

If the API keys are missing, the tools still remain visible in the runtime
catalog, but each invocation returns a typed configuration/auth failure payload
instead of silently disappearing from the OpenClaw surface.

The `ongoing_tasks` wrapper accepts either the secrets values above or the
standard `SUPERMEMORY_BASE_URL` / `SUPERMEMORY_API_KEY` environment variables
as a fallback for hard-memory access.

The `parsed_documents` wrapper accepts either the secrets values above or the
standard `SUPERMEMORY_BASE_URL`, `SUPERMEMORY_API_KEY`, and
`TELEGRAM_BRIDGE_DOCUMENT_CONTAINER_TAG` environment variables as fallbacks.

The `telegram_relay` wrapper accepts either the secrets values above or the
standard `TELEGRAM_BRIDGE_VICUNA_BASE_URL`, `VICUNA_API_KEY`, and
`TELEGRAM_DEFAULT_CHAT_SCOPE` environment variables as fallbacks.
