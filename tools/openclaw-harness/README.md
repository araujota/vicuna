# OpenClaw Harness

This package defines the repo-local OpenClaw capability catalog contract that
Vicuña consumes.

Current scope:

- typed capability descriptors for the current provider-only tool surface
- external runtime-catalog emission for OpenClaw-managed tools
- file-backed OpenClaw tool secrets under `.cache/vicuna` by default, or a
  stable host path such as `/etc/vicuna/openclaw-tool-secrets.json` when
  `VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH` is set
- Tavily-backed `web_search` wrapper command
- Radarr-backed `radarr` wrapper command for movie-library inspection and explicit download-start flows
- Sonarr-backed `sonarr` wrapper command for series-library inspection and explicit download-start flows
- Chaptarr-backed `chaptarr` wrapper command for ebook-library inspection, Hardcover-backed search, and explicit ebook download-start flows
- cron-backed `ongoing_tasks` wrapper command for recurring task CRUD and scheduled execution
- local-filesystem-backed `parsed_documents_search_chunks` wrapper command for retrieval over Docling-parsed Telegram uploads
- provider-backed `telegram_relay` wrapper command for direct user-facing follow-up messages
- bounded `host_shell` wrapper command for last-resort host-side file and shell manipulation inside a dedicated workspace
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

The retained Telegram bridge now consumes the runtime catalog directly. For
live turns it injects every installed provider-tool definition from the
authoritative runtime catalog into `tools[]`, then lets the server derive the
family and method stages from those direct tool definitions. That means future
OpenClaw capabilities must provide all three metadata layers above; otherwise
they will not participate cleanly in the staged controller.

Runtime catalog loading is explicit:

- `node dist/index.js runtime-catalog` returns the authoritative runtime
  catalog, preferring the persisted catalog file when present
- `node dist/index.js runtime-tools` converts that catalog into provider-ready
  direct tool definitions with staged metadata attached
- `node dist/index.js invoke-runtime --tool-name=... --arguments-base64=...`
  executes one direct provider tool by `tool_name`

The runtime catalog now also exposes `host_shell` as a last-resort fallback.
Its routing intent is explicit:

- prefer `media_read`, `media_download`, `media_delete`, `web_search`,
  `hard_memory_read`, `hard_memory_write`, `ongoing_task_create`, and
  `ongoing_task_delete` whenever those specialized tools fit the task
- use `host_shell` only when the runtime genuinely needs direct host-side shell
  or file manipulation that those tools do not provide

`host_shell` executes one bounded `bash -lc` command from a dedicated workspace
root and returns a structured JSON observation envelope instead of a raw shell
dump. The envelope summarizes:

- command status and duration
- bounded stdout/stderr previews with type classification such as `json`,
  `path`, `path_list`, `text`, or `binary`
- created, modified, and deleted workspace paths detected before and after the
  command

Workspace-root policy:

- host deployments should set `VICUNA_HOST_SHELL_ROOT=/home/vicuna/home`
- local/dev runs fall back to `.cache/vicuna/host-shell-home` when the host
  path does not exist

Example direct invocation:

```bash
node dist/index.js invoke-runtime \
  --tool-name=host_shell \
  --arguments-base64="$(printf '%s' '{"command":"pwd","purpose":"Inspect the current host-shell workspace."}' | base64)"
```

When a persisted runtime catalog file exists, it is authoritative for live
turns. This allows operators to expose a richer installed tool set than the
repo-default narrowed fallback catalog without changing the bridge contract.
For host deployments, pair that with a stable catalog path such as
`/var/lib/vicuna/openclaw-catalog.json` via
`VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH` so rebuilds and branch switches do
not discard tool availability.

The wrapper binaries under `tools/openclaw-harness/bin/` honor an explicit
`--secrets-path=...` argument first, and otherwise default to
`VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH` before falling back to the repo-local
`.cache/vicuna/openclaw-tool-secrets.json`. Host services should still set the
environment variable explicitly.

## Markdown Hard Memory

`hard_memory_read` and `hard_memory_write` now use a local markdown store
rather than a remote memory service. The source of truth is one `.md` file per
durable memory primitive.

- host deployments default to `VICUNA_HARD_MEMORY_DIR=/home/vicuna/home/memories`
- local/dev runs fall back to `.cache/vicuna/host-shell-home/memories`
- each keyed memory updates the same markdown file over time
- retrieval stays explicit and local over markdown metadata plus body content

## Markdown Skills

`skill_read` and `skill_create` use the same local-only markdown pattern.

- host deployments default to `VICUNA_SKILLS_DIR=/home/vicuna/home/skills`
- local/dev runs fall back to `.cache/vicuna/host-shell-home/skills`
- `skill_read` returns one exact markdown skill file by normalized name
- `skill_create` writes one markdown skill file, but the server prompt policy
  only authorizes it when the user directly asks to create or update a skill in
  the current conversation
- skill bodies are never auto-injected; the provider server advertises only the
  available file names under the `SKILLS:` prompt section and requires explicit
  reads for file contents

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

For `chaptarr_download_book`, the wrapper now treats `GET /api/v1/search` as a
mixed entity surface, not a book-only lookup. If the mixed search response is
author-first or otherwise lacks a usable book record, the wrapper falls back to
`GET /api/v1/book/lookup` and still keeps ordinary tracked-book or new-book
requests on Chaptarr's explicit `BookSearch` path. It no longer treats a
successful `book/lookup` resolution as an implicit signal to divert into the
legal importer path.

The wrapper also no longer rejects a candidate as audiobook-only just because
the upstream lookup/search payload defaulted `mediaType` or `lastSelectedMediaType`
to `audiobook`. In the current Chaptarr build those flags can appear on plain
book candidates. The wrapper now requires concrete audiobook signals such as
audio file formats, audiobook media types on editions/files, or narrator data
before treating a candidate as audiobook-only.

Chaptarr download flows also keep ebook profile selection on the host side rather
than the model side:

- Chaptarr add-author and add-book inject `qualityProfileId: 1` (`E-Book`)
- Chaptarr add-author and add-book inject `metadataProfileId: 2`
  (`Ebook Default`)
- Chaptarr add-author and add-book inject `ebookMetadataProfileId: 2`

Those Chaptarr download aliases therefore do not expose `quality_profile_id` or
`metadata_profile_id` as model-facing arguments.

When diagnosing Chaptarr ebook acquisition problems on a host, the fastest
useful probes are:

- `GET /api/v1/search?term=...&provider=hardcover`
- `GET /api/v1/book/lookup?term=...`
- `POST /api/v1/command {"name":"BookSearch","bookIds":[...]}`
- `/logfile/chaptarr.txt`

That sequence distinguishes:

- true lookup misses
- mixed search-response routing problems
- indexer searches that run but finish with all releases filtered

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
fields needed for that one action. The durable state is stored locally under
`VICUNA_ONGOING_TASKS_DIR`, while the actual scheduler is the `vicuna` user's
crontab. That keeps create/edit/delete semantics deterministic without a
server-owned idle polling registry.

The wrapper keeps explicit local task metadata and returns only compact
summaries:

- `task_id`
- `task_text`
- `frequency`
- `last_done_at`
- `next_due_at`
- `due_now`
- `active`

The retained host runner loads the canonical env, acquires a per-task lock, and
posts the stored task text back to the live runtime as a `system` message.

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

This wrapper searches only locally stored parsed-document chunk bundles derived
from Telegram-uploaded files. The current contract is intentionally narrow:

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
      "task_dir": "/var/lib/vicuna/ongoing-tasks",
      "runner_script": "/home/vicuna/Projects/vicuna/tools/ops/run-ongoing-task-cron.sh",
      "runtime_url": "http://127.0.0.1:8080/v1/chat/completions",
      "runtime_model": "deepseek-chat",
      "host_user": "vicuna"
    },
    "parsed_documents": {
      "docs_dir": "/home/vicuna/home/docs",
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
standard `VICUNA_ONGOING_TASKS_*` environment variables as fallbacks.

The `parsed_documents` wrapper accepts either the secrets values above or the
standard `VICUNA_DOCS_DIR` environment variable as a fallback.

The `telegram_relay` wrapper accepts either the secrets values above or the
standard `TELEGRAM_BRIDGE_VICUNA_BASE_URL`, `VICUNA_API_KEY`, and
`TELEGRAM_DEFAULT_CHAT_SCOPE` environment variables as fallbacks.
