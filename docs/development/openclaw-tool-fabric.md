# OpenClaw Tool Fabric

Vicuña owns cognition. OpenClaw owns capability delivery.

This integration keeps that separation by exposing OpenClaw-shaped capability
descriptors to the cognitive runtime while forcing execution back through an
exact server-side registry. The runtime can only select from installed tool
specs, and the server will only dispatch a command when all of these fields
match a registered capability:

- `tool_spec_index`
- `tool_kind`
- `capability_id`

The repo-local OpenClaw harness also validates the external selector pair:

- `tool_surface_id`
- `capability_id`

That makes hallucinated tool invocations fail closed on both sides of the
boundary.

For foreground ReAct execution, the server now treats the planner-emitted chat
tool name as the source of truth for the pending tool command. The hidden
preflight runner can still decide that the turn should enter tool mode, but it
does not get to hardwire a specific capability into the visible planner loop.
When the assistant emits a tool call, the server rebinds the pending runtime
command to that emitted tool before request installation and dispatch.

## Functional LoRAs

OpenClaw-backed tool usage still flows through Vicuña's existing
process-functional machinery. Stable `capability_id` and
`provenance_namespace` fields are carried in process signatures so existing and
new tools can accumulate their own tool/process/process-step specializations
without adding new selection code.

At runtime, self-model extension pressure now modulates the applied gain for
functional and process-functional attachments. Selection remains explicit and
ablation still wins: family disables, microphase disables, and stack disables
all suppress attachment before any self-model scaling is applied.

## Shared Context Memory

Foreground assistant replies, proactive emits, tool observations, and DMN
artifacts now share the same Vicuña self-state admission path. The server
admits emitted assistant text as normal system events instead of only noting an
emit counter, so later continuity questions can recover the same memory surface
that already held user and tool events.

Hard-memory result summaries also preserve top-hit titles and short content
snippets so "what have you been up to while I was away?" can recover a usable
narrative instead of only status metadata.

## Current Capabilities

On the current provider-only path, the live catalog is generated in
[`tools/openclaw-harness/src/catalog.ts`](../../tools/openclaw-harness/src/catalog.ts).

Current families include:

- hard memory
- Tavily web search
- narrowed Radarr, Sonarr, and Chaptarr media aliases
- `ongoing_tasks`
- `telegram_relay`

The current provider-only surface intentionally does not include `codex` or
`ask_with_options`.

## Runtime Flags

- `VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED=1`
  Enables catalog installation and strict dispatch resolution.
- `VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS=hard_memory_query,hard_memory_write`
  Optional comma-separated allowlist of registered built-ins.
- `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=/path/to/openclaw-catalog.json`
  Optional external catalog file merged with the built-ins.

If the fabric is enabled but no routable capabilities survive policy and local
tool availability checks, startup refuses to install the catalog.

If a host still carries the older builtin allowlist `hard_memory_query`, the
runtime now compatibility-upgrades that setting so `hard_memory_write` remains
available as part of the hard-memory tool layer instead of being silently
dropped.

At runtime, `llama-server` now logs the effective prerequisite state and the
installed capability set, for example:

- `bash=1 hard_memory=1`
- `catalog_path=/var/cache/vicuna/openclaw-catalog.json catalog_exists=1`
- `OpenClaw capability set: hard_memory_query=..., hard_memory_write=..., ...`

## Live Catalog Reload

When `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH` is set, the running runtime now
checks that file while idle and reloads the catalog if the file changes.

That means new supported tools can be introduced without restarting inference,
as long as they route through an already supported backend such as:

- `legacy_bash`
- `legacy_hard_memory`

The runtime still fails closed:

- selector matching remains exact on `tool_surface_id`, `capability_id`,
  `tool_spec_index`, and `tool_kind`
- unknown `dispatch_backend` values are rejected
- reload happens only while the server is idle, so existing active work does not
  get its tool indices reshuffled mid-turn

## OpenClaw Tool Secrets

OpenClaw-managed API keys for external tools should live in the OpenClaw layer,
not in `VICUNA_*` runtime env vars.

The harness now uses:

- runtime external catalog:
  `REPO_ROOT/.cache/vicuna/openclaw-catalog.json` by default, or
  `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH` when explicitly configured
- OpenClaw tool secrets:
  `REPO_ROOT/.cache/vicuna/openclaw-tool-secrets.json`

That lets key edits happen without restarting inference. The runtime only sees
the installed tool descriptor; the wrapper command reads the key from the
OpenClaw secrets file at execution time.

The live host execution policy for those wrappers still comes from the runtime
env, not from persisted runtime snapshots. `server_context` may restore
historical tool-config fields for compatibility, but it now reapplies the
current env-derived bash and hard-memory policy after restore so stale snapshot
state cannot silently disable Radarr, Sonarr, Chaptarr, or Tavily dispatch on
restart.

## Adding Another External Tool

For a tool that can route through an existing backend such as `legacy_bash`:

1. Add the external descriptor in
   [`tools/openclaw-harness/src/catalog.ts`](../../tools/openclaw-harness/src/catalog.ts).
2. Add any wrapper/config/secrets support in the harness under
   [`tools/openclaw-harness/src`](../../tools/openclaw-harness/src).
3. Emit or sync the external runtime catalog with:
   `node dist/index.js sync-runtime-catalog`
4. Add validation coverage in
   [`tools/openclaw-harness/test/catalog.test.ts`](../../tools/openclaw-harness/test/catalog.test.ts)
   and any affected runtime tests.

If a tool needs a new dispatch backend, then it is no longer external-only and
must add a runtime-side backend mapping in the server.

## Tavily Web Search

The Tavily tool is the reference external network-read capability.

Install or update it from the harness:

```bash
cd tools/openclaw-harness
npm run build
node dist/index.js install-tavily "$TAVILY_API_KEY"
```

That writes the OpenClaw secrets file, emits the external runtime catalog, and
lets the running runtime pick up the new capability on the next idle reload
cycle. When selected, the runtime dispatches the bounded wrapper command
`tools/openclaw-harness/bin/tavily-web-search`.

The same install path now applies to the LAN media tools. For example:

```bash
cd tools/openclaw-harness
npm run build
node dist/index.js install-radarr "$RADARR_API_KEY" "http://10.0.0.218:7878"
node dist/index.js install-sonarr "$SONARR_API_KEY" "http://10.0.0.218:8989"
node dist/index.js install-chaptarr "$CHAPTARR_API_KEY" "http://10.0.0.218:8789"
```

The runtime catalog keeps the Radarr, Sonarr, and Chaptarr media aliases
visible even when their API keys are not yet present. Wrapper execution then
fails with typed configuration or authorization errors instead of silently
removing those tools from the OpenClaw surface.

On the current LAN deployment, Chaptarr's broader discovery surface depends on
Hardcover being configured inside the container. Once the bearer token is
installed through `/api/v1/config/hardcover`, the exposed `chaptarr_search`
and `chaptarr_book_lookup` aliases can successfully drive `/api/v1/search` and
`/api/v1/book/lookup` in addition to `chaptarr_author_lookup`.

On Telegram-origin turns, the bridge prompt now explicitly tells the model to
use those exposed media tools when the user asks about managed media state.
Older assistant refusals in transcript history are treated as stale rather than
as a reason to avoid the currently available tool.

Telegram-origin active requests can also acknowledge immediately now. The
bridge first runs a lightweight acknowledgement-only inference request, emits
that model-generated acknowledgement right away, then starts the full staged
ReAct turn in the background. `server_context` later publishes a chat-scoped
Telegram outbox message when the final completion or terminal failure follow-up
is ready.

For system-service deployments, `tools/ops/install-vicuna-system-service.sh`
and `tools/ops/rebuild-vicuna-runtime.sh` now sync the runtime catalog to the
configured `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH` before service startup or
restart. This closes the drift where the harness had generated a Tavily catalog
under `REPO_ROOT/.cache/vicuna/` but the live runtime was loading
`/var/cache/vicuna/openclaw-catalog.json`.

For active engagement, the runtime now defaults into a ReAct-style foreground
loop when a live tool surface is available and the turn implies open-ended
investigation or fresh current information. The foreground transcript becomes:

- user turn
- assistant first-move planning step
- assistant typed `select_tool_family`
- assistant typed `select_method`
- assistant typed `emit_arguments`
- after each tool result, assistant typed `decide_after_tool`
- if answering or asking, assistant typed `emit_response`
- assistant tool call
- tool observation
- assistant follow-up planning or final answer

Those planner/tool spans are reinserted into the same prompt/KV path used for
normal foreground decoding, so they can influence continuation and later
prompt-eviction behavior instead of living only in side-channel artifacts.

The planner phase now sees the full currently installed tool surface instead of
an internally preselected singleton. Tool preference should emerge from the
canonical shared-context prompt reconstruction and learned LoRA state, not from server-side hardcoded
tool-specific routing.

When authoritative ReAct is active, tool use is now a staged runtime protocol
rather than a single freeform jump directly to tool arguments.

Stage 1 is tool-family selection. The model receives the visible tool families
for that turn and must emit only:

- `Action: select_tool`
- `{"tool_family_id":"..."}`

Stage 2 is method selection. The runtime looks up the selected family and feeds
back only the methods and descriptors for that family. The model must then
emit only:

- `Action: select_method`
- `{"method_name":"..."}`

Stage 3 is final argument emission. The runtime looks up the exact typed
contract for the selected tool family plus method and feeds back only that
method contract. The model must then emit only:

- `Action: act`
- one JSON object containing only the selected method arguments

The runtime, not the model, synthesizes the executable invocation from the
selected family, selected method, and validated argument JSON. Method-local
contracts preserve top-level field descriptions plus enum-constrained allowed
values for string arguments, and the server rejects payloads that use
unsupported enum values, undeclared fields, or mismatch the already selected
tool family and method before dispatch. For media aliases, the planner no
longer needs to emit a broad Servarr `action` selector; the runtime merges the
fixed downstream action for that alias before wrapper dispatch.

This same staged protocol now applies to every OpenClaw-backed tool, not only
Radarr, Sonarr, and Chaptarr. The runtime only dispatches after stage 3 has
resolved to a registered capability and the resulting command is installed as an
async tool job.

Every stage is also traceable in provenance:

- `react_stage_parse_failure` when staged JSON cannot be parsed or is rejected
- `react_stage_parsed` when the stage output is accepted
- `react_stage_transition` when the runtime advances from tool to method or
  method to final call
- `tool_call` and `tool_result` for the actual command lifecycle

Those `react_stage` events include the captured hidden reasoning, raw stage
payload, selected tool family, selected method, and the final accepted
argument JSON. The `tool_call` event then records the controller-synthesized
invocation payload that was actually dispatched. They are for operator
inspection only; user-visible outputs still exclude hidden reasoning.

If a selected tool path already has a learned functional/process-functional
adapter, that adapter is now scoped only to tool-call generation and argument
preparation. The runtime clears the tool-specific layer before the observation
is integrated, but preserves the planner-composition family across the wait and
resume boundary so the post-tool step stays inside the same planner loop until
final answer synthesis.

During tool-class selection and tool-argument preparation, the stack order is:

- base model plus temporal stack
- planner-composition family
- tool-selection family
- any matching tool/process-functional attachment

After the tool call leaves the model, the tool-specific layer is ablated and
the planner family remains. The planner family is only ablated when the runtime
commits to the final user-facing answer.

The invariant is simple: if a tool is not present in both registries, it is not
selectable and it is not dispatchable.

The runtime now enforces one more invariant for external catalog entries: a
descriptor must carry explicit cognitive eligibility flags, not just a backend
and schema. In practice that means external tools must set the relevant
`tool_flags` bits for the origins that should see them:

- `LLAMA_COG_TOOL_ACTIVE_ELIGIBLE` for normal active ReAct turns
- `LLAMA_COG_TOOL_DMN_ELIGIBLE` for DMN ReAct turns
- optional `LLAMA_COG_TOOL_SIMULATION_SAFE` and
  `LLAMA_COG_TOOL_REMEDIATION_SAFE` for narrower internal use

If an external catalog entry omits both active and DMN eligibility, the runtime
now rejects the catalog instead of silently installing a tool that the model
can never see.

## Provenance Retention

Structured provenance now rotates by hour and prunes old segments
automatically.

- the configured `VICUNA_PROVENANCE_LOG_PATH` is treated as the base path
- writes go to hourly segment files such as `runtime-provenance.20260323T17.jsonl`
- `VICUNA_PROVENANCE_RETENTION_HOURS` controls retention and defaults to `48`
- `/health` exposes both the configured base `path` and the current
  `active_path`, plus `retention_ms` and `prune_total`

This keeps the expanded staged-tool and hidden-reasoning trace surface bounded
without requiring external logrotate rules.

## Raw Exec Removal

The raw `exec` tool family is no longer part of the exposed OpenClaw surface.
Host-local bash execution still exists internally for typed wrappers such as
Servarr and Tavily, but planner-visible tool selection must now route through
those structured families instead of a free-form shell-command capability.
