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

## Built-In Capabilities

Current built-ins are registered in
[`tools/server/server-openclaw-fabric.cpp`](../../tools/server/server-openclaw-fabric.cpp)
and mirrored in
[`tools/openclaw-harness/src/catalog.ts`](../../tools/openclaw-harness/src/catalog.ts):

- `exec`
  `tool_surface_id=vicuna.exec.main`
  `capability_id=openclaw.exec.command`
- `hard_memory_query`
  `tool_surface_id=vicuna.memory.hard_query`
  `capability_id=openclaw.vicuna.hard_memory_query`
- `hard_memory_write`
  `tool_surface_id=vicuna.memory.hard_write`
  `capability_id=openclaw.vicuna.hard_memory_write`
- `codex`
  `tool_surface_id=vicuna.codex.main`
  `capability_id=openclaw.vicuna.codex_cli`

## Runtime Flags

- `VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED=1`
  Enables catalog installation and strict dispatch resolution.
- `VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS=exec,hard_memory_query`
  Optional comma-separated allowlist of registered built-ins.
- `VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH=/path/to/openclaw-catalog.json`
  Optional external catalog file merged with the built-ins.

If the fabric is enabled but no routable capabilities survive policy and local
tool availability checks, startup refuses to install the catalog.

At runtime, `llama-server` now logs the effective prerequisite state and the
installed capability set, for example:

- `bash=1 hard_memory=1 codex=1`
- `catalog_path=/var/cache/vicuna/openclaw-catalog.json catalog_exists=1`
- `OpenClaw capability set: exec=..., hard_memory_query=..., ...`

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
- assistant tool call
- tool observation
- assistant follow-up planning or final answer

Those planner/tool spans are reinserted into the same prompt/KV path used for
normal foreground decoding, so they can influence continuation and later
prompt-eviction behavior instead of living only in side-channel artifacts.

The planner phase now sees the full currently installed tool surface instead of
an internally preselected singleton. Tool preference should emerge from the
planner transcript and learned LoRA state, not from server-side hardcoded
tool-specific routing.

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

## Exec Policy

`exec` remains intentionally narrower than a shell script surface. Planner XML
for `exec.command` must describe a single bounded invocation only. The server
still rejects shell metacharacters such as pipes, redirects, chaining, and
substitution, and it now preserves the preflight-safe command if planner XML
tries to override that request with unsafe shell syntax.
