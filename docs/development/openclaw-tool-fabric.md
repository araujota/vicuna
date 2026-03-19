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
[`/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp`](/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp)
and mirrored in
[`/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/catalog.ts`](/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/catalog.ts):

- `exec`
  `tool_surface_id=vicuna.exec.main`
  `capability_id=openclaw.exec.command`
- `hard_memory_query`
  `tool_surface_id=vicuna.memory.hard_query`
  `capability_id=openclaw.vicuna.hard_memory_query`

## Runtime Flags

- `VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED=1`
  Enables catalog installation and strict dispatch resolution.
- `VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS=exec,hard_memory_query`
  Optional comma-separated allowlist of registered built-ins.

If the fabric is enabled but no routable capabilities survive policy and local
tool availability checks, startup refuses to install the catalog.

## Adding Another Tool

1. Add one descriptor builder to the C++ registry in
   [`/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp`](/Users/tyleraraujo/vicuna/tools/server/server-openclaw-fabric.cpp).
2. Mirror the descriptor in the harness registry in
   [`/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/catalog.ts`](/Users/tyleraraujo/vicuna/tools/openclaw-harness/src/catalog.ts).
3. Provide an execution backend mapping in `server_openclaw_dispatch_backend`.
4. Add validation coverage in
   [`/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp`](/Users/tyleraraujo/vicuna/tests/test-openclaw-tool-fabric.cpp)
   and
   [`/Users/tyleraraujo/vicuna/tools/openclaw-harness/test/catalog.test.ts`](/Users/tyleraraujo/vicuna/tools/openclaw-harness/test/catalog.test.ts).

The invariant is simple: if a tool is not present in both registries, it is not
selectable and it is not dispatchable.
