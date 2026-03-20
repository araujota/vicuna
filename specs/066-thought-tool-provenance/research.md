# Research: Full Thought and Tool Provenance Logging

## Local Code Findings

- `tools/server/server-context.cpp` currently appends three main provenance
  event kinds: `active_loop`, `tool_result`, and `dmn_tick`.
- `active_trace_summary_to_json(...)` and `dmn_trace_summary_to_json(...)`
  currently expose only summary fields and omit plan steps, candidate arrays,
  and active planner narration text.
- Exact tool requests are configured server-side through typed request structs,
  but provenance currently records only post-execution result payloads.
- Active planner narration already exists as parsed text in
  `server_task.react_last_planner_reasoning` and is also admitted into the
  cognitive loop via `llama_cognitive_active_planner_reasoning_note(...)`.
- Exact active tool XML is already preserved as
  `server_task.react_last_tool_xml_payload` and routed into the runtime via
  `llama_cognitive_active_tool_emission_note(...)`.
- DMN narration already exists in the translated prompt revision surface:
  `llama_dmn_prompt_revision.rendered_prompt`.

## GitHub History Findings

- Commit [44130a61](https://github.com/araujota/vicuna/commit/44130a61da4a31276f5f70b74d0ea575e1d8197e)
  introduced planner-first runtime and tool persistence updates, including the
  current provenance repository direction and active planner reasoning plumbing.
- Commit [1251628c](https://github.com/araujota/vicuna/commit/1251628c3f4828b3f8e39706097556a32096e002)
  expanded server/runtime observability, proving the repo already treats
  provenance as the canonical append-only inspection surface.

## External Context

- OpenTelemetry’s current logs semantic conventions emphasize preserving the
  original log record and using structured attributes for correlation, which
  supports logging exact request payloads as structured provenance rather than
  flattening them into summary strings.
  Source: [OpenTelemetry general logs attributes](https://opentelemetry.io/docs/specs/semconv/general/logs/)

## Design Conclusions

- The unified provenance repository should remain the canonical location for
  complete narration and exact tool payload capture.
- Journal lines should remain concise breadcrumbs; the structured JSONL stream
  should hold the large narration/payload bodies.
- The implementation should enrich existing provenance events and add one new
  `tool_call` event kind rather than inventing a parallel logging subsystem.
