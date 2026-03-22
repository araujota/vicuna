# Implementation Plan: Tavily Source-First Quality

**Branch**: `[070-truth-runtime-refactor]` | **Date**: 2026-03-22 | **Spec**: [/Users/tyleraraujo/vicuna/specs/075-tavily-source-first-quality/spec.md](/Users/tyleraraujo/vicuna/specs/075-tavily-source-first-quality/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/075-tavily-source-first-quality/spec.md`

## Summary

Refactor the Tavily-backed `web_search` tool into a source-first evidence retriever. Expand the runtime schema to expose better Tavily controls, stop requesting provider-generated answers by default, request richer content chunks, and keep the result shape bounded and inspectable for ReAct synthesis.

## Technical Context

**Language/Version**: TypeScript (Node 22 tool wrapper) plus existing C++ server dispatch/runtime schema  
**Primary Dependencies**: OpenClaw harness, runtime catalog, server dispatch, Tavily Search API  
**Storage**: File-backed OpenClaw secrets and generated runtime catalog  
**Testing**: Node test suite for OpenClaw harness plus targeted runtime-schema validation  
**Target Platform**: Local dev environment and host runtime rebuild  
**Project Type**: Tool-wrapper and server-runtime integration  
**Performance Goals**: Preserve low-latency web search while improving evidence quality and reducing hallucinated source synthesis  
**Constraints**: Keep output bounded, inspectable, and compatible with authoritative ReAct tool results

## Constitution Check

- **Runtime Policy**: Pass. Search quality defaults remain explicit and inspectable.
- **Typed State**: Pass. The runtime catalog schema and C++ dispatch remain typed.
- **Bounded Memory**: Pass. Output stays bounded to capped results and excerpt lengths.
- **Validation**: Pass. Add Node regression tests for schema and request normalization.
- **Documentation & Scope**: Pass. Update harness/server docs with the new source-first policy.

## Implementation Phases

### Phase 1: Wrapper Hardening

- Add explicit Tavily request normalization helpers.
- Disable provider-generated answers by default.
- Request richer source content and enforce a quality floor for result counts.

### Phase 2: Runtime Schema and Dispatch

- Expand the external catalog schema for `web_search`.
- Pass the new supported parameters through the C++ dispatch layer.

### Phase 3: Validation and Docs

- Add Node tests for request normalization and schema exposure.
- Update harness/server docs.

