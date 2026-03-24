# Implementation Plan: Authoritative ReAct Action Contract Guarantee

**Branch**: `[103-react-action-contract-guarantee]` | **Date**: 2026-03-24 | **Spec**: [/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/spec.md](/Users/tyleraraujo/vicuna/specs/103-react-action-contract-guarantee/spec.md)
**Input**: Feature specification from `/specs/103-react-action-contract-guarantee/spec.md`

## Summary

Harden staged authoritative ReAct control so the exact staged action contract becomes runtime-owned during parsing, malformed staged payloads stay inside retry and rewind, and non-staged fallback refuses to treat control-shaped JSON as user-visible prose. The key implementation lives in `tools/server/server-context.cpp`, with staged parser behavior continuing to rely on the strict OpenClaw JSON contracts in `tools/server/server-openclaw-fabric.cpp`.

## Technical Context

**Language/Version**: C++17 and repository-standard shell tooling  
**Primary Dependencies**: `llama-server` runtime, OpenClaw staged tool contracts, nlohmann JSON  
**Storage**: In-memory runtime task state plus existing runtime and Telegram persistence files  
**Testing**: `tests/test-cognitive-loop.cpp` and host runtime trace verification  
**Target Platform**: Linux host runtime and local development build  
**Project Type**: Native inference/runtime server  
**Performance Goals**: Preserve current staged-loop latency and retry behavior without adding new background services  
**Constraints**: Keep policy explicit in CPU control code, avoid hidden heuristics, do not leak malformed controller artifacts to Telegram  
**Scale/Scope**: Focused runtime hardening in existing authoritative ReAct control and retry paths

## Constitution Check

- **Runtime Policy**: Pass. The change stays in CPU-owned `server_context` control flow and existing OpenClaw contract parsing rather than model-only prompting.
- **Typed State**: Pass. The design reuses existing staged phase enums, retry counters, and parse/provenance structs. No new opaque persistence surface is required.
- **Bounded Memory**: Pass. No new memory growth path; the change only alters prefill composition and retry classification.
- **Validation**: Required. Add targeted regression coverage in `tests/test-cognitive-loop.cpp`, run the test binary, and perform a host trace for the Telegram reproduction.
- **Documentation & Scope**: Required. Update `tools/server/README-dev.md` and the feature artifacts under `specs/103-react-action-contract-guarantee/`.

## Project Structure

### Documentation (this feature)

```text
specs/103-react-action-contract-guarantee/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

### Source Code (repository root)

```text
tools/server/
├── README-dev.md
├── server-context.cpp
└── server-openclaw-fabric.cpp

tests/
└── test-cognitive-loop.cpp
```

**Structure Decision**: Keep all behavior in the existing server runtime control path. `server-context.cpp` owns prompt preparation, fallback parsing, retry, and terminal visibility policy. `server-openclaw-fabric.cpp` remains the strict staged JSON contract parser. `tests/test-cognitive-loop.cpp` carries regression coverage.

## Phase 0: Research and Design

- Confirm the current staged prompt path, grammar path, and parse failure retry path.
- Confirm the existing fallback leak path from malformed control JSON into visible terminal text.
- Design host-owned staged prefixes that pin the fixed action bytes while leaving only variable tails for model generation.

## Phase 1: Runtime Hardening

- Normalize missing staged action fields to the exact phase action during parser validation.
- Keep conflicting or otherwise malformed staged payloads in the existing retry and rewind machinery.
- Tighten non-staged fallback so control-shaped JSON cannot be misclassified as visible prose.
- Preserve retry and rewind behavior for staged parse failures without exposing internal details to the user.

## Phase 2: Validation and Docs

- Add targeted regression tests for host-owned prefixes, malformed staged payload retry, and control-shaped JSON rejection.
- Update developer documentation to describe host-owned staged action contracts and retry-only handling of malformed control.
- Run local tests and, if requested later, validate on host with the Telegram reproduction.
