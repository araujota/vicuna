# Implementation Plan: Radarr and Sonarr OpenClaw Tools

**Branch**: `[070-truth-runtime-refactor]` | **Date**: 2026-03-22 | **Spec**: [/Users/tyleraraujo/vicuna/specs/078-radarr-sonarr-openclaw/spec.md](/Users/tyleraraujo/vicuna/specs/078-radarr-sonarr-openclaw/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/078-radarr-sonarr-openclaw/spec.md`

## Summary

Add two new external OpenClaw capabilities, `radarr` and `sonarr`, backed by official `/api/v3` Servarr HTTP wrappers in the TypeScript harness. Keep one authoritative tool fabric by emitting both capabilities through the shared runtime catalog, dispatch them via the existing `legacy_bash` wrapper path, and target the NAS services on `10.0.0.218` by default while allowing secrets overrides for auth and base URLs.

## Technical Context

**Language/Version**: C++17 and TypeScript  
**Primary Dependencies**: `server_openclaw_fabric`, harness runtime catalog, Node `fetch`, official Radarr/Sonarr `/api/v3` APIs  
**Storage**: Existing OpenClaw secrets JSON and runtime catalog JSON  
**Testing**: TypeScript harness tests, native OpenClaw fabric tests, manual host rebuild verification  
**Target Platform**: Linux host runtime on the same LAN as the NAS  
**Project Type**: Native inference runtime plus external tool harness  
**Constraints**: Preserve one authoritative OpenClaw fabric, keep parameter descriptions complete, preserve typed tool observations, avoid requiring a second server-local tool registry

## Constitution Check

- **Runtime Policy**: Pass. Dispatch remains explicit in `server-context.cpp`; no hidden tool-routing registry is introduced.
- **Typed State**: Pass. The work extends typed capability descriptors, secrets, and structured wrapper payloads instead of bypassing them.
- **Bounded Memory**: Pass. No memory or context-window semantics change.
- **Validation**: Pass. Add harness tests for catalog/runtime behavior and wrapper shaping, plus host rebuild verification for live tool visibility.
- **Documentation & Scope**: Pass. Update harness docs and server developer docs to describe the new tools and secrets contract.

## Project Structure

### Documentation (this feature)

```text
specs/078-radarr-sonarr-openclaw/
├── spec.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── radarr-tool-schema.md
│   └── sonarr-tool-schema.md
└── tasks.md
```

### Source Code

```text
tools/openclaw-harness/
├── src/
│   ├── catalog.ts
│   ├── config.ts
│   ├── index.ts
│   ├── runtime-catalog.ts
│   ├── servarr.ts
│   ├── radarr.ts
│   └── sonarr.ts
├── bin/
│   ├── radarr-api
│   └── sonarr-api
└── test/
    ├── catalog.test.ts
    └── servarr.test.ts

tools/server/
├── server-context.cpp
├── README-dev.md
└── server-openclaw-fabric.cpp

tests/
└── test-openclaw-tool-fabric.cpp
```

**Structure Decision**: Implement the new media-service tools as external harness capabilities so they share the same external catalog path as `web_search`, while touching the server only where dispatch construction and documentation must recognize them.

## Implementation Phases

### Phase 1: Harness Capability and Secrets Model

- Extend the OpenClaw secrets/config model with Radarr and Sonarr service config.
- Add `radarr` and `sonarr` capability descriptors to the runtime catalog with full parameter descriptions.
- Ensure the runtime catalog always includes these services with the NAS defaults unless explicitly disabled.

### Phase 2: Servarr Wrapper Execution

- Build shared Servarr HTTP helpers plus service-specific wrappers.
- Implement read actions and add actions using official `/api/v3` endpoints and `X-Api-Key`.
- Return typed JSON observations with explicit error shapes for missing auth, HTTP failures, and validation failures.

### Phase 3: Server Dispatch, Tests, and Deployment

- Teach `server-context.cpp` to dispatch the new capabilities through wrapper scripts.
- Add or update native and TypeScript tests.
- Update docs, sync the runtime catalog, rebuild the host, and verify live capability visibility.
