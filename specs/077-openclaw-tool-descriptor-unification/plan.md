# Implementation Plan: OpenClaw Tool Descriptor Unification

**Branch**: `[070-truth-runtime-refactor]` | **Date**: 2026-03-22 | **Spec**: [/Users/tyleraraujo/vicuna/specs/077-openclaw-tool-descriptor-unification/spec.md](/Users/tyleraraujo/vicuna/specs/077-openclaw-tool-descriptor-unification/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/077-openclaw-tool-descriptor-unification/spec.md`

## Summary

Strengthen the OpenClaw capability contract so every exposed parameter requires a non-empty description, fill in the missing descriptions for builtin and external tools, and document that active and DMN ReAct continue to consume one authoritative tool fabric.

## Technical Context

**Language/Version**: C++17 and TypeScript  
**Primary Dependencies**: Shared OpenClaw descriptor contract, server OpenClaw fabric, harness runtime catalog  
**Storage**: Existing capability descriptors and runtime catalog JSON  
**Testing**: Native OpenClaw fabric tests plus TypeScript harness tests  
**Target Platform**: Local development and Linux host runtime  
**Project Type**: Native inference runtime and TypeScript tool harness  
**Constraints**: Do not introduce a second tool registry; preserve `server_openclaw_fabric` as the sole ReAct tool source

## Constitution Check

- **Runtime Policy**: Pass. The one-fabric rule remains explicit and inspectable.
- **Typed State**: Pass. The work strengthens an existing typed capability contract.
- **Bounded Memory**: Pass. No memory semantics change.
- **Validation**: Pass. Add contract-level tests in both native and TypeScript layers.
- **Documentation & Scope**: Pass. Update operator/developer docs to make the single-fabric rule explicit.

## Implementation Phases

### Phase 1: Contract Validation

- Add recursive parameter-description validation to the shared native OpenClaw descriptor checker.
- Add the equivalent validation to the TypeScript harness contract checker.

### Phase 2: Descriptor Completion

- Add parameter descriptions to every builtin tool in `server_openclaw_fabric`.
- Add parameter descriptions to every external tool emitted through the harness catalog.

### Phase 3: Verification and Docs

- Extend tests to fail on missing descriptions and to verify the builtin and external catalogs pass.
- Update docs to state that active and DMN ReAct consume only the OpenClaw fabric.
