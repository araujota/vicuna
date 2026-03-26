# Implementation Plan: Host Secrets And Verbatim Stage Traces

**Branch**: `codex/129-host-secrets-reasoning` | **Date**: 2026-03-26 | **Spec**: [/Users/tyleraraujo/vicuna/specs/129-host-secrets-reasoning/spec.md](/Users/tyleraraujo/vicuna/specs/129-host-secrets-reasoning/spec.md)
**Input**: Feature specification from `/specs/129-host-secrets-reasoning/spec.md`

## Summary

Make host tool credentials rebuild-safe by loading one durable host env file and
syncing media/Tavily/memory settings into stable OpenClaw secrets/catalog
paths, then expand request tracing so staged provider turns retain verbatim
reasoning/content and explicit additive VAD/heuristic guidance events.

## Technical Context

**Language/Version**: C++17 runtime, Bash startup scripts, Node 20 OpenClaw CLI, Python 3 pytest  
**Primary Dependencies**: `nlohmann::json`, existing DeepSeek adapter, existing OpenClaw CLI installers  
**Storage**: `/etc/vicuna/vicuna.env` plus stable OpenClaw secrets/catalog paths outside the checkout  
**Testing**: runtime build, `tools/server/tests/unit/test_deepseek_provider.py`, targeted script/OpenClaw tests if needed  
**Target Platform**: provider-first runtime on Linux/macOS, deployed host on Linux user services  
**Project Type**: native HTTP service with bridge transport and startup scripts  
**Performance Goals**: preserve current latency profile while increasing debuggability; no unbounded trace growth  
**Constraints**: keep runtime behavior explicit and inspectable, preserve existing staged loop semantics, avoid checkout-bound secret state  
**Scale/Scope**: startup env loading, OpenClaw secret sync, request-trace payload enrichment, docs/tests, host deployment follow-up

## Research Notes

- DeepSeek returns `reasoning_content` as part of the generated output, and the
  same shared `max_tokens` budget governs reasoning plus visible content.
- The current host user services do not use `EnvironmentFile=`, so relying on
  `/etc/vicuna/vicuna.env` only in system-service templates is insufficient.
- OpenClaw already supports configurable stable secrets/catalog paths and
  installer commands that can populate them from API keys.

## Constitution Check

- **Runtime Policy**: Pass. VAD/guidance visibility and request-trace retention
  stay explicit in CPU-side code.
- **Typed State**: Pass. Trace/event payloads remain JSON objects with explicit
  fields; no hidden policy is added.
- **Bounded Memory/Performance**: Pass with care. Request traces remain bounded
  by the existing ring buffer; only event payload richness increases.
- **Validation**: Pass. Tests and docs will be updated, and host deployment will
  validate the persistent env path.
- **Documentation & Scope**: Pass. Operator docs will state exactly which env
  file and OpenClaw paths are authoritative on host deployments.

## Project Structure

### Documentation (this feature)

```text
specs/129-host-secrets-reasoning/
├── spec.md
├── research.md
├── plan.md
└── tasks.md
```

### Source Code

```text
tools/server/
├── server.cpp
├── server-deepseek.cpp
├── README.md
├── README-dev.md
└── tests/unit/test_deepseek_provider.py

tools/ops/
├── runtime-env.sh
├── run-vicuna-runtime.sh
├── run-telegram-bridge.sh
└── install-vicuna-system-service.sh

tools/openclaw-harness/
└── README.md
```

## Implementation Strategy

1. Extend startup env loading so runtime and bridge always source a durable host
   env file when present.
2. Add one startup sync path that writes media/Tavily/memory settings into
   stable OpenClaw secrets/catalog locations outside the checkout.
3. Enrich provider-finished trace events with verbatim `reasoning_content` and
   visible `content`.
4. Add explicit runtime-guidance trace events for VAD/heuristic injection and
   skip reasons.
5. Update tests and docs, then deploy to the host and populate the durable env
   plus durable OpenClaw secrets.

## Complexity Tracking

No constitution violations or justified complexity exceptions.
