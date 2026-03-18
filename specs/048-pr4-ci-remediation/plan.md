# Implementation Plan: PR #4 CI Remediation

**Branch**: `042-discovered-self-state-consolidation` | **Date**: 2026-03-17 | **Spec**: [/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/spec.md](/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/048-pr4-ci-remediation/spec.md`

## Summary

Repair the PR #4 merge blockers by fixing runtime snapshot persistence when optional LoRA subsystems are absent, refactoring the changed cognitive-loop and self-state paths so they no longer regress the repository lizard and clang-tidy baselines, and resolving the valid automated PR review findings while preserving explicit runtime policy.

## Technical Context

**Language/Version**: C++17 and Python 3.11  
**Primary Dependencies**: Existing server runtime persistence code, cognitive loop runtime, self-state runtime, Active LoRA manager, repository CI baseline scripts  
**Storage**: JSON runtime snapshot files and JSONL provenance output for server runtime tests  
**Testing**: `pytest` server tests, repository clang-tidy baseline script, repository lizard baseline script, targeted native builds where needed  
**Target Platform**: Linux CI and the existing CPU-managed server/runtime code paths  
**Project Type**: Native inference/runtime library with server integration and CI quality gates  
**Performance Goals**: No degraded runtime snapshot behavior, no added serving-stack or persistence overhead beyond optional-state checks, and reduced complexity in changed functions  
**Constraints**: Preserve explicit CPU-side policy, avoid weakening CI gates, avoid API ambiguity on new public C surfaces, and keep changes narrowly scoped to PR #4 regressions  
**Scale/Scope**: Targeted remediation across `src/`, `tools/server/`, `tests/`, and the new spec artifacts only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Runtime Policy**: Pass. Refactors will keep DMN, self-state, and persistence policy explicit in CPU-side code instead of pushing logic into opaque helpers.
- **Typed State**: Pass. API cleanup will preserve typed structs and may tighten passing semantics for large typed records.
- **Bounded Memory**: Pass. The persistence fix must treat optional archives as absent rather than forcing initialization or hidden fallback allocation.
- **Validation**: Pass. The work is gated by the same Python and baseline-check commands used in CI.
- **Documentation & Scope**: Pass. The remediation adds aligned Spec Kit artifacts and keeps code changes scoped to the failing checks and valid review findings.

## Project Structure

### Documentation (this feature)

```text
specs/048-pr4-ci-remediation/
├── plan.md
├── spec.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── llama-active-lora.cpp
├── llama-cognitive-loop.cpp
├── llama-context.cpp
└── llama-self-state.cpp

tests/
└── test-serving-lora-stack.cpp

tools/server/
├── server-context.cpp
└── tests/unit/test_basic.py
```

**Structure Decision**: The remediation stays inside the existing runtime, server, and test files implicated by the CI failures and review comments. No new subsystem or dependency is required.

## Phase 0: Research Conclusions

- PR #4 has three failing GitHub Actions jobs: `clang-tidy`, `lint-type-quality`, and `python-server-tests`.
- PR #4 has no Copilot review comments; the only review comments are from `github-advanced-security[bot]`.
- The Python failure is a real defect: server persistence attempts to query functional snapshot archives even when the Active LoRA manager is absent, causing runtime snapshot writes to fail.
- The clang-tidy failure is driven by new or regressed diagnostics in PR #4 code paths, especially `src/llama-active-lora.cpp`, `src/llama-cognitive-loop.cpp`, `src/llama-self-state.cpp`, and `tests/test-serving-lora-stack.cpp`.
- The lizard failure is driven by changed warning entries for `llama_cognitive_loop::dmn_tick`, `llama_cognitive_loop::active_loop_process`, `llama_self_state::update_expanded_model`, `server_context_impl::update_slots`, and large test entrypoints.
- The valid automated review findings overlap with the quality work: variable shadowing, large-object passing by value, and large under-documented functions.

## Implementation Design

### 1. Persistence Guarding

- Make server runtime snapshot export tolerate absent functional and process-functional archives.
- Serialize archive arrays only when the corresponding context queries are available and return valid state.
- Preserve the existing export/import behavior for initialized archives so persistence remains lossless when the subsystem is active.

### 2. Cognitive Loop Quality Refactor

- Split the newly enlarged `active_loop_process` and `dmn_tick` paths into focused helpers for:
  - runtime temporal candidate enumeration
  - functional/process-functional replay candidate generation
  - discrete fallback candidate generation
  - active-loop tracing and task parking integration
- Rename shadowing locals and add brief explanatory comments only where the policy would otherwise be opaque.

### 3. Self-State Quality Refactor

- Simplify nested conditional logic in discovered-state admission and expanded-model update paths.
- Break large heuristic blocks into helper functions that preserve explicit formula ownership close to the call site.

### 4. Public API and Test Cleanup

- Change large-object process-functional import surfaces to pass typed metadata by const pointer/reference instead of by value where the C API allows it.
- Add the direct includes needed by touched tests so include-cleaner no longer reports new regressions.

## Validation Strategy

- `python -m pytest -v tools/server/tests/unit/test_basic.py -k "runtime_snapshot_survives_restart or unified_provenance_repository_records_self_improvement_events"`
- `python .github/scripts/check_lizard_baseline.py --baseline .github/ci/lizard-baseline.txt`
- `python3 .github/scripts/check_clang_tidy_baseline.py --baseline .github/ci/clang-tidy-baseline.txt --build-dir build-clang-tidy --header-filter '^(src|include|common|ggml/(src|include)|tools|examples|tests)/' ...`
- Targeted native builds for touched code paths as needed to keep the compile and test loop tight

## Post-Design Constitution Check

- **Runtime Policy**: Still passes. Helper extraction will organize policy, not hide it.
- **Typed State**: Still passes. Public surfaces remain typed and become more explicit where large objects are passed.
- **Bounded Memory**: Still passes. The persistence fix handles absent state without allocating or mutating hidden runtime memory.
- **Validation**: Still passes. The exact CI checks that failed are part of the local verification plan.
- **Documentation & Scope**: Still passes. The artifact set now reflects the remediation work.
