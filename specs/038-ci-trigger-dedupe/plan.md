# Implementation Plan: CI Trigger Dedupe

**Branch**: `[023-adam-runtime-updates]` | **Date**: 2026-03-14 | **Spec**: [/Users/tyleraraujo/vicuna/specs/038-ci-trigger-dedupe/spec.md](/Users/tyleraraujo/vicuna/specs/038-ci-trigger-dedupe/spec.md)
**Input**: Feature specification from `/specs/038-ci-trigger-dedupe/spec.md`

## Summary

Remove `pull_request` triggers from duplicated validation workflows so branch updates run validation only through `push`, while preserving manual, scheduled, and PR-specific metadata automation. At the same time, make the main quality gates repository-wide and non-fail-fast so a single run reports the full blocking set.

## Technical Context

**Language/Version**: YAML for GitHub Actions workflows  
**Primary Dependencies**: GitHub Actions trigger semantics  
**Storage**: N/A  
**Testing**: YAML inspection and trigger inventory validation  
**Target Platform**: GitHub-hosted CI for Vicuña repository  
**Project Type**: CI configuration  
**Performance Goals**: Reduce duplicate workflow executions and duplicate check reporting on pull requests  
**Constraints**: Preserve existing jobs, path filters, schedules, and manual triggers; do not disable PR-only governance workflows  
**Scale/Scope**: Repository-level validation workflow trigger cleanup

## Constitution Check

- **Runtime Policy**: No runtime behavior changes; only CI trigger policy is adjusted in inspectable workflow YAML.
- **Typed State**: No typed runtime or memory state changes.
- **Bounded Memory**: Not applicable.
- **Validation**: Validate edited YAML by inspecting trigger blocks and confirming only the intended workflows lost `pull_request`.
- **Documentation & Scope**: Spec Kit artifacts document scope and rationale; no operator docs are needed beyond that.

## Project Structure

### Documentation (this feature)

```text
specs/038-ci-trigger-dedupe/
├── spec.md
├── research.md
├── plan.md
└── tasks.md
```

### Source Code (repository root)

```text
.github/workflows/
├── vicuna-ci.yml
├── python-lint.yml
├── python-type-check.yml
├── server-webui.yml
├── check-vendor.yml
├── pre-tokenizer-hashes.yml
├── python-check-requirements.yml
├── editorconfig.yml
├── update-ops-docs.yml
└── copilot-setup-steps.yml
```

## Implementation Strategy

1. Confirm the validation workflow scope and explicitly exclude PR-specific automation such as `labeler.yml`.
2. Remove `pull_request` trigger blocks from the selected validation workflows without altering other trigger or job configuration.
3. Replace diff-scoped or fail-fast quality-gate behavior in `Vicuna CI` with repository-wide aggregated reporting.
4. Add explicit baselines for inherited clang-format and clang-tidy debt so repository-wide scans remain informative without permanently red-lighting on upstream drift.
5. Remove unnecessary inter-job dependencies that suppress independent gate reporting, and repair sanitizer linking for mixed-language tests.
6. Re-scan workflow files and CI results to confirm that only the intended trigger and gate-behavior changes landed.

## Risks

- Removing PR triggers from the wrong workflow would silently disable intended pull-request automation.
- Partial edits could leave inconsistent indentation or malformed YAML.
- Some checks will still appear on PRs indirectly through `push` runs on branch updates; that is expected and desired.
- Repository-wide static analysis may surface pre-existing issues that were previously hidden by changed-file scoping.
- Baseline files can become stale if tool versions or repository layout change; they must therefore stay explicit, inspectable, and version-controlled.
