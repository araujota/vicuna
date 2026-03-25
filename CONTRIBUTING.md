# Contributing

Vicuña is now maintained as a provider-first runtime rather than a general
`llama.cpp` distribution.

## Expectations

- keep changes aligned with the active provider-first architecture
- use Spec Kit artifacts for non-trivial work
- update tests and documentation alongside behavior changes
- remove dead compatibility code instead of preserving it behind stale flags
- validate the retained server build and its targeted integration tests

## Before Opening a PR

- run the relevant local tests for the affected surface
- confirm documentation still matches runtime behavior
- keep the PR scoped to one behavior change or one cleanup slice
- explain any retained legacy code that could not yet be removed
