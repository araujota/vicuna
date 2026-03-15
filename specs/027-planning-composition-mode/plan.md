# Plan

## Architecture

Introduce a shared `cognitive plan` surface in the cognitive loop and route both
active and DMN origins through a shared planning/composition function after
candidate scoring.

The plan layer sits between candidate selection and command emission:

1. build local context and candidate table
2. decide whether planning/composition mode is needed
3. draft a bounded plan
4. select the next executable step
5. emit command / internal write / wait based on that step
6. revise the plan after observation

## Design

### Shared Plan Types

Add public types for:

- plan mode
- plan status
- plan step kind
- plan step status
- plan trace
- plan step

Add a plan summary to active and DMN traces and runner status.

### Functional LoRA

Add:

- `LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION`
- new planning microphases for draft, compose, and revise
- new serving LoRA layer role

Wire the family through:

- adapter initialization
- functional gating input
- activation routing
- update application
- tests

### Cognitive Loop Changes

Refactor active and DMN action finalization so they:

- assemble candidates as before
- call shared plan composer
- execute plan step instead of directly branching on winner action

### Plan Revision

Revision triggers:

- tool result event in active flow
- internal write result / tool result / governance outcome in DMN flow

Revision remains bounded by a small revision budget.

## Validation

- active trace shows explicit plan for tool episodes
- DMN trace shows explicit plan for internal-write-plus-followup sequences
- planning functional LoRA appears in family registry and traces
- targeted `test-active-lora`, `test-cognitive-loop`, and `test-self-state`
  pass
