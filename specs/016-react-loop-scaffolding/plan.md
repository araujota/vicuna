# Implementation Plan: Bounded Tool-Loop Scaffolding For Active And DMN

## Goal

Add a shared explicit tool-loop substrate to Vicuña while preserving the
existing active-loop and DMN decision policies.

## Architecture Decision

Do not implement one generic unconstrained ReAct loop.

Implement:

1. A shared bounded loop substrate with typed phase, budget, proposal, and
   observation state.
2. A foreground policy wrapper for latency-sensitive user-facing episodes.
3. A background policy wrapper for maintenance, remediation, and simulation
   ticks.

## Workstreams

### 1. Public API Surfaces

- Add loop phase, terminal reason, tool flags, and latency-class enums.
- Add bounded tool-registry metadata structs.
- Add bounded episode-state, tool-proposal, and observation structs.
- Embed these in active and DMN traces without removing current fields.

### 2. Runtime State

- Extend `llama_cognitive_loop` with tool specs and persistent episode state.
- Initialize a small default registry from existing tool kinds.
- Keep host-state compatibility while exposing richer loop summaries.

### 3. Foreground Policy

- Preserve current `ANSWER/ASK/ACT/WAIT` scoring.
- Derive an active episode state from the winner action.
- Route `ACT` through an explicit tool proposal scaffold.
- Route `ANSWER`, `ASK`, and `WAIT` through explicit terminal reasons and
  continuation decisions.

### 4. DMN Policy

- Preserve current pressure, favorable-state, counterfactual, remediation, and
  governance ordering.
- Wrap the winning DMN action in explicit background episode state.
- Represent internal write, tool invocation, and emit as bounded phase/plan
  transitions rather than opaque side effects only.

### 5. Testing And Docs

- Extend cognitive-loop tests to assert new scaffolding fields.
- Update architecture and whitepaper docs to describe bounded tool loops and
  distinct active/DMN policies.

## Rollout Notes

- Keep all new state CPU-side and inspectable.
- Preserve current tool-kind semantics so future tool implementations can plug
  in without invalidating traces.
- Treat actual multi-step autonomous continuation as future work; this pass
  establishes the substrate.
