# Implementation Approach: Shared Bounded Tool Loops, Not Generic ReAct Everywhere

## Recommendation

Implement a shared bounded tool-loop substrate and wrap it with two policies:

1. a foreground active-loop policy optimized for latency, clarification, and
   user-facing responsiveness
2. a DMN policy optimized for maintenance, remediation, simulation, and
   governed background action

This is better than copying a standard ReAct loop into both places.

## Why This Is The Right Fit For Vicuña

Vicuña already has:

- explicit self-state
- typed tool state
- explicit favorable-state and remediation machinery
- a separate active loop and DMN

So the correct next step is structural, not wholesale replacement:

- preserve current scoring
- add explicit loop scaffolding around it

## Shared Substrate

The shared substrate should provide:

- loop phases
- bounded step budgets
- tool registry metadata
- typed tool proposals
- typed tool observations
- terminal reasons

These are the pieces that future tools will need regardless of whether they are
called from the active loop or DMN.

## Active Loop Policy

Keep the current winner selection, then project it into a foreground episode:

- `ANSWER` -> `FINISH` with `ANSWER_READY`
- `ASK` -> `FINISH` with `ASK_USER`
- `ACT` -> `PREPARE_TOOL` with a tool proposal and future wait/observe path
- `WAIT` -> `FINISH` with `WAITING_ON_TOOL` or low-priority defer semantics

Use small budgets, for example two or three bounded logical steps, because this
loop is user-facing.

## DMN Policy

Keep current pressure, counterfactual, remediation, and governance ordering.
Then project the winning result into a background episode:

- `SILENT` -> `FINISH` with `PRESSURE_NOT_ADMITTED` or `GOVERNANCE_BLOCKED`
- `INTERNAL_WRITE` -> `OBSERVE` / `FINISH` with `INTERNAL_WRITE_READY`
- `INVOKE_TOOL` -> `PREPARE_TOOL` with remediation-aware tool proposal
- `EMIT` -> `FINISH` with `EMIT_READY`

DMN budgets can be slightly larger because the loop is maintenance-oriented,
but they still need to remain bounded and preemptible.

## Tool Registry Design

The runtime should initialize a small default registry from current tool kinds:

- `GENERIC`
- `HARD_MEMORY_QUERY`
- `HARD_MEMORY_WRITE`

Registry metadata should express:

- active eligibility
- DMN eligibility
- latency class
- simulation safety
- remediation safety
- whether side effects are external

This makes future tool insertion additive.

## Parity Rules

- do not replace current candidate scoring with free-form prompt loops
- do not change favorable-state, counterfactual, remediation, or governance
  ordering
- do not break Active/Past LoRA update triggers
- do not remove existing trace fields; extend them

## Implementation Scope For This Pass

This pass should implement:

- public tool-loop structs and enums
- persistent loop state in `llama_cognitive_loop`
- active and DMN trace population
- default registry population from existing tool kinds
- regression coverage

This pass should not try to implement:

- a large new real tool suite
- unconstrained self-directed looping
- opaque chain-of-thought storage
