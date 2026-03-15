# Spec: Planning/Composition Mode

## Summary

Replace direct state-switching action selection in the active and DMN loops with
a shared bounded planning/composition mode. The planner is not a third loop. It
is a reusable control mode invoked from active or DMN origins to draft,
compose, revise, and execute a short plan that may include internal writes, tool
invocation, observation integration, and emission.

The planner must remain typed, bounded, CPU-visible, and compatible with the
existing functional LoRA control system. A new planning/composition functional
LoRA family must participate in the same gain prediction and Adam-backed update
path as the existing functional families.

## Problem

The current runtime has bounded runners, candidate tables, tool proposals, and
queue-visible commands, but it still behaves as a routed state switch:

- candidate scoring picks a winning action directly
- there is no explicit reusable plan object
- there is no subgoal or composition surface shared by active and DMN flows
- tool use is attached after action choice rather than being part of a durable
  plan
- tool results can resume a runner, but there is no explicit plan revision step
- the functional LoRA stack does not have a dedicated planning/composition
  family

This keeps the system closer to a finite action router than to a proper
planner/executor/composer.

## Goals

- Introduce a first-class bounded cognitive plan surface shared by active and
  DMN origins.
- Make planning/composition a mode of active and DMN operation, not a third
  loop.
- Represent plans explicitly with typed steps, statuses, revision counts, and
  next-step pointers.
- Let plans include tool use, internal writes, observation integration, and
  outward emission.
- Route tool returns back into plan revision rather than treating them only as a
  fresh state-switch.
- Add a planning/composition functional LoRA family and microphases that
  participate in the same gain control and Adam update path as the existing
  functional families.
- Keep the runtime bounded and inspectable.

## Non-Goals

- No unbounded search tree or recursive planner.
- No generic external workflow engine.
- No autonomous skill library in this change.
- No replacement of active or DMN origins with a third scheduler.

## User-Facing Behavior

- Active and DMN traces expose an explicit plan summary and plan steps.
- Tool proposals are derived from the current executable plan step rather than
  directly from a winning action alone.
- Tool-result integration can revise a pending plan before the loop settles.
- Planning/composition can select simple one-step plans when no composition is
  needed, so the system remains fast for easy cases.

## Functional Requirements

### FR1: Shared Cognitive Plan Surface

The runtime must expose a bounded shared cognitive plan representation that can
be used by both active and DMN flows.

Each plan must include:

- origin (`ACTIVE` or `DMN`)
- mode (`NONE` or `PLANNING_COMPOSITION`)
- plan id
- plan status
- step count
- current step index
- revision count
- a bounded array of plan steps

### FR2: Typed Plan Steps

Plan steps must be typed and bounded. Supported step kinds must include:

- internal write
- invoke tool
- observe tool result
- emit answer
- emit ask
- emit background
- wait

Each step must include a kind, status, score/priority, optional tool kind, and
reason mask.

### FR3: Shared Planner/Composer

The cognitive loop must use a shared planner/composer function for both active
and DMN origins.

The planner must:

- consume the local loop context and candidate table
- decide whether planning/composition mode is needed
- create a bounded plan
- choose the next executable step
- revise the plan after tool observation or internal-write observation

### FR4: Planning Replaces Direct Action Switching Where Obsoleted

Active and DMN loops may still keep lifecycle phases for bookkeeping, but final
action selection must come from the plan’s executable step when planning mode is
active.

### FR5: Tool Integration

Planning/composition must be able to include tool invocation and observation as
plan steps. Tool results must map back into plan revision.

### FR6: Functional LoRA Integration

A new planning/composition functional LoRA family must exist.

It must:

- have its own serving adapter layer
- be eligible during planning microphases in both active and DMN flows
- be gated by the same learned gain MLP
- receive stochastic gain exploration like the other functional families
- be updated through the same Adam-backed functional update path

### FR7: Boundedness

The planner must remain bounded.

- bounded number of steps per plan
- bounded number of revisions
- bounded number of planning candidates
- bounded plan metadata in traces

### FR8: Observability

The public trace surface must expose enough information to understand:

- whether planning/composition mode was used
- the produced plan
- the executed step
- whether revision occurred
- which functional LoRA microphase/family was engaged

## Acceptance Criteria

- Active traces show explicit plans for tool-using or multi-step episodes.
- DMN traces show explicit plans for internal-write-plus-followup,
  tool-gathering, or emit sequences.
- Simple cases can still collapse to one-step plans.
- Tool result resumption revises or advances the plan rather than bypassing the
  plan surface.
- The planning/composition functional LoRA family exists and updates.
- Existing targeted active/DMN/functional tests pass, with new tests covering
  planning traces and planner-driven tool flow.
