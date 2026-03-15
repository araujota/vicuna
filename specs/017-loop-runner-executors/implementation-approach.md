# Implementation Approach: Event-Driven Planner-Executors For Vicuña

## Recommendation

Build a proper runner as an event-driven planner-executor, not as a free-form
recursive prompt loop.

The runner should:

1. use the existing explicit scoring logic to choose the next move
2. materialize that move as runner state plus a host-visible command
3. continue locally when the action is internal and cheap
4. suspend when external work is needed
5. resume on observation or command completion

## Active Runner

The active runner should be conservative and user-facing:

- low step budget
- typically one command followed by wait or finish
- resume only on tool observations or host completion

Expected path:

- user event -> score -> answer / ask / act / wait
- answer or ask => enqueue emit command and finish
- act => enqueue tool command and wait
- tool observation => resume and finish with bounded follow-up behavior

## DMN Runner

The DMN runner should support bounded internal continuation:

- admitted by pressure
- seeded from self-state, reactivation, and tool state
- computes favorable, counterfactual, remediation, governance
- may perform internal write inline
- may then re-plan once or twice within budget
- yields tool or emit commands when externalization is justified

This makes DMN a real bounded maintenance process rather than a single-step
router.

## Host Integration

The host should not have to infer behavior from traces. It should see explicit
pending commands via a queue API.

Required host interactions:

- poll pending commands
- acknowledge command start if desired
- mark command completion or cancellation

This is enough for future tool integrations and background emit orchestration.

## Parity Constraints

- do not replace existing winner scoring
- do not bypass remediation or governance
- do not create unbounded autonomous recursion
- keep all continuation bounded by explicit step counts
- preserve current tests where possible and extend them for the runner surface
