# Implementation Approach: Adam Expansion For Runtime Updates

## Decision

Adopt Adam in two additional self-state-driven parameter-update paths:

1. runtime LoRA tensor writes
2. temporal reward/dampening bias updates

Do not adopt Adam in the current counterfactual ablation ladder.

## Runtime LoRA Path

### Before

- compute self-state-derived delta
- add the delta directly into flattened `A` and `B` tensor values
- normalize gain bookkeeping

### After

- compute the same directional deltas
- reinterpret them as pseudo-gradients over `A` and `B`
- update per-parameter first and second moments
- apply an Adam step with bounded effective learning rate derived from the
  existing budgeted tensor scale
- keep weight decay and gain normalization explicit

## Temporal Bias Path

### Before

- threshold signed and efficiency advantages
- add or decay reward and dampening biases heuristically

### After

- derive bounded signed pseudo-gradients from the same advantage signals
- update scalar Adam state for reward and dampening biases
- clip reward and dampening biases to existing bounds
- recompute effective write scale from bounded biases

## Counterfactual Ladder

Keep it unchanged aside from documentation and tests clarifying why it remains
explicit policy. It ranks interventions; it does not update model parameters.

## Validation Strategy

- prove Active LoRA stats expose optimizer advancement
- prove functional-family writes still settle correctly because they share the
  same writer
- prove temporal bias state exposes optimizer advancement after DMN temporal
  self-improvement
- rerun targeted regression tests for Active LoRA and cognitive loop behavior
