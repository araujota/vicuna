# Data Model: Adam Expansion For Runtime Updates

## Runtime LoRA Weight Adam State

Represents persistent optimizer state for one runtime-mutable LoRA tensor pair.

### Fields

- `step`: monotonic optimizer step counter for the tensor pair
- `m_a`, `v_a`: first and second moments for LoRA tensor `A`
- `m_b`, `v_b`: first and second moments for LoRA tensor `B`
- `last_update_norm`: last combined parameter update norm for observability

### Invariants

- sizes of `m_a` and `v_a` match the flattened size of tensor `A`
- sizes of `m_b` and `v_b` match the flattened size of tensor `B`
- state exists only for runtime-mutable adapters owned by the active LoRA
  manager
- memory growth is bounded by the fixed runtime adapter/tensor inventory

## Runtime LoRA Optimizer Summary

Public observability summary for Active LoRA writes.

### Fields

- `optimizer_step_count`: total number of optimizer-backed runtime write steps
- `optimizer_last_update_norm`: norm of the most recent optimizer-backed write

### Invariants

- advances only when an optimizer-backed write actually occurs
- remains zero for contexts that never performed runtime LoRA writes

## Temporal Bias Adam State

Represents optimizer state for reward and dampening scalar biases.

### Fields

- `reward_step`, `reward_m`, `reward_v`
- `dampening_step`, `dampening_m`, `dampening_v`
- `last_update_norm`

### Invariants

- all state is scalar and bounded
- applied biases remain clipped to the existing runtime bounds
- `effective_write_scale` continues to derive from bounded reward and
  dampening biases

## Non-Entity: Counterfactual Ranking

The counterfactual ladder remains a scored discrete policy surface, not an
optimizer-backed parameter model.

### Explicit Non-Invariants

- no Adam moments
- no hidden learned ranking weights added in this feature
- no reinterpretation of explicit ranking heuristics as gradients
