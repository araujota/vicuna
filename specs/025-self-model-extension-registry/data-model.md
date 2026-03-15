# Data Model: Self-Model Extension Registry

## New Public Enums

- `llama_self_model_extension_kind`
  - `MEMORY_CONTEXT`
  - `SCALAR_PARAM`
- `llama_self_model_extension_source`
  - `COUNTERFACTUAL`
  - `TOOL_HARD_MEMORY`
  - `TOOL_BASH_CLI`
  - `TOOL_EXTERNAL`
- `llama_self_model_extension_domain`
  - `GOAL_PROGRESS`
  - `USER_OUTCOME`
  - `EPISTEMIC`
  - `EFFICIENCY`
  - `RECOVERY`
  - `STRATEGY`
  - `SELF_IMPROVEMENT`
- `llama_self_model_extension_flags`
  - `ACTIVE`
  - `HAS_DESIRED_STATE`
  - `AFFECT_GAIN`
  - `AFFECT_ALLOSTASIS`
  - `DISCOVERED`

## New Public Structs

### `llama_self_model_extension_update`

Write contract for host/tool code.

Fields:

- key / label
- source
- source_tool_kind
- kind
- domain
- flags
- value
- desired_value
- confidence
- salience
- gain_weight
- allostatic_weight

### `llama_self_model_extension_info`

Read contract for inspection.

Fields:

- slot
- key / label
- source
- source_tool_kind
- kind
- domain
- flags
- value
- desired_value
- confidence
- salience
- gain_weight
- allostatic_weight
- last_update_monotonic_ms
- activation_count

### `llama_self_model_extension_summary`

Fixed-size summary consumed by control paths.

Fields:

- active_count
- gain_count
- allostatic_count
- hard_memory_count
- tool_count
- mean_confidence
- mean_salience
- max_salience
- gain_signal
- gain_signal_abs
- context_activation
- allostatic_divergence

### `llama_self_model_extension_candidate`

Counterfactual promotion candidate.

Fields:

- key / label
- source_tool_kind
- domain
- expected_gain_improvement
- expected_allostatic_delta
- confidence
- promoted

### `llama_self_model_extension_trace`

Last discovery/promotion record.

Fields:

- valid
- candidate_count
- promoted_count
- winner_index
- candidates[]

## Internal Storage

- bounded vector/array of extensions, capacity fixed at compile time
- LRU/low-salience eviction when capacity is exhausted
- per-extension sketch for memory-context activation

## Policy Rules

- Authored core is always present and remains the main self-model.
- Extensions are additive and bounded.
- Hard-memory-derived `MEMORY_CONTEXT` entries default to:
  - `AFFECT_GAIN = true`
  - `AFFECT_ALLOSTASIS = false`
- Tool-authored `SCALAR_PARAM` entries may set both flags when they also supply
  a desired state.
- Extensions without desired state cannot contribute to allostatic divergence.
- Extension summaries are fixed-size aggregates so gating input dimensionality
  stays bounded.
