# Data Model

## `llama_cognitive_plan_step`

- `kind`
- `status`
- `tool_kind`
- `reason_mask`
- `score`
- `expected_gain`

## `llama_cognitive_plan_trace`

- `valid`
- `origin`
- `mode`
- `plan_id`
- `status`
- `revision_count`
- `step_count`
- `current_step`
- `selected_step_kind`
- `aggregate_score`
- `steps[]`

## Runner additions

Both active and DMN runner status need:

- current `plan_id`
- `planning_active`
- `plan_revision_count`
- `current_plan_step`

## Trace additions

Both active and DMN traces need:

- embedded `plan`
- executed step kind
- whether planning/composition mode was used
