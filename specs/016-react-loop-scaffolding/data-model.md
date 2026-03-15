# Data Model: Shared Tool-Loop Substrate With Distinct Active And DMN Policies

## New Public Enum Families

- `llama_cognitive_loop_phase`
  - `ASSEMBLE`
  - `PROPOSE`
  - `PREPARE_TOOL`
  - `WAIT_TOOL`
  - `OBSERVE`
  - `FINISH`
- `llama_cognitive_terminal_reason`
  - `NONE`
  - `ANSWER_READY`
  - `ASK_USER`
  - `TOOL_REQUIRED`
  - `WAITING_ON_TOOL`
  - `BACKGROUND_DEFERRED`
  - `PRESSURE_NOT_ADMITTED`
  - `INTERNAL_WRITE_READY`
  - `EMIT_READY`
  - `GOVERNANCE_BLOCKED`
- `llama_cognitive_tool_flags`
  - `ACTIVE_ELIGIBLE`
  - `DMN_ELIGIBLE`
  - `SIMULATION_SAFE`
  - `REMEDIATION_SAFE`
  - `EXTERNAL_SIDE_EFFECT`
- `llama_cognitive_tool_latency_class`
  - `LOW`
  - `MEDIUM`
  - `HIGH`

## New Public Structs

### Loop Tool Spec

Bounded metadata for future tool insertion:

- `tool_kind`
- `flags`
- `latency_class`
- `max_steps_reserved`
- `name`

### Loop Tool Proposal

Bounded summary of the currently intended tool action:

- `valid`
- `tool_kind`
- `reason_mask`
- `source_family`
- `safety_flags`
- `expected_steps`
- `expected_observation_gain`
- `job_id`

### Loop Observation

Bounded summary of the latest observation:

- `valid`
- `tool_kind`
- `job_id`
- `status`
- `signal`
- `followup_affinity`

### Loop Episode State

Bounded episode/tick state:

- `phase`
- `terminal_reason`
- `max_steps`
- `steps_taken`
- `continuation_allowed`
- `waiting_on_tool`
- `tool_registry_count`

## Trace Embedding

Embed:

- `loop_state`
- `tool_proposal`
- `observation`

into both `llama_active_loop_trace` and `llama_dmn_tick_trace`.

The DMN trace also keeps its current remediation and governance surfaces
unchanged.
