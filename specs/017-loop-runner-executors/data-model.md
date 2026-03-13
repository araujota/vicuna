# Data Model: Runner Status And Command Queue

## New Public Enums

- `llama_cognitive_command_kind`
  - `NONE`
  - `EMIT_ANSWER`
  - `EMIT_ASK`
  - `INVOKE_TOOL`
  - `INTERNAL_WRITE`
  - `EMIT_BACKGROUND`
- `llama_cognitive_command_origin`
  - `ACTIVE`
  - `DMN`
- `llama_cognitive_command_status`
  - `PENDING`
  - `ACKED`
  - `COMPLETED`
  - `CANCELLED`

## New Public Structs

### Runner Command

- `command_id`
- `origin`
- `kind`
- `status`
- `episode_id`
- `tick_id`
- `tool_kind`
- `tool_job_id`
- `reason_mask`
- `priority`
- `source_family`
- `loop_phase`

### Active Runner Status

- `episode_id`
- `active`
- `waiting_on_tool`
- `completed`
- `steps_taken`
- `max_steps`
- `pending_command_id`
- `last_command_id`

### DMN Runner Status

- `tick_id`
- `active`
- `waiting_on_tool`
- `completed`
- `steps_taken`
- `max_steps`
- `pending_command_id`
- `last_command_id`

## Queue Behavior

- bounded FIFO queue in runtime
- host can inspect by index
- host can acknowledge or complete by `command_id`
- completion feeds back into runner state
