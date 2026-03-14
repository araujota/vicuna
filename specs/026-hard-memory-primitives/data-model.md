# Data Model: Hard Memory Primitives

## Hard Memory Primitive

- `kind`: semantic class of durable memory
  - `EVENT_FRAGMENT`
  - `TRAJECTORY`
  - `OUTCOME`
  - `TOOL_OBSERVATION`
  - `USER_MODEL`
  - `SELF_MODEL_FRAGMENT`
- `domain`: primary control domain
  - goal progress
  - user outcome
  - epistemic
  - efficiency
  - recovery
  - strategy
  - self improvement
- `source_role`: user, tool, or system
- `source_channel`: primary or counterfactual
- `source_tool_kind`: none, bash, hard-memory query, hard-memory write, generic
- `importance`: bounded `[0, 1]`
- `confidence`: bounded `[0, 1]`
- `gain_bias`: bounded `[0, 1]`
- `allostatic_relevance`: bounded `[0, 1]`
- `event_id` / `episode_id`: bounded identifiers
- `key`: stable key for de-duplication
- `title`: short label
- `content`: bounded textual payload
- `tags`: bounded tag list for retrieval filtering and reconstruction

## Hard Memory Archive Batch

- `primitive_count`: number of primitives sent in one transaction
- `primitives[]`: bounded array of archived primitives
- `container_tag`
- `runtime_identity`
- `request_started_us`, `request_completed_us`
- `status_code`
- `attempted`, `archived`
- `error`

Semantics:

- one event or one settled loop transaction can emit multiple primitives
- primitive budget is capped
- over-budget candidates are trimmed by salience

## Hard Memory Hit Metadata

- `kind`
- `domain`
- `source_role`
- `source_channel`
- `source_tool_kind`
- `importance`
- `confidence`
- `gain_bias`
- `allostatic_relevance`
- `tags`

Semantics:

- reconstructed from Supermemory metadata when available
- safe defaults apply when metadata is missing or malformed

## Hard Memory Retrieval Summary

- counts by kind:
  - event
  - trajectory
  - outcome
  - tool observation
  - user model
  - self model
- domain-weighted activation:
  - epistemic
  - goal
  - user
  - efficiency
  - recovery
  - strategy
  - self improvement
- aggregate fields:
  - mean similarity
  - max similarity
  - importance signal
  - confidence signal
  - gain support
  - allostatic support

Semantics:

- fixed-width summary consumed by self-state promotion and functional gating
- updated on each query result

## Archive Sources

### Self-State Event Archive

- emits `EVENT_FRAGMENT`
- can also emit `USER_MODEL` or `SELF_MODEL_FRAGMENT` when social or self-model
  signal crosses thresholds

### Active-Loop Settlement Archive

- emits `TRAJECTORY`
- emits `OUTCOME`
- emits `TOOL_OBSERVATION` when a tool result exists

### DMN Settlement Archive

- emits `TRAJECTORY`
- emits `OUTCOME`
- can emit `SELF_MODEL_FRAGMENT` from counterfactual/governance/remediation
  traces

## Interaction With Self-Model And LoRA Bias

- retrieved primitives can be promoted to `MEMORY_CONTEXT` self-model
  extensions
- retrieval summary is appended to the functional-gating observation
- memory-context retrieval remains non-allostatic by default
- user-model and self-model fragments may contribute allostatic support only
  when explicitly marked and converted into tool-authored scalar state through
  separate policy
