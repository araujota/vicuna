# Data Model: Authoritative ReAct Action Contract Guarantee

## RuntimeOwnedStageAction

- **Purpose**: Runtime-owned exact action value for one staged ReAct phase.
- **Fields affected**:
  - existing staged phase enum on `server_task.react_tool_stage`
- **Invariants**:
  - resolves to the exact expected action for the current phase
  - can be supplied explicitly by the model or normalized by the runtime when omitted
  - never allows a conflicting action value for the current phase

## StageVariablePayload

- **Purpose**: Model-generated portion of the staged JSON other than the fixed phase action.
- **Fields affected**:
  - generated assistant visible JSON payload
- **Invariants**:
  - contains only the phase-specific variable payload
  - must not restate or override the runtime-owned fixed action contract

## MalformedControlRetry

- **Purpose**: Explicit internal recovery state for malformed staged control payloads.
- **Fields affected**:
  - existing `react_retry_feedback`
  - existing `react_retry_count`
  - existing `react_stage_retry_count`
  - existing `react_last_failure_class`
  - existing `react_last_failure_detail`
- **Invariants**:
  - malformed staged control remains an internal retry condition
  - repeated failure can rewind to an earlier stage
  - internal parse details are not emitted as user-visible text

## ControlShapedVisiblePayload

- **Purpose**: Detect visible JSON fragments that look like controller artifacts rather than prose.
- **Fields affected**:
  - fallback visible content examined in `infer_authoritative_react_step_without_action_label(...)`
- **Invariants**:
  - JSON containing keys such as `tool_family_id`, `method_name`, `decision`, or controller-only `action` fields is not accepted as terminal visible prose without a valid contract
  - ordinary prose fallback remains allowed when no control-shaped payload is present
