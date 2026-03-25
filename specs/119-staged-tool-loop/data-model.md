# Data Model: Staged Tool Loop

## ToolFamilySummary

- `family_id`: stable machine identifier
- `family_name`: provider-facing selection name
- `family_description`: concise description of what the family enables for the runtime and user
- `method_count`: number of callable methods available in the family

## ToolMethodSummary

- `capability_id`: stable execution target
- `family_id`: parent family identifier
- `method_name`: provider-facing selection name
- `method_description`: concise description of what the method does
- `tool_name`: underlying runtime tool name
- `dispatch_backend`: current runtime backend used for execution

## ToolMethodContract

- `capability_id`: stable execution target
- `family_id`: parent family identifier
- `method_name`: provider-facing selection name
- `schema`: normalized JSON schema object
- `required_fields`: ordered required field names
- `field_descriptions`: extracted short descriptions per field path

## StagedToolLoopState

- `stage`: `family_select` | `method_select` | `payload_build` | `completed`
- `selected_family_id`: currently selected family, if any
- `selected_method_name`: currently selected method, if any
- `last_tool_capability_id`: last executed capability, if any
- `last_tool_observation_summary`: short result summary used to resume the loop
- `mode`: `foreground` | `ongoing_task` | `background_active`
- `suppress_replay_admission`: whether this loop must not admit cognitive replay entries

## Stage Response Shapes

### FamilySelectionResponse

- `family`: selected family name

### MethodSelectionResponse

- `method`: selected method name, or `back`, or `complete`

### PayloadSelectionResponse

- `action`: `submit` or `back`
- `payload`: typed payload object when `action=submit`

## Normalization Rules

- Families are deduplicated by `tool_family_id`.
- Methods are grouped under their family by `capability_id`.
- Only catalog entries with non-empty family, method, and contract metadata are eligible for staged exposure.
- Contract field descriptions are extracted recursively from `input_schema_json`; nodes missing descriptions are invalid for staged exposure.
