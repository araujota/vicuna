# Data Model: Strict Live Memory Runtime

## ServingAdapterLayer

- `adapter`: pointer/handle to the underlying `llama_adapter_lora`
- `scale`: effective inference scale applied to this layer
- `role`: one of `request`, `past_week`, `past_month`, `past_quarter`, `past_year`, `all_time`, `active`
- `precedence`: deterministic ordering index used when rebuilding the serving stack
- `enabled`: whether the layer should participate in the next decode

Validation rules:

- `adapter` must be non-null
- `scale == 0` disables the layer but does not erase its ownership record
- `precedence` must be unique within the effective stack

## RequestAdapterSet

- Ordered or keyed collection of request-managed adapter layers
- Mutated by request or server task state
- Must not own runtime memory layers

State transitions:

1. `empty -> populated` when request adapters are attached
2. `populated -> updated` when task/request changes LoRAs
3. `populated -> empty` when request adapters are cleared

## RuntimeMemoryStack

- Ordered collection of runtime-managed memory layers
- Contains zero or more frozen past buckets plus the editable Active LoRA
- Owned by the runtime memory manager, not by request code

State transitions:

1. `uninitialized -> active_only`
2. `active_only -> active_plus_past`
3. `active_plus_past -> updated_scales` on decay/rollover/tick

## EffectiveServingStack

- Rebuilt composition of `RequestAdapterSet + RuntimeMemoryStack`
- Passed into graph construction for live decode
- Must preserve deterministic precedence and explicit roles

Derived invariants:

- Memory layers remain present across request-adapter changes
- Active layer precedes no newer memory layer
- Rebuild is triggered whenever request adapters, runtime scales, or runtime membership changes

## StrictReplayState

- `pending`: whether retained-suffix replay must run before further generation
- `start_index`: first prompt-token index that must be replayed
- `end_index`: one-past-end prompt-token index to replay
- `cursor`: current replay progress
- `trigger_reason`: enum such as `active_write`, `redundant_skip`, `zero_suffix_skip`
- `weights_changed`: whether Active ingestion materially changed serving weights

State transitions:

1. `idle -> pending` after accepted Active write with retained suffix
2. `pending -> replaying` when replay batches begin
3. `replaying -> completed` when the retained suffix is fully re-decoded
4. `idle -> skipped` when replay is unnecessary
5. `pending/replaying -> failed` when replay decode returns error

## ReplayAuditRecord

- `slot_id`
- `evicted_token_count`
- `retained_token_count`
- `weights_changed`
- `action`: `scheduled`, `skipped`, `completed`, `failed`
- `reason`
- `timestamp_us`

## Relationships

- `RuntimeMemoryStack` owns runtime `ServingAdapterLayer` entries.
- `RequestAdapterSet` owns request `ServingAdapterLayer` entries.
- `EffectiveServingStack` is a rebuild product of both sets.
- `StrictReplayState` is slot-local and depends on Active-ingestion outcome plus retained suffix length.
- `ReplayAuditRecord` is emitted from replay-state transitions.
