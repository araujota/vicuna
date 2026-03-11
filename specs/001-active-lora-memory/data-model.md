# Data Model: Sliding-Window Active LoRA Memory

## ActiveLoRAConfig

- `enabled`: whether the feature is active for a context
- `host_memory_ratio`: fraction of currently available host memory reserved for this LoRA unit
- `device_memory_ratio`: fraction of currently available aggregate device memory reserved for this LoRA unit
- `min_rank`: minimum rank required for activation
- `max_rank`: hard ceiling for rank planning
- `train_context_tokens`: token window used by the shadow training context
- `train_stride_tokens`: stride used when building training samples from an evicted span
- `max_updates_before_rollover`: write-count boundary for rollover readiness
- `adapter_scale`: inference scaling applied to the Active LoRA
- `embedding_strategy`: selected embedder identity
- `learning_rate`: optimizer learning rate for span writes
- `weight_decay`: optimizer weight decay for span writes

## ActiveLoRABudget

- `host_free_bytes`: host memory observed at planning time
- `host_budget_bytes`: host memory allowance derived from `host_memory_ratio`
- `device_free_bytes`: aggregate device memory observed at planning time
- `device_budget_bytes`: device allowance derived from `device_memory_ratio`
- `bytes_per_rank_host`: per-rank host-memory cost for selected targets
- `bytes_per_rank_device`: per-rank device-memory cost for selected targets
- `selected_rank`: largest rank that satisfies all active budget constraints
- `placement_summary`: human-readable summary of CPU/device placement for selected targets

## ActiveLoRATarget

- `base_tensor_name`: name of the base model tensor to be adapted
- `tensor_rows`: first dimension of the base tensor
- `tensor_cols`: second dimension of the base tensor
- `memory_domain`: host or device placement used for the runtime adapter tensors
- `rank_cost_bytes`: bytes consumed per unit of rank for this target

## EvictedSpan

- `seq_id`: logical sequence identifier for the source context
- `tokens`: evicted token payload
- `token_count`: number of tokens in the span
- `source_window_start`: source position where the evicted span began
- `source_window_end`: source position where the evicted span ended
- `reason`: why the span was evicted or admitted

## SpanEmbedding

- `strategy_id`: embedder used for this span
- `dimension`: vector dimension returned by the embedder
- `l2_norm`: norm of the span embedding
- `values`: vector payload used for admission comparison and future writer policy

## ActiveLoRAUpdateRecord

- `update_id`: monotonically increasing write identifier
- `span_token_count`: size of the admitted evicted span
- `embedding_strategy`: strategy used for admission
- `admission_decision`: accepted, skipped, truncated, or failed
- `budget_rank`: rank active at write time
- `optimizer_steps`: number of writer steps executed
- `rollover_ready`: whether the write crossed the rollover boundary
- `message`: human-readable audit note

## ActiveLoRAState

- `budget`: current `ActiveLoRABudget`
- `targets`: selected `ActiveLoRATarget` list
- `updates_applied`: total successful writes
- `tokens_ingested`: total admitted tokens
- `rollover_ready`: whether the current Active LoRA should be frozen on the next lifecycle transition
- `last_embedding`: most recent accepted `SpanEmbedding`
- `last_update`: most recent `ActiveLoRAUpdateRecord`

## State Transitions

1. `disabled -> planned`
   The runtime computes budgets and target tensors.
2. `planned -> active`
   The runtime allocates the mutable Active LoRA and the shadow training context.
3. `active -> updating`
   An evicted span is embedded, admitted, and written through the shadow context.
4. `updating -> active`
   The write succeeds and updates counters, traces, and last-span metadata.
5. `active -> rollover_ready`
   Update count or policy threshold crosses the configured boundary.
6. `rollover_ready -> frozen_handoff`
   A future past-LoRA implementation freezes the unit and records the handoff.
