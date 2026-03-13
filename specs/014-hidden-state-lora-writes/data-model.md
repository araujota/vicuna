# Data Model: Hidden-State-Derived Active LoRA Embeddings And Feature-Derived Write Rules

## Overview

These entities define the runtime surfaces needed to implement the new
hidden-state embedder and writer while preserving current Active/Past LoRA,
remediation, and serving-stack parity.

## Proposed Entities

### Hidden-State Embedder Params

Defines how span embeddings are extracted from the serving model family.

Suggested fields:

- `mode`
  - `FINAL_OUTPUT_POOL`
  - `FINAL_TOKEN_STACK`
  - `LATE_LAYER_TAP`
- `pooling`
  - `MEAN`
  - `ATTENTION_WEIGHTED`
  - `LAST_TOKEN`
  - `TOPK_TOKEN_MEAN`
- `normalize`
- `max_tokens_per_pass`
- `chunk_overlap`
- `use_aux_context`
- `aux_context_quantization`
  - `NONE`
  - `Q4_NF4`
- `late_layer_indices[]`
- `fallback_mode`

### Hidden-State Span Embedding

The content representation used for admission, redundancy suppression, and
cluster similarity.

Suggested fields:

- `dim`
- `values[]`
- `norm`
- `source_mode`
- `source_layer_mask`
- `token_count`
- `chunk_count`

### Write Feature Vector

The typed feature bundle used for writer decisions. This must be distinct from
the pure admission embedding.

Suggested components:

- `content_embedding`
- `goal_projection`
- `social_projection`
- `register_snapshot`
- `register_delta`
- `event_role`
- `channel`
- `tool_state_summary`
- `continuation_pressure`
- `repair_pressure`
- `temporal_phase`
- `remediation_budget_scale`

### Layer Sensitivity Profile

Controls how write budget is distributed across target layer groups.

Suggested fields:

- `group_count`
- `groups[]`
  - `name`
  - `layer_start`
  - `layer_end`
  - `tensor_roles_mask`
  - `base_rank_budget`
  - `allocated_rank_budget`
  - `sensitivity_score`
  - `write_energy_ema`
  - `governance_bonus`

### Write Direction Slice

One bounded update slice for a target layer/tensor group.

Suggested fields:

- `target_role`
- `layer_index`
- `rank_used`
- `left_direction`
- `right_direction`
- `gain_delta`
- `feature_summary`
- `source_kind`
  - `INGEST`
  - `REMEDIATE`
  - `CONDENSE`

### Active LoRA Write Trace V2

Detailed observability for the new embedder/writer pipeline.

Suggested fields:

- `embedder_mode`
- `fallback_used`
- `span_embedding_dim`
- `similarity_to_previous`
- `similarity_to_memory_handles`
- `layer_budget_before`
- `layer_budget_after`
- `selected_layer_groups`
- `write_feature_norm`
- `direction_energy`
- `gain_mean_before`
- `gain_mean_after`
- `skipped_reason`
- `parity_flags`

### Rank Allocation Policy

Defines how AdaLoRA-like ideas are adapted for runtime use.

Suggested fields:

- `strategy`
  - `FIXED`
  - `PERIODIC_SENSITIVITY_REALLOCATION`
- `reallocation_interval_updates`
- `min_rank_per_group`
- `max_rank_per_group`
- `target_total_rank`
- `sensitivity_beta`
- `freeze_threshold`

### Quantized Auxiliary Context Policy

Optional support surface for a hidden-state extraction context that is cheaper
than a full-precision duplicate.

Suggested fields:

- `enabled`
- `quant_type`
- `compute_dtype`
- `double_quant`
- `max_memory_bytes`
- `allowed_backends_mask`

## Parity-Carrying Entities

### Layering Parity Invariants

These are not new runtime objects but explicit invariants the implementation
must carry forward:

- Active role remains `LLAMA_ADAPTER_LORA_LAYER_ACTIVE`
- Past roles remain week/month/quarter/year/all-time
- serving-order remains request -> all_time -> year -> quarter -> month -> week -> active

### Ablation Parity Invariants

- `LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION` must still refer to meaningful
  runtime memory layers after the writer changes.
- Any new layer sensitivity or group allocation must remain introspectable to
  the ablation and counterfactual machinery.

### Remediation Parity Invariants

- ordinary ingestion and remediation must share the same writer core
- remediation may override budgets or feature emphasis
- remediation must still avoid base-weight mutation and past-bucket mutation
