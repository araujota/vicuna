# Data Model: Frozen Past-LoRA Condensation Stack

## TemporalBucketId

- `past_week`
- `past_month`
- `past_quarter`
- `past_year`
- `all_time`

## DirectionGainConfig

- `direction_epsilon`: lower bound for normalization stability
- `gain_max`: maximum effective gain allowed for one adapter weight
- `gain_learning_rate`: update rate for Active-stage gain changes
- `gain_decay`: passive shrink factor applied before or during condensation
- `singular_value_floor`: threshold for dropping low-energy condensed components

## BucketConfig

- `bucket_id`: one `TemporalBucketId`
- `host_memory_ratio`: fraction of available host memory reserved for this bucket
- `device_memory_ratio`: fraction of available device memory reserved for this bucket
- `min_rank`: minimum viable rank for activation
- `max_rank`: hard rank ceiling
- `base_scale`: pre-decay inference scale for this bucket
- `decay_half_life_us`: half-life used to convert age into effective inference scale
- `condensation_period_us`: how often the bucket should compact into the next older bucket
- `merge_source_weight`: how strongly the younger source bucket contributes during condensation
- `merge_target_retention`: how much of the current target bucket survives before the new merge is injected

## ActiveDirectionGainState

- `selected_rank`: planned rank for the current Active LoRA
- `direction_targets`: curated list of adapted tensors
- `per_target_gain`: bounded scalar gain per adapted tensor
- `updates_applied`: successful live write count
- `tokens_ingested`: admitted token count
- `rollover_ready`: whether Active should freeze on the next eligible condensation tick
- `last_embedding`: most recent accepted span embedding
- `last_update_us`: timestamp of the last accepted write

## FrozenBucketArtifact

- `bucket_id`: owning temporal bucket
- `version`: monotonically increasing snapshot version for that bucket
- `selected_rank`: rank of the frozen snapshot
- `created_at_us`: timestamp when this snapshot was produced
- `source_window_start_us`: earliest source time represented by the artifact
- `source_window_end_us`: latest source time represented by the artifact
- `effective_gain_max`: maximum per-target gain stored in the artifact
- `effective_gain_mean`: mean per-target gain stored in the artifact
- `embedder_id`: embedder identity used by the source stage
- `normalized`: whether the directional component passed normalization checks
- `populated`: whether the bucket currently contributes to inference

## CondensationJob

- `job_id`: monotonic identifier
- `source_stage`: `active` or one `TemporalBucketId`
- `target_bucket`: one `TemporalBucketId`
- `due_at_us`: next time the job becomes eligible
- `last_run_us`: most recent execution timestamp
- `status`: idle, due, running, skipped, failed, completed
- `skip_reason`: optional human-readable reason for a skipped job

## CondensationRecord

- `job_id`: originating job
- `source_stage`: younger stage being compacted
- `target_bucket`: older bucket receiving the condensed result
- `source_rank`: source rank used for the merge
- `target_rank`: target rank after recompression
- `source_gain_sum`: total source gain entering the merge
- `target_gain_sum_before`: total target gain before merge
- `target_gain_sum_after`: total target gain after merge
- `singular_values_kept`: number of components retained after recompression
- `clipped_gain_count`: number of target gains clipped by policy
- `result`: completed, skipped_redundant, skipped_uninitialized, failed_budget, failed_writer, failed_normalization

## PastLoRAStackState

- `buckets`: ordered set of `FrozenBucketArtifact` states
- `jobs`: explicit `CondensationJob` set for each stage transition
- `last_tick_us`: timestamp of the most recent scheduler tick
- `pending_job_mask`: bitmask describing which transitions are due
- `effective_scales`: current per-bucket decay-adjusted inference scales

## State Transitions

1. `active.editable -> active.rollover_ready`
   Active reaches its update boundary while preserving bounded direction/gain state.
2. `active.rollover_ready -> past_week.snapshot`
   A tick freezes Active into a new Past Week snapshot and replaces the editable Active state with a fresh budgeted adapter.
3. `past_week.snapshot -> past_month.snapshot`
   A scheduled condensation job merges Past Week into Past Month and atomically replaces the Past Month snapshot.
4. `past_month.snapshot -> past_quarter.snapshot`
   The same merge-and-recompress pattern advances memory into the quarter bucket.
5. `past_quarter.snapshot -> past_year.snapshot`
   A scheduled condensation job advances quarter memory into year memory.
6. `past_year.snapshot -> all_time.snapshot`
   The oldest periodic condensation accumulates durable long-horizon bias in the all-time bucket.
7. `snapshot -> decayed_inference_state`
   Inference-time scales are recomputed from bucket age and configured decay parameters without rewriting frozen artifacts.
