# Data Model: Typed Persistent Self-State

## SelfState

- `schema_version`
- `updater_version`
- `registers`
- `time_surface`
- `event_anchors`
- `channel_state`
- `identity_surface`
- `goals`
- `commitments`
- `working_memory`
- `memory_handles`
- `reactivation_priorities`
- `tool_jobs`
- `social_state`
- `trace_items`

Invariants:
- one instance per `llama_context`
- owned in CPU-side control code
- inspectable without reading prompt text or LoRA weights

## SelfRegisterDefinition

- `register_id`
- `name`
- `family`
- `default_scalar_value`
- `default_categorical_value`
- `value_min`
- `value_max`
- `uses_scalar_value`
- `updater_version`

Initial implemented register ids:
- `r_uncertainty`
- `r_contradiction`
- `r_novelty`
- `r_topic_shift`
- `r_goal_relevance`
- `r_self_relevance`
- `r_social_relevance`
- `r_affordance`
- `r_broadcast_pressure`
- `r_broadcast_inhibition`
- `r_followup_continuation`
- `r_memory_write_priority`
- `r_time_phase`
- `r_tool_salience`
- `r_channel_state`

## SelfRegisterValue

- `scalar_value`
- `categorical_value`
- `confidence`
- `last_update_wall_ms`
- `last_update_monotonic_ms`
- `source_mask`
- `updater_version`
- `dirty`

Rules:
- bounded scalar registers clamp to `[value_min, value_max]`
- categorical registers use `categorical_value`
- every update stamps both wall-clock and monotonic update times

## SelfTimePoint

- `wall_clock_ms`
- `monotonic_ms`
- `timezone_offset_minutes`

Rules:
- `monotonic_ms` must not move backward
- timezone offset must remain in a sane bounded range

## SelfDatetimeSurface

Raw fields:
- `wall_clock_ms`
- `monotonic_ms`
- `timezone_offset_minutes`
- `local_year`
- `local_month`
- `local_day`
- `local_hour`
- `local_minute`
- `local_second`
- `day_of_week`
- `day_of_year`

Derived cyclic fields:
- `hour_sin`
- `hour_cos`
- `weekday_sin`
- `weekday_cos`
- `year_day_sin`
- `year_day_cos`

Derived elapsed fields:
- `delta_since_last_user_ms`
- `delta_since_last_tool_event_ms`
- `delta_since_last_emit_ms`
- `session_age_ms`

## SelfEventAnchors

- `session_start_wall_ms`
- `session_start_monotonic_ms`
- `last_user_monotonic_ms`
- `last_tool_monotonic_ms`
- `last_emit_monotonic_ms`

Rules:
- unset anchor values use `-1`
- elapsed deltas derive from monotonic time only

## SelfChannelState

Enum-backed categorical state:
- `waiting`
- `active`
- `do_not_interrupt`

Rules:
- mirrored into `r_channel_state`
- updated explicitly through the self-state API

## SelfStateParams

- `enable_learned_contradiction_head`
- `enable_learned_uncertainty_head`
- `enable_learned_broadcast_head`
- `enable_builtin_contradiction_probe`
- `enable_builtin_uncertainty_probe`
- `enable_builtin_broadcast_probe`
- `tool_salience_half_life_ms`
- `prewrite_gain`
- `postwrite_gain`
- `contradiction_head_callback`
- `uncertainty_head_callback`
- `broadcast_head_callback`

Rules:
- callbacks are optional
- disabled or missing callbacks fall back to the analytic path

## SelfStateEvent

- `tokens`
- `n_tokens`
- `role`
- `channel`
- `flags`
- `decoder_entropy`
- `decoder_top_margin`

Roles:
- `user`
- `tool`
- `system`

Channels:
- `primary`
- `counterfactual`

Flags:
- `admitted`
- `tool_completed`
- `tool_failed`
- `emit_followup`

## SelfStateFeatureVector

- `token_count_log`
- `unique_token_ratio`
- `novelty`
- `topic_shift`
- `working_memory_top_similarity`
- `working_memory_similarity_variance`
- `memory_handle_top_similarity`
- `memory_handle_similarity_variance`
- `goal_top_similarity`
- `commitment_top_similarity`
- `identity_similarity`
- `self_reference_ratio`
- `negation_ratio`
- `uncertainty_lexical_ratio`
- `error_ratio`
- `recency_user`
- `recency_tool`
- `recency_emit`
- `tool_readiness_score`
- `tool_pending_pressure`
- `decoder_entropy`
- `decoder_top_margin`
- `contradiction_score`
- `uncertainty_score`
- `memory_write_pressure`
- `broadcast_pressure_hint`
- `broadcast_inhibition_hint`
- `followup_hint`

Rules:
- built separately for prewrite and postwrite phases
- all float outputs are bounded to `[0, 1]` in the initial slice
- counterfactual-channel events share the same typed feature surface but use a distinct broadcast policy

## SelfSketchSurface

- `id`
- `priority`
- `unresolved`
- `last_update_monotonic_ms`
- `sketch[32]`

Rules:
- used for identity-adjacent retrieval surfaces without tying the runtime to one embedder family
- priorities remain bounded to `[0, 1]`

## SelfWorkingMemoryItem

- `event_id`
- `role`
- `flags`
- `salience`
- `unresolved_question`
- `tool_affordance_hint`
- `admitted_monotonic_ms`
- `sketch[32]`

Rules:
- admitted postwrite events enter a fixed-budget ring
- the ring is retrieval-facing state, not prompt text

## SelfMemoryHandle

- `handle_id`
- `kind`
- `priority`
- `last_update_monotonic_ms`
- `member_count`
- `centroid[32]`

Kinds:
- `working_memory_cluster`
- `frozen_bucket`
- `active_memory`
- `external`

Rules:
- handles are typed retrieval namespaces, not prompt-visible memory
- centroid sketches stay normalized and bounded
- frozen-bucket handles are synchronized from real Past LoRA bucket stats during consolidation ticks, not from every admitted message
- frozen-bucket handles remain part of the persistent prefix even after KV eviction

## SelfReactivationInfo

- `handle_id`
- `kind`
- `priority`
- `top_similarity`
- `last_update_monotonic_ms`

Rules:
- values form a sparse keyed priority surface over memory handles
- priorities are recomputed in postwrite against the current event sketch

## SelfToolJob

- `job_id`
- `status`
- `importance`
- `last_update_monotonic_ms`

Statuses:
- `idle`
- `pending`
- `running`
- `completed`
- `failed`

Rules:
- tool-job state is typed control-plane state, not inferred only from embeddings
- job status feeds readiness and pending-pressure features

## SelfSocialState

- `familiarity`
- `trust`
- `reciprocity`
- `bond_strength`
- `user_turn_count`
- `system_turn_count`
- `last_update_monotonic_ms`

Rules:
- represented as bounded persistent scalars rather than free-form text summaries where feasible
- updated only from explicit control-side event transitions
- survives KV eviction and trace replay as part of the self-state prefix

## SelfTraceItem

- `time_point`
- `event`
- `tokens`

Rules:
- trace items preserve the deterministic event/update order used for replay
- replay rebuilds dynamic state from preserved static surfaces plus frozen trace items

## SelfUpdaterProgram

- `version`
- `memory_novelty_weight`
- `memory_working_similarity_weight`
- `memory_handle_similarity_weight`
- `memory_uncertainty_weight`
- `memory_contradiction_weight`
- `memory_handle_variance_weight`
- `broadcast_social_weight`
- `broadcast_contradiction_weight`
- `broadcast_uncertainty_weight`
- `broadcast_tool_pending_weight`
- `broadcast_tool_unready_weight`
- `broadcast_failure_weight`
- `broadcast_question_weight`
- `broadcast_goal_weight`
- `rule_count`
- `rules[]`

## SelfRegisterUpdaterRule

- `register_id`
- `phase_mask`
- `baseline`
- `rise_gain`
- `fall_gain`
- `baseline_pull`
- `feature_ids[]`
- `feature_weights[]`
- `source_register_ids[]`
- `source_register_weights[]`

Rules:
- updater programs combine global feature weights with per-register bounded update rules
- each rule is a constrained scalar dynamical update, not free-form code
- rules may only target bounded scalar registers
- rules may only read typed feature ids and bounded scalar source registers
- invalid versions are rejected explicitly
- invalid rules are rejected explicitly
- counterfactual replay evaluates candidate programs without mutating live traces

## InTreeProbeHeads

- `contradiction_probe`
- `uncertainty_probe`
- `broadcast_probe`

Rules:
- implemented as fixed in-tree linear probes over typed scalar features
- may be blended with analytic scores and then overridden by optional callbacks
- remain CPU-side policy modules, not backend-kernel behavior

## SelfCounterfactualResult

- `updater_version`
- `replay_channel`
- `replayed_events`
- `working_memory_count`
- `reactivation_count`
- `uncertainty`
- `contradiction`
- `memory_write_priority`
- `broadcast_pressure`

Rules:
- results summarize replay outcomes for a candidate updater program
- results are derived from replayed traces, not live prompt text
- results identify which replay channel was used for the simulation
