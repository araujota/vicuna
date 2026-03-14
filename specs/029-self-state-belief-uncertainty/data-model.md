# Data Model: Partial-Observation Self-State Belief Gradient

## Core Runtime Types

### `llama_self_belief_params`

Purpose:

- Configure the bounded belief layer.

Fields:

- `enabled`
- `latent_slot_count`
- `residual_decay`
- `pressure_clip`
- `confidence_floor`
- `promotion_threshold`
- `max_update_step`
- `missing_observation_weight`
- `unmodeled_care_weight`
- `forecast_error_weight`
- `counterfactual_miss_weight`
- `memory_residue_weight`

## `llama_self_belief_slot`

Purpose:

- Track one bounded hypothesized hidden concern class.

Fields:

- `pressure`
- `confidence`
- `novelty_support`
- `memory_support`
- `forecast_error_support`
- `last_update_step`

Invariants:

- all fields clipped to bounded numeric ranges
- no free-form text payload
- fixed count per context

## `llama_self_belief_summary`

Purpose:

- Fixed-width summary exposed to gain control and traces.

Fields:

- `known_care_uncertainty`
- `missing_observation_uncertainty`
- `unmodeled_care_uncertainty`
- `residual_allostatic_pressure`
- `promotion_readiness`
- `belief_entropy`
- `belief_confidence`
- `max_slot_pressure`
- `slot_pressure_mean`

Invariants:

- all values normalized into stable bounded ranges
- can be serialized into public stats
- does not expose variable-length slot internals to the gating MLP

## `llama_self_belief_evidence`

Purpose:

- Capture one settled update tuple for belief filtering.

Fields:

- `snapshot_before`
- `snapshot_after`
- `predicted_delta`
- `realized_delta`
- `counterfactual_miss`
- `retrieval_support`
- `tool_outcome_miss`
- `user_outcome_miss`
- `unexplained_delta_mass`

## `llama_self_model_promotion_candidate`

Purpose:

- Bridge latent residue into explicit self-model extension proposals.

Fields:

- `candidate_kind`
- `support_score`
- `allostatic_relevance`
- `memory_anchor_id`
- `suggested_label`
- `suggested_desired_value`
- `stability_score`

## Control-Path Integration

### Observation Layer

- existing self-state registers
- existing self-model profiles and extension summaries

### Belief Layer

- belief slots
- belief summary
- promotion candidates

### Gain Layer

Input vector becomes:

- explicit self-state gradient features
- allostatic distance
- existing loop/tool/memory features
- belief summary features

The gain layer never consumes raw belief slots directly.
