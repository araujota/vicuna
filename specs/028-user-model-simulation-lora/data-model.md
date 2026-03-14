# Data Model

## User Preference Summary

Bounded self-state summary of the current user model.

Fields:

- `directness_preference`
- `verbosity_preference`
- `structure_preference`
- `clarification_preference`
- `autonomy_preference`
- `disagreement_sensitivity`
- `rhetorical_intensity`
- `preference_confidence`
- `rhetorical_confidence`
- `simulator_readiness`

Semantics:

- values are bounded floats in `[0, 1]`
- confidence fields represent confidence in the estimate, not desirability
- the summary complements existing social and outcome state rather than
  replacing it

## Enriched User Model Primitive

Hard-memory `USER_MODEL` primitive content will encode a bounded summary of:

- relationship residue
- preference residue
- rhetorical residue
- uncertainty or confidence
- provenance tags such as `social`, `preference`, `rhetoric`, or `repair`

These remain compact text summaries inside the existing primitive structure to
avoid changing the external Supermemory payload shape, while retrieval summaries
gain additional typed aggregate signals.

## User Personality Adapter State

Dedicated runtime adapter owned by the active-lora manager.

Fields:

- `enabled`
- `selected_rank`
- `updates_applied`
- `optimizer_step_count`
- `tokens_ingested`
- `gain_mean`
- `gain_max`
- `optimizer_last_update_norm`
- `confidence`

Semantics:

- fixed-size runtime LoRA
- trained only from user-authored evicted spans
- no temporal versioning or bucket rollover

## User Simulation Trace

Typed trace for DMN simulated-user counterfactual pass.

Fields:

- `valid`
- `used_user_personality_adapter`
- `temporal_layers_ablated`
- `candidate_family`
- `candidate_message`
- `simulated_user_reply`
- `message_token_count`
- `reply_token_count`
- `simulation_confidence`
- `pre_simulation_divergence`
- `post_simulation_divergence`
- `signed_self_state_outcome`

Semantics:

- only records bounded excerpts and metrics
- attached to counterfactual traces and DMN traces
- never implies that the reply came from the real user
