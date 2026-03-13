# Data Model: Expanded Self-Model For Efficiency-Oriented Improvement

## Design Principle

Do not flatten the entire expansion into one larger `llama_self_register_id`
enum. Preserve the existing bank as a fast loop-control surface and add grouped
typed profiles on top of it.

## Layered Representation

### Layer 0: Fast Control Register Bank

Purpose:

- cheap scalar updates
- immediate loop routing
- parity with current Active Loop and DMN logic

Recommended changes:

- keep all existing registers
- add only the most routing-critical new scalars, for example:
  - `LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK`
  - `LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE`
  - `LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY`
  - `LLAMA_SELF_REGISTER_RECOVERY_URGENCY`
  - `LLAMA_SELF_REGISTER_ANSWERABILITY`
  - `LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY`

These remain summary controls, not the full representation.

### Layer 1: Typed Self Profiles

Add bounded structs for coherent semantic domains.

```c
enum llama_self_profile_id {
    LLAMA_SELF_PROFILE_GOAL_PROGRESS = 0,
    LLAMA_SELF_PROFILE_USER_OUTCOME,
    LLAMA_SELF_PROFILE_EPISTEMIC,
    LLAMA_SELF_PROFILE_EFFICIENCY,
    LLAMA_SELF_PROFILE_RECOVERY,
    LLAMA_SELF_PROFILE_STRATEGY,
    LLAMA_SELF_PROFILE_SELF_IMPROVEMENT,
};
```

```c
enum llama_self_horizon_id {
    LLAMA_SELF_HORIZON_INSTANT = 0,
    LLAMA_SELF_HORIZON_SHORT,
    LLAMA_SELF_HORIZON_LONG,
};
```

### Layer 2: Forecasts And Prediction Error

Add explicit bounded forecast structs that can be updated after each loop or
event to compare expected versus observed outcome.

## Proposed Typed Profiles

### Goal Progress Profile

```c
struct llama_self_goal_progress_profile {
    float goal_progress_estimate;
    float blocker_severity;
    float dependency_readiness;
    float urgency;
    float expected_next_action_gain;
    float commitment_slippage_risk;
    float confidence;
};
```

Semantics:

- Tracks whether the system is converging on the user's task.
- Makes "progress" explicit instead of inferring it only from goal similarity.

### User Outcome Profile

```c
struct llama_self_user_outcome_profile {
    float satisfaction_estimate;
    float frustration_risk;
    float misunderstanding_risk;
    float trust_repair_need;
    float preference_uncertainty;
    float cognitive_load_estimate;
    float autonomy_tolerance_estimate;
    float confidence;
};
```

Semantics:

- Separates likely user outcome from raw social valence.
- Supports later evaluator training without changing the runtime state schema.

### Epistemic Profile

```c
struct llama_self_epistemic_profile {
    float answerability;
    float evidence_sufficiency;
    float ambiguity_concentration;
    float self_estimate_confidence;
    float tool_need_confidence;
    float contradiction_load;
    float uncertainty_load;
};
```

Semantics:

- Distinguishes "I do not know", "I have weak evidence", and
  "I may need a tool or clarifying question".

### Efficiency Profile

```c
struct llama_self_efficiency_profile {
    float expected_steps_remaining;
    float expected_inference_cost_remaining;
    float loop_inefficiency;
    float repetition_risk;
    float context_thrash_risk;
    float tool_roundtrip_cost;
    float response_compaction_opportunity;
};
```

Semantics:

- Explicitly represents the user's efficiency objective.
- Gives the runtime a place to reason about step count, not only correctness.

### Recovery Profile

```c
struct llama_self_recovery_profile {
    float favorable_divergence_goal;
    float favorable_divergence_social;
    float favorable_divergence_epistemic;
    float favorable_divergence_action;
    float recovery_momentum;
    float regulation_debt;
    float unresolved_tension_load;
    float recovery_cost_estimate;
};
```

Semantics:

- Decomposes favorable-state return into families instead of one aggregate gap.
- Supports later remediation policies without changing the concept of favorable
  state.

### Strategy Profile

```c
struct llama_self_strategy_profile {
    float answer_bias;
    float ask_bias;
    float act_bias;
    float wait_bias;
    float exploit_bias;
    float deliberate_bias;
    float write_internal_bias;
    float act_external_bias;
};
```

Semantics:

- Makes current strategic posture explicit and inspectable.
- Useful for both debugging and later evaluator shaping.

### Self-Improvement Profile

```c
struct llama_self_improvement_profile {
    float update_worthiness;
    float expected_gain;
    float evidence_deficit;
    float reversibility;
    float blast_radius_risk;
    float observability_deficit;
    float readiness;
};
```

Semantics:

- Provides explicit motivation surfaces for future self-improvement systems.
- Keeps governance separate from user-facing action urgency.

## Horizon Slices

Each profile should be represented at three timescales.

```c
template <typename T>
struct llama_self_horizon_slice {
    int32_t horizon_id;
    int64_t last_update_monotonic_ms;
    T value;
};
```

Recommended interpretation:

- `INSTANT`: current event-local estimate
- `SHORT`: short EMA over recent events or loop iterations
- `LONG`: persistent baseline or slower EMA

The public C API can avoid templates by defining concrete structs.

## Forecast And Prediction Error

```c
struct llama_self_forecast_trace {
    float predicted_steps_remaining;
    float predicted_inference_cost_remaining;
    float predicted_satisfaction_delta;
    float predicted_recovery_delta;
    float predicted_goal_progress_delta;
    float confidence;
};
```

```c
struct llama_self_prediction_error_trace {
    float steps_error;
    float inference_cost_error;
    float satisfaction_error;
    float recovery_error;
    float goal_progress_error;
    int64_t observed_after_monotonic_ms;
};
```

Purpose:

- Allows later calibration and evaluator layers to compare forecast versus
  outcome.
- Keeps self-estimation explicit and inspectable.

## Integration With Existing Surfaces

### Favorable State

Do not replace `llama_favorable_state_profile`.

Instead:

- keep the existing favorable profile as the compact public summary
- derive it from the richer recovery, social, epistemic, and actionability
  profiles
- optionally add new favorable dimensions later only after the richer profile
  layer exists

### Active Loop And DMN

Use the new profiles as inputs into existing decision logic, but keep the
output trace shapes stable where possible.

Examples:

- `winner_action` still lives in `llama_active_loop_trace`
- new profile summaries become additional trace fields or side-channel APIs

### Import / Export / Replay

All new profiles should remain:

- bounded
- serializable
- versioned
- optional for backward compatibility

## Update Order

Recommended order inside self-state update flow:

1. Compute current event-local feature vector.
2. Update fast control registers as today.
3. Update typed profile instant slices.
4. Update short and long horizon slices.
5. Emit forecasts for steps, recovery, and satisfaction.
6. On later observation, compute prediction-error traces.

This preserves the current loop-critical path while adding richer state above
it.
