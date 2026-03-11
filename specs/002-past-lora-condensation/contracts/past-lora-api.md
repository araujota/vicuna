# Contract: Past-LoRA Condensation Runtime API

## Purpose

Expose a runtime surface for enabling the frozen temporal memory stack, driving condensation ticks, and inspecting bucket state without binding callers to one frontend.

## Proposed C API

### Configuration

```c
enum llama_memory_lora_bucket {
    LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK = 0,
    LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH = 1,
    LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER = 2,
    LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR = 3,
    LLAMA_MEMORY_LORA_BUCKET_ALL_TIME = 4,
    LLAMA_MEMORY_LORA_BUCKET_COUNT = 5,
};

struct llama_past_lora_params {
    bool     enabled;
    float    host_memory_ratio[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    float    device_memory_ratio[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    uint32_t min_rank[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    uint32_t max_rank[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    float    base_scale[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    uint64_t decay_half_life_us[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    uint64_t condensation_period_us[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    float    merge_source_weight[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    float    merge_target_retention[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    float    gain_max;
    float    singular_value_floor;
};

struct llama_past_lora_bucket_stats {
    bool     populated;
    uint32_t version;
    uint32_t selected_rank;
    uint64_t created_at_us;
    uint64_t source_window_start_us;
    uint64_t source_window_end_us;
    uint64_t host_budget_bytes;
    uint64_t device_budget_bytes;
    float    base_scale;
    float    effective_scale;
    float    gain_mean;
    float    gain_max;
};

struct llama_past_lora_stats {
    bool     enabled;
    uint64_t last_tick_us;
    uint64_t pending_job_mask;
    struct llama_past_lora_bucket_stats buckets[LLAMA_MEMORY_LORA_BUCKET_COUNT];
};
```

### Lifecycle

```c
LLAMA_API struct llama_past_lora_params llama_past_lora_default_params(void);

LLAMA_API int32_t llama_past_lora_init(
        struct llama_context * ctx,
        struct llama_past_lora_params params);

LLAMA_API int32_t llama_past_lora_tick(
        struct llama_context * ctx,
        uint64_t now_us);

LLAMA_API int32_t llama_past_lora_get_stats(
        const struct llama_context * ctx,
        struct llama_past_lora_stats * out_stats);
```

## Behavioral Contract

- `llama_past_lora_init()` returns success only when the runtime can plan fixed budgets for each bucket and allocate the bounded frozen-stack structures without mutating base model weights.
- `llama_past_lora_tick()` recomputes decay scales, marks due condensation jobs, and executes any eligible bucket handoffs in order from younger to older stages.
- `llama_past_lora_tick()` may replace a bucket snapshot only during an explicit condensation event; it must not continuously rewrite past buckets during ordinary inference updates.
- `llama_past_lora_get_stats()` must always reflect the current bucket population state, effective decay scales, and pending condensation jobs after initialization.
- Past-bucket condensation must preserve direction-versus-gain separation and enforce configured gain or singular-value limits.

## Internal Integration Contract

- The existing Active LoRA manager remains the owner of the editable stage and exposes rollover readiness to the past-stack scheduler.
- Runtime adapter allocation remains in `src/llama-adapter.*`.
- Inference continues to apply memory stages through the existing LoRA graph path in `src/llama-graph.cpp`.
- Frontends or runtime loops may call `llama_past_lora_tick()` on explicit ticks to keep the stack decayed and condensed.
