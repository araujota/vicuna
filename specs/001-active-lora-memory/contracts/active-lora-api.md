# Contract: Active LoRA Runtime API

## Purpose

Expose a minimal runtime surface for enabling, feeding, and inspecting the Active LoRA memory stage without binding callers to one embedding implementation or one frontend.

## Proposed C API

### Configuration

```c
enum llama_active_lora_embedding_type {
    LLAMA_ACTIVE_LORA_EMBEDDING_HASH = 0,
    LLAMA_ACTIVE_LORA_EMBEDDING_TOKEN_POOL = 1,
};

typedef bool (*llama_active_lora_embedding_callback)(
    const struct llama_context * ctx,
    const llama_token * tokens,
    size_t n_tokens,
    float * out_embedding,
    size_t n_embedding,
    void * user_data);

struct llama_active_lora_params {
    bool     enabled;
    float    host_memory_ratio;
    float    device_memory_ratio;
    uint32_t min_rank;
    uint32_t max_rank;
    uint32_t train_context_tokens;
    uint32_t train_stride_tokens;
    uint32_t max_updates_before_rollover;
    float    adapter_scale;
    float    learning_rate;
    float    weight_decay;
    uint32_t embedding_dim;
    int32_t  embedding_type;
    llama_active_lora_embedding_callback embedding_callback;
    void *   embedding_callback_user_data;
};

struct llama_active_lora_stats {
    bool     enabled;
    bool     rollover_ready;
    uint32_t selected_rank;
    uint32_t updates_applied;
    uint64_t tokens_ingested;
    float    host_memory_ratio;
    float    device_memory_ratio;
    uint64_t host_budget_bytes;
    uint64_t device_budget_bytes;
    uint32_t embedding_dim;
    bool     embedding_is_custom;
    int32_t  embedding_type;
};
```

### Lifecycle

```c
LLAMA_API struct llama_active_lora_params llama_active_lora_default_params(void);

LLAMA_API int32_t llama_active_lora_init(
        struct llama_context * ctx,
        struct llama_active_lora_params params);

LLAMA_API int32_t llama_active_lora_ingest(
        struct llama_context * ctx,
        const llama_token * tokens,
        size_t n_tokens);

LLAMA_API int32_t llama_active_lora_get_stats(
        const struct llama_context * ctx,
        struct llama_active_lora_stats * out_stats);
```

## Behavioral Contract

 - `llama_active_lora_init()` returns success only when the runtime can compute a valid fixed-size budget, select a non-zero rank, and allocate the mutable adapter within that budget.
- `llama_active_lora_ingest()` accepts an evicted token span, performs embedding-based admission, updates the Active LoRA if the span is accepted, and records an inspectable update result.
- `llama_active_lora_get_stats()` must be safe to call after initialization regardless of whether any writes have occurred.
- Initialization must not mutate base model weights.
- Callers may use built-in embedders or provide a custom embedding callback without changing the surrounding ingestion API.

## Internal Integration Contract

- `src/llama-context.*` owns the manager lifetime.
- `src/llama-adapter.*` provides mutable runtime adapter allocation hooks.
- `src/llama-graph.cpp` continues to apply Active LoRA through the existing LoRA path.
- Frontends such as the server may call `llama_active_lora_ingest()` when their context-shift logic discards prompt tokens.
