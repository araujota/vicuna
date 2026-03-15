# Quickstart: Sliding-Window Active LoRA Memory

## Goal

Enable a bounded Active LoRA memory stage on a context, then feed discarded tokens from a context-shift event into that stage.

## Steps

1. Create the model and the main inference context as usual.
2. Construct `llama_active_lora_params` from `llama_active_lora_default_params()`.
3. Set `enabled = true` and choose the host/device memory ratios and training window sizes for the current deployment.
4. Call `llama_active_lora_init(ctx, params)`.
5. When a frontend discards tokens from the sliding window, pass the evicted token slice to `llama_active_lora_ingest(ctx, tokens, n_tokens)`.
6. Periodically call `llama_active_lora_get_stats(ctx, &stats)` to inspect budget selection, update counters, and rollover readiness.

## Expected Outcome

- The context now carries an Active LoRA that sits in the inference adapter stack.
- Each admitted eviction span produces an update record and advances the Active LoRA state without changing base model weights.
- The selected rank remains bounded by the configured proportions of currently available host and device memory.

## Verification Focus

- Force a context shift and confirm that `updates_applied` increases.
- Confirm that the selected rank is non-zero and does not exceed the planned budget.
- Switch the embedding type and confirm that the ingestion path still executes through the same API.
