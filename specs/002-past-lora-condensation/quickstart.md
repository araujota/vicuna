# Quickstart: Past-LoRA Condensation Stack

## Prerequisites

- Active LoRA memory is already available in the current branch history.
- Build with tests enabled:

```bash
cmake -S /Users/tyleraraujo/vicuna -B /Users/tyleraraujo/vicuna/build -DLLAMA_BUILD_TESTS=ON
cmake --build /Users/tyleraraujo/vicuna/build -j4 --target test-active-lora test-past-lora
```

## Runtime Smoke Test

1. Initialize a model and context.
2. Enable Active LoRA with a small rollover threshold.
3. Enable the past stack with default temporal-bucket parameters.
4. Ingest enough evicted spans to mark Active as rollover-ready.
5. Call `llama_past_lora_tick(ctx, now_us)` with a monotonic timestamp.
6. Query `llama_past_lora_get_stats()` and confirm `past_week` is populated.
7. Advance `now_us` across later condensation boundaries and call `llama_past_lora_tick()` again.
8. Confirm that older buckets populate in order and that every populated bucket exposes a non-negative `effective_scale`.

## Expected Verification

- Active writes maintain bounded gain values after ingestion.
- `past_week` becomes populated after the first rollover tick.
- Later ticks move condensed state into `past_month`, `past_quarter`, `past_year`, and `all_time`.
- Frozen bucket versions only change during explicit condensation ticks.
- Repeated ticks without a due job leave populated bucket versions unchanged.
