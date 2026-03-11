# Quickstart: Strict Live Memory Runtime

## Goal

Verify that live decode preserves memory adapters during serving and replays the retained suffix after an accepted Active LoRA update.

## Steps

1. Enable runtime memory in the server:

```bash
export VICUNA_ACTIVE_LORA=1
export VICUNA_PAST_LORA=1
```

2. Start the server with a small enough context to force context shift during generation.

3. Send a completion request long enough to:
   - fill the live window
   - trigger context shift
   - produce at least one Active LoRA ingest event

4. Inspect logs for:
   - effective serving-stack rebuilds
   - memory-layer presence during live decode
   - Active LoRA ingestion result
   - strict replay scheduling or skip reason
   - replay completion before the next sampled token continues generation

## Targeted Test Commands

```bash
ctest --test-dir build --output-on-failure -R 'test-active-lora|test-past-lora'
```

## Expected Outcomes

- Request adapter changes do not remove runtime memory layers from live serving.
- Accepted Active writes schedule retained-suffix replay.
- Redundant-span or zero-suffix shifts log replay skip reasons instead of replaying.
- Generation resumes only after replay completes when replay was scheduled.
