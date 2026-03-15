# Quickstart: Hard Memory Primitives

## Goal

Verify that Vicuña archives typed hard-memory primitives and uses typed query
metadata to improve retrieval cooperation with self-state and LoRA bias.

## Steps

1. Configure hard memory with a Supermemory endpoint or the native mock server
   tests.
2. Drive a user or tool event through `llama_self_state_apply_postwrite()`.
3. Run an active-loop or DMN cycle that settles with meaningful traces.
4. Confirm the last hard-memory archive trace reports multiple primitives and
   typed kinds.
5. Run `llama_hard_memory_query()` against typed mock results.
6. Confirm:
   - typed hit metadata is parsed
   - retrieval summary is non-zero
   - self-model promotion still works
   - functional-gating tests still pass

## Validation Commands

```sh
cmake --build /Users/tyleraraujo/vicuna/build-codex --target test-self-state test-cognitive-loop test-active-lora -j4
ctest --test-dir /Users/tyleraraujo/vicuna/build-codex --output-on-failure -R 'test-self-state|test-cognitive-loop|test-active-lora'
```
