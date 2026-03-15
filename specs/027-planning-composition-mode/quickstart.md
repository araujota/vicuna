# Quickstart

After implementation:

1. Run an active-loop tool-seeking episode and inspect the active trace for a
   multi-step plan.
2. Run a DMN tick that chooses internal write plus follow-up and inspect the DMN
   trace for a composed plan.
3. Verify the functional LoRA registry reports the planning/composition family.
4. Run targeted tests:

```sh
ctest --test-dir /Users/tyleraraujo/vicuna/build-codex --output-on-failure -R 'test-active-lora|test-cognitive-loop|test-self-state'
```
