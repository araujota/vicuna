# Quickstart: Validate Adam Expansion For Runtime Updates

## Build

```sh
cmake --build build-codex --target test-active-lora test-cognitive-loop -j4
```

## Run targeted tests

```sh
ctest --test-dir build-codex --output-on-failure -R "test-active-lora|test-cognitive-loop"
```

## What to verify

- Active LoRA stats now report optimizer advancement after runtime writes
- functional-family updates still pass because they share the optimizer-backed
  runtime writer
- temporal encoding bias exposes optimizer advancement after temporal
  self-improvement
- counterfactual ladder tests still behave as explicit ranking logic, not as a
  learned optimizer path
