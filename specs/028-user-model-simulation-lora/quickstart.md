# Quickstart

1. Build the focused test targets:

```sh
cmake --build /Users/tyleraraujo/vicuna/build-codex --target test-active-lora test-self-state test-cognitive-loop -j4
```

2. Run the focused suite:

```sh
ctest --test-dir /Users/tyleraraujo/vicuna/build-codex --output-on-failure -R 'test-active-lora|test-self-state|test-cognitive-loop'
```

3. Inspect:

- hard-memory archive traces for enriched `USER_MODEL` primitives
- self-state model info for user-preference summary fields
- serving-stack traces during DMN user simulation
- counterfactual traces for candidate message and simulated reply excerpts
