# Quickstart: Staged Tool Loop

## Goal

Run all tool-backed work through a family→method→payload controller so the provider touches the runtime at more explicit checkpoints.

## Flow

1. Build the core concierge system prompt.
2. Render family selection from normalized catalog families.
3. Parse strict JSON family choice.
4. Render method selection for the chosen family.
5. Parse strict JSON method choice or `back` or `complete`.
6. Render payload construction for the chosen method.
7. Parse strict JSON payload or `back`.
8. Execute the validated method.
9. Feed the observation back into history.
10. Restart at family selection unless the provider already chose `complete`.

## Validation

1. Run focused staged-loop provider tests.
2. Run the full DeepSeek provider test file.
3. Build `llama-server`.
