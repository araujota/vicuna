# Contract: Memory Runtime Serving Semantics

## Scope

This contract describes the required runtime behavior for live serving composition and strict KV coherence. It is primarily an internal runtime contract spanning:

- `src/llama-context.*`
- `src/llama-graph.*`
- `src/llama-active-lora.*`
- `tools/server/server-context.cpp`

## Serving Composition Contract

### Inputs

- Request-managed LoRA adapters configured for the current request or slot
- Runtime-managed memory adapters owned by the Active/Past memory manager

### Required Behavior

1. The runtime MUST rebuild one effective serving stack whenever either adapter source changes.
2. The effective stack MUST preserve runtime memory layers even when request adapters are updated.
3. The effective stack MUST expose deterministic precedence for:
   - older past buckets
   - newer past buckets
   - active layer
   - request adapters
4. The decode graph MUST use the effective serving stack, not a request-only adapter set.

## Strict Replay Contract

### Trigger

- A generation-time context shift that:
  - evicts text into Active LoRA
  - and produces an accepted Active weight change
  - and leaves a retained suffix

### Required Behavior

1. The runtime MUST invalidate stale retained KV state for the affected suffix.
2. The runtime MUST replay the retained suffix under the updated effective serving stack before further sampling.
3. The runtime MUST invalidate or bypass stale checkpoints captured under the pre-update stack.
4. The runtime MUST reset sampler state to reflect the compacted retained prompt.

### Skip Cases

Replay MUST be skipped when:

- Active ingestion reported no effective weight change
- or the retained suffix length is zero

Skip behavior MUST emit an inspectable reason.

## Observability Contract

The runtime MUST emit logs or trace state for:

- serving-stack rebuilds
- layer ordering and effective scales
- strict replay scheduling
- strict replay completion
- strict replay skip reasons
- strict replay failures
