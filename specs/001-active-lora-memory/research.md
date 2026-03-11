# Research: Sliding-Window Active LoRA Memory

## Decision: Use a runtime-generated mutable LoRA instead of a prompt summary buffer

**Rationale**:
- The repository already applies LoRAs centrally through `llm_graph_context::build_lora_mm`, so a runtime-generated adapter can enter the existing inference stack without patching each model backend.
- A mutable adapter preserves the architecture’s requirement that the first durable stage after context eviction is an Active LoRA rather than more prompt text.
- The existing `ggml_opt` training path offers a credible write mechanism for evicted spans using next-token loss, which is closer to LoRA’s intended use than a heuristic projection-only memory layer.

**Alternatives considered**:
- Prompt-summary stuffing: rejected because it violates the memory cascade and typed-state principles.
- Control-vector-only memory: rejected because a control vector is not a LoRA stage and does not satisfy the requested inference-stack placement.
- Projection-only synthetic LoRA updates: rejected as the primary writer because they are harder to justify and audit than training an explicit low-rank adapter on evicted spans.

## Decision: Train the Active LoRA through a shadow context that shares the same model

**Rationale**:
- `llama_opt_epoch()` clears the context memory during training, so reusing the serving context would destroy the live KV state.
- A second `llama_context` created from the same model can train the shared Active LoRA weights while the inference context keeps its current memory intact.
- This keeps the CPU-side policy and scheduling separate from backend math while reusing the current optimizer and graph path.

**Alternatives considered**:
- Train inside the serving context: rejected because it would corrupt the active prompt/KV state.
- Save and restore serving state around each write: rejected as more fragile and more expensive than maintaining a dedicated training context.

## Decision: Size the Active LoRA rank from live host and device memory budgets

**Rationale**:
- The codebase already exposes free and total memory through ggml backend device APIs and already uses free-memory-based device splitting for model loading.
- A rank planner can convert a per-rank byte cost for selected target tensors into the largest rank that fits both the configured host-memory ratio and the configured aggregate device-memory ratio.
- This satisfies the “fixed-size as a proportion of available RAM/VRAM” requirement while remaining model- and machine-dependent.

**Alternatives considered**:
- Fixed adapter byte size: rejected because the user explicitly forbids it.
- Fixed LoRA rank across all systems: rejected because the same rank has very different memory costs across model sizes and devices.

## Decision: Start with a curated cross-architecture target tensor subset

**Rationale**:
- `llama_layer` already exposes common attention and feed-forward tensors across many architectures.
- A curated subset such as attention projections plus core feed-forward tensors gives meaningful write capacity while keeping budget planning, training cost, and portability tractable.
- The plan leaves room to widen the target set later without changing the budget model or public manager interface.

**Alternatives considered**:
- All model tensors: rejected as too expensive and harder to bound tightly.
- One tensor only: rejected as too narrow to serve as the default durable memory stage.

## Decision: Make span embeddings pluggable and use them for admission, tracing, and future writer selection

**Rationale**:
- The architecture docs already separate embeddings from higher-level control and memory policy.
- Different open-source models may not share one embedding space, so the eviction pipeline needs an interface boundary rather than a hard-coded embedding assumption.
- A pluggable embedder lets the runtime start with local strategies and later swap to an external or model-family-specific embedder without redesigning the Active LoRA manager.

**Alternatives considered**:
- No embedding stage at all: rejected because the user explicitly requested a swappable embedding strategy.
- Hard-code one model-specific embedding path: rejected because it would not generalize across incompatible model families.

## Decision: Expose rollover readiness and audit records now, even before a full past-LoRA stack exists

**Rationale**:
- The working paper and architecture document require frozen past LoRAs after rollover.
- The current implementation scope can deliver Active LoRA budgeting, writing, and boundary tracking before fully wiring past-LoRA retrieval and weighting.
- Recording rollover metadata now preserves auditability and makes the future past-stack implementation additive instead of disruptive.

**Alternatives considered**:
- Delay all rollover state until past-LoRA retrieval exists: rejected because it hides a critical memory-cascade boundary.

## External Sources Consulted

- LoRA original paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Long-horizon compressed-memory reference: [Compressive Transformers for Long-Range Sequence Modelling](https://openreview.net/forum?id=SylKikSYDH)
- Long-horizon memory reference: [Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention](https://arxiv.org/abs/2404.07143)
- Official llama.cpp embedding example reference: [llama.cpp embedding example README](https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/README.md)
