# Research Notes: Hidden-State-Derived Active LoRA Embeddings And Feature-Derived Write Rules

## Scope

This note supports the design in
`specs/014-hidden-state-lora-writes/implementation-approach.md`.

The core question is not whether LoRA memory should replace external memory.
The design assumption is that hard memory remains the declarative/episodic
substrate, while LoRA memory is used to accumulate durable inference-level
behavior and style biases.

## Local Evidence

### Current embedder and write bottlenecks

- The Active LoRA manager currently supports:
  - 64-dim hash embeddings
  - 64-dim token-pool embeddings
  - a callback-backed custom embedder
  (`src/llama-active-lora.cpp:130-279`).
- Current target tensors are biased toward the last four transformer layers and
  include attention plus MLP tensors (`src/llama-active-lora.cpp:102-123`).
- Current write behavior is simple:
  - select one rank slot by `updates_seen % rank`
  - update LoRA `A` from the span embedding
  - update LoRA `B` from token IDs modulo 1024
  - renormalize and adjust gain
  (`src/llama-active-lora.cpp:860-909`).
- This means current rank expansion mostly increases slot capacity and slows
  overwrite pressure; it does not solve semantic aliasing in the write signal.

### Existing direction/gain and condensation surfaces

- The current implementation already carries a direction/gain-style
  representation:
  - low-rank directions can be read and written
  - merged
  - normalized
  - and clipped by gain
  (`src/llama-active-lora.cpp:347-401`,
  `src/llama-active-lora.cpp:590-717`,
  `src/llama-active-lora.cpp:720-748`).
- Past-bucket condensation already depends on those surfaces and should remain
  compatible with any embedder/write-rule change
  (`src/llama-active-lora.cpp:955-971`).

### Existing hidden-state extraction surfaces

- The local runtime already exposes embedding outputs through:
  - `llama_set_embeddings(...)`
  - `llama_get_embeddings_ith(...)`
  - `llama_get_embeddings_seq(...)`
  (`include/llama.h:1963-2015`,
  `src/llama-context.cpp:782-820`,
  `src/llama-context.cpp:3191-3242`).
- `llama-server` already uses those embedding outputs in embedding-oriented task
  paths (`tools/server/server-context.cpp:1633-1663`).
- That makes a minimally invasive base-model-derived embedder possible without
  inventing a completely separate external embedding model.

### Existing parity invariants that must not break

- Serving-layer order is fixed and regression-tested:
  `request -> all_time -> year -> quarter -> month -> week -> active`
  (`include/llama.h:433-439`,
  `tests/test-serving-lora-stack.cpp:99-155`).
- Counterfactual LoRA ablation is a first-class family and must remain valid
  after writer changes (`include/llama.h:919-925`,
  `tests/test-cognitive-loop.cpp:483-505`,
  `src/llama-cognitive-loop.cpp:427-548`).
- Remediation currently routes through `active_lora_remediate(...)` and must
  stay behaviorally aligned with ordinary Active writes
  (`src/llama-cognitive-loop.cpp:1173-1181`).

## External Primary Sources

### PEFT methods

- [LoRA](https://arxiv.org/abs/2106.09685)
  establishes low-rank adaptation as a compact update space.
- [AdaLoRA](https://arxiv.org/abs/2303.10512)
  reallocates rank budget adaptively over training, using a staged schedule and
  sensitivity-aware rank allocation.
- [DoRA](https://arxiv.org/abs/2402.09353)
  decomposes weights into magnitude and direction, and official PEFT docs and
  examples emphasize its strength especially at lower ranks.
  Relevant code/docs:
  - [PEFT DoRA example README](https://github.com/huggingface/peft/blob/main/examples/dora_finetuning/README.md)
- [QLoRA](https://arxiv.org/abs/2305.14314)
  shows how to fine-tune adapters against a frozen 4-bit quantized base model
  using NF4, double quantization, and paged optimizers.
  Relevant repo:
  - [artidoro/qlora](https://github.com/artidoro/qlora)

### Memory systems using hidden or latent state

- [LongMem](https://arxiv.org/abs/2306.07174)
  augments language models with long-term memory via retrieved memory and
  side-network fusion. It is relevant as a hidden-state-aware memory reference,
  but not a direct LoRA-memory template.
  Repo:
  - [Victorwz/LongMem](https://github.com/Victorwz/LongMem)
- [MemoryLLM](https://arxiv.org/abs/2402.04624)
  and [M+](https://arxiv.org/abs/2502.00592)
  are direct comparisons for self-updatable or latent memory inside the model
  rather than only external retrieval.
  Repo:
  - [wangyu-ustc/MemoryLLM](https://github.com/wangyu-ustc/MemoryLLM)

## External Findings

### What is relevant from AdaLoRA

- Relevant:
  - sensitivity-aware allocation
  - adaptive rank budget across layers
  - distinguishing initial capacity from later compacted capacity
- Not directly portable:
  - training-step schedules such as `tinit`, `tfinal`, and `total_step`
  - frequent optimizer-driven reallocation on every step
- Vicuña translation:
  use periodic runtime rebudgeting or sensitivity accumulation, not literal
  SGD-training schedules.

### What is relevant from DoRA

- Highly relevant:
  - explicit separation of magnitude from direction
  - better low-rank behavior when rank is small
  - clearer control over update strength
- Vicuña already has a partial analogue through explicit `gain` and normalized
  direction recompression.
- The missing piece is that the writer still derives direction poorly.

### What is relevant from QLoRA

- Relevant:
  - use of a frozen quantized base for auxiliary or shadow operations
  - NF4/double-quant ideas for memory-constrained auxiliary contexts
- Not directly relevant:
  - paged optimizers for large gradient-based training loops
  - treating the primary live serving path as a gradient-finetuning workflow
- Vicuña translation:
  quantization is a possible implementation aid for an auxiliary hidden-state
  extraction context, not the main memory-writing idea.

## Synthesis

### Main design conclusion

The next move should be:

1. replace the default hash/token-pool embedder with a base-model-derived
   hidden-state embedder
2. split admission embedding from write features
3. replace token-modulo write directions with feature-derived directions
4. keep direction/gain separation
5. adopt AdaLoRA ideas only for periodic per-layer budget allocation
6. treat QLoRA as optional support for an auxiliary context, not as the live
   runtime-memory core

### Architectural implication

The least invasive path is to use current embedding outputs from the serving
model family first. A deeper late-layer-tap design can follow if the first path
shows that final output embeddings are not rich enough.
