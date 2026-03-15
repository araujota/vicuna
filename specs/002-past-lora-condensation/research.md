# Research: Frozen Past-LoRA Condensation Stack

## Decision: Keep one explicit frozen snapshot per temporal bucket and apply all populated buckets concurrently during inference

**Rationale**:
- The current `llama.cpp` runtime already supports multiple LoRAs in the inference stack with independent scales, so a five-bucket temporal stack can reuse existing graph application logic without model-specific graph rewrites.
- Vicuña’s architecture requires auditable frozen stages, which is better served by named bucket snapshots than by one opaque merged long-term adapter.
- One snapshot per bucket keeps the first implementation bounded while preserving the causal chain `active -> past_week -> past_month -> past_quarter -> past_year -> all_time`.

**Alternatives considered**:
- Keep an unbounded list of frozen artifacts per bucket: rejected for the first implementation because memory accounting and retrieval policy become much harder to bound.
- Merge all past memory into one monolithic adapter: rejected because it destroys temporal inspectability and conflicts with the working paper’s staged stack.

## Decision: Use DoRA-style direction-versus-gain decomposition for both Active updates and frozen artifacts

**Rationale**:
- DoRA explicitly separates weight magnitude from directional adaptation, which matches the requirement that memory updates should mostly move direction while tightly constraining size.
- A normalized low-rank direction plus explicit gain scalar keeps update strength inspectable and clip-able.
- The same representation works for both live Active writes and frozen-bucket condensation outputs, avoiding one format for recent memory and another for long-term memory.

**Alternatives considered**:
- Raw LoRA factor updates only: rejected because magnitude can drift implicitly and becomes harder to audit or constrain.
- Full dense DoRA magnitude vectors per weight matrix: rejected for the first implementation because it adds more memory overhead than needed for an initial bounded system.

## Decision: Condense buckets by additive low-rank composition followed by bounded low-rank recompression

**Rationale**:
- The PEFT merge literature and tooling show that adapter merging can be expressed as weighted low-rank composition plus recompression rather than retraining from scratch.
- For Vicuña, condensation is a periodic snapshot operation, so a more expensive but auditable merge is acceptable as long as it stays bounded.
- A small-matrix recompression path avoids materializing the entire base model or retraining a new adapter and preserves a fixed rank per bucket.

**Alternatives considered**:
- Retrain an older bucket from raw past data: rejected because the architecture wants frozen auditable artifacts and bounded periodic jobs, not a second training pipeline.
- Simple slot-wise averaging of LoRA factors: rejected because factor spaces are non-identifiable and naive averaging can destroy the represented direction.

## Decision: Use deterministic pruning and singular-value thresholding for condensation noise control instead of stochastic merge policies by default

**Rationale**:
- TIES identifies sign disagreement and noisy small updates as major causes of merge interference, and DARE shows that dropping and rescaling can improve robustness.
- For Vicuña, auditability matters more than stochastic merge diversity, so a deterministic policy based on thresholded low-energy components and bounded gains is easier to reason about and replay.
- Singular-value thresholding on the recompression step provides a natural deterministic way to drop weak condensed directions while staying within fixed rank.

**Alternatives considered**:
- Random drop-and-rescale as the default merge rule: rejected because it is less auditable and introduces randomness into long-term memory formation.
- No pruning at all: rejected because older buckets would accumulate low-value noise and gain inflation.

## Decision: Drive condensation from explicit runtime ticks and due-job state instead of hidden background mutation

**Rationale**:
- The current codebase does not yet have the full DMN loop, but it does have server and runtime control paths that can call explicit tick functions.
- An explicit tick with inspectable due-state matches the architecture requirement that policy remain CPU-side, inspectable, and auditable.
- This lets the same logic work in tests, CLI-like runtimes, and server loops without depending on wall-clock threads hidden behind the backend.

**Alternatives considered**:
- Dedicated background thread inside the low-level runtime: rejected because it obscures scheduling policy and complicates deterministic tests.
- Only condense on process restart or manual commands: rejected because the user explicitly asked for periodic jobs.

## Decision: Use deterministic time-decay scales per bucket during normal inference

**Rationale**:
- The working paper requires time-decayed influence for past buckets but forbids destructive rewriting of frozen artifacts.
- The existing inference stack already multiplies each adapter by a scale, so time decay is naturally implemented as an explicit per-bucket effective scale.
- Deterministic decay parameters keep the policy visible and easy to test while leaving room for future learned decay surfaces.

**Alternatives considered**:
- Rewriting older buckets to simulate decay: rejected because past artifacts must remain frozen.
- One constant scale for all frozen buckets: rejected because it ignores temporal separation and weakens the intended memory hierarchy.

## Decision: Keep frozen bucket replacement atomic and explicit rather than mutating frozen units incrementally

**Rationale**:
- The architecture permits condensation into older time artifacts but insists that past LoRAs are not updated live like the Active LoRA.
- Replacing a bucket snapshot only during a logged condensation job preserves that invariant better than letting evicted spans continuously touch old buckets.
- Atomic replacement also creates clean version boundaries for audit and tests.

**Alternatives considered**:
- Incrementally writing directly into frozen buckets on every Active update: rejected because it violates the frozen-stage model.

## External Sources Consulted

- DoRA paper: [Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- TIES paper: [Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)
- DARE paper: [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)
- PEFT merge methods and utilities: [Hugging Face PEFT model merging docs](https://huggingface.co/docs/peft/developer_guides/model_merging)
- O-LoRA continual-learning reference: [O-LoRA: Orthonormal Low-Rank Adaptation of Large Language Models](https://aclanthology.org/2026.findings-naacl.45/)
- LoRI interference reference: [LoRI: Reducing Cross-Task Modality Misalignment in Multi-Task LoRA](https://arxiv.org/abs/2410.17633)
