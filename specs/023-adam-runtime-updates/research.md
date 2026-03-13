# Research: Adam For Additional Self-State-Driven Runtime Updates

## Question

Should Adam be inserted into other self-state-driven runtime update paths in
Vicuña beyond the new functional gating MLP?

## Sources Consulted

- Local runtime implementation in
  `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` and
  `/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp`
- GitHub primary sources:
  - `huggingface/peft`
  - `artidoro/qlora`
  - `EricLBuehler/xlora`
  - `adapter-hub/adapters`
- Web primary sources:
  - Adam paper (`arXiv:1412.6980`)
  - LoRA paper (`arXiv:2106.09685`)
  - QLoRA paper (`arXiv:2305.14314`)

## External Findings

### Adam is the default baseline for noisy low-rank adapter training

The original Adam paper motivates the optimizer for stochastic objectives with
noisy and potentially sparse gradients. That matches the character of Vicuña's
online self-state-driven updates better than plain additive writes.

The LoRA and QLoRA literature both train low-rank adapters with Adam-family
optimizers in ordinary practice. GitHub examples in `huggingface/peft` and
`artidoro/qlora` use `AdamW` or paged AdamW for adapter training, which is
strong evidence that adaptive moments are considered normal for low-rank weight
updates.

### Adam fits genuine parameter-update paths, not discrete ranking paths

Vicuña has at least three relevant classes of self-state-driven updates:

1. runtime LoRA tensor mutation
2. temporal write-bias scalar updates
3. counterfactual intervention ranking

Only the first two are actual parameter updates. The counterfactual ladder
scores and ranks interventions; it does not maintain differentiable parameters
that are updated from an optimization objective. Forcing Adam into that path
would be conceptually wrong and would hide explicit runtime policy inside an
optimizer metaphor.

## Local Findings

### Strong candidate: runtime LoRA writes

`train_on_adapter()` in
`/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` currently converts
self-state-derived content and state signals into direct additive updates on
runtime LoRA `A` and `B` tensors.

Why Adam fits:

- the code already computes per-parameter directional deltas
- updates are online, noisy, and non-stationary
- the tensors are low-rank and small enough for explicit per-parameter moment
  state to remain bounded
- Active LoRA and functional family updates both flow through this shared path,
  so one change covers multiple runtime learning mechanisms

Decision: adopt Adam here.

### Strong candidate: temporal write-bias controller

`temporal_encoding_bias_apply()` in
`/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp` updates reward and
dampening scalar biases from signed and efficiency advantages. This is a true
bias-update path driven by self-state outcome deltas.

Why Adam fits:

- the controller updates trainable scalar parameters, not discrete choices
- signals are noisy and can alternate direction
- moment smoothing is useful to avoid threshold-jump behavior
- the state is tiny and trivially bounded

Decision: adopt Adam here.

### Non-candidate: counterfactual LoRA ablation ladder

`compute_counterfactual_trace()` in
`/Users/tyleraraujo/vicuna/src/llama-cognitive-loop.cpp` ranks discrete
candidate interventions, including LoRA ablation proposals.

Why Adam does not fit:

- the path has no differentiable parameters to optimize
- scores are explicit runtime policy, not trainable weights
- replacing it with optimizer state would reduce inspectability without a sound
  objective or gradient

Decision: do not adopt Adam here in this change set.

## Implementation Implications

1. Add explicit typed Adam state for runtime LoRA tensors.
2. Route Active LoRA and functional-family writes through the new optimizer
   path while preserving existing weight decay and gain normalization.
3. Add explicit typed Adam state for temporal reward and dampening biases.
4. Extend public observability surfaces so tests can prove the optimizer paths
   advanced.
5. Document the explicit rejection of Adam for the discrete counterfactual
   ranking ladder.
