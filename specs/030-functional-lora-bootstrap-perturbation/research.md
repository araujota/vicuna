# Research: Functional LoRA Bootstrap Perturbation

## Question

Do Vicuña functional LoRAs already behave like "no-op learned adapters plus decaying stochastic bootstrap perturbation," and if not, what is the cleanest production-ready way to make them do so?

Short answer: not yet. Today they are true no-op learned adapters, but the stochasticity lives in the gating MLP output rather than in the functional LoRA effect itself. The clean fix is to keep the learned adapter zeroed and add a separate tiny bootstrap adapter per family with sampled signed scale that decays toward a nonzero floor as usage increases.

## Local Findings

### The Learned Functional Adapters Already Start As True No-Ops

Relevant local files:

- `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- `/Users/tyleraraujo/vicuna/src/llama-adapter.cpp`

Observed behavior:

- `create_runtime_adapter(...)` calls `zero_adapter(...)` immediately after runtime allocation.
- `zero_adapter(...)` zeroes both LoRA tensors and sets per-weight gain to `0.0f`.
- Functional families therefore begin with no learned effect at all.

Implication:

- The "functional LoRA begins as a no-op" part of the requested behavior is already true.

### Current Exploration Is In Gain Routing, Not In The Functional LoRA Effect

Relevant local files:

- `/Users/tyleraraujo/vicuna/src/llama-active-lora.cpp`
- `/Users/tyleraraujo/vicuna/include/llama.h`

Observed behavior:

- `initialize_functional_defaults(...)` configures `exploration_noise_initial_std`, `exploration_noise_min_std`, and `exploration_noise_decay_invocations`.
- `predict_activation(...)` samples Gaussian noise into the gain predicted by the gating MLP.
- If the functional adapter remains zeroed, noisy gain alone cannot create a functional bias from the adapter itself.

Implication:

- Vicuña currently explores *whether* to apply a family, but not *what accidental bias* that family might express before it has learned anything.
- The user’s requested semantics are therefore only half implemented.

### The Existing Runtime Stack Supports A Separate Bootstrap Adapter Cleanly

Observed behavior:

- Runtime LoRAs are attached by explicit adapter pointer and role through `attach_adapter_runtime(...)`.
- Multiple runtime adapters can coexist in the stack as long as they are distinct pointers.
- Functional families already have per-family activation, ablation, trace, and hold policy in CPU-side control code.

Implication:

- The cleanest implementation is not to make the learned adapter nonzero.
- It is to add a second adapter per family that carries only bootstrap perturbation.

## GitHub Findings

### The Original Microsoft LoRA Implementation Starts From A No-Op Effective Delta

Source:

- `microsoft/LoRA` [`loralib/layers.py`](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py)

Observed pattern:

- The original LoRA implementation initializes one factor with a standard initializer and the other factor to zero so the effective low-rank delta begins at zero.

Implication for Vicuña:

- Keeping the learned functional adapter as an initial no-op is consistent with canonical LoRA behavior.
- We should not smuggle exploration in by making the learned adapter itself nonzero.

### Hugging Face PEFT Preserves The Same No-Op LoRA Initialization Contract

Source:

- `huggingface/peft` [`src/peft/tuners/lora/layer.py`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)

Observed pattern:

- PEFT’s default reset path initializes one LoRA factor and zeros the other, again keeping the effective delta at startup as a no-op.
- Alternative initializations exist, but the default behavior still preserves the "start from no effective delta" contract.

Implication for Vicuña:

- The right place for exploration is an explicit auxiliary mechanism, not hidden inside the learned adapter initialization.

## Web / Paper Findings

### Parameter-Space Noise Is The Right Analogy For Early Accidental Discovery

Sources:

- Plappert et al., "Parameter Space Noise for Exploration" [arXiv](https://arxiv.org/abs/1706.01905)
- Fortunato et al., "Noisy Networks for Exploration" [arXiv](https://arxiv.org/abs/1706.10295)

Relevant point:

- Exploration can come from stochastic perturbations to model parameters or parameterized function outputs rather than only from action-space randomness.
- Decaying but nonzero perturbation preserves continued discovery while reducing interference as learning matures.

Implication for Vicuña:

- A tiny signed bootstrap perturbation on the functional-LoRA effect is conceptually aligned with parameter-space exploration.
- The perturbation should be explicit, bounded, and decay with family usage while keeping a positive floor.

## Recommended Architecture

### Keep The Learned Functional Adapter Zeroed

Do not change the current learned-adapter initialization contract.

Why:

- It is canonical LoRA behavior.
- It preserves the meaning of runtime updates.
- It keeps "learned state" and "bootstrap exploration" separate.

### Add A Separate Bootstrap Adapter Per Functional Family

Implementation shape:

- allocate a second runtime adapter per functional family,
- initialize it with tiny random weights,
- never update it with Adam,
- attach it only when the family is activated,
- scale it by a sampled signed perturbation value.

Why:

- It gives the family a real accidental effect even when the learned adapter is still zero.
- It keeps the learned adapter semantically clean.
- It makes observability straightforward.

### Decay Bootstrap Magnitude From Usage, Not Wall Clock

Recommended policy:

- maintain a per-family usage or activation count,
- compute `bootstrap_std = floor + (initial - floor) / sqrt(1 + usage / decay)`.

Why:

- This matches the existing gating-noise decay style.
- It ties exploration to actual family experience.
- It naturally approaches a nonzero floor.

### Expose Bootstrap Perturbation Separately From Gate Exploration

Trace surfaces should include:

- gate exploration std and sampled gate noise,
- bootstrap perturbation std and sampled bootstrap perturbation,
- family usage count.

Why:

- Operators need to distinguish "the gate selected this family more strongly" from "the family’s early bootstrap substrate perturbed the output."

## Rejected Alternatives

### Make The Learned Functional Adapter Nonzero At Initialization

Rejected because:

- it breaks the clean "learned adapter starts as no-op" contract,
- it muddies the meaning of runtime updates,
- and it makes accidental exploration indistinguishable from learned state.

### Rely Only On Gating Noise

Rejected because:

- gain noise on a zeroed adapter still yields zero functional bias from the adapter itself.
- it does not satisfy the requested behavior that the functional LoRA can accidentally discover useful biases early.

### Resample The Learned Adapter Weights Directly On Every Activation

Rejected because:

- it entangles exploration with learned parameters,
- complicates optimizer state and observability,
- and makes family state harder to reason about than a fixed bootstrap substrate plus sampled scale.

## Implementation Consequences

- Functional families will still be "effectively no-op learned adapters" at birth.
- Early activations will now have a tiny signed stochastic effect from a separate bootstrap substrate.
- Exploration pressure will be strongest early and decay toward a nonzero floor with actual family usage.
- Counterfactual, ablation, and user-simulation flows remain coherent because the bootstrap substrate is just another explicit runtime adapter controlled by the same family activation path.
