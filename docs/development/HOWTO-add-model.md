# Add a new base model architecture to Vicuña

Vicuña still follows essentially the same high-level process as upstream
`llama.cpp` when adding support for a new base model:

1. Convert the model to GGUF
2. Define the architecture and runtime metadata
3. Build and register the GGML graph implementation

The main difference is file layout. The upstream workflow is still the right
mental model, but Vicuña's current source tree spreads the work across
`src/llama-arch.*`, `src/llama-model-loader.cpp`, `src/llama-model.cpp`, and
`src/models/*.cpp` instead of putting all graph code in one file.

After following these steps and validating the affected tools/backends, you can
open a PR.

Also, it is important to check that the examples and main ggml backends (CUDA, METAL, CPU) are working with the new architecture, especially:
- [cli](/tools/cli/)
- [completion](/tools/completion/)
- [imatrix](/tools/imatrix/)
- [quantize](/tools/quantize/)
- [server](/tools/server/)

### 1. Convert the model to GGUF

This step is done in python with a `convert` script using the [gguf](https://pypi.org/project/gguf/) library.
Depending on the model architecture, you can use either [convert_hf_to_gguf.py](/convert_hf_to_gguf.py) or [examples/convert_legacy_llama.py](/examples/convert_legacy_llama.py) (for `llama/llama2` models in `.pth` format).

The convert script reads the model configuration, tokenizer, tensor names+data and converts them to GGUF metadata and tensors.

The required steps to implement for an HF model are:

1. Define the model `ModelBase.register` annotation in a new `TextModel` or `MmprojModel` subclass, example:

```python
@ModelBase.register("MyModelForCausalLM")
class MyModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MYMODEL
```

or

```python
@ModelBase.register("MyModelForConditionalGeneration")
class MyModel(MmprojModel):
    model_arch = gguf.MODEL_ARCH.MYMODEL
```

2. Define the layout of the GGUF tensors in [constants.py](/gguf-py/gguf/constants.py)

Add an enum entry in `MODEL_ARCH`, the model human friendly name in `MODEL_ARCH_NAMES` and the GGUF tensor names in `MODEL_TENSORS`.

Example for `falcon` model:
```python
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ]
```

3. Map the original tensor names to the standardize equivalent in GGUF

As a general rule, before adding a new tensor name to GGUF, be sure the equivalent naming does not already exist.

Once you have found the GGUF tensor name equivalent, add it to the [tensor_mapping.py](/gguf-py/gguf/tensor_mapping.py) file.

If the tensor name is part of a repetitive layer/block, the key word `bid` substitutes it.

Example for the normalization tensor in attention layers:

```python
block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: (
            "gpt_neox.layers.{bid}.input_layernorm",                # gptneox
            "transformer.h.{bid}.ln_1",                             # gpt2 gpt-j refact qwen
            "transformer.blocks.{bid}.norm_1",                      # mpt
            ...
        )
}
```

`transformer.blocks.{bid}.norm_1` will be mapped to `blk.{bid}.attn_norm` in GGUF.

Depending on the model configuration, tokenizer, code and tensors layout, you will have to override:
- `TextModel#set_gguf_parameters`
- `MmprojModel#set_gguf_parameters`
- `ModelBase#set_vocab`
- `ModelBase#modify_tensors`

NOTE: Tensor names must end with `.weight` or `.bias` suffixes, that is the convention and several tools like `quantize` expect this to proceed the weights.

### 2. Define the model architecture and runtime metadata

The model params and tensor layout are still defined in the same core runtime
areas used by `llama.cpp`, but in Vicuña the important integration points are:

1. Define a new `llm_arch` enum value in `src/llama-arch.h`.
2. In `src/llama-arch.cpp`:
    - Add the architecture name to the `LLM_ARCH_NAMES` map.
    - Add the list of model tensors to `llm_get_tensor_names` (you may also need to update `LLM_TENSOR_NAMES`)
3. Add any non-standard metadata loading in the `llama_model_loader` constructor in `src/llama-model-loader.cpp`.
4. Wire any architecture-specific tensor creation or runtime switches in `src/llama-model.cpp`.
5. If the model has a RoPE operation, add a case for the architecture in `llama_model_rope_type` in `src/llama-model.cpp`.

NOTE: The dimensions in `ggml` are typically in the reverse order of the `pytorch` dimensions.

### 3. Build the GGML graph implementation

This is still the core implementation step, but in Vicuña the graph builders
live under `src/models/`.

1. Add a new model-specific graph builder declaration to `src/models/models.h`.
2. Implement the graph builder in a new `src/models/<model>.cpp` file.
3. Add the new source file to `src/CMakeLists.txt` so it is compiled.
4. In `llama_model::build_graph` in `src/llama-model.cpp`, add a case for your architecture to instantiate the new graph-building struct.

Have a look at existing implementations like `src/models/llama.cpp`,
`src/models/dbrx.cpp`, or `src/models/bert.cpp`.

Some `ggml` backends do not support all operations. Backend implementations can be added in a separate PR.

Note: to debug the inference graph: you can use [llama-eval-callback](/examples/eval-callback/).

## Validation before opening a PR

At minimum, confirm the new architecture works in the tools and backends that
exercise the path you changed. The usual checklist is:

- `tools/cli`
- `tools/completion`
- `tools/imatrix`
- `tools/quantize`
- `tools/server`
- CPU plus any backend you expect to support immediately, especially CUDA and
  METAL when relevant

## GGUF specification

https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## Resources

- YaRN RoPE scaling https://github.com/ggml-org/llama.cpp/pull/2268
- support Baichuan serial models https://github.com/ggml-org/llama.cpp/pull/3009
- support attention bias https://github.com/ggml-org/llama.cpp/pull/4283
- Mixtral support https://github.com/ggml-org/llama.cpp/pull/4406
- BERT embeddings https://github.com/ggml-org/llama.cpp/pull/5423
- Grok-1 support https://github.com/ggml-org/llama.cpp/pull/6204
- Command R Plus support https://github.com/ggml-org/llama.cpp/pull/6491
- support arch DBRX https://github.com/ggml-org/llama.cpp/pull/6515
- How to convert HuggingFace model to GGUF format https://github.com/ggml-org/llama.cpp/discussions/2948
