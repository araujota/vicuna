#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

enum llama_adapter_lora_layer_role : int32_t {
    LLAMA_ADAPTER_LORA_LAYER_REQUEST = 0,
    LLAMA_ADAPTER_LORA_LAYER_ALL_TIME,
    LLAMA_ADAPTER_LORA_LAYER_PAST_YEAR,
    LLAMA_ADAPTER_LORA_LAYER_PAST_QUARTER,
    LLAMA_ADAPTER_LORA_LAYER_PAST_MONTH,
    LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK,
    LLAMA_ADAPTER_LORA_LAYER_ACTIVE,
    LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION,
    LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_PLANNING_COMPOSITION,
    LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_COUNTERFACTUAL,
    LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_MEMORY_COMPRESSION,
    LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_SELF_OBSERVATION,
    LLAMA_ADAPTER_LORA_LAYER_USER_PERSONALITY,
    LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_PROCESS_LEARNED,
    LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_PROCESS_BOOTSTRAP,
};

struct llama_adapter_lora_entry {
    llama_adapter_lora * adapter = nullptr;
    float scale = 0.0f;
    int32_t precedence = 0;
    llama_adapter_lora_layer_role role = LLAMA_ADAPTER_LORA_LAYER_REQUEST;
};

using llama_adapter_lora_stack = std::vector<llama_adapter_lora_entry>;
using llama_adapter_lora_stack_ptr = std::unique_ptr<llama_adapter_lora_stack>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    float gain = 1.0f;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = (alpha ? adapter_scale * alpha / rank : adapter_scale) * gain;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b, float gain = 1.0f) : a(a), b(b), gain(gain) {}
};

struct llama_adapter_lora {
    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    bool is_runtime_mutable = false;

    llama_adapter_lora() = default;
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

bool llama_adapter_lora_init_runtime(
        llama_model & model,
        llama_adapter_lora & adapter,
        const std::vector<std::pair<ggml_tensor *, uint32_t>> & targets,
        const std::string & name_prefix);

int32_t llama_adapter_lora_layer_precedence(llama_adapter_lora_layer_role role);
const char * llama_adapter_lora_layer_role_name(llama_adapter_lora_layer_role role);
