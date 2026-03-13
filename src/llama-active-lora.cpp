#include "llama-active-lora.h"

#include "llama.h"
#include "llama-adapter.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t ACTIVE_TO_WEEK_JOB = 0;
constexpr uint32_t WEEK_TO_MONTH_JOB = 1;
constexpr uint32_t MONTH_TO_QUARTER_JOB = 2;
constexpr uint32_t QUARTER_TO_YEAR_JOB = 3;
constexpr uint32_t YEAR_TO_ALL_TIME_JOB = 4;
constexpr size_t PAST_BUCKET_COUNT = LLAMA_MEMORY_LORA_BUCKET_COUNT;
constexpr float DIRECTION_EPS = 1.0e-8f;
constexpr float FUNCTIONAL_GAIN_CLIP_MIN = 0.0f;
constexpr float FUNCTIONAL_GAIN_CLIP_MAX = 2.0f;
constexpr size_t FUNCTIONAL_GATING_INPUT_DIM = 21;
constexpr size_t FUNCTIONAL_GATING_HIDDEN_DIM = 16;
constexpr uint64_t FUNCTIONAL_GATING_INIT_SEED = 0x6d6574616c6f7373ULL;

float clamp_unit(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

float clamp_range(float value, float lo, float hi) {
    return std::min(hi, std::max(lo, value));
}

const char * past_bucket_name(size_t bucket) {
    switch (bucket) {
        case LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK:    return "past_week";
        case LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH:   return "past_month";
        case LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER: return "past_quarter";
        case LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR:    return "past_year";
        case LLAMA_MEMORY_LORA_BUCKET_ALL_TIME:     return "all_time";
        default:                                    return "unknown";
    }
}

uint64_t get_host_free_bytes() {
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return 0;
    }

    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(cpu_dev, &free, &total);
    return free ? free : total;
}

uint64_t get_device_free_bytes(const llama_model & model) {
    uint64_t total_free = 0;
    for (auto * dev : model.devices) {
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        total_free += free ? free : total;
    }
    return total_free;
}

uint64_t scale_memory_budget(uint64_t free_bytes, float ratio) {
    if (ratio <= 0.0f) {
        return 0;
    }

    // Some backends do not report host/device memory. Treat that as "unknown"
    // and avoid collapsing the planned rank to zero.
    if (free_bytes == 0) {
        return std::numeric_limits<uint64_t>::max();
    }

    if (ratio >= 1.0f) {
        return free_bytes;
    }

    const long double scaled = static_cast<long double>(free_bytes) * static_cast<long double>(ratio);
    return std::max<uint64_t>(1, static_cast<uint64_t>(scaled));
}

void maybe_add_target(std::vector<ggml_tensor *> & targets, ggml_tensor * tensor) {
    if (!tensor || tensor->ne[1] <= 1) {
        return;
    }

    if (std::find(targets.begin(), targets.end(), tensor) == targets.end()) {
        targets.push_back(tensor);
    }
}

std::vector<ggml_tensor *> collect_memory_targets(const llama_model & model) {
    std::vector<ggml_tensor *> targets;
    const int32_t n_layer = model.hparams.n_layer;
    const int32_t first_layer = std::max(0, n_layer - 4);

    for (int32_t il = first_layer; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        maybe_add_target(targets, layer.wq);
        maybe_add_target(targets, layer.wv);
        maybe_add_target(targets, layer.wo);
        maybe_add_target(targets, layer.wqkv);
        maybe_add_target(targets, layer.ffn_gate);
        maybe_add_target(targets, layer.ffn_up);
        maybe_add_target(targets, layer.ffn_down);
    }

    if (targets.empty()) {
        maybe_add_target(targets, model.output);
    }

    return targets;
}

struct active_lora_embedding {
    std::vector<float> values;
    float norm = 0.0f;
};

struct llama_context_deleter {
    void operator()(llama_context * ctx) const {
        if (ctx) {
            llama_free(ctx);
        }
    }
};

using llama_context_ptr = std::unique_ptr<llama_context, llama_context_deleter>;

struct active_lora_write_features {
    std::vector<float> content_signal;
    std::vector<float> state_signal;
    float update_emphasis = 0.0f;
    float goal_alignment = 0.0f;
    float repair_pressure = 0.0f;
    float social_alignment = 0.0f;
};

void normalize_vector(std::vector<float> & vec);

uint64_t mix_seed(uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

float decoder_entropy_feature(float entropy) {
    if (entropy <= 0.0f) {
        return 0.0f;
    }
    return entropy / (entropy + 1.0f);
}

float sample_signal(const std::vector<float> & signal, size_t index, uint64_t seed) {
    if (signal.empty()) {
        return 0.0f;
    }

    const size_t n = signal.size();
    const size_t base = seed % n;
    const size_t stride = ((seed >> 17) % n) | size_t(1);
    const float first = signal[(base + index * stride) % n];
    if (n == 1) {
        return first;
    }

    const float second = signal[(base + index * (stride + 2) + 1) % n];
    const float third = signal[(base + index * (stride + 4) + 3) % n];
    const float sign = ((seed >> 41) & 1ULL) ? -1.0f : 1.0f;
    return 0.60f * first + 0.30f * second + 0.10f * third * sign;
}

std::vector<float> project_signal(const std::vector<float> & src, size_t out_dim, uint64_t seed) {
    std::vector<float> out(out_dim, 0.0f);
    if (src.empty() || out_dim == 0) {
        return out;
    }

    if (src.size() == out_dim) {
        out = src;
    } else {
        for (size_t i = 0; i < out_dim; ++i) {
            out[i] = sample_signal(src, i, seed + i * 0x9e3779b97f4a7c15ULL);
        }
    }

    normalize_vector(out);
    return out;
}

uint64_t hash_signal_prefix(const std::vector<float> & signal, size_t limit, uint64_t seed) {
    uint64_t hash = mix_seed(seed ^ 1469598103934665603ULL);
    for (size_t i = 0; i < std::min(limit, signal.size()); ++i) {
        const int32_t quantized = static_cast<int32_t>(std::lrint(signal[i] * 4096.0f));
        hash ^= static_cast<uint32_t>(quantized);
        hash *= 1099511628211ULL;
        hash = mix_seed(hash + i);
    }
    return hash;
}

size_t parse_layer_index(const std::string & tensor_name) {
    const std::string marker = "blk.";
    const size_t pos = tensor_name.find(marker);
    if (pos == std::string::npos) {
        return std::numeric_limits<size_t>::max();
    }

    size_t value = 0;
    bool saw_digit = false;
    for (size_t i = pos + marker.size(); i < tensor_name.size(); ++i) {
        const char ch = tensor_name[i];
        if (ch < '0' || ch > '9') {
            break;
        }
        saw_digit = true;
        value = value * 10 + static_cast<size_t>(ch - '0');
    }

    return saw_digit ? value : std::numeric_limits<size_t>::max();
}

float functional_default_gain(int32_t family) {
    (void) family;
    return 1.0f;
}

uint32_t functional_top_k_priority(int32_t family) {
    switch (family) {
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL:
            return 4;
        case LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION:
            return 3;
        case LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION:
            return 2;
        case LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION:
        default:
            return 1;
    }
}

uint32_t functional_update_horizon_steps(int32_t family) {
    switch (family) {
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL:
            return 3;
        case LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION:
            return 2;
        case LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION:
        case LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION:
        default:
            return 1;
    }
}

float functional_target_value(size_t index) {
    switch (index) {
        case 5: return 1.0f; // answerability
        default: return 0.0f;
    }
}

float functional_snapshot_component(const llama_functional_outcome_snapshot & snapshot, size_t index) {
    switch (index) {
        case 0: return clamp_unit(snapshot.favorable_divergence);
        case 1: return clamp_unit(snapshot.user_satisfaction_risk);
        case 2: return clamp_unit(snapshot.goal_progress_pressure);
        case 3: return clamp_unit(snapshot.loop_inefficiency);
        case 4: return clamp_unit(snapshot.recovery_urgency);
        case 5: return clamp_unit(snapshot.answerability);
        case 6: return clamp_unit(snapshot.preference_uncertainty);
        case 7: return clamp_unit(snapshot.expected_steps_remaining);
        case 8: return clamp_unit(snapshot.expected_inference_cost_remaining);
        default: return 0.0f;
    }
}

float functional_allostatic_distance(const llama_functional_outcome_snapshot & snapshot) {
    static const float weights[] = { 0.25f, 0.12f, 0.12f, 0.10f, 0.10f, 0.10f, 0.08f, 0.08f, 0.05f };
    float accum = 0.0f;
    float weight_sum = 0.0f;
    for (size_t i = 0; i < sizeof(weights)/sizeof(weights[0]); ++i) {
        const float error = functional_target_value(i) - functional_snapshot_component(snapshot, i);
        accum += weights[i] * error * error;
        weight_sum += weights[i];
    }
    return weight_sum > 0.0f ? std::sqrt(accum / weight_sum) : 0.0f;
}

struct functional_gating_training_tuple {
    bool valid = false;
    int32_t family = -1;
    int32_t loop_origin = 0;
    int32_t microphase = 0;
    uint64_t eligible_mask = 0;
    uint64_t activated_mask = 0;
    uint64_t invocation_count = 0;
    float allostatic_distance_before = 0.0f;
    float exploration_std = 0.0f;
    std::array<float, FUNCTIONAL_GATING_INPUT_DIM> input = {};
    std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> hidden = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> predicted_gains = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> sampled_noise = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> applied_gains = {};
};

struct functional_gating_network {
    std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM * FUNCTIONAL_GATING_INPUT_DIM> w1 = {};
    std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> b1 = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT * FUNCTIONAL_GATING_HIDDEN_DIM> w2 = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> b2 = {};
};

struct functional_gating_adam_state {
    std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM * FUNCTIONAL_GATING_INPUT_DIM> m_w1 = {};
    std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM * FUNCTIONAL_GATING_INPUT_DIM> v_w1 = {};
    std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> m_b1 = {};
    std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> v_b1 = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT * FUNCTIONAL_GATING_HIDDEN_DIM> m_w2 = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT * FUNCTIONAL_GATING_HIDDEN_DIM> v_w2 = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> m_b2 = {};
    std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> v_b2 = {};
    uint64_t step = 0;
    float learning_rate = 3.0e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1.0e-8f;
};

struct runtime_lora_adam_state {
    std::vector<float> m_a;
    std::vector<float> v_a;
    std::vector<float> m_b;
    std::vector<float> v_b;
    uint64_t step = 0;
    float learning_rate = 1.0e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1.0e-8f;
    float last_update_norm = 0.0f;
};

struct temporal_bias_adam_state {
    uint64_t step = 0;
    float learning_rate = 1.5e-2f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1.0e-8f;
    float m_reward = 0.0f;
    float v_reward = 0.0f;
    float m_dampening = 0.0f;
    float v_dampening = 0.0f;
};

struct runtime_lora_write_result {
    bool updated = false;
    uint64_t transaction_step = 0;
    float update_norm = 0.0f;
};

float gating_noise_std(const llama_functional_lora_family_config & config, uint64_t invocation_count) {
    const float init = std::max(0.0f, config.exploration_noise_initial_std);
    const float min_std = std::max(0.0f, std::min(init, config.exploration_noise_min_std));
    const float decay = std::max<uint32_t>(1u, config.exploration_noise_decay_invocations);
    const float scaled = std::sqrt(1.0f + static_cast<float>(invocation_count) / static_cast<float>(decay));
    return min_std + (init - min_std) / scaled;
}

class active_lora_embedder {
public:
    virtual ~active_lora_embedder() = default;
    virtual int32_t type() const = 0;
    virtual bool is_custom() const = 0;
    virtual uint32_t dim() const = 0;
    virtual bool ready() const {
        return dim() > 0;
    }
    virtual active_lora_embedding embed(const llama_token * tokens, size_t n_tokens) const = 0;
};

class active_lora_hash_embedder final : public active_lora_embedder {
public:
    int32_t type() const override {
        return LLAMA_ACTIVE_LORA_EMBEDDING_HASH;
    }

    bool is_custom() const override {
        return false;
    }

    uint32_t dim() const override {
        return 64;
    }

    active_lora_embedding embed(const llama_token * tokens, size_t n_tokens) const override {
        active_lora_embedding result;
        result.values.assign(dim(), 0.0f);

        for (size_t i = 0; i < n_tokens; ++i) {
            const uint32_t value = static_cast<uint32_t>(tokens[i]);
            const size_t bucket = value % result.values.size();
            const float sign = (value & 1u) ? -1.0f : 1.0f;
            result.values[bucket] += sign;
        }

        float sum_sq = 0.0f;
        for (float & value : result.values) {
            value /= std::max<size_t>(1, n_tokens);
            sum_sq += value * value;
        }
        result.norm = std::sqrt(sum_sq);
        return result;
    }
};

class active_lora_token_pool_embedder final : public active_lora_embedder {
public:
    int32_t type() const override {
        return LLAMA_ACTIVE_LORA_EMBEDDING_TOKEN_POOL;
    }

    bool is_custom() const override {
        return false;
    }

    uint32_t dim() const override {
        return 64;
    }

    active_lora_embedding embed(const llama_token * tokens, size_t n_tokens) const override {
        active_lora_embedding result;
        result.values.assign(dim(), 0.0f);

        for (size_t i = 0; i < n_tokens; ++i) {
            const size_t bucket = static_cast<uint32_t>(tokens[i]) % result.values.size();
            result.values[bucket] += 1.0f;
        }

        float sum_sq = 0.0f;
        for (float & value : result.values) {
            value /= std::max<size_t>(1, n_tokens);
            sum_sq += value * value;
        }
        result.norm = std::sqrt(sum_sq);
        return result;
    }
};

class active_lora_callback_embedder final : public active_lora_embedder {
public:
    active_lora_callback_embedder(
            const llama_context & owner,
            llama_active_lora_embedding_callback callback,
            void * user_data,
            uint32_t dim,
            int32_t declared_type) :
        owner(owner),
        callback(callback),
        user_data(user_data),
        embedding_dim(dim),
        declared_type(declared_type) {
    }

    int32_t type() const override {
        return declared_type;
    }

    bool is_custom() const override {
        return true;
    }

    uint32_t dim() const override {
        return embedding_dim;
    }

    active_lora_embedding embed(const llama_token * tokens, size_t n_tokens) const override {
        active_lora_embedding result;
        result.values.assign(embedding_dim, 0.0f);

        if (!callback || embedding_dim == 0) {
            return result;
        }

        if (!callback(&owner, tokens, n_tokens, result.values.data(), result.values.size(), user_data)) {
            result.values.clear();
            return result;
        }

        float sum_sq = 0.0f;
        for (float value : result.values) {
            sum_sq += value * value;
        }
        result.norm = std::sqrt(sum_sq);
        return result;
    }

private:
    const llama_context & owner;
    llama_active_lora_embedding_callback callback = nullptr;
    void * user_data = nullptr;
    uint32_t embedding_dim = 0;
    int32_t declared_type = LLAMA_ACTIVE_LORA_EMBEDDING_HASH;
};

class active_lora_hidden_state_embedder final : public active_lora_embedder {
public:
    active_lora_hidden_state_embedder(const llama_context & owner, const llama_active_lora_params & params) {
        const auto & cparams = owner.get_cparams();
        llama_context_params aux_params = llama_context_default_params();
        const uint32_t planned_tokens = std::max<uint32_t>(8, params.train_context_tokens);
        const uint32_t aux_ctx_tokens = std::max<uint32_t>(
                16,
                std::min<uint32_t>(std::max<uint32_t>(planned_tokens, params.train_stride_tokens + 1), cparams.n_ctx));

        aux_params.n_ctx = aux_ctx_tokens;
        aux_params.n_batch = aux_ctx_tokens;
        aux_params.n_ubatch = aux_ctx_tokens;
        aux_params.n_seq_max = 1;
        aux_params.n_threads = cparams.n_threads;
        aux_params.n_threads_batch = cparams.n_threads_batch;
        aux_params.rope_freq_base = cparams.rope_freq_base;
        aux_params.rope_freq_scale = cparams.rope_freq_scale;
        aux_params.yarn_orig_ctx = cparams.n_ctx_orig_yarn;
        aux_params.yarn_ext_factor = cparams.yarn_ext_factor;
        aux_params.yarn_attn_factor = cparams.yarn_attn_factor;
        aux_params.yarn_beta_fast = cparams.yarn_beta_fast;
        aux_params.yarn_beta_slow = cparams.yarn_beta_slow;
        aux_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        aux_params.embeddings = true;
        aux_params.offload_kqv = cparams.offload_kqv;
        aux_params.no_perf = true;
        aux_params.op_offload = cparams.op_offload;
        aux_params.kv_unified = cparams.kv_unified;

        aux_ctx.reset(llama_init_from_model(const_cast<llama_model *>(&owner.get_model()), aux_params));
        if (!aux_ctx) {
            return;
        }

        llama_set_embeddings(aux_ctx.get(), true);
        source_dim = static_cast<uint32_t>(std::max(0, llama_model_n_embd_out(&owner.get_model())));
        embedding_dim = params.embedding_dim ? params.embedding_dim : source_dim;
        chunk_tokens = std::max<uint32_t>(1, std::min<uint32_t>(aux_ctx_tokens, planned_tokens));
        stride_tokens = params.train_stride_tokens == 0 ? chunk_tokens : std::min<uint32_t>(chunk_tokens, params.train_stride_tokens);
    }

    int32_t type() const override {
        return LLAMA_ACTIVE_LORA_EMBEDDING_HIDDEN_STATE;
    }

    bool is_custom() const override {
        return false;
    }

    uint32_t dim() const override {
        return embedding_dim;
    }

    bool ready() const override {
        return aux_ctx && source_dim > 0 && embedding_dim > 0;
    }

    active_lora_embedding embed(const llama_token * tokens, size_t n_tokens) const override {
        active_lora_embedding result;
        if (!ready() || !tokens || n_tokens == 0) {
            return result;
        }

        result.values.assign(embedding_dim, 0.0f);
        uint32_t chunks = 0;
        size_t offset = 0;

        while (offset < n_tokens) {
            const size_t count = std::min<size_t>(chunk_tokens, n_tokens - offset);
            llama_memory_clear(llama_get_memory(aux_ctx.get()), true);

            llama_batch batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);
            batch.n_tokens = static_cast<int32_t>(count);
            for (size_t i = 0; i < count; ++i) {
                batch.token[i] = tokens[offset + i];
                batch.pos[i] = static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            const int32_t decode_rc = llama_decode(aux_ctx.get(), batch);
            llama_batch_free(batch);
            if (decode_rc != 0) {
                result.values.clear();
                result.norm = 0.0f;
                return result;
            }

            std::vector<float> chunk_values(source_dim, 0.0f);
            uint32_t seen = 0;
            for (size_t i = 0; i < count; ++i) {
                const float * token_embedding = llama_get_embeddings_ith(aux_ctx.get(), static_cast<int32_t>(i));
                if (!token_embedding) {
                    continue;
                }

                float token_norm = 0.0f;
                for (uint32_t d = 0; d < source_dim; ++d) {
                    token_norm += token_embedding[d] * token_embedding[d];
                }
                token_norm = std::sqrt(token_norm);
                const float scale = token_norm > DIRECTION_EPS ? 1.0f / token_norm : 1.0f;
                for (uint32_t d = 0; d < source_dim; ++d) {
                    chunk_values[d] += token_embedding[d] * scale;
                }
                ++seen;
            }

            if (seen > 0) {
                for (float & value : chunk_values) {
                    value /= static_cast<float>(seen);
                }
                std::vector<float> projected = project_signal(chunk_values, embedding_dim, 0x6a09e667f3bcc909ULL + chunks);
                for (uint32_t d = 0; d < embedding_dim; ++d) {
                    result.values[d] += projected[d];
                }
                ++chunks;
            }

            if (offset + count >= n_tokens) {
                break;
            }
            const size_t step = stride_tokens == 0 ? count : std::max<size_t>(1, stride_tokens);
            offset += std::min(step, count);
        }

        if (chunks == 0) {
            result.values.clear();
            return result;
        }

        float sum_sq = 0.0f;
        for (float & value : result.values) {
            value /= static_cast<float>(chunks);
            sum_sq += value * value;
        }
        result.norm = std::sqrt(sum_sq);
        return result;
    }

private:
    llama_context_ptr aux_ctx;
    uint32_t source_dim = 0;
    uint32_t embedding_dim = 0;
    uint32_t chunk_tokens = 0;
    uint32_t stride_tokens = 0;
};

std::unique_ptr<active_lora_embedder> make_embedder(const llama_context & owner, const llama_active_lora_params & params) {
    if (params.embedding_callback != nullptr) {
        return std::make_unique<active_lora_callback_embedder>(
                owner,
                params.embedding_callback,
                params.embedding_callback_user_data,
                params.embedding_dim,
                params.embedding_type);
    }

    switch (params.embedding_type) {
        case LLAMA_ACTIVE_LORA_EMBEDDING_HIDDEN_STATE:
            {
                auto embedder = std::make_unique<active_lora_hidden_state_embedder>(owner, params);
                if (embedder->ready()) {
                    return embedder;
                }

                LLAMA_LOG_WARN("%s: hidden-state embedder unavailable, falling back to hash embedder\n", __func__);
                return std::make_unique<active_lora_hash_embedder>();
            }
        case LLAMA_ACTIVE_LORA_EMBEDDING_TOKEN_POOL:
            return std::make_unique<active_lora_token_pool_embedder>();
        case LLAMA_ACTIVE_LORA_EMBEDDING_HASH:
        default:
            return std::make_unique<active_lora_hash_embedder>();
    }
}

float cosine_similarity(const active_lora_embedding & a, const active_lora_embedding & b) {
    if (a.values.empty() || b.values.empty() || a.values.size() != b.values.size() || a.norm == 0.0f || b.norm == 0.0f) {
        return -1.0f;
    }

    float dot = 0.0f;
    for (size_t i = 0; i < a.values.size(); ++i) {
        dot += a.values[i] * b.values[i];
    }

    return dot / (a.norm * b.norm);
}

float fro_norm(const std::vector<float> & data) {
    double sum_sq = 0.0;
    for (float value : data) {
        sum_sq += (double) value * (double) value;
    }
    return std::sqrt((float) sum_sq);
}

float compute_effective_scale(float base_scale, uint64_t created_at_us, uint64_t now_us, uint64_t half_life_us) {
    if (base_scale <= 0.0f) {
        return 0.0f;
    }
    if (half_life_us == 0 || now_us <= created_at_us) {
        return base_scale;
    }

    const double age = (double) (now_us - created_at_us);
    const double half_lives = age / (double) half_life_us;
    return (float) (base_scale * std::pow(0.5, half_lives));
}

std::pair<float, float> adapter_gain_stats(const llama_adapter_lora & adapter) {
    float gain_sum = 0.0f;
    float gain_max = 0.0f;
    size_t count = 0;

    for (const auto & it : adapter.ab_map) {
        gain_sum += it.second.gain;
        gain_max = std::max(gain_max, it.second.gain);
        ++count;
    }

    return {
        count == 0 ? 0.0f : gain_sum / count,
        gain_max,
    };
}

void zero_weight(llama_adapter_lora_weight & weight) {
    std::vector<float> data_a(weight.a->ne[0] * weight.a->ne[1], 0.0f);
    std::vector<float> data_b(weight.b->ne[0] * weight.b->ne[1], 0.0f);
    ggml_backend_tensor_set(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_set(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
    weight.gain = 0.0f;
}

void zero_adapter(llama_adapter_lora & adapter) {
    for (auto & it : adapter.ab_map) {
        zero_weight(it.second);
    }
}

struct low_rank_direction {
    size_t out_dim = 0;
    size_t in_dim = 0;
    size_t rank = 0;
    float gain = 0.0f;
    std::vector<float> left;  // [out_dim, rank], row-major
    std::vector<float> right; // [in_dim, rank], row-major
};

low_rank_direction read_direction(const llama_adapter_lora_weight & weight) {
    low_rank_direction result;
    result.in_dim = weight.a->ne[0];
    result.rank = weight.a->ne[1];
    result.out_dim = weight.b->ne[1];
    result.gain = weight.gain;
    result.left.assign(result.out_dim * result.rank, 0.0f);
    result.right.assign(result.in_dim * result.rank, 0.0f);

    std::vector<float> data_a(result.in_dim * result.rank);
    std::vector<float> data_b(result.rank * result.out_dim);
    ggml_backend_tensor_get(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_get(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));

    for (size_t k = 0; k < result.rank; ++k) {
        for (size_t i = 0; i < result.in_dim; ++i) {
            result.right[i*result.rank + k] = data_a[k*result.in_dim + i];
        }
        for (size_t o = 0; o < result.out_dim; ++o) {
            result.left[o*result.rank + k] = data_b[o*result.rank + k];
        }
    }

    return result;
}

void write_direction(llama_adapter_lora_weight & weight, const low_rank_direction & direction) {
    const size_t alloc_rank = weight.a->ne[1];
    const size_t in_dim = weight.a->ne[0];
    const size_t out_dim = weight.b->ne[1];
    std::vector<float> data_a(in_dim * alloc_rank, 0.0f);
    std::vector<float> data_b(alloc_rank * out_dim, 0.0f);

    const size_t used_rank = std::min(direction.rank, alloc_rank);
    for (size_t k = 0; k < used_rank; ++k) {
        for (size_t i = 0; i < in_dim; ++i) {
            data_a[k*in_dim + i] = direction.right[i*direction.rank + k];
        }
        for (size_t o = 0; o < out_dim; ++o) {
            data_b[o*alloc_rank + k] = direction.left[o*direction.rank + k];
        }
    }

    ggml_backend_tensor_set(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_set(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
    weight.gain = direction.gain;
}

float dot_columns(const std::vector<float> & mat, size_t rows, size_t cols, size_t col_a, size_t col_b) {
    float result = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        result += mat[r*cols + col_a] * mat[r*cols + col_b];
    }
    return result;
}

float column_norm(const std::vector<float> & mat, size_t rows, size_t cols, size_t col) {
    return std::sqrt(std::max(0.0f, dot_columns(mat, rows, cols, col, col)));
}

void scale_column(std::vector<float> & mat, size_t rows, size_t cols, size_t col, float scale) {
    for (size_t r = 0; r < rows; ++r) {
        mat[r*cols + col] *= scale;
    }
}

void subtract_column(std::vector<float> & mat, size_t rows, size_t cols, size_t dst_col, size_t src_col, float scale) {
    for (size_t r = 0; r < rows; ++r) {
        mat[r*cols + dst_col] -= scale * mat[r*cols + src_col];
    }
}

struct thin_qr_result {
    std::vector<float> q;
    std::vector<float> r;
    size_t rows = 0;
    size_t cols = 0;
};

thin_qr_result thin_qr(const std::vector<float> & mat, size_t rows, size_t cols) {
    thin_qr_result result;
    result.q = mat;
    result.r.assign(cols * cols, 0.0f);
    result.rows = rows;
    result.cols = cols;

    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < j; ++i) {
            const float proj = dot_columns(result.q, rows, cols, i, j);
            result.r[i*cols + j] = proj;
            subtract_column(result.q, rows, cols, j, i, proj);
        }

        const float norm = column_norm(result.q, rows, cols, j);
        result.r[j*cols + j] = norm;
        if (norm > DIRECTION_EPS) {
            scale_column(result.q, rows, cols, j, 1.0f / norm);
        } else {
            for (size_t r = 0; r < rows; ++r) {
                result.q[r*cols + j] = 0.0f;
            }
        }
    }

    return result;
}

std::vector<float> small_mat_mul_r_rt(const std::vector<float> & lhs, const std::vector<float> & rhs, size_t n) {
    std::vector<float> out(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += lhs[i*n + k] * rhs[j*n + k];
            }
            out[i*n + j] = sum;
        }
    }
    return out;
}

std::vector<float> small_mat_transpose_mul(const std::vector<float> & mat, size_t n) {
    std::vector<float> out(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += mat[k*n + i] * mat[k*n + j];
            }
            out[i*n + j] = sum;
        }
    }
    return out;
}

std::pair<std::vector<float>, std::vector<float>> jacobi_eigen(std::vector<float> mat, size_t n) {
    std::vector<float> eigenvectors(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        eigenvectors[i*n + i] = 1.0f;
    }

    const size_t max_iter = 64 * n * n;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        size_t p = 0;
        size_t q = 0;
        float max_val = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const float value = std::fabs(mat[i*n + j]);
                if (value > max_val) {
                    max_val = value;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_val < 1.0e-6f) {
            break;
        }

        const float app = mat[p*n + p];
        const float aqq = mat[q*n + q];
        const float apq = mat[p*n + q];
        const float phi = 0.5f * std::atan2(2.0f * apq, aqq - app);
        const float c = std::cos(phi);
        const float s = std::sin(phi);

        for (size_t k = 0; k < n; ++k) {
            const float mkp = mat[k*n + p];
            const float mkq = mat[k*n + q];
            mat[k*n + p] = c * mkp - s * mkq;
            mat[k*n + q] = s * mkp + c * mkq;
        }

        for (size_t k = 0; k < n; ++k) {
            const float mpk = mat[p*n + k];
            const float mqk = mat[q*n + k];
            mat[p*n + k] = c * mpk - s * mqk;
            mat[q*n + k] = s * mpk + c * mqk;
        }

        for (size_t k = 0; k < n; ++k) {
            const float vkp = eigenvectors[k*n + p];
            const float vkq = eigenvectors[k*n + q];
            eigenvectors[k*n + p] = c * vkp - s * vkq;
            eigenvectors[k*n + q] = s * vkp + c * vkq;
        }
    }

    std::vector<float> eigenvalues(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        eigenvalues[i] = mat[i*n + i];
    }

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return eigenvalues[a] > eigenvalues[b];
    });

    std::vector<float> sorted_values(n, 0.0f);
    std::vector<float> sorted_vectors(n * n, 0.0f);
    for (size_t col = 0; col < n; ++col) {
        sorted_values[col] = std::max(0.0f, eigenvalues[order[col]]);
        for (size_t row = 0; row < n; ++row) {
            sorted_vectors[row*n + col] = eigenvectors[row*n + order[col]];
        }
    }

    return { sorted_values, sorted_vectors };
}

std::vector<float> multiply_small_matrix_vector(const std::vector<float> & mat, size_t n, const std::vector<float> & vec) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            sum += mat[i*n + j] * vec[j];
        }
        out[i] = sum;
    }
    return out;
}

void normalize_vector(std::vector<float> & vec) {
    const float norm = fro_norm(vec);
    if (norm > DIRECTION_EPS) {
        for (float & value : vec) {
            value /= norm;
        }
    }
}

bool merge_directions(
        const low_rank_direction & target,
        float target_weight,
        const low_rank_direction & source,
        float source_weight,
        size_t target_rank,
        float gain_cap,
        float singular_value_floor,
        low_rank_direction & out) {
    const size_t in_dim = source.in_dim ? source.in_dim : target.in_dim;
    const size_t out_dim = source.out_dim ? source.out_dim : target.out_dim;

    if (in_dim == 0 || out_dim == 0 || target_rank == 0) {
        return false;
    }

    size_t composed_rank = 0;
    if (target.gain > DIRECTION_EPS && target_weight > 0.0f) {
        composed_rank += target.rank;
    }
    if (source.gain > DIRECTION_EPS && source_weight > 0.0f) {
        composed_rank += source.rank;
    }
    if (composed_rank == 0) {
        return false;
    }

    std::vector<float> u_total(out_dim * composed_rank, 0.0f);
    std::vector<float> v_total(in_dim * composed_rank, 0.0f);

    size_t col_offset = 0;
    auto append_direction = [&](const low_rank_direction & dir, float mix_weight) {
        if (dir.gain <= DIRECTION_EPS || mix_weight <= 0.0f || dir.rank == 0) {
            return;
        }

        const float scale = std::sqrt(std::max(0.0f, dir.gain * mix_weight));
        for (size_t k = 0; k < dir.rank; ++k) {
            for (size_t o = 0; o < out_dim; ++o) {
                u_total[o*composed_rank + col_offset + k] = dir.left[o*dir.rank + k] * scale;
            }
            for (size_t i = 0; i < in_dim; ++i) {
                v_total[i*composed_rank + col_offset + k] = dir.right[i*dir.rank + k] * scale;
            }
        }
        col_offset += dir.rank;
    };

    append_direction(target, target_weight);
    append_direction(source, source_weight);

    const thin_qr_result qr_u = thin_qr(u_total, out_dim, composed_rank);
    const thin_qr_result qr_v = thin_qr(v_total, in_dim, composed_rank);
    const std::vector<float> small = small_mat_mul_r_rt(qr_u.r, qr_v.r, composed_rank);
    const std::vector<float> gram = small_mat_transpose_mul(small, composed_rank);
    const auto eig = jacobi_eigen(gram, composed_rank);
    const std::vector<float> & eigenvalues = eig.first;
    const std::vector<float> & v_small = eig.second;

    std::vector<float> kept_sigma;
    std::vector<std::vector<float>> kept_u_small;
    std::vector<std::vector<float>> kept_v_small;

    for (size_t col = 0; col < composed_rank && kept_sigma.size() < target_rank; ++col) {
        const float sigma = std::sqrt(std::max(0.0f, eigenvalues[col]));
        if (sigma <= DIRECTION_EPS) {
            continue;
        }

        if (!kept_sigma.empty() && sigma < kept_sigma.front() * singular_value_floor) {
            continue;
        }

        std::vector<float> v_vec(composed_rank, 0.0f);
        for (size_t i = 0; i < composed_rank; ++i) {
            v_vec[i] = v_small[i*composed_rank + col];
        }

        std::vector<float> u_vec = multiply_small_matrix_vector(small, composed_rank, v_vec);
        for (float & value : u_vec) {
            value /= sigma;
        }
        normalize_vector(u_vec);
        normalize_vector(v_vec);

        kept_sigma.push_back(sigma);
        kept_u_small.push_back(std::move(u_vec));
        kept_v_small.push_back(std::move(v_vec));
    }

    if (kept_sigma.empty()) {
        return false;
    }

    const float natural_gain = fro_norm(kept_sigma);
    if (natural_gain <= DIRECTION_EPS) {
        return false;
    }

    out.out_dim = out_dim;
    out.in_dim = in_dim;
    out.rank = kept_sigma.size();
    out.gain = std::min(natural_gain, gain_cap);
    out.left.assign(out_dim * out.rank, 0.0f);
    out.right.assign(in_dim * out.rank, 0.0f);

    for (size_t comp = 0; comp < out.rank; ++comp) {
        const float direction_sigma = kept_sigma[comp] / natural_gain;
        const float factor_scale = std::sqrt(std::max(0.0f, direction_sigma));

        for (size_t o = 0; o < out_dim; ++o) {
            float value = 0.0f;
            for (size_t k = 0; k < composed_rank; ++k) {
                value += qr_u.q[o*composed_rank + k] * kept_u_small[comp][k];
            }
            out.left[o*out.rank + comp] = value * factor_scale;
        }

        for (size_t i = 0; i < in_dim; ++i) {
            float value = 0.0f;
            for (size_t k = 0; k < composed_rank; ++k) {
                value += qr_v.q[i*composed_rank + k] * kept_v_small[comp][k];
            }
            out.right[i*out.rank + comp] = value * factor_scale;
        }
    }

    return true;
}

void normalize_active_weight(
        llama_adapter_lora_weight & weight,
        float update_energy,
        float gain_decay,
        float gain_max) {
    std::vector<float> data_a(weight.a->ne[0] * weight.a->ne[1]);
    std::vector<float> data_b(weight.b->ne[0] * weight.b->ne[1]);
    ggml_backend_tensor_get(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_get(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));

    const float norm_a = fro_norm(data_a);
    const float norm_b = fro_norm(data_b);
    if (norm_a > DIRECTION_EPS) {
        for (float & value : data_a) {
            value /= norm_a;
        }
    }
    if (norm_b > DIRECTION_EPS) {
        for (float & value : data_b) {
            value /= norm_b;
        }
    }

    const float energy = update_energy * std::max(norm_a * norm_b, 1.0f);
    weight.gain = std::min(gain_max, std::max(0.0f, weight.gain * (1.0f - gain_decay) + energy));

    ggml_backend_tensor_set(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_set(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
}

llama_adapter_lora_layer_role past_bucket_role(size_t bucket) {
    switch (bucket) {
        case LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK:    return LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK;
        case LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH:   return LLAMA_ADAPTER_LORA_LAYER_PAST_MONTH;
        case LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER: return LLAMA_ADAPTER_LORA_LAYER_PAST_QUARTER;
        case LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR:    return LLAMA_ADAPTER_LORA_LAYER_PAST_YEAR;
        case LLAMA_MEMORY_LORA_BUCKET_ALL_TIME:
        default:                                    return LLAMA_ADAPTER_LORA_LAYER_ALL_TIME;
    }
}

llama_adapter_lora_layer_role functional_family_role(int32_t family) {
    switch (family) {
        case LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION:    return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION;
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL:    return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_COUNTERFACTUAL;
        case LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION:return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_MEMORY_COMPRESSION;
        case LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION: return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_SELF_OBSERVATION;
        default:                                      return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION;
    }
}

const char * functional_family_name(int32_t family) {
    switch (family) {
        case LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION:     return "tool_selection";
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL:     return "counterfactual";
        case LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION: return "memory_compression";
        case LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION:   return "self_observation";
        default:                                       return "unknown";
    }
}

} // namespace

struct llama_active_lora_manager::impl {
    explicit impl(llama_context & owner) : owner(owner) {}

    struct bucket_runtime {
        std::unique_ptr<llama_adapter_lora> adapter;
        llama_past_lora_bucket_stats stats = {};
    };

    struct job_runtime {
        uint64_t last_run_us = 0;
        uint32_t last_source_version = 0;
    };

    llama_context & owner;
    llama_active_lora_params params = llama_active_lora_default_params();
    llama_active_lora_stats stats = {};
    llama_past_lora_params past_params = llama_past_lora_default_params();
    llama_past_lora_stats past_stats = {};
    std::unique_ptr<llama_adapter_lora> adapter;
    std::unique_ptr<active_lora_embedder> embedder;
    std::array<bucket_runtime, PAST_BUCKET_COUNT> buckets = {};
    std::array<job_runtime, PAST_BUCKET_COUNT> jobs = {};
    std::array<std::unique_ptr<llama_adapter_lora>, LLAMA_FUNCTIONAL_LORA_COUNT> functional_adapters = {};
    std::array<llama_functional_lora_family_config, LLAMA_FUNCTIONAL_LORA_COUNT> functional_configs = {};
    std::array<llama_functional_lora_family_state, LLAMA_FUNCTIONAL_LORA_COUNT> functional_states = {};
    std::array<llama_functional_lora_update_info, LLAMA_FUNCTIONAL_LORA_COUNT> functional_updates = {};
    llama_functional_lora_trace functional_trace = {};
    llama_functional_lora_ablation_config functional_ablation = {};
    std::vector<ggml_tensor *> targets;
    active_lora_embedding last_embedding;
    uint64_t updates_seen = 0;
    uint64_t active_started_at_us = 0;
    uint64_t active_last_update_us = 0;
    uint32_t active_rollover_version = 0;
    llama_active_temporal_encoding_bias temporal_encoding_bias = {};
    temporal_bias_adam_state temporal_bias_adam = {};
    functional_gating_network gating_network = {};
    functional_gating_adam_state gating_adam = {};
    std::array<functional_gating_training_tuple, LLAMA_FUNCTIONAL_LORA_COUNT> gating_training = {};
    std::unordered_map<const ggml_tensor *, runtime_lora_adam_state> runtime_lora_adam = {};
    std::unordered_map<const llama_adapter_lora *, uint64_t> runtime_lora_write_count = {};
    std::mt19937_64 gating_rng { FUNCTIONAL_GATING_INIT_SEED };
    uint64_t gating_invocation_count = 0;
    bool initialized = false;
    bool past_initialized = false;

    void initialize_functional_defaults(uint32_t selected_rank) {
        functional_ablation = {};
        functional_trace = {};
        initialize_gating_network();
        for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
            auto & cfg = functional_configs[family];
            cfg.enabled = true;
            cfg.rank_min = std::max<uint32_t>(1, std::min<uint32_t>(selected_rank, params.min_rank));
            cfg.rank_max = std::max<uint32_t>(cfg.rank_min, selected_rank);
            cfg.gain_min = FUNCTIONAL_GAIN_CLIP_MIN;
            cfg.gain_max = FUNCTIONAL_GAIN_CLIP_MAX;
            cfg.gain_clip_min = FUNCTIONAL_GAIN_CLIP_MIN;
            cfg.gain_clip_max = FUNCTIONAL_GAIN_CLIP_MAX;
            cfg.default_gain = functional_default_gain(family);
            cfg.exploration_noise_initial_std = 0.08f;
            cfg.exploration_noise_min_std = 0.01f;
            cfg.exploration_noise_decay_invocations = 512;
            cfg.negative_update_scale = 0.65f;
            cfg.positive_update_scale = 1.0f;
            cfg.top_k_priority = functional_top_k_priority(family);
            cfg.update_horizon_steps = functional_update_horizon_steps(family);
            cfg.update_horizon_commands =
                    family == LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ? 1 : 0;
            cfg.allow_active_loop = family != LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
            cfg.allow_dmn_loop = true;

            auto & state = functional_states[family];
            state = {};
            state.family = family;
            state.enabled = true;
            state.compatible = true;
            state.predicted_gain = cfg.default_gain;

            functional_trace.holds[family] = {};
            functional_trace.holds[family].family = family;
            functional_trace.family_state[family] = state;
            functional_updates[family] = {};
            functional_updates[family].family = family;
            gating_training[family] = {};
        }
    }

    void initialize_gating_network() {
        std::normal_distribution<float> dist(0.0f, 0.05f);
        for (float & value : gating_network.w1) {
            value = dist(gating_rng);
        }
        for (float & value : gating_network.w2) {
            value = dist(gating_rng);
        }
        for (float & value : gating_network.b1) {
            value = 0.0f;
        }
        for (float & value : gating_network.b2) {
            value = 1.0f;
        }
        gating_adam = {};
        gating_invocation_count = 0;
    }

    void clear_runtime_optimizer(llama_adapter_lora & target_adapter) {
        runtime_lora_write_count.erase(&target_adapter);
        for (const auto & it : target_adapter.ab_map) {
            runtime_lora_adam.erase(it.second.a);
        }
    }

    runtime_lora_adam_state & ensure_runtime_adam_state(
            const llama_adapter_lora_weight & weight,
            size_t size_a,
            size_t size_b) {
        auto & state = runtime_lora_adam[weight.a];
        if (state.m_a.size() != size_a) {
            state.m_a.assign(size_a, 0.0f);
            state.v_a.assign(size_a, 0.0f);
        }
        if (state.m_b.size() != size_b) {
            state.m_b.assign(size_b, 0.0f);
            state.v_b.assign(size_b, 0.0f);
        }
        return state;
    }

    std::array<float, FUNCTIONAL_GATING_INPUT_DIM> build_gating_input(
            const llama_functional_gating_observation & observation,
            float * out_gradient_norm = nullptr) const {
        std::array<float, FUNCTIONAL_GATING_INPUT_DIM> input = {};
        float norm_sq = 0.0f;
        for (size_t i = 0; i < 9; ++i) {
            const float error = functional_target_value(i) - functional_snapshot_component(observation.snapshot, i);
            input[i] = error;
            norm_sq += error * error;
        }

        input[9]  = -clamp_unit(observation.uncertainty);
        input[10] = clamp_unit(observation.tool_affinity);
        input[11] = clamp_unit(observation.continuation);
        input[12] = clamp_unit(observation.memory_pressure);
        input[13] = clamp_unit(observation.recovery_urgency);
        input[14] = -clamp_unit(observation.prediction_error);
        input[15] = observation.loop_origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? 1.0f : 0.0f;
        input[16] = observation.loop_origin == LLAMA_COG_COMMAND_ORIGIN_DMN ? 1.0f : 0.0f;
        input[17] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION)) != 0 ? 1.0f : 0.0f;
        input[18] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL)) != 0 ? 1.0f : 0.0f;
        input[19] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION)) != 0 ? 1.0f : 0.0f;
        input[20] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION)) != 0 ? 1.0f : 0.0f;

        if (out_gradient_norm) {
            *out_gradient_norm = std::sqrt(norm_sq);
        }
        return input;
    }

    void forward_gating(
            const std::array<float, FUNCTIONAL_GATING_INPUT_DIM> & input,
            std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> * out_hidden,
            std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> * out_means) const {
        for (size_t h = 0; h < FUNCTIONAL_GATING_HIDDEN_DIM; ++h) {
            float sum = gating_network.b1[h];
            for (size_t i = 0; i < FUNCTIONAL_GATING_INPUT_DIM; ++i) {
                sum += gating_network.w1[h * FUNCTIONAL_GATING_INPUT_DIM + i] * input[i];
            }
            (*out_hidden)[h] = std::tanh(sum);
        }

        for (size_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
            float sum = gating_network.b2[family];
            for (size_t h = 0; h < FUNCTIONAL_GATING_HIDDEN_DIM; ++h) {
                sum += gating_network.w2[family * FUNCTIONAL_GATING_HIDDEN_DIM + h] * (*out_hidden)[h];
            }
            (*out_means)[family] = sum;
        }
    }

    bool predict_activation(
            const llama_functional_gating_observation & observation,
            const llama_functional_activation_decision & policy_seed,
            llama_functional_activation_decision * out_decision) {
        if (!out_decision) {
            return false;
        }

        llama_functional_activation_decision decision = policy_seed;
        decision.loop_origin = observation.loop_origin;
        decision.microphase = observation.microphase;
        decision.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
        decision.eligible_mask = policy_seed.eligible_mask;
        decision.activated_mask = 0;
        decision.top_family = -1;

        float gradient_norm = 0.0f;
        const auto input = build_gating_input(observation, &gradient_norm);
        std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> hidden = {};
        std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> means = {};
        forward_gating(input, &hidden, &means);

        const float exploration_std = gating_noise_std(functional_configs[0], gating_invocation_count);
        std::normal_distribution<float> noise_dist(0.0f, exploration_std);

        for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
            const bool eligible = (decision.eligible_mask & (1ull << family)) != 0;
            const float sampled_noise = eligible ? noise_dist(gating_rng) : 0.0f;
            const float unclipped = means[family] + sampled_noise;
            const float gain = eligible ?
                    clamp_range(unclipped, functional_configs[family].gain_clip_min, functional_configs[family].gain_clip_max) :
                    0.0f;
            decision.predicted_gains[family] = means[family];
            decision.sampled_noise[family] = sampled_noise;
            decision.gains[family] = gain;
            decision.priority[family] = gain;

            if (eligible && gain > 0.0f) {
                decision.activated_mask |= (1ull << family);
                if (decision.top_family < 0 || gain > decision.gains[decision.top_family]) {
                    decision.top_family = family;
                }

                auto & tuple = gating_training[family];
                tuple = {};
                tuple.valid = true;
                tuple.family = family;
                tuple.loop_origin = observation.loop_origin;
                tuple.microphase = observation.microphase;
                tuple.eligible_mask = decision.eligible_mask;
                tuple.activated_mask = decision.activated_mask;
                tuple.invocation_count = gating_invocation_count;
                tuple.allostatic_distance_before = functional_allostatic_distance(observation.snapshot);
                tuple.exploration_std = exploration_std;
                tuple.input = input;
                tuple.hidden = hidden;
                for (int32_t i = 0; i < LLAMA_FUNCTIONAL_LORA_COUNT; ++i) {
                    tuple.predicted_gains[i] = means[i];
                    tuple.sampled_noise[i] = decision.sampled_noise[i];
                    tuple.applied_gains[i] = decision.gains[i];
                }
            } else {
                gating_training[family].valid = false;
            }
        }

        decision.exploration_std = exploration_std;
        decision.allostatic_distance = functional_allostatic_distance(observation.snapshot);
        decision.allostatic_gradient_norm = gradient_norm;
        decision.gating_invocation_count = gating_invocation_count;
        gating_invocation_count += 1;
        *out_decision = decision;
        return true;
    }

    template<size_t N>
    static float adam_update_array(
            std::array<float, N> & params,
            std::array<float, N> & m,
            std::array<float, N> & v,
            const std::array<float, N> & grad,
            const functional_gating_adam_state & adam) {
        float update_norm = 0.0f;
        const float step_f = static_cast<float>(adam.step);
        const float beta1h = 1.0f - std::pow(adam.beta1, step_f);
        const float beta2h = 1.0f - std::pow(adam.beta2, step_f);
        for (size_t i = 0; i < N; ++i) {
            m[i] = adam.beta1 * m[i] + (1.0f - adam.beta1) * grad[i];
            v[i] = adam.beta2 * v[i] + (1.0f - adam.beta2) * grad[i] * grad[i];
            const float m_hat = beta1h > 0.0f ? m[i] / beta1h : m[i];
            const float v_hat = beta2h > 0.0f ? v[i] / beta2h : v[i];
            const float delta = adam.learning_rate * m_hat / (std::sqrt(v_hat) + adam.epsilon);
            params[i] -= delta;
            update_norm += delta * delta;
        }
        return update_norm;
    }

    static float adam_update_vector(
            std::vector<float> & params,
            std::vector<float> & m,
            std::vector<float> & v,
            const std::vector<float> & grad,
            const runtime_lora_adam_state & adam) {
        float update_norm = 0.0f;
        const float step_f = static_cast<float>(adam.step);
        const float beta1h = 1.0f - std::pow(adam.beta1, step_f);
        const float beta2h = 1.0f - std::pow(adam.beta2, step_f);
        for (size_t i = 0; i < params.size(); ++i) {
            m[i] = adam.beta1 * m[i] + (1.0f - adam.beta1) * grad[i];
            v[i] = adam.beta2 * v[i] + (1.0f - adam.beta2) * grad[i] * grad[i];
            const float m_hat = beta1h > 0.0f ? m[i] / beta1h : m[i];
            const float v_hat = beta2h > 0.0f ? v[i] / beta2h : v[i];
            const float delta = adam.learning_rate * m_hat / (std::sqrt(v_hat) + adam.epsilon);
            params[i] -= delta;
            update_norm += delta * delta;
        }
        return update_norm;
    }

    static float adam_update_scalar(
            float & param,
            float & m,
            float & v,
            float grad,
            const temporal_bias_adam_state & adam) {
        const float step_f = static_cast<float>(adam.step);
        const float beta1h = 1.0f - std::pow(adam.beta1, step_f);
        const float beta2h = 1.0f - std::pow(adam.beta2, step_f);
        m = adam.beta1 * m + (1.0f - adam.beta1) * grad;
        v = adam.beta2 * v + (1.0f - adam.beta2) * grad * grad;
        const float m_hat = beta1h > 0.0f ? m / beta1h : m;
        const float v_hat = beta2h > 0.0f ? v / beta2h : v;
        const float delta = adam.learning_rate * m_hat / (std::sqrt(v_hat) + adam.epsilon);
        param -= delta;
        return std::fabs(delta);
    }

    float apply_meta_update(
            int32_t family,
            const llama_functional_outcome_snapshot & before,
            const llama_functional_outcome_snapshot & after,
            llama_functional_lora_update_info * out_update) {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
            return 0.0f;
        }
        (void) before;
        auto & tuple = gating_training[family];
        if (!tuple.valid) {
            return 0.0f;
        }

        const float before_distance = tuple.allostatic_distance_before;
        const float after_distance = functional_allostatic_distance(after);
        const float meta_loss = after_distance - before_distance;
        const float std = std::max(1.0e-4f, tuple.exploration_std);
        const float effective_noise = tuple.applied_gains[family] - tuple.predicted_gains[family];
        const float grad_output = meta_loss * effective_noise / (std * std);

        std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT * FUNCTIONAL_GATING_HIDDEN_DIM> grad_w2 = {};
        std::array<float, LLAMA_FUNCTIONAL_LORA_COUNT> grad_b2 = {};
        std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> grad_hidden = {};
        std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM> grad_b1 = {};
        std::array<float, FUNCTIONAL_GATING_HIDDEN_DIM * FUNCTIONAL_GATING_INPUT_DIM> grad_w1 = {};

        grad_b2[family] = grad_output;
        for (size_t h = 0; h < FUNCTIONAL_GATING_HIDDEN_DIM; ++h) {
            grad_w2[family * FUNCTIONAL_GATING_HIDDEN_DIM + h] = grad_output * tuple.hidden[h];
            grad_hidden[h] = grad_output * gating_network.w2[family * FUNCTIONAL_GATING_HIDDEN_DIM + h];
        }
        for (size_t h = 0; h < FUNCTIONAL_GATING_HIDDEN_DIM; ++h) {
            const float local = grad_hidden[h] * (1.0f - tuple.hidden[h] * tuple.hidden[h]);
            grad_b1[h] = local;
            for (size_t i = 0; i < FUNCTIONAL_GATING_INPUT_DIM; ++i) {
                grad_w1[h * FUNCTIONAL_GATING_INPUT_DIM + i] = local * tuple.input[i];
            }
        }

        gating_adam.step += 1;
        float update_norm_sq = 0.0f;
        update_norm_sq += adam_update_array(gating_network.w2, gating_adam.m_w2, gating_adam.v_w2, grad_w2, gating_adam);
        update_norm_sq += adam_update_array(gating_network.b2, gating_adam.m_b2, gating_adam.v_b2, grad_b2, gating_adam);
        update_norm_sq += adam_update_array(gating_network.w1, gating_adam.m_w1, gating_adam.v_w1, grad_w1, gating_adam);
        update_norm_sq += adam_update_array(gating_network.b1, gating_adam.m_b1, gating_adam.v_b1, grad_b1, gating_adam);

        if (out_update) {
            out_update->meta_loss = meta_loss;
            out_update->allostatic_distance_before = before_distance;
            out_update->allostatic_distance_after = after_distance;
            out_update->exploration_std = tuple.exploration_std;
            out_update->parameter_update_norm = std::sqrt(update_norm_sq);
            out_update->adam_step = gating_adam.step;
        }

        tuple.valid = false;
        return meta_loss;
    }

    static llama_self_state_event default_event(
            const llama_token * tokens,
            size_t n_tokens,
            bool remediation) {
        return {
            /*.tokens =*/ tokens,
            /*.n_tokens =*/ n_tokens,
            /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
            /*.channel =*/ remediation ? LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL : LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ remediation ? 1.0f : 0.85f,
        };
    }

    active_lora_write_features build_write_features(
            const active_lora_embedding & embedding,
            const llama_self_state_event & event,
            const llama_self_state_feature_vector * provided_features,
            bool remediation) const {
        active_lora_write_features result;
        result.content_signal = embedding.values;
        normalize_vector(result.content_signal);

        llama_self_state_feature_vector features = {};
        bool have_features = false;
        if (provided_features) {
            features = *provided_features;
            have_features = true;
        } else {
            have_features = owner.self_state_build_postwrite_features(event, &features);
        }

        if (!have_features) {
            result.state_signal = {
                clamp_unit(static_cast<float>(std::log1p(static_cast<double>(event.n_tokens)) / 6.0)),
                event.role == LLAMA_SELF_STATE_EVENT_USER ? 1.0f : 0.0f,
                event.role == LLAMA_SELF_STATE_EVENT_TOOL ? 1.0f : 0.0f,
                event.role == LLAMA_SELF_STATE_EVENT_SYSTEM ? 1.0f : 0.0f,
                event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ? 1.0f : 0.0f,
                remediation ? 1.0f : 0.0f,
                decoder_entropy_feature(event.decoder_entropy),
                clamp_unit(1.0f - event.decoder_top_margin),
            };
            normalize_vector(result.state_signal);
            result.update_emphasis = remediation ? 0.75f : 0.55f;
            result.goal_alignment = remediation ? 0.60f : 0.40f;
            result.repair_pressure = remediation ? 0.70f : 0.30f;
            result.social_alignment = event.role == LLAMA_SELF_STATE_EVENT_USER ? 0.65f : 0.45f;
            return result;
        }

        result.goal_alignment = clamp_unit(
                0.45f * features.goal_top_similarity +
                0.20f * features.commitment_top_similarity +
                0.15f * features.social_trust +
                0.10f * features.social_reciprocity +
                0.10f * (1.0f - features.negative_user_valence));
        result.repair_pressure = clamp_unit(
                0.35f * features.contradiction_score +
                0.25f * features.uncertainty_score +
                0.20f * features.followup_hint +
                0.10f * features.tool_pending_pressure +
                0.10f * (1.0f - features.social_trust));
        result.social_alignment = clamp_unit(
                0.35f * features.social_trust +
                0.20f * features.social_reciprocity +
                0.15f * features.social_familiarity +
                0.15f * (1.0f - features.negative_user_valence) +
                0.15f * features.broadcast_inhibition_hint);
        result.update_emphasis = clamp_unit(
                0.15f +
                0.18f * features.novelty +
                0.18f * features.memory_write_pressure +
                0.17f * result.goal_alignment +
                0.12f * result.social_alignment +
                0.10f * features.tool_pending_pressure +
                0.10f * (remediation ? result.repair_pressure : 0.0f));

        result.state_signal = {
            clamp_unit(features.novelty),
            clamp_unit(features.topic_shift),
            clamp_unit(features.goal_top_similarity),
            clamp_unit(features.commitment_top_similarity),
            clamp_unit(features.working_memory_top_similarity),
            clamp_unit(features.memory_handle_top_similarity),
            clamp_unit(features.social_trust),
            clamp_unit(features.social_reciprocity),
            clamp_unit(features.tool_readiness_score),
            clamp_unit(features.tool_pending_pressure),
            clamp_unit(features.contradiction_score),
            clamp_unit(features.uncertainty_score),
            clamp_unit(features.memory_write_pressure),
            clamp_unit(features.broadcast_pressure_hint),
            clamp_unit(features.broadcast_inhibition_hint),
            clamp_unit(features.followup_hint),
            clamp_unit(features.negative_user_valence),
            clamp_unit(1.0f - features.negative_user_valence),
            clamp_unit(result.goal_alignment),
            clamp_unit(result.repair_pressure),
            clamp_unit(result.social_alignment),
            event.role == LLAMA_SELF_STATE_EVENT_USER ? 1.0f : 0.0f,
            event.role == LLAMA_SELF_STATE_EVENT_TOOL ? 1.0f : 0.0f,
            event.role == LLAMA_SELF_STATE_EVENT_SYSTEM ? 1.0f : 0.0f,
            event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ? 1.0f : 0.0f,
            (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED) ? 1.0f : 0.0f,
            remediation ? 1.0f : 0.0f,
            decoder_entropy_feature(event.decoder_entropy),
            clamp_unit(1.0f - event.decoder_top_margin),
        };
        normalize_vector(result.state_signal);
        return result;
    }

    static size_t select_slot(const active_lora_write_features & features, size_t rank) {
        if (rank == 0) {
            return 0;
        }

        uint64_t hash = hash_signal_prefix(features.content_signal, 24, 0x243f6a8885a308d3ULL);
        hash ^= hash_signal_prefix(features.state_signal, 24, 0x13198a2e03707344ULL);
        return static_cast<size_t>(hash % rank);
    }

    float tensor_update_scale(
            const std::string & tensor_name,
            const active_lora_write_features & features,
            bool remediation) const {
        float scale = 0.55f + 0.70f * features.update_emphasis;

        const bool is_attention = tensor_name.find("attn") != std::string::npos ||
                tensor_name.find("wq") != std::string::npos ||
                tensor_name.find("wv") != std::string::npos ||
                tensor_name.find("wo") != std::string::npos ||
                tensor_name.find("wqkv") != std::string::npos;
        const bool is_mlp = tensor_name.find("ffn") != std::string::npos;
        if (is_attention) {
            scale *= 0.90f + 0.40f * std::max(features.goal_alignment, remediation ? features.repair_pressure : features.social_alignment);
        } else if (is_mlp) {
            scale *= 0.85f + 0.35f * (0.5f * features.goal_alignment + 0.5f * features.social_alignment);
        }

        const size_t layer_index = parse_layer_index(tensor_name);
        if (layer_index != std::numeric_limits<size_t>::max()) {
            const float denom = static_cast<float>(std::max<int32_t>(1, owner.get_model().hparams.n_layer - 1));
            const float layer_ratio = static_cast<float>(layer_index) / denom;
            scale *= 0.70f + 0.30f * layer_ratio;
        }

        if (remediation) {
            scale *= 0.90f + 0.35f * features.repair_pressure;
        }

        return std::max(0.0f, scale);
    }

    uint32_t compute_rank(
            uint64_t host_budget_bytes,
            uint64_t device_budget_bytes,
            uint32_t max_rank) const {
        const uint64_t bytes_per_scalar = ggml_type_size(GGML_TYPE_F32);
        uint64_t host_bytes_per_rank = 0;
        uint64_t device_bytes_per_rank = 0;

        for (ggml_tensor * tensor : targets) {
            const uint64_t bytes_per_rank = bytes_per_scalar * (tensor->ne[0] + tensor->ne[1]);
            auto * buft = ggml_backend_buffer_get_type(tensor->buffer);
            if (ggml_backend_buft_is_host(buft)) {
                host_bytes_per_rank += bytes_per_rank;
            } else {
                device_bytes_per_rank += bytes_per_rank;
            }
        }

        uint64_t rank_limit = max_rank;
        if (host_bytes_per_rank > 0) {
            if (host_budget_bytes == 0) {
                return 0;
            }
            if (host_budget_bytes != std::numeric_limits<uint64_t>::max()) {
                rank_limit = std::min<uint64_t>(rank_limit, host_budget_bytes / host_bytes_per_rank);
            }
        }
        if (device_bytes_per_rank > 0) {
            if (device_budget_bytes == 0) {
                return 0;
            }
            if (device_budget_bytes != std::numeric_limits<uint64_t>::max()) {
                rank_limit = std::min<uint64_t>(rank_limit, device_budget_bytes / device_bytes_per_rank);
            }
        }

        return static_cast<uint32_t>(rank_limit);
    }

    bool create_runtime_adapter(
            std::unique_ptr<llama_adapter_lora> & out,
            uint32_t rank,
            const std::string & name_prefix,
            float initial_scale,
            llama_adapter_lora_layer_role role) {
        out = std::make_unique<llama_adapter_lora>();
        std::vector<std::pair<ggml_tensor *, uint32_t>> runtime_targets;
        runtime_targets.reserve(targets.size());
        for (ggml_tensor * tensor : targets) {
            runtime_targets.emplace_back(tensor, rank);
        }

        if (!llama_adapter_lora_init_runtime(const_cast<llama_model &>(owner.model), *out, runtime_targets, name_prefix)) {
            out.reset();
            return false;
        }

        zero_adapter(*out);
        owner.attach_adapter_runtime(out.get(), initial_scale, role);
        return true;
    }

    void set_adapter_scale(llama_adapter_lora * adapter_ptr, float scale, llama_adapter_lora_layer_role role) {
        owner.attach_adapter_runtime(adapter_ptr, scale, role);
    }

    static active_lora_write_features augment_functional_write_features(
            active_lora_write_features base,
            int32_t family,
            int32_t loop_origin,
            const float * metrics,
            size_t metric_count,
            float signed_outcome,
            float magnitude) {
        if (base.content_signal.empty()) {
            return base;
        }

        base.state_signal.push_back(family == LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ? 1.0f : 0.0f);
        base.state_signal.push_back(family == LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL ? 1.0f : 0.0f);
        base.state_signal.push_back(family == LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION ? 1.0f : 0.0f);
        base.state_signal.push_back(family == LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION ? 1.0f : 0.0f);
        base.state_signal.push_back(loop_origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? 1.0f : 0.0f);
        base.state_signal.push_back(loop_origin == LLAMA_COG_COMMAND_ORIGIN_DMN ? 1.0f : 0.0f);
        base.state_signal.push_back(clamp_unit(0.5f + 0.5f * signed_outcome));
        base.state_signal.push_back(clamp_unit(0.5f - 0.5f * signed_outcome));
        base.state_signal.push_back(clamp_unit(magnitude));
        for (size_t i = 0; i < metric_count; ++i) {
            base.state_signal.push_back(clamp_unit(metrics[i]));
        }

        base.update_emphasis = clamp_unit(std::max(base.update_emphasis, 0.15f + 0.85f * magnitude));
        base.goal_alignment = clamp_unit(std::max(base.goal_alignment, 0.5f + 0.5f * std::max(0.0f, signed_outcome)));
        base.repair_pressure = clamp_unit(std::max(base.repair_pressure, 0.5f + 0.5f * std::max(0.0f, -signed_outcome)));
        normalize_vector(base.state_signal);
        return base;
    }

    bool train_on_adapter(
            llama_adapter_lora & target_adapter,
            const active_lora_embedding & embedding,
            const active_lora_write_features & write_features,
            float signed_budget_scale,
            bool remediation,
            runtime_lora_write_result * out_result = nullptr) {
        if (out_result) {
            *out_result = {};
        }
        if (embedding.values.empty()) {
            return false;
        }

        const float sign = signed_budget_scale >= 0.0f ? 1.0f : -1.0f;
        const float base_scale = params.learning_rate > 0.0f ? params.learning_rate : 1.0e-4f;
        const float update_scale = base_scale * clamp_unit(std::fabs(signed_budget_scale)) * std::max(0.15f, write_features.update_emphasis);
        if (update_scale <= 0.0f) {
            return false;
        }

        bool any_written = false;
        float total_update_norm_sq = 0.0f;
        for (auto & it : target_adapter.ab_map) {
            ggml_tensor * tensor_a = it.second.a;
            ggml_tensor * tensor_b = it.second.b;
            const size_t ne0_a = tensor_a->ne[0];
            const size_t ne1_a = tensor_a->ne[1];
            const size_t ne0_b = tensor_b->ne[0];
            const size_t ne1_b = tensor_b->ne[1];

            if (ne0_a == 0 || ne1_a == 0 || ne0_b == 0 || ne1_b == 0) {
                continue;
            }

            std::vector<float> data_a(ne0_a * ne1_a);
            std::vector<float> data_b(ne0_b * ne1_b);
            std::vector<float> grad_a(data_a.size(), 0.0f);
            std::vector<float> grad_b(data_b.size(), 0.0f);
            ggml_backend_tensor_get(tensor_a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_get(tensor_b, data_b.data(), 0, data_b.size() * sizeof(float));

            const size_t slot = select_slot(write_features, ne1_a);
            const float tensor_scale = sign * update_scale * tensor_update_scale(it.first, write_features, remediation);
            if (tensor_scale == 0.0f) {
                continue;
            }

            if (params.weight_decay > 0.0f) {
                const float decay = std::max(0.0f, 1.0f - std::min(0.95f, params.weight_decay));
                for (float & value : data_a) {
                    value *= decay;
                }
                for (float & value : data_b) {
                    value *= decay;
                }
            }

            const uint64_t right_seed = hash_signal_prefix(write_features.content_signal, 16, mix_seed(hash_signal_prefix(write_features.state_signal, 16, 0x9e3779b97f4a7c15ULL) + slot));
            const uint64_t left_seed = mix_seed(right_seed ^ hash_signal_prefix(write_features.state_signal, 16, 0xbf58476d1ce4e5b9ULL));
            for (size_t i = 0; i < ne0_a; ++i) {
                const float content = sample_signal(write_features.content_signal, i, right_seed);
                const float state = sample_signal(write_features.state_signal, i, right_seed ^ 0x94d049bb133111ebULL);
                grad_a[slot*ne0_a + i] = -sign * (0.78f * content + 0.22f * state);
            }

            for (size_t j = 0; j < ne1_b; ++j) {
                const float steer = sample_signal(write_features.state_signal, j, left_seed);
                const float content = sample_signal(write_features.content_signal, j, left_seed ^ 0xd6e8feb86659fd93ULL);
                grad_b[j*ne0_b + slot] = -sign * (0.65f * steer + 0.35f * content);
            }

            auto & adam = ensure_runtime_adam_state(it.second, data_a.size(), data_b.size());
            adam.learning_rate = std::fabs(tensor_scale);
            adam.step += 1;
            float update_norm_sq = 0.0f;
            update_norm_sq += adam_update_vector(data_a, adam.m_a, adam.v_a, grad_a, adam);
            update_norm_sq += adam_update_vector(data_b, adam.m_b, adam.v_b, grad_b, adam);
            adam.last_update_norm = std::sqrt(update_norm_sq);

            ggml_backend_tensor_set(tensor_a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_set(tensor_b, data_b.data(), 0, data_b.size() * sizeof(float));
            normalize_active_weight(it.second, std::fabs(tensor_scale), params.gain_decay, params.gain_max);
            total_update_norm_sq += update_norm_sq;
            any_written = true;
        }

        if (!any_written) {
            return false;
        }

        const uint64_t transaction_step = ++runtime_lora_write_count[&target_adapter];
        if (out_result) {
            out_result->updated = true;
            out_result->transaction_step = transaction_step;
            out_result->update_norm = std::sqrt(total_update_norm_sq);
        }
        return true;
    }

    bool train_on_span(
            const active_lora_embedding & embedding,
            const active_lora_write_features & write_features,
            float budget_scale,
            bool remediation,
            runtime_lora_write_result * out_result = nullptr) {
        if (embedding.values.empty() || !adapter) {
            return false;
        }
        return train_on_adapter(*adapter, embedding, write_features, budget_scale, remediation, out_result);
    }

    bool apply_write(
            const llama_self_state_event & event,
            const llama_self_state_feature_vector * features,
            float budget_scale,
            bool remediation) {
        if (!event.tokens || event.n_tokens == 0) {
            return false;
        }

        const active_lora_embedding embedding = embedder->embed(event.tokens, event.n_tokens);
        if (embedding.values.empty()) {
            return false;
        }

        if (!remediation) {
            const float similarity = cosine_similarity(embedding, last_embedding);
            if (similarity > 0.995f) {
                LLAMA_LOG_INFO("%s: skipping Active LoRA write for highly redundant span (cos=%.4f)\n", __func__, similarity);
                return true;
            }
        }

        float effective_budget_scale = budget_scale;
        if (!remediation) {
            const float write_scale =
                    std::max(0.20f, temporal_encoding_bias.effective_write_scale > 0.0f ?
                            temporal_encoding_bias.effective_write_scale :
                            1.0f);
            effective_budget_scale *= write_scale;
        }

        const active_lora_write_features write_features = build_write_features(embedding, event, features, remediation);
        runtime_lora_write_result write_result = {};
        if (!train_on_span(embedding, write_features, effective_budget_scale, remediation, &write_result)) {
            return false;
        }

        last_embedding = embedding;
        stats.updates_applied++;
        stats.optimizer_step_count = write_result.transaction_step;
        stats.tokens_ingested += event.n_tokens;
        updates_seen++;
        stats.rollover_ready = updates_seen >= params.max_updates_before_rollover;
        active_last_update_us = llama_time_us();
        update_active_gain_stats();
        stats.optimizer_last_update_norm = write_result.update_norm;

        if (remediation) {
            LLAMA_LOG_INFO("%s: Active LoRA remediation applied (%zu tokens, budget=%.3f, update=%u)\n",
                    __func__, event.n_tokens, budget_scale, stats.updates_applied);
        } else {
            LLAMA_LOG_INFO("%s: Active LoRA write accepted (%zu tokens, update=%u, rollover_ready=%s)\n",
                    __func__, event.n_tokens, stats.updates_applied, stats.rollover_ready ? "true" : "false");
        }

        return true;
    }

    void update_active_gain_stats() {
        const auto gains = adapter ? adapter_gain_stats(*adapter) : std::pair<float, float>(0.0f, 0.0f);
        stats.gain_mean = gains.first;
        stats.gain_max = gains.second;
    }

    void reset_active_stage(uint64_t now_us) {
        if (adapter) {
            clear_runtime_optimizer(*adapter);
            zero_adapter(*adapter);
        }

        last_embedding = {};
        updates_seen = 0;
        stats.rollover_ready = false;
        stats.updates_applied = 0;
        stats.optimizer_step_count = 0;
        stats.tokens_ingested = 0;
        stats.gain_mean = 0.0f;
        stats.gain_max = 0.0f;
        stats.optimizer_last_update_norm = 0.0f;
        active_started_at_us = now_us;
        active_last_update_us = now_us;
        temporal_encoding_bias.effective_write_scale =
                clamp_range(1.0f + temporal_encoding_bias.reward_bias - temporal_encoding_bias.dampening_bias, 0.20f, 1.80f);
    }

    bool merge_source_into_bucket(
            const llama_adapter_lora & source_adapter,
            uint64_t source_window_start_us,
            uint64_t source_window_end_us,
            size_t target_bucket,
            uint64_t now_us) {
        auto & bucket = buckets[target_bucket];
        if (!bucket.adapter) {
            return false;
        }

        const float source_weight = past_params.merge_source_weight[target_bucket];
        const float target_weight = bucket.stats.populated ? past_params.merge_target_retention[target_bucket] : 0.0f;
        bool any_written = false;

        for (auto & it : bucket.adapter->ab_map) {
            auto src_it = source_adapter.ab_map.find(it.first);
            if (src_it == source_adapter.ab_map.end()) {
                continue;
            }

            low_rank_direction merged;
            const low_rank_direction target_dir = read_direction(it.second);
            const low_rank_direction source_dir = read_direction(src_it->second);
            if (!merge_directions(
                        target_dir,
                        target_weight,
                        source_dir,
                        source_weight,
                        bucket.stats.selected_rank,
                        past_params.gain_max,
                        past_params.singular_value_floor,
                        merged)) {
                zero_weight(it.second);
                continue;
            }

            write_direction(it.second, merged);
            any_written = true;
        }

        if (!any_written) {
            return false;
        }

        bucket.stats.populated = true;
        bucket.stats.version += 1;
        bucket.stats.created_at_us = now_us;
        if (bucket.stats.source_window_start_us == 0 || target_weight == 0.0f) {
            bucket.stats.source_window_start_us = source_window_start_us;
        } else {
            bucket.stats.source_window_start_us = std::min(bucket.stats.source_window_start_us, source_window_start_us);
        }
        bucket.stats.source_window_end_us = std::max(bucket.stats.source_window_end_us, source_window_end_us);
        bucket.stats.base_scale = past_params.base_scale[target_bucket];
        const auto gains = adapter_gain_stats(*bucket.adapter);
        bucket.stats.gain_mean = gains.first;
        bucket.stats.gain_max = gains.second;
        return true;
    }

    bool freeze_active_into_week(uint64_t now_us) {
        if (!stats.rollover_ready || !adapter) {
            return false;
        }

        if (!merge_source_into_bucket(*adapter, active_started_at_us, active_last_update_us, LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK, now_us)) {
            return false;
        }

        jobs[ACTIVE_TO_WEEK_JOB].last_run_us = now_us;
        active_rollover_version += 1;
        reset_active_stage(now_us);
        return true;
    }

    bool condense_bucket(size_t source_bucket, size_t target_bucket, uint64_t now_us) {
        auto & source = buckets[source_bucket];
        if (!source.stats.populated || !source.adapter) {
            return false;
        }

        if (!merge_source_into_bucket(
                    *source.adapter,
                    source.stats.source_window_start_us,
                    source.stats.source_window_end_us,
                    target_bucket,
                    now_us)) {
            return false;
        }

        jobs[source_bucket + 1].last_run_us = now_us;
        jobs[source_bucket + 1].last_source_version = source.stats.version;
        return true;
    }

    void refresh_bucket_scales(uint64_t now_us) {
        if (!past_initialized) {
            return;
        }

        for (size_t bucket = 0; bucket < PAST_BUCKET_COUNT; ++bucket) {
            float scale = 0.0f;
            if (buckets[bucket].stats.populated) {
                scale = compute_effective_scale(
                        past_params.base_scale[bucket],
                        buckets[bucket].stats.created_at_us,
                        now_us,
                        past_params.decay_half_life_us[bucket]);
            }

            buckets[bucket].stats.base_scale = past_params.base_scale[bucket];
            buckets[bucket].stats.effective_scale = scale;
            if (buckets[bucket].adapter) {
                set_adapter_scale(buckets[bucket].adapter.get(), scale, past_bucket_role(bucket));
            }
            past_stats.buckets[bucket] = buckets[bucket].stats;
        }
    }

    uint64_t compute_pending_job_mask(uint64_t now_us) const {
        if (!past_initialized) {
            return 0;
        }

        uint64_t mask = 0;
        const bool active_period_elapsed =
                stats.updates_applied > 0 &&
                past_params.condensation_period_us[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK] > 0 &&
                now_us >= active_started_at_us + past_params.condensation_period_us[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK];
        if (stats.rollover_ready || active_period_elapsed) {
            mask |= (1ull << ACTIVE_TO_WEEK_JOB);
        }

        for (size_t source_bucket = 0; source_bucket + 1 < PAST_BUCKET_COUNT; ++source_bucket) {
            const auto & source = buckets[source_bucket];
            if (!source.stats.populated) {
                continue;
            }

            const uint32_t job_idx = source_bucket + 1;
            if (jobs[job_idx].last_source_version == source.stats.version) {
                continue;
            }

            const uint64_t period = past_params.condensation_period_us[source_bucket];
            if (period == 0 || now_us >= source.stats.created_at_us + period) {
                mask |= (1ull << job_idx);
            }
        }

        return mask;
    }
};

llama_active_lora_manager::llama_active_lora_manager(llama_context & owner) :
    pimpl(std::make_unique<impl>(owner)) {
}

llama_active_lora_manager::~llama_active_lora_manager() {
    if (pimpl->adapter) {
        pimpl->owner.detach_adapter_runtime(pimpl->adapter.get());
        const_cast<llama_model &>(pimpl->owner.model).loras.erase(pimpl->adapter.get());
    }
    for (auto & adapter : pimpl->functional_adapters) {
        if (adapter) {
            pimpl->owner.detach_adapter_runtime(adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(adapter.get());
        }
    }
    for (auto & bucket : pimpl->buckets) {
        if (bucket.adapter) {
            pimpl->owner.detach_adapter_runtime(bucket.adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(bucket.adapter.get());
        }
    }
}

bool llama_active_lora_manager::init(const llama_active_lora_params & params) {
    if (!params.enabled) {
        LLAMA_LOG_ERROR("%s: Active LoRA init requested while disabled\n", __func__);
        return false;
    }

    if (params.embedding_callback != nullptr && params.embedding_dim == 0) {
        LLAMA_LOG_ERROR("%s: custom Active LoRA embedder requires a non-zero embedding_dim\n", __func__);
        return false;
    }

    pimpl->params = params;
    pimpl->embedder = make_embedder(pimpl->owner, params);
    pimpl->targets = collect_memory_targets(pimpl->owner.model);

    const uint64_t host_free_bytes = get_host_free_bytes();
    const uint64_t device_free_bytes = get_device_free_bytes(pimpl->owner.model);
    const uint64_t host_budget_bytes = scale_memory_budget(host_free_bytes, params.host_memory_ratio);
    const uint64_t device_budget_bytes = scale_memory_budget(device_free_bytes, params.device_memory_ratio);
    const uint32_t selected_rank = pimpl->compute_rank(host_budget_bytes, device_budget_bytes, params.max_rank);
    if (selected_rank < params.min_rank) {
        LLAMA_LOG_ERROR("%s: failed to plan Active LoRA rank (planned %u, min %u)\n",
                __func__, selected_rank, params.min_rank);
        return false;
    }

    if (!pimpl->adapter) {
        if (!pimpl->create_runtime_adapter(
                    pimpl->adapter,
                    selected_rank,
                    "memory.active",
                    params.adapter_scale,
                    LLAMA_ADAPTER_LORA_LAYER_ACTIVE)) {
            return false;
        }
    }

    pimpl->stats.enabled = true;
    pimpl->stats.rollover_ready = false;
    pimpl->stats.selected_rank = selected_rank;
    pimpl->stats.updates_applied = 0;
    pimpl->stats.optimizer_step_count = 0;
    pimpl->stats.tokens_ingested = 0;
    pimpl->stats.host_memory_ratio = params.host_memory_ratio;
    pimpl->stats.device_memory_ratio = params.device_memory_ratio;
    pimpl->stats.host_budget_bytes = host_budget_bytes;
    pimpl->stats.device_budget_bytes = device_budget_bytes;
    pimpl->stats.embedding_dim = pimpl->embedder->dim();
    pimpl->stats.embedding_is_custom = pimpl->embedder->is_custom();
    pimpl->stats.embedding_type = pimpl->embedder->type();
    pimpl->stats.gain_mean = 0.0f;
    pimpl->stats.gain_max = 0.0f;
    pimpl->stats.optimizer_last_update_norm = 0.0f;
    pimpl->runtime_lora_adam.clear();
    pimpl->runtime_lora_write_count.clear();
    pimpl->temporal_bias_adam = {};
    pimpl->temporal_encoding_bias.adam_step = 0;
    pimpl->temporal_encoding_bias.last_update_norm = 0.0f;
    pimpl->initialized = true;
    pimpl->reset_active_stage(llama_time_us());
    pimpl->initialize_functional_defaults(selected_rank);

    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        if (!pimpl->functional_adapters[family]) {
            if (!pimpl->create_runtime_adapter(
                        pimpl->functional_adapters[family],
                        selected_rank,
                        std::string("functional.") + functional_family_name(family),
                        0.0f,
                        functional_family_role(family))) {
                return false;
            }
        }
        pimpl->functional_states[family].compatible = pimpl->functional_adapters[family] != nullptr;
        pimpl->functional_trace.family_state[family] = pimpl->functional_states[family];
    }

    LLAMA_LOG_INFO("%s: initialized Active LoRA (rank=%u, host_budget=%" PRIu64 ", device_budget=%" PRIu64 ", embedder=%d)\n",
            __func__, selected_rank, host_budget_bytes, device_budget_bytes, pimpl->stats.embedding_type);

    return true;
}

bool llama_active_lora_manager::init_past(const llama_past_lora_params & params) {
    if (!pimpl->initialized || !pimpl->adapter) {
        LLAMA_LOG_ERROR("%s: Active LoRA must be initialized before the past stack\n", __func__);
        return false;
    }
    if (!params.enabled) {
        LLAMA_LOG_ERROR("%s: past LoRA init requested while disabled\n", __func__);
        return false;
    }

    pimpl->past_params = params;
    const uint64_t host_free_bytes = get_host_free_bytes();
    const uint64_t device_free_bytes = get_device_free_bytes(pimpl->owner.model);

    for (size_t bucket = 0; bucket < PAST_BUCKET_COUNT; ++bucket) {
        const uint64_t host_budget_bytes = scale_memory_budget(host_free_bytes, params.host_memory_ratio[bucket]);
        const uint64_t device_budget_bytes = scale_memory_budget(device_free_bytes, params.device_memory_ratio[bucket]);
        const uint32_t selected_rank = pimpl->compute_rank(host_budget_bytes, device_budget_bytes, params.max_rank[bucket]);
        if (selected_rank < params.min_rank[bucket]) {
            LLAMA_LOG_ERROR("%s: failed to plan rank for %s (planned %u, min %u)\n",
                    __func__, past_bucket_name(bucket), selected_rank, params.min_rank[bucket]);
            return false;
        }

        if (!pimpl->buckets[bucket].adapter) {
            if (!pimpl->create_runtime_adapter(
                        pimpl->buckets[bucket].adapter,
                        selected_rank,
                        std::string("memory.") + past_bucket_name(bucket),
                        0.0f,
                        past_bucket_role(bucket))) {
                return false;
            }
        }

        pimpl->buckets[bucket].stats = {};
        pimpl->buckets[bucket].stats.selected_rank = selected_rank;
        pimpl->buckets[bucket].stats.host_budget_bytes = host_budget_bytes;
        pimpl->buckets[bucket].stats.device_budget_bytes = device_budget_bytes;
        pimpl->buckets[bucket].stats.base_scale = params.base_scale[bucket];
        pimpl->set_adapter_scale(pimpl->buckets[bucket].adapter.get(), 0.0f, past_bucket_role(bucket));
    }

    pimpl->past_stats = {};
    pimpl->past_stats.enabled = true;
    pimpl->past_stats.last_tick_us = llama_time_us();
    pimpl->past_initialized = true;
    pimpl->refresh_bucket_scales(pimpl->past_stats.last_tick_us);
    pimpl->past_stats.pending_job_mask = pimpl->compute_pending_job_mask(pimpl->past_stats.last_tick_us);

    return true;
}

bool llama_active_lora_manager::ingest(const llama_token * tokens, size_t n_tokens) {
    return pimpl->initialized && pimpl->apply_write(pimpl->default_event(tokens, n_tokens, false), nullptr, 1.0f, false);
}

bool llama_active_lora_manager::ingest(const llama_self_state_event & event, const llama_self_state_feature_vector * features) {
    return pimpl->initialized && pimpl->apply_write(event, features, 1.0f, false);
}

bool llama_active_lora_manager::remediate(const llama_token * tokens, size_t n_tokens, float budget_scale) {
    return pimpl->initialized && pimpl->apply_write(pimpl->default_event(tokens, n_tokens, true), nullptr, budget_scale, true);
}

bool llama_active_lora_manager::remediate(
        const llama_self_state_event & event,
        float budget_scale,
        const llama_self_state_feature_vector * features) {
    return pimpl->initialized && pimpl->apply_write(event, features, budget_scale, true);
}

bool llama_active_lora_manager::get_stats(llama_active_lora_stats * out_stats) const {
    if (!out_stats || !pimpl->initialized) {
        return false;
    }

    *out_stats = pimpl->stats;
    return true;
}

bool llama_active_lora_manager::tick_past(uint64_t now_us) {
    if (!pimpl->past_initialized) {
        return false;
    }

    const uint64_t pending = pimpl->compute_pending_job_mask(now_us);
    pimpl->past_stats.pending_job_mask = pending;
    pimpl->past_stats.last_tick_us = now_us;

    if (pending & (1ull << ACTIVE_TO_WEEK_JOB)) {
        (void) pimpl->freeze_active_into_week(now_us);
    }
    if (pending & (1ull << WEEK_TO_MONTH_JOB)) {
        (void) pimpl->condense_bucket(LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK, LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH, now_us);
    }
    if (pending & (1ull << MONTH_TO_QUARTER_JOB)) {
        (void) pimpl->condense_bucket(LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH, LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER, now_us);
    }
    if (pending & (1ull << QUARTER_TO_YEAR_JOB)) {
        (void) pimpl->condense_bucket(LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER, LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR, now_us);
    }
    if (pending & (1ull << YEAR_TO_ALL_TIME_JOB)) {
        (void) pimpl->condense_bucket(LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR, LLAMA_MEMORY_LORA_BUCKET_ALL_TIME, now_us);
    }

    pimpl->refresh_bucket_scales(now_us);
    pimpl->past_stats.pending_job_mask = pimpl->compute_pending_job_mask(now_us);
    return true;
}

bool llama_active_lora_manager::get_past_stats(llama_past_lora_stats * out_stats) const {
    if (!out_stats || !pimpl->past_initialized) {
        return false;
    }

    *out_stats = pimpl->past_stats;
    return true;
}

int32_t llama_active_lora_manager::functional_family_count() {
    return LLAMA_FUNCTIONAL_LORA_COUNT;
}

bool llama_active_lora_manager::functional_family_config_get(int32_t family, llama_functional_lora_family_config * out_config) const {
    if (!out_config || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }
    *out_config = pimpl->functional_configs[family];
    return true;
}

bool llama_active_lora_manager::functional_family_state_get(int32_t family, llama_functional_lora_family_state * out_state) const {
    if (!out_state || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }
    *out_state = pimpl->functional_states[family];
    return true;
}

bool llama_active_lora_manager::functional_get_last_trace(llama_functional_lora_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = pimpl->functional_trace;
    return true;
}

bool llama_active_lora_manager::functional_get_last_update(int32_t family, llama_functional_lora_update_info * out_update) const {
    if (!out_update || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }
    *out_update = pimpl->functional_updates[family];
    return true;
}

bool llama_active_lora_manager::functional_set_ablation(const llama_functional_lora_ablation_config & config) {
    pimpl->functional_ablation = config;
    return true;
}

bool llama_active_lora_manager::functional_get_ablation(llama_functional_lora_ablation_config * out_config) const {
    if (!out_config) {
        return false;
    }
    *out_config = pimpl->functional_ablation;
    return true;
}

bool llama_active_lora_manager::temporal_encoding_bias_get(llama_active_temporal_encoding_bias * out_bias) const {
    if (!out_bias) {
        return false;
    }
    *out_bias = pimpl->temporal_encoding_bias;
    return true;
}

bool llama_active_lora_manager::temporal_encoding_bias_apply(
        float signed_advantage,
        float efficiency_advantage,
        int64_t monotonic_ms) {
    if (!pimpl->initialized) {
        return false;
    }

    auto & bias = pimpl->temporal_encoding_bias;
    const float magnitude = clamp_unit(
            0.65f * std::fabs(clamp_range(signed_advantage, -1.0f, 1.0f)) +
            0.35f * std::fabs(clamp_range(efficiency_advantage, -1.0f, 1.0f)));
    const float reward_signal = clamp_unit(
            0.75f * std::max(0.0f, signed_advantage) +
            0.25f * std::max(0.0f, efficiency_advantage));
    const float dampening_signal = clamp_unit(
            0.75f * std::max(0.0f, -signed_advantage) +
            0.25f * std::max(0.0f, -efficiency_advantage));
    auto & adam = pimpl->temporal_bias_adam;
    adam.learning_rate = 0.01f + 0.04f * magnitude;
    adam.step += 1;

    const float reward_grad =
            reward_signal > 0.0f ?
            -(reward_signal - 0.10f * bias.reward_bias) :
            0.10f * bias.reward_bias;
    const float dampening_grad =
            dampening_signal > 0.0f ?
            -(dampening_signal - 0.10f * bias.dampening_bias) :
            0.10f * bias.dampening_bias;
    const float reward_delta = llama_active_lora_manager::impl::adam_update_scalar(
            bias.reward_bias,
            adam.m_reward,
            adam.v_reward,
            reward_grad,
            adam);
    const float dampening_delta = llama_active_lora_manager::impl::adam_update_scalar(
            bias.dampening_bias,
            adam.m_dampening,
            adam.v_dampening,
            dampening_grad,
            adam);
    bias.reward_bias = clamp_range(bias.reward_bias, 0.0f, 0.35f);
    bias.dampening_bias = clamp_range(bias.dampening_bias, 0.0f, 0.35f);

    bias.effective_write_scale = clamp_range(1.0f + bias.reward_bias - bias.dampening_bias, 0.20f, 1.80f);
    bias.last_update_norm = std::sqrt(reward_delta * reward_delta + dampening_delta * dampening_delta);
    bias.adam_step = adam.step;
    bias.applied_update_count += 1;
    bias.last_update_monotonic_ms = monotonic_ms;
    return true;
}

bool llama_active_lora_manager::functional_predict_activation(
        const llama_functional_gating_observation & observation,
        const llama_functional_activation_decision & policy_seed,
        llama_functional_activation_decision * out_decision) {
    if (!pimpl->initialized) {
        return false;
    }
    return pimpl->predict_activation(observation, policy_seed, out_decision);
}

bool llama_active_lora_manager::functional_activate(const llama_functional_activation_decision & decision) {
    if (!pimpl->initialized) {
        return false;
    }

    const bool disable_stack = pimpl->functional_ablation.disable_functional_stack;
    const bool disable_holds = pimpl->functional_ablation.disable_hold_windows;
    pimpl->functional_trace.last_activation = decision;

    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        auto & hold = pimpl->functional_trace.holds[family];
        auto & state = pimpl->functional_states[family];
        hold.family = family;

        if (hold.active && hold.loop_origin == decision.loop_origin) {
            if (hold.hold_unit == LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS && hold.hold_remaining > 0) {
                --hold.hold_remaining;
            } else if (hold.hold_unit == LLAMA_FUNCTIONAL_HOLD_PHASE_EXIT && hold.microphase_current != decision.microphase) {
                hold.hold_remaining = 0;
            }
            if (hold.hold_remaining == 0) {
                hold.active = false;
                hold.gain = 0.0f;
            }
        }

        const bool family_disabled = (pimpl->functional_ablation.disabled_family_mask & (1ull << family)) != 0;
        const bool microphase_disabled =
                decision.microphase >= 0 &&
                decision.microphase < 63 &&
                (pimpl->functional_ablation.disabled_microphase_mask & (1ull << decision.microphase)) != 0;
        const bool activate_now =
                !disable_stack &&
                !family_disabled &&
                !microphase_disabled &&
                (decision.activated_mask & (1ull << family)) != 0 &&
                decision.gains[family] > 0.0f;

        if (activate_now) {
            hold.active = true;
            hold.loop_origin = decision.loop_origin;
            hold.microphase_started = decision.microphase;
            hold.microphase_current = decision.microphase;
            hold.hold_unit = disable_holds ? LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS : decision.hold_unit[family];
            hold.hold_budget = disable_holds ? 1u : std::max<uint32_t>(1u, decision.hold_value[family]);
            hold.hold_remaining = hold.hold_budget;
            hold.gain = clamp_range(
                    decision.gains[family],
                    pimpl->functional_configs[family].gain_clip_min,
                    pimpl->functional_configs[family].gain_clip_max);
        } else if (hold.active) {
            hold.microphase_current = decision.microphase;
        }

        const float gain = hold.active ? hold.gain : 0.0f;
        if (pimpl->functional_adapters[family]) {
            pimpl->set_adapter_scale(pimpl->functional_adapters[family].get(), gain, functional_family_role(family));
        }

        state.family = family;
        state.enabled = pimpl->functional_configs[family].enabled && !family_disabled;
        state.compatible = pimpl->functional_adapters[family] != nullptr;
        state.active_now = gain > 0.0f;
        state.current_gain = gain;
        state.predicted_gain = decision.predicted_gains[family];
        state.last_noise = decision.sampled_noise[family];
        state.current_microphase = hold.active ? decision.microphase : LLAMA_FUNCTIONAL_MICROPHASE_NONE;
        state.current_hold_unit = hold.active ? hold.hold_unit : LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS;
        state.current_hold_remaining = hold.active ? hold.hold_remaining : 0;
        pimpl->functional_trace.family_state[family] = state;
    }

    return true;
}

bool llama_active_lora_manager::functional_note_command_complete(int32_t origin) {
    if (!pimpl->initialized) {
        return false;
    }

    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        auto & hold = pimpl->functional_trace.holds[family];
        if (!hold.active || hold.loop_origin != origin) {
            continue;
        }
        if (hold.hold_unit == LLAMA_FUNCTIONAL_HOLD_COMMANDS && hold.hold_remaining > 0) {
            --hold.hold_remaining;
            if (hold.hold_remaining == 0) {
                hold.active = false;
                hold.gain = 0.0f;
                if (pimpl->functional_adapters[family]) {
                    pimpl->set_adapter_scale(pimpl->functional_adapters[family].get(), 0.0f, functional_family_role(family));
                }
                pimpl->functional_states[family].active_now = false;
                pimpl->functional_states[family].current_gain = 0.0f;
                pimpl->functional_states[family].current_hold_remaining = 0;
                pimpl->functional_trace.family_state[family] = pimpl->functional_states[family];
            }
        }
    }
    return true;
}

bool llama_active_lora_manager::functional_apply_update(
        int32_t family,
        int32_t loop_origin,
        int32_t start_microphase,
        int32_t settle_microphase,
        const llama_functional_outcome_snapshot & before,
        const llama_functional_outcome_snapshot & after,
        int32_t selected_tool_kind,
        int32_t candidate_count,
        const float * metrics,
        size_t metric_count,
        float signed_outcome,
        float magnitude,
        const llama_self_state_event & event,
        const llama_self_state_feature_vector * features) {
    if (!pimpl->initialized ||
            family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
            !event.tokens || event.n_tokens == 0 ||
            pimpl->functional_ablation.disable_update_writes ||
            (pimpl->functional_ablation.disabled_family_mask & (1ull << family)) != 0 ||
            !pimpl->functional_adapters[family]) {
        return false;
    }

    const active_lora_embedding embedding = pimpl->embedder->embed(event.tokens, event.n_tokens);
    if (embedding.values.empty()) {
        return false;
    }

    active_lora_write_features write_features = pimpl->build_write_features(embedding, event, features, false);
    write_features = pimpl->augment_functional_write_features(
            write_features,
            family,
            loop_origin,
            metrics,
            metric_count,
            signed_outcome,
            magnitude);

    const float direction_scale =
            std::max(0.05f, std::min(1.0f, magnitude)) *
            (signed_outcome >= 0.0f ?
                pimpl->functional_configs[family].positive_update_scale :
                pimpl->functional_configs[family].negative_update_scale);
    runtime_lora_write_result write_result = {};
    if (!pimpl->train_on_adapter(
                *pimpl->functional_adapters[family],
                embedding,
                write_features,
                signed_outcome >= 0.0f ? direction_scale : -direction_scale,
                false,
                &write_result)) {
        return false;
    }

    auto & update = pimpl->functional_updates[family];
    update = {};
    update.valid = true;
    update.family = family;
    update.loop_origin = loop_origin;
    update.start_microphase = start_microphase;
    update.settle_microphase = settle_microphase;
    update.selected_tool_kind = selected_tool_kind;
    update.candidate_count = candidate_count;
    update.signed_outcome = signed_outcome;
    update.magnitude = magnitude;
    update.before_snapshot = before;
    update.after_snapshot = after;
    for (size_t i = 0; i < std::min(metric_count, sizeof(update.metrics)/sizeof(update.metrics[0])); ++i) {
        update.metrics[i] = metrics[i];
    }
    (void) pimpl->apply_meta_update(family, before, after, &update);
    update.adapter_update_norm = write_result.update_norm;
    update.adapter_optimizer_step = write_result.transaction_step;

    pimpl->functional_states[family].update_count += 1;
    pimpl->functional_states[family].last_signed_outcome = signed_outcome;
    pimpl->functional_states[family].last_meta_loss = update.meta_loss;
    pimpl->functional_trace.family_state[family] = pimpl->functional_states[family];
    return true;
}
