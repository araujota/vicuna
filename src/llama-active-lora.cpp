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
#include <unordered_map>
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
constexpr size_t FUNCTIONAL_GATING_INPUT_DIM = 50;
constexpr size_t FUNCTIONAL_GATING_HIDDEN_DIM = 16;
constexpr uint64_t FUNCTIONAL_GATING_INIT_SEED = 0x6d6574616c6f7373ULL;
constexpr uint64_t FUNCTIONAL_BOOTSTRAP_INIT_SEED = 0x626f6f7473747270ULL;

float clamp_unit(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

float clamp_range(float value, float lo, float hi) {
    return std::min(hi, std::max(lo, value));
}

float clamp_signed_unit(float value) {
    return clamp_range(value, -1.0f, 1.0f);
}

uint64_t hash_event_tokens(const llama_self_state_event & event) {
    if (!event.tokens || event.n_tokens == 0) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < event.n_tokens; ++i) {
        hash ^= (uint64_t) (uint32_t) event.tokens[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

float decayed_std(float initial_std, float min_std, uint64_t count, uint32_t decay_horizon) {
    const float init = std::max(0.0f, initial_std);
    const float floor = std::max(0.0f, std::min(init, min_std));
    const float decay = std::max<uint32_t>(1u, decay_horizon);
    const float scaled = std::sqrt(1.0f + static_cast<float>(count) / static_cast<float>(decay));
    return floor + (init - floor) / scaled;
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
        case LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION:
            return 4;
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL:
            return 5;
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
        case LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION:
            return 2;
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
    return index == 5 ? 1.0f : 0.0f; // answerability
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
    return decayed_std(
            config.exploration_noise_initial_std,
            config.exploration_noise_min_std,
            invocation_count,
            config.exploration_noise_decay_invocations);
}

float bootstrap_noise_std(const llama_functional_lora_family_config & config, uint64_t activation_count) {
    return decayed_std(
            config.bootstrap_perturbation_initial_std,
            config.bootstrap_perturbation_min_std,
            activation_count,
            config.bootstrap_perturbation_decay_activations);
}

float sample_bounded_gaussian(std::mt19937_64 & rng, float stddev) {
    if (stddev <= 0.0f) {
        return 0.0f;
    }

    std::normal_distribution<float> dist(0.0f, stddev);
    return clamp_range(dist(rng), -3.0f * stddev, 3.0f * stddev);
}

constexpr uint64_t FUNCTIONAL_SNAPSHOT_PERIOD_US = 7ull * 24ull * 60ull * 60ull * 1000000ull;
constexpr uint64_t FUNCTIONAL_SNAPSHOT_RETENTION_US = 31ull * 24ull * 60ull * 60ull * 1000000ull;
constexpr size_t FUNCTIONAL_DIRECTION_SKETCH_DIMS = 32;

struct functional_snapshot_runtime {
    llama_functional_lora_snapshot_info info = {};
    std::unique_ptr<llama_adapter_lora> adapter;
};

struct process_functional_ledger_runtime {
    llama_process_functional_ledger_info info = {};
    uint32_t last_creation_attempt_observation = 0;
};

struct process_functional_entry_runtime {
    llama_process_functional_entry_info info = {};
    std::unique_ptr<llama_adapter_lora> adapter;
    std::unique_ptr<llama_adapter_lora> bootstrap_adapter;
    std::unique_ptr<llama_adapter_lora> replay_adapter;
    std::array<functional_snapshot_runtime, LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY> snapshots = {};
    llama_functional_lora_snapshot_archive snapshot_archive = {};
    llama_functional_lora_replay_override replay_override = {};
    llama_functional_lora_differential_update differential_update = {};
    std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> dominant_direction = {};
    std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> last_signature = {};
    bool signature_valid = false;
};

float vector_cosine_similarity(const std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> & a,
                               const std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> & b) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < FUNCTIONAL_DIRECTION_SKETCH_DIMS; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a <= DIRECTION_EPS || norm_b <= DIRECTION_EPS) {
        return 0.0f;
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void normalize_array(std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> & values) {
    float norm = 0.0f;
    for (float value : values) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm <= DIRECTION_EPS) {
        return;
    }
    for (float & value : values) {
        value /= norm;
    }
}

std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> project_adapter_signature(const llama_adapter_lora & adapter) {
    std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> signature = {};
    size_t cursor = 0;
    for (const auto & it : adapter.ab_map) {
        const auto & weight = it.second;
        const size_t size_a = weight.a->ne[0] * weight.a->ne[1];
        const size_t size_b = weight.b->ne[0] * weight.b->ne[1];
        std::vector<float> data_a(size_a, 0.0f);
        std::vector<float> data_b(size_b, 0.0f);
        ggml_backend_tensor_get(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
        ggml_backend_tensor_get(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
        for (float value : data_a) {
            signature[cursor % FUNCTIONAL_DIRECTION_SKETCH_DIMS] += value;
            cursor += 1;
        }
        for (float value : data_b) {
            signature[cursor % FUNCTIONAL_DIRECTION_SKETCH_DIMS] += value;
            cursor += 1;
        }
        signature[cursor % FUNCTIONAL_DIRECTION_SKETCH_DIMS] += weight.gain;
        cursor += 1;
    }
    normalize_array(signature);
    return signature;
}

void blend_toward_signature(std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> & dominant,
                            const std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> & update,
                            float mix) {
    const float clamped_mix = clamp_unit(mix);
    for (size_t i = 0; i < FUNCTIONAL_DIRECTION_SKETCH_DIMS; ++i) {
        dominant[i] = (1.0f - clamped_mix) * dominant[i] + clamped_mix * update[i];
    }
    normalize_array(dominant);
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

void randomize_adapter(llama_adapter_lora & adapter, std::mt19937_64 & rng, float weight_std) {
    if (weight_std <= 0.0f) {
        zero_adapter(adapter);
        return;
    }

    std::normal_distribution<float> dist(0.0f, weight_std);
    for (auto & it : adapter.ab_map) {
        auto & weight = it.second;
        std::vector<float> data_a(weight.a->ne[0] * weight.a->ne[1], 0.0f);
        std::vector<float> data_b(weight.b->ne[0] * weight.b->ne[1], 0.0f);
        for (float & value : data_a) {
            value = dist(rng);
        }
        for (float & value : data_b) {
            value = dist(rng);
        }
        ggml_backend_tensor_set(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
        ggml_backend_tensor_set(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
        weight.gain = 1.0f;
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
        case LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION:
            return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_PLANNING_COMPOSITION;
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL:    return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_COUNTERFACTUAL;
        case LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION:return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_MEMORY_COMPRESSION;
        case LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION: return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_SELF_OBSERVATION;
        default:                                      return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION;
    }
}

const char * functional_family_name(int32_t family) {
    switch (family) {
        case LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION:     return "tool_selection";
        case LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION:return "planning_composition";
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL:     return "counterfactual";
        case LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION: return "memory_compression";
        case LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION:   return "self_observation";
        default:                                       return "unknown";
    }
}

llama_adapter_lora_layer_role process_functional_learned_role() {
    return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_PROCESS_LEARNED;
}

llama_adapter_lora_layer_role process_functional_bootstrap_role() {
    return LLAMA_ADAPTER_LORA_LAYER_FUNCTIONAL_PROCESS_BOOTSTRAP;
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
    llama_user_personality_lora_stats user_personality_stats = {};
    llama_past_lora_params past_params = llama_past_lora_default_params();
    llama_past_lora_stats past_stats = {};
    std::unique_ptr<llama_adapter_lora> adapter;
    std::unique_ptr<llama_adapter_lora> user_personality_adapter;
    std::unique_ptr<active_lora_embedder> embedder;
    std::array<bucket_runtime, PAST_BUCKET_COUNT> buckets = {};
    std::array<job_runtime, PAST_BUCKET_COUNT> jobs = {};
    std::array<std::unique_ptr<llama_adapter_lora>, LLAMA_FUNCTIONAL_LORA_COUNT> functional_adapters = {};
    std::array<std::unique_ptr<llama_adapter_lora>, LLAMA_FUNCTIONAL_LORA_COUNT> functional_bootstrap_adapters = {};
    std::array<std::unique_ptr<llama_adapter_lora>, LLAMA_FUNCTIONAL_LORA_COUNT> functional_replay_adapters = {};
    std::array<llama_functional_lora_family_config, LLAMA_FUNCTIONAL_LORA_COUNT> functional_configs = {};
    std::array<llama_functional_lora_family_state, LLAMA_FUNCTIONAL_LORA_COUNT> functional_states = {};
    std::array<llama_functional_lora_update_info, LLAMA_FUNCTIONAL_LORA_COUNT> functional_updates = {};
    std::array<llama_functional_lora_differential_update, LLAMA_FUNCTIONAL_LORA_COUNT> functional_differential_updates = {};
    std::array<llama_functional_lora_snapshot_archive, LLAMA_FUNCTIONAL_LORA_COUNT> functional_snapshot_archives = {};
    std::array<std::array<functional_snapshot_runtime, LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY>, LLAMA_FUNCTIONAL_LORA_COUNT> functional_snapshots = {};
    std::array<llama_functional_lora_replay_override, LLAMA_FUNCTIONAL_LORA_COUNT> functional_replay_overrides = {};
    std::array<std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS>, LLAMA_FUNCTIONAL_LORA_COUNT> functional_dominant_directions = {};
    std::array<std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS>, LLAMA_FUNCTIONAL_LORA_COUNT> functional_last_signatures = {};
    std::array<bool, LLAMA_FUNCTIONAL_LORA_COUNT> functional_signature_valid = {};
    llama_functional_snapshot_maintenance_trace functional_snapshot_trace = {};
    llama_functional_lora_trace functional_trace = {};
    llama_functional_lora_ablation_config functional_ablation = {};
    llama_process_functional_params process_params = llama_process_functional_default_params();
    std::array<process_functional_ledger_runtime, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES> process_ledgers = {};
    std::array<process_functional_entry_runtime, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES> process_entries = {};
    llama_process_functional_signature current_process_signature = {};
    llama_process_functional_trace last_process_trace = {};
    llama_functional_snapshot_maintenance_trace process_snapshot_trace = {};
    llama_self_model_extension_summary last_extension_summary = {};
    int32_t current_process_entry_slot = -1;
    std::vector<ggml_tensor *> targets;
    active_lora_embedding last_embedding;
    active_lora_embedding last_user_personality_embedding;
    uint64_t updates_seen = 0;
    uint64_t active_started_at_us = 0;
    uint64_t active_last_update_us = 0;
    uint32_t active_rollover_version = 0;
    llama_active_temporal_encoding_bias temporal_encoding_bias = {};
    temporal_bias_adam_state temporal_bias_adam = {};
    functional_gating_network gating_network = {};
    functional_gating_adam_state gating_adam = {};
    std::array<functional_gating_training_tuple, LLAMA_FUNCTIONAL_LORA_COUNT> gating_training = {};
    std::unordered_map<const ggml_tensor *, runtime_lora_adam_state> runtime_lora_adam;
    std::unordered_map<const llama_adapter_lora *, uint64_t> runtime_lora_write_count;
    std::mt19937_64 gating_rng { FUNCTIONAL_GATING_INIT_SEED };
    std::mt19937_64 bootstrap_rng { FUNCTIONAL_BOOTSTRAP_INIT_SEED };
    uint64_t gating_invocation_count = 0;
    uint64_t next_functional_snapshot_id = 1;
    bool initialized = false;
    bool past_initialized = false;

    void initialize_process_defaults() {
        process_params = llama_process_functional_default_params();
        current_process_signature = {};
        last_process_trace = {};
        current_process_entry_slot = -1;
        for (auto & ledger : process_ledgers) {
            ledger = {};
        }
        for (auto & entry : process_entries) {
            entry = {};
        }
        process_snapshot_trace = {};
    }

    static bool process_signature_matches(
            const llama_process_functional_signature & lhs,
            const llama_process_functional_signature & rhs) {
        return lhs.valid && rhs.valid &&
                lhs.signature_hash == rhs.signature_hash &&
                lhs.family == rhs.family;
    }

    int32_t process_entry_count() const {
        int32_t count = 0;
        for (const auto & entry : process_entries) {
            count += entry.info.valid ? 1 : 0;
        }
        return count;
    }

    int32_t process_ledger_count() const {
        int32_t count = 0;
        for (const auto & ledger : process_ledgers) {
            count += ledger.info.valid ? 1 : 0;
        }
        return count;
    }

    int32_t find_process_entry_slot(const llama_process_functional_signature & signature) const {
        if (!signature.valid) {
            return -1;
        }
        for (int32_t i = 0; i < LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES; ++i) {
            if (process_entries[i].info.valid &&
                    process_signature_matches(process_entries[i].info.signature, signature)) {
                return i;
            }
        }
        return -1;
    }

    int32_t find_process_ledger_slot(const llama_process_functional_signature & signature) const {
        if (!signature.valid) {
            return -1;
        }
        for (int32_t i = 0; i < LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES; ++i) {
            if (process_ledgers[i].info.valid &&
                    process_signature_matches(process_ledgers[i].info.signature, signature)) {
                return i;
            }
        }
        return -1;
    }

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
            cfg.bootstrap_perturbation_initial_std = 0.05f;
            cfg.bootstrap_perturbation_min_std = 0.005f;
            cfg.bootstrap_perturbation_decay_activations = 256;
            cfg.bootstrap_weight_init_std = 0.01f;
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
            state.current_bootstrap_std = bootstrap_noise_std(cfg, 0);

            functional_trace.holds[family] = {};
            functional_trace.holds[family].family = family;
            functional_trace.family_state[family] = state;
            functional_updates[family] = {};
            functional_updates[family].family = family;
            functional_differential_updates[family] = {};
            gating_training[family] = {};
            functional_snapshot_archives[family] = {};
            functional_snapshot_archives[family].family = family;
            functional_snapshot_archives[family].next_capture_due_us = FUNCTIONAL_SNAPSHOT_PERIOD_US;
            functional_replay_overrides[family] = {};
            functional_signature_valid[family] = false;
            for (size_t slot = 0; slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++slot) {
                functional_snapshots[family][slot] = {};
                functional_snapshots[family][slot].info.family = family;
                functional_snapshots[family][slot].info.slot = static_cast<int32_t>(slot);
            }
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

    static std::array<float, FUNCTIONAL_GATING_INPUT_DIM> build_gating_input(
            const llama_functional_gating_observation & observation,
            float * out_gradient_norm = nullptr) {
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
        input[17] = clamp_unit(observation.planning_pressure);
        input[18] = clamp_unit(observation.plan_complexity);
        input[19] = clamp_unit(observation.plan_revision);
        input[20] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION)) != 0 ? 1.0f : 0.0f;
        input[21] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION)) != 0 ? 1.0f : 0.0f;
        input[22] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL)) != 0 ? 1.0f : 0.0f;
        input[23] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION)) != 0 ? 1.0f : 0.0f;
        input[24] = (observation.eligible_mask & (1ull << LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION)) != 0 ? 1.0f : 0.0f;
        input[25] = clamp_signed_unit(observation.extension_summary.gain_signal);
        input[26] = clamp_unit(observation.extension_summary.gain_signal_abs);
        input[27] = clamp_unit(observation.extension_summary.context_activation);
        input[28] = clamp_unit(observation.extension_summary.allostatic_divergence);
        input[29] = clamp_unit(observation.extension_summary.mean_confidence);
        input[30] = clamp_unit(observation.extension_summary.max_salience);
        input[31] = clamp_unit(observation.hard_memory_summary.mean_similarity);
        input[32] = clamp_unit(observation.hard_memory_summary.max_similarity);
        input[33] = clamp_unit(observation.hard_memory_summary.importance_signal);
        input[34] = clamp_unit(observation.hard_memory_summary.confidence_signal);
        input[35] = clamp_unit(observation.hard_memory_summary.gain_support);
        input[36] = clamp_unit(observation.hard_memory_summary.goal_support);
        input[37] = clamp_unit(observation.hard_memory_summary.user_support);
        input[38] = clamp_unit(observation.hard_memory_summary.epistemic_support);
        input[39] = clamp_unit(observation.hard_memory_summary.efficiency_support);
        input[40] = clamp_unit(observation.hard_memory_summary.recovery_support);
        input[41] = observation.belief_summary.valid ? 1.0f : 0.0f;
        input[42] = clamp_unit(observation.belief_summary.known_care_uncertainty);
        input[43] = clamp_unit(observation.belief_summary.missing_observation_uncertainty);
        input[44] = clamp_unit(observation.belief_summary.unmodeled_care_uncertainty);
        input[45] = clamp_unit(observation.belief_summary.residual_allostatic_pressure);
        input[46] = clamp_unit(observation.belief_summary.promotion_readiness);
        input[47] = clamp_unit(observation.belief_summary.belief_entropy);
        input[48] = clamp_unit(observation.belief_summary.belief_confidence);
        input[49] = clamp_unit(observation.belief_summary.max_slot_pressure);

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
        last_extension_summary = observation.extension_summary;
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
            decision.bootstrap_std[family] = 0.0f;
            decision.bootstrap_perturbation[family] = 0.0f;
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
            llama_adapter_lora_layer_role role,
            bool attach_runtime = true) {
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
        if (attach_runtime) {
            owner.attach_adapter_runtime(out.get(), initial_scale, role);
        }
        return true;
    }

    static bool copy_adapter(llama_adapter_lora & dst, const llama_adapter_lora & src) {
        if (dst.ab_map.size() != src.ab_map.size()) {
            return false;
        }
        for (const auto & it : src.ab_map) {
            auto dst_it = dst.ab_map.find(it.first);
            if (dst_it == dst.ab_map.end()) {
                return false;
            }
            const auto & src_weight = it.second;
            auto & dst_weight = dst_it->second;
            const size_t size_a = src_weight.a->ne[0] * src_weight.a->ne[1];
            const size_t size_b = src_weight.b->ne[0] * src_weight.b->ne[1];
            std::vector<float> data_a(size_a, 0.0f);
            std::vector<float> data_b(size_b, 0.0f);
            ggml_backend_tensor_get(src_weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_get(src_weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
            ggml_backend_tensor_set(dst_weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_set(dst_weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
            dst_weight.gain = src_weight.gain;
        }
        return true;
    }

    void clear_process_runtime_attachments() {
        for (auto & entry : process_entries) {
            if (entry.adapter) {
                owner.detach_adapter_runtime(entry.adapter.get());
            }
            if (entry.bootstrap_adapter) {
                owner.detach_adapter_runtime(entry.bootstrap_adapter.get());
            }
            if (entry.replay_adapter) {
                owner.detach_adapter_runtime(entry.replay_adapter.get());
            }
            for (auto & snapshot : entry.snapshots) {
                if (snapshot.adapter) {
                    owner.detach_adapter_runtime(snapshot.adapter.get());
                }
            }
        }
    }

    bool ensure_process_entry_adapters(int32_t slot, int32_t family) {
        if (slot < 0 || slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
                family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
            return false;
        }
        auto & entry = process_entries[slot];
        const uint32_t rank = stats.selected_rank > 0 ? stats.selected_rank : params.min_rank;
        if (!entry.adapter) {
            if (!create_runtime_adapter(
                        entry.adapter,
                        std::max<uint32_t>(1, rank),
                        std::string("functional.process.") + std::to_string(slot) + "." + functional_family_name(family),
                        0.0f,
                        process_functional_learned_role(),
                        false)) {
                return false;
            }
        }
        if (!entry.bootstrap_adapter) {
            if (!create_runtime_adapter(
                        entry.bootstrap_adapter,
                        std::max<uint32_t>(1, rank),
                        std::string("functional.process.bootstrap.") + std::to_string(slot) + "." + functional_family_name(family),
                        0.0f,
                        process_functional_bootstrap_role(),
                        false)) {
                return false;
            }
            randomize_adapter(
                    *entry.bootstrap_adapter,
                    bootstrap_rng,
                    functional_configs[family].bootstrap_weight_init_std);
        }
        return true;
    }

    int32_t first_free_process_entry_slot() const {
        for (int32_t i = 0; i < std::min<int32_t>((int32_t) process_params.max_entries, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES); ++i) {
            if (!process_entries[i].info.valid) {
                return i;
            }
        }
        return -1;
    }

    int32_t first_free_process_ledger_slot() const {
        for (int32_t i = 0; i < LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES; ++i) {
            if (!process_ledgers[i].info.valid) {
                return i;
            }
        }
        return -1;
    }

    static float process_entry_eviction_score(const process_functional_entry_runtime & entry, uint64_t now_us) {
        if (!entry.info.valid) {
            return -1.0e9f;
        }
        const float staleness_penalty =
                entry.info.last_used_us > 0 && now_us > entry.info.last_used_us ?
                        std::min(1.0f, (float) (now_us - entry.info.last_used_us) / (7.0f * 24.0f * 60.0f * 60.0f * 1000000.0f)) :
                        0.0f;
        return entry.info.utility_score +
                0.02f * std::min<uint64_t>(32, entry.info.update_count) -
                0.10f * staleness_penalty;
    }

    int32_t select_process_entry_eviction_slot(uint64_t now_us) const {
        int32_t slot = -1;
        float best_score = std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < std::min<int32_t>((int32_t) process_params.max_entries, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES); ++i) {
            const auto & entry = process_entries[i];
            if (!entry.info.valid) {
                continue;
            }
            if (current_process_entry_slot == i) {
                continue;
            }
            const float score = process_entry_eviction_score(entry, now_us);
            if (score < best_score) {
                best_score = score;
                slot = i;
            }
        }
        return slot;
    }

    bool create_process_entry(
            const llama_process_functional_signature & signature,
            int32_t ledger_slot,
            int32_t * out_slot,
            int32_t * out_reason) {
        const uint64_t now_us = llama_time_us();
        if (out_slot) {
            *out_slot = -1;
        }
        if (out_reason) {
            *out_reason = LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_INVALID_SIGNATURE;
        }
        if (!signature.valid ||
                signature.family < 0 ||
                signature.family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
            return false;
        }

        int32_t slot = first_free_process_entry_slot();
        int32_t reason = LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_CREATED;
        if (slot < 0) {
            slot = select_process_entry_eviction_slot(now_us);
            if (slot < 0) {
                reason = LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_CAPACITY_DENIED;
                if (out_reason) {
                    *out_reason = reason;
                }
                return false;
            }
            if (process_entries[slot].adapter) {
                owner.detach_adapter_runtime(process_entries[slot].adapter.get());
                const_cast<llama_model &>(owner.model).loras.erase(process_entries[slot].adapter.get());
            }
            if (process_entries[slot].bootstrap_adapter) {
                owner.detach_adapter_runtime(process_entries[slot].bootstrap_adapter.get());
                const_cast<llama_model &>(owner.model).loras.erase(process_entries[slot].bootstrap_adapter.get());
            }
            if (process_entries[slot].replay_adapter) {
                owner.detach_adapter_runtime(process_entries[slot].replay_adapter.get());
                const_cast<llama_model &>(owner.model).loras.erase(process_entries[slot].replay_adapter.get());
            }
            for (auto & snapshot : process_entries[slot].snapshots) {
                if (snapshot.adapter) {
                    owner.detach_adapter_runtime(snapshot.adapter.get());
                    const_cast<llama_model &>(owner.model).loras.erase(snapshot.adapter.get());
                }
            }
            process_entries[slot] = {};
            reason = LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_EVICTED_REPLACEMENT;
        }

        if (!ensure_process_entry_adapters(slot, signature.family)) {
            process_entries[slot] = {};
            return false;
        }

        auto & entry = process_entries[slot];
        entry.info = {};
        entry.info.valid = true;
        entry.info.slot = slot;
        entry.info.signature = signature;
        entry.info.created_at_us = now_us;
        entry.info.last_used_us = now_us;
        entry.info.utility_score = process_ledgers[ledger_slot].info.weak_or_worse_ratio;
        entry.info.current_gain = functional_configs[signature.family].default_gain;
        entry.info.current_bootstrap_std = bootstrap_noise_std(functional_configs[signature.family], 0);
        entry.info.last_bootstrap_perturbation = 0.0f;
        entry.snapshot_archive = {};
        entry.snapshot_archive.family = signature.family;
        entry.snapshot_archive.next_capture_due_us = FUNCTIONAL_SNAPSHOT_PERIOD_US;
        entry.replay_override = {};
        entry.differential_update = {};
        entry.signature_valid = false;
        for (size_t snapshot_slot = 0; snapshot_slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++snapshot_slot) {
            entry.snapshots[snapshot_slot] = {};
            entry.snapshots[snapshot_slot].info.family = signature.family;
            entry.snapshots[snapshot_slot].info.slot = static_cast<int32_t>(snapshot_slot);
        }
        zero_adapter(*entry.adapter);
        zero_adapter(*entry.bootstrap_adapter);
        randomize_adapter(
                *entry.bootstrap_adapter,
                bootstrap_rng,
                functional_configs[signature.family].bootstrap_weight_init_std);
        if (out_slot) {
            *out_slot = slot;
        }
        if (out_reason) {
            *out_reason = reason;
        }
        return true;
    }

    bool process_qualifies_for_creation(const process_functional_ledger_runtime & ledger) const {
        if (!process_params.enabled || !ledger.info.valid) {
            return false;
        }
        if (ledger.info.observation_count < process_params.min_observations) {
            return false;
        }
        if (ledger.info.weak_or_worse_ratio < process_params.weak_or_worse_ratio_threshold) {
            return false;
        }
        return ledger.info.mean_signed_outcome <= process_params.mean_outcome_ceiling;
    }

    static float adapter_difference_norm(const llama_adapter_lora & lhs, const llama_adapter_lora & rhs) {
        double sum_sq = 0.0;
        for (const auto & it : lhs.ab_map) {
            auto rhs_it = rhs.ab_map.find(it.first);
            if (rhs_it == rhs.ab_map.end()) {
                continue;
            }
            const auto & lhs_weight = it.second;
            const auto & rhs_weight = rhs_it->second;
            const size_t size_a = lhs_weight.a->ne[0] * lhs_weight.a->ne[1];
            const size_t size_b = lhs_weight.b->ne[0] * lhs_weight.b->ne[1];
            std::vector<float> lhs_a(size_a, 0.0f);
            std::vector<float> rhs_a(size_a, 0.0f);
            std::vector<float> lhs_b(size_b, 0.0f);
            std::vector<float> rhs_b(size_b, 0.0f);
            ggml_backend_tensor_get(lhs_weight.a, lhs_a.data(), 0, lhs_a.size() * sizeof(float));
            ggml_backend_tensor_get(rhs_weight.a, rhs_a.data(), 0, rhs_a.size() * sizeof(float));
            ggml_backend_tensor_get(lhs_weight.b, lhs_b.data(), 0, lhs_b.size() * sizeof(float));
            ggml_backend_tensor_get(rhs_weight.b, rhs_b.data(), 0, rhs_b.size() * sizeof(float));
            for (size_t i = 0; i < size_a; ++i) {
                const double delta = (double) lhs_a[i] - (double) rhs_a[i];
                sum_sq += delta * delta;
            }
            for (size_t i = 0; i < size_b; ++i) {
                const double delta = (double) lhs_b[i] - (double) rhs_b[i];
                sum_sq += delta * delta;
            }
            const double gain_delta = (double) lhs_weight.gain - (double) rhs_weight.gain;
            sum_sq += gain_delta * gain_delta;
        }
        return std::sqrt((float) sum_sq);
    }

    static void record_adapter_signature(
            llama_adapter_lora * live_adapter,
            std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> & dominant_direction,
            std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> & last_signature,
            bool & signature_valid) {
        if (!live_adapter) {
            return;
        }
        const std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> current =
                project_adapter_signature(*live_adapter);
        if (signature_valid) {
            std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> delta = {};
            for (size_t i = 0; i < FUNCTIONAL_DIRECTION_SKETCH_DIMS; ++i) {
                delta[i] = current[i] - last_signature[i];
            }
            normalize_array(delta);
            if (fro_norm(std::vector<float>(delta.begin(), delta.end())) > DIRECTION_EPS) {
                blend_toward_signature(dominant_direction, delta, 0.35f);
            }
        } else {
            dominant_direction = current;
            signature_valid = true;
        }
        last_signature = current;
    }

    void record_functional_signature(int32_t family) {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
            return;
        }
        record_adapter_signature(
                functional_adapters[family].get(),
                functional_dominant_directions[family],
                functional_last_signatures[family],
                functional_signature_valid[family]);
    }

    void record_process_signature(int32_t entry_slot) {
        if (entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
            return;
        }
        auto & entry = process_entries[entry_slot];
        record_adapter_signature(
                entry.adapter.get(),
                entry.dominant_direction,
                entry.last_signature,
                entry.signature_valid);
    }

    bool ensure_snapshot_adapter_runtime(
            std::array<functional_snapshot_runtime, LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY> & snapshots,
            int32_t family,
            int32_t slot,
            const std::string & name_prefix,
            llama_adapter_lora_layer_role role) {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT || slot < 0 || slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
            return false;
        }
        auto & runtime = snapshots[slot];
        if (runtime.adapter) {
            return true;
        }
        const uint32_t rank = stats.selected_rank > 0 ? stats.selected_rank : params.min_rank;
        return create_runtime_adapter(
                runtime.adapter,
                std::max<uint32_t>(1, rank),
                name_prefix,
                0.0f,
                role,
                false);
    }

    bool ensure_snapshot_adapter(int32_t family, int32_t slot) {
        return ensure_snapshot_adapter_runtime(
                functional_snapshots[family],
                family,
                slot,
                std::string("functional.snapshot.") + functional_family_name(family) + "." + std::to_string(slot),
                functional_family_role(family));
    }

    bool ensure_process_snapshot_adapter(int32_t entry_slot, int32_t slot) {
        if (entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
                !process_entries[entry_slot].info.valid) {
            return false;
        }
        const int32_t family = process_entries[entry_slot].info.signature.family;
        return ensure_snapshot_adapter_runtime(
                process_entries[entry_slot].snapshots,
                family,
                slot,
                std::string("functional.process.snapshot.") + std::to_string(entry_slot) + "." + std::to_string(slot) + "." + functional_family_name(family),
                process_functional_learned_role());
    }

    bool ensure_replay_adapter_runtime(
            std::unique_ptr<llama_adapter_lora> & replay_adapter,
            int32_t family,
            const std::string & name_prefix,
            llama_adapter_lora_layer_role role) {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
            return false;
        }
        if (replay_adapter) {
            return true;
        }
        const uint32_t rank = stats.selected_rank > 0 ? stats.selected_rank : params.min_rank;
        return create_runtime_adapter(
                replay_adapter,
                std::max<uint32_t>(1, rank),
                name_prefix,
                0.0f,
                role,
                false);
    }

    bool ensure_replay_adapter(int32_t family) {
        return ensure_replay_adapter_runtime(
                functional_replay_adapters[family],
                family,
                std::string("functional.replay.") + functional_family_name(family),
                functional_family_role(family));
    }

    bool ensure_process_replay_adapter(int32_t entry_slot) {
        if (entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
                !process_entries[entry_slot].info.valid) {
            return false;
        }
        const int32_t family = process_entries[entry_slot].info.signature.family;
        return ensure_replay_adapter_runtime(
                process_entries[entry_slot].replay_adapter,
                family,
                std::string("functional.process.replay.") + std::to_string(entry_slot) + "." + functional_family_name(family),
                process_functional_learned_role());
    }

    void clear_family_runtime_attachments(int32_t family) {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
            return;
        }
        if (functional_adapters[family]) {
            owner.detach_adapter_runtime(functional_adapters[family].get());
        }
        if (functional_bootstrap_adapters[family]) {
            owner.detach_adapter_runtime(functional_bootstrap_adapters[family].get());
        }
        if (functional_replay_adapters[family]) {
            owner.detach_adapter_runtime(functional_replay_adapters[family].get());
        }
        for (size_t slot = 0; slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++slot) {
            if (functional_snapshots[family][slot].adapter) {
                owner.detach_adapter_runtime(functional_snapshots[family][slot].adapter.get());
            }
        }
    }

    bool process_uses_external_tool_context() const {
        return current_process_signature.valid &&
                (current_process_signature.tool_kind != LLAMA_TOOL_KIND_NONE ||
                 current_process_signature.capability_id[0] != '\0' ||
                 current_process_signature.provenance_namespace[0] != '\0');
    }

    float self_model_runtime_gain_scale(int32_t family) const {
        const auto & summary = last_extension_summary;
        if (summary.active_count <= 0 || summary.gain_count <= 0) {
            return 1.0f;
        }

        const bool tool_context =
                family == LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ||
                process_uses_external_tool_context();
        const float activation = clamp_unit(std::max(summary.context_activation, summary.gain_signal_abs));
        const float signed_gain = clamp_signed_unit(summary.gain_signal);
        const float direction = tool_context ? signed_gain : 0.50f * signed_gain;
        const float strength = tool_context ?
                (0.35f + 0.45f * activation) :
                (0.20f + 0.25f * activation);
        return clamp_range(1.0f + direction * strength, 0.25f, 1.75f);
    }

    float apply_self_model_runtime_gain(int32_t family, float gain) const {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT || gain <= 0.0f) {
            return 0.0f;
        }

        const float scaled_gain = gain * self_model_runtime_gain_scale(family);
        return clamp_range(
                scaled_gain,
                functional_configs[family].gain_clip_min,
                functional_configs[family].gain_clip_max);
    }

    void apply_functional_runtime_scale(int32_t family, float gain, float bootstrap_perturbation) {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
            return;
        }

        const auto & replay = functional_replay_overrides[family];
        llama_adapter_lora * active_adapter = functional_adapters[family].get();
        if (replay.active) {
            if (replay.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED &&
                    replay.snapshot_slot >= 0 &&
                    replay.snapshot_slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
                active_adapter = functional_snapshots[family][replay.snapshot_slot].adapter.get();
            } else if (replay.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ||
                       replay.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED) {
                active_adapter = functional_replay_adapters[family].get();
            }
        }

        clear_family_runtime_attachments(family);

        if (active_adapter) {
            owner.attach_adapter_runtime(active_adapter, gain, functional_family_role(family));
        }
        if (!replay.active || !replay.disable_bootstrap) {
            if (functional_bootstrap_adapters[family]) {
                owner.attach_adapter_runtime(
                        functional_bootstrap_adapters[family].get(),
                        replay.active && replay.disable_bootstrap ? 0.0f : bootstrap_perturbation,
                        functional_family_role(family));
            }
        }
    }

    bool capture_functional_snapshot(int32_t family, uint64_t now_us) {
        if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
                !functional_adapters[family] ||
                !functional_configs[family].enabled) {
            return false;
        }

        int32_t slot = -1;
        for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
            if (!functional_snapshots[family][i].info.valid) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            uint64_t oldest = std::numeric_limits<uint64_t>::max();
            for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
                if (functional_snapshots[family][i].info.captured_at_us < oldest) {
                    oldest = functional_snapshots[family][i].info.captured_at_us;
                    slot = i;
                }
            }
        }
        if (slot < 0 || !ensure_snapshot_adapter(family, slot) ||
                !copy_adapter(*functional_snapshots[family][slot].adapter, *functional_adapters[family])) {
            return false;
        }

        auto & archive = functional_snapshot_archives[family];
        auto & runtime = functional_snapshots[family][slot];
        runtime.info = {};
        runtime.info.valid = true;
        runtime.info.family = family;
        runtime.info.slot = slot;
        runtime.info.source = LLAMA_FUNCTIONAL_SNAPSHOT_SOURCE_WEEKLY_ARCHIVE;
        runtime.info.snapshot_id = next_functional_snapshot_id++;
        runtime.info.captured_at_us = now_us;
        runtime.info.expires_at_us = now_us + FUNCTIONAL_SNAPSHOT_RETENTION_US;
        runtime.info.source_update_count = functional_states[family].update_count;
        runtime.info.self_state_gradient_norm = functional_trace.last_activation.allostatic_gradient_norm;
        runtime.info.robustness_score = clamp_unit(0.5f + 0.5f * functional_states[family].last_signed_outcome);
        runtime.info.last_signed_outcome = functional_states[family].last_signed_outcome;
        runtime.info.dominant_direction_cosine =
                vector_cosine_similarity(functional_dominant_directions[family], functional_last_signatures[family]);
        archive.last_capture_us = now_us;
        archive.next_capture_due_us = now_us + FUNCTIONAL_SNAPSHOT_PERIOD_US;
        archive.items[slot] = runtime.info;

        uint32_t count = 0;
        for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
            if (functional_snapshots[family][i].info.valid) {
                count += 1;
            }
            archive.items[i] = functional_snapshots[family][i].info;
        }
        archive.count = count;
        return true;
    }

    void expire_functional_snapshots(uint64_t now_us, uint32_t * out_expired_count = nullptr) {
        for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
            auto & archive = functional_snapshot_archives[family];
            uint32_t count = 0;
            for (int32_t slot = 0; slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++slot) {
                auto & runtime = functional_snapshots[family][slot];
                if (runtime.info.valid && runtime.info.expires_at_us <= now_us) {
                    runtime.info = {};
                    if (runtime.adapter) {
                        zero_adapter(*runtime.adapter);
                    }
                    if (out_expired_count) {
                        *out_expired_count += 1;
                    }
                }
                archive.items[slot] = runtime.info;
                if (runtime.info.valid) {
                    count += 1;
                }
            }
            archive.count = count;
        }
    }

    bool capture_process_snapshot(int32_t entry_slot, uint64_t now_us) {
        if (entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
            return false;
        }
        auto & entry = process_entries[entry_slot];
        const int32_t family = entry.info.signature.family;
        if (!entry.info.valid ||
                family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
                !entry.adapter ||
                !functional_configs[family].enabled) {
            return false;
        }

        int32_t slot = -1;
        for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
            if (!entry.snapshots[i].info.valid) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            uint64_t oldest = std::numeric_limits<uint64_t>::max();
            for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
                if (entry.snapshots[i].info.captured_at_us < oldest) {
                    oldest = entry.snapshots[i].info.captured_at_us;
                    slot = i;
                }
            }
        }
        if (slot < 0 || !ensure_process_snapshot_adapter(entry_slot, slot) ||
                !copy_adapter(*entry.snapshots[slot].adapter, *entry.adapter)) {
            return false;
        }

        auto & archive = entry.snapshot_archive;
        auto & runtime = entry.snapshots[slot];
        runtime.info = {};
        runtime.info.valid = true;
        runtime.info.family = family;
        runtime.info.slot = slot;
        runtime.info.source = LLAMA_FUNCTIONAL_SNAPSHOT_SOURCE_WEEKLY_ARCHIVE;
        runtime.info.snapshot_id = next_functional_snapshot_id++;
        runtime.info.captured_at_us = now_us;
        runtime.info.expires_at_us = now_us + FUNCTIONAL_SNAPSHOT_RETENTION_US;
        runtime.info.source_update_count = entry.info.update_count;
        runtime.info.self_state_gradient_norm = functional_trace.last_activation.allostatic_gradient_norm;
        runtime.info.robustness_score = clamp_unit(0.5f + 0.5f * entry.info.last_signed_outcome);
        runtime.info.last_signed_outcome = entry.info.last_signed_outcome;
        runtime.info.dominant_direction_cosine =
                vector_cosine_similarity(entry.dominant_direction, entry.last_signature);
        archive.family = family;
        archive.last_capture_us = now_us;
        archive.next_capture_due_us = now_us + FUNCTIONAL_SNAPSHOT_PERIOD_US;
        archive.items[slot] = runtime.info;

        uint32_t count = 0;
        for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
            if (entry.snapshots[i].info.valid) {
                count += 1;
            }
            archive.items[i] = entry.snapshots[i].info;
        }
        archive.count = count;
        return true;
    }

    void expire_process_snapshots(uint64_t now_us, uint32_t * out_expired_count = nullptr) {
        for (int32_t entry_slot = 0; entry_slot < LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES; ++entry_slot) {
            auto & entry = process_entries[entry_slot];
            auto & archive = entry.snapshot_archive;
            uint32_t count = 0;
            for (int32_t slot = 0; slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++slot) {
                auto & runtime = entry.snapshots[slot];
                if (runtime.info.valid && runtime.info.expires_at_us <= now_us) {
                    runtime.info = {};
                    if (runtime.adapter) {
                        zero_adapter(*runtime.adapter);
                    }
                    if (out_expired_count) {
                        *out_expired_count += 1;
                    }
                }
                archive.items[slot] = runtime.info;
                if (runtime.info.valid) {
                    count += 1;
                }
            }
            archive.count = count;
        }
    }

    static size_t serialized_adapter_size(const llama_adapter_lora & adapter) {
        size_t total = sizeof(uint32_t);
        for (const auto & it : adapter.ab_map) {
            const size_t size_a = it.second.a->ne[0] * it.second.a->ne[1];
            const size_t size_b = it.second.b->ne[0] * it.second.b->ne[1];
            total += sizeof(uint32_t) + it.first.size();
            total += sizeof(float);
            total += sizeof(uint64_t) * 2;
            total += (size_a + size_b) * sizeof(float);
        }
        return total;
    }

    bool serialized_adapter_export(const llama_adapter_lora & adapter, void * dst, size_t size) const {
        if (!dst || size < serialized_adapter_size(adapter)) {
            return false;
        }
        uint8_t * cursor = static_cast<uint8_t *>(dst);
        const uint32_t count = static_cast<uint32_t>(adapter.ab_map.size());
        std::memcpy(cursor, &count, sizeof(count));
        cursor += sizeof(count);
        for (const auto & it : adapter.ab_map) {
            const uint32_t name_len = static_cast<uint32_t>(it.first.size());
            const uint64_t size_a = it.second.a->ne[0] * it.second.a->ne[1];
            const uint64_t size_b = it.second.b->ne[0] * it.second.b->ne[1];
            std::vector<float> data_a(size_a, 0.0f);
            std::vector<float> data_b(size_b, 0.0f);
            ggml_backend_tensor_get(it.second.a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_get(it.second.b, data_b.data(), 0, data_b.size() * sizeof(float));
            std::memcpy(cursor, &name_len, sizeof(name_len));
            cursor += sizeof(name_len);
            std::copy_n(it.first.begin(), name_len, cursor);
            cursor += name_len;
            std::memcpy(cursor, &it.second.gain, sizeof(float));
            cursor += sizeof(float);
            std::memcpy(cursor, &size_a, sizeof(size_a));
            cursor += sizeof(size_a);
            std::memcpy(cursor, &size_b, sizeof(size_b));
            cursor += sizeof(size_b);
            std::memcpy(cursor, data_a.data(), data_a.size() * sizeof(float));
            cursor += data_a.size() * sizeof(float);
            std::memcpy(cursor, data_b.data(), data_b.size() * sizeof(float));
            cursor += data_b.size() * sizeof(float);
        }
        return true;
    }

    static bool serialized_adapter_import(llama_adapter_lora & adapter, const void * src, size_t size) {
        if (!src || size < sizeof(uint32_t)) {
            return false;
        }
        const uint8_t * cursor = static_cast<const uint8_t *>(src);
        const uint8_t * end = cursor + size;
        uint32_t count = 0;
        std::memcpy(&count, cursor, sizeof(count));
        cursor += sizeof(count);
        for (uint32_t idx = 0; idx < count; ++idx) {
            if (cursor + sizeof(uint32_t) > end) {
                return false;
            }
            uint32_t name_len = 0;
            std::memcpy(&name_len, cursor, sizeof(name_len));
            cursor += sizeof(name_len);
            if (cursor + name_len + sizeof(float) + sizeof(uint64_t) * 2 > end) {
                return false;
            }
            const std::string name(reinterpret_cast<const char *>(cursor), name_len);
            cursor += name_len;
            auto it = adapter.ab_map.find(name);
            if (it == adapter.ab_map.end()) {
                return false;
            }
            float gain = 0.0f;
            uint64_t size_a = 0;
            uint64_t size_b = 0;
            std::memcpy(&gain, cursor, sizeof(gain));
            cursor += sizeof(gain);
            std::memcpy(&size_a, cursor, sizeof(size_a));
            cursor += sizeof(size_a);
            std::memcpy(&size_b, cursor, sizeof(size_b));
            cursor += sizeof(size_b);
            const size_t expected_a = it->second.a->ne[0] * it->second.a->ne[1];
            const size_t expected_b = it->second.b->ne[0] * it->second.b->ne[1];
            if (size_a != expected_a || size_b != expected_b ||
                    cursor + (size_a + size_b) * sizeof(float) > end) {
                return false;
            }
            ggml_backend_tensor_set(it->second.a, cursor, 0, size_a * sizeof(float));
            cursor += size_a * sizeof(float);
            ggml_backend_tensor_set(it->second.b, cursor, 0, size_b * sizeof(float));
            cursor += size_b * sizeof(float);
            it->second.gain = gain;
        }
        return cursor == end;
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
        base.state_signal.push_back(family == LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION ? 1.0f : 0.0f);
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

    void refresh_user_personality_gain_stats() {
        const auto gains = user_personality_adapter ?
                adapter_gain_stats(*user_personality_adapter) :
                std::pair<float, float>(0.0f, 0.0f);
        user_personality_stats.gain_mean = gains.first;
        user_personality_stats.gain_max = gains.second;
    }

    float user_personality_confidence() const {
        if (!user_personality_stats.enabled) {
            return 0.0f;
        }
        const float update_signal = clamp_unit((float) user_personality_stats.updates_applied / 8.0f);
        const float token_signal = clamp_unit((float) std::min<uint64_t>(user_personality_stats.tokens_ingested, 2048) / 2048.0f);
        return clamp_unit(0.55f * update_signal + 0.45f * token_signal);
    }

    bool apply_user_personality_write(
            const llama_self_state_event & event,
            const llama_self_state_feature_vector * features) {
        if (!user_personality_adapter ||
                !event.tokens || event.n_tokens == 0 ||
                event.role != LLAMA_SELF_STATE_EVENT_USER) {
            return false;
        }

        const active_lora_embedding embedding = embedder->embed(event.tokens, event.n_tokens);
        if (embedding.values.empty()) {
            return false;
        }

        const float similarity = cosine_similarity(embedding, last_user_personality_embedding);
        if (similarity > 0.997f) {
            return true;
        }

        active_lora_write_features write_features = build_write_features(embedding, event, features, false);
        write_features.update_emphasis = clamp_unit(std::max(write_features.update_emphasis, 0.20f + 0.45f * write_features.social_alignment));
        write_features.social_alignment = clamp_unit(std::max(write_features.social_alignment, 0.55f));

        runtime_lora_write_result write_result = {};
        if (!train_on_adapter(*user_personality_adapter, embedding, write_features, 1.0f, false, &write_result)) {
            return false;
        }

        last_user_personality_embedding = embedding;
        user_personality_stats.updates_applied++;
        user_personality_stats.optimizer_step_count = write_result.transaction_step;
        user_personality_stats.tokens_ingested += event.n_tokens;
        user_personality_stats.optimizer_last_update_norm = write_result.update_norm;
        user_personality_stats.confidence = user_personality_confidence();
        refresh_user_personality_gain_stats();
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
    if (pimpl->user_personality_adapter) {
        pimpl->owner.detach_adapter_runtime(pimpl->user_personality_adapter.get());
        const_cast<llama_model &>(pimpl->owner.model).loras.erase(pimpl->user_personality_adapter.get());
    }
    for (auto & adapter : pimpl->functional_adapters) {
        if (adapter) {
            pimpl->owner.detach_adapter_runtime(adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(adapter.get());
        }
    }
    for (auto & adapter : pimpl->functional_bootstrap_adapters) {
        if (adapter) {
            pimpl->owner.detach_adapter_runtime(adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(adapter.get());
        }
    }
    for (auto & adapter : pimpl->functional_replay_adapters) {
        if (adapter) {
            pimpl->owner.detach_adapter_runtime(adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(adapter.get());
        }
    }
    for (auto & family_snapshots : pimpl->functional_snapshots) {
        for (auto & snapshot : family_snapshots) {
            if (snapshot.adapter) {
                pimpl->owner.detach_adapter_runtime(snapshot.adapter.get());
                const_cast<llama_model &>(pimpl->owner.model).loras.erase(snapshot.adapter.get());
            }
        }
    }
    for (auto & bucket : pimpl->buckets) {
        if (bucket.adapter) {
            pimpl->owner.detach_adapter_runtime(bucket.adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(bucket.adapter.get());
        }
    }
    for (auto & entry : pimpl->process_entries) {
        if (entry.adapter) {
            pimpl->owner.detach_adapter_runtime(entry.adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(entry.adapter.get());
        }
        if (entry.bootstrap_adapter) {
            pimpl->owner.detach_adapter_runtime(entry.bootstrap_adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(entry.bootstrap_adapter.get());
        }
        if (entry.replay_adapter) {
            pimpl->owner.detach_adapter_runtime(entry.replay_adapter.get());
            const_cast<llama_model &>(pimpl->owner.model).loras.erase(entry.replay_adapter.get());
        }
        for (auto & snapshot : entry.snapshots) {
            if (snapshot.adapter) {
                pimpl->owner.detach_adapter_runtime(snapshot.adapter.get());
                const_cast<llama_model &>(pimpl->owner.model).loras.erase(snapshot.adapter.get());
            }
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

    if (!pimpl->user_personality_adapter) {
        if (!pimpl->create_runtime_adapter(
                    pimpl->user_personality_adapter,
                    selected_rank,
                    "memory.user_personality",
                    params.adapter_scale,
                    LLAMA_ADAPTER_LORA_LAYER_USER_PERSONALITY,
                    false)) {
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
    pimpl->user_personality_stats = {};
    pimpl->user_personality_stats.enabled = pimpl->user_personality_adapter != nullptr;
    pimpl->user_personality_stats.selected_rank = selected_rank;
    pimpl->user_personality_stats.adapter_scale = params.adapter_scale;
    pimpl->runtime_lora_adam.clear();
    pimpl->runtime_lora_write_count.clear();
    pimpl->temporal_bias_adam = {};
    pimpl->temporal_encoding_bias.adam_step = 0;
    pimpl->temporal_encoding_bias.last_update_norm = 0.0f;
    pimpl->initialized = true;
    pimpl->reset_active_stage(llama_time_us());
    pimpl->last_user_personality_embedding = {};
    pimpl->initialize_functional_defaults(selected_rank);
    pimpl->initialize_process_defaults();

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
        if (!pimpl->functional_bootstrap_adapters[family]) {
            if (!pimpl->create_runtime_adapter(
                        pimpl->functional_bootstrap_adapters[family],
                        selected_rank,
                        std::string("functional.bootstrap.") + functional_family_name(family),
                        0.0f,
                        functional_family_role(family))) {
                return false;
            }
            randomize_adapter(
                    *pimpl->functional_bootstrap_adapters[family],
                    pimpl->bootstrap_rng,
                    pimpl->functional_configs[family].bootstrap_weight_init_std);
        }
        pimpl->functional_states[family].compatible =
                pimpl->functional_adapters[family] != nullptr &&
                pimpl->functional_bootstrap_adapters[family] != nullptr;
        pimpl->functional_trace.family_state[family] = pimpl->functional_states[family];
        pimpl->record_functional_signature(family);
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
    if (!pimpl->initialized) {
        return false;
    }
    const bool active_ok = pimpl->apply_write(event, features, 1.0f, false);
    if (event.role == LLAMA_SELF_STATE_EVENT_USER) {
        (void) pimpl->apply_user_personality_write(event, features);
    }
    return active_ok;
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

bool llama_active_lora_manager::user_personality_get_stats(llama_user_personality_lora_stats * out_stats) const {
    if (!out_stats || !pimpl->initialized) {
        return false;
    }

    llama_user_personality_lora_stats stats = pimpl->user_personality_stats;
    stats.attached_for_simulation = false;
    if (pimpl->user_personality_adapter) {
        llama_serving_lora_layer_info layer = {};
        const int32_t count = pimpl->owner.serving_lora_stack_count();
        for (int32_t i = 0; i < count; ++i) {
            if (pimpl->owner.serving_lora_stack_layer(i, &layer) &&
                    layer.role == LLAMA_SERVING_LORA_LAYER_USER_PERSONALITY &&
                    layer.scale > 0.0f) {
                stats.attached_for_simulation = true;
                break;
            }
        }
    }
    *out_stats = stats;
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

bool llama_active_lora_manager::functional_snapshot_archive_get(
        int32_t family,
        llama_functional_lora_snapshot_archive * out_archive) const {
    if (!out_archive || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }
    *out_archive = pimpl->functional_snapshot_archives[family];
    return true;
}

bool llama_active_lora_manager::functional_snapshot_info_get(
        int32_t family,
        int32_t slot,
        llama_functional_lora_snapshot_info * out_info) const {
    if (!out_info ||
            family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
            slot < 0 || slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
        return false;
    }
    *out_info = pimpl->functional_snapshots[family][slot].info;
    return true;
}

bool llama_active_lora_manager::functional_get_last_snapshot_maintenance(
        llama_functional_snapshot_maintenance_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = pimpl->functional_snapshot_trace;
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

bool llama_active_lora_manager::functional_replay_override_begin(const llama_functional_lora_replay_override & config) {
    if (!pimpl->initialized ||
            config.family < 0 || config.family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }

    const int32_t family = config.family;
    if (config.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED) {
        if (config.snapshot_slot < 0 || config.snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY ||
                !pimpl->functional_snapshots[family][config.snapshot_slot].info.valid ||
                !pimpl->functional_snapshots[family][config.snapshot_slot].adapter) {
            return false;
        }
    } else if (config.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ||
               config.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED) {
        if (!pimpl->functional_adapters[family] || !pimpl->ensure_replay_adapter(family)) {
            return false;
        }
        if (!pimpl->copy_adapter(*pimpl->functional_replay_adapters[family], *pimpl->functional_adapters[family])) {
            return false;
        }

        const float signed_scale =
                config.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ?
                        std::max(0.02f, config.perturbation_scale) :
                        std::max(0.01f, config.perturbation_scale * 0.5f);
        std::normal_distribution<float> dist(0.0f, signed_scale);
        std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> perturb_signature = {};
        for (float & value : perturb_signature) {
            value = dist(pimpl->bootstrap_rng);
        }
        if (config.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL &&
                pimpl->functional_signature_valid[family]) {
            const auto & dominant = pimpl->functional_dominant_directions[family];
            float dot = 0.0f;
            for (size_t i = 0; i < FUNCTIONAL_DIRECTION_SKETCH_DIMS; ++i) {
                dot += perturb_signature[i] * dominant[i];
            }
            for (size_t i = 0; i < FUNCTIONAL_DIRECTION_SKETCH_DIMS; ++i) {
                perturb_signature[i] -= dot * dominant[i];
            }
        }
        normalize_array(perturb_signature);
        for (auto & it : pimpl->functional_replay_adapters[family]->ab_map) {
            const size_t size_a = it.second.a->ne[0] * it.second.a->ne[1];
            const size_t size_b = it.second.b->ne[0] * it.second.b->ne[1];
            std::vector<float> data_a(size_a, 0.0f);
            std::vector<float> data_b(size_b, 0.0f);
            ggml_backend_tensor_get(it.second.a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_get(it.second.b, data_b.data(), 0, data_b.size() * sizeof(float));
            for (size_t i = 0; i < data_a.size(); ++i) {
                data_a[i] += signed_scale * perturb_signature[i % FUNCTIONAL_DIRECTION_SKETCH_DIMS];
            }
            for (size_t i = 0; i < data_b.size(); ++i) {
                data_b[i] += signed_scale * perturb_signature[(i + data_a.size()) % FUNCTIONAL_DIRECTION_SKETCH_DIMS];
            }
            ggml_backend_tensor_set(it.second.a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_set(it.second.b, data_b.data(), 0, data_b.size() * sizeof(float));
        }
    }

    pimpl->functional_replay_overrides[family] = config;
    return true;
}

bool llama_active_lora_manager::functional_replay_override_end(int32_t family) {
    if (!pimpl->initialized || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }
    pimpl->functional_replay_overrides[family] = {};
    pimpl->apply_functional_runtime_scale(
            family,
            pimpl->functional_states[family].current_gain,
            pimpl->functional_states[family].last_bootstrap_perturbation);
    return true;
}

bool llama_active_lora_manager::functional_get_last_differential_update(
        int32_t family,
        llama_functional_lora_differential_update * out_update) const {
    if (!out_update || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }
    *out_update = pimpl->functional_differential_updates[family];
    return true;
}

size_t llama_active_lora_manager::functional_snapshot_blob_size(int32_t family, int32_t slot) const {
    if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
            slot < 0 || slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
        return 0;
    }
    const auto & runtime = pimpl->functional_snapshots[family][slot];
    if (!runtime.info.valid || !runtime.adapter) {
        return 0;
    }
    return pimpl->serialized_adapter_size(*runtime.adapter);
}

bool llama_active_lora_manager::functional_snapshot_blob_export(int32_t family, int32_t slot, void * dst, size_t size) const {
    if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
            slot < 0 || slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
        return false;
    }
    const auto & runtime = pimpl->functional_snapshots[family][slot];
    return runtime.info.valid && runtime.adapter &&
            pimpl->serialized_adapter_export(*runtime.adapter, dst, size);
}

bool llama_active_lora_manager::functional_snapshot_blob_import(
        int32_t family,
        int32_t slot,
        const llama_functional_lora_snapshot_info & info,
        const void * src,
        size_t size) {
    if (!pimpl->initialized ||
            family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
            slot < 0 || slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY ||
            !pimpl->ensure_snapshot_adapter(family, slot) ||
            !pimpl->serialized_adapter_import(*pimpl->functional_snapshots[family][slot].adapter, src, size)) {
        return false;
    }
    pimpl->functional_snapshots[family][slot].info = info;
    pimpl->functional_snapshot_archives[family].items[slot] = info;
    uint32_t count = 0;
    uint64_t last_capture = 0;
    for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
        if (pimpl->functional_snapshots[family][i].info.valid) {
            count += 1;
            last_capture = std::max(last_capture, pimpl->functional_snapshots[family][i].info.captured_at_us);
        }
    }
    pimpl->functional_snapshot_archives[family].family = family;
    pimpl->functional_snapshot_archives[family].count = count;
    pimpl->functional_snapshot_archives[family].last_capture_us = last_capture;
    pimpl->functional_snapshot_archives[family].next_capture_due_us = last_capture == 0 ? FUNCTIONAL_SNAPSHOT_PERIOD_US : last_capture + FUNCTIONAL_SNAPSHOT_PERIOD_US;
    return true;
}

bool llama_active_lora_manager::functional_snapshot_maintain(uint64_t now_us) {
    if (!pimpl->initialized) {
        return false;
    }
    pimpl->functional_snapshot_trace = {};
    pimpl->functional_snapshot_trace.ran = true;
    pimpl->functional_snapshot_trace.now_us = now_us;
    pimpl->expire_functional_snapshots(now_us, &pimpl->functional_snapshot_trace.expired_count);
    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        auto & archive = pimpl->functional_snapshot_archives[family];
        if (archive.next_capture_due_us == 0) {
            archive.next_capture_due_us = now_us + FUNCTIONAL_SNAPSHOT_PERIOD_US;
        }
        if (now_us >= archive.next_capture_due_us && pimpl->capture_functional_snapshot(family, now_us)) {
            pimpl->functional_snapshot_trace.captured_any = true;
            pimpl->functional_snapshot_trace.captured_count += 1;
            pimpl->functional_snapshot_trace.captured_family_mask |= (1ull << family);
        }
        if (pimpl->functional_snapshot_trace.next_due_us == 0 ||
                (archive.next_capture_due_us > 0 && archive.next_capture_due_us < pimpl->functional_snapshot_trace.next_due_us)) {
            pimpl->functional_snapshot_trace.next_due_us = archive.next_capture_due_us;
        }
    }
    return true;
}

bool llama_active_lora_manager::process_functional_get_params(llama_process_functional_params * out_params) const {
    if (!out_params) {
        return false;
    }
    *out_params = pimpl->process_params;
    return true;
}

bool llama_active_lora_manager::process_functional_set_params(const llama_process_functional_params & params) {
    if (!pimpl->initialized) {
        return false;
    }
    pimpl->process_params = params;
    pimpl->process_params.max_entries = std::max<uint32_t>(1, std::min<uint32_t>(params.max_entries, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES));
    pimpl->process_params.min_observations = std::max<uint32_t>(1, params.min_observations);
    pimpl->process_params.noop_abs_ceiling = clamp_unit(params.noop_abs_ceiling);
    pimpl->process_params.weak_positive_ceiling = clamp_unit(params.weak_positive_ceiling);
    pimpl->process_params.mean_outcome_ceiling = clamp_signed_unit(params.mean_outcome_ceiling);
    pimpl->process_params.weak_or_worse_ratio_threshold = clamp_unit(params.weak_or_worse_ratio_threshold);
    pimpl->process_params.utility_decay = clamp_range(params.utility_decay, 0.0f, 1.0f);
    return true;
}

int32_t llama_active_lora_manager::process_functional_entry_count() const {
    return pimpl->initialized ? pimpl->process_entry_count() : 0;
}

bool llama_active_lora_manager::process_functional_entry_get(int32_t index, llama_process_functional_entry_info * out_info) const {
    if (!out_info || !pimpl->initialized || index < 0 || index >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }
    *out_info = pimpl->process_entries[index].info;
    return true;
}

int32_t llama_active_lora_manager::process_functional_ledger_count() const {
    return pimpl->initialized ? pimpl->process_ledger_count() : 0;
}

bool llama_active_lora_manager::process_functional_ledger_get(int32_t index, llama_process_functional_ledger_info * out_info) const {
    if (!out_info || !pimpl->initialized || index < 0 || index >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }
    *out_info = pimpl->process_ledgers[index].info;
    return true;
}

bool llama_active_lora_manager::process_functional_get_last_trace(llama_process_functional_trace * out_trace) const {
    if (!out_trace || !pimpl->initialized) {
        return false;
    }
    *out_trace = pimpl->last_process_trace;
    return true;
}

bool llama_active_lora_manager::process_functional_get_current_signature(llama_process_functional_signature * out_signature) const {
    if (!out_signature || !pimpl->initialized) {
        return false;
    }
    *out_signature = pimpl->current_process_signature;
    return true;
}

bool llama_active_lora_manager::process_functional_snapshot_archive_get(
        int32_t entry_slot,
        llama_functional_lora_snapshot_archive * out_archive) const {
    if (!out_archive || !pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }
    *out_archive = pimpl->process_entries[entry_slot].snapshot_archive;
    return true;
}

bool llama_active_lora_manager::process_functional_snapshot_info_get(
        int32_t entry_slot,
        int32_t snapshot_slot,
        llama_functional_lora_snapshot_info * out_info) const {
    if (!out_info || !pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
            snapshot_slot < 0 || snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
        return false;
    }
    *out_info = pimpl->process_entries[entry_slot].snapshots[snapshot_slot].info;
    return true;
}

bool llama_active_lora_manager::process_functional_get_last_snapshot_maintenance(
        llama_functional_snapshot_maintenance_trace * out_trace) const {
    if (!out_trace || !pimpl->initialized) {
        return false;
    }
    *out_trace = pimpl->process_snapshot_trace;
    return true;
}

bool llama_active_lora_manager::process_functional_replay_override_begin(
        int32_t entry_slot,
        const llama_functional_lora_replay_override & config) {
    if (!pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }

    auto & entry = pimpl->process_entries[entry_slot];
    const int32_t family = entry.info.signature.family;
    if (!entry.info.valid || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return false;
    }

    llama_functional_lora_replay_override applied = config;
    applied.family = family;

    if (applied.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED) {
        if (applied.snapshot_slot < 0 ||
                applied.snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY ||
                !entry.snapshots[applied.snapshot_slot].info.valid ||
                !entry.snapshots[applied.snapshot_slot].adapter) {
            return false;
        }
    } else if (applied.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ||
               applied.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED) {
        if (!entry.adapter || !pimpl->ensure_process_replay_adapter(entry_slot) ||
                !pimpl->copy_adapter(*entry.replay_adapter, *entry.adapter)) {
            return false;
        }

        const float signed_scale =
                applied.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ?
                        std::max(0.02f, applied.perturbation_scale) :
                        std::max(0.01f, applied.perturbation_scale * 0.5f);
        std::normal_distribution<float> dist(0.0f, signed_scale);
        std::array<float, FUNCTIONAL_DIRECTION_SKETCH_DIMS> perturb_signature = {};
        for (float & value : perturb_signature) {
            value = dist(pimpl->bootstrap_rng);
        }
        if (applied.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL && entry.signature_valid) {
            const auto & dominant = entry.dominant_direction;
            float dot = 0.0f;
            for (size_t i = 0; i < FUNCTIONAL_DIRECTION_SKETCH_DIMS; ++i) {
                dot += perturb_signature[i] * dominant[i];
            }
            for (size_t i = 0; i < FUNCTIONAL_DIRECTION_SKETCH_DIMS; ++i) {
                perturb_signature[i] -= dot * dominant[i];
            }
        }
        normalize_array(perturb_signature);
        for (auto & it : entry.replay_adapter->ab_map) {
            const size_t size_a = it.second.a->ne[0] * it.second.a->ne[1];
            const size_t size_b = it.second.b->ne[0] * it.second.b->ne[1];
            std::vector<float> data_a(size_a, 0.0f);
            std::vector<float> data_b(size_b, 0.0f);
            ggml_backend_tensor_get(it.second.a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_get(it.second.b, data_b.data(), 0, data_b.size() * sizeof(float));
            for (size_t i = 0; i < data_a.size(); ++i) {
                data_a[i] += signed_scale * perturb_signature[i % FUNCTIONAL_DIRECTION_SKETCH_DIMS];
            }
            for (size_t i = 0; i < data_b.size(); ++i) {
                data_b[i] += signed_scale * perturb_signature[(i + data_a.size()) % FUNCTIONAL_DIRECTION_SKETCH_DIMS];
            }
            ggml_backend_tensor_set(it.second.a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_set(it.second.b, data_b.data(), 0, data_b.size() * sizeof(float));
        }
    } else {
        return false;
    }

    entry.replay_override = applied;
    return true;
}

bool llama_active_lora_manager::process_functional_replay_override_end(int32_t entry_slot) {
    if (!pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }
    pimpl->process_entries[entry_slot].replay_override = {};
    return true;
}

bool llama_active_lora_manager::process_functional_get_last_differential_update(
        int32_t entry_slot,
        llama_functional_lora_differential_update * out_update) const {
    if (!out_update || !pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }
    *out_update = pimpl->process_entries[entry_slot].differential_update;
    return true;
}

bool llama_active_lora_manager::process_functional_apply_differential_update(
        int32_t entry_slot,
        int32_t proposal_family,
        int32_t replay_mode,
        int32_t snapshot_slot,
        float signed_score_delta,
        float magnitude,
        float robustness_score) {
    if (!pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }
    auto & entry = pimpl->process_entries[entry_slot];
    const int32_t family = entry.info.signature.family;
    if (!entry.info.valid ||
            family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
            !entry.adapter) {
        return false;
    }

    llama_adapter_lora * source_adapter = nullptr;
    if (replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED) {
        if (snapshot_slot < 0 || snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
            return false;
        }
        source_adapter = entry.snapshots[snapshot_slot].adapter.get();
    } else if (replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ||
               replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED) {
        source_adapter = entry.replay_adapter.get();
    } else {
        return false;
    }
    if (!source_adapter) {
        return false;
    }

    const float signed_step = clamp_range(signed_score_delta * std::max(0.05f, magnitude), -0.75f, 0.75f);
    if (std::fabs(signed_step) <= 1.0e-5f) {
        return false;
    }

    double sum_sq = 0.0;
    for (auto & it : entry.adapter->ab_map) {
        auto src_it = source_adapter->ab_map.find(it.first);
        if (src_it == source_adapter->ab_map.end()) {
            continue;
        }
        auto & dst_weight = it.second;
        const auto & src_weight = src_it->second;
        const size_t size_a = dst_weight.a->ne[0] * dst_weight.a->ne[1];
        const size_t size_b = dst_weight.b->ne[0] * dst_weight.b->ne[1];
        std::vector<float> dst_a(size_a, 0.0f);
        std::vector<float> src_a(size_a, 0.0f);
        std::vector<float> dst_b(size_b, 0.0f);
        std::vector<float> src_b(size_b, 0.0f);
        ggml_backend_tensor_get(dst_weight.a, dst_a.data(), 0, dst_a.size() * sizeof(float));
        ggml_backend_tensor_get(src_weight.a, src_a.data(), 0, src_a.size() * sizeof(float));
        ggml_backend_tensor_get(dst_weight.b, dst_b.data(), 0, dst_b.size() * sizeof(float));
        ggml_backend_tensor_get(src_weight.b, src_b.data(), 0, src_b.size() * sizeof(float));
        for (size_t i = 0; i < size_a; ++i) {
            const float delta = signed_step * (src_a[i] - dst_a[i]);
            dst_a[i] += delta;
            sum_sq += (double) delta * (double) delta;
        }
        for (size_t i = 0; i < size_b; ++i) {
            const float delta = signed_step * (src_b[i] - dst_b[i]);
            dst_b[i] += delta;
            sum_sq += (double) delta * (double) delta;
        }
        ggml_backend_tensor_set(dst_weight.a, dst_a.data(), 0, dst_a.size() * sizeof(float));
        ggml_backend_tensor_set(dst_weight.b, dst_b.data(), 0, dst_b.size() * sizeof(float));
        const float gain_delta = signed_step * (src_weight.gain - dst_weight.gain);
        dst_weight.gain = clamp_range(
                dst_weight.gain + gain_delta,
                pimpl->functional_configs[family].gain_clip_min,
                pimpl->functional_configs[family].gain_clip_max);
        sum_sq += (double) gain_delta * (double) gain_delta;
    }

    const float difference_norm = pimpl->adapter_difference_norm(*entry.adapter, *source_adapter);
    const float applied_update_norm = std::sqrt((float) sum_sq);
    const uint64_t transaction_step = ++pimpl->runtime_lora_write_count[entry.adapter.get()];

    entry.differential_update = {};
    entry.differential_update.valid = true;
    entry.differential_update.family = family;
    entry.differential_update.proposal_family = proposal_family;
    entry.differential_update.source_snapshot_slot = snapshot_slot;
    entry.differential_update.signed_score_delta = signed_score_delta;
    entry.differential_update.magnitude = magnitude;
    entry.differential_update.lora_difference_norm = difference_norm;
    entry.differential_update.applied_update_norm = applied_update_norm;
    entry.differential_update.robustness_score = robustness_score;
    entry.differential_update.adapter_optimizer_step = transaction_step;
    pimpl->record_process_signature(entry_slot);
    return true;
}

size_t llama_active_lora_manager::process_functional_entry_blob_size(int32_t index) const {
    if (!pimpl->initialized || index < 0 || index >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return 0;
    }
    const auto & entry = pimpl->process_entries[index];
    if (!entry.info.valid || !entry.adapter || !entry.bootstrap_adapter) {
        return 0;
    }
    return sizeof(uint64_t) * 2 +
            pimpl->serialized_adapter_size(*entry.adapter) +
            pimpl->serialized_adapter_size(*entry.bootstrap_adapter);
}

bool llama_active_lora_manager::process_functional_entry_blob_export(int32_t index, void * dst, size_t size) const {
    if (!dst || !pimpl->initialized || index < 0 || index >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
        return false;
    }
    const auto & entry = pimpl->process_entries[index];
    if (!entry.info.valid || !entry.adapter || !entry.bootstrap_adapter) {
        return false;
    }
    const uint64_t learned_size = (uint64_t) pimpl->serialized_adapter_size(*entry.adapter);
    const uint64_t bootstrap_size = (uint64_t) pimpl->serialized_adapter_size(*entry.bootstrap_adapter);
    if (size < sizeof(uint64_t) * 2 + learned_size + bootstrap_size) {
        return false;
    }
    uint8_t * cursor = static_cast<uint8_t *>(dst);
    std::memcpy(cursor, &learned_size, sizeof(learned_size));
    cursor += sizeof(learned_size);
    std::memcpy(cursor, &bootstrap_size, sizeof(bootstrap_size));
    cursor += sizeof(bootstrap_size);
    if (!pimpl->serialized_adapter_export(*entry.adapter, cursor, (size_t) learned_size)) {
        return false;
    }
    cursor += learned_size;
    return pimpl->serialized_adapter_export(*entry.bootstrap_adapter, cursor, (size_t) bootstrap_size);
}

bool llama_active_lora_manager::process_functional_entry_blob_import(
        int32_t index,
        const llama_process_functional_entry_info & info,
        const void * src,
        size_t size) {
    if (!pimpl->initialized ||
            !src ||
            index < 0 ||
            index >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
            !info.valid ||
            !pimpl->ensure_process_entry_adapters(index, info.signature.family)) {
        return false;
    }
    if (size < sizeof(uint64_t) * 2) {
        return false;
    }
    const uint8_t * cursor = static_cast<const uint8_t *>(src);
    uint64_t learned_size = 0;
    uint64_t bootstrap_size = 0;
    std::memcpy(&learned_size, cursor, sizeof(learned_size));
    cursor += sizeof(learned_size);
    std::memcpy(&bootstrap_size, cursor, sizeof(bootstrap_size));
    cursor += sizeof(bootstrap_size);
    if (sizeof(uint64_t) * 2 + learned_size + bootstrap_size != size) {
        return false;
    }
    auto & entry = pimpl->process_entries[index];
    if (!pimpl->serialized_adapter_import(*entry.adapter, cursor, (size_t) learned_size)) {
        return false;
    }
    cursor += learned_size;
    if (!pimpl->serialized_adapter_import(*entry.bootstrap_adapter, cursor, (size_t) bootstrap_size)) {
        return false;
    }
    entry.info = info;
    entry.info.slot = index;
    entry.snapshot_archive.family = info.signature.family;
    entry.snapshot_archive.next_capture_due_us = FUNCTIONAL_SNAPSHOT_PERIOD_US;
    entry.replay_override = {};
    entry.differential_update = {};
    entry.signature_valid = false;
    for (int32_t slot = 0; slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++slot) {
        entry.snapshots[slot].info.family = info.signature.family;
        entry.snapshots[slot].info.slot = slot;
    }
    return true;
}

size_t llama_active_lora_manager::process_functional_snapshot_blob_size(int32_t entry_slot, int32_t snapshot_slot) const {
    if (!pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
            snapshot_slot < 0 || snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
        return 0;
    }
    const auto & runtime = pimpl->process_entries[entry_slot].snapshots[snapshot_slot];
    if (!runtime.info.valid || !runtime.adapter) {
        return 0;
    }
    return pimpl->serialized_adapter_size(*runtime.adapter);
}

bool llama_active_lora_manager::process_functional_snapshot_blob_export(
        int32_t entry_slot,
        int32_t snapshot_slot,
        void * dst,
        size_t size) const {
    if (!pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
            snapshot_slot < 0 || snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
        return false;
    }
    const auto & runtime = pimpl->process_entries[entry_slot].snapshots[snapshot_slot];
    return runtime.info.valid && runtime.adapter &&
            pimpl->serialized_adapter_export(*runtime.adapter, dst, size);
}

bool llama_active_lora_manager::process_functional_snapshot_blob_import(
        int32_t entry_slot,
        int32_t snapshot_slot,
        const llama_functional_lora_snapshot_info & info,
        const void * src,
        size_t size) {
    if (!pimpl->initialized ||
            entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
            snapshot_slot < 0 || snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY ||
            !pimpl->process_entries[entry_slot].info.valid ||
            !pimpl->ensure_process_snapshot_adapter(entry_slot, snapshot_slot) ||
            !pimpl->serialized_adapter_import(*pimpl->process_entries[entry_slot].snapshots[snapshot_slot].adapter, src, size)) {
        return false;
    }
    auto & entry = pimpl->process_entries[entry_slot];
    entry.snapshots[snapshot_slot].info = info;
    entry.snapshot_archive.items[snapshot_slot] = info;
    uint32_t count = 0;
    uint64_t last_capture = 0;
    for (int32_t i = 0; i < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++i) {
        if (entry.snapshots[i].info.valid) {
            count += 1;
            last_capture = std::max(last_capture, entry.snapshots[i].info.captured_at_us);
        }
    }
    entry.snapshot_archive.family = entry.info.signature.family;
    entry.snapshot_archive.count = count;
    entry.snapshot_archive.last_capture_us = last_capture;
    entry.snapshot_archive.next_capture_due_us = last_capture == 0 ? FUNCTIONAL_SNAPSHOT_PERIOD_US : last_capture + FUNCTIONAL_SNAPSHOT_PERIOD_US;
    return true;
}

bool llama_active_lora_manager::process_functional_snapshot_maintain(uint64_t now_us) {
    if (!pimpl->initialized) {
        return false;
    }
    pimpl->process_snapshot_trace = {};
    pimpl->process_snapshot_trace.ran = true;
    pimpl->process_snapshot_trace.now_us = now_us;
    pimpl->expire_process_snapshots(now_us, &pimpl->process_snapshot_trace.expired_count);
    for (int32_t entry_slot = 0; entry_slot < LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES; ++entry_slot) {
        auto & entry = pimpl->process_entries[entry_slot];
        auto & archive = entry.snapshot_archive;
        if (!entry.info.valid) {
            continue;
        }
        if (archive.next_capture_due_us == 0) {
            archive.next_capture_due_us = now_us + FUNCTIONAL_SNAPSHOT_PERIOD_US;
        }
        if (now_us >= archive.next_capture_due_us && pimpl->capture_process_snapshot(entry_slot, now_us)) {
            pimpl->process_snapshot_trace.captured_any = true;
            pimpl->process_snapshot_trace.captured_count += 1;
            pimpl->process_snapshot_trace.captured_family_mask |= (1ull << std::max(0, entry.info.signature.family));
        }
        if (pimpl->process_snapshot_trace.next_due_us == 0 ||
                (archive.next_capture_due_us > 0 && archive.next_capture_due_us < pimpl->process_snapshot_trace.next_due_us)) {
            pimpl->process_snapshot_trace.next_due_us = archive.next_capture_due_us;
        }
    }
    return true;
}

bool llama_active_lora_manager::process_functional_set_execution(const llama_process_functional_signature & signature) {
    if (!pimpl->initialized) {
        return false;
    }
    pimpl->current_process_signature = signature;
    pimpl->current_process_entry_slot = pimpl->find_process_entry_slot(signature);
    pimpl->last_process_trace = {};
    pimpl->last_process_trace.valid = signature.valid;
    pimpl->last_process_trace.signature = signature;
    pimpl->last_process_trace.matched_existing_entry = pimpl->current_process_entry_slot >= 0;
    pimpl->last_process_trace.matched_entry_slot = pimpl->current_process_entry_slot;
    pimpl->last_process_trace.bank_size = (uint32_t) pimpl->process_entry_count();
    pimpl->last_process_trace.bank_capacity =
            std::min<uint32_t>(pimpl->process_params.max_entries, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES);
    if (!signature.valid) {
        pimpl->clear_process_runtime_attachments();
    }
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
    llama_functional_activation_decision applied = decision;
    pimpl->last_process_trace.valid = pimpl->current_process_signature.valid;
    pimpl->last_process_trace.signature = pimpl->current_process_signature;
    pimpl->last_process_trace.matched_existing_entry = pimpl->current_process_entry_slot >= 0;
    pimpl->last_process_trace.matched_entry_slot = pimpl->current_process_entry_slot;
    pimpl->last_process_trace.activation_attached = false;
    pimpl->last_process_trace.bank_size = (uint32_t) pimpl->process_entry_count();
    pimpl->last_process_trace.bank_capacity =
            std::min<uint32_t>(pimpl->process_params.max_entries, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES);

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
                hold.bootstrap_std = 0.0f;
                hold.bootstrap_perturbation = 0.0f;
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
            const float bootstrap_std = bootstrap_noise_std(
                    pimpl->functional_configs[family],
                    state.activation_count);
            const float bootstrap_perturbation = sample_bounded_gaussian(pimpl->bootstrap_rng, bootstrap_std);
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
            hold.bootstrap_std = bootstrap_std;
            hold.bootstrap_perturbation = bootstrap_perturbation;
        } else if (hold.active) {
            hold.microphase_current = decision.microphase;
        }

        const float gain = hold.active ? pimpl->apply_self_model_runtime_gain(family, hold.gain) : 0.0f;
        const float bootstrap_perturbation = hold.active ? hold.bootstrap_perturbation : 0.0f;
        pimpl->apply_functional_runtime_scale(family, gain, bootstrap_perturbation);
        applied.gains[family] = gain;

        state.family = family;
        state.enabled = pimpl->functional_configs[family].enabled && !family_disabled;
        state.compatible =
                pimpl->functional_adapters[family] != nullptr &&
                pimpl->functional_bootstrap_adapters[family] != nullptr;
        state.active_now = gain > 0.0f;
        state.current_gain = gain;
        state.predicted_gain = decision.predicted_gains[family];
        state.last_noise = decision.sampled_noise[family];
        state.current_bootstrap_std = hold.active ?
                hold.bootstrap_std :
                bootstrap_noise_std(pimpl->functional_configs[family], state.activation_count);
        state.last_bootstrap_perturbation = bootstrap_perturbation;
        state.current_microphase = hold.active ? decision.microphase : LLAMA_FUNCTIONAL_MICROPHASE_NONE;
        state.current_hold_unit = hold.active ? hold.hold_unit : LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS;
        state.current_hold_remaining = hold.active ? hold.hold_remaining : 0;
        if (activate_now) {
            state.activation_count += 1;
        }
        applied.bootstrap_std[family] = hold.active ? hold.bootstrap_std : state.current_bootstrap_std;
        applied.bootstrap_perturbation[family] = bootstrap_perturbation;
        pimpl->functional_trace.holds[family] = hold;
        pimpl->functional_trace.family_state[family] = state;
    }

    pimpl->clear_process_runtime_attachments();
    if (pimpl->current_process_entry_slot >= 0 &&
            pimpl->current_process_signature.valid &&
            pimpl->current_process_signature.family >= 0 &&
            pimpl->current_process_signature.family < LLAMA_FUNCTIONAL_LORA_COUNT) {
        auto & entry = pimpl->process_entries[pimpl->current_process_entry_slot];
        const int32_t family = pimpl->current_process_signature.family;
        if (entry.info.valid &&
                pimpl->functional_states[family].active_now &&
                entry.adapter &&
                entry.bootstrap_adapter) {
            const auto & replay = entry.replay_override;
            llama_adapter_lora * active_adapter = entry.adapter.get();
            if (replay.active) {
                if (replay.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED &&
                        replay.snapshot_slot >= 0 &&
                        replay.snapshot_slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
                    active_adapter = entry.snapshots[replay.snapshot_slot].adapter.get();
                } else if (replay.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ||
                           replay.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED) {
                    active_adapter = entry.replay_adapter.get();
                }
            }
            if (!active_adapter) {
                pimpl->last_process_trace.activation_attached = false;
                pimpl->functional_trace.last_activation = applied;
                return true;
            }
            const float gain = clamp_range(
                    pimpl->functional_states[family].current_gain,
                    pimpl->functional_configs[family].gain_clip_min,
                    pimpl->functional_configs[family].gain_clip_max);
            const float bootstrap_std = bootstrap_noise_std(
                    pimpl->functional_configs[family],
                    entry.info.activation_count);
            const float bootstrap_perturbation = sample_bounded_gaussian(pimpl->bootstrap_rng, bootstrap_std);
            pimpl->owner.attach_adapter_runtime(
                    active_adapter,
                    gain,
                    process_functional_learned_role());
            if (!replay.active || !replay.disable_bootstrap) {
                pimpl->owner.attach_adapter_runtime(
                        entry.bootstrap_adapter.get(),
                        replay.active && replay.disable_bootstrap ? 0.0f : bootstrap_perturbation,
                        process_functional_bootstrap_role());
            }
            entry.info.activation_count += 1;
            entry.info.last_used_us = llama_time_us();
            entry.info.current_gain = gain;
            entry.info.current_bootstrap_std = bootstrap_std;
            entry.info.last_bootstrap_perturbation = bootstrap_perturbation;
            pimpl->last_process_trace.activation_attached = true;
            pimpl->last_process_trace.matched_existing_entry = true;
            pimpl->last_process_trace.matched_entry_slot = pimpl->current_process_entry_slot;
        }
    }

    pimpl->functional_trace.last_activation = applied;
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
                hold.bootstrap_std = 0.0f;
                hold.bootstrap_perturbation = 0.0f;
                pimpl->apply_functional_runtime_scale(family, 0.0f, 0.0f);
                pimpl->functional_states[family].active_now = false;
                pimpl->functional_states[family].current_gain = 0.0f;
                pimpl->functional_states[family].current_bootstrap_std = bootstrap_noise_std(
                        pimpl->functional_configs[family],
                        pimpl->functional_states[family].activation_count);
                pimpl->functional_states[family].last_bootstrap_perturbation = 0.0f;
                pimpl->functional_states[family].current_hold_remaining = 0;
                pimpl->functional_trace.holds[family] = hold;
                pimpl->functional_trace.family_state[family] = pimpl->functional_states[family];
            }
        }
    }
    if (pimpl->current_process_entry_slot >= 0 &&
            pimpl->current_process_signature.valid &&
            pimpl->current_process_signature.family >= 0 &&
            pimpl->current_process_signature.family < LLAMA_FUNCTIONAL_LORA_COUNT &&
            !pimpl->functional_states[pimpl->current_process_signature.family].active_now) {
        pimpl->clear_process_runtime_attachments();
        pimpl->last_process_trace.activation_attached = false;
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
    update.source_token_hash = hash_event_tokens(event);
    update.source_token_count = (int32_t) std::min<size_t>(event.n_tokens, (size_t) std::numeric_limits<int32_t>::max());
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
    pimpl->record_functional_signature(family);
    pimpl->functional_trace.family_state[family] = pimpl->functional_states[family];

    if (pimpl->process_params.enabled &&
            pimpl->current_process_signature.valid &&
            pimpl->current_process_signature.family == family) {
        int32_t ledger_slot = pimpl->find_process_ledger_slot(pimpl->current_process_signature);
        if (ledger_slot < 0) {
            ledger_slot = pimpl->first_free_process_ledger_slot();
            if (ledger_slot >= 0) {
                pimpl->process_ledgers[ledger_slot] = {};
                pimpl->process_ledgers[ledger_slot].info.valid = true;
                pimpl->process_ledgers[ledger_slot].info.signature = pimpl->current_process_signature;
            }
        }

        if (ledger_slot >= 0) {
            auto & ledger = pimpl->process_ledgers[ledger_slot];
            auto & info = ledger.info;
            info.valid = true;
            info.signature = pimpl->current_process_signature;
            info.observation_count += 1;
            info.last_observed_us = llama_time_us();
            info.mean_signed_outcome =
                    info.observation_count > 1 ?
                            ((info.mean_signed_outcome * (float) (info.observation_count - 1)) + signed_outcome) /
                                    (float) info.observation_count :
                            signed_outcome;
            info.mean_magnitude =
                    info.observation_count > 1 ?
                            ((info.mean_magnitude * (float) (info.observation_count - 1)) + magnitude) /
                                    (float) info.observation_count :
                            magnitude;
            info.ema_signed_outcome =
                    info.observation_count > 1 ?
                            (0.75f * info.ema_signed_outcome + 0.25f * signed_outcome) :
                            signed_outcome;

            const float abs_outcome = std::fabs(signed_outcome);
            if (signed_outcome < -pimpl->process_params.noop_abs_ceiling) {
                info.negative_count += 1;
            } else if (abs_outcome <= pimpl->process_params.noop_abs_ceiling) {
                info.noop_count += 1;
            } else if (signed_outcome <= pimpl->process_params.weak_positive_ceiling) {
                info.weak_positive_count += 1;
            } else {
                info.strong_positive_count += 1;
            }
            info.weak_or_worse_ratio = info.observation_count > 0 ?
                    (float) (info.negative_count + info.noop_count + info.weak_positive_count) /
                            (float) info.observation_count :
                    0.0f;

            pimpl->last_process_trace.valid = true;
            pimpl->last_process_trace.signature = pimpl->current_process_signature;
            pimpl->last_process_trace.signed_outcome = signed_outcome;
            pimpl->last_process_trace.magnitude = magnitude;
            pimpl->last_process_trace.weak_or_worse_ratio = info.weak_or_worse_ratio;
            pimpl->last_process_trace.bank_size = (uint32_t) pimpl->process_entry_count();
            pimpl->last_process_trace.bank_capacity =
                    std::min<uint32_t>(pimpl->process_params.max_entries, LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES);

            int32_t process_slot = pimpl->find_process_entry_slot(pimpl->current_process_signature);
            pimpl->current_process_entry_slot = process_slot;
            if (process_slot >= 0 && pimpl->process_entries[process_slot].adapter) {
                runtime_lora_write_result process_write_result = {};
                if (pimpl->train_on_adapter(
                            *pimpl->process_entries[process_slot].adapter,
                            embedding,
                            write_features,
                            signed_outcome >= 0.0f ? direction_scale : -direction_scale,
                            false,
                            &process_write_result)) {
                    auto & process_entry = pimpl->process_entries[process_slot];
                    process_entry.info.update_count += 1;
                    process_entry.info.last_used_us = llama_time_us();
                    process_entry.info.last_signed_outcome = signed_outcome;
                    process_entry.info.utility_score = clamp_range(
                            pimpl->process_params.utility_decay * process_entry.info.utility_score +
                                    0.35f * std::max(0.0f, signed_outcome) +
                                    0.25f * info.weak_or_worse_ratio,
                            0.0f,
                            2.0f);
                    pimpl->record_process_signature(process_slot);
                    pimpl->last_process_trace.matched_existing_entry = true;
                    pimpl->last_process_trace.matched_entry_slot = process_slot;
                    pimpl->last_process_trace.creation_reason = LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_EXISTING_ENTRY;
                }
            } else {
                const bool cooldown_ok =
                        info.observation_count >= ledger.last_creation_attempt_observation + pimpl->process_params.creation_cooldown_updates;
                if (!cooldown_ok) {
                    pimpl->last_process_trace.creation_reason = LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_COOLDOWN;
                } else if (pimpl->process_qualifies_for_creation(ledger)) {
                    info.creation_attempt_count += 1;
                    info.last_creation_attempt_us = info.last_observed_us;
                    ledger.last_creation_attempt_observation = info.observation_count;
                    int32_t created_slot = -1;
                    int32_t reason = LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_NONE;
                    if (pimpl->create_process_entry(
                                pimpl->current_process_signature,
                                ledger_slot,
                                &created_slot,
                                &reason)) {
                        pimpl->current_process_entry_slot = created_slot;
                        pimpl->last_process_trace.created_entry = true;
                        pimpl->last_process_trace.created_entry_slot = created_slot;
                        pimpl->last_process_trace.creation_reason = reason;
                        pimpl->last_process_trace.evicted_entry_slot =
                                reason == LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_EVICTED_REPLACEMENT ? created_slot : -1;
                    } else {
                        pimpl->last_process_trace.creation_reason = reason;
                    }
                } else {
                    pimpl->last_process_trace.creation_reason =
                            LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_THRESHOLD_NOT_MET;
                }
                info.last_creation_reason = pimpl->last_process_trace.creation_reason;
            }
        }
    }
    return true;
}

bool llama_active_lora_manager::functional_apply_differential_update(
        int32_t family,
        int32_t proposal_family,
        int32_t replay_mode,
        int32_t snapshot_slot,
        float signed_score_delta,
        float magnitude,
        float robustness_score) {
    if (!pimpl->initialized ||
            family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
            !pimpl->functional_adapters[family]) {
        return false;
    }

    llama_adapter_lora * source_adapter = nullptr;
    if (replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED) {
        if (snapshot_slot < 0 || snapshot_slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY) {
            return false;
        }
        source_adapter = pimpl->functional_snapshots[family][snapshot_slot].adapter.get();
    } else if (replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL ||
               replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED) {
        source_adapter = pimpl->functional_replay_adapters[family].get();
    } else {
        return false;
    }
    if (!source_adapter) {
        return false;
    }

    const float signed_step = clamp_range(signed_score_delta * std::max(0.05f, magnitude), -0.75f, 0.75f);
    if (std::fabs(signed_step) <= 1.0e-5f) {
        return false;
    }

    double sum_sq = 0.0;
    for (auto & it : pimpl->functional_adapters[family]->ab_map) {
        auto src_it = source_adapter->ab_map.find(it.first);
        if (src_it == source_adapter->ab_map.end()) {
            continue;
        }
        auto & dst_weight = it.second;
        const auto & src_weight = src_it->second;
        const size_t size_a = dst_weight.a->ne[0] * dst_weight.a->ne[1];
        const size_t size_b = dst_weight.b->ne[0] * dst_weight.b->ne[1];
        std::vector<float> dst_a(size_a, 0.0f);
        std::vector<float> src_a(size_a, 0.0f);
        std::vector<float> dst_b(size_b, 0.0f);
        std::vector<float> src_b(size_b, 0.0f);
        ggml_backend_tensor_get(dst_weight.a, dst_a.data(), 0, dst_a.size() * sizeof(float));
        ggml_backend_tensor_get(src_weight.a, src_a.data(), 0, src_a.size() * sizeof(float));
        ggml_backend_tensor_get(dst_weight.b, dst_b.data(), 0, dst_b.size() * sizeof(float));
        ggml_backend_tensor_get(src_weight.b, src_b.data(), 0, src_b.size() * sizeof(float));
        for (size_t i = 0; i < size_a; ++i) {
            const float delta = signed_step * (src_a[i] - dst_a[i]);
            dst_a[i] += delta;
            sum_sq += (double) delta * (double) delta;
        }
        for (size_t i = 0; i < size_b; ++i) {
            const float delta = signed_step * (src_b[i] - dst_b[i]);
            dst_b[i] += delta;
            sum_sq += (double) delta * (double) delta;
        }
        ggml_backend_tensor_set(dst_weight.a, dst_a.data(), 0, dst_a.size() * sizeof(float));
        ggml_backend_tensor_set(dst_weight.b, dst_b.data(), 0, dst_b.size() * sizeof(float));
        const float gain_delta = signed_step * (src_weight.gain - dst_weight.gain);
        dst_weight.gain = clamp_range(
                dst_weight.gain + gain_delta,
                pimpl->functional_configs[family].gain_clip_min,
                pimpl->functional_configs[family].gain_clip_max);
        sum_sq += (double) gain_delta * (double) gain_delta;
    }

    const float difference_norm = pimpl->adapter_difference_norm(*pimpl->functional_adapters[family], *source_adapter);
    const float applied_update_norm = std::sqrt((float) sum_sq);
    const uint64_t transaction_step = ++pimpl->runtime_lora_write_count[pimpl->functional_adapters[family].get()];

    auto & diff = pimpl->functional_differential_updates[family];
    diff = {};
    diff.valid = true;
    diff.family = family;
    diff.proposal_family = proposal_family;
    diff.source_snapshot_slot = snapshot_slot;
    diff.signed_score_delta = signed_score_delta;
    diff.magnitude = magnitude;
    diff.lora_difference_norm = difference_norm;
    diff.applied_update_norm = applied_update_norm;
    diff.robustness_score = robustness_score;
    diff.adapter_optimizer_step = transaction_step;
    pimpl->record_functional_signature(family);
    return true;
}

llama_adapter_lora * llama_active_lora_manager::user_personality_adapter() const {
    return pimpl->user_personality_adapter.get();
}

float llama_active_lora_manager::user_personality_scale() const {
    return pimpl->user_personality_stats.adapter_scale;
}

bool llama_active_lora_manager::user_personality_set_attached(bool attached) {
    if (!pimpl->initialized || !pimpl->user_personality_adapter) {
        return false;
    }

    if (attached) {
        pimpl->owner.attach_adapter_runtime(
                pimpl->user_personality_adapter.get(),
                std::max(0.0f, pimpl->user_personality_stats.adapter_scale),
                LLAMA_ADAPTER_LORA_LAYER_USER_PERSONALITY);
    } else {
        pimpl->owner.detach_adapter_runtime(pimpl->user_personality_adapter.get());
    }
    pimpl->user_personality_stats.attached_for_simulation = attached;
    return true;
}
