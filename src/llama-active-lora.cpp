#include "llama-active-lora.h"

#include "llama-adapter.h"
#include "llama-context.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
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

static const char * past_bucket_name(size_t bucket) {
    switch (bucket) {
        case LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK:    return "past_week";
        case LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH:   return "past_month";
        case LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER: return "past_quarter";
        case LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR:    return "past_year";
        case LLAMA_MEMORY_LORA_BUCKET_ALL_TIME:     return "all_time";
        default:                                    return "unknown";
    }
}

static uint64_t get_host_free_bytes() {
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return 0;
    }

    size_t free = 0;
    size_t total = 0;
    ggml_backend_dev_memory(cpu_dev, &free, &total);
    return free ? free : total;
}

static uint64_t get_device_free_bytes(const llama_model & model) {
    uint64_t total_free = 0;
    for (auto * dev : model.devices) {
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        total_free += free ? free : total;
    }
    return total_free;
}

static void maybe_add_target(std::vector<ggml_tensor *> & targets, ggml_tensor * tensor) {
    if (!tensor || tensor->ne[1] <= 1) {
        return;
    }

    if (std::find(targets.begin(), targets.end(), tensor) == targets.end()) {
        targets.push_back(tensor);
    }
}

static std::vector<ggml_tensor *> collect_memory_targets(const llama_model & model) {
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

class active_lora_embedder {
public:
    virtual ~active_lora_embedder() = default;
    virtual int32_t type() const = 0;
    virtual bool is_custom() const = 0;
    virtual uint32_t dim() const = 0;
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

static std::unique_ptr<active_lora_embedder> make_embedder(const llama_context & owner, const llama_active_lora_params & params) {
    if (params.embedding_callback != nullptr) {
        return std::make_unique<active_lora_callback_embedder>(
                owner,
                params.embedding_callback,
                params.embedding_callback_user_data,
                params.embedding_dim,
                params.embedding_type);
    }

    switch (params.embedding_type) {
        case LLAMA_ACTIVE_LORA_EMBEDDING_TOKEN_POOL:
            return std::make_unique<active_lora_token_pool_embedder>();
        case LLAMA_ACTIVE_LORA_EMBEDDING_HASH:
        default:
            return std::make_unique<active_lora_hash_embedder>();
    }
}

static float cosine_similarity(const active_lora_embedding & a, const active_lora_embedding & b) {
    if (a.values.empty() || b.values.empty() || a.values.size() != b.values.size() || a.norm == 0.0f || b.norm == 0.0f) {
        return -1.0f;
    }

    float dot = 0.0f;
    for (size_t i = 0; i < a.values.size(); ++i) {
        dot += a.values[i] * b.values[i];
    }

    return dot / (a.norm * b.norm);
}

static float fro_norm(const std::vector<float> & data) {
    double sum_sq = 0.0;
    for (float value : data) {
        sum_sq += (double) value * (double) value;
    }
    return std::sqrt((float) sum_sq);
}

static float compute_effective_scale(float base_scale, uint64_t created_at_us, uint64_t now_us, uint64_t half_life_us) {
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

static std::pair<float, float> adapter_gain_stats(const llama_adapter_lora & adapter) {
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

static void zero_weight(llama_adapter_lora_weight & weight) {
    std::vector<float> data_a(weight.a->ne[0] * weight.a->ne[1], 0.0f);
    std::vector<float> data_b(weight.b->ne[0] * weight.b->ne[1], 0.0f);
    ggml_backend_tensor_set(weight.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_set(weight.b, data_b.data(), 0, data_b.size() * sizeof(float));
    weight.gain = 0.0f;
}

static void zero_adapter(llama_adapter_lora & adapter) {
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

static low_rank_direction read_direction(const llama_adapter_lora_weight & weight) {
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

static void write_direction(llama_adapter_lora_weight & weight, const low_rank_direction & direction) {
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

static float dot_columns(const std::vector<float> & mat, size_t rows, size_t cols, size_t col_a, size_t col_b) {
    float result = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        result += mat[r*cols + col_a] * mat[r*cols + col_b];
    }
    return result;
}

static float column_norm(const std::vector<float> & mat, size_t rows, size_t cols, size_t col) {
    return std::sqrt(std::max(0.0f, dot_columns(mat, rows, cols, col, col)));
}

static void scale_column(std::vector<float> & mat, size_t rows, size_t cols, size_t col, float scale) {
    for (size_t r = 0; r < rows; ++r) {
        mat[r*cols + col] *= scale;
    }
}

static void subtract_column(std::vector<float> & mat, size_t rows, size_t cols, size_t dst_col, size_t src_col, float scale) {
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

static thin_qr_result thin_qr(const std::vector<float> & mat, size_t rows, size_t cols) {
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

static std::vector<float> small_mat_mul_r_rt(const std::vector<float> & lhs, const std::vector<float> & rhs, size_t n) {
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

static std::vector<float> small_mat_transpose_mul(const std::vector<float> & mat, size_t n) {
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

static std::pair<std::vector<float>, std::vector<float>> jacobi_eigen(std::vector<float> mat, size_t n) {
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

static std::vector<float> multiply_small_matrix_vector(const std::vector<float> & mat, size_t n, const std::vector<float> & vec) {
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

static void normalize_vector(std::vector<float> & vec) {
    const float norm = fro_norm(vec);
    if (norm > DIRECTION_EPS) {
        for (float & value : vec) {
            value /= norm;
        }
    }
}

static bool merge_directions(
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

static void normalize_active_weight(
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
    std::vector<ggml_tensor *> targets;
    active_lora_embedding last_embedding;
    uint64_t updates_seen = 0;
    uint64_t active_started_at_us = 0;
    uint64_t active_last_update_us = 0;
    uint32_t active_rollover_version = 0;
    bool initialized = false;
    bool past_initialized = false;

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
            rank_limit = std::min<uint64_t>(rank_limit, host_budget_bytes / host_bytes_per_rank);
        }
        if (device_bytes_per_rank > 0) {
            if (device_budget_bytes == 0) {
                return 0;
            }
            rank_limit = std::min<uint64_t>(rank_limit, device_budget_bytes / device_bytes_per_rank);
        }

        return static_cast<uint32_t>(rank_limit);
    }

    bool create_runtime_adapter(
            std::unique_ptr<llama_adapter_lora> & out,
            uint32_t rank,
            const std::string & name_prefix,
            float initial_scale) {
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
        owner.attach_adapter_runtime(out.get(), initial_scale);
        return true;
    }

    void set_adapter_scale(llama_adapter_lora * adapter_ptr, float scale) {
        if (!owner.loras) {
            owner.loras = std::make_unique<llama_adapter_loras>();
        }
        (*owner.loras)[adapter_ptr] = scale;
    }

    bool train_on_span(const active_lora_embedding & embedding, const llama_token * tokens, size_t n_tokens) {
        if (embedding.values.empty() || n_tokens == 0 || !adapter) {
            return false;
        }

        const float token_scale = 1.0f / std::max<size_t>(1, n_tokens);
        const float update_scale = params.learning_rate > 0.0f ? params.learning_rate : 1.0e-4f;

        for (auto & it : adapter->ab_map) {
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
            ggml_backend_tensor_get(tensor_a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_get(tensor_b, data_b.data(), 0, data_b.size() * sizeof(float));

            const size_t slot = updates_seen % ne1_a;
            for (size_t i = 0; i < ne0_a; ++i) {
                const float emb = embedding.values[i % embedding.values.size()];
                data_a[slot*ne0_a + i] += update_scale * emb;
            }

            for (size_t j = 0; j < ne1_b; ++j) {
                const float tok = static_cast<float>(tokens[j % n_tokens] % 1024) * token_scale;
                data_b[j*ne0_b + slot] += update_scale * tok;
            }

            ggml_backend_tensor_set(tensor_a, data_a.data(), 0, data_a.size() * sizeof(float));
            ggml_backend_tensor_set(tensor_b, data_b.data(), 0, data_b.size() * sizeof(float));
            normalize_active_weight(it.second, update_scale, params.gain_decay, params.gain_max);
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
            zero_adapter(*adapter);
        }

        last_embedding = {};
        updates_seen = 0;
        stats.rollover_ready = false;
        stats.updates_applied = 0;
        stats.tokens_ingested = 0;
        stats.gain_mean = 0.0f;
        stats.gain_max = 0.0f;
        active_started_at_us = now_us;
        active_last_update_us = now_us;
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
        if (bucket.stats.source_window_start_us == 0 || !bucket.stats.source_window_start_us || target_weight == 0.0f) {
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
                set_adapter_scale(buckets[bucket].adapter.get(), scale);
            }
            past_stats.buckets[bucket] = buckets[bucket].stats;
        }
    }

    uint64_t compute_pending_job_mask(uint64_t now_us) const {
        if (!past_initialized) {
            return 0;
        }

        uint64_t mask = 0;
        if (stats.rollover_ready) {
            mask |= (1ull << ACTIVE_TO_WEEK_JOB);
        } else if (stats.updates_applied > 0 && past_params.condensation_period_us[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK] > 0 &&
                now_us >= active_started_at_us + past_params.condensation_period_us[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK]) {
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
        if (pimpl->owner.loras) {
            pimpl->owner.loras->erase(pimpl->adapter.get());
        }
        const_cast<llama_model &>(pimpl->owner.model).loras.erase(pimpl->adapter.get());
    }
    for (auto & bucket : pimpl->buckets) {
        if (bucket.adapter) {
            if (pimpl->owner.loras) {
                pimpl->owner.loras->erase(bucket.adapter.get());
            }
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
    const uint64_t host_budget_bytes = static_cast<uint64_t>(host_free_bytes * params.host_memory_ratio);
    const uint64_t device_budget_bytes = static_cast<uint64_t>(device_free_bytes * params.device_memory_ratio);
    const uint32_t selected_rank = pimpl->compute_rank(host_budget_bytes, device_budget_bytes, params.max_rank);
    if (selected_rank < params.min_rank) {
        LLAMA_LOG_ERROR("%s: failed to plan Active LoRA rank (planned %u, min %u)\n",
                __func__, selected_rank, params.min_rank);
        return false;
    }

    if (!pimpl->adapter) {
        if (!pimpl->create_runtime_adapter(pimpl->adapter, selected_rank, "memory.active", params.adapter_scale)) {
            return false;
        }
    }

    pimpl->stats.enabled = true;
    pimpl->stats.rollover_ready = false;
    pimpl->stats.selected_rank = selected_rank;
    pimpl->stats.updates_applied = 0;
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
    pimpl->initialized = true;
    pimpl->reset_active_stage(llama_time_us());

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
        const uint64_t host_budget_bytes = static_cast<uint64_t>(host_free_bytes * params.host_memory_ratio[bucket]);
        const uint64_t device_budget_bytes = static_cast<uint64_t>(device_free_bytes * params.device_memory_ratio[bucket]);
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
                        0.0f)) {
                return false;
            }
        }

        pimpl->buckets[bucket].stats = {};
        pimpl->buckets[bucket].stats.selected_rank = selected_rank;
        pimpl->buckets[bucket].stats.host_budget_bytes = host_budget_bytes;
        pimpl->buckets[bucket].stats.device_budget_bytes = device_budget_bytes;
        pimpl->buckets[bucket].stats.base_scale = params.base_scale[bucket];
        pimpl->set_adapter_scale(pimpl->buckets[bucket].adapter.get(), 0.0f);
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
    if (!pimpl->initialized || !tokens || n_tokens == 0) {
        return false;
    }

    const active_lora_embedding embedding = pimpl->embedder->embed(tokens, n_tokens);
    const float similarity = cosine_similarity(embedding, pimpl->last_embedding);
    if (similarity > 0.995f) {
        LLAMA_LOG_INFO("%s: skipping Active LoRA write for highly redundant span (cos=%.4f)\n", __func__, similarity);
        return true;
    }

    if (!pimpl->train_on_span(embedding, tokens, n_tokens)) {
        return false;
    }

    pimpl->last_embedding = embedding;
    pimpl->stats.updates_applied++;
    pimpl->stats.tokens_ingested += n_tokens;
    pimpl->updates_seen++;
    pimpl->stats.rollover_ready = pimpl->updates_seen >= pimpl->params.max_updates_before_rollover;
    pimpl->active_last_update_us = llama_time_us();
    pimpl->update_active_gain_stats();

    LLAMA_LOG_INFO("%s: Active LoRA write accepted (%zu tokens, update=%u, rollover_ready=%s)\n",
            __func__, n_tokens, pimpl->stats.updates_applied, pimpl->stats.rollover_ready ? "true" : "false");

    return true;
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
