#include "server-embedding-backend.h"

#include "server-runtime.h"

#include <algorithm>
#include <array>

namespace {

static int32_t env_to_int(const char * name, int32_t default_value) {
    if (const char * value = std::getenv(name)) {
        const int parsed = std::atoi(value);
        if (parsed > 0) {
            return parsed;
        }
    }
    return default_value;
}

static bool env_to_bool(const char * name, bool default_value) {
    if (const char * value = std::getenv(name)) {
        const std::string parsed = value;
        if (parsed == "0" || parsed == "false" || parsed == "FALSE") {
            return false;
        }
        if (parsed == "1" || parsed == "true" || parsed == "TRUE") {
            return true;
        }
    }
    return default_value;
}

static std::string model_metadata_value(const llama_model * model, const char * key) {
    if (!model || !key) {
        return {};
    }

    std::array<char, 128> key_buf = {};
    std::array<char, 256> val_buf = {};
    const int32_t count = llama_model_meta_count(model);
    for (int32_t i = 0; i < count; ++i) {
        if (llama_model_meta_key_by_index(model, i, key_buf.data(), key_buf.size()) < 0) {
            continue;
        }
        if (std::string(key_buf.data()) != key) {
            continue;
        }
        if (llama_model_meta_val_str_by_index(model, i, val_buf.data(), val_buf.size()) < 0) {
            return {};
        }
        return std::string(val_buf.data());
    }

    return {};
}

} // namespace

server_embedding_backend_config server_embedding_backend_config_from_env() {
    server_embedding_backend_config config;
    config.enabled = env_to_bool("VICUNA_EMOTIVE_EMBED_ENABLED", true);
    if (const char * value = std::getenv("VICUNA_EMOTIVE_EMBED_MODEL")) {
        config.model_path = value;
    }
    config.gpu_layers = env_to_int("VICUNA_EMOTIVE_EMBED_GPU_LAYERS", config.gpu_layers);
    config.ctx = env_to_int("VICUNA_EMOTIVE_EMBED_CTX", config.ctx);
    if (const char * value = std::getenv("VICUNA_EMOTIVE_EMBED_POOLING")) {
        config.pooling = value;
    }
    return config;
}

server_embedding_backend::server_embedding_backend() :
        mode_label_("lexical_only"),
        model_(nullptr),
        ctx_(nullptr),
        n_embd_(0) {
}

server_embedding_backend::~server_embedding_backend() {
    shutdown();
}

bool server_embedding_backend::configure(const server_embedding_backend_config & config, std::string * out_error) {
    std::lock_guard<std::mutex> lock(mutex_);

    shutdown();
    config_ = config;
    mode_label_ = "lexical_only";
    error_message_.clear();

    if (!config_.enabled || config_.model_path.empty()) {
        return true;
    }

    if (!initialize(out_error)) {
        mode_label_ = "lexical_only";
        return false;
    }

    return true;
}

bool server_embedding_backend::initialize(std::string * out_error) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.gpu_layers;
    model_ = llama_model_load_from_file(config_.model_path.c_str(), model_params);
    if (!model_) {
        error_message_ = "failed to load embedding model";
        if (out_error) {
            *out_error = error_message_;
        }
        return false;
    }

    const std::string architecture = model_metadata_value(model_, "general.architecture");
    if (architecture != "qwen3") {
        error_message_ = architecture.empty()
                ? "embedding model is missing general.architecture metadata"
                : "embedding model architecture must be qwen3";
        if (out_error) {
            *out_error = error_message_;
        }
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.n_ctx = config_.ctx;
    ctx_params.n_batch = config_.ctx;
    ctx_params.n_ubatch = config_.ctx;
    ctx_params.n_seq_max = 1;
    ctx_params.pooling_type = parse_pooling_type();

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        error_message_ = "failed to initialize embedding context";
        if (out_error) {
            *out_error = error_message_;
        }
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    n_embd_ = llama_model_n_embd_out(model_);
    mode_label_ = "embedding_fused";
    return true;
}

void server_embedding_backend::shutdown() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    n_embd_ = 0;
}

enum llama_pooling_type server_embedding_backend::parse_pooling_type() const {
    if (config_.pooling == "mean") {
        return LLAMA_POOLING_TYPE_MEAN;
    }
    if (config_.pooling == "cls") {
        return LLAMA_POOLING_TYPE_CLS;
    }
    if (config_.pooling == "none") {
        return LLAMA_POOLING_TYPE_NONE;
    }
    return LLAMA_POOLING_TYPE_LAST;
}

bool server_embedding_backend::embed_text(const std::string & text, std::vector<float> & out_embedding, std::string * out_error) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!ready()) {
        return false;
    }

    std::vector<llama_token> tokens = server_tokenize(ctx_, text, true, true);
    if (tokens.empty()) {
        out_embedding.assign((size_t) std::max(1, n_embd_), 0.0f);
        return true;
    }

    if ((int32_t) tokens.size() > config_.ctx) {
        tokens.resize((size_t) config_.ctx);
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        server_batch_add(batch, tokens[i], (llama_pos) i, {0}, i + 1 == tokens.size());
    }

    const int decode_status = llama_decode(ctx_, batch);
    llama_batch_free(batch);

    if (decode_status < 0) {
        error_message_ = "embedding decode failed";
        if (out_error) {
            *out_error = error_message_;
        }
        return false;
    }

    const float * embd = llama_get_embeddings_seq(ctx_, 0);
    if (!embd) {
        embd = llama_get_embeddings_ith(ctx_, (int32_t) tokens.size() - 1);
    }
    if (!embd) {
        error_message_ = "embedding output was unavailable";
        if (out_error) {
            *out_error = error_message_;
        }
        return false;
    }

    out_embedding.resize((size_t) n_embd_);
    server_embd_normalize(embd, out_embedding.data(), n_embd_, 2);
    return true;
}

bool server_embedding_backend::configured() const {
    return !config_.model_path.empty() && config_.enabled;
}

bool server_embedding_backend::ready() const {
    return model_ != nullptr && ctx_ != nullptr && n_embd_ > 0;
}

bool server_embedding_backend::degraded() const {
    return !ready();
}

const std::string & server_embedding_backend::mode_label() const {
    return mode_label_;
}

const std::string & server_embedding_backend::error_message() const {
    return error_message_;
}

json server_embedding_backend::health_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        {"configured", configured()},
        {"ready", ready()},
        {"mode", mode_label_},
        {"model_path", config_.model_path},
        {"pooling", config_.pooling},
        {"ctx", config_.ctx},
        {"gpu_layers", config_.gpu_layers},
        {"error", error_message_},
    };
}
