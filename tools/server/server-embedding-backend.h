#pragma once

#include "server-common.h"

#include "llama.h"

#include <mutex>
#include <string>
#include <vector>

struct server_embedding_backend_config {
    bool enabled = true;
    std::string model_path;
    int32_t gpu_layers = 999;
    int32_t ctx = 4096;
    std::string pooling = "last";
};

server_embedding_backend_config server_embedding_backend_config_from_env();

class server_embedding_backend {
public:
    server_embedding_backend();
    ~server_embedding_backend();

    bool configure(const server_embedding_backend_config & config, std::string * out_error = nullptr);
    bool embed_text(const std::string & text, std::vector<float> & out_embedding, std::string * out_error = nullptr);

    bool configured() const;
    bool ready() const;
    bool degraded() const;
    const std::string & mode_label() const;
    const std::string & error_message() const;

    json health_json() const;

private:
    bool initialize(std::string * out_error);
    void shutdown();
    enum llama_pooling_type parse_pooling_type() const;

    server_embedding_backend_config config_;
    std::string mode_label_;
    std::string error_message_;
    llama_model * model_;
    llama_context * ctx_;
    int32_t n_embd_;
    mutable std::mutex mutex_;
};
