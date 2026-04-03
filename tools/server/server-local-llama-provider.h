#pragma once

#include "server-common.h"
#include "server-deepseek.h"
#include "server-emotive-runtime.h"
#include "server-runpod.h"
#include "server-runtime-control.h"
#include "server-runtime.h"

enum class server_provider_kind {
    deepseek,
};

struct server_provider_runtime_config {
    server_provider_kind kind = server_provider_kind::deepseek;
    std::string provider_label = "deepseek";
    std::string resolved_model;
};

struct server_local_llama_provider_config {
    bool enabled = false;
};

server_provider_runtime_config server_provider_runtime_config_from_env(
        const server_runtime_params & params,
        const runpod_inference_runtime_config & runpod_config);

server_local_llama_provider_config server_local_llama_provider_config_from_env(
        const server_runtime_params & params,
        const server_provider_runtime_config & provider_config);

class server_local_llama_provider {
public:
    bool configure(const server_local_llama_provider_config & config, std::string * out_error = nullptr);
    bool ready() const;
    const std::string & resolved_model() const;
    json health_json() const;
    json models_json() const;
    bool complete_chat(
            const json & body,
            deepseek_chat_result * out_result,
            const deepseek_stream_observer * observer,
            json * out_error,
            const deepseek_request_trace * trace,
            server_llama_runtime_control_state * runtime_control_state,
            server_emotive_turn_builder * turn_builder);

private:
    server_local_llama_provider_config config_ = {};
    std::string resolved_model_ = "disabled";
};
