#include "server-local-llama-provider.h"

server_provider_runtime_config server_provider_runtime_config_from_env(
        const server_runtime_params & params,
        const runpod_inference_runtime_config & runpod_config) {
    (void) params;
    server_provider_runtime_config config = {};
    config.resolved_model = runpod_inference_host_relay_enabled(runpod_config) ?
            runpod_config.resolved_model :
            std::string("deepseek-chat");
    return config;
}

server_local_llama_provider_config server_local_llama_provider_config_from_env(
        const server_runtime_params & params,
        const server_provider_runtime_config & provider_config) {
    (void) params;
    (void) provider_config;
    return {};
}

bool server_local_llama_provider::configure(
        const server_local_llama_provider_config & config,
        std::string * out_error) {
    config_ = config;
    if (out_error) {
        out_error->clear();
    }
    return true;
}

bool server_local_llama_provider::ready() const {
    return false;
}

const std::string & server_local_llama_provider::resolved_model() const {
    return resolved_model_;
}

json server_local_llama_provider::health_json() const {
    return {
        {"enabled", false},
        {"ready", false},
        {"status", "removed"},
        {"message", "Local pod inference moved to a separate repository"},
    };
}

json server_local_llama_provider::models_json() const {
    return json::array();
}

bool server_local_llama_provider::complete_chat(
        const json & body,
        deepseek_chat_result * out_result,
        const deepseek_stream_observer * observer,
        json * out_error,
        const deepseek_request_trace * trace,
        server_llama_runtime_control_state * runtime_control_state,
        server_emotive_turn_builder * turn_builder) {
    (void) body;
    (void) out_result;
    (void) observer;
    (void) trace;
    (void) runtime_control_state;
    (void) turn_builder;
    if (out_error) {
        *out_error = format_error_response(
                "Local pod inference is no longer supported in this repository",
                ERROR_TYPE_NOT_SUPPORTED);
    }
    return false;
}
