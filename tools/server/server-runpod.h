#pragma once

#include "server-deepseek.h"

#include <string>

enum class runpod_inference_role {
    disabled = 0,
    host = 1,
    node = 2,
};

enum class host_inference_mode {
    standard = 0,
    experimental = 1,
};

struct runpod_inference_runtime_config {
    bool enabled = false;
    runpod_inference_role role = runpod_inference_role::disabled;
    std::string role_label = "disabled";
    host_inference_mode host_mode = host_inference_mode::standard;
    std::string host_mode_label = "standard";
    std::string relay_url;
    std::string auth_token;
    int32_t timeout_ms = 1800000;
    std::string requested_model = "google/gemma-4-31B-it";
    std::string resolved_model = "google/gemma-4-31B-it";
    std::string serving_dtype = "q8_0";
    std::string kv_profile = "q8_0_64k";
    int32_t context_limit = 65536;
    int32_t default_max_tokens = 4096;
    std::string node_id = "runpod-a100";
    std::string config_error;
};

runpod_inference_runtime_config runpod_inference_runtime_config_from_env();
void configure_runpod_inference_runtime(const runpod_inference_runtime_config & config);
const runpod_inference_runtime_config & runpod_inference_runtime_config_instance();

bool runpod_inference_local_provider_required(const runpod_inference_runtime_config & config);
bool runpod_inference_host_relay_enabled(const runpod_inference_runtime_config & config);
bool runpod_inference_node_execution_enabled(const runpod_inference_runtime_config & config);
bool host_inference_experimental_enabled(const runpod_inference_runtime_config & config);
bool host_inference_standard_enabled(const runpod_inference_runtime_config & config);

json runpod_build_health_json(const runpod_inference_runtime_config & config);
json runpod_build_models_json(const runpod_inference_runtime_config & config);

bool runpod_validate_bearer_auth(
        const runpod_inference_runtime_config & config,
        const std::string & authorization_header,
        std::string * out_error = nullptr);

bool runpod_relay_inference_request(
        const runpod_inference_runtime_config & config,
        const json & request_body,
        deepseek_chat_result * out_result,
        json * out_error,
        json * out_meta = nullptr);
