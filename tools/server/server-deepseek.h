#pragma once

#include "server-common.h"

#include <functional>
#include <string>
#include <vector>

struct deepseek_runtime_config {
    bool enabled = false;
    std::string api_key;
    std::string base_url = "https://api.deepseek.com";
    std::string model = "deepseek-reasoner";
    int32_t timeout_ms = 60000;
};

struct deepseek_stream_observer {
    std::function<void(const std::string &)> on_reasoning_delta;
    std::function<void(const std::string &)> on_content_delta;
    std::function<void(const std::string &)> on_runtime_event;
};

struct deepseek_tool_call {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct deepseek_chat_result {
    std::string content;
    std::string reasoning_content;
    std::string finish_reason = "stop";
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
    std::vector<deepseek_tool_call> tool_calls;
    json emotive_trace = nullptr;
};

deepseek_runtime_config deepseek_runtime_config_from_env();
bool deepseek_validate_runtime_config(const deepseek_runtime_config & config, json * out_error = nullptr);
bool deepseek_complete_chat(
        const deepseek_runtime_config & config,
        const json & body,
        deepseek_chat_result * out_result,
        const deepseek_stream_observer * observer = nullptr,
        json * out_error = nullptr);

json deepseek_build_health_json(const deepseek_runtime_config & config);
json deepseek_build_models_json(const deepseek_runtime_config & config);

json deepseek_format_chat_completion_response(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result,
        const std::string & completion_id);
json deepseek_format_text_completion_response(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result,
        const std::string & completion_id);
json deepseek_format_responses_response(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result);
std::vector<json> deepseek_format_chat_completion_stream(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result,
        const std::string & completion_id,
        bool include_usage);
