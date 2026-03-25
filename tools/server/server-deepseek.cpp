#include "server-deepseek.h"
#include "server-runtime.h"

#include "common/http.h"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <stdexcept>

static std::string trim_ascii_copy_local(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return value.substr(begin, end - begin);
}

static std::string json_content_to_text(const json & content) {
    if (content.is_null()) {
        return std::string();
    }
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        throw std::invalid_argument("message content must be a string or text-part array");
    }

    std::string text;
    for (const auto & part : content) {
        if (part.is_string()) {
            text += part.get<std::string>();
            continue;
        }
        if (!part.is_object()) {
            continue;
        }
        const std::string type = json_value(part, "type", std::string());
        if (type.empty() || type == "text" || type == "input_text") {
            text += json_value(part, "text", std::string());
        }
    }

    return text;
}

static json deepseek_normalize_outbound_tool_calls(const json & tool_calls) {
    if (!tool_calls.is_array()) {
        throw std::invalid_argument("\"tool_calls\" must be an array when present");
    }

    json normalized = json::array();
    for (const auto & item : tool_calls) {
        if (!item.is_object()) {
            throw std::invalid_argument("each tool call must be an object");
        }

        const std::string type = json_value(item, "type", std::string("function"));
        if (!type.empty() && type != "function") {
            throw std::invalid_argument("only function tool calls are supported");
        }
        if (!item.contains("function") || !item.at("function").is_object()) {
            throw std::invalid_argument("each tool call must include a function object");
        }

        const json & function = item.at("function");
        const std::string name = json_value(function, "name", std::string());
        if (name.empty()) {
            throw std::invalid_argument("each tool call must include a non-empty function name");
        }

        json normalized_call = {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"arguments", function.contains("arguments") ? function.at("arguments") : json("{}")},
            }},
        };
        const std::string id = json_value(item, "id", std::string());
        if (!id.empty()) {
            normalized_call["id"] = id;
        }
        normalized.push_back(std::move(normalized_call));
    }

    return normalized;
}

static std::string deepseek_chat_completions_url(const deepseek_runtime_config & config) {
    std::string url = trim_ascii_copy_local(config.base_url);
    if (url.empty()) {
        url = "https://api.deepseek.com";
    }
    if (url.size() >= std::string("/chat/completions").size() &&
            url.compare(url.size() - std::string("/chat/completions").size(),
                        std::string("/chat/completions").size(),
                        "/chat/completions") == 0) {
        return url;
    }
    if (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url + "/chat/completions";
}

static json deepseek_build_provider_messages(const json & body) {
    json messages = json::array();
    if (body.contains("messages")) {
        if (!body.at("messages").is_array()) {
            throw std::invalid_argument("\"messages\" must be an array");
        }

        for (const auto & item : body.at("messages")) {
            if (!item.is_object()) {
                throw std::invalid_argument("each message must be an object");
            }

            const std::string role = json_value(item, "role", std::string());
            if (role.empty()) {
                throw std::invalid_argument("each message must include a role");
            }

            json provider_message = {
                {"role", role},
            };
            if (item.contains("content")) {
                provider_message["content"] = json_content_to_text(item.at("content"));
            } else if (item.contains("tool_calls")) {
                provider_message["content"] = nullptr;
            } else {
                throw std::invalid_argument("each message must include content unless it only carries tool_calls");
            }

            if (item.contains("tool_calls")) {
                provider_message["tool_calls"] = deepseek_normalize_outbound_tool_calls(item.at("tool_calls"));
            }
            if (role == "assistant" && item.contains("tool_calls")) {
                const auto reasoning_it = item.find("reasoning_content");
                if (reasoning_it != item.end() && reasoning_it->is_string()) {
                    provider_message["reasoning_content"] = reasoning_it->get<std::string>();
                }
            }
            const std::string tool_call_id = json_value(item, "tool_call_id", std::string());
            if (!tool_call_id.empty()) {
                provider_message["tool_call_id"] = tool_call_id;
            }
            const std::string name = json_value(item, "name", std::string());
            if (!name.empty()) {
                provider_message["name"] = name;
            }

            messages.push_back(std::move(provider_message));
        }
    } else if (body.contains("prompt")) {
        messages.push_back(json{
            {"role", "user"},
            {"content", json_content_to_text(body.at("prompt"))},
        });
    } else {
        throw std::invalid_argument("request must include either \"messages\" or \"prompt\"");
    }

    if (messages.empty()) {
        throw std::invalid_argument("request must include at least one non-empty message");
    }

    return messages;
}

static constexpr int64_t VICUNA_OUTBOUND_MAX_TOKENS = 1024;

static json deepseek_build_provider_body(const deepseek_runtime_config & config, const json & body) {
    json provider_body = {
        {"model", config.model},
        {"messages", deepseek_build_provider_messages(body)},
        {"stream", true},
        {"max_tokens", VICUNA_OUTBOUND_MAX_TOKENS},
    };

    if (body.contains("temperature")) {
        provider_body["temperature"] = body.at("temperature");
    }
    if (body.contains("top_p")) {
        provider_body["top_p"] = body.at("top_p");
    }
    if (body.contains("presence_penalty")) {
        provider_body["presence_penalty"] = body.at("presence_penalty");
    }
    if (body.contains("frequency_penalty")) {
        provider_body["frequency_penalty"] = body.at("frequency_penalty");
    }
    if (body.contains("tools")) {
        if (!body.at("tools").is_array()) {
            throw std::invalid_argument("\"tools\" must be an array");
        }
        provider_body["tools"] = body.at("tools");
    }
    if (body.contains("tool_choice")) {
        provider_body["tool_choice"] = body.at("tool_choice");
    }
    if (body.contains("parallel_tool_calls")) {
        provider_body["parallel_tool_calls"] = body.at("parallel_tool_calls");
    }
    if (body.contains("response_format")) {
        provider_body["response_format"] = body.at("response_format");
    }
    if (body.contains("thinking")) {
        provider_body["thinking"] = body.at("thinking");
    }

    return provider_body;
}

static bool deepseek_finalize_tool_calls(deepseek_chat_result * out_result, json * out_error) {
    for (auto & tool_call : out_result->tool_calls) {
        if (tool_call.name.empty()) {
            if (out_error) {
                *out_error = format_error_response(
                        "DeepSeek provider returned a tool call without a function name.",
                        ERROR_TYPE_SERVER);
            }
            return false;
        }

        if (tool_call.arguments_json.empty()) {
            tool_call.arguments_json = "{}";
        }

        try {
            const json parsed = json::parse(tool_call.arguments_json);
            if (!parsed.is_object()) {
                if (out_error) {
                    *out_error = format_error_response(
                            "DeepSeek provider returned tool arguments that were not a JSON object.",
                            ERROR_TYPE_SERVER);
                }
                return false;
            }
            tool_call.arguments_json = parsed.dump();
        } catch (const std::exception & e) {
            if (out_error) {
                *out_error = format_error_response(
                        std::string("DeepSeek provider returned malformed tool arguments JSON: ") + e.what(),
                        ERROR_TYPE_SERVER);
            }
            return false;
        }

        if (tool_call.id.empty()) {
            tool_call.id = "call_" + random_string();
        }
    }

    return true;
}

static json deepseek_message_json(const deepseek_chat_result & result) {
    json message = {
        {"role", "assistant"},
        {"content", result.content.empty() && !result.tool_calls.empty() ? json(nullptr) : json(result.content)},
    };
    if (!result.reasoning_content.empty()) {
        message["reasoning_content"] = result.reasoning_content;
    }
    if (!result.tool_calls.empty()) {
        json tool_calls = json::array();
        for (const auto & tool_call : result.tool_calls) {
            tool_calls.push_back({
                {"id", tool_call.id},
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", tool_call.arguments_json},
                }},
            });
        }
        message["tool_calls"] = std::move(tool_calls);
    }
    return message;
}

static void deepseek_apply_usage(const json & response_body, deepseek_chat_result * out_result) {
    if (!response_body.contains("usage") || !response_body.at("usage").is_object()) {
        return;
    }

    const json & usage = response_body.at("usage");
    out_result->prompt_tokens = json_value(usage, "prompt_tokens", out_result->prompt_tokens);
    out_result->completion_tokens = json_value(usage, "completion_tokens", out_result->completion_tokens);
}

static void deepseek_append_delta_text(
        const json & object,
        const char * key,
        std::string * out_text,
        const std::function<void(const std::string &)> & observer) {
    if (!object.contains(key)) {
        return;
    }

    const std::string fragment = json_content_to_text(object.at(key));
    if (fragment.empty()) {
        return;
    }

    *out_text += fragment;
    if (observer) {
        observer(fragment);
    }
}

static bool deepseek_merge_tool_calls(
        const json & tool_calls,
        deepseek_chat_result * out_result,
        bool finalize,
        json * out_error) {
    if (!tool_calls.is_array()) {
        if (out_error) {
            *out_error = format_error_response(
                    "DeepSeek provider returned a non-array tool_calls payload.",
                    ERROR_TYPE_SERVER);
        }
        return false;
    }

    for (size_t i = 0; i < tool_calls.size(); ++i) {
        const auto & item = tool_calls.at(i);
        if (!item.is_object()) {
            if (out_error) {
                *out_error = format_error_response(
                        "DeepSeek provider returned a malformed tool call entry.",
                        ERROR_TYPE_SERVER);
            }
            return false;
        }

        const size_t index = item.contains("index") && item.at("index").is_number_unsigned() ?
                item.at("index").get<size_t>() :
                i;
        while (out_result->tool_calls.size() <= index) {
            out_result->tool_calls.push_back(deepseek_tool_call());
        }

        deepseek_tool_call & target = out_result->tool_calls[index];
        const std::string id = json_value(item, "id", std::string());
        if (!id.empty()) {
            target.id = id;
        }

        if (item.contains("type")) {
            const std::string type = json_value(item, "type", std::string("function"));
            if (!type.empty() && type != "function") {
                if (out_error) {
                    *out_error = format_error_response(
                            "DeepSeek provider returned an unsupported tool call type.",
                            ERROR_TYPE_SERVER);
                }
                return false;
            }
        }

        const json function = item.value("function", json::object());
        if (!function.is_object()) {
            if (out_error) {
                *out_error = format_error_response(
                        "DeepSeek provider returned a tool call without a function object.",
                        ERROR_TYPE_SERVER);
            }
            return false;
        }

        if (function.contains("name")) {
            const std::string name_fragment = json_value(function, "name", std::string());
            if (!name_fragment.empty()) {
                target.name += name_fragment;
            }
        }
        if (function.contains("arguments")) {
            if (function.at("arguments").is_string()) {
                target.arguments_json += function.at("arguments").get<std::string>();
            } else if (target.arguments_json.empty()) {
                target.arguments_json = function.at("arguments").dump();
            } else {
                target.arguments_json = function.at("arguments").dump();
            }
        }
    }

    if (finalize) {
        return deepseek_finalize_tool_calls(out_result, out_error);
    }

    return true;
}

static bool deepseek_parse_response_json(
        const json & response_body,
        deepseek_chat_result * out_result,
        const deepseek_stream_observer * observer,
        json * out_error) {
    if (!response_body.contains("choices") || !response_body.at("choices").is_array() ||
            response_body.at("choices").empty()) {
        if (out_error) {
            *out_error = format_error_response(
                    "DeepSeek provider response did not include any choices.",
                    ERROR_TYPE_SERVER);
        }
        return false;
    }

    const json & choice = response_body.at("choices").at(0);
    const json & payload = choice.contains("delta") && choice.at("delta").is_object() ?
            choice.at("delta") :
            choice.value("message", json::object());
    if (!payload.is_object()) {
        if (out_error) {
            *out_error = format_error_response(
                    "DeepSeek provider response did not include a usable message delta.",
                    ERROR_TYPE_SERVER);
        }
        return false;
    }

    deepseek_append_delta_text(
            payload,
            "reasoning_content",
            &out_result->reasoning_content,
            observer ? observer->on_reasoning_delta : std::function<void(const std::string &)>());
    deepseek_append_delta_text(
            payload,
            "content",
            &out_result->content,
            observer ? observer->on_content_delta : std::function<void(const std::string &)>());
    const bool finalize_tool_calls = !choice.contains("delta");
    if (payload.contains("tool_calls") &&
            !deepseek_merge_tool_calls(payload.at("tool_calls"), out_result, finalize_tool_calls, out_error)) {
        return false;
    }

    const std::string finish_reason = json_value(choice, "finish_reason", std::string());
    if (!finish_reason.empty()) {
        out_result->finish_reason = finish_reason;
    }

    deepseek_apply_usage(response_body, out_result);
    return true;
}

static bool deepseek_consume_sse_event(
        const std::string & raw_event,
        deepseek_chat_result * out_result,
        const deepseek_stream_observer * observer,
        json * out_error) {
    const std::string event = trim_ascii_copy_local(raw_event);
    if (event.empty() || event == "[DONE]") {
        return true;
    }

    try {
        return deepseek_parse_response_json(json::parse(event), out_result, observer, out_error);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = format_error_response(
                    std::string("DeepSeek provider stream parse failed: ") + e.what(),
                    ERROR_TYPE_SERVER);
        }
        return false;
    }
}

deepseek_runtime_config deepseek_runtime_config_from_env() {
    deepseek_runtime_config config;

    if (const char * value = std::getenv("VICUNA_DEEPSEEK_API_KEY")) {
        config.api_key = value;
    }
    if (const char * value = std::getenv("VICUNA_DEEPSEEK_BASE_URL")) {
        config.base_url = value;
    }
    if (const char * value = std::getenv("VICUNA_DEEPSEEK_MODEL")) {
        config.model = value;
    }
    if (const char * value = std::getenv("VICUNA_DEEPSEEK_TIMEOUT_MS")) {
        const int timeout_ms = std::atoi(value);
        if (timeout_ms > 0) {
            config.timeout_ms = timeout_ms;
        }
    }

    config.enabled = !trim_ascii_copy_local(config.api_key).empty();
    return config;
}

bool deepseek_validate_runtime_config(const deepseek_runtime_config & config, json * out_error) {
    if (!config.enabled) {
        if (out_error) {
            *out_error = format_error_response(
                    "DeepSeek provider mode is not enabled. Set VICUNA_DEEPSEEK_API_KEY.",
                    ERROR_TYPE_NOT_SUPPORTED);
        }
        return false;
    }

    if (trim_ascii_copy_local(config.api_key).empty()) {
        if (out_error) {
            *out_error = format_error_response(
                    "DeepSeek provider mode requires VICUNA_DEEPSEEK_API_KEY.",
                    ERROR_TYPE_AUTHENTICATION);
        }
        return false;
    }

    if (trim_ascii_copy_local(config.model).empty()) {
        if (out_error) {
            *out_error = format_error_response(
                    "DeepSeek provider mode requires a non-empty model.",
                    ERROR_TYPE_INVALID_REQUEST);
        }
        return false;
    }

    if (config.timeout_ms <= 0) {
        if (out_error) {
            *out_error = format_error_response(
                    "DeepSeek provider mode requires a positive timeout.",
                    ERROR_TYPE_INVALID_REQUEST);
        }
        return false;
    }

    return true;
}

bool deepseek_complete_chat(
        const deepseek_runtime_config & config,
        const json & body,
        deepseek_chat_result * out_result,
        const deepseek_stream_observer * observer,
        json * out_error) {
    if (!out_result) {
        if (out_error) {
            *out_error = format_error_response("DeepSeek result output must not be null.", ERROR_TYPE_SERVER);
        }
        return false;
    }

    *out_result = deepseek_chat_result();

    json config_error;
    if (!deepseek_validate_runtime_config(config, &config_error)) {
        if (out_error) {
            *out_error = config_error;
        }
        return false;
    }

    json provider_body;
    try {
        provider_body = deepseek_build_provider_body(config, body);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
        }
        return false;
    }

    try {
        const std::string url = deepseek_chat_completions_url(config);
        auto [client, parts] = server_http_client(url);
        client.set_connection_timeout(std::chrono::milliseconds(config.timeout_ms));
        client.set_read_timeout(std::chrono::milliseconds(config.timeout_ms));
        client.set_write_timeout(std::chrono::milliseconds(config.timeout_ms));

        httplib::Headers headers = {
            {"Authorization", "Bearer " + config.api_key},
            {"Accept", "text/event-stream, application/json"},
        };
        std::string raw_body;
        std::string line_buffer;
        std::string event_data;
        bool saw_stream_event = false;
        json stream_error;

        auto flush_event = [&]() -> bool {
            if (event_data.empty()) {
                return true;
            }

            saw_stream_event = true;
            const bool ok = deepseek_consume_sse_event(event_data, out_result, observer, &stream_error);
            event_data.clear();
            return ok;
        };

        auto response = client.Post(
                parts.path.c_str(),
                headers,
                provider_body.dump(),
                "application/json",
                [&](const char * data, size_t data_length) {
                    if (!stream_error.is_null()) {
                        return false;
                    }

                    raw_body.append(data, data_length);
                    line_buffer.append(data, data_length);

                    size_t line_end = std::string::npos;
                    while ((line_end = line_buffer.find('\n')) != std::string::npos) {
                        std::string line = line_buffer.substr(0, line_end);
                        line_buffer.erase(0, line_end + 1);
                        if (!line.empty() && line.back() == '\r') {
                            line.pop_back();
                        }

                        if (line.empty()) {
                            if (!flush_event()) {
                                return false;
                            }
                            continue;
                        }

                        if (line.rfind("data:", 0) == 0) {
                            std::string data_line = trim_ascii_copy_local(line.substr(5));
                            if (!event_data.empty()) {
                                event_data.push_back('\n');
                            }
                            event_data += data_line;
                        }
                    }

                    return true;
                });
        if (!response) {
            if (out_error) {
                *out_error = format_error_response(
                        "DeepSeek provider request failed before a response was received.",
                        ERROR_TYPE_UNAVAILABLE);
            }
            return false;
        }

        if (!line_buffer.empty()) {
            std::string line = line_buffer;
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.rfind("data:", 0) == 0) {
                std::string data_line = trim_ascii_copy_local(line.substr(5));
                if (!event_data.empty()) {
                    event_data.push_back('\n');
                }
                event_data += data_line;
            }
        }
        if (!event_data.empty() && !flush_event()) {
            if (out_error) {
                *out_error = stream_error;
            }
            return false;
        }
        if (!stream_error.is_null()) {
            if (out_error) {
                *out_error = stream_error;
            }
            return false;
        }

        if (response->status == 401 || response->status == 403) {
            if (out_error) {
                *out_error = format_error_response(
                        "DeepSeek authentication failed. Check VICUNA_DEEPSEEK_API_KEY.",
                        ERROR_TYPE_AUTHENTICATION);
            }
            return false;
        }

        if (response->status < 200 || response->status >= 300) {
            error_type type = response->status >= 500 ? ERROR_TYPE_UNAVAILABLE : ERROR_TYPE_INVALID_REQUEST;
            std::string message = "DeepSeek request failed";
            try {
                const json error_body = json::parse(raw_body);
                if (error_body.contains("error") && error_body.at("error").is_object()) {
                    message = json_value(error_body.at("error"), "message", message);
                } else if (error_body.contains("message") && error_body.at("message").is_string()) {
                    message = error_body.at("message").get<std::string>();
                }
            } catch (...) {
                if (!trim_ascii_copy_local(raw_body).empty()) {
                    message = trim_ascii_copy_local(raw_body);
                }
            }
            if (out_error) {
                *out_error = format_error_response("DeepSeek provider error: " + message, type);
            }
            return false;
        }

        if (!saw_stream_event) {
            try {
                const json response_body = json::parse(raw_body);
                if (!deepseek_parse_response_json(response_body, out_result, observer, out_error)) {
                    return false;
                }
            } catch (const std::exception & e) {
                if (out_error) {
                    *out_error = format_error_response(
                            std::string("DeepSeek provider response parse failed: ") + e.what(),
                            ERROR_TYPE_SERVER);
                }
                return false;
            }
        }
        if (!deepseek_finalize_tool_calls(out_result, out_error)) {
            return false;
        }

        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = format_error_response(
                    std::string("DeepSeek provider request failed: ") + e.what(),
                    ERROR_TYPE_SERVER);
        }
        return false;
    }
}

json deepseek_build_health_json(const deepseek_runtime_config & config) {
    return {
        {"status", "ok"},
        {"state", "ready"},
        {"provider", {
            {"name", "deepseek"},
            {"model", config.model},
            {"base_url", server_http_show_masked_url(server_http_parse_url(deepseek_chat_completions_url(config)))},
        }},
    };
}

json deepseek_build_models_json(const deepseek_runtime_config & config) {
    const std::time_t now = std::time(nullptr);
    return {
        {"models", json::array({
            {
                {"name", config.model},
                {"model", config.model},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""},
                {"type", "model"},
                {"description", "DeepSeek provider-backed reasoning model"},
                {"tags", json::array({"deepseek", "provider", "reasoning"})},
                {"capabilities", json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "provider"},
                    {"family", "deepseek"},
                    {"families", json::array({"deepseek"})},
                    {"parameter_size", ""},
                    {"quantization_level", ""},
                }},
            },
        })},
        {"object", "list"},
        {"data", json::array({
            {
                {"id", config.model},
                {"aliases", json::array({config.model})},
                {"tags", json::array({"deepseek", "provider", "reasoning"})},
                {"object", "model"},
                {"created", now},
                {"owned_by", "deepseek"},
                {"meta", {
                    {"provider", "deepseek"},
                    {"endpoint", server_http_show_masked_url(server_http_parse_url(deepseek_chat_completions_url(config)))},
                }},
            },
        })},
    };
}

json deepseek_format_chat_completion_response(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result,
        const std::string & completion_id) {
    const std::time_t now = std::time(nullptr);
    json response = {
        {"choices", json::array({
            {
                {"finish_reason", result.finish_reason.empty() ? "stop" : result.finish_reason},
                {"index", 0},
                {"message", deepseek_message_json(result)},
            },
        })},
        {"created", now},
        {"model", config.model},
        {"system_fingerprint", server_runtime_build_info()},
        {"object", "chat.completion"},
        {"usage", {
            {"completion_tokens", result.completion_tokens},
            {"prompt_tokens", result.prompt_tokens},
            {"total_tokens", result.prompt_tokens + result.completion_tokens},
        }},
        {"id", completion_id},
    };
    if (!result.emotive_trace.is_null()) {
        response["vicuna_emotive_trace"] = result.emotive_trace;
    }
    return response;
}

json deepseek_format_text_completion_response(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result,
        const std::string & completion_id) {
    const std::time_t now = std::time(nullptr);
    json response = {
        {"choices", json::array({
            {
                {"text", result.content},
                {"index", 0},
                {"logprobs", nullptr},
                {"finish_reason", result.finish_reason.empty() ? "stop" : result.finish_reason},
            },
        })},
        {"created", now},
        {"model", config.model},
        {"system_fingerprint", server_runtime_build_info()},
        {"object", "text_completion"},
        {"usage", {
            {"completion_tokens", result.completion_tokens},
            {"prompt_tokens", result.prompt_tokens},
            {"total_tokens", result.prompt_tokens + result.completion_tokens},
        }},
        {"id", completion_id},
    };
    if (!result.emotive_trace.is_null()) {
        response["vicuna_emotive_trace"] = result.emotive_trace;
    }
    return response;
}

json deepseek_format_responses_response(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result) {
    std::vector<json> output;
    if (!result.reasoning_content.empty()) {
        output.push_back({
            {"id", "rs_" + random_string()},
            {"summary", json::array()},
            {"type", "reasoning"},
            {"content", json::array({
                {
                    {"text", result.reasoning_content},
                    {"type", "reasoning_text"},
                },
            })},
            {"encrypted_content", ""},
            {"status", "completed"},
        });
    }

    if (!result.content.empty()) {
        output.push_back({
            {"content", json::array({
                {
                    {"type", "output_text"},
                    {"annotations", json::array()},
                    {"logprobs", json::array()},
                    {"text", result.content},
                },
            })},
            {"id", "msg_" + random_string()},
            {"role", "assistant"},
            {"status", "completed"},
            {"type", "message"},
        });
    }
    for (const auto & tool_call : result.tool_calls) {
        output.push_back({
            {"arguments", tool_call.arguments_json},
            {"call_id", tool_call.id},
            {"id", "fc_" + random_string()},
            {"name", tool_call.name},
            {"status", "completed"},
            {"type", "function_call"},
        });
    }

    const std::time_t now = std::time(nullptr);
    json response = {
        {"completed_at", now},
        {"created_at", now},
        {"id", "resp_" + random_string()},
        {"model", config.model},
        {"object", "response"},
        {"output", output},
        {"status", "completed"},
        {"usage", {
            {"input_tokens", result.prompt_tokens},
            {"output_tokens", result.completion_tokens},
            {"total_tokens", result.prompt_tokens + result.completion_tokens},
        }},
    };
    if (!result.emotive_trace.is_null()) {
        response["vicuna_emotive_trace"] = result.emotive_trace;
    }
    return response;
}

std::vector<json> deepseek_format_chat_completion_stream(
        const deepseek_runtime_config & config,
        const deepseek_chat_result & result,
        const std::string & completion_id,
        bool include_usage) {
    const std::time_t now = std::time(nullptr);
    std::vector<json> deltas;

    auto push_delta = [&](const json & delta, json finish_reason = nullptr, bool include_trace = false) {
        json item = {
            {"choices", json::array({
                {
                    {"finish_reason", finish_reason},
                    {"index", 0},
                    {"delta", delta},
                },
            })},
            {"created", now},
            {"id", completion_id},
            {"model", config.model},
            {"system_fingerprint", server_runtime_build_info()},
            {"object", "chat.completion.chunk"},
        };
        if (include_trace && !result.emotive_trace.is_null()) {
            item["vicuna_emotive_trace"] = result.emotive_trace;
        }
        deltas.push_back(std::move(item));
    };

    push_delta({
        {"role", "assistant"},
        {"content", nullptr},
    });

    if (!result.reasoning_content.empty()) {
        push_delta({
            {"reasoning_content", result.reasoning_content},
        });
    }

    if (!result.content.empty()) {
        push_delta({
            {"content", result.content},
        });
    }
    if (!result.tool_calls.empty()) {
        json tool_calls = json::array();
        for (size_t index = 0; index < result.tool_calls.size(); ++index) {
            const auto & tool_call = result.tool_calls[index];
            tool_calls.push_back({
                {"index", index},
                {"id", tool_call.id},
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", tool_call.arguments_json},
                }},
            });
        }
        push_delta({
            {"tool_calls", std::move(tool_calls)},
        });
    }

    push_delta(json::object(), result.finish_reason.empty() ? "stop" : result.finish_reason, true);

    if (include_usage) {
        deltas.push_back({
            {"choices", json::array()},
            {"created", now},
            {"id", completion_id},
            {"model", config.model},
            {"system_fingerprint", server_runtime_build_info()},
            {"object", "chat.completion.chunk"},
            {"usage", {
                {"completion_tokens", result.completion_tokens},
                {"prompt_tokens", result.prompt_tokens},
                {"total_tokens", result.prompt_tokens + result.completion_tokens},
            }},
        });
    }

    return deltas;
}
