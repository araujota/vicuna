#include "server-deepseek.h"
#include "server-runtime.h"

#include "common/http.h"

#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
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

static std::string deepseek_chat_completions_url(const deepseek_runtime_config & config, bool use_beta = false) {
    std::string url = trim_ascii_copy_local(config.base_url);
    if (url.empty()) {
        url = "https://api.deepseek.com";
    }
    const std::string suffix = use_beta ? "/beta/chat/completions" : "/chat/completions";
    const std::string alternate_suffix = use_beta ? "/chat/completions" : "/beta/chat/completions";
    if (url.size() >= suffix.size() &&
            url.compare(url.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return url;
    }
    if (url.size() >= alternate_suffix.size() &&
            url.compare(url.size() - alternate_suffix.size(), alternate_suffix.size(), alternate_suffix) == 0) {
        url.erase(url.size() - alternate_suffix.size());
    }
    if (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url + suffix;
}

static bool deepseek_request_uses_strict_tools(const json & provider_body) {
    if (!provider_body.contains("tools") || !provider_body.at("tools").is_array()) {
        return false;
    }
    for (const auto & item : provider_body.at("tools")) {
        if (!item.is_object() || json_value(item, "type", std::string()) != "function") {
            continue;
        }
        const json function = item.value("function", json::object());
        if (!function.is_object()) {
            continue;
        }
        if (function.contains("strict") && function.at("strict").is_boolean() && function.at("strict").get<bool>()) {
            return true;
        }
    }
    return false;
}

static bool deepseek_request_uses_prefix_completion(const json & provider_body) {
    if (!provider_body.contains("messages") || !provider_body.at("messages").is_array() ||
            provider_body.at("messages").empty()) {
        return false;
    }
    const json & last_message = provider_body.at("messages").back();
    return last_message.is_object() &&
            json_value(last_message, "role", std::string()) == "assistant" &&
            json_value(last_message, "prefix", false);
}

static bool deepseek_request_thinking_enabled(const json & body, bool default_enabled) {
    if (!body.contains("thinking") || !body.at("thinking").is_object()) {
        return default_enabled;
    }
    return json_value(body.at("thinking"), "type", std::string()) == "enabled";
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
            if (role == "assistant") {
                const auto reasoning_it = item.find("reasoning_content");
                if (reasoning_it != item.end() && reasoning_it->is_string()) {
                    provider_message["reasoning_content"] = reasoning_it->get<std::string>();
                }
                if (item.contains("prefix") && item.at("prefix").is_boolean()) {
                    provider_message["prefix"] = item.at("prefix").get<bool>();
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

static constexpr int64_t VICUNA_OUTBOUND_MAX_TOKENS = 768;
static constexpr double VICUNA_OUTBOUND_TEMPERATURE = 0.2;

struct deepseek_shared_client_state {
    std::mutex mutex;
    std::string url;
    server_http_url parts;
    std::unique_ptr<httplib::Client> client;
    int64_t client_builds = 0;
    int64_t request_count = 0;
    int64_t reused_request_count = 0;
};

static deepseek_shared_client_state & deepseek_shared_client_state_instance() {
    static deepseek_shared_client_state state;
    return state;
}

static void deepseek_emit_trace_event(
        const deepseek_request_trace * trace,
        const std::string & event,
        json data = json::object()) {
    if (!trace || !trace->emit) {
        return;
    }
    if (!data.is_object()) {
        data = json::object();
    }
    if (!trace->request_id.empty() && !data.contains("request_id")) {
        data["request_id"] = trace->request_id;
    }
    if (!trace->mode_label.empty() && !data.contains("mode_label")) {
        data["mode_label"] = trace->mode_label;
    }
    trace->emit(event, std::move(data));
}

static bool deepseek_prepare_shared_client_locked(
        deepseek_shared_client_state & state,
        const deepseek_runtime_config & config,
        const std::string & request_url,
        std::string * out_url,
        server_http_url * out_parts,
        bool * out_reused = nullptr) {
    const std::string url = request_url.empty() ? deepseek_chat_completions_url(config) : request_url;
    const bool reuse_existing = state.client && state.url == url;
    if (!reuse_existing) {
        auto [client, parts] = server_http_client(url);
        auto shared_client = std::make_unique<httplib::Client>(std::move(client));
        shared_client->set_keep_alive(true);
        state.client = std::move(shared_client);
        state.parts = std::move(parts);
        state.url = url;
        ++state.client_builds;
    } else {
        ++state.reused_request_count;
    }

    if (state.client) {
        state.client->set_connection_timeout(std::chrono::milliseconds(config.timeout_ms));
        state.client->set_read_timeout(std::chrono::milliseconds(config.timeout_ms));
        state.client->set_write_timeout(std::chrono::milliseconds(config.timeout_ms));
    }

    if (out_url) {
        *out_url = state.url;
    }
    if (out_parts) {
        *out_parts = state.parts;
    }
    if (out_reused) {
        *out_reused = reuse_existing;
    }
    return true;
}

static json deepseek_transport_health_json() {
    auto & state = deepseek_shared_client_state_instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    return {
        {"shared_client_cached", state.client != nullptr},
        {"client_builds", state.client_builds},
        {"request_count", state.request_count},
        {"reused_request_count", state.reused_request_count},
        {"base_url", state.url.empty() ? json(nullptr) : json(server_http_show_masked_url(state.parts))},
    };
}

static json deepseek_build_provider_body(const deepseek_runtime_config & config, const json & body) {
    const int64_t outbound_max_tokens = json_value(body, "x-vicuna-provider-max-tokens-override", VICUNA_OUTBOUND_MAX_TOKENS);
    const bool thinking_enabled = deepseek_request_thinking_enabled(body, config.default_thinking_enabled);
    json provider_body = {
        {"model", config.model},
        {"messages", deepseek_build_provider_messages(body)},
        {"stream", true},
        {"max_tokens", outbound_max_tokens},
    };
    if (body.contains("temperature")) {
        provider_body["temperature"] = body.at("temperature");
    } else if (!thinking_enabled && !body.contains("top_p")) {
        provider_body["temperature"] = VICUNA_OUTBOUND_TEMPERATURE;
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
    if (body.contains("stop")) {
        provider_body["stop"] = body.at("stop");
    }
    if (body.contains("logprobs")) {
        provider_body["logprobs"] = body.at("logprobs");
    }
    if (body.contains("top_logprobs")) {
        provider_body["top_logprobs"] = body.at("top_logprobs");
    }
    if (body.contains("thinking")) {
        provider_body["thinking"] = body.at("thinking");
    } else if (config.default_thinking_enabled) {
        provider_body["thinking"] = {
            {"type", "enabled"},
        };
    }

    return provider_body;
}

static bool deepseek_finalize_tool_calls(
        deepseek_chat_result * out_result,
        json * out_error,
        const deepseek_request_trace * trace = nullptr) {
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
            if (trace) {
                deepseek_emit_trace_event(trace, "provider_tool_arguments_parse_failed", {
                    {"provider", "deepseek"},
                    {"tool_name", tool_call.name.empty() ? json(nullptr) : json(tool_call.name)},
                    {"tool_call_id", tool_call.id.empty() ? json(nullptr) : json(tool_call.id)},
                    {"raw_arguments", tool_call.arguments_json},
                    {"message", std::string("DeepSeek provider returned malformed tool arguments JSON: ") + e.what()},
                });
            }
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

static json deepseek_collect_system_messages(const json & provider_body) {
    json system_messages = json::array();
    if (!provider_body.contains("messages") || !provider_body.at("messages").is_array()) {
        return system_messages;
    }
    for (const auto & item : provider_body.at("messages")) {
        if (!item.is_object() || json_value(item, "role", std::string()) != "system") {
            continue;
        }
        if (item.contains("content")) {
            system_messages.push_back(item.at("content"));
        }
    }
    return system_messages;
}

static void deepseek_apply_usage(const json & response_body, deepseek_chat_result * out_result) {
    if (!response_body.contains("usage") || !response_body.at("usage").is_object()) {
        return;
    }

    const json & usage = response_body.at("usage");
    out_result->prompt_tokens = json_value(usage, "prompt_tokens", out_result->prompt_tokens);
    out_result->completion_tokens = json_value(usage, "completion_tokens", out_result->completion_tokens);
}

static std::string deepseek_decode_dsml_text(const std::string & value) {
    std::string decoded = value;
    const std::pair<const char *, const char *> replacements[] = {
        {"&quot;", "\""},
        {"&apos;", "'"},
        {"&gt;", ">"},
        {"&lt;", "<"},
        {"&amp;", "&"},
    };
    for (const auto & replacement : replacements) {
        size_t cursor = 0;
        while ((cursor = decoded.find(replacement.first, cursor)) != std::string::npos) {
            decoded.replace(cursor, std::strlen(replacement.first), replacement.second);
            cursor += std::strlen(replacement.second);
        }
    }
    return decoded;
}

static std::string deepseek_extract_dsml_attr(const std::string & tag, const std::string & name) {
    const std::string needle = name + "=\"";
    const size_t begin = tag.find(needle);
    if (begin == std::string::npos) {
        return std::string();
    }
    const size_t value_begin = begin + needle.size();
    const size_t value_end = tag.find('"', value_begin);
    if (value_end == std::string::npos) {
        return std::string();
    }
    return deepseek_decode_dsml_text(tag.substr(value_begin, value_end - value_begin));
}

static json deepseek_parse_dsml_parameter_value(const std::string & raw_value, bool force_string) {
    const std::string decoded = deepseek_decode_dsml_text(raw_value);
    if (force_string) {
        return decoded;
    }

    const std::string trimmed = trim_ascii_copy_local(decoded);
    if (trimmed.empty()) {
        return std::string();
    }
    try {
        return json::parse(trimmed);
    } catch (...) {
        return decoded;
    }
}

static bool deepseek_recover_dsml_tool_calls_from_content(
        deepseek_chat_result * out_result,
        const deepseek_request_trace * trace) {
    if (!out_result || !out_result->tool_calls.empty()) {
        return false;
    }

    static const std::string function_calls_open = u8"<｜DSML｜function_calls>";
    static const std::string function_calls_close = u8"</｜DSML｜function_calls>";
    static const std::string invoke_open = u8"<｜DSML｜invoke";
    static const std::string invoke_close = u8"</｜DSML｜invoke>";
    static const std::string parameter_open = u8"<｜DSML｜parameter";
    static const std::string parameter_close = u8"</｜DSML｜parameter>";

    const size_t block_begin = out_result->content.find(function_calls_open);
    if (block_begin == std::string::npos) {
        return false;
    }
    const size_t block_end = out_result->content.find(function_calls_close, block_begin + function_calls_open.size());
    if (block_end == std::string::npos) {
        return false;
    }

    const std::string dsml_block = out_result->content.substr(
            block_begin + function_calls_open.size(),
            block_end - (block_begin + function_calls_open.size()));
    std::vector<deepseek_tool_call> recovered;
    size_t cursor = 0;
    while (true) {
        const size_t invoke_begin = dsml_block.find(invoke_open, cursor);
        if (invoke_begin == std::string::npos) {
            break;
        }
        const size_t invoke_tag_end = dsml_block.find('>', invoke_begin);
        const size_t invoke_end = dsml_block.find(invoke_close, invoke_tag_end == std::string::npos ? invoke_begin : invoke_tag_end + 1);
        if (invoke_tag_end == std::string::npos || invoke_end == std::string::npos) {
            return false;
        }

        const std::string invoke_tag = dsml_block.substr(invoke_begin, invoke_tag_end - invoke_begin + 1);
        const std::string tool_name = trim_ascii_copy_local(deepseek_extract_dsml_attr(invoke_tag, "name"));
        if (tool_name.empty()) {
            return false;
        }

        json arguments = json::object();
        const std::string invoke_body = dsml_block.substr(invoke_tag_end + 1, invoke_end - (invoke_tag_end + 1));
        size_t parameter_cursor = 0;
        while (true) {
            const size_t parameter_begin = invoke_body.find(parameter_open, parameter_cursor);
            if (parameter_begin == std::string::npos) {
                break;
            }
            const size_t parameter_tag_end = invoke_body.find('>', parameter_begin);
            const size_t parameter_end = invoke_body.find(
                    parameter_close,
                    parameter_tag_end == std::string::npos ? parameter_begin : parameter_tag_end + 1);
            if (parameter_tag_end == std::string::npos || parameter_end == std::string::npos) {
                return false;
            }

            const std::string parameter_tag = invoke_body.substr(
                    parameter_begin,
                    parameter_tag_end - parameter_begin + 1);
            const std::string parameter_name = trim_ascii_copy_local(
                    deepseek_extract_dsml_attr(parameter_tag, "name"));
            if (parameter_name.empty()) {
                return false;
            }
            const std::string parameter_string = trim_ascii_copy_local(
                    deepseek_extract_dsml_attr(parameter_tag, "string"));
            const std::string parameter_value = invoke_body.substr(
                    parameter_tag_end + 1,
                    parameter_end - (parameter_tag_end + 1));
            arguments[parameter_name] = deepseek_parse_dsml_parameter_value(
                    parameter_value,
                    parameter_string == "true");
            parameter_cursor = parameter_end + parameter_close.size();
        }

        deepseek_tool_call tool_call;
        tool_call.name = tool_name;
        tool_call.arguments_json = arguments.dump();
        recovered.push_back(std::move(tool_call));
        cursor = invoke_end + invoke_close.size();
    }

    if (recovered.empty()) {
        return false;
    }

    const std::string prefix = trim_ascii_copy_local(out_result->content.substr(0, block_begin));
    const std::string suffix = trim_ascii_copy_local(out_result->content.substr(block_end + function_calls_close.size()));
    if (prefix.empty()) {
        out_result->content = suffix;
    } else if (suffix.empty()) {
        out_result->content = prefix;
    } else {
        out_result->content = prefix + "\n\n" + suffix;
    }
    out_result->tool_calls = std::move(recovered);
    out_result->finish_reason = "tool_calls";

    if (trace) {
        deepseek_emit_trace_event(trace, "provider_dsml_tool_calls_recovered", {
            {"provider", "deepseek"},
            {"tool_call_count", static_cast<int64_t>(out_result->tool_calls.size())},
            {"remaining_content", out_result->content},
        });
    }
    return true;
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
        return deepseek_finalize_tool_calls(out_result, out_error, nullptr);
    }

    return true;
}

static bool deepseek_parse_response_json(
        const json & response_body,
        deepseek_chat_result * out_result,
        const deepseek_stream_observer * observer,
        json * out_error,
        const deepseek_request_trace * trace = nullptr) {
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
    if (!payload.contains("tool_calls")) {
        deepseek_recover_dsml_tool_calls_from_content(out_result, trace);
    }

    const std::string finish_reason = json_value(choice, "finish_reason", std::string());
    if (!finish_reason.empty() && out_result->tool_calls.empty()) {
        out_result->finish_reason = finish_reason;
    }

    deepseek_apply_usage(response_body, out_result);
    return true;
}

static bool deepseek_consume_sse_event(
        const std::string & raw_event,
        deepseek_chat_result * out_result,
        const deepseek_stream_observer * observer,
        json * out_error,
        const deepseek_request_trace * trace = nullptr) {
    const std::string event = trim_ascii_copy_local(raw_event);
    if (event.empty() || event == "[DONE]") {
        return true;
    }

    try {
        return deepseek_parse_response_json(json::parse(event), out_result, observer, out_error, trace);
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
        json * out_error,
        const deepseek_request_trace * trace) {
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
    const bool use_strict_tools = deepseek_request_uses_strict_tools(provider_body);
    const bool use_prefix_completion = deepseek_request_uses_prefix_completion(provider_body);
    const bool use_beta_endpoint = use_strict_tools || use_prefix_completion;
    const std::string request_url = deepseek_chat_completions_url(config, use_beta_endpoint);

    const auto request_started_at = std::chrono::steady_clock::now();
    deepseek_emit_trace_event(trace, "provider_request_started", {
        {"provider", "deepseek"},
        {"model", json_value(provider_body, "model", config.model)},
        {"message_count", provider_body.contains("messages") && provider_body.at("messages").is_array() ?
                static_cast<int64_t>(provider_body.at("messages").size()) : 0},
        {"tool_count", provider_body.contains("tools") && provider_body.at("tools").is_array() ?
                static_cast<int64_t>(provider_body.at("tools").size()) : 0},
        {"max_tokens", json_value(provider_body, "max_tokens", int64_t(0))},
        {"temperature", provider_body.contains("temperature") ? provider_body.at("temperature") : json(nullptr)},
        {"top_p", provider_body.contains("top_p") ? provider_body.at("top_p") : json(nullptr)},
        {"thinking", provider_body.contains("thinking") ? provider_body.at("thinking") : json(nullptr)},
        {"response_format", provider_body.contains("response_format") ? provider_body.at("response_format") : json(nullptr)},
        {"stop", provider_body.contains("stop") ? provider_body.at("stop") : json(nullptr)},
        {"strict_tools", use_strict_tools},
        {"prefix_completion", use_prefix_completion},
        {"system_messages", deepseek_collect_system_messages(provider_body)},
        {"endpoint", server_http_show_masked_url(server_http_parse_url(request_url))},
    });

    try {
        auto & shared_client_state = deepseek_shared_client_state_instance();
        std::unique_lock<std::mutex> client_lock(shared_client_state.mutex);
        server_http_url parts;
        bool reused_client = false;
        deepseek_prepare_shared_client_locked(shared_client_state, config, request_url, nullptr, &parts, &reused_client);
        ++shared_client_state.request_count;

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
            const bool ok = deepseek_consume_sse_event(event_data, out_result, observer, &stream_error, trace);
            event_data.clear();
            return ok;
        };

        auto response = shared_client_state.client->Post(
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
            deepseek_emit_trace_event(trace, "provider_request_error", {
                {"provider", "deepseek"},
                {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_started_at).count()},
                {"reused_client", reused_client},
                {"message", "DeepSeek provider request failed before a response was received."},
            });
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
            deepseek_emit_trace_event(trace, "provider_request_error", {
                {"provider", "deepseek"},
                {"status", response->status},
                {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_started_at).count()},
                {"reused_client", reused_client},
                {"message", json_value(stream_error, "message", std::string("provider stream error"))},
            });
            if (out_error) {
                *out_error = stream_error;
            }
            return false;
        }
        if (!stream_error.is_null()) {
            deepseek_emit_trace_event(trace, "provider_request_error", {
                {"provider", "deepseek"},
                {"status", response->status},
                {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_started_at).count()},
                {"reused_client", reused_client},
                {"message", json_value(stream_error, "message", std::string("provider stream error"))},
            });
            if (out_error) {
                *out_error = stream_error;
            }
            return false;
        }

        if (response->status == 401 || response->status == 403) {
            deepseek_emit_trace_event(trace, "provider_request_error", {
                {"provider", "deepseek"},
                {"status", response->status},
                {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_started_at).count()},
                {"reused_client", reused_client},
                {"message", "DeepSeek authentication failed. Check VICUNA_DEEPSEEK_API_KEY."},
            });
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
            deepseek_emit_trace_event(trace, "provider_request_error", {
                {"provider", "deepseek"},
                {"status", response->status},
                {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_started_at).count()},
                {"reused_client", reused_client},
                {"message", message},
            });
            return false;
        }

        if (!saw_stream_event) {
            try {
                const json response_body = json::parse(raw_body);
                if (!deepseek_parse_response_json(response_body, out_result, observer, out_error, trace)) {
                    deepseek_emit_trace_event(trace, "provider_request_error", {
                        {"provider", "deepseek"},
                        {"status", response->status},
                        {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - request_started_at).count()},
                        {"reused_client", reused_client},
                        {"message", out_error ? json_value(*out_error, "message", std::string("provider response parse failed")) : std::string("provider response parse failed")},
                    });
                    return false;
                }
            } catch (const std::exception & e) {
                deepseek_emit_trace_event(trace, "provider_request_error", {
                    {"provider", "deepseek"},
                    {"status", response->status},
                    {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - request_started_at).count()},
                    {"reused_client", reused_client},
                    {"message", std::string("DeepSeek provider response parse failed: ") + e.what()},
                });
                if (out_error) {
                    *out_error = format_error_response(
                            std::string("DeepSeek provider response parse failed: ") + e.what(),
                            ERROR_TYPE_SERVER);
                }
                return false;
            }
        }
        if (!deepseek_finalize_tool_calls(out_result, out_error, trace)) {
            deepseek_emit_trace_event(trace, "provider_request_error", {
                {"provider", "deepseek"},
                {"status", response->status},
                {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_started_at).count()},
                {"reused_client", reused_client},
                {"message", out_error ? json_value(*out_error, "message", std::string("provider tool finalization failed")) : std::string("provider tool finalization failed")},
            });
            return false;
        }

        deepseek_emit_trace_event(trace, "provider_request_finished", {
            {"provider", "deepseek"},
            {"status", response->status},
            {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - request_started_at).count()},
            {"reused_client", reused_client},
            {"finish_reason", out_result->finish_reason},
            {"prompt_tokens", out_result->prompt_tokens},
            {"completion_tokens", out_result->completion_tokens},
            {"tool_call_count", static_cast<int64_t>(out_result->tool_calls.size())},
            {"reasoning_chars", static_cast<int64_t>(out_result->reasoning_content.size())},
            {"content_chars", static_cast<int64_t>(out_result->content.size())},
            {"reasoning_content", out_result->reasoning_content},
            {"content", out_result->content},
            {"saw_stream_event", saw_stream_event},
        });
        return true;
    } catch (const std::exception & e) {
        deepseek_emit_trace_event(trace, "provider_request_error", {
            {"provider", "deepseek"},
            {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - request_started_at).count()},
            {"message", std::string("DeepSeek provider request failed: ") + e.what()},
        });
        if (out_error) {
            *out_error = format_error_response(
                    std::string("DeepSeek provider request failed: ") + e.what(),
                    ERROR_TYPE_SERVER);
        }
        return false;
    }
}

json deepseek_tool_call_to_json(const deepseek_tool_call & tool_call) {
    return {
        {"id", tool_call.id},
        {"name", tool_call.name},
        {"arguments_json", tool_call.arguments_json},
    };
}

bool deepseek_tool_call_from_json(
        const json & body,
        deepseek_tool_call * out_tool_call,
        std::string * out_error) {
    if (!out_tool_call) {
        if (out_error) {
            *out_error = "tool call output must not be null";
        }
        return false;
    }
    if (!body.is_object()) {
        if (out_error) {
            *out_error = "tool call payload must be an object";
        }
        return false;
    }

    out_tool_call->id = json_value(body, "id", std::string());
    out_tool_call->name = json_value(body, "name", std::string());
    out_tool_call->arguments_json = json_value(body, "arguments_json", std::string());
    if (trim_ascii_copy_local(out_tool_call->name).empty()) {
        if (out_error) {
            *out_error = "tool call payload did not include a name";
        }
        return false;
    }
    return true;
}

json deepseek_chat_result_to_json(const deepseek_chat_result & result) {
    json tool_calls = json::array();
    for (const auto & tool_call : result.tool_calls) {
        tool_calls.push_back(deepseek_tool_call_to_json(tool_call));
    }

    json body = {
        {"content", result.content},
        {"reasoning_content", result.reasoning_content},
        {"finish_reason", result.finish_reason},
        {"prompt_tokens", result.prompt_tokens},
        {"completion_tokens", result.completion_tokens},
        {"tool_calls", std::move(tool_calls)},
        {"emotive_trace", result.emotive_trace},
        {"rich_response", result.rich_response},
    };
    return body;
}

bool deepseek_chat_result_from_json(
        const json & body,
        deepseek_chat_result * out_result,
        std::string * out_error) {
    if (!out_result) {
        if (out_error) {
            *out_error = "chat result output must not be null";
        }
        return false;
    }
    if (!body.is_object()) {
        if (out_error) {
            *out_error = "chat result payload must be an object";
        }
        return false;
    }

    deepseek_chat_result parsed;
    parsed.content = json_value(body, "content", std::string());
    parsed.reasoning_content = json_value(body, "reasoning_content", std::string());
    parsed.finish_reason = json_value(body, "finish_reason", std::string("stop"));
    parsed.prompt_tokens = json_value(body, "prompt_tokens", int32_t(0));
    parsed.completion_tokens = json_value(body, "completion_tokens", int32_t(0));
    if (body.contains("tool_calls") && !body.at("tool_calls").is_null()) {
        if (!body.at("tool_calls").is_array()) {
            if (out_error) {
                *out_error = "chat result tool_calls payload must be an array";
            }
            return false;
        }
        for (const auto & item : body.at("tool_calls")) {
            deepseek_tool_call tool_call;
            if (!deepseek_tool_call_from_json(item, &tool_call, out_error)) {
                return false;
            }
            parsed.tool_calls.push_back(std::move(tool_call));
        }
    }
    parsed.emotive_trace = body.contains("emotive_trace") ? body.at("emotive_trace") : json(nullptr);
    parsed.rich_response = body.contains("rich_response") ? body.at("rich_response") : json(nullptr);
    *out_result = std::move(parsed);
    return true;
}

json deepseek_build_health_json(const deepseek_runtime_config & config) {
    return {
        {"status", "ok"},
        {"state", "ready"},
        {"provider", {
            {"name", "deepseek"},
            {"model", config.model},
            {"base_url", server_http_show_masked_url(server_http_parse_url(deepseek_chat_completions_url(config)))},
            {"transport", deepseek_transport_health_json()},
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
