#include "server-runpod.h"

#include "server-runtime.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <random>

namespace {

static std::string trim_ascii_copy(const std::string & value) {
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

static std::string to_lower_ascii_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static std::string normalize_compact_identifier(std::string value) {
    std::string compact;
    compact.reserve(value.size());
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) != 0) {
            compact.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return compact;
}

static int32_t env_to_int(const char * name, int32_t default_value) {
    if (const char * value = std::getenv(name)) {
        const int parsed = std::atoi(value);
        if (parsed > 0) {
            return parsed;
        }
    }
    return default_value;
}

static std::string join_url_path_local(const server_http_url & parts, const std::string & suffix) {
    std::string path = parts.path;
    if (path.empty() || path == "/") {
        path.clear();
    }
    if (!path.empty() && path.back() == '/' && !suffix.empty() && suffix.front() == '/') {
        path.pop_back();
    }
    if (path.empty()) {
        return suffix.empty() ? std::string("/") : suffix;
    }
    if (suffix.empty()) {
        return path;
    }
    if (suffix.front() != '/') {
        return path + "/" + suffix;
    }
    return path + suffix;
}

static std::string resolve_runpod_model_target(const std::string & requested_model) {
    const std::string compact = normalize_compact_identifier(requested_model);
    if (compact.empty()) {
        return "google/gemma-4-31B-it";
    }
    if (compact.find("gemma431bit") != std::string::npos ||
            compact.find("gemma431bitgguf") != std::string::npos) {
        return "google/gemma-4-31B-it";
    }
    if (compact.find("mistralsmall324binstruct2506") != std::string::npos ||
            compact.find("mistralsmall3224binstruct2506") != std::string::npos) {
        return "mistralai/Mistral-Small-3.2-24B-Instruct-2506";
    }
    if (compact.find("qwen335ba3b") != std::string::npos ||
            compact.find("qwen3535ba3b") != std::string::npos) {
        return "Qwen/Qwen3.5-35B-A3B";
    }
    if (compact.find("qwen332b") != std::string::npos) {
        return "Qwen/Qwen3-32B";
    }
    if (compact.find("qwen330ba3b") != std::string::npos) {
        return "Qwen/Qwen3-30B-A3B";
    }
    if (compact.find("qwen3") != std::string::npos && compact.find("a3b") != std::string::npos) {
        return "Qwen/Qwen3.5-35B-A3B";
    }
    return trim_ascii_copy(requested_model);
}

static runpod_inference_runtime_config & runpod_runtime_config_storage() {
    static runpod_inference_runtime_config config;
    return config;
}

static std::string parse_error_message(const json & body, const std::string & fallback) {
    if (body.is_object()) {
        if (body.contains("error") && body.at("error").is_object()) {
            return json_value(body.at("error"), "message", fallback);
        }
        return json_value(body, "message", fallback);
    }
    return fallback;
}

static std::string runpod_model_family(const std::string & resolved_model) {
    const std::string compact = normalize_compact_identifier(resolved_model);
    if (compact.find("gemma") != std::string::npos) {
        return "gemma4";
    }
    if (compact.find("mistral") != std::string::npos) {
        return "mistral";
    }
    if (compact.find("qwen") != std::string::npos) {
        return "qwen3";
    }
    return "runpod";
}

static std::string runpod_generate_recovered_tool_call_id() {
    static std::mt19937 rng(std::random_device{}());
    static constexpr char alphabet[] =
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789";
    std::uniform_int_distribution<size_t> dist(0, sizeof(alphabet) - 2);
    std::string id;
    id.reserve(9);
    for (int i = 0; i < 9; ++i) {
        id.push_back(alphabet[dist(rng)]);
    }
    return id;
}

static std::string runpod_normalize_mistral_tool_arguments(const std::string & raw_arguments) {
    const std::string trimmed = trim_ascii_copy(raw_arguments);
    if (trimmed.empty()) {
        return "{}";
    }
    try {
        const json parsed = json::parse(trimmed);
        if (parsed.is_string()) {
            const std::string decoded = trim_ascii_copy(parsed.get<std::string>());
            if (decoded.empty()) {
                return "{}";
            }
            try {
                return json::parse(decoded).dump();
            } catch (...) {
                return decoded;
            }
        }
        return parsed.dump();
    } catch (...) {
        return trimmed;
    }
}

static std::string runpod_strip_markdown_code_fence(const std::string & raw_content) {
    const std::string trimmed = trim_ascii_copy(raw_content);
    if (trimmed.size() < 6 || trimmed.rfind("```", 0) != 0) {
        return trimmed;
    }
    const size_t first_newline = trimmed.find('\n');
    if (first_newline == std::string::npos) {
        return trimmed;
    }
    const size_t closing = trimmed.rfind("```");
    if (closing == std::string::npos || closing <= first_newline) {
        return trimmed;
    }
    return trim_ascii_copy(trimmed.substr(first_newline + 1, closing - (first_newline + 1)));
}

static bool runpod_parse_tool_call_candidate(
        const json & candidate,
        deepseek_tool_call * out_tool_call) {
    if (!candidate.is_object() || !out_tool_call) {
        return false;
    }

    std::string tool_name;
    std::string tool_call_id = trim_ascii_copy(json_value(candidate, "id", std::string()));
    json arguments_payload = nullptr;

    if (candidate.contains("function") && candidate.at("function").is_object()) {
        const json & function = candidate.at("function");
        tool_name = trim_ascii_copy(json_value(function, "name", std::string()));
        if (function.contains("parameters")) {
            arguments_payload = function.at("parameters");
        } else if (function.contains("arguments")) {
            arguments_payload = function.at("arguments");
        }
    } else {
        tool_name = trim_ascii_copy(json_value(candidate, "name", std::string()));
        if (candidate.contains("parameters")) {
            arguments_payload = candidate.at("parameters");
        } else if (candidate.contains("arguments")) {
            arguments_payload = candidate.at("arguments");
        }
        if (tool_call_id.empty()) {
            tool_call_id = trim_ascii_copy(json_value(candidate, "call_id", std::string()));
        }
    }

    if (tool_name.empty()) {
        return false;
    }
    if (tool_call_id.empty()) {
        tool_call_id = runpod_generate_recovered_tool_call_id();
    }

    std::string arguments_json = "{}";
    if (!arguments_payload.is_null()) {
        if (arguments_payload.is_string()) {
            arguments_json = runpod_normalize_mistral_tool_arguments(arguments_payload.get<std::string>());
        } else {
            arguments_json = arguments_payload.dump();
        }
    }

    out_tool_call->id = tool_call_id;
    out_tool_call->name = tool_name;
    out_tool_call->arguments_json = arguments_json;
    return true;
}

static bool runpod_recover_gemma_json_tool_calls_from_content(deepseek_chat_result * out_result) {
    if (!out_result || !out_result->tool_calls.empty()) {
        return false;
    }

    const std::string candidate_text = runpod_strip_markdown_code_fence(out_result->content);
    if (candidate_text.empty()) {
        return false;
    }

    json parsed_payload;
    try {
        parsed_payload = json::parse(candidate_text);
    } catch (...) {
        return false;
    }

    std::vector<deepseek_tool_call> recovered;
    auto append_candidate = [&recovered](const json & candidate) -> bool {
        deepseek_tool_call tool_call;
        if (!runpod_parse_tool_call_candidate(candidate, &tool_call)) {
            return false;
        }
        recovered.push_back(std::move(tool_call));
        return true;
    };

    if (parsed_payload.is_object()) {
        if (!append_candidate(parsed_payload)) {
            return false;
        }
    } else if (parsed_payload.is_array()) {
        if (parsed_payload.empty()) {
            return false;
        }
        for (const auto & candidate : parsed_payload) {
            if (!append_candidate(candidate)) {
                return false;
            }
        }
    } else {
        return false;
    }

    if (recovered.empty()) {
        return false;
    }

    out_result->content.clear();
    out_result->tool_calls = std::move(recovered);
    out_result->finish_reason = "tool_calls";
    return true;
}

static bool runpod_is_gemma_identifier_char(const char ch) {
    const unsigned char value = static_cast<unsigned char>(ch);
    return std::isalnum(value) != 0 || ch == '_' || ch == '-' || ch == ':' || ch == '.';
}

static std::vector<std::string> runpod_split_top_level_arguments(const std::string & text) {
    std::vector<std::string> parts;
    std::string current;
    int paren_depth = 0;
    int brace_depth = 0;
    int bracket_depth = 0;
    bool in_string = false;
    char string_quote = '\0';
    bool escape = false;

    for (const char ch : text) {
        if (in_string) {
            current.push_back(ch);
            if (escape) {
                escape = false;
                continue;
            }
            if (ch == '\\') {
                escape = true;
                continue;
            }
            if (ch == string_quote) {
                in_string = false;
                string_quote = '\0';
            }
            continue;
        }

        if (ch == '"' || ch == '\'') {
            in_string = true;
            string_quote = ch;
            current.push_back(ch);
            continue;
        }

        switch (ch) {
            case '(':
                ++paren_depth;
                break;
            case ')':
                --paren_depth;
                break;
            case '{':
                ++brace_depth;
                break;
            case '}':
                --brace_depth;
                break;
            case '[':
                ++bracket_depth;
                break;
            case ']':
                --bracket_depth;
                break;
            case ',':
                if (paren_depth == 0 && brace_depth == 0 && bracket_depth == 0) {
                    const std::string trimmed = trim_ascii_copy(current);
                    if (!trimmed.empty()) {
                        parts.push_back(trimmed);
                    }
                    current.clear();
                    continue;
                }
                break;
            default:
                break;
        }

        current.push_back(ch);
    }

    const std::string trimmed = trim_ascii_copy(current);
    if (!trimmed.empty()) {
        parts.push_back(trimmed);
    }
    return parts;
}

static bool runpod_find_top_level_equals(const std::string & text, size_t * out_pos) {
    int paren_depth = 0;
    int brace_depth = 0;
    int bracket_depth = 0;
    bool in_string = false;
    char string_quote = '\0';
    bool escape = false;

    for (size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        if (in_string) {
            if (escape) {
                escape = false;
                continue;
            }
            if (ch == '\\') {
                escape = true;
                continue;
            }
            if (ch == string_quote) {
                in_string = false;
                string_quote = '\0';
            }
            continue;
        }
        if (ch == '"' || ch == '\'') {
            in_string = true;
            string_quote = ch;
            continue;
        }
        switch (ch) {
            case '(':
                ++paren_depth;
                break;
            case ')':
                --paren_depth;
                break;
            case '{':
                ++brace_depth;
                break;
            case '}':
                --brace_depth;
                break;
            case '[':
                ++bracket_depth;
                break;
            case ']':
                --bracket_depth;
                break;
            case '=':
                if (paren_depth == 0 && brace_depth == 0 && bracket_depth == 0) {
                    if (out_pos) {
                        *out_pos = index;
                    }
                    return true;
                }
                break;
            default:
                break;
        }
    }

    return false;
}

static std::string runpod_unescape_single_quoted_string(const std::string & text) {
    std::string result;
    result.reserve(text.size());
    bool escape = false;
    for (const char ch : text) {
        if (escape) {
            switch (ch) {
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                default:
                    result.push_back(ch);
                    break;
            }
            escape = false;
            continue;
        }
        if (ch == '\\') {
            escape = true;
            continue;
        }
        result.push_back(ch);
    }
    if (escape) {
        result.push_back('\\');
    }
    return result;
}

static bool runpod_parse_gemma_argument_literal(const std::string & raw_value, json * out_value) {
    if (!out_value) {
        return false;
    }

    const std::string value = trim_ascii_copy(raw_value);
    if (value.empty()) {
        *out_value = "";
        return true;
    }

    if ((value.front() == '"' && value.back() == '"') ||
            (value.front() == '\'' && value.back() == '\'')) {
        if (value.front() == '"') {
            try {
                *out_value = json::parse(value);
                return true;
            } catch (...) {
                return false;
            }
        }
        *out_value = runpod_unescape_single_quoted_string(value.substr(1, value.size() - 2));
        return true;
    }

    if (value == "true") {
        *out_value = true;
        return true;
    }
    if (value == "false") {
        *out_value = false;
        return true;
    }
    if (value == "null" || value == "None") {
        *out_value = nullptr;
        return true;
    }

    if (value.front() == '{' || value.front() == '[') {
        try {
            *out_value = json::parse(value);
            return true;
        } catch (...) {
            return false;
        }
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed_int = std::strtoll(value.c_str(), &end, 10);
    if (end != nullptr && *end == '\0' && errno == 0) {
        *out_value = parsed_int;
        return true;
    }

    end = nullptr;
    errno = 0;
    const double parsed_double = std::strtod(value.c_str(), &end);
    if (end != nullptr && *end == '\0' && errno == 0 &&
            parsed_double >= -std::numeric_limits<double>::max() &&
            parsed_double <= std::numeric_limits<double>::max()) {
        *out_value = parsed_double;
        return true;
    }

    if (std::all_of(value.begin(), value.end(), [](const char ch) {
            return runpod_is_gemma_identifier_char(ch);
        })) {
        *out_value = value;
        return true;
    }

    return false;
}

static bool runpod_parse_gemma_python_invocation(
        const std::string & invocation,
        deepseek_tool_call * out_tool_call) {
    if (!out_tool_call) {
        return false;
    }

    const std::string trimmed = trim_ascii_copy(invocation);
    const size_t paren_open = trimmed.find('(');
    const size_t paren_close = trimmed.rfind(')');
    if (paren_open == std::string::npos || paren_close == std::string::npos || paren_close <= paren_open) {
        return false;
    }
    if (trim_ascii_copy(trimmed.substr(paren_close + 1)).size() != 0) {
        return false;
    }

    const std::string tool_name = trim_ascii_copy(trimmed.substr(0, paren_open));
    if (tool_name.empty() ||
            !std::all_of(tool_name.begin(), tool_name.end(), [](const char ch) {
                return runpod_is_gemma_identifier_char(ch);
            })) {
        return false;
    }

    const std::string arguments_body = trimmed.substr(paren_open + 1, paren_close - paren_open - 1);
    json arguments = json::object();
    for (const std::string & item : runpod_split_top_level_arguments(arguments_body)) {
        size_t equals_pos = std::string::npos;
        if (!runpod_find_top_level_equals(item, &equals_pos)) {
            return false;
        }
        const std::string argument_name = trim_ascii_copy(item.substr(0, equals_pos));
        if (argument_name.empty() ||
                !std::all_of(argument_name.begin(), argument_name.end(), [](const char ch) {
                    return runpod_is_gemma_identifier_char(ch);
                })) {
            return false;
        }
        json argument_value;
        if (!runpod_parse_gemma_argument_literal(item.substr(equals_pos + 1), &argument_value)) {
            return false;
        }
        arguments[argument_name] = std::move(argument_value);
    }

    out_tool_call->id = runpod_generate_recovered_tool_call_id();
    out_tool_call->name = tool_name;
    out_tool_call->arguments_json = arguments.dump();
    return true;
}

static bool runpod_recover_gemma_python_tool_calls_from_content(deepseek_chat_result * out_result) {
    if (!out_result || !out_result->tool_calls.empty()) {
        return false;
    }

    std::string candidate_text = runpod_strip_markdown_code_fence(out_result->content);
    if (candidate_text.empty()) {
        return false;
    }

    if (candidate_text.front() == '[' && candidate_text.back() == ']') {
        candidate_text = trim_ascii_copy(candidate_text.substr(1, candidate_text.size() - 2));
    }
    if (candidate_text.empty()) {
        return false;
    }

    std::vector<deepseek_tool_call> recovered;
    for (const std::string & invocation : runpod_split_top_level_arguments(candidate_text)) {
        deepseek_tool_call tool_call;
        if (!runpod_parse_gemma_python_invocation(invocation, &tool_call)) {
            return false;
        }
        recovered.push_back(std::move(tool_call));
    }

    if (recovered.empty()) {
        return false;
    }

    out_result->content.clear();
    out_result->tool_calls = std::move(recovered);
    out_result->finish_reason = "tool_calls";
    return true;
}

static bool runpod_recover_mistral_tool_calls_from_content(deepseek_chat_result * out_result) {
    if (!out_result || !out_result->tool_calls.empty()) {
        return false;
    }

    static const std::string tool_calls_open = "[TOOL_CALLS]";
    static const std::string call_id_open = "[CALL_ID]";
    static const std::string args_open = "[ARGS]";

    const size_t first_tool_pos = out_result->content.find(tool_calls_open);
    if (first_tool_pos == std::string::npos) {
        return false;
    }

    std::vector<deepseek_tool_call> recovered;
    std::string remaining_content = trim_ascii_copy(out_result->content.substr(0, first_tool_pos));
    size_t cursor = first_tool_pos;
    while (cursor != std::string::npos && cursor < out_result->content.size()) {
        const size_t tool_name_begin = cursor + tool_calls_open.size();
        const size_t call_id_pos = out_result->content.find(call_id_open, tool_name_begin);
        if (call_id_pos == std::string::npos) {
            return false;
        }
        const std::string tool_name = trim_ascii_copy(
                out_result->content.substr(tool_name_begin, call_id_pos - tool_name_begin));
        if (tool_name.empty()) {
            return false;
        }

        const size_t call_id_begin = call_id_pos + call_id_open.size();
        const size_t args_pos = out_result->content.find(args_open, call_id_begin);
        if (args_pos == std::string::npos) {
            return false;
        }
        std::string tool_call_id = trim_ascii_copy(
                out_result->content.substr(call_id_begin, args_pos - call_id_begin));
        if (tool_call_id.empty()) {
            tool_call_id = runpod_generate_recovered_tool_call_id();
        }

        const size_t args_begin = args_pos + args_open.size();
        const size_t next_tool_pos = out_result->content.find(tool_calls_open, args_begin);
        const size_t args_end = next_tool_pos == std::string::npos ? out_result->content.size() : next_tool_pos;
        const std::string raw_arguments = trim_ascii_copy(
                out_result->content.substr(args_begin, args_end - args_begin));
        if (raw_arguments.empty()) {
            return false;
        }

        deepseek_tool_call tool_call;
        tool_call.id = tool_call_id;
        tool_call.name = tool_name;
        tool_call.arguments_json = runpod_normalize_mistral_tool_arguments(raw_arguments);
        recovered.push_back(std::move(tool_call));

        cursor = next_tool_pos;
    }

    if (recovered.empty()) {
        return false;
    }

    out_result->content = remaining_content;
    out_result->tool_calls = std::move(recovered);
    out_result->finish_reason = "tool_calls";
    return true;
}

static bool runpod_parse_openai_chat_result(
        const json & payload,
        deepseek_chat_result * out_result,
        json * out_error) {
    if (!payload.is_object()) {
        if (out_error) {
            *out_error = format_error_response(
                    "RunPod relay returned a non-object chat payload",
                    ERROR_TYPE_SERVER);
        }
        return false;
    }
    if (!payload.contains("choices") || !payload.at("choices").is_array() || payload.at("choices").empty()) {
        if (out_error) {
            *out_error = format_error_response(
                    "RunPod relay did not include any chat choices",
                    ERROR_TYPE_SERVER);
        }
        return false;
    }

    const json & choice = payload.at("choices").at(0);
    const json message = choice.value("message", json::object());
    if (!message.is_object()) {
        if (out_error) {
            *out_error = format_error_response(
                    "RunPod relay did not include a usable chat message",
                    ERROR_TYPE_SERVER);
        }
        return false;
    }

    deepseek_chat_result parsed;
    parsed.content = json_value(message, "content", std::string());
    parsed.reasoning_content = json_value(message, "reasoning_content", std::string());
    parsed.finish_reason = json_value(choice, "finish_reason", std::string("stop"));

    const json usage = payload.value("usage", json::object());
    if (usage.is_object()) {
        parsed.prompt_tokens = json_value(usage, "prompt_tokens", int32_t(0));
        parsed.completion_tokens = json_value(usage, "completion_tokens", int32_t(0));
    }

    if (message.contains("tool_calls") && !message.at("tool_calls").is_null()) {
        if (!message.at("tool_calls").is_array()) {
            if (out_error) {
                *out_error = format_error_response(
                        "RunPod relay returned a non-array tool_calls payload",
                        ERROR_TYPE_SERVER);
            }
            return false;
        }
        for (const auto & item : message.at("tool_calls")) {
            deepseek_tool_call tool_call;
            std::string parse_error;
            if (!deepseek_tool_call_from_json(item, &tool_call, &parse_error)) {
                if (out_error) {
                    *out_error = format_error_response(
                            "RunPod relay returned invalid tool_calls: " + parse_error,
                            ERROR_TYPE_SERVER);
                }
                return false;
            }
            parsed.tool_calls.push_back(std::move(tool_call));
        }
    }
    if (parsed.tool_calls.empty()) {
        runpod_recover_mistral_tool_calls_from_content(&parsed);
    }
    if (parsed.tool_calls.empty()) {
        runpod_recover_gemma_json_tool_calls_from_content(&parsed);
    }
    if (parsed.tool_calls.empty()) {
        runpod_recover_gemma_python_tool_calls_from_content(&parsed);
    }

    parsed.emotive_trace = payload.contains("vicuna_emotive_trace") ?
            payload.at("vicuna_emotive_trace") :
            json(nullptr);
    *out_result = std::move(parsed);
    return true;
}

static bool runpod_relay_preflight_health(
        const runpod_inference_runtime_config & config,
        const server_http_url & parts,
        json * out_meta) {
    auto [health_client, health_parts] = server_http_client(config.relay_url);
    const int32_t preflight_timeout_ms = std::max<int32_t>(1000, std::min<int32_t>(config.timeout_ms, 5000));
    health_client.set_connection_timeout(std::chrono::milliseconds(preflight_timeout_ms));
    health_client.set_read_timeout(std::chrono::milliseconds(preflight_timeout_ms));
    health_client.set_write_timeout(std::chrono::milliseconds(preflight_timeout_ms));

    const std::string health_path = join_url_path_local(health_parts, "/health");
    auto response = health_client.Get(health_path.c_str());
    if (out_meta) {
        (*out_meta)["endpoint"] = server_http_show_masked_url(parts);
        (*out_meta)["preflight_timeout_ms"] = preflight_timeout_ms;
        (*out_meta)["health_path"] = health_path;
        (*out_meta)["preflight_http_status"] = response ? json(response->status) : json(nullptr);
    }
    return response && response->status >= 200 && response->status < 300;
}

} // namespace

runpod_inference_runtime_config runpod_inference_runtime_config_from_env() {
    runpod_inference_runtime_config config;

    if (const char * value = std::getenv("VICUNA_HOST_INFERENCE_MODE")) {
        config.host_mode_label = trim_ascii_copy(value);
    }
    const std::string normalized_host_mode = to_lower_ascii_copy(trim_ascii_copy(config.host_mode_label));
    if (normalized_host_mode.empty() || normalized_host_mode == "standard") {
        config.host_mode = host_inference_mode::standard;
        config.host_mode_label = "standard";
    } else if (normalized_host_mode == "experimental") {
        config.host_mode = host_inference_mode::experimental;
        config.host_mode_label = "experimental";
    } else {
        config.host_mode = host_inference_mode::standard;
        config.host_mode_label = "standard";
        config.config_error = "VICUNA_HOST_INFERENCE_MODE must be one of: standard, experimental";
        return config;
    }

    if (const char * value = std::getenv("VICUNA_RUNPOD_INFERENCE_ROLE")) {
        config.role_label = trim_ascii_copy(value);
    }
    const std::string normalized_role = to_lower_ascii_copy(trim_ascii_copy(config.role_label));
    if (normalized_role.empty() || normalized_role == "disabled" || normalized_role == "off" || normalized_role == "false") {
        config.role = runpod_inference_role::disabled;
        config.role_label = "disabled";
    } else if (normalized_role == "host" || normalized_role == "relay") {
        config.role = runpod_inference_role::host;
        config.role_label = "host";
    } else if (normalized_role == "node" || normalized_role == "remote") {
        config.role = runpod_inference_role::node;
        config.role_label = "node";
    } else {
        config.role = runpod_inference_role::disabled;
        config.role_label = "disabled";
        config.config_error = "VICUNA_RUNPOD_INFERENCE_ROLE must be one of: disabled, host, node";
        return config;
    }

    config.enabled = config.role != runpod_inference_role::disabled;
    if (const char * value = std::getenv("VICUNA_RUNPOD_INFERENCE_URL")) {
        config.relay_url = trim_ascii_copy(value);
    }
    if (const char * value = std::getenv("VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN")) {
        config.auth_token = trim_ascii_copy(value);
    }
    const char * model_env = std::getenv("VICUNA_RUNPOD_MODEL_TARGET");
    if (!model_env) {
        model_env = std::getenv("VICUNA_RUNPOD_INFERENCE_MODEL");
    }
    if (model_env) {
        config.requested_model = trim_ascii_copy(model_env);
    }
    if (const char * value = std::getenv("VICUNA_RUNPOD_SERVING_DTYPE")) {
        const std::string parsed = trim_ascii_copy(value);
        if (!parsed.empty()) {
            config.serving_dtype = parsed;
        }
    }
    if (const char * value = std::getenv("VICUNA_RUNPOD_KV_PROFILE")) {
        const std::string parsed = trim_ascii_copy(value);
        if (!parsed.empty()) {
            config.kv_profile = parsed;
        }
    }
    if (const char * value = std::getenv("VICUNA_RUNPOD_NODE_ID")) {
        const std::string parsed = trim_ascii_copy(value);
        if (!parsed.empty()) {
            config.node_id = parsed;
        }
    }

    config.timeout_ms = env_to_int("VICUNA_RUNPOD_INFERENCE_TIMEOUT_MS", config.timeout_ms);
    config.context_limit = env_to_int("VICUNA_RUNPOD_CONTEXT_LIMIT", config.context_limit);
    config.default_max_tokens = env_to_int(
            "VICUNA_RUNPOD_INFERENCE_DEFAULT_MAX_TOKENS",
            config.default_max_tokens);
    config.resolved_model = resolve_runpod_model_target(config.requested_model);

    if (!config.enabled) {
        return config;
    }
    if ((config.role == runpod_inference_role::host) && config.relay_url.empty()) {
        config.config_error = "VICUNA_RUNPOD_INFERENCE_URL is required when VICUNA_RUNPOD_INFERENCE_ROLE=host";
        return config;
    }
    if (config.auth_token.empty()) {
        config.config_error = "VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN is required when RunPod inference mode is enabled";
        return config;
    }
    return config;
}

void configure_runpod_inference_runtime(const runpod_inference_runtime_config & config) {
    runpod_runtime_config_storage() = config;
}

const runpod_inference_runtime_config & runpod_inference_runtime_config_instance() {
    return runpod_runtime_config_storage();
}

bool runpod_inference_local_provider_required(const runpod_inference_runtime_config & config) {
    return config.role != runpod_inference_role::host;
}

bool runpod_inference_host_relay_enabled(const runpod_inference_runtime_config & config) {
    return config.enabled &&
            config.role == runpod_inference_role::host &&
            config.host_mode == host_inference_mode::experimental &&
            config.config_error.empty();
}

bool runpod_inference_node_execution_enabled(const runpod_inference_runtime_config & config) {
    return config.enabled && config.role == runpod_inference_role::node && config.config_error.empty();
}

bool host_inference_experimental_enabled(const runpod_inference_runtime_config & config) {
    return config.host_mode == host_inference_mode::experimental;
}

bool host_inference_standard_enabled(const runpod_inference_runtime_config & config) {
    return config.host_mode != host_inference_mode::experimental;
}

json runpod_build_health_json(const runpod_inference_runtime_config & config) {
    json payload = {
        {"enabled", config.enabled},
        {"role", config.role_label},
        {"host_inference_mode", config.host_mode_label},
        {"resolved_model", config.resolved_model},
        {"requested_model", config.requested_model},
        {"serving_dtype", config.serving_dtype},
        {"kv_profile", config.kv_profile},
        {"context_limit", config.context_limit},
        {"default_max_tokens", config.default_max_tokens},
        {"node_id", config.node_id},
        {"local_provider_required", runpod_inference_local_provider_required(config)},
    };
    if (!config.relay_url.empty()) {
        payload["relay_url"] = server_http_show_masked_url(server_http_parse_url(config.relay_url));
    } else {
        payload["relay_url"] = nullptr;
    }
    if (!config.config_error.empty()) {
        payload["config_error"] = config.config_error;
    } else {
        payload["config_error"] = nullptr;
    }
    return payload;
}

json runpod_build_models_json(const runpod_inference_runtime_config & config) {
    const std::time_t now = std::time(nullptr);
    const std::string family = runpod_model_family(config.resolved_model);
    const json common_meta = {
        {"provider", "runpod"},
        {"role", config.role_label},
        {"serving_dtype", config.serving_dtype},
        {"kv_profile", config.kv_profile},
        {"context_limit", config.context_limit},
    };
    return {
        {"models", json::array({
            {
                {"name", config.resolved_model},
                {"model", config.resolved_model},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""},
                {"type", "model"},
                {"description", "RunPod relay target for the remote Vicuña inference plane"},
                {"tags", json::array({"runpod", family, "a100", config.serving_dtype, config.kv_profile})},
                {"capabilities", json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", config.requested_model},
                    {"format", "runpod-relay"},
                    {"family", family},
                    {"families", json::array({family, "runpod"})},
                    {"parameter_size", ""},
                    {"quantization_level", config.serving_dtype},
                }},
            },
        })},
        {"object", "list"},
        {"data", json::array({
            {
                {"id", config.resolved_model},
                {"aliases", json::array({config.resolved_model})},
                {"tags", json::array({"runpod", family, "a100", config.serving_dtype, config.kv_profile})},
                {"object", "model"},
                {"created", now},
                {"owned_by", "runpod"},
                {"meta", common_meta},
            },
        })},
    };
}

bool runpod_validate_bearer_auth(
        const runpod_inference_runtime_config & config,
        const std::string & authorization_header,
        std::string * out_error) {
    const std::string trimmed = trim_ascii_copy(authorization_header);
    const std::string prefix = "Bearer ";
    if (trimmed.size() <= prefix.size() || trimmed.compare(0, prefix.size(), prefix) != 0) {
        if (out_error) {
            *out_error = "missing bearer authorization header";
        }
        return false;
    }
    const std::string provided = trim_ascii_copy(trimmed.substr(prefix.size()));
    if (provided.empty() || provided != config.auth_token) {
        if (out_error) {
            *out_error = "invalid runpod inference bearer token";
        }
        return false;
    }
    return true;
}

bool runpod_relay_inference_request(
        const runpod_inference_runtime_config & config,
        const json & request_body,
        deepseek_chat_result * out_result,
        json * out_error,
        json * out_meta) {
    if (out_meta) {
        *out_meta = json::object();
    }
    if (!out_result) {
        if (out_error) {
            *out_error = format_error_response("RunPod relay requires a non-null result output", ERROR_TYPE_SERVER);
        }
        return false;
    }
    if (!runpod_inference_host_relay_enabled(config)) {
        if (out_error) {
            *out_error = format_error_response("RunPod host relay mode is not enabled", ERROR_TYPE_NOT_SUPPORTED);
        }
        return false;
    }

    try {
        auto [client, parts] = server_http_client(config.relay_url);
        client.set_connection_timeout(std::chrono::milliseconds(config.timeout_ms));
        client.set_read_timeout(std::chrono::milliseconds(config.timeout_ms));
        client.set_write_timeout(std::chrono::milliseconds(config.timeout_ms));

        json preflight_meta = json::object();
        if (!runpod_relay_preflight_health(config, parts, &preflight_meta)) {
            if (out_meta) {
                *out_meta = preflight_meta;
            }
            if (out_error) {
                *out_error = format_error_response(
                        "RunPod inference relay tunnel is unavailable or the pod runtime is not healthy",
                        ERROR_TYPE_UNAVAILABLE);
            }
            return false;
        }

        const httplib::Headers headers = {
            {"Accept", "application/json"},
            {"Authorization", "Bearer " + config.auth_token},
        };
        const std::string path = join_url_path_local(parts, "/v1/chat/completions");
        auto response = client.Post(path.c_str(), headers, request_body.dump(), "application/json");
        if (!response) {
            if (out_error) {
                *out_error = format_error_response(
                        "RunPod inference relay failed before a response was received",
                        ERROR_TYPE_UNAVAILABLE);
            }
            return false;
        }

        const json payload = response->body.empty() ? json::object() : json::parse(response->body);
        if (out_meta) {
            *out_meta = {
                {"http_status", response->status},
                {"endpoint", server_http_show_masked_url(parts)},
                {"preflight_http_status", preflight_meta.value("preflight_http_status", json(nullptr))},
                {"preflight_timeout_ms", preflight_meta.value("preflight_timeout_ms", json(nullptr))},
            };
        }
        if (response->status < 200 || response->status >= 300) {
            if (out_error) {
                *out_error = format_error_response(
                        parse_error_message(payload, server_string_format(
                                "RunPod inference relay failed with HTTP %d",
                                response->status)),
                        response->status == 401 ? ERROR_TYPE_AUTHENTICATION : ERROR_TYPE_UNAVAILABLE);
            }
            return false;
        }
        if (payload.contains("ok")) {
            if (!json_value(payload, "ok", false)) {
                if (out_error) {
                    *out_error = format_error_response(
                            parse_error_message(payload, "RunPod inference relay returned ok=false"),
                            ERROR_TYPE_SERVER);
                }
                return false;
            }
            std::string parse_error;
            if (!deepseek_chat_result_from_json(payload.value("result", json::object()), out_result, &parse_error)) {
                if (out_error) {
                    *out_error = format_error_response(
                            "RunPod inference relay returned an invalid result payload: " + parse_error,
                            ERROR_TYPE_SERVER);
                }
                return false;
            }
            return true;
        }

        if (!runpod_parse_openai_chat_result(payload, out_result, out_error)) {
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = format_error_response(
                    std::string("RunPod inference relay request failed: ") + e.what(),
                    ERROR_TYPE_SERVER);
        }
        return false;
    }
}
