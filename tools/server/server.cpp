#include "server-common.h"
#include "server-deepseek.h"
#include "server-emotive-runtime.h"
#include "server-http.h"
#include "server-runtime.h"
#include "../../common/base64.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <clocale>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <future>
#include <iomanip>
#include <memory>
#include <set>
#include <mutex>
#include <sstream>
#include <signal.h>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;
static std::string extract_json_object_payload(const std::string & text);
static bool execute_deepseek_chat_with_emotive(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const json & body,
        deepseek_chat_result * out_result,
        json * out_error,
        server_emotive_trace * out_trace = nullptr,
        bool cognitive_replay = false,
        const std::string & cognitive_replay_entry_id = std::string(),
        bool enable_heuristic_guidance = true,
        bool suppress_replay_admission = false,
        const std::string & mode_label = std::string());

static std::string trim_copy(std::string value) {
    const auto is_not_space = [](unsigned char ch) {
        return std::isspace(ch) == 0;
    };

    auto begin = std::find_if(value.begin(), value.end(), is_not_space);
    auto end = std::find_if(value.rbegin(), value.rend(), is_not_space).base();
    if (begin >= end) {
        return std::string();
    }
    return std::string(begin, end);
}

static std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static std::string request_header_value(const server_http_req & req, const std::string & key) {
    const std::string needle = to_lower_copy(key);
    for (const auto & [header_key, header_value] : req.headers) {
        if (to_lower_copy(header_key) == needle) {
            return header_value;
        }
    }
    return std::string();
}

static bool parse_truthy_header(const std::string & value) {
    const std::string normalized = to_lower_copy(trim_copy(value));
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

static server_http_res_ptr make_json_response(const json & payload, int status = 200) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data = safe_json_to_str(payload);
    return res;
}

static server_http_res_ptr make_error_response(const json & error) {
    const int status = json_value(error, "code", 500);
    return make_json_response({{"error", error}}, status);
}

struct telegram_outbox_item {
    uint64_t sequence_number = 0;
    std::string kind = "message";
    std::string chat_scope;
    std::string telegram_method = "sendMessage";
    json telegram_payload = json::object();
    std::string text;
    int64_t reply_to_message_id = 0;
    std::string intent;
    std::string dedupe_key;
    double urgency = 0.0;
};

struct telegram_outbox_state {
    mutable std::mutex mutex;
    std::deque<telegram_outbox_item> items;
    uint64_t next_sequence_number = 1;
    size_t max_items = 128;
};

struct telegram_outbox_enqueue_result {
    bool queued = false;
    bool deduplicated = false;
    uint64_t sequence_number = 0;
    std::string chat_scope;
};

struct telegram_bridge_request_context {
    bool active = false;
    bool deferred_delivery = false;
    std::string chat_scope;
    std::string conversation_id;
    int64_t reply_to_message_id = 0;
    int32_t history_turns = 0;
};

struct telegram_delivery_result {
    bool handled = false;
    bool queued = false;
    bool deduplicated = false;
    uint64_t sequence_number = 0;
    std::string chat_scope;
    std::string telegram_method = "sendMessage";
    int64_t reply_to_message_id = 0;
    std::string source;
};

struct telegram_runtime_tool_adapter_config {
    std::string node_bin = "node";
    std::string entry_path;
    int32_t max_rounds = 8;
};

static telegram_bridge_request_context parse_telegram_bridge_request_context(const server_http_req & req) {
    telegram_bridge_request_context context = {};
    context.chat_scope = trim_copy(request_header_value(req, "X-Vicuna-Telegram-Chat-Id"));
    if (context.chat_scope.empty()) {
        return context;
    }

    context.active = true;
    context.deferred_delivery = parse_truthy_header(request_header_value(req, "X-Vicuna-Telegram-Deferred-Delivery"));
    context.conversation_id = trim_copy(request_header_value(req, "X-Vicuna-Telegram-Conversation-Id"));
    context.reply_to_message_id = std::max<int64_t>(
            0,
            std::strtoll(request_header_value(req, "X-Vicuna-Telegram-Message-Id").c_str(), nullptr, 10));
    context.history_turns = std::max<int32_t>(
            0,
            static_cast<int32_t>(std::strtol(request_header_value(req, "X-Vicuna-Telegram-History-Turns").c_str(), nullptr, 10)));
    return context;
}

static std::string build_telegram_bridge_system_prompt() {
    return "You are Vicuña, a helpful personal concierge replying to a Telegram user through the retained transport bridge. "
           "Maintain continuity from the carried Telegram transcript and the current user turn. "
           "Use the live runtime tools made available on this turn instead of claiming you lack access to external systems. "
           "If earlier dialogue said a live system was unavailable, treat that as stale and use the tool that is available now. "
           "Deliver user-visible Telegram replies through telegram_relay instead of plain assistant text. "
           "For simple responses, call telegram_relay with text. "
           "For richer Telegram-native output, send request={method,payload}, prefer sendMessage plus parse_mode and reply_markup for most rich text/button replies, "
           "and prefer parse_mode over raw Telegram entity offsets unless you intentionally provide valid UTF-16-based entities.";
}

static json build_telegram_bridge_relay_tool(const telegram_bridge_request_context & context) {
    std::ostringstream reply_id_description;
    if (context.reply_to_message_id > 0) {
        reply_id_description << "Optional Telegram message id to use as the reply anchor. Use "
                             << context.reply_to_message_id
                             << " to reply to the current user turn.";
    } else {
        reply_id_description << "Optional Telegram message id to use as the reply anchor.";
    }

    return json{
        {"type", "function"},
        {"function", {
            {"name", "telegram_relay"},
            {"description", "Queue one Telegram follow-up message through the provider-only bridge outbox."},
            {"parameters", {
                {"type", "object"},
                {"description", "The Telegram relay payload. Use text for simple replies or request={method,payload} for structured Telegram-native output."},
                {"anyOf", json::array({
                    json{{"required", json::array({"text"})}},
                    json{{"required", json::array({"request"})}},
                })},
                {"properties", {
                    {"text", {
                        {"type", "string"},
                        {"description", "Optional simple plain-text Telegram reply."},
                    }},
                    {"request", {
                        {"type", "object"},
                        {"description", "Optional structured Telegram Bot API send request. Allowed methods: sendMessage, sendPhoto, sendDocument, sendAudio, sendVoice, sendVideo, sendAnimation, sendSticker, sendMediaGroup, sendLocation, sendVenue, sendContact, sendPoll, or sendDice. Do not include chat_id; the relay fills routing."},
                        {"required", json::array({"method", "payload"})},
                        {"properties", {
                            {"method", {
                                {"type", "string"},
                                {"description", "Allowed outbound Telegram method, such as sendMessage, sendPhoto, sendDocument, sendAudio, sendVoice, sendVideo, sendAnimation, sendSticker, sendMediaGroup, sendLocation, sendVenue, sendContact, sendPoll, or sendDice."},
                            }},
                            {"payload", {
                                {"type", "object"},
                                {"description", "Telegram Bot API payload for the selected method. Prefer sendMessage plus parse_mode, link_preview_options, and reply_markup for most rich text and button replies."},
                            }},
                        }},
                    }},
                    {"chat_scope", {
                        {"type", "string"},
                        {"description", "The Telegram chat scope to route the follow-up into. Use " + json(context.chat_scope).dump() + " for this turn."},
                    }},
                    {"reply_to_message_id", {
                        {"type", "integer"},
                        {"minimum", 1},
                        {"description", reply_id_description.str()},
                    }},
                    {"intent", {
                        {"type", "string"},
                        {"description", "Optional intent label that classifies the follow-up, such as question or conclusion."},
                    }},
                    {"dedupe_key", {
                        {"type", "string"},
                        {"description", "Optional dedupe key used to suppress duplicate queued follow-ups for the same chat."},
                    }},
                    {"urgency", {
                        {"type", "number"},
                        {"minimum", 0},
                        {"description", "Optional normalized urgency score for the queued follow-up."},
                    }},
                }},
            }},
            {"x-vicuna-family-id", "telegram"},
            {"x-vicuna-family-name", "Telegram"},
            {"x-vicuna-family-description", "Send direct user-facing follow-up messages through the Telegram bridge outbox."},
            {"x-vicuna-method-name", "relay"},
            {"x-vicuna-method-description", "Queue one Telegram follow-up message as plain text or a structured Bot API send request."},
        }},
    };
}

static telegram_runtime_tool_adapter_config telegram_runtime_tool_adapter_config_from_env() {
    telegram_runtime_tool_adapter_config config;
    if (const char * value = std::getenv("VICUNA_OPENCLAW_NODE_BIN")) {
        const std::string parsed = trim_copy(value);
        if (!parsed.empty()) {
            config.node_bin = parsed;
        }
    }

    if (const char * value = std::getenv("VICUNA_OPENCLAW_ENTRY_PATH")) {
        config.entry_path = trim_copy(value);
    } else if (const char * value = std::getenv("TELEGRAM_BRIDGE_OPENCLAW_ENTRY_PATH")) {
        config.entry_path = trim_copy(value);
    }

    if (config.entry_path.empty()) {
        try {
            const std::filesystem::path cwd = std::filesystem::current_path();
            const std::filesystem::path source_repo_root =
                    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
            const std::vector<std::filesystem::path> candidates = {
                source_repo_root / "tools" / "openclaw-harness" / "dist" / "index.js",
                cwd / "tools" / "openclaw-harness" / "dist" / "index.js",
                cwd / ".." / "tools" / "openclaw-harness" / "dist" / "index.js",
                cwd / ".." / ".." / "tools" / "openclaw-harness" / "dist" / "index.js",
            };
            for (const auto & candidate : candidates) {
                if (std::filesystem::exists(candidate)) {
                    config.entry_path = candidate.lexically_normal().string();
                    break;
                }
            }
        } catch (const std::exception &) {
        }
        if (config.entry_path.empty()) {
            config.entry_path = "tools/openclaw-harness/dist/index.js";
        }
    }

    if (const char * value = std::getenv("VICUNA_TELEGRAM_RUNTIME_MAX_ROUNDS")) {
        const int parsed = std::atoi(value);
        if (parsed > 0) {
            config.max_rounds = parsed;
        }
    }
    return config;
}

static std::string shell_escape_single_quoted(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    escaped.push_back('\'');
    for (const char ch : value) {
        if (ch == '\'') {
            escaped += "'\"'\"'";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

static bool run_shell_json_command(
        const std::string & command_line,
        json * out_json,
        std::string * out_error) {
    if (!out_json) {
        if (out_error) {
            *out_error = "json command output must not be null";
        }
        return false;
    }

#if defined(_WIN32)
    FILE * pipe = _popen(command_line.c_str(), "r");
#else
    FILE * pipe = popen(command_line.c_str(), "r");
#endif
    if (!pipe) {
        if (out_error) {
            *out_error = "failed to open subprocess pipe";
        }
        return false;
    }

    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

#if defined(_WIN32)
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif

    const std::string trimmed = trim_copy(output);
    if (status != 0) {
        if (out_error) {
            *out_error = trimmed.empty() ?
                    server_string_format("subprocess failed with status %d", status) :
                    trimmed;
        }
        return false;
    }
    if (trimmed.empty()) {
        if (out_error) {
            *out_error = "subprocess produced no JSON output";
        }
        return false;
    }

    try {
        *out_json = json::parse(trimmed);
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("subprocess returned invalid JSON: %s", e.what());
        }
        return false;
    }
}

static bool parse_runtime_tools_override(json * out_payload, std::string * out_error) {
    const char * value = std::getenv("VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON");
    if (!value || trim_copy(value).empty()) {
        return false;
    }
    if (!out_payload) {
        if (out_error) {
            *out_error = "runtime tools output must not be null";
        }
        return true;
    }
    try {
        *out_payload = json::parse(value);
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON was not valid JSON: %s", e.what());
        }
        return true;
    }
}

static bool load_server_owned_telegram_runtime_tools(
        const telegram_runtime_tool_adapter_config & config,
        const telegram_bridge_request_context & context,
        json * out_tools,
        std::string * out_error) {
    if (!out_tools) {
        if (out_error) {
            *out_error = "runtime tool output must not be null";
        }
        return false;
    }

    json payload;
    std::string command_error;
    const bool used_override = parse_runtime_tools_override(&payload, &command_error);
    if (used_override && !command_error.empty()) {
        if (out_error) {
            *out_error = command_error;
        }
        return false;
    }

    if (!used_override) {
        const std::string command_line =
                shell_escape_single_quoted(config.node_bin) + " " +
                shell_escape_single_quoted(config.entry_path) + " runtime-tools --exclude-tool-name=telegram_relay 2>&1";
        if (!run_shell_json_command(command_line, &payload, &command_error)) {
            if (out_error) {
                *out_error = "unable to load Telegram runtime tool catalog: " + command_error;
            }
            return false;
        }
    }

    json tools = payload;
    if (payload.is_object()) {
        tools = json_value(payload, "tools", json::array());
    }
    if (!tools.is_array()) {
        if (out_error) {
            *out_error = "Telegram runtime tool catalog did not contain a tools array";
        }
        return false;
    }

    json final_tools = tools;
    final_tools.push_back(build_telegram_bridge_relay_tool(context));
    *out_tools = std::move(final_tools);
    return true;
}

static bool lookup_runtime_tool_observation_override(
        const std::string & tool_name,
        json * out_observation,
        std::string * out_error) {
    const char * value = std::getenv("VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON");
    if (!value || trim_copy(value).empty()) {
        return false;
    }
    try {
        const json payload = json::parse(value);
        json observations = payload;
        if (payload.is_object() && payload.contains("observations")) {
            observations = payload.at("observations");
        }
        if (!observations.is_object()) {
            if (out_error) {
                *out_error = "VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON must decode to an object";
            }
            return true;
        }
        if (!observations.contains(tool_name)) {
            if (out_error) {
                *out_error = "VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON does not contain " + tool_name;
            }
            return true;
        }
        if (out_observation) {
            *out_observation = observations.at(tool_name);
        }
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON was not valid JSON: %s", e.what());
        }
        return true;
    }
}

static bool invoke_server_owned_telegram_runtime_tool(
        const telegram_runtime_tool_adapter_config & config,
        const deepseek_tool_call & tool_call,
        json * out_observation,
        std::string * out_error) {
    if (!out_observation) {
        if (out_error) {
            *out_error = "runtime tool observation output must not be null";
        }
        return false;
    }

    std::string override_error;
    if (lookup_runtime_tool_observation_override(tool_call.name, out_observation, &override_error)) {
        if (!override_error.empty()) {
            if (out_error) {
                *out_error = override_error;
            }
            return false;
        }
        return true;
    }

    json payload;
    std::string command_error;
    const std::string arguments_base64 = base64::encode(tool_call.arguments_json);
    const std::string command_line =
            shell_escape_single_quoted(config.node_bin) + " " +
            shell_escape_single_quoted(config.entry_path) + " invoke-runtime " +
            "--tool-name=" + shell_escape_single_quoted(tool_call.name) + " " +
            "--arguments-base64=" + shell_escape_single_quoted(arguments_base64) +
            " 2>&1";
    if (!run_shell_json_command(command_line, &payload, &command_error)) {
        if (out_error) {
            *out_error = "runtime tool execution failed for " + tool_call.name + ": " + command_error;
        }
        return false;
    }

    *out_observation = payload.contains("observation") ? payload.at("observation") : json::object();
    return true;
}

static json build_bridge_assistant_tool_replay_message(const deepseek_chat_result & result) {
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

    json message = {
        {"role", "assistant"},
        {"content", result.content},
        {"tool_calls", std::move(tool_calls)},
    };
    if (!result.reasoning_content.empty()) {
        message["reasoning_content"] = result.reasoning_content;
    }
    return message;
}

static json build_bridge_tool_result_message(const deepseek_tool_call & tool_call, const json & observation) {
    return {
        {"role", "tool"},
        {"tool_call_id", tool_call.id},
        {"name", tool_call.name},
        {"content", observation.is_string() ? observation.get<std::string>() : safe_json_to_str(observation)},
    };
}

static json build_server_owned_bridge_request_body(
        const json & body,
        const telegram_bridge_request_context & context,
        const json & runtime_tools) {
    json augmented = body;
    json messages = json::array();
    messages.push_back({
        {"role", "system"},
        {"content", build_telegram_bridge_system_prompt()},
    });
    for (const auto & item : body.at("messages")) {
        messages.push_back(item);
    }
    augmented["messages"] = std::move(messages);
    augmented["tools"] = runtime_tools;
    augmented["tool_choice"] = "auto";
    augmented["parallel_tool_calls"] = false;
    (void) context;
    return augmented;
}

struct request_activity_state {
    std::atomic<int32_t> active_requests = 0;
    mutable std::mutex mutex;
    std::chrono::steady_clock::time_point last_foreground_activity = std::chrono::steady_clock::now();

    void note_request_start() {
        active_requests.fetch_add(1);
        std::lock_guard<std::mutex> lock(mutex);
        last_foreground_activity = std::chrono::steady_clock::now();
    }

    void note_request_end() {
        active_requests.fetch_sub(1);
        std::lock_guard<std::mutex> lock(mutex);
        last_foreground_activity = std::chrono::steady_clock::now();
    }

    int64_t idle_ms() const {
        std::lock_guard<std::mutex> lock(mutex);
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - last_foreground_activity).count();
    }
};

struct cognitive_replay_worker_state {
    mutable std::mutex mutex;
    bool running = false;
    std::string active_entry_id;
    std::string last_error;
    int64_t last_started_at_ms = 0;
    int64_t last_finished_at_ms = 0;
};

struct server_ongoing_task_config {
    bool enabled = false;
    std::string base_url = "https://api.supermemory.ai";
    std::string auth_token;
    std::string container_tag = "vicuna";
    std::string runtime_identity = "vicuna";
    std::string registry_key = "ongoing-tasks-registry";
    std::string registry_title = "Ongoing task registry";
    float query_threshold = 0.0f;
    int32_t poll_interval_ms = 60000;
    int32_t timeout_ms = 5000;
};

struct server_ongoing_task_frequency {
    int32_t interval = 1;
    std::string unit = "days";
};

struct server_ongoing_task_record {
    std::string task_id;
    std::string task_text;
    server_ongoing_task_frequency frequency;
    std::string created_at;
    std::string updated_at;
    std::string last_done_at;
    bool active = true;
};

struct server_ongoing_task_summary {
    std::string task_id;
    std::string task_text;
    server_ongoing_task_frequency frequency;
    std::string last_done_at;
    std::string next_due_at;
    bool due_now = false;
    bool active = true;
};

struct server_ongoing_task_registry {
    int32_t schema_version = 1;
    std::string updated_at;
    std::vector<server_ongoing_task_record> tasks;
};

struct server_ongoing_task_decision {
    bool valid = false;
    bool should_run = false;
    std::string selected_task_id;
    std::string rationale;
    int64_t decided_at_ms = 0;
    std::string current_time_iso;
    int32_t task_count = 0;
};

struct ongoing_task_worker_state {
    mutable std::mutex mutex;
    bool running = false;
    std::string mode = "idle";
    std::string active_task_id;
    std::string last_error;
    int64_t last_poll_at_ms = 0;
    int64_t last_finished_at_ms = 0;
    std::string last_completed_task_id;
    std::string last_completed_at_iso;
    server_ongoing_task_decision last_decision;
};

enum staged_tool_stage_kind {
    STAGED_TOOL_STAGE_FAMILY_SELECT,
    STAGED_TOOL_STAGE_METHOD_SELECT,
    STAGED_TOOL_STAGE_PAYLOAD_BUILD,
};

enum staged_tool_method_choice_kind {
    STAGED_TOOL_METHOD_CHOOSE_METHOD,
    STAGED_TOOL_METHOD_GO_BACK,
    STAGED_TOOL_METHOD_COMPLETE,
};

enum staged_tool_payload_choice_kind {
    STAGED_TOOL_PAYLOAD_SUBMIT,
    STAGED_TOOL_PAYLOAD_GO_BACK,
};

struct staged_tool_method {
    std::string tool_name;
    std::string method_name;
    std::string method_description;
    json parameters = json::object();
    json original_tool = json::object();
};

struct staged_tool_family {
    std::string family_id;
    std::string family_name;
    std::string family_description;
    std::vector<staged_tool_method> methods;
};

struct staged_tool_catalog {
    std::vector<staged_tool_family> families;
};

struct staged_tool_method_choice {
    staged_tool_method_choice_kind kind = STAGED_TOOL_METHOD_CHOOSE_METHOD;
    std::string method_name;
};

struct staged_tool_payload_choice {
    staged_tool_payload_choice_kind kind = STAGED_TOOL_PAYLOAD_SUBMIT;
    json payload = json::object();
};

static std::string staged_tool_titleize(const std::string & raw) {
    std::ostringstream out;
    bool capitalize = true;
    for (unsigned char ch : raw) {
        if (ch == '_' || ch == '-' || ch == '.') {
            out << ' ';
            capitalize = true;
            continue;
        }
        out << static_cast<char>(capitalize ? std::toupper(ch) : ch);
        capitalize = false;
    }
    return trim_copy(out.str());
}

static std::vector<std::string> staged_tool_known_family_prefixes() {
    return {
        "ongoing_tasks",
        "hard_memory",
        "parsed_documents",
        "web_search",
        "telegram",
        "radarr",
        "sonarr",
        "chaptarr",
        "runtime",
    };
}

static std::string derive_staged_tool_family_id(const std::string & tool_name) {
    for (const auto & prefix : staged_tool_known_family_prefixes()) {
        if (tool_name == prefix || tool_name.rfind(prefix + "_", 0) == 0) {
            return prefix;
        }
    }

    const size_t first_underscore = tool_name.find('_');
    if (first_underscore == std::string::npos) {
        return trim_copy(tool_name);
    }

    const size_t second_underscore = tool_name.find('_', first_underscore + 1);
    if (second_underscore != std::string::npos) {
        return trim_copy(tool_name.substr(0, second_underscore));
    }
    return trim_copy(tool_name.substr(0, first_underscore));
}

static std::string derive_staged_method_name(const std::string & tool_name, const std::string & family_id) {
    const std::string prefix = family_id + "_";
    if (tool_name.rfind(prefix, 0) == 0 && tool_name.size() > prefix.size()) {
        return trim_copy(tool_name.substr(prefix.size()));
    }
    return trim_copy(tool_name);
}

static std::string summarize_staged_family_description(const staged_tool_family & family) {
    if (!family.family_description.empty()) {
        return family.family_description;
    }
    if (family.methods.empty()) {
        return "Use this family to take the next relevant action.";
    }
    if (family.methods.size() == 1) {
        return family.methods.front().method_description;
    }
    std::ostringstream out;
    out << "Use this family to";
    for (size_t index = 0; index < family.methods.size() && index < 2; ++index) {
        out << (index == 0 ? " " : ", ");
        out << trim_copy(family.methods[index].method_description);
    }
    if (family.methods.size() > 2) {
        out << ", and related actions.";
    } else {
        out << ".";
    }
    return trim_copy(out.str());
}

static bool staged_tool_schema_has_descriptions(const json & schema) {
    if (!schema.is_object()) {
        return true;
    }
    if ((schema.contains("type") || schema.contains("properties") || schema.contains("items")) &&
            !schema.value("description", std::string()).empty()) {
        // fall through
    } else if (schema.contains("type") || schema.contains("properties") || schema.contains("items")) {
        return false;
    }

    if (schema.contains("properties") && schema.at("properties").is_object()) {
        for (const auto & item : schema.at("properties").items()) {
            if (!staged_tool_schema_has_descriptions(item.value())) {
                return false;
            }
        }
    }
    if (schema.contains("items")) {
        const auto & items = schema.at("items");
        if (items.is_array()) {
            for (const auto & item : items) {
                if (!staged_tool_schema_has_descriptions(item)) {
                    return false;
                }
            }
        } else if (!staged_tool_schema_has_descriptions(items)) {
            return false;
        }
    }
    return true;
}

static staged_tool_catalog build_staged_tool_catalog_from_request(const json & tools) {
    staged_tool_catalog catalog = {};
    if (!tools.is_array()) {
        return catalog;
    }

    std::map<std::string, size_t> family_index_by_id;
    for (const auto & item : tools) {
        if (!item.is_object() || json_value(item, "type", std::string()) != "function" || !item.contains("function")) {
            continue;
        }
        const json & function = item.at("function");
        if (!function.is_object()) {
            continue;
        }

        const std::string tool_name = trim_copy(json_value(function, "name", std::string()));
        if (tool_name.empty()) {
            continue;
        }

        const json parameters = function.contains("parameters") ? function.at("parameters") : json::object();
        if (!parameters.is_object()) {
            continue;
        }

        const std::string family_id = trim_copy(
                json_value(function, "x-vicuna-family-id", std::string()).empty() ?
                        derive_staged_tool_family_id(tool_name) :
                        json_value(function, "x-vicuna-family-id", std::string()));
        const std::string family_name = trim_copy(
                json_value(function, "x-vicuna-family-name", std::string()).empty() ?
                        staged_tool_titleize(family_id) :
                        json_value(function, "x-vicuna-family-name", std::string()));
        const std::string family_description = trim_copy(json_value(function, "x-vicuna-family-description", std::string()));
        const std::string method_name = trim_copy(
                json_value(function, "x-vicuna-method-name", std::string()).empty() ?
                        derive_staged_method_name(tool_name, family_id) :
                        json_value(function, "x-vicuna-method-name", std::string()));
        const std::string method_description = trim_copy(
                json_value(function, "x-vicuna-method-description", std::string()).empty() ?
                        json_value(function, "description", std::string()) :
                        json_value(function, "x-vicuna-method-description", std::string()));
        if (family_id.empty() || family_name.empty() || method_name.empty() || method_description.empty()) {
            continue;
        }
        if (!staged_tool_schema_has_descriptions(parameters)) {
            continue;
        }

        size_t family_index = 0;
        auto existing = family_index_by_id.find(family_id);
        if (existing == family_index_by_id.end()) {
            family_index = catalog.families.size();
            family_index_by_id.emplace(family_id, family_index);
            catalog.families.push_back({
                family_id,
                family_name,
                family_description,
                {},
            });
        } else {
            family_index = existing->second;
        }

        catalog.families[family_index].methods.push_back({
            tool_name,
            method_name,
            method_description,
            parameters,
            item,
        });
    }

    for (auto & family : catalog.families) {
        if (family.family_description.empty()) {
            family.family_description = summarize_staged_family_description(family);
        }
        std::sort(family.methods.begin(), family.methods.end(), [](const staged_tool_method & lhs, const staged_tool_method & rhs) {
            return lhs.method_name < rhs.method_name;
        });
    }
    std::sort(catalog.families.begin(), catalog.families.end(), [](const staged_tool_family & lhs, const staged_tool_family & rhs) {
        return lhs.family_name < rhs.family_name;
    });
    return catalog;
}

static const staged_tool_family * staged_tool_find_family(
        const staged_tool_catalog & catalog,
        const std::string & family_name) {
    for (const auto & family : catalog.families) {
        if (family.family_name == family_name) {
            return &family;
        }
    }
    return nullptr;
}

static const staged_tool_method * staged_tool_find_method(
        const staged_tool_family & family,
        const std::string & method_name) {
    for (const auto & method : family.methods) {
        if (method.method_name == method_name) {
            return &method;
        }
    }
    return nullptr;
}

static std::string build_staged_tool_core_system_prompt() {
    return
            "You are Vicuña, a helpful personal concierge with access to tool families. "
            "Choose deliberately, stay grounded in the conversation, and use tools when they let "
            "you help the user more directly or safely.";
}

static std::string build_staged_family_selection_prompt(const staged_tool_catalog & catalog) {
    std::ostringstream out;
    out << "You are choosing one tool family.\n";
    out << "Return exactly one JSON object with this shape: {\"family\":\"<exact family name>\"}.\n";
    out << "Do not add any keys, markdown, or explanation.\n";
    out << "Available families:\n";
    for (const auto & family : catalog.families) {
        out << "- " << family.family_name << ": " << family.family_description << "\n";
    }
    return out.str();
}

static std::string build_staged_method_selection_prompt(
        const staged_tool_family & family,
        bool allow_completion) {
    std::ostringstream out;
    out << "You are choosing a method of the " << family.family_name << " tool family.\n";
    out << "Return exactly one JSON object with this shape: {\"method\":\"<exact method name>\"}.\n";
    out << "Allowed sentinels:\n";
    out << "- {\"method\":\"back\"} to go back to tool-family selection.\n";
    if (allow_completion) {
        out << "- {\"method\":\"complete\"} if the active tool loop is finished and you should stop selecting tools.\n";
    }
    out << "Available methods:\n";
    for (const auto & method : family.methods) {
        out << "- " << method.method_name << ": " << method.method_description << "\n";
    }
    return out.str();
}

static void append_staged_contract_lines(
        std::ostringstream & out,
        const std::string & path,
        const json & schema,
        int depth = 0) {
    if (!schema.is_object()) {
        return;
    }
    const std::string indent(static_cast<size_t>(depth) * 2, ' ');
    const std::string type = json_value(schema, "type", std::string("object"));
    const std::string description = trim_copy(json_value(schema, "description", std::string()));
    out << indent << "- " << path << " [" << type << "]";
    if (!description.empty()) {
        out << ": " << description;
    }
    if (schema.contains("enum") && schema.at("enum").is_array()) {
        out << " allowed=" << schema.at("enum").dump();
    }
    out << "\n";

    if (schema.contains("properties") && schema.at("properties").is_object()) {
        const std::set<std::string> required = [&]() {
            std::set<std::string> names;
            if (schema.contains("required") && schema.at("required").is_array()) {
                for (const auto & item : schema.at("required")) {
                    if (item.is_string()) {
                        names.insert(item.get<std::string>());
                    }
                }
            }
            return names;
        }();
        for (const auto & item : schema.at("properties").items()) {
            append_staged_contract_lines(
                    out,
                    path + "." + item.key() + (required.count(item.key()) > 0 ? " (required)" : ""),
                    item.value(),
                    depth + 1);
        }
    }
    if (schema.contains("items")) {
        append_staged_contract_lines(out, path + "[]", schema.at("items"), depth + 1);
    }
}

static std::string build_staged_payload_prompt(
        const staged_tool_family & family,
        const staged_tool_method & method,
        const std::string & validation_error = std::string()) {
    std::ostringstream out;
    out << "You are constructing a payload for the " << method.method_name
        << " method of the " << family.family_name << " tool family.\n";
    out << "Return exactly one JSON object.\n";
    out << "Use {\"action\":\"back\"} to go back to method selection.\n";
    out << "Otherwise use {\"action\":\"submit\",\"payload\":{...}} with a payload that satisfies the typed contract.\n";
    if (!validation_error.empty()) {
        out << "Previous payload error: " << validation_error << "\n";
    }
    out << "Method description: " << method.method_description << "\n";
    out << "Typed contract:\n";
    append_staged_contract_lines(out, "payload", method.parameters);
    out << "Return JSON only.\n";
    return out.str();
}

static json build_staged_messages(
        const json & original_messages,
        const std::string & stage_prompt) {
    json messages = json::array();
    messages.push_back({
        {"role", "system"},
        {"content", build_staged_tool_core_system_prompt()},
    });
    if (original_messages.is_array()) {
        for (const auto & item : original_messages) {
            messages.push_back(item);
        }
    }
    messages.push_back({
        {"role", "system"},
        {"content", stage_prompt},
    });
    return messages;
}

static json build_staged_provider_body(const json & base_body, const json & messages) {
    static constexpr int64_t staged_max_tokens = 1024;
    json staged = base_body;
    staged.erase("tools");
    staged.erase("tool_choice");
    staged.erase("parallel_tool_calls");
    staged["messages"] = messages;
    staged["stream"] = false;
    staged["max_tokens"] = staged_max_tokens;
    staged["response_format"] = {
        {"type", "json_object"},
    };
    return staged;
}

static std::string build_staged_retry_prompt(
        const std::string & stage_prompt,
        const std::string & validation_error) {
    if (validation_error.empty()) {
        return stage_prompt;
    }
    std::ostringstream out;
    out << stage_prompt;
    out << "Previous response error: " << validation_error << "\n";
    out << "Return exactly one non-empty JSON object with no prose, markdown, or code fences.\n";
    return out.str();
}

static bool parse_staged_family_selection_response(
        const std::string & text,
        const staged_tool_catalog & catalog,
        std::string * out_family_name,
        std::string * out_error) {
    try {
        const json payload = json::parse(extract_json_object_payload(text));
        const std::string family_name = trim_copy(json_value(payload, "family", std::string()));
        if (family_name.empty()) {
            if (out_error) {
                *out_error = "staged family selection did not include family";
            }
            return false;
        }
        if (!staged_tool_find_family(catalog, family_name)) {
            if (out_error) {
                *out_error = "staged family selection chose an unknown family";
            }
            return false;
        }
        if (out_family_name) {
            *out_family_name = family_name;
        }
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("staged family selection was not valid JSON: %s", e.what());
        }
        return false;
    }
}

static bool parse_staged_method_selection_response(
        const std::string & text,
        const staged_tool_family & family,
        bool allow_completion,
        staged_tool_method_choice * out_choice,
        std::string * out_error) {
    if (!out_choice) {
        if (out_error) {
            *out_error = "staged method selection output must not be null";
        }
        return false;
    }
    try {
        const json payload = json::parse(extract_json_object_payload(text));
        const std::string method_name = trim_copy(json_value(payload, "method", std::string()));
        if (method_name == "back") {
            out_choice->kind = STAGED_TOOL_METHOD_GO_BACK;
            out_choice->method_name.clear();
            return true;
        }
        if (allow_completion && method_name == "complete") {
            out_choice->kind = STAGED_TOOL_METHOD_COMPLETE;
            out_choice->method_name.clear();
            return true;
        }
        if (!staged_tool_find_method(family, method_name)) {
            if (out_error) {
                *out_error = "staged method selection chose an unknown method";
            }
            return false;
        }
        out_choice->kind = STAGED_TOOL_METHOD_CHOOSE_METHOD;
        out_choice->method_name = method_name;
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("staged method selection was not valid JSON: %s", e.what());
        }
        return false;
    }
}

static bool validate_payload_against_schema(
        const json & value,
        const json & schema,
        std::string * out_error,
        const std::string & path = "payload") {
    if (!schema.is_object()) {
        return true;
    }
    const std::string type = json_value(schema, "type", std::string());
    if (type == "object") {
        if (!value.is_object()) {
            if (out_error) {
                *out_error = path + " must be an object";
            }
            return false;
        }
        if (schema.contains("required") && schema.at("required").is_array()) {
            for (const auto & item : schema.at("required")) {
                if (item.is_string() && !value.contains(item.get<std::string>())) {
                    if (out_error) {
                        *out_error = path + "." + item.get<std::string>() + " is required";
                    }
                    return false;
                }
            }
        }
        if (schema.contains("properties") && schema.at("properties").is_object()) {
            for (const auto & item : schema.at("properties").items()) {
                if (value.contains(item.key()) &&
                        !validate_payload_against_schema(value.at(item.key()), item.value(), out_error, path + "." + item.key())) {
                    return false;
                }
            }
        }
    } else if (type == "array") {
        if (!value.is_array()) {
            if (out_error) {
                *out_error = path + " must be an array";
            }
            return false;
        }
        if (schema.contains("minItems")) {
            const size_t min_items = static_cast<size_t>(json_value(schema, "minItems", int32_t(0)));
            if (value.size() < min_items) {
                if (out_error) {
                    *out_error = path + " must contain at least " + std::to_string(min_items) + " items";
                }
                return false;
            }
        }
        if (schema.contains("items")) {
            size_t index = 0;
            for (const auto & item : value) {
                if (!validate_payload_against_schema(item, schema.at("items"), out_error, path + "[" + std::to_string(index) + "]")) {
                    return false;
                }
                ++index;
            }
        }
    } else if (type == "string") {
        if (!value.is_string()) {
            if (out_error) {
                *out_error = path + " must be a string";
            }
            return false;
        }
    } else if (type == "boolean") {
        if (!value.is_boolean()) {
            if (out_error) {
                *out_error = path + " must be a boolean";
            }
            return false;
        }
    } else if (type == "integer") {
        if (!value.is_number_integer()) {
            if (out_error) {
                *out_error = path + " must be an integer";
            }
            return false;
        }
    } else if (type == "number") {
        if (!value.is_number()) {
            if (out_error) {
                *out_error = path + " must be a number";
            }
            return false;
        }
    }

    if (schema.contains("enum") && schema.at("enum").is_array()) {
        bool matched = false;
        for (const auto & candidate : schema.at("enum")) {
            if (candidate == value) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            if (out_error) {
                *out_error = path + " must match one of the allowed enum values";
            }
            return false;
        }
    }
    return true;
}

static bool parse_staged_payload_response(
        const std::string & text,
        const staged_tool_method & method,
        staged_tool_payload_choice * out_choice,
        std::string * out_error) {
    if (!out_choice) {
        if (out_error) {
            *out_error = "staged payload output must not be null";
        }
        return false;
    }
    try {
        const json payload = json::parse(extract_json_object_payload(text));
        const std::string action = trim_copy(json_value(payload, "action", std::string()));
        if (action == "back") {
            out_choice->kind = STAGED_TOOL_PAYLOAD_GO_BACK;
            out_choice->payload = json::object();
            return true;
        }
        if (action != "submit" || !payload.contains("payload")) {
            if (out_error) {
                *out_error = "staged payload response must use action=submit with payload or action=back";
            }
            return false;
        }
        std::string validation_error;
        if (!validate_payload_against_schema(payload.at("payload"), method.parameters, &validation_error)) {
            if (out_error) {
                *out_error = validation_error;
            }
            return false;
        }
        out_choice->kind = STAGED_TOOL_PAYLOAD_SUBMIT;
        out_choice->payload = payload.at("payload");
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("staged payload response was not valid JSON: %s", e.what());
        }
        return false;
    }
}

static bool execute_staged_selection_turn(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const json & base_body,
        const std::string & stage_prompt,
        deepseek_chat_result * out_result,
        json * out_error,
        bool suppress_replay_admission = false,
        const std::string & mode_label = std::string()) {
    const json messages = build_staged_messages(base_body.at("messages"), stage_prompt);
    const json staged_body = build_staged_provider_body(base_body, messages);
    return execute_deepseek_chat_with_emotive(
            config,
            emotive_runtime,
            staged_body,
            out_result,
            out_error,
            nullptr,
            false,
            std::string(),
            true,
            suppress_replay_admission,
            mode_label);
}

static bool execute_final_completion_after_staged_loop(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const json & body,
        deepseek_chat_result * out_result,
        json * out_error,
        bool suppress_replay_admission = false,
        const std::string & mode_label = std::string()) {
    json completion_body = body;
    completion_body.erase("tools");
    completion_body.erase("tool_choice");
    completion_body.erase("parallel_tool_calls");
    completion_body["messages"] = build_staged_messages(
            body.at("messages"),
            "The staged tool loop is complete. Reply directly to the user or conclude the task without JSON.");
    return execute_deepseek_chat_with_emotive(
            config,
            emotive_runtime,
            completion_body,
            out_result,
            out_error,
            nullptr,
            false,
            std::string(),
            true,
            suppress_replay_admission,
            mode_label);
}

static bool should_use_staged_tool_loop_for_request(const json & body) {
    if (!body.contains("tools") || !body.at("tools").is_array() || body.at("tools").empty()) {
        return false;
    }
    if (body.contains("tool_choice")) {
        const auto & tool_choice = body.at("tool_choice");
        if ((tool_choice.is_string() && tool_choice.get<std::string>() == "none") ||
                (tool_choice.is_object() && json_value(tool_choice, "type", std::string()) == "function")) {
            return false;
        }
    }
    return body.contains("messages") && body.at("messages").is_array();
}

static bool execute_deepseek_chat_with_staged_tools(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const json & body,
        deepseek_chat_result * out_result,
        json * out_error,
        bool suppress_replay_admission = false,
        const std::string & mode_label = std::string()) {
    if (!out_result) {
        if (out_error) {
            *out_error = format_error_response("staged tool loop requires a result output", ERROR_TYPE_SERVER);
        }
        return false;
    }
    *out_result = deepseek_chat_result();

    const staged_tool_catalog catalog = build_staged_tool_catalog_from_request(body.at("tools"));
    if (catalog.families.empty()) {
        return execute_deepseek_chat_with_emotive(
                config,
                emotive_runtime,
                body,
                out_result,
                out_error,
                nullptr,
                false,
                std::string(),
                true,
                suppress_replay_admission,
                mode_label);
    }

    const int max_stage_turns = 24;
    const int max_stage_json_attempts = 2;
    int stage_turn = 0;
    std::string selected_family_name;
    while (stage_turn < max_stage_turns) {
        bool family_valid = false;
        std::string family_validation_error;
        for (int family_attempt = 0; family_attempt < max_stage_json_attempts && stage_turn < max_stage_turns; ++family_attempt) {
            ++stage_turn;
            deepseek_chat_result family_result;
            if (!execute_staged_selection_turn(
                        config,
                        emotive_runtime,
                        body,
                        build_staged_retry_prompt(
                                build_staged_family_selection_prompt(catalog),
                                family_validation_error),
                        &family_result,
                        out_error,
                        suppress_replay_admission,
                        mode_label.empty() ? "staged_family_select" : mode_label + "_family_select")) {
                return false;
            }

            std::string parse_error;
            if (parse_staged_family_selection_response(family_result.content, catalog, &selected_family_name, &parse_error)) {
                family_valid = true;
                break;
            }
            family_validation_error = parse_error;
        }
        if (!family_valid) {
            if (out_error) {
                *out_error = format_error_response(family_validation_error, ERROR_TYPE_SERVER);
            }
            return false;
        }
        const staged_tool_family * family = staged_tool_find_family(catalog, selected_family_name);
        if (!family) {
            if (out_error) {
                *out_error = format_error_response("staged tool loop selected a missing family", ERROR_TYPE_SERVER);
            }
            return false;
        }

        while (stage_turn < max_stage_turns) {
            staged_tool_method_choice method_choice = {};
            bool method_valid = false;
            std::string method_validation_error;
            for (int method_attempt = 0; method_attempt < max_stage_json_attempts && stage_turn < max_stage_turns; ++method_attempt) {
                ++stage_turn;
                deepseek_chat_result method_result;
                if (!execute_staged_selection_turn(
                            config,
                            emotive_runtime,
                            body,
                            build_staged_retry_prompt(
                                    build_staged_method_selection_prompt(*family, true),
                                    method_validation_error),
                            &method_result,
                            out_error,
                            suppress_replay_admission,
                            mode_label.empty() ? "staged_method_select" : mode_label + "_method_select")) {
                    return false;
                }

                std::string method_error;
                if (parse_staged_method_selection_response(method_result.content, *family, true, &method_choice, &method_error)) {
                    method_valid = true;
                    break;
                }
                method_validation_error = method_error;
            }
            if (!method_valid) {
                if (out_error) {
                    *out_error = format_error_response(method_validation_error, ERROR_TYPE_SERVER);
                }
                return false;
            }

            if (method_choice.kind == STAGED_TOOL_METHOD_GO_BACK) {
                break;
            }
            if (method_choice.kind == STAGED_TOOL_METHOD_COMPLETE) {
                return execute_final_completion_after_staged_loop(
                        config,
                        emotive_runtime,
                        body,
                        out_result,
                        out_error,
                        suppress_replay_admission,
                        mode_label.empty() ? "staged_complete" : mode_label + "_complete");
            }

            const staged_tool_method * method = staged_tool_find_method(*family, method_choice.method_name);
            if (!method) {
                if (out_error) {
                    *out_error = format_error_response("staged tool loop selected a missing method", ERROR_TYPE_SERVER);
                }
                return false;
            }

            std::string payload_validation_error;
            while (stage_turn < max_stage_turns) {
                ++stage_turn;
                deepseek_chat_result payload_result;
                if (!execute_staged_selection_turn(
                            config,
                            emotive_runtime,
                            body,
                            build_staged_payload_prompt(*family, *method, payload_validation_error),
                            &payload_result,
                            out_error,
                            suppress_replay_admission,
                            mode_label.empty() ? "staged_payload_build" : mode_label + "_payload_build")) {
                    return false;
                }

                staged_tool_payload_choice payload_choice = {};
                std::string payload_error;
                if (!parse_staged_payload_response(payload_result.content, *method, &payload_choice, &payload_error)) {
                    payload_validation_error = payload_error;
                    continue;
                }
                if (payload_choice.kind == STAGED_TOOL_PAYLOAD_GO_BACK) {
                    break;
                }

                out_result->content.clear();
                out_result->reasoning_content = payload_result.reasoning_content;
                out_result->finish_reason = "tool_calls";
                out_result->prompt_tokens = payload_result.prompt_tokens;
                out_result->completion_tokens = payload_result.completion_tokens;
                out_result->emotive_trace = payload_result.emotive_trace;
                out_result->tool_calls = {
                    {
                        gen_chatcmplid(),
                        method->tool_name,
                        payload_choice.payload.dump(),
                    }
                };
                return true;
            }
        }
    }

    if (out_error) {
        *out_error = format_error_response("staged tool loop exceeded its maximum step budget", ERROR_TYPE_SERVER);
    }
    return false;
}

static int64_t current_epoch_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static bool env_to_bool_local(const char * name, bool default_value) {
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

static int32_t env_to_int_local(const char * name, int32_t default_value) {
    if (const char * value = std::getenv(name)) {
        const int parsed = std::atoi(value);
        if (parsed > 0) {
            return parsed;
        }
    }
    return default_value;
}

static float env_to_float_local(const char * name, float default_value) {
    if (const char * value = std::getenv(name)) {
        const float parsed = std::strtof(value, nullptr);
        if (std::isfinite(parsed)) {
            return parsed;
        }
    }
    return default_value;
}

static server_ongoing_task_config server_ongoing_task_config_from_env() {
    server_ongoing_task_config config;
    config.enabled = env_to_bool_local("VICUNA_ONGOING_TASKS_ENABLED", false);
    if (const char * value = std::getenv("VICUNA_ONGOING_TASKS_BASE_URL")) {
        config.base_url = trim_copy(value);
    } else if (const char * value = std::getenv("SUPERMEMORY_BASE_URL")) {
        config.base_url = trim_copy(value);
    }
    if (const char * value = std::getenv("VICUNA_ONGOING_TASKS_AUTH_TOKEN")) {
        config.auth_token = trim_copy(value);
    } else if (const char * value = std::getenv("SUPERMEMORY_API_KEY")) {
        config.auth_token = trim_copy(value);
    }
    if (const char * value = std::getenv("VICUNA_ONGOING_TASKS_CONTAINER_TAG")) {
        config.container_tag = trim_copy(value);
    } else if (const char * value = std::getenv("VICUNA_HARD_MEMORY_RUNTIME_IDENTITY")) {
        config.container_tag = trim_copy(value);
    }
    if (const char * value = std::getenv("VICUNA_ONGOING_TASKS_RUNTIME_IDENTITY")) {
        config.runtime_identity = trim_copy(value);
    } else if (const char * value = std::getenv("VICUNA_HARD_MEMORY_RUNTIME_IDENTITY")) {
        config.runtime_identity = trim_copy(value);
    }
    if (const char * value = std::getenv("VICUNA_ONGOING_TASKS_REGISTRY_KEY")) {
        config.registry_key = trim_copy(value);
    }
    if (const char * value = std::getenv("VICUNA_ONGOING_TASKS_REGISTRY_TITLE")) {
        config.registry_title = trim_copy(value);
    }
    config.query_threshold = env_to_float_local("VICUNA_ONGOING_TASKS_QUERY_THRESHOLD", config.query_threshold);
    config.poll_interval_ms = env_to_int_local("VICUNA_ONGOING_TASKS_POLL_MS", config.poll_interval_ms);
    config.timeout_ms = env_to_int_local("VICUNA_ONGOING_TASKS_TIMEOUT_MS", config.timeout_ms);
    return config;
}

static int64_t current_time_ms_utc() {
    return current_epoch_ms();
}

static std::string iso_from_epoch_ms(int64_t epoch_ms) {
    const std::time_t seconds = (std::time_t) (epoch_ms / 1000);
    const int32_t millis = (int32_t) std::abs((int) (epoch_ms % 1000));
    std::tm tm = {};
#if defined(_WIN32)
    gmtime_s(&tm, &seconds);
#else
    gmtime_r(&seconds, &tm);
#endif
    char buffer[32];
    std::snprintf(
            buffer,
            sizeof(buffer),
            "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec,
            millis);
    return buffer;
}

static bool parse_iso8601_utc_ms(const std::string & value, int64_t * out_ms) {
    if (!out_ms) {
        return false;
    }
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    int millis = 0;
    if (std::sscanf(value.c_str(), "%d-%d-%dT%d:%d:%d.%dZ", &year, &month, &day, &hour, &minute, &second, &millis) < 6) {
        return false;
    }
    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
#if defined(_WIN32)
    const std::time_t seconds = _mkgmtime(&tm);
#else
    const std::time_t seconds = timegm(&tm);
#endif
    if (seconds < 0) {
        return false;
    }
    *out_ms = (int64_t) seconds * 1000 + millis;
    return true;
}

static json ongoing_task_decision_to_json(const server_ongoing_task_decision & decision) {
    return {
        {"valid", decision.valid},
        {"should_run", decision.should_run},
        {"selected_task_id", decision.selected_task_id},
        {"rationale", decision.rationale},
        {"decided_at_ms", decision.decided_at_ms},
        {"current_time_iso", decision.current_time_iso},
        {"task_count", decision.task_count},
    };
}

static int64_t ongoing_task_frequency_window_ms(const server_ongoing_task_frequency & frequency) {
    const int64_t interval = std::max<int32_t>(1, frequency.interval);
    if (frequency.unit == "hours") {
        return interval * 60LL * 60LL * 1000LL;
    }
    if (frequency.unit == "days") {
        return interval * 24LL * 60LL * 60LL * 1000LL;
    }
    if (frequency.unit == "weeks") {
        return interval * 7LL * 24LL * 60LL * 60LL * 1000LL;
    }
    return 24LL * 60LL * 60LL * 1000LL;
}

static std::string join_url_path(const server_http_url & parts, const std::string & suffix) {
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

static bool parse_and_normalize_iso8601_utc(const json & value, std::string * out_iso) {
    if (!out_iso || !value.is_string()) {
        return false;
    }
    int64_t parsed_ms = 0;
    if (!parse_iso8601_utc_ms(value.get<std::string>(), &parsed_ms)) {
        return false;
    }
    *out_iso = iso_from_epoch_ms(parsed_ms);
    return true;
}

static bool parse_ongoing_task_frequency(
        const json & value,
        server_ongoing_task_frequency * out_frequency,
        std::string * out_error) {
    if (!out_frequency || !value.is_object()) {
        if (out_error) {
            *out_error = "ongoing-task frequency must be an object";
        }
        return false;
    }

    const int32_t interval = json_value(value, "interval", 0);
    const std::string unit = trim_copy(json_value(value, "unit", std::string()));
    if (interval < 1) {
        if (out_error) {
            *out_error = "ongoing-task frequency interval must be >= 1";
        }
        return false;
    }
    if (unit != "hours" && unit != "days" && unit != "weeks") {
        if (out_error) {
            *out_error = "ongoing-task frequency unit must be one of hours, days, or weeks";
        }
        return false;
    }

    out_frequency->interval = interval;
    out_frequency->unit = unit;
    return true;
}

static bool parse_ongoing_task_record(
        const json & value,
        server_ongoing_task_record * out_record,
        std::string * out_error) {
    if (!out_record || !value.is_object()) {
        if (out_error) {
            *out_error = "ongoing-task record must be an object";
        }
        return false;
    }

    server_ongoing_task_record record = {};
    record.task_id = trim_copy(json_value(value, "task_id", std::string()));
    record.task_text = trim_copy(json_value(value, "task_text", std::string()));
    if (record.task_id.empty() || record.task_text.empty()) {
        if (out_error) {
            *out_error = "ongoing-task record must include non-empty task_id and task_text";
        }
        return false;
    }
    if (!parse_ongoing_task_frequency(json_value(value, "frequency", json::object()), &record.frequency, out_error)) {
        return false;
    }
    if (!parse_and_normalize_iso8601_utc(value.value("created_at", json()), &record.created_at)) {
        if (out_error) {
            *out_error = "ongoing-task record created_at must be a valid ISO-8601 UTC timestamp";
        }
        return false;
    }
    if (!parse_and_normalize_iso8601_utc(value.value("updated_at", json()), &record.updated_at)) {
        if (out_error) {
            *out_error = "ongoing-task record updated_at must be a valid ISO-8601 UTC timestamp";
        }
        return false;
    }
    if (value.contains("last_done_at") && !value.at("last_done_at").is_null()) {
        if (!parse_and_normalize_iso8601_utc(value.at("last_done_at"), &record.last_done_at)) {
            if (out_error) {
                *out_error = "ongoing-task record last_done_at must be null or a valid ISO-8601 UTC timestamp";
            }
            return false;
        }
    }
    record.active = json_value(value, "active", true);
    *out_record = std::move(record);
    return true;
}

static bool parse_ongoing_task_registry_payload(
        const std::string & raw_content,
        server_ongoing_task_registry * out_registry,
        std::string * out_error) {
    if (!out_registry) {
        if (out_error) {
            *out_error = "ongoing-task registry output must not be null";
        }
        return false;
    }

    json payload;
    try {
        payload = json::parse(raw_content);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("ongoing-task registry was not valid JSON: %s", e.what());
        }
        return false;
    }

    if (!payload.is_object() || !payload.contains("tasks") || !payload.at("tasks").is_array()) {
        if (out_error) {
            *out_error = "ongoing-task registry must contain a tasks array";
        }
        return false;
    }

    server_ongoing_task_registry registry = {};
    registry.schema_version = json_value(payload, "schema_version", 1);
    if (!parse_and_normalize_iso8601_utc(payload.value("updated_at", json()), &registry.updated_at)) {
        if (out_error) {
            *out_error = "ongoing-task registry updated_at must be a valid ISO-8601 UTC timestamp";
        }
        return false;
    }

    for (const auto & item : payload.at("tasks")) {
        server_ongoing_task_record record = {};
        if (!parse_ongoing_task_record(item, &record, out_error)) {
            return false;
        }
        registry.tasks.push_back(std::move(record));
    }

    *out_registry = std::move(registry);
    return true;
}

static server_ongoing_task_registry empty_ongoing_task_registry() {
    server_ongoing_task_registry registry = {};
    registry.schema_version = 1;
    registry.updated_at = iso_from_epoch_ms(current_time_ms_utc());
    return registry;
}

static server_ongoing_task_summary summarize_ongoing_task(
        const server_ongoing_task_record & record,
        int64_t now_ms) {
    server_ongoing_task_summary summary = {};
    summary.task_id = record.task_id;
    summary.task_text = record.task_text;
    summary.frequency = record.frequency;
    summary.last_done_at = record.last_done_at;
    summary.active = record.active;

    int64_t due_anchor_ms = now_ms;
    if (!record.last_done_at.empty()) {
        int64_t last_done_ms = 0;
        if (parse_iso8601_utc_ms(record.last_done_at, &last_done_ms)) {
            due_anchor_ms = last_done_ms + ongoing_task_frequency_window_ms(record.frequency);
        }
    } else {
        int64_t created_ms = 0;
        if (parse_iso8601_utc_ms(record.created_at, &created_ms)) {
            due_anchor_ms = created_ms;
        }
    }

    summary.next_due_at = iso_from_epoch_ms(due_anchor_ms);
    summary.due_now = record.active && due_anchor_ms <= now_ms;
    return summary;
}

static std::vector<server_ongoing_task_summary> list_active_ongoing_task_summaries(
        const server_ongoing_task_registry & registry,
        int64_t now_ms) {
    std::vector<server_ongoing_task_summary> tasks;
    for (const auto & record : registry.tasks) {
        if (!record.active) {
            continue;
        }
        tasks.push_back(summarize_ongoing_task(record, now_ms));
    }
    std::sort(
            tasks.begin(),
            tasks.end(),
            [](const server_ongoing_task_summary & lhs, const server_ongoing_task_summary & rhs) {
                if (lhs.next_due_at == rhs.next_due_at) {
                    return lhs.task_id < rhs.task_id;
                }
                return lhs.next_due_at < rhs.next_due_at;
            });
    return tasks;
}

static bool ongoing_task_hard_memory_request(
        const server_ongoing_task_config & config,
        const std::string & path_suffix,
        const json & request_body,
        json * out_response,
        std::string * out_error) {
    if (!out_response) {
        if (out_error) {
            *out_error = "ongoing-task response output must not be null";
        }
        return false;
    }
    if (config.auth_token.empty()) {
        if (out_error) {
            *out_error = "missing ongoing-task hard-memory auth token";
        }
        return false;
    }

    try {
        auto [client, parts] = server_http_client(config.base_url);
        client.set_connection_timeout(std::chrono::milliseconds(config.timeout_ms));
        client.set_read_timeout(std::chrono::milliseconds(config.timeout_ms));
        client.set_write_timeout(std::chrono::milliseconds(config.timeout_ms));

        const httplib::Headers headers = {
            {"Authorization", "Bearer " + config.auth_token},
            {"x-supermemory-api-key", config.auth_token},
            {"Accept", "application/json"},
        };

        const std::string path = join_url_path(parts, path_suffix);
        auto response = client.Post(path.c_str(), headers, request_body.dump(), "application/json");
        if (!response) {
            if (out_error) {
                *out_error = "ongoing-task hard-memory request failed before a response was received";
            }
            return false;
        }
        if (response->status < 200 || response->status >= 300) {
            if (out_error) {
                *out_error = server_string_format(
                        "ongoing-task hard-memory request failed with HTTP %d",
                        response->status);
            }
            return false;
        }

        *out_response = response->body.empty() ? json::object() : json::parse(response->body);
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = e.what();
        }
        return false;
    }
}

static bool load_ongoing_task_registry(
        const server_ongoing_task_config & config,
        server_ongoing_task_registry * out_registry,
        std::string * out_error) {
    if (!out_registry) {
        if (out_error) {
            *out_error = "ongoing-task registry output must not be null";
        }
        return false;
    }

    json response;
    if (!ongoing_task_hard_memory_request(config, "/v4/profile", {
                {"containerTag", config.container_tag},
                {"q", config.registry_key},
                {"threshold", config.query_threshold},
            }, &response, out_error)) {
        return false;
    }

    const json search_results = json_value(response, "searchResults", json::object());
    const json results = json_value(search_results, "results", json::array());
    if (!results.is_array()) {
        *out_registry = empty_ongoing_task_registry();
        return true;
    }

    bool found_match = false;
    server_ongoing_task_registry best_registry = empty_ongoing_task_registry();
    for (const auto & item : results) {
        if (!item.is_object()) {
            continue;
        }

        const json metadata = json_value(item, "metadata", json::object());
        const std::string metadata_key = trim_copy(json_value(metadata, "key", std::string()));
        const std::string title = trim_copy(json_value(item, "title", json_value(metadata, "title", std::string())));
        if (metadata_key != config.registry_key && title != config.registry_title) {
            continue;
        }

        std::string raw_content;
        if (item.contains("memory") && item.at("memory").is_string()) {
            raw_content = item.at("memory").get<std::string>();
        } else if (item.contains("chunk") && item.at("chunk").is_string()) {
            raw_content = item.at("chunk").get<std::string>();
        } else if (item.contains("content") && item.at("content").is_string()) {
            raw_content = item.at("content").get<std::string>();
        } else {
            if (out_error) {
                *out_error = "ongoing-task registry result was missing content";
            }
            return false;
        }

        server_ongoing_task_registry parsed = {};
        if (!parse_ongoing_task_registry_payload(raw_content, &parsed, out_error)) {
            return false;
        }
        if (!found_match || parsed.updated_at > best_registry.updated_at) {
            best_registry = std::move(parsed);
            found_match = true;
        }
    }

    *out_registry = found_match ? best_registry : empty_ongoing_task_registry();
    return true;
}

static json ongoing_task_registry_to_json(const server_ongoing_task_registry & registry) {
    json tasks = json::array();
    for (const auto & task : registry.tasks) {
        tasks.push_back({
            {"task_id", task.task_id},
            {"task_text", task.task_text},
            {"frequency", {
                {"interval", task.frequency.interval},
                {"unit", task.frequency.unit},
            }},
            {"created_at", task.created_at},
            {"updated_at", task.updated_at},
            {"last_done_at", task.last_done_at.empty() ? json(nullptr) : json(task.last_done_at)},
            {"active", task.active},
        });
    }
    return {
        {"schema_version", 1},
        {"updated_at", registry.updated_at},
        {"tasks", std::move(tasks)},
    };
}

static bool save_ongoing_task_registry(
        const server_ongoing_task_config & config,
        server_ongoing_task_registry registry,
        std::string * out_error) {
    registry.schema_version = 1;
    registry.updated_at = iso_from_epoch_ms(current_time_ms_utc());

    json ignored_response;
    return ongoing_task_hard_memory_request(config, "/v4/memories", {
                {"containerTag", config.container_tag},
                {"memories", json::array({
                    {
                        {"content", safe_json_to_str(ongoing_task_registry_to_json(registry))},
                        {"metadata", {
                            {"source", "vicuna"},
                            {"runtimeIdentity", config.runtime_identity},
                            {"kind", "tool_observation"},
                            {"domain", "strategy"},
                            {"key", config.registry_key},
                            {"title", config.registry_title},
                            {"tags", json::array({"ongoing_tasks", "registry"})},
                            {"importance", 0.8},
                            {"confidence", 1.0},
                            {"gainBias", 0.3},
                            {"allostaticRelevance", 0.0},
                        }},
                    },
                })},
            }, &ignored_response, out_error);
}

static bool mark_ongoing_task_complete(
        server_ongoing_task_registry * registry,
        const std::string & task_id,
        const std::string & completed_at_iso,
        std::string * out_error) {
    if (!registry) {
        if (out_error) {
            *out_error = "ongoing-task registry must not be null";
        }
        return false;
    }
    for (auto & task : registry->tasks) {
        if (task.task_id != task_id) {
            continue;
        }
        task.last_done_at = completed_at_iso;
        task.updated_at = iso_from_epoch_ms(current_time_ms_utc());
        return true;
    }
    if (out_error) {
        *out_error = "selected ongoing task was not present in the loaded registry";
    }
    return false;
}

static std::string build_ongoing_task_decision_system_prompt() {
    return
            "You are deciding whether one recurring ongoing task should run now before true idle. "
            "Use both the explicit cadence/timestamp fields and the task wording. "
            "Select at most one task. Return exactly one JSON object and no markdown. "
            "The object must contain: should_run (boolean), selected_task_id (string or empty string), "
            "rationale (short string).";
}

static std::string build_ongoing_task_decision_user_prompt(
        const std::vector<server_ongoing_task_summary> & tasks,
        const std::string & current_time_iso) {
    std::ostringstream out;
    out << "Current system time: " << current_time_iso << "\n";
    out << "Decide whether exactly one ongoing task should run now.\n";
    out << "Prefer the most overdue or most clearly due task. If none should run, return should_run=false.\n";
    out << "Tasks:\n";
    for (const auto & task : tasks) {
        out << "- task_id: " << task.task_id << "\n";
        out << "  task_text: " << task.task_text << "\n";
        out << "  frequency_interval: " << task.frequency.interval << "\n";
        out << "  frequency_unit: " << task.frequency.unit << "\n";
        out << "  last_done_at: " << (task.last_done_at.empty() ? "null" : task.last_done_at) << "\n";
        out << "  next_due_at: " << task.next_due_at << "\n";
        out << "  due_now: " << (task.due_now ? "true" : "false") << "\n";
        out << "  active: " << (task.active ? "true" : "false") << "\n";
    }
    return out.str();
}

static bool parse_ongoing_task_decision_response(
        const std::string & text,
        server_ongoing_task_decision * out_decision,
        std::string * out_error) {
    if (!out_decision) {
        if (out_error) {
            *out_error = "ongoing-task decision output must not be null";
        }
        return false;
    }

    json payload;
    try {
        payload = json::parse(extract_json_object_payload(text));
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("ongoing-task decision was not valid JSON: %s", e.what());
        }
        return false;
    }

    server_ongoing_task_decision decision = {};
    decision.valid = true;
    decision.should_run = json_value(payload, "should_run", false);
    decision.selected_task_id = trim_copy(json_value(payload, "selected_task_id", std::string()));
    decision.rationale = trim_copy(json_value(payload, "rationale", std::string()));
    if (!decision.should_run) {
        decision.selected_task_id.clear();
    }
    if (decision.should_run && decision.selected_task_id.empty()) {
        if (out_error) {
            *out_error = "ongoing-task decision selected work but did not provide selected_task_id";
        }
        return false;
    }

    *out_decision = std::move(decision);
    return true;
}

static const server_ongoing_task_summary * find_ongoing_task_summary(
        const std::vector<server_ongoing_task_summary> & tasks,
        const std::string & task_id) {
    for (const auto & task : tasks) {
        if (task.task_id == task_id) {
            return &task;
        }
    }
    return nullptr;
}

static bool telegram_method_is_allowed(const std::string & method) {
    return method == "sendMessage" ||
            method == "sendPhoto" ||
            method == "sendDocument" ||
            method == "sendAudio" ||
            method == "sendVoice" ||
            method == "sendVideo" ||
            method == "sendAnimation" ||
            method == "sendSticker" ||
            method == "sendMediaGroup" ||
            method == "sendLocation" ||
            method == "sendVenue" ||
            method == "sendContact" ||
            method == "sendPoll" ||
            method == "sendDice";
}

static std::string telegram_payload_string_field(const json & payload, const char * key) {
    if (!payload.is_object() || !payload.contains(key) || !payload.at(key).is_string()) {
        return std::string();
    }
    return trim_copy(payload.at(key).get<std::string>());
}

static void validate_telegram_payload_fields(const std::string & method, const json & payload) {
    if (!telegram_method_is_allowed(method)) {
        throw std::invalid_argument("telegram outbox write requires an allowlisted telegram_method");
    }
    if (!payload.is_object()) {
        throw std::invalid_argument("telegram outbox write requires telegram_payload to be an object");
    }

    if (method == "sendMessage") {
        if (telegram_payload_string_field(payload, "text").empty()) {
            throw std::invalid_argument("telegram sendMessage payload requires non-empty text");
        }
        return;
    }
    if (method == "sendPhoto") {
        if (telegram_payload_string_field(payload, "photo").empty()) {
            throw std::invalid_argument("telegram sendPhoto payload requires non-empty photo");
        }
        return;
    }
    if (method == "sendDocument") {
        if (telegram_payload_string_field(payload, "document").empty()) {
            throw std::invalid_argument("telegram sendDocument payload requires non-empty document");
        }
        return;
    }
    if (method == "sendAudio") {
        if (telegram_payload_string_field(payload, "audio").empty()) {
            throw std::invalid_argument("telegram sendAudio payload requires non-empty audio");
        }
        return;
    }
    if (method == "sendVoice") {
        if (telegram_payload_string_field(payload, "voice").empty()) {
            throw std::invalid_argument("telegram sendVoice payload requires non-empty voice");
        }
        return;
    }
    if (method == "sendVideo") {
        if (telegram_payload_string_field(payload, "video").empty()) {
            throw std::invalid_argument("telegram sendVideo payload requires non-empty video");
        }
        return;
    }
    if (method == "sendAnimation") {
        if (telegram_payload_string_field(payload, "animation").empty()) {
            throw std::invalid_argument("telegram sendAnimation payload requires non-empty animation");
        }
        return;
    }
    if (method == "sendSticker") {
        if (telegram_payload_string_field(payload, "sticker").empty()) {
            throw std::invalid_argument("telegram sendSticker payload requires non-empty sticker");
        }
        return;
    }
    if (method == "sendMediaGroup") {
        if (!payload.contains("media") || !payload.at("media").is_array() || payload.at("media").empty()) {
            throw std::invalid_argument("telegram sendMediaGroup payload requires non-empty media array");
        }
        return;
    }
    if (method == "sendLocation") {
        if (!payload.contains("latitude") || !payload.at("latitude").is_number()) {
            throw std::invalid_argument("telegram sendLocation payload requires numeric latitude");
        }
        if (!payload.contains("longitude") || !payload.at("longitude").is_number()) {
            throw std::invalid_argument("telegram sendLocation payload requires numeric longitude");
        }
        return;
    }
    if (method == "sendVenue") {
        if (!payload.contains("latitude") || !payload.at("latitude").is_number()) {
            throw std::invalid_argument("telegram sendVenue payload requires numeric latitude");
        }
        if (!payload.contains("longitude") || !payload.at("longitude").is_number()) {
            throw std::invalid_argument("telegram sendVenue payload requires numeric longitude");
        }
        if (telegram_payload_string_field(payload, "title").empty()) {
            throw std::invalid_argument("telegram sendVenue payload requires non-empty title");
        }
        if (telegram_payload_string_field(payload, "address").empty()) {
            throw std::invalid_argument("telegram sendVenue payload requires non-empty address");
        }
        return;
    }
    if (method == "sendContact") {
        if (telegram_payload_string_field(payload, "phone_number").empty()) {
            throw std::invalid_argument("telegram sendContact payload requires non-empty phone_number");
        }
        if (telegram_payload_string_field(payload, "first_name").empty()) {
            throw std::invalid_argument("telegram sendContact payload requires non-empty first_name");
        }
        return;
    }
    if (method == "sendPoll") {
        if (telegram_payload_string_field(payload, "question").empty()) {
            throw std::invalid_argument("telegram sendPoll payload requires non-empty question");
        }
        if (!payload.contains("options") || !payload.at("options").is_array() || payload.at("options").size() < 2) {
            throw std::invalid_argument("telegram sendPoll payload requires at least two options");
        }
        return;
    }
}

static std::string derive_telegram_summary_text(const std::string & method, const json & payload) {
    const std::string text = telegram_payload_string_field(payload, "text");
    if (!text.empty()) {
        return text;
    }

    const std::string caption = telegram_payload_string_field(payload, "caption");
    if (!caption.empty()) {
        return caption;
    }

    if (method == "sendPoll") {
        const std::string question = telegram_payload_string_field(payload, "question");
        return question.empty() ? std::string("Telegram poll sent.") : question;
    }
    if (method == "sendVenue") {
        const std::string title = telegram_payload_string_field(payload, "title");
        const std::string address = telegram_payload_string_field(payload, "address");
        if (!title.empty() && !address.empty()) {
            return title + "\n" + address;
        }
        if (!title.empty()) {
            return title;
        }
        if (!address.empty()) {
            return address;
        }
        return "Telegram venue sent.";
    }
    if (method == "sendContact") {
        const std::string first_name = telegram_payload_string_field(payload, "first_name");
        const std::string phone_number = telegram_payload_string_field(payload, "phone_number");
        if (!first_name.empty() && !phone_number.empty()) {
            return first_name + "\n" + phone_number;
        }
        if (!first_name.empty()) {
            return first_name;
        }
        if (!phone_number.empty()) {
            return phone_number;
        }
        return "Telegram contact sent.";
    }
    if (method == "sendMediaGroup" && payload.is_object() && payload.contains("media") && payload.at("media").is_array()) {
        for (const auto & media_item : payload.at("media")) {
            const std::string media_caption = telegram_payload_string_field(media_item, "caption");
            if (!media_caption.empty()) {
                return media_caption;
            }
        }
        return "Telegram media group sent.";
    }
    if (method == "sendLocation") {
        return "Telegram location sent.";
    }
    if (method == "sendDice") {
        return "Telegram dice sent.";
    }
    if (method == "sendSticker") {
        return "Telegram sticker sent.";
    }

    return "Telegram " + method + " sent.";
}

static std::string build_telegram_bridge_dedupe_key(
        const telegram_bridge_request_context & context,
        const std::string & telegram_method) {
    std::ostringstream out;
    out << "bridge:";
    if (!context.conversation_id.empty()) {
        out << context.conversation_id;
    } else {
        out << context.chat_scope;
    }
    out << ":" << context.reply_to_message_id;
    out << ":" << telegram_method;
    return out.str();
}

static telegram_outbox_enqueue_result telegram_outbox_enqueue(telegram_outbox_state * outbox, telegram_outbox_item request) {
    if (!outbox) {
        throw std::invalid_argument("telegram outbox state was not configured");
    }
    if (request.chat_scope.empty()) {
        throw std::invalid_argument("telegram outbox write requires non-empty chat_scope");
    }
    validate_telegram_payload_fields(request.telegram_method, request.telegram_payload);
    if (request.text.empty()) {
        request.text = derive_telegram_summary_text(request.telegram_method, request.telegram_payload);
    }
    if (request.text.empty()) {
        throw std::invalid_argument("telegram outbox write requires relayable summary text");
    }

    std::lock_guard<std::mutex> lock(outbox->mutex);
    if (!request.dedupe_key.empty()) {
        for (const auto & existing : outbox->items) {
            if (existing.kind == request.kind &&
                    existing.chat_scope == request.chat_scope &&
                    existing.dedupe_key == request.dedupe_key) {
                return {
                    false,
                    true,
                    existing.sequence_number,
                    existing.chat_scope,
                };
            }
        }
    }

    request.sequence_number = outbox->next_sequence_number++;
    outbox->items.push_back(request);
    while (outbox->items.size() > outbox->max_items) {
        outbox->items.pop_front();
    }
    return {
        true,
        false,
        request.sequence_number,
        request.chat_scope,
    };
}

static telegram_outbox_item telegram_outbox_item_from_json(const json & body) {
    const std::string kind = trim_copy(json_value(body, "kind", std::string("message")));
    if (kind != "message") {
        throw std::invalid_argument("telegram outbox only supports kind=message on the provider-only path");
    }

    telegram_outbox_item request = {};
    request.kind = kind;
    request.chat_scope = trim_copy(json_value(body, "chat_scope", std::string()));
    request.text = trim_copy(json_value(body, "text", std::string()));
    request.telegram_method = trim_copy(json_value(body, "telegram_method", std::string("sendMessage")));
    if (body.contains("telegram_payload")) {
        request.telegram_payload = body.at("telegram_payload");
    } else {
        request.telegram_payload = json{
            {"text", request.text},
        };
    }
    request.reply_to_message_id = std::max<int64_t>(0, json_value(body, "reply_to_message_id", int64_t(0)));
    request.intent = trim_copy(json_value(body, "intent", std::string()));
    request.dedupe_key = trim_copy(json_value(body, "dedupe_key", std::string()));
    request.urgency = std::max(0.0, json_value(body, "urgency", 0.0));
    return request;
}

static bool parse_telegram_relay_tool_call(
        const deepseek_tool_call & tool_call,
        const telegram_bridge_request_context & context,
        telegram_outbox_item * out_item,
        std::string * out_error) {
    if (!out_item) {
        if (out_error) {
            *out_error = "telegram relay parsing requires an output item";
        }
        return false;
    }

    json arguments;
    try {
        arguments = json::parse(tool_call.arguments_json.empty() ? std::string("{}") : tool_call.arguments_json);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("telegram_relay arguments were not valid JSON: %s", e.what());
        }
        return false;
    }
    if (!arguments.is_object()) {
        if (out_error) {
            *out_error = "telegram_relay arguments must be a JSON object";
        }
        return false;
    }

    const bool has_text = !trim_copy(json_value(arguments, "text", std::string())).empty();
    const bool has_request = arguments.contains("request") && arguments.at("request").is_object();
    if (has_text == has_request) {
        if (out_error) {
            *out_error = "telegram_relay requires exactly one of text or request";
        }
        return false;
    }

    telegram_outbox_item item = {};
    item.chat_scope = trim_copy(json_value(arguments, "chat_scope", context.chat_scope));
    item.reply_to_message_id = std::max<int64_t>(
            0,
            json_value(arguments, "reply_to_message_id", context.reply_to_message_id));
    item.intent = trim_copy(json_value(arguments, "intent", std::string()));
    item.dedupe_key = trim_copy(json_value(arguments, "dedupe_key", std::string()));
    item.urgency = std::max(0.0, json_value(arguments, "urgency", 0.0));

    if (item.chat_scope.empty()) {
        if (out_error) {
            *out_error = "telegram_relay requires chat_scope when no bridge-scoped default is available";
        }
        return false;
    }

    if (has_text) {
        item.text = trim_copy(json_value(arguments, "text", std::string()));
        item.telegram_method = "sendMessage";
        item.telegram_payload = json{
            {"text", item.text},
        };
    } else {
        const json & request = arguments.at("request");
        item.telegram_method = trim_copy(json_value(request, "method", std::string()));
        if (request.contains("payload")) {
            item.telegram_payload = request.at("payload");
        } else {
            item.telegram_payload = json::object();
        }
        if (item.telegram_payload.is_object()) {
            item.telegram_payload.erase("chat_id");
        }
    }

    try {
        validate_telegram_payload_fields(item.telegram_method, item.telegram_payload);
        if (item.text.empty()) {
            item.text = derive_telegram_summary_text(item.telegram_method, item.telegram_payload);
        }
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = e.what();
        }
        return false;
    }

    if (item.text.empty()) {
        if (out_error) {
            *out_error = "telegram_relay did not produce relayable summary text";
        }
        return false;
    }
    if (item.dedupe_key.empty()) {
        item.dedupe_key = build_telegram_bridge_dedupe_key(context, item.telegram_method);
    }
    *out_item = std::move(item);
    return true;
}

static json telegram_outbox_item_to_json(const telegram_outbox_item & item) {
    json payload = {
        {"sequence_number", item.sequence_number},
        {"kind", item.kind},
        {"chat_scope", item.chat_scope},
        {"telegram_method", item.telegram_method},
        {"telegram_payload", item.telegram_payload},
        {"text", item.text},
    };
    if (item.reply_to_message_id > 0) {
        payload["reply_to_message_id"] = item.reply_to_message_id;
    }
    if (!item.intent.empty()) {
        payload["intent"] = item.intent;
    }
    if (!item.dedupe_key.empty()) {
        payload["dedupe_key"] = item.dedupe_key;
    }
    if (item.urgency > 0.0) {
        payload["urgency"] = item.urgency;
    }
    return payload;
}

static json telegram_outbox_read_json(const telegram_outbox_state & outbox, uint64_t after) {
    std::lock_guard<std::mutex> lock(outbox.mutex);

    json items = json::array();
    uint64_t newest_sequence = 0;
    uint64_t oldest_sequence = 0;
    if (!outbox.items.empty()) {
        oldest_sequence = outbox.items.front().sequence_number;
        newest_sequence = outbox.items.back().sequence_number;
    }

    for (const auto & item : outbox.items) {
        if (item.sequence_number > after) {
            items.push_back(telegram_outbox_item_to_json(item));
        }
    }

    return {
        {"items", std::move(items)},
        {"last_sequence", newest_sequence},
        {"stored_items", outbox.items.size()},
        {"next_sequence_number", outbox.next_sequence_number},
        {"oldest_sequence", oldest_sequence},
        {"newest_sequence", newest_sequence},
    };
}

static json telegram_outbox_health_json(const telegram_outbox_state & outbox) {
    std::lock_guard<std::mutex> lock(outbox.mutex);
    const uint64_t newest_sequence = outbox.items.empty() ? 0 : outbox.items.back().sequence_number;
    const uint64_t oldest_sequence = outbox.items.empty() ? 0 : outbox.items.front().sequence_number;
    return {
        {"stored_items", outbox.items.size()},
        {"next_sequence_number", outbox.next_sequence_number},
        {"oldest_sequence", oldest_sequence},
        {"newest_sequence", newest_sequence},
    };
}

static json telegram_outbox_enqueue_json(telegram_outbox_state * outbox, const json & body) {
    const telegram_outbox_item request = telegram_outbox_item_from_json(body);
    const telegram_outbox_enqueue_result enqueue = telegram_outbox_enqueue(outbox, request);
    std::lock_guard<std::mutex> lock(outbox->mutex);
    return {
        {"ok", true},
        {"queued", enqueue.queued},
        {"deduplicated", enqueue.deduplicated},
        {"sequence_number", enqueue.sequence_number},
        {"chat_scope", enqueue.chat_scope},
        {"stored_items", outbox->items.size()},
        {"next_sequence_number", outbox->next_sequence_number},
    };
}

static json telegram_delivery_result_to_json(const telegram_delivery_result & delivery) {
    if (!delivery.handled) {
        return nullptr;
    }
    return {
        {"handled", true},
        {"queued", delivery.queued},
        {"deduplicated", delivery.deduplicated},
        {"sequence_number", delivery.sequence_number},
        {"chat_scope", delivery.chat_scope},
        {"telegram_method", delivery.telegram_method},
        {"reply_to_message_id", delivery.reply_to_message_id},
        {"source", delivery.source},
    };
}

static bool execute_telegram_delivery_for_bridge_request(
        const server_http_req & req,
        telegram_outbox_state * outbox,
        deepseek_chat_result * result,
        telegram_delivery_result * out_delivery,
        json * out_error) {
    if (!result) {
        if (out_error) {
            *out_error = format_error_response("telegram delivery execution requires a result", ERROR_TYPE_SERVER);
        }
        return false;
    }
    if (out_delivery) {
        *out_delivery = telegram_delivery_result();
    }
    if (!outbox) {
        return true;
    }

    const telegram_bridge_request_context context = parse_telegram_bridge_request_context(req);
    if (!context.active) {
        return true;
    }

    telegram_outbox_item item = {};
    std::string delivery_source;
    if (result->tool_calls.size() == 1 && result->tool_calls.front().name == "telegram_relay") {
        std::string parse_error;
        if (!parse_telegram_relay_tool_call(result->tool_calls.front(), context, &item, &parse_error)) {
            if (out_error) {
                *out_error = format_error_response(parse_error, ERROR_TYPE_SERVER);
            }
            return false;
        }
        delivery_source = "tool_call";
    } else {
        if (!result->tool_calls.empty()) {
            // Bridge-scoped requests may now continue through direct runtime tool calls.
            // Only intercept the bridge-local telegram_relay tool here; leave other tool calls
            // intact so the bridge can execute them and continue the loop client-side.
            return true;
        }
        const std::string text = trim_copy(result->content);
        if (text.empty()) {
            return true;
        }
        item.chat_scope = context.chat_scope;
        item.reply_to_message_id = context.reply_to_message_id;
        item.text = text;
        item.telegram_method = "sendMessage";
        item.telegram_payload = json{
            {"text", text},
        };
        item.dedupe_key = build_telegram_bridge_dedupe_key(context, item.telegram_method);
        delivery_source = "compat_plain_text";
    }

    telegram_outbox_enqueue_result enqueue = {};
    try {
        enqueue = telegram_outbox_enqueue(outbox, item);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = format_error_response(e.what(), ERROR_TYPE_SERVER);
        }
        return false;
    }

    result->content.clear();
    result->tool_calls.clear();
    result->finish_reason = "stop";

    if (out_delivery) {
        out_delivery->handled = true;
        out_delivery->queued = enqueue.queued;
        out_delivery->deduplicated = enqueue.deduplicated;
        out_delivery->sequence_number = enqueue.sequence_number;
        out_delivery->chat_scope = enqueue.chat_scope;
        out_delivery->telegram_method = item.telegram_method;
        out_delivery->reply_to_message_id = item.reply_to_message_id;
        out_delivery->source = delivery_source;
    }
    return true;
}

static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func, request_activity_state * activity_state = nullptr) {
    return [func = std::move(func), activity_state](const server_http_req & req) -> server_http_res_ptr {
        struct request_activity_guard {
            request_activity_state * state = nullptr;

            explicit request_activity_guard(request_activity_state * state) : state(state) {
                if (this->state) {
                    this->state->note_request_start();
                }
            }

            ~request_activity_guard() {
                if (state) {
                    state->note_request_end();
                }
            }
        } guard(activity_state);

        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            return make_error_response(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            return make_error_response(format_error_response(e.what(), ERROR_TYPE_SERVER));
        } catch (...) {
            return make_error_response(format_error_response("unknown error", ERROR_TYPE_SERVER));
        }
    };
}

static std::string request_content_to_text(const json & content) {
    if (content.is_null()) {
        return std::string();
    }
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        return std::string();
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
        if (type.empty() || type == "text" || type == "input_text" || type == "output_text") {
            text += json_value(part, "text", std::string());
        }
    }

    return text;
}

static std::string format_tool_calls_runtime_text(const json & tool_calls) {
    if (!tool_calls.is_array() || tool_calls.empty()) {
        return std::string();
    }

    std::ostringstream out;
    bool first = true;
    for (const auto & item : tool_calls) {
        if (!item.is_object()) {
            continue;
        }
        const json function = item.value("function", json::object());
        const std::string name = json_value(function, "name", std::string("unknown_tool"));
        const std::string arguments = json_value(function, "arguments", std::string("{}"));
        if (!first) {
            out << "; ";
        }
        out << "tool_selection:" << name << " args=" << arguments;
        first = false;
    }

    return out.str();
}

static void seed_emotive_turn_from_request(const json & body, server_emotive_turn_builder * builder) {
    if (!builder) {
        return;
    }

    if (body.contains("messages") && body.at("messages").is_array()) {
        for (const auto & item : body.at("messages")) {
            if (!item.is_object()) {
                continue;
            }

            const std::string role = json_value(item, "role", std::string());
            if (role == "system") {
                const std::string text = item.contains("content") ?
                        request_content_to_text(item.at("content")) :
                        std::string();
                if (!text.empty()) {
                    builder->add_replay_block(SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT, "system_prompt: " + text);
                }
                continue;
            }

            if (role == "user") {
                const std::string text = item.contains("content") ?
                        request_content_to_text(item.at("content")) :
                        std::string();
                if (!text.empty()) {
                    builder->add_user_message(text);
                }
                continue;
            }

            if (role == "assistant") {
                const std::string reasoning = json_value(item, "reasoning_content", std::string());
                if (!reasoning.empty()) {
                    builder->add_replay_block(SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING, reasoning);
                }

                const std::string content = item.contains("content") ?
                        request_content_to_text(item.at("content")) :
                        std::string();
                if (!content.empty()) {
                    builder->add_replay_block(SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT, content);
                }

                if (item.contains("tool_calls")) {
                    const std::string tool_runtime_text = format_tool_calls_runtime_text(item.at("tool_calls"));
                    if (!tool_runtime_text.empty()) {
                        builder->add_replay_block(SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT, tool_runtime_text);
                    }
                }
                continue;
            }

            if (role == "tool") {
                const std::string tool_call_id = json_value(item, "tool_call_id", std::string());
                const std::string content = item.contains("content") ?
                        request_content_to_text(item.at("content")) :
                        std::string();
                std::string runtime_text = "tool_result";
                if (!tool_call_id.empty()) {
                    runtime_text += ":" + tool_call_id;
                }
                if (!content.empty()) {
                    runtime_text += " " + content;
                }
                builder->add_replay_block(SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT, runtime_text);
            }
        }
        return;
    }

    if (body.contains("prompt")) {
        const std::string prompt_text = request_content_to_text(body.at("prompt"));
        if (!prompt_text.empty()) {
            builder->add_user_message(prompt_text);
        }
    }
}

struct active_tool_continuation_span {
    bool valid = false;
    size_t assistant_index = 0;
    size_t first_tool_index = 0;
    size_t last_tool_index = 0;
};

static active_tool_continuation_span find_active_tool_continuation_span(const json & messages) {
    active_tool_continuation_span span;
    if (!messages.is_array() || messages.empty()) {
        return span;
    }

    for (size_t i = messages.size(); i > 0; --i) {
        const size_t current = i - 1;
        const auto & item = messages.at(current);
        if (!item.is_object() || json_value(item, "role", std::string()) != "tool") {
            continue;
        }

        size_t first_tool = current;
        while (first_tool > 0) {
            const auto & previous = messages.at(first_tool - 1);
            if (!previous.is_object() || json_value(previous, "role", std::string()) != "tool") {
                break;
            }
            --first_tool;
        }

        if (first_tool == 0) {
            continue;
        }

        const auto & assistant = messages.at(first_tool - 1);
        if (!assistant.is_object() || json_value(assistant, "role", std::string()) != "assistant") {
            continue;
        }
        if (!assistant.contains("tool_calls") || !assistant.at("tool_calls").is_array() || assistant.at("tool_calls").empty()) {
            continue;
        }

        span.valid = true;
        span.assistant_index = first_tool - 1;
        span.first_tool_index = first_tool;
        span.last_tool_index = current;
        return span;
    }

    return span;
}

static std::string normalize_guidance_text(const std::string & text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (const char ch : text) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) || std::isspace(uch)) {
            normalized.push_back((char) std::tolower(uch));
        } else {
            normalized.push_back(' ');
        }
    }
    return normalized;
}

static std::vector<std::string> extract_live_struct_tags(const std::string & text, const std::string & role_hint) {
    std::vector<std::string> tags;
    if (!role_hint.empty()) {
        tags.push_back(role_hint);
    }
    const std::string normalized = normalize_guidance_text(text);
    if (normalized.find("tool") != std::string::npos || normalized.find("function") != std::string::npos) {
        tags.push_back("tool_context");
    }
    if (normalized.find("error") != std::string::npos ||
            normalized.find("fail") != std::string::npos ||
            normalized.find("timeout") != std::string::npos ||
            normalized.find("retry") != std::string::npos) {
        tags.push_back("runtime_failure");
    }
    if (normalized.find("unclear") != std::string::npos ||
            normalized.find("unsure") != std::string::npos ||
            normalized.find("maybe") != std::string::npos ||
            normalized.find("stuck") != std::string::npos ||
            normalized.find("cannot") != std::string::npos) {
        tags.push_back("uncertainty_spike");
    }
    if (normalized.find("validate") != std::string::npos ||
            normalized.find("confirm") != std::string::npos ||
            normalized.find("check") != std::string::npos ||
            normalized.find("schema") != std::string::npos ||
            normalized.find("contract") != std::string::npos) {
        tags.push_back("validation_step");
    }
    if (normalized.find("plan") != std::string::npos ||
            normalized.find("next") != std::string::npos ||
            normalized.find("step") != std::string::npos ||
            normalized.find("first") != std::string::npos ||
            normalized.find("then") != std::string::npos) {
        tags.push_back("planning_step");
    }
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    return tags;
}

static std::string build_live_heuristic_query_text(
        const json & messages,
        std::vector<std::string> * out_struct_tags = nullptr) {
    if (!messages.is_array() || messages.empty()) {
        return std::string();
    }

    std::vector<std::string> fragments;
    std::vector<std::string> tags;
    for (size_t i = messages.size(); i > 0 && fragments.size() < 3; --i) {
        const auto & item = messages.at(i - 1);
        if (!item.is_object()) {
            continue;
        }

        const std::string role = json_value(item, "role", std::string());
        if (role == "system") {
            continue;
        }

        const std::string reasoning = json_value(item, "reasoning_content", std::string());
        if (!reasoning.empty()) {
            fragments.push_back(reasoning);
            const auto reasoning_tags = extract_live_struct_tags(reasoning, "assistant_reasoning");
            tags.insert(tags.end(), reasoning_tags.begin(), reasoning_tags.end());
            continue;
        }

        const std::string content = item.contains("content") ?
                request_content_to_text(item.at("content")) :
                std::string();
        if (!content.empty()) {
            fragments.push_back(content);
            const auto content_tags = extract_live_struct_tags(content, role);
            tags.insert(tags.end(), content_tags.begin(), content_tags.end());
        }
    }

    std::reverse(fragments.begin(), fragments.end());
    std::ostringstream out;
    for (size_t i = 0; i < fragments.size(); ++i) {
        if (i > 0) {
            out << "\n";
        }
        out << fragments[i];
    }

    if (out_struct_tags) {
        std::sort(tags.begin(), tags.end());
        tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
        *out_struct_tags = std::move(tags);
    }

    return trim_copy(out.str());
}

static std::string format_interleaved_vad_guidance(const server_emotive_vad & vad) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "Current emotive guidance: valence=" << vad.valence
        << " arousal=" << vad.arousal
        << " dominance=" << vad.dominance;
    if (!vad.style_guide.tone_label.empty()) {
        out << "; tone=" << vad.style_guide.tone_label;
    }
    out << "; keep warmth=" << vad.style_guide.warmth
        << ", energy=" << vad.style_guide.energy
        << ", assertiveness=" << vad.style_guide.assertiveness
        << ", hedging=" << vad.style_guide.hedging
        << ", directness=" << vad.style_guide.directness;
    if (!vad.style_guide.prompt_hints.empty()) {
        out << "; hint=" << vad.style_guide.prompt_hints.front();
    }
    out << ".";
    return out.str();
}

static std::string extract_json_object_payload(const std::string & text) {
    const std::string trimmed = trim_copy(text);
    const size_t json_start = trimmed.find('{');
    const size_t json_end = trimmed.rfind('}');
    if (json_start == std::string::npos || json_end == std::string::npos || json_end < json_start) {
        return trimmed;
    }
    return trimmed.substr(json_start, json_end - json_start + 1);
}

static server_heuristic_object parse_heuristic_object_response(const std::string & text) {
    const json payload = json::parse(extract_json_object_payload(text));
    server_heuristic_object heuristic = {};
    heuristic.heuristic_id = json_value(payload, "heuristic_id", std::string());
    heuristic.title = json_value(payload, "title", std::string());
    const json trigger = json_value(payload, "trigger", json::object());
    heuristic.task_types = json_value(trigger, "task_types", std::vector<std::string>());
    heuristic.tool_names = json_value(trigger, "tool_names", std::vector<std::string>());
    heuristic.struct_tags = json_value(trigger, "struct_tags", std::vector<std::string>());
    heuristic.emotive_conditions = json_value(trigger, "emotive_conditions", std::map<std::string, std::string>());
    heuristic.semantic_trigger_text = json_value(trigger, "semantic_trigger_text", std::string());
    const json diagnosis = json_value(payload, "diagnosis", json::object());
    heuristic.failure_mode = json_value(diagnosis, "failure_mode", std::string());
    heuristic.evidence = json_value(diagnosis, "evidence", std::vector<std::string>());
    const json intervention = json_value(payload, "intervention", json::object());
    heuristic.constraints = json_value(intervention, "constraints", std::vector<std::string>());
    heuristic.preferred_actions = json_value(intervention, "preferred_actions", std::vector<std::string>());
    heuristic.action_ranking_rules = json_value(intervention, "action_ranking_rules", std::vector<std::string>());
    heuristic.mid_reasoning_correction = json_value(intervention, "mid_reasoning_correction", std::string());
    const json scope = json_value(payload, "scope", json::object());
    heuristic.applies_when = json_value(scope, "applies_when", std::vector<std::string>());
    heuristic.avoid_when = json_value(scope, "avoid_when", std::vector<std::string>());
    const json confidence = json_value(payload, "confidence", json::object());
    heuristic.p_success = json_value(confidence, "p_success", heuristic.p_success);
    heuristic.calibration = json_value(confidence, "calibration", heuristic.calibration);
    heuristic.confidence_notes = json_value(confidence, "notes", std::string());
    return heuristic;
}

static json inject_additive_runtime_guidance(
        const json & body,
        const server_emotive_turn_builder * builder,
        server_emotive_runtime * emotive_runtime,
        bool enable_heuristic_guidance) {
    if ((!builder || !builder->has_current_state()) && !emotive_runtime) {
        return body;
    }
    if (!body.contains("messages") || !body.at("messages").is_array()) {
        return body;
    }

    json messages = body.at("messages");
    const active_tool_continuation_span span = find_active_tool_continuation_span(messages);
    const size_t insertion_index = span.valid ? span.last_tool_index + 1 : messages.size();

    std::string vad_guidance;
    if (builder && builder->has_current_state() && span.valid) {
        const auto & assistant = messages.at(span.assistant_index);
        const std::string reasoning = json_value(assistant, "reasoning_content", std::string());
        if (!reasoning.empty()) {
            vad_guidance = format_interleaved_vad_guidance(builder->current_vad());
        }
    }

    std::future<std::string> heuristic_guidance_future;
    bool launched_heuristic_search = false;
    if (enable_heuristic_guidance && emotive_runtime) {
        std::vector<std::string> struct_tags;
        const std::string query_text = build_live_heuristic_query_text(messages, &struct_tags);
        if (!query_text.empty()) {
            const bool have_state = builder && builder->has_current_state();
            const server_emotive_vector moment = have_state ? builder->current_moment() : server_emotive_vector();
            const server_emotive_vad vad = have_state ? builder->current_vad() : server_emotive_vad();
            heuristic_guidance_future = std::async(
                    std::launch::async,
                    [emotive_runtime, query_text, struct_tags, moment, vad, have_state]() {
                        std::string guidance;
                        emotive_runtime->retrieve_matching_heuristic(
                                query_text,
                                struct_tags,
                                have_state ? &moment : nullptr,
                                have_state ? &vad : nullptr,
                                &guidance);
                        return guidance;
                    });
            launched_heuristic_search = true;
        }
    }

    std::string heuristic_guidance;
    if (launched_heuristic_search) {
        heuristic_guidance = heuristic_guidance_future.get();
    }
    if (heuristic_guidance.empty() && vad_guidance.empty()) {
        return body;
    }

    auto insert_it = messages.begin() + static_cast<json::difference_type>(insertion_index);
    if (!heuristic_guidance.empty()) {
        insert_it = messages.insert(insert_it, json{
            {"role", "system"},
            {"content", heuristic_guidance},
        });
        ++insert_it;
    }
    if (!vad_guidance.empty()) {
        messages.insert(insert_it, json{
            {"role", "system"},
            {"content", vad_guidance},
        });
    }

    json augmented = body;
    augmented["messages"] = std::move(messages);
    return augmented;
}

static bool execute_deepseek_chat_with_emotive(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const json & body,
        deepseek_chat_result * out_result,
        json * out_error,
        server_emotive_trace * out_trace,
        bool cognitive_replay,
        const std::string & cognitive_replay_entry_id,
        bool enable_heuristic_guidance,
        bool suppress_replay_admission,
        const std::string & mode_label) {
    if (!out_result) {
        if (out_error) {
            *out_error = format_error_response("DeepSeek result output must not be null.", ERROR_TYPE_SERVER);
        }
        return false;
    }

    *out_result = deepseek_chat_result();
    if (out_trace) {
        *out_trace = server_emotive_trace();
    }

    std::unique_ptr<server_emotive_turn_builder> turn_builder;
    if (emotive_runtime.config().enabled) {
        turn_builder = std::make_unique<server_emotive_turn_builder>(
                emotive_runtime,
                config.model,
                cognitive_replay,
                cognitive_replay_entry_id,
                suppress_replay_admission,
                mode_label);
        seed_emotive_turn_from_request(body, turn_builder.get());
    }

    deepseek_stream_observer observer;
    if (turn_builder) {
        observer.on_reasoning_delta = [&turn_builder](const std::string & text) {
            turn_builder->observe_reasoning_delta(text);
        };
        observer.on_content_delta = [&turn_builder](const std::string & text) {
            turn_builder->observe_content_delta(text);
        };
        observer.on_runtime_event = [&turn_builder](const std::string & text) {
            turn_builder->observe_runtime_event(text);
        };
    }

    const json provider_body = inject_additive_runtime_guidance(
            body,
            turn_builder.get(),
            &emotive_runtime,
            enable_heuristic_guidance && !cognitive_replay);
    if (!deepseek_complete_chat(config, provider_body, out_result, turn_builder ? &observer : nullptr, out_error)) {
        return false;
    }

    if (turn_builder) {
        turn_builder->observe_runtime_event(
                "provider_finish:" + (out_result->finish_reason.empty() ? std::string("stop") : out_result->finish_reason));
        const server_emotive_trace trace = turn_builder->finalize();
        out_result->emotive_trace = server_emotive_trace_to_json(trace);
        if (out_trace) {
            *out_trace = trace;
        }
    }

    return true;
}

static json convert_responses_input_to_chat(const json & body) {
    json converted = {
        {"messages", json::array()},
        {"stream", json_value(body, "stream", false)},
    };

    const std::string model = json_value(body, "model", std::string());
    if (!model.empty()) {
        converted["model"] = model;
    }
    if (body.contains("stream_options")) {
        converted["stream_options"] = body.at("stream_options");
    }
    if (body.contains("temperature")) {
        converted["temperature"] = body.at("temperature");
    }
    if (body.contains("top_p")) {
        converted["top_p"] = body.at("top_p");
    }
    if (body.contains("presence_penalty")) {
        converted["presence_penalty"] = body.at("presence_penalty");
    }
    if (body.contains("frequency_penalty")) {
        converted["frequency_penalty"] = body.at("frequency_penalty");
    }
    if (body.contains("max_output_tokens")) {
        converted["max_output_tokens"] = body.at("max_output_tokens");
    }
    if (body.contains("max_completion_tokens")) {
        converted["max_completion_tokens"] = body.at("max_completion_tokens");
    }
    if (body.contains("tools")) {
        converted["tools"] = body.at("tools");
    }
    if (body.contains("tool_choice")) {
        converted["tool_choice"] = body.at("tool_choice");
    }
    if (body.contains("parallel_tool_calls")) {
        converted["parallel_tool_calls"] = body.at("parallel_tool_calls");
    }

    const std::string instructions = json_value(body, "instructions", std::string());
    if (!instructions.empty()) {
        converted["messages"].push_back({
            {"role", "system"},
            {"content", instructions},
        });
    }

    if (!body.contains("input")) {
        throw std::invalid_argument("Responses request must include \"input\".");
    }

    const json & input = body.at("input");
    if (input.is_string()) {
        converted["messages"].push_back({
            {"role", "user"},
            {"content", input},
        });
        return converted;
    }

    if (!input.is_array()) {
        throw std::invalid_argument("\"input\" must be a string or an array.");
    }

    for (const auto & item : input) {
        if (item.is_string()) {
            converted["messages"].push_back({
                {"role", "user"},
                {"content", item},
            });
            continue;
        }

        if (!item.is_object()) {
            continue;
        }

        const std::string type = json_value(item, "type", std::string());
        if (type == "input_text") {
            converted["messages"].push_back({
                {"role", "user"},
                {"content", json_value(item, "text", std::string())},
            });
            continue;
        }
        if (type == "function_call_output") {
            converted["messages"].push_back({
                {"role", "tool"},
                {"tool_call_id", json_value(item, "call_id", std::string())},
                {"content", item.contains("output") ? item.at("output") : item.value("content", json(""))},
            });
            continue;
        }

        if (item.contains("role")) {
            json message = {
                {"role", json_value(item, "role", std::string("user"))},
                {"content", item.contains("content") ? item.at("content") : json("")},
            };
            if (item.contains("reasoning_content")) {
                message["reasoning_content"] = item.at("reasoning_content");
            }
            if (item.contains("tool_calls")) {
                message["tool_calls"] = item.at("tool_calls");
            }
            if (item.contains("tool_call_id")) {
                message["tool_call_id"] = item.at("tool_call_id");
            }
            converted["messages"].push_back(std::move(message));
            continue;
        }

        if (type == "message") {
            json message = {
                {"role", json_value(item, "role", std::string("user"))},
                {"content", item.contains("content") ? item.at("content") : json("")},
            };
            if (item.contains("reasoning_content")) {
                message["reasoning_content"] = item.at("reasoning_content");
            }
            if (item.contains("tool_calls")) {
                message["tool_calls"] = item.at("tool_calls");
            }
            if (item.contains("tool_call_id")) {
                message["tool_call_id"] = item.at("tool_call_id");
            }
            converted["messages"].push_back(std::move(message));
        }
    }

    if (converted.at("messages").empty()) {
        throw std::invalid_argument("Responses request did not contain any usable input messages.");
    }

    return converted;
}

struct bridge_compat_state {
    std::atomic<bool> live_stream_connected = false;
};

static json cognitive_replay_worker_json(
        const cognitive_replay_worker_state & worker_state,
        const request_activity_state & activity_state) {
    std::lock_guard<std::mutex> lock(worker_state.mutex);
    return {
        {"running", worker_state.running},
        {"active_entry_id", worker_state.active_entry_id},
        {"last_error", worker_state.last_error},
        {"last_started_at_ms", worker_state.last_started_at_ms},
        {"last_finished_at_ms", worker_state.last_finished_at_ms},
        {"idle_ms", activity_state.idle_ms()},
        {"active_request_count", activity_state.active_requests.load()},
    };
}

static json ongoing_task_worker_json(
        const ongoing_task_worker_state & worker_state,
        const request_activity_state & activity_state,
        const server_ongoing_task_config & config) {
    std::lock_guard<std::mutex> lock(worker_state.mutex);
    return {
        {"enabled", config.enabled},
        {"mode", worker_state.mode},
        {"running", worker_state.running},
        {"active_task_id", worker_state.active_task_id},
        {"last_error", worker_state.last_error},
        {"last_poll_at_ms", worker_state.last_poll_at_ms},
        {"last_finished_at_ms", worker_state.last_finished_at_ms},
        {"last_completed_task_id", worker_state.last_completed_task_id},
        {"last_completed_at_iso", worker_state.last_completed_at_iso},
        {"last_decision", ongoing_task_decision_to_json(worker_state.last_decision)},
        {"poll_interval_ms", config.poll_interval_ms},
        {"timeout_ms", config.timeout_ms},
        {"registry_key", config.registry_key},
        {"container_tag", config.container_tag},
        {"idle_ms", activity_state.idle_ms()},
        {"active_request_count", activity_state.active_requests.load()},
    };
}

static bool execute_bridge_scoped_telegram_request(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const server_http_req & req,
        const json & body,
        deepseek_chat_result * out_result,
        json * out_error) {
    if (!out_result) {
        if (out_error) {
            *out_error = format_error_response("bridge-scoped Telegram execution requires a result output", ERROR_TYPE_SERVER);
        }
        return false;
    }
    if (!body.contains("messages") || !body.at("messages").is_array() || body.at("messages").empty()) {
        if (out_error) {
            *out_error = format_error_response("bridge-scoped Telegram requests must include a non-empty messages array", ERROR_TYPE_INVALID_REQUEST);
        }
        return false;
    }

    const telegram_bridge_request_context context = parse_telegram_bridge_request_context(req);
    if (!context.active) {
        if (out_error) {
            *out_error = format_error_response("bridge-scoped Telegram execution requires Telegram request headers", ERROR_TYPE_SERVER);
        }
        return false;
    }

    const telegram_runtime_tool_adapter_config adapter_config = telegram_runtime_tool_adapter_config_from_env();
    json runtime_tools;
    std::string tool_error;
    if (!load_server_owned_telegram_runtime_tools(adapter_config, context, &runtime_tools, &tool_error)) {
        if (out_error) {
            *out_error = format_error_response(tool_error, ERROR_TYPE_SERVER);
        }
        return false;
    }

    json current_body = build_server_owned_bridge_request_body(body, context, runtime_tools);
    for (int32_t round = 0; round < std::max<int32_t>(1, adapter_config.max_rounds); ++round) {
        deepseek_chat_result round_result;
        const bool use_staged_tools = should_use_staged_tool_loop_for_request(current_body);
        const bool executed = use_staged_tools ?
                execute_deepseek_chat_with_staged_tools(
                        config,
                        emotive_runtime,
                        current_body,
                        &round_result,
                        out_error,
                        false,
                        "telegram_bridge") :
                execute_deepseek_chat_with_emotive(
                        config,
                        emotive_runtime,
                        current_body,
                        &round_result,
                        out_error,
                        nullptr,
                        false,
                        std::string(),
                        true,
                        false,
                        "telegram_bridge");
        if (!executed) {
            return false;
        }

        if (round_result.tool_calls.empty()) {
            *out_result = std::move(round_result);
            return true;
        }

        bool contains_telegram_relay = false;
        bool contains_runtime_tool = false;
        for (const auto & tool_call : round_result.tool_calls) {
            if (tool_call.name == "telegram_relay") {
                contains_telegram_relay = true;
            } else {
                contains_runtime_tool = true;
            }
        }
        if (contains_telegram_relay && contains_runtime_tool) {
            if (out_error) {
                *out_error = format_error_response("bridge-scoped Telegram request returned mixed telegram_relay and runtime tool calls", ERROR_TYPE_SERVER);
            }
            return false;
        }
        if (contains_telegram_relay) {
            *out_result = std::move(round_result);
            return true;
        }

        json messages = current_body.at("messages");
        messages.push_back(build_bridge_assistant_tool_replay_message(round_result));
        for (const auto & tool_call : round_result.tool_calls) {
            json observation;
            std::string observation_error;
            if (!invoke_server_owned_telegram_runtime_tool(adapter_config, tool_call, &observation, &observation_error)) {
                if (out_error) {
                    *out_error = format_error_response(observation_error, ERROR_TYPE_SERVER);
                }
                return false;
            }
            messages.push_back(build_bridge_tool_result_message(tool_call, observation));
        }
        current_body["messages"] = std::move(messages);
    }

    if (out_error) {
        *out_error = format_error_response("bridge-scoped Telegram tool loop exceeded its maximum round budget", ERROR_TYPE_SERVER);
    }
    return false;
}

static server_http_res_ptr handle_deepseek_chat(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const server_http_req & req,
        telegram_outbox_state * telegram_outbox,
        const json & body,
        bool responses_api,
        bool text_completion_api) {
    json error;
    if (!deepseek_validate_runtime_config(config, &error)) {
        return make_error_response(error);
    }

    deepseek_chat_result result;
    const telegram_bridge_request_context telegram_context = parse_telegram_bridge_request_context(req);
    const bool executed = telegram_context.active ?
            execute_bridge_scoped_telegram_request(config, emotive_runtime, req, body, &result, &error) :
            (should_use_staged_tool_loop_for_request(body) ?
                    execute_deepseek_chat_with_staged_tools(config, emotive_runtime, body, &result, &error) :
                    execute_deepseek_chat_with_emotive(config, emotive_runtime, body, &result, &error));
    if (!executed) {
        return make_error_response(error);
    }

    telegram_delivery_result telegram_delivery = {};
    if (!execute_telegram_delivery_for_bridge_request(req, telegram_outbox, &result, &telegram_delivery, &error)) {
        return make_error_response(error);
    }

    const std::string completion_id = gen_chatcmplid();
    if (responses_api) {
        json response = deepseek_format_responses_response(config, result);
        const json delivery_json = telegram_delivery_result_to_json(telegram_delivery);
        if (!delivery_json.is_null()) {
            response["vicuna_telegram_delivery"] = delivery_json;
        }
        return make_json_response(response);
    }
    if (text_completion_api) {
        json response = deepseek_format_text_completion_response(config, result, completion_id);
        const json delivery_json = telegram_delivery_result_to_json(telegram_delivery);
        if (!delivery_json.is_null()) {
            response["vicuna_telegram_delivery"] = delivery_json;
        }
        return make_json_response(response);
    }

    const bool stream = json_value(body, "stream", false);
    if (stream) {
        const json stream_options = json_value(body, "stream_options", json::object());
        const bool include_usage = json_value(stream_options, "include_usage", false);

        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        res->data = format_oai_sse(
                deepseek_format_chat_completion_stream(config, result, completion_id, include_usage));
        return res;
    }

    json response = deepseek_format_chat_completion_response(config, result, completion_id);
    const json delivery_json = telegram_delivery_result_to_json(telegram_delivery);
    if (!delivery_json.is_null()) {
        response["vicuna_telegram_delivery"] = delivery_json;
    }
    return make_json_response(response);
}

static void ongoing_task_worker_set_mode(
        ongoing_task_worker_state * worker_state,
        const std::string & mode,
        bool running,
        const std::string & active_task_id = std::string()) {
    if (!worker_state) {
        return;
    }
    std::lock_guard<std::mutex> lock(worker_state->mutex);
    worker_state->mode = mode;
    worker_state->running = running;
    worker_state->active_task_id = active_task_id;
}

static void ongoing_task_worker_set_error(ongoing_task_worker_state * worker_state, const std::string & error) {
    if (!worker_state) {
        return;
    }
    std::lock_guard<std::mutex> lock(worker_state->mutex);
    worker_state->last_error = error;
}

static void ongoing_task_worker_set_decision(
        ongoing_task_worker_state * worker_state,
        const server_ongoing_task_decision & decision,
        const std::string & error = std::string()) {
    if (!worker_state) {
        return;
    }
    std::lock_guard<std::mutex> lock(worker_state->mutex);
    worker_state->last_decision = decision;
    worker_state->last_error = error;
}

static void ongoing_task_worker_mark_poll(ongoing_task_worker_state * worker_state, int64_t poll_ms) {
    if (!worker_state) {
        return;
    }
    std::lock_guard<std::mutex> lock(worker_state->mutex);
    worker_state->last_poll_at_ms = poll_ms;
}

static void ongoing_task_worker_mark_completion(
        ongoing_task_worker_state * worker_state,
        const std::string & task_id,
        const std::string & completed_at_iso) {
    if (!worker_state) {
        return;
    }
    std::lock_guard<std::mutex> lock(worker_state->mutex);
    worker_state->last_completed_task_id = task_id;
    worker_state->last_completed_at_iso = completed_at_iso;
    worker_state->last_finished_at_ms = current_epoch_ms();
    worker_state->last_error.clear();
}

static void ongoing_task_worker_mark_finished(ongoing_task_worker_state * worker_state) {
    if (!worker_state) {
        return;
    }
    std::lock_guard<std::mutex> lock(worker_state->mutex);
    worker_state->last_finished_at_ms = current_epoch_ms();
    if (worker_state->mode != "error") {
        worker_state->mode = "idle";
    }
    worker_state->running = false;
    worker_state->active_task_id.clear();
}

static bool request_activity_is_idle_for_background(
        const request_activity_state * activity_state,
        int64_t idle_after_ms) {
    if (!activity_state) {
        return true;
    }
    return activity_state->active_requests.load() <= 0 && activity_state->idle_ms() >= idle_after_ms;
}

static bool decide_due_ongoing_task(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const server_ongoing_task_config & ongoing_config,
        ongoing_task_worker_state * worker_state,
        server_ongoing_task_registry * out_registry,
        server_ongoing_task_summary * out_selected_task) {
    if (!out_registry) {
        ongoing_task_worker_set_error(worker_state, "ongoing-task registry output must not be null");
        return false;
    }
    if (out_selected_task) {
        *out_selected_task = server_ongoing_task_summary();
    }

    const int64_t poll_ms = current_time_ms_utc();
    ongoing_task_worker_mark_poll(worker_state, poll_ms);
    ongoing_task_worker_set_mode(worker_state, "ongoing_task_poll", true);

    server_ongoing_task_registry registry = {};
    std::string registry_error;
    if (!load_ongoing_task_registry(ongoing_config, &registry, &registry_error)) {
        ongoing_task_worker_set_error(worker_state, registry_error);
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    const std::string current_time_iso = iso_from_epoch_ms(poll_ms);
    const std::vector<server_ongoing_task_summary> tasks = list_active_ongoing_task_summaries(registry, poll_ms);
    server_ongoing_task_decision decision = {};
    decision.valid = true;
    decision.decided_at_ms = poll_ms;
    decision.current_time_iso = current_time_iso;
    decision.task_count = (int32_t) tasks.size();

    if (tasks.empty()) {
        decision.should_run = false;
        decision.rationale = "No active ongoing tasks are registered.";
        ongoing_task_worker_set_decision(worker_state, decision);
        *out_registry = std::move(registry);
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    ongoing_task_worker_set_mode(worker_state, "ongoing_task_decision", true);
    json body = {
        {"model", config.model},
        {"messages", json::array({
            {
                {"role", "system"},
                {"content", build_ongoing_task_decision_system_prompt()},
            },
            {
                {"role", "user"},
                {"content", build_ongoing_task_decision_user_prompt(tasks, current_time_iso)},
            },
        })},
        {"stream", false},
    };

    deepseek_chat_result result;
    json error;
    if (!execute_deepseek_chat_with_emotive(
                config,
                emotive_runtime,
                body,
                &result,
                &error,
                nullptr,
                false,
                std::string(),
                true,
                true,
                "ongoing_task_decision")) {
        ongoing_task_worker_set_error(
                worker_state,
                json_value(error, "message", std::string("ongoing-task decision request failed")));
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    std::string parse_error;
    if (!parse_ongoing_task_decision_response(result.content, &decision, &parse_error)) {
        ongoing_task_worker_set_error(worker_state, parse_error);
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }
    decision.decided_at_ms = poll_ms;
    decision.current_time_iso = current_time_iso;
    decision.task_count = (int32_t) tasks.size();

    if (decision.should_run && !find_ongoing_task_summary(tasks, decision.selected_task_id)) {
        ongoing_task_worker_set_error(worker_state, "ongoing-task decision selected an unknown task_id");
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    ongoing_task_worker_set_decision(worker_state, decision);
    *out_registry = std::move(registry);
    if (!decision.should_run) {
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    if (out_selected_task) {
        *out_selected_task = *find_ongoing_task_summary(tasks, decision.selected_task_id);
    }
    return true;
}

static bool execute_selected_ongoing_task(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const server_ongoing_task_config & ongoing_config,
        const server_ongoing_task_summary & task,
        server_ongoing_task_registry registry,
        ongoing_task_worker_state * worker_state) {
    ongoing_task_worker_set_mode(worker_state, "ongoing_task_execution", true, task.task_id);

    json body = {
        {"model", config.model},
        {"messages", json::array({
            {
                {"role", "user"},
                {"content", task.task_text},
            },
        })},
        {"stream", false},
    };

    deepseek_chat_result result;
    json error;
    if (!execute_deepseek_chat_with_emotive(
                config,
                emotive_runtime,
                body,
                &result,
                &error,
                nullptr,
                false,
                std::string(),
                true,
                true,
                "ongoing_task_execution")) {
        ongoing_task_worker_set_error(
                worker_state,
                json_value(error, "message", std::string("ongoing-task execution request failed")));
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    const std::string completed_at_iso = iso_from_epoch_ms(current_time_ms_utc());
    std::string completion_error;
    if (!mark_ongoing_task_complete(&registry, task.task_id, completed_at_iso, &completion_error)) {
        ongoing_task_worker_set_error(worker_state, completion_error);
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }
    if (!save_ongoing_task_registry(ongoing_config, std::move(registry), &completion_error)) {
        ongoing_task_worker_set_error(worker_state, completion_error);
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    ongoing_task_worker_mark_completion(worker_state, task.task_id, completed_at_iso);
    ongoing_task_worker_mark_finished(worker_state);
    return true;
}

static std::string build_cognitive_replay_system_prompt() {
    return
            "You are in cognitive replay mode. Review the stored episode and propose the action or "
            "action sequence that would likely have produced a less negative self-state transition. "
            "Stay grounded in the provided trace. Prefer concrete checks, ordering changes, and "
            "decision points over abstract reflection.";
}

static const char * cognitive_replay_block_kind_name(server_emotive_block_kind kind) {
    switch (kind) {
        case SERVER_EMOTIVE_BLOCK_USER_MESSAGE: return "user_message";
        case SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING: return "assistant_reasoning";
        case SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT: return "assistant_content";
        case SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT: return "runtime_event";
    }
    return "runtime_event";
}

static std::string build_cognitive_replay_user_prompt(const server_cognitive_replay_entry & entry) {
    std::ostringstream out;
    out << "Replay the following negative episode and find a better path.\n";
    out << "Measured drop:\n";
    out << "- negative_mass=" << std::fixed << std::setprecision(2) << entry.negative_mass << "\n";
    out << "- valence_drop=" << entry.valence_drop << "\n";
    out << "- dominance_drop=" << entry.dominance_drop << "\n";
    out << "Baseline after-state:\n";
    out << "- valence=" << entry.baseline_vad_after.valence << "\n";
    out << "- arousal=" << entry.baseline_vad_after.arousal << "\n";
    out << "- dominance=" << entry.baseline_vad_after.dominance << "\n";
    out << "Trace window:\n";
    for (const auto & block : entry.window_blocks) {
        out << block.block_index << " | " << cognitive_replay_block_kind_name(block.kind)
            << " | " << block.text << "\n";
    }
    out << "Return a concise answer with these sections:\n";
    out << "1. Better Path\n";
    out << "2. Why It Improves State\n";
    return out.str();
}

static std::string build_heuristic_compression_system_prompt() {
    return
            "You are compressing a resolved cognitive replay into exactly one reusable heuristic. "
            "Return exactly one JSON object and no markdown. The object must contain these top-level "
            "fields: heuristic_id, title, trigger, diagnosis, intervention, scope, confidence. "
            "Use the trigger/diagnosis/intervention/scope/confidence structure described in the prompt. "
            "Prefer concrete constraints and action ranking over abstract advice.";
}

static std::string build_heuristic_compression_user_prompt(
        const server_cognitive_replay_entry & entry,
        const server_cognitive_replay_result & result) {
    std::ostringstream out;
    out << "Compress the resolved replay into one heuristic JSON object.\n";
    out << "Measured improvement:\n";
    out << "- baseline_negative_mass=" << std::fixed << std::setprecision(2) << result.comparison.baseline_negative_mass << "\n";
    out << "- replay_negative_mass=" << result.comparison.replay_negative_mass << "\n";
    out << "- baseline_valence=" << result.comparison.baseline_valence << "\n";
    out << "- replay_valence=" << result.comparison.replay_valence << "\n";
    out << "- baseline_dominance=" << result.comparison.baseline_dominance << "\n";
    out << "- replay_dominance=" << result.comparison.replay_dominance << "\n";
    out << "Bad Path:\n";
    for (const auto & block : entry.window_blocks) {
        out << "- " << cognitive_replay_block_kind_name(block.kind) << ": " << block.text << "\n";
    }
    out << "Better Path reasoning_content:\n" << result.reasoning_content << "\n";
    out << "Better Path content:\n" << result.content << "\n";
    out << "Schema requirements:\n";
    out << "{\n";
    out << "  \"heuristic_id\": \"optional-string\",\n";
    out << "  \"title\": \"short title\",\n";
    out << "  \"trigger\": {\n";
    out << "    \"task_types\": [\"...\"],\n";
    out << "    \"tool_names\": [\"...\"],\n";
    out << "    \"struct_tags\": [\"...\"],\n";
    out << "    \"emotive_conditions\": {\"negative_mass\": \"...\", \"valence\": \"...\", \"dominance\": \"...\"},\n";
    out << "    \"semantic_trigger_text\": \"...\"\n";
    out << "  },\n";
    out << "  \"diagnosis\": {\n";
    out << "    \"failure_mode\": \"...\",\n";
    out << "    \"evidence\": [\"...\"]\n";
    out << "  },\n";
    out << "  \"intervention\": {\n";
    out << "    \"constraints\": [\"...\"],\n";
    out << "    \"preferred_actions\": [\"...\"],\n";
    out << "    \"action_ranking_rules\": [\"...\"],\n";
    out << "    \"mid_reasoning_correction\": \"...\"\n";
    out << "  },\n";
    out << "  \"scope\": {\n";
    out << "    \"applies_when\": [\"...\"],\n";
    out << "    \"avoid_when\": [\"...\"]\n";
    out << "  },\n";
    out << "  \"confidence\": {\n";
    out << "    \"p_success\": 0.0,\n";
    out << "    \"calibration\": \"manual\",\n";
    out << "    \"notes\": \"...\"\n";
    out << "  }\n";
    out << "}\n";
    out << "Return only valid JSON.\n";
    return out.str();
}

static void compress_resolved_replay_into_heuristic(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const server_cognitive_replay_entry & entry,
        const server_cognitive_replay_result & result,
        cognitive_replay_worker_state * worker_state) {
    json body = {
        {"model", config.model},
        {"messages", json::array({
            {
                {"role", "system"},
                {"content", build_heuristic_compression_system_prompt()},
            },
            {
                {"role", "user"},
                {"content", build_heuristic_compression_user_prompt(entry, result)},
            },
        })},
        {"stream", false},
    };

    deepseek_chat_result compression_result;
    json error;
    if (!execute_deepseek_chat_with_emotive(
                config,
                emotive_runtime,
                body,
                &compression_result,
                &error,
                nullptr,
                true,
                entry.entry_id,
                false)) {
        if (worker_state) {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->last_error = json_value(error, "message", std::string("heuristic compression request failed"));
        }
        return;
    }

    try {
        const server_heuristic_object heuristic = parse_heuristic_object_response(compression_result.content);
        std::string persistence_error;
        if (!emotive_runtime.store_heuristic_memory_record(entry.entry_id, heuristic, &persistence_error) &&
                worker_state) {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->last_error = persistence_error.empty() ?
                    std::string("heuristic persistence failed") :
                    persistence_error;
        }
    } catch (const std::exception & e) {
        if (worker_state) {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->last_error = e.what();
        }
    }
}

static void run_cognitive_replay_once(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const server_cognitive_replay_entry & entry,
        cognitive_replay_worker_state * worker_state) {
    json body = {
        {"model", config.model},
        {"messages", json::array({
            {
                {"role", "system"},
                {"content", build_cognitive_replay_system_prompt()},
            },
            {
                {"role", "user"},
                {"content", build_cognitive_replay_user_prompt(entry)},
            },
        })},
        {"stream", false},
    };

    deepseek_chat_result result;
    json error;
    server_emotive_trace replay_trace;
    if (!execute_deepseek_chat_with_emotive(
                config,
                emotive_runtime,
                body,
                &result,
                &error,
                &replay_trace,
                true,
                entry.entry_id)) {
        emotive_runtime.fail_cognitive_replay_entry(
                entry.entry_id,
                json_value(error, "message", std::string("cognitive replay request failed")));
        if (worker_state) {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->last_error = json_value(error, "message", std::string("cognitive replay request failed"));
        }
        return;
    }

    emotive_runtime.record_cognitive_replay_result(
            entry.entry_id,
            result.reasoning_content,
            result.content,
            replay_trace);

    server_cognitive_replay_entry resolved_entry;
    server_cognitive_replay_result resolved_result;
    if (emotive_runtime.get_cognitive_replay_resolution(entry.entry_id, &resolved_entry, &resolved_result) &&
            resolved_entry.status == SERVER_COGNITIVE_REPLAY_RESOLVED &&
            resolved_result.comparison.improved) {
        compress_resolved_replay_into_heuristic(config, emotive_runtime, resolved_entry, resolved_result, worker_state);
    }

    if (worker_state) {
        std::lock_guard<std::mutex> lock(worker_state->mutex);
        if (worker_state->last_error.empty()) {
            worker_state->last_error.clear();
        }
    }
}

static void cognitive_replay_worker_loop(
        std::atomic<bool> * stop_flag,
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        request_activity_state * activity_state,
        cognitive_replay_worker_state * worker_state,
        const server_ongoing_task_config & ongoing_config,
        ongoing_task_worker_state * ongoing_worker_state) {
    const auto & replay_config = emotive_runtime.config().cognitive_replay;
    const int32_t worker_poll_ms = [&]() {
        int32_t poll_ms = replay_config.enabled ? replay_config.poll_interval_ms : 250;
        if (ongoing_config.enabled) {
            poll_ms = replay_config.enabled ?
                    std::min(poll_ms, ongoing_config.poll_interval_ms) :
                    ongoing_config.poll_interval_ms;
        }
        return std::max<int32_t>(50, poll_ms);
    }();
    const auto sleep_for_poll = [worker_poll_ms]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(worker_poll_ms));
    };

    while (stop_flag && !stop_flag->load()) {
        if (!replay_config.enabled && !ongoing_config.enabled) {
            sleep_for_poll();
            continue;
        }
        if (!request_activity_is_idle_for_background(activity_state, replay_config.idle_after_ms)) {
            ongoing_task_worker_set_mode(ongoing_worker_state, "idle", false);
            sleep_for_poll();
            continue;
        }

        if (replay_config.enabled) {
            server_cognitive_replay_entry entry;
            if (emotive_runtime.try_claim_cognitive_replay_entry(&entry)) {
                if (worker_state) {
                    std::lock_guard<std::mutex> lock(worker_state->mutex);
                    worker_state->running = true;
                    worker_state->active_entry_id = entry.entry_id;
                    worker_state->last_started_at_ms = current_epoch_ms();
                }

                run_cognitive_replay_once(config, emotive_runtime, entry, worker_state);

                if (worker_state) {
                    std::lock_guard<std::mutex> lock(worker_state->mutex);
                    worker_state->running = false;
                    worker_state->active_entry_id.clear();
                    worker_state->last_finished_at_ms = current_epoch_ms();
                }

                sleep_for_poll();
                continue;
            }
        }

        if (!ongoing_config.enabled) {
            sleep_for_poll();
            continue;
        }

        int64_t last_poll_at_ms = 0;
        if (ongoing_worker_state) {
            std::lock_guard<std::mutex> lock(ongoing_worker_state->mutex);
            last_poll_at_ms = ongoing_worker_state->last_poll_at_ms;
        }
        const int64_t now_ms = current_epoch_ms();
        if (last_poll_at_ms > 0 && now_ms - last_poll_at_ms < ongoing_config.poll_interval_ms) {
            sleep_for_poll();
            continue;
        }

        server_ongoing_task_registry registry = {};
        server_ongoing_task_summary selected_task = {};
        const bool should_run_task = decide_due_ongoing_task(
                config,
                emotive_runtime,
                ongoing_config,
                ongoing_worker_state,
                &registry,
                &selected_task);
        if (!should_run_task) {
            sleep_for_poll();
            continue;
        }

        if (!request_activity_is_idle_for_background(activity_state, replay_config.idle_after_ms)) {
            ongoing_task_worker_set_mode(ongoing_worker_state, "idle", false);
            ongoing_task_worker_set_error(ongoing_worker_state, std::string());
            ongoing_task_worker_mark_finished(ongoing_worker_state);
            sleep_for_poll();
            continue;
        }

        execute_selected_ongoing_task(
                config,
                emotive_runtime,
                ongoing_config,
                selected_task,
                std::move(registry),
                ongoing_worker_state);
        sleep_for_poll();
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    server_runtime_params params;
    bool show_help = false;
    std::string arg_error;
    if (!server_runtime_params_parse(argc, argv, params, &show_help, &arg_error)) {
        if (!arg_error.empty()) {
            LOG_ERR("%s\n", arg_error.c_str());
        }
        return arg_error.empty() ? 0 : 1;
    }
    if (show_help) {
        server_runtime_print_usage(argv[0]);
        return 0;
    }

    if (params.n_parallel < 0) {
        params.n_parallel = 4;
    }

    server_runtime_init(params.verbose);

    if (params.n_threads < 0) {
        params.n_threads = std::max<int32_t>(1, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    if (params.n_threads_batch < 0) {
        params.n_threads_batch = params.n_threads;
    }

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n",
            params.n_threads,
            params.n_threads_batch,
            std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", server_runtime_system_info(params).c_str());
    LOG_INF("\n");

    const deepseek_runtime_config deepseek_config = deepseek_runtime_config_from_env();
    const server_emotive_runtime_config emotive_config = server_emotive_runtime_config_from_env();
    const server_ongoing_task_config ongoing_task_config = server_ongoing_task_config_from_env();
    server_emotive_runtime emotive_runtime(emotive_config);
    json config_error;
    if (!deepseek_validate_runtime_config(deepseek_config, &config_error)) {
        LOG_ERR("%s: %s\n", __func__, json_value(config_error, "message", std::string("invalid provider config")).c_str());
        return 1;
    }

    bridge_compat_state bridge_state;
    request_activity_state request_activity;
    cognitive_replay_worker_state replay_worker_state;
    ongoing_task_worker_state ongoing_worker_state;
    std::atomic<bool> replay_worker_stop(false);
    telegram_outbox_state telegram_outbox;
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    const auto get_health =
            [&bridge_state,
             &deepseek_config,
             &emotive_runtime,
             &ongoing_task_config,
             &telegram_outbox,
             &request_activity,
             &replay_worker_state,
             &ongoing_worker_state](const server_http_req &) {
        json payload = deepseek_build_health_json(deepseek_config);
        payload["proactive_mailbox"] = {
            {"stored_responses", 0},
            {"publish_total", 0},
            {"live_stream_connected", bridge_state.live_stream_connected.load()},
        };
        payload["telegram_outbox"] = telegram_outbox_health_json(telegram_outbox);
        payload["emotive_runtime"] = emotive_runtime.health_json();
        payload["emotive_runtime"]["cognitive_replay"]["worker"] =
                cognitive_replay_worker_json(replay_worker_state, request_activity);
        payload["emotive_runtime"]["ongoing_tasks"] =
                ongoing_task_worker_json(ongoing_worker_state, request_activity, ongoing_task_config);
        return make_json_response(payload);
    };

    const auto get_models = [&deepseek_config](const server_http_req &) {
        return make_json_response(deepseek_build_models_json(deepseek_config));
    };

    const auto get_latest_emotive_trace = [&emotive_runtime](const server_http_req &) {
        return make_json_response(emotive_runtime.latest_trace_json());
    };

    const auto get_cognitive_replay = [&emotive_runtime, &request_activity, &replay_worker_state](const server_http_req &) {
        json payload = emotive_runtime.cognitive_replay_json();
        payload["worker"] = cognitive_replay_worker_json(replay_worker_state, request_activity);
        return make_json_response(payload);
    };

    const auto get_heuristic_memory = [&emotive_runtime](const server_http_req &) {
        return make_json_response(emotive_runtime.heuristic_memory_json());
    };

    const auto get_ongoing_tasks =
            [&request_activity, &ongoing_task_config, &ongoing_worker_state](const server_http_req &) {
        return make_json_response({
            {"object", "vicuna.emotive.ongoing_tasks"},
            {"worker", ongoing_task_worker_json(ongoing_worker_state, request_activity, ongoing_task_config)},
        });
    };

    const auto post_chat_completions = [&deepseek_config, &emotive_runtime, &telegram_outbox](const server_http_req & req) {
        const json body = json::parse(req.body);
        return handle_deepseek_chat(deepseek_config, emotive_runtime, req, &telegram_outbox, body, false, false);
    };

    const auto post_completions = [&deepseek_config, &emotive_runtime, &telegram_outbox](const server_http_req & req) {
        const json body = json::parse(req.body);
        return handle_deepseek_chat(deepseek_config, emotive_runtime, req, &telegram_outbox, body, false, true);
    };

    const auto post_responses = [&deepseek_config, &emotive_runtime, &telegram_outbox](const server_http_req & req) {
        const json body = convert_responses_input_to_chat(json::parse(req.body));
        return handle_deepseek_chat(deepseek_config, emotive_runtime, req, &telegram_outbox, body, true, false);
    };

    const auto get_responses_stream = [&bridge_state](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        res->headers["Cache-Control"] = "no-cache";
        res->headers["Connection"] = "keep-alive";
        res->data = ": connected\n\n";

        bridge_state.live_stream_connected.store(true);
        auto should_stop = req.should_stop;
        res->next = [&bridge_state, should_stop, last_heartbeat = std::chrono::steady_clock::now()](std::string & output) mutable -> bool {
            if (should_stop()) {
                bridge_state.live_stream_connected.store(false);
                return false;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const auto now = std::chrono::steady_clock::now();
            if (now - last_heartbeat >= std::chrono::seconds(15)) {
                output = ": keepalive\n\n";
                last_heartbeat = now;
            } else {
                output.clear();
            }
            return true;
        };
        return res;
    };

    const auto get_telegram_outbox = [&telegram_outbox](const server_http_req & req) {
        const uint64_t after = std::max<uint64_t>(0, std::strtoull(req.get_param("after", "0").c_str(), nullptr, 10));
        return make_json_response(telegram_outbox_read_json(telegram_outbox, after));
    };

    const auto post_telegram_outbox = [&telegram_outbox](const server_http_req & req) {
        const json body = req.body.empty() ? json::object() : json::parse(req.body);
        return make_json_response(telegram_outbox_enqueue_json(&telegram_outbox, body));
    };

    const auto post_telegram_approval = [](const server_http_req & req) {
        const json body = req.body.empty() ? json::object() : json::parse(req.body);
        return make_json_response({
            {"ok", true},
            {"status", "accepted"},
            {"approval_id", json_value(body, "approval_id", std::string())},
        });
    };

    const auto post_telegram_interruption = [](const server_http_req &) {
        return make_json_response({
            {"ok", true},
            {"cancelled_approval_ids", json::array()},
        });
    };

    ctx_http.get("/health", ex_wrapper(get_health));
    ctx_http.get("/v1/health", ex_wrapper(get_health));
    ctx_http.get("/v1/models", ex_wrapper(get_models));
    ctx_http.get("/v1/emotive/trace/latest", ex_wrapper(get_latest_emotive_trace));
    ctx_http.get("/v1/emotive/cognitive-replay", ex_wrapper(get_cognitive_replay));
    ctx_http.get("/v1/emotive/heuristics", ex_wrapper(get_heuristic_memory));
    ctx_http.get("/v1/emotive/ongoing-tasks", ex_wrapper(get_ongoing_tasks));
    ctx_http.post("/v1/chat/completions", ex_wrapper(post_chat_completions, &request_activity));
    ctx_http.post("/v1/completions", ex_wrapper(post_completions, &request_activity));
    ctx_http.post("/v1/responses", ex_wrapper(post_responses, &request_activity));
    ctx_http.get("/v1/responses/stream", ex_wrapper(get_responses_stream));
    ctx_http.get("/v1/telegram/outbox", ex_wrapper(get_telegram_outbox));
    ctx_http.post("/v1/telegram/outbox", ex_wrapper(post_telegram_outbox));
    ctx_http.post("/v1/telegram/approval", ex_wrapper(post_telegram_approval));
    ctx_http.post("/v1/telegram/interruption", ex_wrapper(post_telegram_interruption));

    if (params.api_surface != SERVER_API_SURFACE_OPENAI) {
        ctx_http.post("/chat/completions", ex_wrapper(post_chat_completions, &request_activity));
        ctx_http.post("/completions", ex_wrapper(post_completions, &request_activity));
        ctx_http.post("/responses", ex_wrapper(post_responses, &request_activity));
        ctx_http.get("/models", ex_wrapper(get_models));
    }

    if (!ctx_http.start()) {
        LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
        return 1;
    }
    ctx_http.is_ready.store(true);

    std::thread replay_worker_thread;
    if (emotive_runtime.config().cognitive_replay.enabled || ongoing_task_config.enabled) {
        replay_worker_thread = std::thread(
                cognitive_replay_worker_loop,
                &replay_worker_stop,
                std::cref(deepseek_config),
                std::ref(emotive_runtime),
                &request_activity,
                &replay_worker_state,
                std::cref(ongoing_task_config),
                &ongoing_worker_state);
    }

    shutdown_handler = [&](int) {
        replay_worker_stop.store(true);
        ctx_http.stop();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    LOG_INF("%s: DeepSeek provider server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    replay_worker_stop.store(true);
    if (replay_worker_thread.joinable()) {
        replay_worker_thread.join();
    }

    return 0;
}
