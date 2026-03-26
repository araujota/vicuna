#include "server-common.h"
#include "server-deepseek.h"
#include "server-emotive-runtime.h"
#include "server-http.h"
#include "server-runtime.h"
#include "../../common/base64.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <clocale>
#include <cctype>
#include <cmath>
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
struct runtime_request_trace_context;
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
        float ongoing_task_due = 0.0f,
        const std::string & mode_label = std::string(),
        const runtime_request_trace_context * trace_context = nullptr);

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

struct telegram_emotive_dimension_spec {
    const char * id;
    const char * label;
};

static const std::array<telegram_emotive_dimension_spec, 14> k_telegram_emotive_dimensions = {{
    {"epistemic_pressure", "Epistemic Pressure"},
    {"confidence", "Confidence"},
    {"contradiction_pressure", "Contradiction Pressure"},
    {"planning_clarity", "Planning Clarity"},
    {"curiosity", "Curiosity"},
    {"caution", "Caution"},
    {"frustration", "Frustration"},
    {"satisfaction", "Satisfaction"},
    {"momentum", "Momentum"},
    {"stall", "Stall"},
    {"semantic_novelty", "Semantic Novelty"},
    {"user_alignment", "User Alignment"},
    {"runtime_trust", "Runtime Trust"},
    {"runtime_failure_pressure", "Runtime Failure Pressure"},
}};

static json telegram_emotive_dimension_direction_json(size_t index, size_t total) {
    if (total == 0) {
        return json::array({0.0, 1.0, 0.0});
    }
    static const double pi = 3.14159265358979323846;
    const double golden_angle = pi * (3.0 - std::sqrt(5.0));
    const double y = 1.0 - (2.0 * (static_cast<double>(index) + 0.5) / static_cast<double>(total));
    const double radius = std::sqrt(std::max(0.0, 1.0 - y * y));
    const double theta = golden_angle * static_cast<double>(index);
    return json::array({
        std::cos(theta) * radius,
        y,
        std::sin(theta) * radius,
    });
}

static json build_telegram_emotive_moment_json(const json & raw_moment) {
    if (!raw_moment.is_object()) {
        return nullptr;
    }

    json moment = json::object();
    for (const auto & dimension : k_telegram_emotive_dimensions) {
        moment[dimension.id] = json_value(raw_moment, dimension.id, 0.0f);
    }
    return moment;
}

static json build_telegram_emotive_animation_bundle(const json & trace) {
    if (!trace.is_object()) {
        return nullptr;
    }

    const json blocks = trace.value("blocks", json::array());
    if (!blocks.is_array() || blocks.empty()) {
        return nullptr;
    }

    const int32_t total_blocks = static_cast<int32_t>(blocks.size());
    const int32_t requested_start = std::max<int32_t>(0, json_value(trace, "live_generation_start_block_index", int32_t(0)));
    const int32_t live_start = std::min<int32_t>(requested_start, total_blocks);

    json dimensions = json::array();
    for (size_t index = 0; index < k_telegram_emotive_dimensions.size(); ++index) {
        const auto & dimension = k_telegram_emotive_dimensions[index];
        dimensions.push_back({
            {"id", dimension.id},
            {"label", dimension.label},
            {"direction_index", static_cast<int32_t>(index)},
            {"direction_xyz", telegram_emotive_dimension_direction_json(index, k_telegram_emotive_dimensions.size())},
        });
    }

    json keyframes = json::array();
    for (int32_t block_index = live_start; block_index < total_blocks; ++block_index) {
        const json & block = blocks.at(static_cast<size_t>(block_index));
        if (!block.is_object()) {
            continue;
        }

        const json moment = build_telegram_emotive_moment_json(block.value("moment", json::object()));
        if (moment.is_null()) {
            continue;
        }

        keyframes.push_back({
            {"ordinal", static_cast<int32_t>(keyframes.size())},
            {"trace_block_index", json_value(block, "block_index", block_index)},
            {"source_kind", json_value(block.value("source", json::object()), "kind", std::string("runtime_event"))},
            {"moment", moment},
            {"dominant_dimensions", block.value("vad", json::object()).value("dominant_dimensions", json::array())},
        });
    }

    if (keyframes.empty()) {
        return nullptr;
    }

    const double seconds_per_keyframe = 0.5;
    return {
        {"bundle_version", 1},
        {"trace_id", json_value(trace, "trace_id", std::string())},
        {"generation_start_block_index", live_start},
        {"seconds_per_keyframe", seconds_per_keyframe},
        {"fps", 24},
        {"viewport_width", 720},
        {"viewport_height", 720},
        {"rotation_period_seconds", 12.0},
        {"duration_seconds", seconds_per_keyframe * static_cast<double>(keyframes.size())},
        {"dimensions", std::move(dimensions)},
        {"keyframes", std::move(keyframes)},
    };
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

struct runtime_request_trace_context {
    std::string request_id;
    std::string route;
    std::string mode_label;
    bool bridge_scoped = false;
    std::string telegram_chat_scope;
    std::string telegram_conversation_id;
    int64_t telegram_message_id = 0;
    int64_t started_at_ms = 0;
};

struct runtime_request_trace_state {
    mutable std::mutex mutex;
    std::deque<json> events;
    size_t max_events = 512;
    uint64_t total_events = 0;
};

static runtime_request_trace_state & runtime_request_trace_state_instance() {
    static runtime_request_trace_state state;
    return state;
}

static std::string generate_runtime_trace_request_id() {
    static std::atomic<uint64_t> counter = 0;
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    const uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);
    return "vicuna_req_" + std::to_string(now) + "_" + std::to_string(seq);
}

static int64_t runtime_now_epoch_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string request_trace_id_from_headers(const server_http_req & req) {
    std::string request_id = trim_copy(request_header_value(req, "X-Client-Request-Id"));
    if (request_id.empty()) {
        request_id = trim_copy(request_header_value(req, "X-Vicuna-Request-Id"));
    }
    if (request_id.empty()) {
        request_id = generate_runtime_trace_request_id();
    }
    return request_id;
}

static void runtime_request_trace_log(
        const runtime_request_trace_context & context,
        const std::string & component,
        const std::string & event,
        json data = json::object()) {
    if (!data.is_object()) {
        data = json::object();
    }
    const int64_t now_ms = runtime_now_epoch_ms();
    json payload = {
        {"timestamp_ms", now_ms},
        {"request_id", context.request_id},
        {"component", component},
        {"event", event},
        {"route", context.route},
        {"mode_label", context.mode_label},
        {"bridge_scoped", context.bridge_scoped},
        {"data", std::move(data)},
    };
    if (!context.telegram_chat_scope.empty()) {
        payload["telegram_chat_scope"] = context.telegram_chat_scope;
    }
    if (!context.telegram_conversation_id.empty()) {
        payload["telegram_conversation_id"] = context.telegram_conversation_id;
    }
    if (context.telegram_message_id > 0) {
        payload["telegram_message_id"] = context.telegram_message_id;
    }

    auto & state = runtime_request_trace_state_instance();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.events.push_back(payload);
        ++state.total_events;
        while (state.events.size() > state.max_events) {
            state.events.pop_front();
        }
    }

    SRV_INF("request_trace: %s\n", safe_json_to_str(payload).c_str());
}

static json runtime_request_trace_read_json(const server_http_req & req) {
    const std::string request_id = trim_copy(req.get_param("request_id", ""));
    const int64_t limit = std::max<int64_t>(1, std::min<int64_t>(
            512,
            std::strtoll(req.get_param("limit", "100").c_str(), nullptr, 10)));

    auto & state = runtime_request_trace_state_instance();
    json items = json::array();
    std::lock_guard<std::mutex> lock(state.mutex);
    for (auto it = state.events.rbegin(); it != state.events.rend(); ++it) {
        if (!request_id.empty() && json_value(*it, "request_id", std::string()) != request_id) {
            continue;
        }
        items.push_back(*it);
        if ((int64_t) items.size() >= limit) {
            break;
        }
    }
    std::reverse(items.begin(), items.end());
    return {
        {"object", "vicuna.request_traces"},
        {"request_id", request_id.empty() ? json(nullptr) : json(request_id)},
        {"count", static_cast<int64_t>(items.size())},
        {"items", std::move(items)},
    };
}

static json runtime_request_trace_health_json() {
    auto & state = runtime_request_trace_state_instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    return {
        {"stored_events", static_cast<int64_t>(state.events.size())},
        {"max_events", static_cast<int64_t>(state.max_events)},
        {"total_events", static_cast<int64_t>(state.total_events)},
        {"latest_request_id", state.events.empty() ? json(nullptr) : json(json_value(state.events.back(), "request_id", std::string()))},
        {"latest_event", state.events.empty() ? json(nullptr) : json(json_value(state.events.back(), "event", std::string()))},
    };
}

struct telegram_outbox_item {
    uint64_t sequence_number = 0;
    std::string kind = "message";
    std::string chat_scope;
    std::string telegram_method = "sendMessage";
    json telegram_payload = json::object();
    json emotive_animation = nullptr;
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

static runtime_request_trace_context make_runtime_request_trace_context(
        const server_http_req & req,
        const std::string & route,
        const telegram_bridge_request_context & telegram_context,
        const std::string & mode_label = std::string()) {
    runtime_request_trace_context context;
    context.request_id = request_trace_id_from_headers(req);
    context.route = route;
    context.mode_label = mode_label;
    context.bridge_scoped = telegram_context.active;
    context.telegram_chat_scope = telegram_context.chat_scope;
    context.telegram_conversation_id = telegram_context.conversation_id;
    context.telegram_message_id = telegram_context.reply_to_message_id;
    context.started_at_ms = runtime_now_epoch_ms();
    return context;
}

static runtime_request_trace_context make_background_trace_context(
        const std::string & route,
        const std::string & mode_label) {
    runtime_request_trace_context context;
    context.request_id = generate_runtime_trace_request_id();
    context.route = route;
    context.mode_label = mode_label;
    context.started_at_ms = runtime_now_epoch_ms();
    return context;
}

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

static json build_telegram_inline_button_schema() {
    return json{
        {"type", "object"},
        {"description", "One inline keyboard button. Set text plus either url or callback_data when needed."},
        {"properties", {
            {"text", {
                {"type", "string"},
                {"description", "Visible button label shown to the user."},
            }},
            {"url", {
                {"type", "string"},
                {"description", "Optional HTTPS URL to open when the user taps the button."},
            }},
            {"callback_data", {
                {"type", "string"},
                {"description", "Optional callback payload to send back to the bot when the button is tapped."},
            }},
        }},
        {"required", json::array({"text"})},
        {"additionalProperties", false},
    };
}

static json build_telegram_reply_markup_schema() {
    return json{
        {"type", "object"},
        {"description", "Optional Telegram reply markup. Use inline_keyboard for button rows."},
        {"properties", {
            {"inline_keyboard", {
                {"type", "array"},
                {"description", "Optional inline keyboard rows. Each row is an array of buttons."},
                {"items", {
                    {"type", "array"},
                    {"description", "One inline keyboard row."},
                    {"minItems", 1},
                    {"items", build_telegram_inline_button_schema()},
                }},
            }},
        }},
        {"additionalProperties", false},
    };
}

static json build_telegram_bridge_tool(
        const std::string & tool_name,
        const std::string & method_name,
        const std::string & method_description,
        json parameters) {
    return json{
        {"type", "function"},
        {"function", {
            {"name", tool_name},
            {"description", method_description},
            {"parameters", std::move(parameters)},
            {"x-vicuna-family-id", "telegram"},
            {"x-vicuna-family-name", "Telegram"},
            {"x-vicuna-family-description", "MANDATORY: THIS IS THE ONLY WAY TO COMMUNICATE WITH THE USER on bridge-scoped Telegram requests. Send direct user-facing follow-up messages through the Telegram bridge outbox."},
            {"x-vicuna-method-name", method_name},
            {"x-vicuna-method-description", method_description},
        }},
    };
}

static json build_telegram_bridge_tools() {
    const json reply_markup = build_telegram_reply_markup_schema();

    json tools = json::array();

    json plain_props = json::object();
    plain_props["text"] = {
        {"type", "string"},
        {"description", "Plain Telegram text to send to the user."},
    };
    tools.push_back(build_telegram_bridge_tool(
            "telegram_send_plain_text",
            "send_plain_text",
            "Send a simple plain-text Telegram reply. Do not include Markdown or HTML formatting markers.",
            {
                {"type", "object"},
                {"description", "Payload for a simple Telegram plain-text reply."},
                {"required", json::array({"text"})},
                {"properties", std::move(plain_props)},
                {"additionalProperties", false},
            }));

    json rich_props = json::object();
    rich_props["text"] = {
        {"type", "string"},
        {"description", "Telegram message text with formatting markup."},
    };
    rich_props["parse_mode"] = {
        {"type", "string"},
        {"enum", json::array({"HTML"})},
        {"description", "Formatting mode for the message text. Prefer HTML tags for Telegram-rich text."},
    };
    rich_props["disable_web_page_preview"] = {
        {"type", "boolean"},
        {"description", "Set true to disable link previews for the message."},
    };
    rich_props["reply_markup"] = reply_markup;
    tools.push_back(build_telegram_bridge_tool(
            "telegram_send_formatted_text",
            "send_formatted_text",
            "Send a formatted Telegram text reply with HTML parse_mode and optional reply markup.",
            {
                {"type", "object"},
                {"description", "Payload for a formatted Telegram text reply."},
                {"required", json::array({"text"})},
                {"properties", std::move(rich_props)},
                {"additionalProperties", false},
            }));

    json photo_props = json::object();
    photo_props["photo"] = {
        {"type", "string"},
        {"description", "Telegram file_id, URL, or attachable reference for the photo."},
    };
    photo_props["caption"] = {
        {"type", "string"},
        {"description", "Optional caption for the photo."},
    };
    photo_props["parse_mode"] = {
        {"type", "string"},
        {"enum", json::array({"HTML"})},
        {"description", "Formatting mode for the photo caption. Prefer HTML tags for Telegram-rich text."},
    };
    photo_props["reply_markup"] = reply_markup;
    tools.push_back(build_telegram_bridge_tool(
            "telegram_send_photo",
            "send_photo",
            "Send a Telegram photo with an optional formatted caption.",
            {
                {"type", "object"},
                {"description", "Payload for a Telegram photo reply."},
                {"required", json::array({"photo"})},
                {"properties", std::move(photo_props)},
                {"additionalProperties", false},
            }));

    json document_props = json::object();
    document_props["document"] = {
        {"type", "string"},
        {"description", "Telegram file_id, URL, or attachable reference for the document."},
    };
    document_props["caption"] = {
        {"type", "string"},
        {"description", "Optional caption for the document."},
    };
    document_props["parse_mode"] = {
        {"type", "string"},
        {"enum", json::array({"HTML"})},
        {"description", "Formatting mode for the document caption. Prefer HTML tags for Telegram-rich text."},
    };
    document_props["reply_markup"] = reply_markup;
    tools.push_back(build_telegram_bridge_tool(
            "telegram_send_document",
            "send_document",
            "Send a Telegram document with an optional formatted caption.",
            {
                {"type", "object"},
                {"description", "Payload for a Telegram document reply."},
                {"required", json::array({"document"})},
                {"properties", std::move(document_props)},
                {"additionalProperties", false},
            }));

    json poll_props = json::object();
    poll_props["question"] = {
        {"type", "string"},
        {"description", "Poll question to show to the user."},
    };
    poll_props["options"] = {
        {"type", "array"},
        {"minItems", 2},
        {"items", {
            {"type", "string"},
        }},
        {"description", "Poll answer options. Provide at least two."},
    };
    poll_props["is_anonymous"] = {
        {"type", "boolean"},
        {"description", "Whether the poll is anonymous."},
    };
    poll_props["allows_multiple_answers"] = {
        {"type", "boolean"},
        {"description", "Whether the poll allows multiple answers."},
    };
    tools.push_back(build_telegram_bridge_tool(
            "telegram_send_poll",
            "send_poll",
            "Send a Telegram poll.",
            {
                {"type", "object"},
                {"description", "Payload for a Telegram poll reply."},
                {"required", json::array({"question", "options"})},
                {"properties", std::move(poll_props)},
                {"additionalProperties", false},
            }));

    json dice_props = json::object();
    dice_props["emoji"] = {
        {"type", "string"},
        {"enum", json::array({"🎲", "🎯", "🏀", "⚽", "🎳", "🎰"})},
        {"description", "Optional dice emoji variant to send."},
    };
    tools.push_back(build_telegram_bridge_tool(
            "telegram_send_dice",
            "send_dice",
            "Send a Telegram dice-style emoji roll.",
            {
                {"type", "object"},
                {"description", "Payload for a Telegram dice reply."},
                {"properties", std::move(dice_props)},
                {"additionalProperties", false},
            }));

    return tools;
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

struct telegram_runtime_tool_cache_state {
    mutable std::mutex mutex;
    std::string cache_key;
    json tools = json::array();
    bool loaded_from_override = false;
    uint64_t hits = 0;
    uint64_t misses = 0;
};

struct staged_tool_prompt_cache_state;

static telegram_runtime_tool_cache_state & telegram_runtime_tool_cache_state_instance() {
    static telegram_runtime_tool_cache_state state;
    return state;
}

static staged_tool_prompt_cache_state & staged_tool_prompt_cache_state_instance();

static std::string build_telegram_runtime_tool_cache_key(
        const telegram_runtime_tool_adapter_config & config,
        bool used_override,
        const json & payload) {
    if (used_override) {
        return "override:" + safe_json_to_str(payload);
    }

    std::string mtime_token = "missing";
    try {
        const auto mtime = std::filesystem::last_write_time(config.entry_path).time_since_epoch().count();
        mtime_token = std::to_string(static_cast<long long>(mtime));
    } catch (...) {
    }
    return "entry:" + config.node_bin + "|" + config.entry_path + "|" + mtime_token;
}

static bool load_server_owned_telegram_runtime_tools(
        const telegram_runtime_tool_adapter_config & config,
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

    const std::string cache_key = build_telegram_runtime_tool_cache_key(config, used_override, payload);
    auto & cache_state = telegram_runtime_tool_cache_state_instance();
    {
        std::lock_guard<std::mutex> lock(cache_state.mutex);
        if (!cache_state.cache_key.empty() && cache_state.cache_key == cache_key) {
            ++cache_state.hits;
            *out_tools = cache_state.tools;
            return true;
        }
    }

    if (!used_override) {
        const std::string command_line =
                shell_escape_single_quoted(config.node_bin) + " " +
                shell_escape_single_quoted(config.entry_path) + " runtime-tools 2>&1";
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

    json final_tools = json::array();
    for (const auto & runtime_tool : tools) {
        if (runtime_tool.is_object() && json_value(runtime_tool, "type", std::string()) == "function") {
            const json function = runtime_tool.value("function", json::object());
            const std::string tool_name = trim_copy(json_value(function, "name", std::string()));
            const std::string family_id = trim_copy(json_value(function, "x-vicuna-family-id", std::string()));
            if (tool_name == "telegram_relay" || family_id == "telegram") {
                continue;
            }
        }
        final_tools.push_back(runtime_tool);
    }

    {
        std::lock_guard<std::mutex> lock(cache_state.mutex);
        cache_state.cache_key = cache_key;
        cache_state.tools = final_tools;
        cache_state.loaded_from_override = used_override;
        ++cache_state.misses;
    }

    *out_tools = std::move(final_tools);
    return true;
}

static json telegram_runtime_tool_cache_health_json() {
    const auto & cache_state = telegram_runtime_tool_cache_state_instance();
    std::lock_guard<std::mutex> lock(cache_state.mutex);
    return {
        {"cached", !cache_state.cache_key.empty()},
        {"hits", cache_state.hits},
        {"misses", cache_state.misses},
        {"loaded_from_override", cache_state.loaded_from_override},
        {"tool_count", cache_state.tools.is_array() ? static_cast<int64_t>(cache_state.tools.size()) : 0},
    };
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
        const telegram_bridge_request_context & /*context*/,
        const json & runtime_tools) {
    json augmented = body;
    json messages = json::array();
    messages.push_back({
        {"role", "system"},
        {"content",
                "Bridge-scoped delivery contract: when you are ready to respond to the user, do not call a Telegram tool. "
                "Return normal assistant content instead. You may optionally start with YAML front matter delimited by --- lines. "
                "Supported fields are format (plain_text|markdown|html), title, disable_web_page_preview, delivery_hint (reply|replace_prompt|silent), "
                "and reply_markup. After the closing --- line, include the user-facing rich text body."},
    });
    for (const auto & item : body.at("messages")) {
        messages.push_back(item);
    }
    augmented["messages"] = std::move(messages);
    augmented["tools"] = runtime_tools;
    augmented["tool_choice"] = "auto";
    augmented["parallel_tool_calls"] = false;
    augmented["x-vicuna-bridge-scoped"] = true;
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

struct staged_tool_prompt_spec {
    std::string stable_prefix;
    std::string dynamic_suffix;
};

struct staged_tool_prompt_bundle {
    staged_tool_catalog catalog;
    staged_tool_prompt_spec family_selection_prompt;
    json family_selection_tools = json::array();
    std::map<std::string, staged_tool_prompt_spec> method_selection_prompts;
    std::map<std::string, json> method_selection_tools;
    std::map<std::string, std::map<std::string, staged_tool_prompt_spec>> payload_prompts;
    std::map<std::string, std::map<std::string, json>> payload_tools;
};

struct staged_tool_prompt_cache_state {
    mutable std::mutex mutex;
    std::string cache_key;
    std::shared_ptr<const staged_tool_prompt_bundle> bundle;
    uint64_t hits = 0;
    uint64_t misses = 0;
};

static staged_tool_prompt_cache_state & staged_tool_prompt_cache_state_instance() {
    static staged_tool_prompt_cache_state state;
    return state;
}

struct staged_tool_method_choice {
    staged_tool_method_choice_kind kind = STAGED_TOOL_METHOD_CHOOSE_METHOD;
    std::string method_name;
};

struct staged_tool_payload_choice {
    staged_tool_payload_choice_kind kind = STAGED_TOOL_PAYLOAD_SUBMIT;
    json payload = json::object();
};

static constexpr const char * STAGED_SELECTOR_TOOL_FAMILY = "select_family";
static constexpr const char * STAGED_SELECTOR_TOOL_METHOD = "select_method";
static constexpr const char * STAGED_SELECTOR_TOOL_SUBMIT_PAYLOAD = "submit_payload";
static constexpr const char * STAGED_SELECTOR_TOOL_GO_BACK = "go_back";

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
            "You are Vicuña, a helpful personal concierge. Select the next tool step that best helps the user. "
            "When a selector tool is available, call exactly one selector tool immediately and do not answer in plain text. "
            "Plain assistant text is invalid while a staged tool turn is active.";
}

static json build_staged_selector_tool(
        const std::string & name,
        const std::string & description,
        const json & parameters) {
    return {
        {"type", "function"},
        {"function", {
            {"name", name},
            {"description", description},
            {"strict", true},
            {"parameters", parameters},
        }},
    };
}

static json build_staged_selector_enum_tool(
        const std::string & name,
        const std::string & description,
        const std::string & field_name,
        const std::string & field_description,
        const std::vector<std::string> & values) {
    json enum_values = json::array();
    for (const auto & value : values) {
        enum_values.push_back(value);
    }
    return build_staged_selector_tool(name, description, {
        {"type", "object"},
        {"description", description},
        {"properties", {
            {field_name, {
                {"type", "string"},
                {"description", field_description},
                {"enum", enum_values},
            }},
        }},
        {"required", json::array({field_name})},
        {"additionalProperties", false},
    });
}

static json build_staged_back_selector_tool() {
    return build_staged_selector_tool(
            STAGED_SELECTOR_TOOL_GO_BACK,
            "Go back to the previous stage in the staged tool-selection loop.",
            {
                {"type", "object"},
                {"description", "No arguments. Call this tool to return to the previous stage."},
                {"properties", json::object()},
                {"required", json::array()},
                {"additionalProperties", false},
            });
}

static bool staged_schema_property_is_required(const json & schema, const std::string & key) {
    if (!schema.is_object() || !schema.contains("required") || !schema.at("required").is_array()) {
        return false;
    }
    for (const auto & item : schema.at("required")) {
        if (item.is_string() && item.get<std::string>() == key) {
            return true;
        }
    }
    return false;
}

static json build_staged_payload_provider_schema(const json & schema) {
    if (!schema.is_object()) {
        return json::object();
    }

    if (schema.contains("anyOf") && schema.at("anyOf").is_array()) {
        json strict_any_of = json::array();
        for (const auto & item : schema.at("anyOf")) {
            strict_any_of.push_back(build_staged_payload_provider_schema(item));
        }
        json result = {
            {"anyOf", std::move(strict_any_of)},
        };
        const std::string description = trim_copy(json_value(schema, "description", std::string()));
        if (!description.empty()) {
            result["description"] = description;
        }
        return result;
    }

    const std::string type = trim_copy(json_value(schema, "type", std::string()));
    if (type == "object" || schema.contains("properties")) {
        json properties = json::object();
        json required = json::array();
        if (schema.contains("properties") && schema.at("properties").is_object()) {
            for (const auto & item : schema.at("properties").items()) {
                const bool is_required = staged_schema_property_is_required(schema, item.key());
                if (is_required) {
                    properties[item.key()] = build_staged_payload_provider_schema(item.value());
                    required.push_back(item.key());
                }
            }
        }
        // DeepSeek strict mode requires every exposed object property to also
        // appear in `required`, so the provider-facing payload contract is a
        // projection to the original required subset only.
        json result = {
            {"type", "object"},
            {"properties", std::move(properties)},
            {"required", std::move(required)},
            {"additionalProperties", false},
        };
        const std::string description = trim_copy(json_value(schema, "description", std::string()));
        if (!description.empty()) {
            result["description"] = description;
        }
        return result;
    }

    if (type == "array") {
        json result = {
            {"type", "array"},
            {"items", build_staged_payload_provider_schema(schema.value("items", json::object()))},
        };
        const std::string description = trim_copy(json_value(schema, "description", std::string()));
        if (!description.empty()) {
            result["description"] = description;
        }
        return result;
    }

    json result = json::object();
    for (const char * key : {
                 "type",
                 "description",
                 "enum",
                 "const",
                 "default",
                 "minimum",
                 "maximum",
                 "exclusiveMinimum",
                 "exclusiveMaximum",
                 "multipleOf",
                 "pattern",
                 "format",
             }) {
        if (schema.contains(key)) {
            result[key] = schema.at(key);
        }
    }
    return result;
}

static void strip_staged_optional_nulls(
        json * value,
        const json & schema) {
    if (!value) {
        return;
    }
    if (!schema.is_object()) {
        return;
    }
    const std::string type = trim_copy(json_value(schema, "type", std::string()));
    if ((type == "object" || schema.contains("properties")) && value->is_object() &&
            schema.contains("properties") && schema.at("properties").is_object()) {
        std::vector<std::string> keys_to_erase;
        for (const auto & item : schema.at("properties").items()) {
            if (!value->contains(item.key())) {
                continue;
            }
            json & child = (*value)[item.key()];
            if (!staged_schema_property_is_required(schema, item.key()) &&
                    child.is_null()) {
                keys_to_erase.push_back(item.key());
                continue;
            }
            strip_staged_optional_nulls(&child, item.value());
        }
        for (const auto & key : keys_to_erase) {
            value->erase(key);
        }
        return;
    }
    if (type == "array" && value->is_array() && schema.contains("items")) {
        for (auto & item : *value) {
            strip_staged_optional_nulls(&item, schema.at("items"));
        }
    }
}

static staged_tool_prompt_spec build_staged_family_selection_prompt(const staged_tool_catalog & catalog) {
    staged_tool_prompt_spec prompt = {};
    prompt.stable_prefix =
            "Stage: select a tool family.\n"
            "Call the selector tool immediately.\n"
            "Choose the family whose brief description best matches the next step.";
    std::ostringstream out;
    out << "Available families:\n";
    for (const auto & family : catalog.families) {
        out << "- " << family.family_name << ": " << family.family_description << "\n";
    }
    prompt.dynamic_suffix = out.str();
    return prompt;
}

static json build_staged_family_selection_tools(const staged_tool_catalog & catalog) {
    std::vector<std::string> family_names;
    family_names.reserve(catalog.families.size());
    for (const auto & family : catalog.families) {
        family_names.push_back(family.family_name);
    }
    return json::array({
        build_staged_selector_enum_tool(
                STAGED_SELECTOR_TOOL_FAMILY,
                "Select the next tool family to use.",
                "family",
                "The exact family name to use for the next staged step.",
                family_names),
    });
}

static staged_tool_prompt_spec build_staged_method_selection_prompt(
        const staged_tool_family & family,
        bool allow_completion) {
    staged_tool_prompt_spec prompt = {};
    prompt.stable_prefix =
            "Stage: select a method within the chosen family.\n"
            "Call the selector tool immediately.\n"
            "Choose back to return to family selection.";
    std::ostringstream out;
    out << "Chosen family: " << family.family_name << "\n";
    out << "Available methods:\n";
    if (allow_completion) {
        out << "- complete: Finish the active staged loop without selecting another method.\n";
    }
    out << "- back: Return to family selection.\n";
    for (const auto & method : family.methods) {
        out << "- " << method.method_name << ": " << method.method_description << "\n";
    }
    prompt.dynamic_suffix = out.str();
    return prompt;
}

static json build_staged_method_selection_tools(
        const staged_tool_family & family,
        bool allow_completion) {
    std::vector<std::string> method_names = {"back"};
    if (allow_completion) {
        method_names.push_back("complete");
    }
    for (const auto & method : family.methods) {
        method_names.push_back(method.method_name);
    }
    return json::array({
        build_staged_selector_enum_tool(
                STAGED_SELECTOR_TOOL_METHOD,
                "Select the next method within the chosen tool family.",
                "method",
                "The exact method name to use next. Choose back to return or complete when the staged loop is done.",
                method_names),
    });
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

static staged_tool_prompt_spec build_staged_payload_prompt(
        const staged_tool_family & family,
        const staged_tool_method & method,
        const json & provider_schema,
        const std::string & validation_error = std::string()) {
    staged_tool_prompt_spec prompt = {};
    prompt.stable_prefix =
            "Stage: construct a payload for the selected method.\n"
            "Call exactly one payload tool immediately.\n"
            "Call go_back to return to method selection.\n"
            "Only provide the fields shown in this contract. "
            "The runtime keeps all other method parameters absent or fills transport defaults itself.";
    std::ostringstream out;
    out << "Chosen family: " << family.family_name << "\n";
    out << "Chosen method: " << method.method_name << "\n";
    out << "Method summary: " << method.method_description << "\n";
    if (!validation_error.empty()) {
        out << "Previous payload error: " << validation_error << "\n";
    }
    out << "Contract:\n";
    append_staged_contract_lines(out, "payload", provider_schema);
    prompt.dynamic_suffix = out.str();
    return prompt;
}

static json build_staged_payload_tools(
        const json & provider_schema) {
    return json::array({
        build_staged_selector_tool(
                STAGED_SELECTOR_TOOL_SUBMIT_PAYLOAD,
                "Submit the payload for the selected method.",
                provider_schema),
        build_staged_back_selector_tool(),
    });
}

static staged_tool_prompt_bundle build_staged_tool_prompt_bundle(const json & tools) {
    staged_tool_prompt_bundle bundle = {};
    bundle.catalog = build_staged_tool_catalog_from_request(tools);
    if (bundle.catalog.families.empty()) {
        return bundle;
    }

    bundle.family_selection_prompt = build_staged_family_selection_prompt(bundle.catalog);
    bundle.family_selection_tools = build_staged_family_selection_tools(bundle.catalog);
    for (const auto & family : bundle.catalog.families) {
        bundle.method_selection_prompts[family.family_name] =
                build_staged_method_selection_prompt(family, true);
        bundle.method_selection_tools[family.family_name] =
                build_staged_method_selection_tools(family, true);
        auto & payload_by_method = bundle.payload_prompts[family.family_name];
        auto & payload_tools_by_method = bundle.payload_tools[family.family_name];
        for (const auto & method : family.methods) {
            const json provider_schema = build_staged_payload_provider_schema(method.parameters);
            payload_by_method[method.method_name] = build_staged_payload_prompt(family, method, provider_schema);
            payload_tools_by_method[method.method_name] = build_staged_payload_tools(provider_schema);
        }
    }
    return bundle;
}

static std::shared_ptr<const staged_tool_prompt_bundle> get_cached_staged_tool_prompt_bundle(const json & tools) {
    auto & cache_state = staged_tool_prompt_cache_state_instance();
    const std::string cache_key = safe_json_to_str(tools);
    {
        std::lock_guard<std::mutex> lock(cache_state.mutex);
        if (cache_state.bundle && cache_state.cache_key == cache_key) {
            ++cache_state.hits;
            return cache_state.bundle;
        }
    }

    auto bundle = std::make_shared<staged_tool_prompt_bundle>(build_staged_tool_prompt_bundle(tools));
    {
        std::lock_guard<std::mutex> lock(cache_state.mutex);
        cache_state.cache_key = cache_key;
        cache_state.bundle = bundle;
        ++cache_state.misses;
    }
    return bundle;
}

static json staged_tool_prompt_cache_health_json() {
    const auto & cache_state = staged_tool_prompt_cache_state_instance();
    std::lock_guard<std::mutex> lock(cache_state.mutex);
    return {
        {"cached", cache_state.bundle != nullptr},
        {"hits", cache_state.hits},
        {"misses", cache_state.misses},
        {"family_count", cache_state.bundle ? static_cast<int64_t>(cache_state.bundle->catalog.families.size()) : 0},
    };
}

static json build_staged_messages(
        const json & original_messages,
        const json & base_body,
        const staged_tool_prompt_spec & stage_prompt,
        const std::string & validation_error = std::string()) {
    json messages = json::array();
    messages.push_back({
        {"role", "system"},
        {"content", build_staged_tool_core_system_prompt()},
    });
    if (json_value(base_body, "x-vicuna-bridge-scoped", false)) {
        messages.push_back({
            {"role", "system"},
            {"content",
                    "Bridge-scoped note: plain assistant text is never delivered to the user from a staged turn. "
                    "MANDATORY: use the Telegram family for every user-facing reply."},
        });
    }
    if (!stage_prompt.stable_prefix.empty()) {
        messages.push_back({
            {"role", "system"},
            {"content", stage_prompt.stable_prefix},
        });
    }
    if (!stage_prompt.dynamic_suffix.empty()) {
        messages.push_back({
            {"role", "system"},
            {"content", stage_prompt.dynamic_suffix},
        });
    }
    if (!validation_error.empty()) {
        messages.push_back({
            {"role", "system"},
            {"content",
                    "Previous response error: " + validation_error + "\n"
                    "MANDATORY: call exactly one selector tool immediately. "
                    "Do not answer in plain text."},
        });
    }
    if (original_messages.is_array()) {
        for (const auto & item : original_messages) {
            messages.push_back(item);
        }
    }
    return messages;
}

static json build_staged_provider_body(
        const json & base_body,
        const json & messages,
        const json & tools,
        bool stage_followup_guidance,
        int64_t stage_max_tokens = 768) {
    json staged = base_body;
    staged["parallel_tool_calls"] = false;
    staged["tools"] = tools;
    staged.erase("tool_choice");
    staged["messages"] = messages;
    staged["stream"] = false;
    staged["max_tokens"] = stage_max_tokens;
    staged["x-vicuna-provider-max-tokens-override"] = stage_max_tokens;
    staged.erase("response_format");
    staged["x-vicuna-stage-followup-guidance"] = stage_followup_guidance;
    return staged;
}

static bool staged_body_is_bridge_scoped(const json & body) {
    return json_value(body, "x-vicuna-bridge-scoped", false);
}

struct staged_utf8_parse_result {
    enum status_t { success, incomplete, invalid } status = invalid;
    size_t bytes_consumed = 0;
};

static staged_utf8_parse_result staged_parse_utf8_codepoint(
        std::string_view input,
        size_t offset) {
    staged_utf8_parse_result result = {};
    if (offset >= input.size()) {
        result.status = staged_utf8_parse_result::incomplete;
        return result;
    }

    const unsigned char first = static_cast<unsigned char>(input[offset]);
    if ((first & 0x80u) == 0) {
        result.status = staged_utf8_parse_result::success;
        result.bytes_consumed = 1;
        return result;
    }
    if ((first & 0x40u) == 0) {
        result.status = staged_utf8_parse_result::invalid;
        return result;
    }

    size_t expected = 0;
    if ((first & 0x20u) == 0) {
        expected = 2;
    } else if ((first & 0x10u) == 0) {
        expected = 3;
    } else if ((first & 0x08u) == 0) {
        expected = 4;
    } else {
        result.status = staged_utf8_parse_result::invalid;
        return result;
    }

    if (offset + expected > input.size()) {
        result.status = staged_utf8_parse_result::incomplete;
        return result;
    }
    for (size_t i = 1; i < expected; ++i) {
        const unsigned char continuation = static_cast<unsigned char>(input[offset + i]);
        if ((continuation & 0xc0u) != 0x80u) {
            result.status = staged_utf8_parse_result::invalid;
            return result;
        }
    }

    result.status = staged_utf8_parse_result::success;
    result.bytes_consumed = expected;
    return result;
}

static std::string make_compact_utf8_preview(
        const std::string & raw_text,
        size_t max_bytes) {
    const std::string trimmed = trim_copy(raw_text);
    std::string compact;
    compact.reserve(std::min(max_bytes, trimmed.size()));

    bool pending_space = false;
    size_t offset = 0;
    while (offset < trimmed.size()) {
        const auto parsed = staged_parse_utf8_codepoint(trimmed, offset);
        if (parsed.status == staged_utf8_parse_result::incomplete) {
            break;
        }
        if (parsed.status == staged_utf8_parse_result::invalid) {
            ++offset;
            continue;
        }

        const std::string_view encoded(trimmed.data() + offset, parsed.bytes_consumed);
        offset += parsed.bytes_consumed;

        const bool is_space = encoded.size() == 1 && std::isspace(static_cast<unsigned char>(encoded.front())) != 0;
        if (is_space) {
            pending_space = !compact.empty();
            continue;
        }

        if (pending_space) {
            if (compact.size() + 1 > max_bytes) {
                break;
            }
            compact.push_back(' ');
            pending_space = false;
        }
        if (compact.size() + encoded.size() > max_bytes) {
            break;
        }
        compact.append(encoded.data(), encoded.size());
    }

    return trim_copy(compact);
}

static std::string staged_plain_text_retry_error(
        const json & base_body,
        const std::string & stage_label,
        const std::string & raw_content,
        bool telegram_available) {
    std::string message = "staged " + stage_label + " returned plain assistant text instead of a selector tool call";
    if (staged_body_is_bridge_scoped(base_body)) {
        message += "; bridge-scoped staged turns cannot communicate with the user via plain text";
        if (telegram_available) {
            message += "; Telegram is the only valid user-communication path";
        }
    }
    const std::string preview = make_compact_utf8_preview(raw_content, 160);
    if (!preview.empty()) {
        message += ": ";
        message += preview;
    }
    return message;
}

static bool parse_staged_family_selection_response(
        const json & base_body,
        const deepseek_chat_result & result,
        const staged_tool_catalog & catalog,
        std::string * out_family_name,
        std::string * out_error) {
    if (!result.tool_calls.empty()) {
        const deepseek_tool_call & tool_call = result.tool_calls.front();
        if (tool_call.name != STAGED_SELECTOR_TOOL_FAMILY) {
            if (out_error) {
                *out_error = "staged family selection returned an unexpected tool call";
            }
            return false;
        }
        try {
            const json payload = json::parse(tool_call.arguments_json);
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
                *out_error = server_string_format("staged family selection tool arguments were not valid JSON: %s", e.what());
            }
            return false;
        }
    }
    if (!trim_copy(result.content).empty()) {
        if (out_error) {
            *out_error = staged_plain_text_retry_error(
                    base_body,
                    "family selection",
                    result.content,
                    staged_tool_find_family(catalog, "Telegram") != nullptr);
        }
        return false;
    }
    try {
        const json payload = json::parse(extract_json_object_payload(result.content));
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
        const json & base_body,
        const deepseek_chat_result & result,
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
        json payload;
        if (!result.tool_calls.empty()) {
            const deepseek_tool_call & tool_call = result.tool_calls.front();
            if (tool_call.name != STAGED_SELECTOR_TOOL_METHOD) {
                if (out_error) {
                    *out_error = "staged method selection returned an unexpected tool call";
                }
                return false;
            }
            payload = json::parse(tool_call.arguments_json);
        } else {
            if (!trim_copy(result.content).empty()) {
                if (out_error) {
                    *out_error = staged_plain_text_retry_error(
                            base_body,
                            "method selection",
                            result.content,
                            family.family_name == "Telegram");
                }
                return false;
            }
            payload = json::parse(extract_json_object_payload(result.content));
        }
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
        const json & base_body,
        const deepseek_chat_result & result,
        const staged_tool_family & family,
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
        if (!result.tool_calls.empty()) {
            const deepseek_tool_call & tool_call = result.tool_calls.front();
            if (tool_call.name == STAGED_SELECTOR_TOOL_GO_BACK) {
                out_choice->kind = STAGED_TOOL_PAYLOAD_GO_BACK;
                out_choice->payload = json::object();
                return true;
            }
            if (tool_call.name != STAGED_SELECTOR_TOOL_SUBMIT_PAYLOAD) {
                if (out_error) {
                    *out_error = "staged payload selection returned an unexpected tool call";
                }
                return false;
            }
            json payload = json::parse(tool_call.arguments_json);
            strip_staged_optional_nulls(&payload, method.parameters);
            std::string validation_error;
            if (!validate_payload_against_schema(payload, method.parameters, &validation_error)) {
                if (out_error) {
                    *out_error = validation_error;
                }
                return false;
            }
            out_choice->kind = STAGED_TOOL_PAYLOAD_SUBMIT;
            out_choice->payload = std::move(payload);
            return true;
        }
        if (!trim_copy(result.content).empty()) {
                if (out_error) {
                    *out_error = staged_plain_text_retry_error(
                            base_body,
                            "payload construction",
                            result.content,
                            family.family_name == "Telegram");
                }
                return false;
            }
        const json payload = json::parse(extract_json_object_payload(result.content));
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
        const staged_tool_prompt_spec & stage_prompt,
        const json & stage_tools,
        const std::string & validation_error,
        bool stage_followup_guidance,
        int64_t stage_max_tokens,
        deepseek_chat_result * out_result,
        json * out_error,
        bool suppress_replay_admission = false,
        const std::string & mode_label = std::string(),
        const runtime_request_trace_context * trace_context = nullptr) {
    const json messages = build_staged_messages(base_body.at("messages"), base_body, stage_prompt, validation_error);
    const json staged_body = build_staged_provider_body(base_body, messages, stage_tools, stage_followup_guidance, stage_max_tokens);
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
            0.0f,
            mode_label,
            trace_context);
}

static bool execute_final_completion_after_staged_loop(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const json & body,
        deepseek_chat_result * out_result,
        json * out_error,
        bool suppress_replay_admission = false,
        const std::string & mode_label = std::string(),
        const runtime_request_trace_context * trace_context = nullptr) {
    json completion_body = body;
    completion_body.erase("tools");
    completion_body.erase("tool_choice");
    completion_body.erase("parallel_tool_calls");
    completion_body["x-vicuna-stage-followup-guidance"] = true;
    completion_body["messages"] = json::array({
        {
            {"role", "system"},
            {"content", build_staged_tool_core_system_prompt()},
        },
        {
            {"role", "system"},
            {"content", "The staged tool loop is complete. Reply directly to the user or conclude the task without JSON."},
        },
    });
    if (body.at("messages").is_array()) {
        for (const auto & item : body.at("messages")) {
            completion_body["messages"].push_back(item);
        }
    }
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
            0.0f,
            mode_label,
            trace_context);
}

static bool should_use_staged_tool_loop_for_request(const json & body) {
    const char * staged_fallback_env = std::getenv("VICUNA_ENABLE_STAGED_TOOL_FALLBACK");
    if (!staged_fallback_env || !parse_truthy_header(staged_fallback_env)) {
        return false;
    }
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
        const std::string & mode_label = std::string(),
        const runtime_request_trace_context * trace_context = nullptr) {
    if (!out_result) {
        if (out_error) {
            *out_error = format_error_response("staged tool loop requires a result output", ERROR_TYPE_SERVER);
        }
        return false;
    }
    *out_result = deepseek_chat_result();

    const auto prompt_bundle = get_cached_staged_tool_prompt_bundle(body.at("tools"));
    if (!prompt_bundle || prompt_bundle->catalog.families.empty()) {
        if (trace_context) {
            runtime_request_trace_log(*trace_context, "staged_tool_loop", "staged_prompt_bundle_missing", {
                {"tool_count", body.contains("tools") && body.at("tools").is_array() ?
                        static_cast<int64_t>(body.at("tools").size()) : 0},
                {"message", "strict staged tool mode requires describable family, method, and parameter metadata for every tool-bearing request"},
            });
        }
        if (out_error) {
            *out_error = format_error_response(
                    "strict staged tool mode requires describable family, method, and parameter metadata for every tool-bearing request",
                    ERROR_TYPE_INVALID_REQUEST);
        }
        return false;
    }
    const staged_tool_catalog & catalog = prompt_bundle->catalog;

    const int max_stage_turns = 24;
    const int max_stage_selector_attempts = 2;
    int stage_turn = 0;
    std::string selected_family_name;
    while (stage_turn < max_stage_turns) {
        bool family_valid = false;
        std::string family_validation_error;
        for (int family_attempt = 0; family_attempt < max_stage_selector_attempts && stage_turn < max_stage_turns; ++family_attempt) {
            ++stage_turn;
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "staged_tool_loop", "family_selection_attempt_started", {
                    {"stage_turn", stage_turn},
                    {"attempt", family_attempt + 1},
                    {"family_count", static_cast<int64_t>(catalog.families.size())},
                    {"previous_error", family_validation_error.empty() ? json(nullptr) : json(family_validation_error)},
                });
            }
            deepseek_chat_result family_result;
                if (!execute_staged_selection_turn(
                        config,
                        emotive_runtime,
                        body,
                        prompt_bundle->family_selection_prompt,
                        prompt_bundle->family_selection_tools,
                        family_validation_error,
                        stage_turn > 1 || family_attempt > 0,
                        768,
                        &family_result,
                        out_error,
                        suppress_replay_admission,
                        mode_label.empty() ? "staged_family_select" : mode_label + "_family_select",
                        trace_context)) {
                return false;
            }

            std::string parse_error;
            if (parse_staged_family_selection_response(body, family_result, catalog, &selected_family_name, &parse_error)) {
                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "staged_tool_loop", "family_selection_attempt_succeeded", {
                        {"stage_turn", stage_turn},
                        {"attempt", family_attempt + 1},
                        {"selected_family", selected_family_name},
                    });
                }
                family_valid = true;
                break;
            }
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "staged_tool_loop", "family_selection_attempt_failed", {
                    {"stage_turn", stage_turn},
                    {"attempt", family_attempt + 1},
                    {"error", parse_error},
                });
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
            const bool allow_completion = !staged_body_is_bridge_scoped(body);
            const staged_tool_prompt_spec method_prompt = allow_completion ?
                    prompt_bundle->method_selection_prompts.at(family->family_name) :
                    build_staged_method_selection_prompt(*family, false);
            const json method_tools = allow_completion ?
                    prompt_bundle->method_selection_tools.at(family->family_name) :
                    build_staged_method_selection_tools(*family, false);
            for (int method_attempt = 0; method_attempt < max_stage_selector_attempts && stage_turn < max_stage_turns; ++method_attempt) {
                ++stage_turn;
                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "staged_tool_loop", "method_selection_attempt_started", {
                        {"stage_turn", stage_turn},
                        {"attempt", method_attempt + 1},
                        {"selected_family", family->family_name},
                        {"previous_error", method_validation_error.empty() ? json(nullptr) : json(method_validation_error)},
                    });
                }
                deepseek_chat_result method_result;
                if (!execute_staged_selection_turn(
                            config,
                            emotive_runtime,
                            body,
                            method_prompt,
                            method_tools,
                            method_validation_error,
                            true,
                            768,
                            &method_result,
                            out_error,
                            suppress_replay_admission,
                            mode_label.empty() ? "staged_method_select" : mode_label + "_method_select",
                            trace_context)) {
                    return false;
                }

                std::string method_error;
                if (parse_staged_method_selection_response(body, method_result, *family, allow_completion, &method_choice, &method_error)) {
                    if (trace_context) {
                        runtime_request_trace_log(*trace_context, "staged_tool_loop", "method_selection_attempt_succeeded", {
                            {"stage_turn", stage_turn},
                            {"attempt", method_attempt + 1},
                            {"selected_family", family->family_name},
                            {"selection", method_choice.kind == STAGED_TOOL_METHOD_CHOOSE_METHOD ? json(method_choice.method_name) :
                                    (method_choice.kind == STAGED_TOOL_METHOD_GO_BACK ? json("back") : json("complete"))},
                        });
                    }
                    method_valid = true;
                    break;
                }
                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "staged_tool_loop", "method_selection_attempt_failed", {
                        {"stage_turn", stage_turn},
                        {"attempt", method_attempt + 1},
                        {"selected_family", family->family_name},
                        {"error", method_error},
                    });
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
                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "staged_tool_loop", "method_selection_back", {
                        {"stage_turn", stage_turn},
                        {"selected_family", family->family_name},
                    });
                }
                break;
            }
            if (method_choice.kind == STAGED_TOOL_METHOD_COMPLETE) {
                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "staged_tool_loop", "method_selection_complete", {
                        {"stage_turn", stage_turn},
                        {"selected_family", family->family_name},
                    });
                }
                return execute_final_completion_after_staged_loop(
                        config,
                        emotive_runtime,
                        body,
                        out_result,
                        out_error,
                        suppress_replay_admission,
                        mode_label.empty() ? "staged_complete" : mode_label + "_complete",
                        trace_context);
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
                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "staged_tool_loop", "payload_attempt_started", {
                        {"stage_turn", stage_turn},
                        {"selected_family", family->family_name},
                        {"selected_method", method->method_name},
                        {"previous_error", payload_validation_error.empty() ? json(nullptr) : json(payload_validation_error)},
                    });
                }
                deepseek_chat_result payload_result;
                const int64_t payload_stage_max_tokens = family->family_name == "Telegram" ? 1024 : 768;
                if (!execute_staged_selection_turn(
                            config,
                            emotive_runtime,
                            body,
                            payload_validation_error.empty() ?
                                    prompt_bundle->payload_prompts.at(family->family_name).at(method->method_name) :
                                    build_staged_payload_prompt(*family, *method, build_staged_payload_provider_schema(method->parameters), payload_validation_error),
                            prompt_bundle->payload_tools.at(family->family_name).at(method->method_name),
                            payload_validation_error,
                            true,
                            payload_stage_max_tokens,
                            &payload_result,
                            out_error,
                            suppress_replay_admission,
                            mode_label.empty() ? "staged_payload_build" : mode_label + "_payload_build",
                            trace_context)) {
                    return false;
                }

                staged_tool_payload_choice payload_choice = {};
                std::string payload_error;
                if (!parse_staged_payload_response(body, payload_result, *family, *method, &payload_choice, &payload_error)) {
                    if (trace_context) {
                        runtime_request_trace_log(*trace_context, "staged_tool_loop", "payload_attempt_failed", {
                            {"stage_turn", stage_turn},
                            {"selected_family", family->family_name},
                            {"selected_method", method->method_name},
                            {"error", payload_error},
                        });
                    }
                    payload_validation_error = payload_error;
                    continue;
                }
                if (payload_choice.kind == STAGED_TOOL_PAYLOAD_GO_BACK) {
                    if (trace_context) {
                        runtime_request_trace_log(*trace_context, "staged_tool_loop", "payload_back", {
                            {"stage_turn", stage_turn},
                            {"selected_family", family->family_name},
                            {"selected_method", method->method_name},
                        });
                    }
                    break;
                }

                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "staged_tool_loop", "payload_submitted", {
                        {"stage_turn", stage_turn},
                        {"selected_family", family->family_name},
                        {"selected_method", method->method_name},
                        {"tool_name", method->tool_name},
                        {"payload", payload_choice.payload},
                    });
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
    if (frequency.unit == "minutes") {
        return interval * 60LL * 1000LL;
    }
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
    if (unit != "minutes" && unit != "hours" && unit != "days" && unit != "weeks") {
        if (out_error) {
            *out_error = "ongoing-task frequency unit must be one of minutes, hours, days, or weeks";
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
    if (body.contains("emotive_animation")) {
        request.emotive_animation = body.at("emotive_animation");
    }
    request.reply_to_message_id = std::max<int64_t>(0, json_value(body, "reply_to_message_id", int64_t(0)));
    request.intent = trim_copy(json_value(body, "intent", std::string()));
    request.dedupe_key = trim_copy(json_value(body, "dedupe_key", std::string()));
    request.urgency = std::max(0.0, json_value(body, "urgency", 0.0));
    return request;
}

static bool is_server_owned_telegram_delivery_tool_name(const std::string & tool_name) {
    return tool_name == "telegram_send_plain_text" ||
            tool_name == "telegram_send_formatted_text" ||
            tool_name == "telegram_send_photo" ||
            tool_name == "telegram_send_document" ||
            tool_name == "telegram_send_poll" ||
            tool_name == "telegram_send_dice";
}

static bool parse_server_owned_telegram_delivery_tool_call(
        const deepseek_tool_call & tool_call,
        const telegram_bridge_request_context & context,
        telegram_outbox_item * out_item,
        std::string * out_error) {
    if (!out_item) {
        if (out_error) {
            *out_error = "telegram delivery parsing requires an output item";
        }
        return false;
    }

    json arguments;
    try {
        arguments = json::parse(tool_call.arguments_json.empty() ? std::string("{}") : tool_call.arguments_json);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = server_string_format("%s arguments were not valid JSON: %s", tool_call.name.c_str(), e.what());
        }
        return false;
    }
    if (!arguments.is_object()) {
        if (out_error) {
            *out_error = "Telegram delivery arguments must be a JSON object";
        }
        return false;
    }

    telegram_outbox_item item = {};
    item.chat_scope = context.chat_scope;
    item.reply_to_message_id = std::max<int64_t>(0, context.reply_to_message_id);
    item.intent.clear();
    item.dedupe_key.clear();
    item.urgency = 0.0;

    if (item.chat_scope.empty()) {
        if (out_error) {
            *out_error = "Telegram delivery requires chat_scope when no bridge-scoped default is available";
        }
        return false;
    }

    if (tool_call.name == "telegram_send_plain_text") {
        item.text = trim_copy(json_value(arguments, "text", std::string()));
        item.telegram_method = "sendMessage";
        item.telegram_payload = json{
            {"text", item.text},
        };
    } else if (tool_call.name == "telegram_send_formatted_text") {
        item.telegram_method = "sendMessage";
        item.telegram_payload = json{
            {"text", trim_copy(json_value(arguments, "text", std::string()))},
        };
        item.telegram_payload["parse_mode"] = arguments.contains("parse_mode") ? arguments.at("parse_mode") : json("HTML");
        if (arguments.contains("disable_web_page_preview")) {
            item.telegram_payload["disable_web_page_preview"] = arguments.at("disable_web_page_preview");
        }
        if (arguments.contains("reply_markup")) {
            item.telegram_payload["reply_markup"] = arguments.at("reply_markup");
        }
    } else if (tool_call.name == "telegram_send_photo") {
        item.telegram_method = "sendPhoto";
        item.telegram_payload = json{
            {"photo", trim_copy(json_value(arguments, "photo", std::string()))},
        };
        if (arguments.contains("caption")) {
            item.telegram_payload["caption"] = arguments.at("caption");
        }
        if (arguments.contains("parse_mode")) {
            item.telegram_payload["parse_mode"] = arguments.at("parse_mode");
        }
        if (arguments.contains("reply_markup")) {
            item.telegram_payload["reply_markup"] = arguments.at("reply_markup");
        }
    } else if (tool_call.name == "telegram_send_document") {
        item.telegram_method = "sendDocument";
        item.telegram_payload = json{
            {"document", trim_copy(json_value(arguments, "document", std::string()))},
        };
        if (arguments.contains("caption")) {
            item.telegram_payload["caption"] = arguments.at("caption");
        }
        if (arguments.contains("parse_mode")) {
            item.telegram_payload["parse_mode"] = arguments.at("parse_mode");
        }
        if (arguments.contains("reply_markup")) {
            item.telegram_payload["reply_markup"] = arguments.at("reply_markup");
        }
    } else if (tool_call.name == "telegram_send_poll") {
        item.telegram_method = "sendPoll";
        item.telegram_payload = json{
            {"question", trim_copy(json_value(arguments, "question", std::string()))},
            {"options", arguments.contains("options") ? arguments.at("options") : json::array()},
        };
        if (arguments.contains("is_anonymous")) {
            item.telegram_payload["is_anonymous"] = arguments.at("is_anonymous");
        }
        if (arguments.contains("allows_multiple_answers")) {
            item.telegram_payload["allows_multiple_answers"] = arguments.at("allows_multiple_answers");
        }
    } else if (tool_call.name == "telegram_send_dice") {
        item.telegram_method = "sendDice";
        item.telegram_payload = json::object();
        if (arguments.contains("emoji")) {
            item.telegram_payload["emoji"] = arguments.at("emoji");
        }
    } else {
        if (out_error) {
            *out_error = "unknown Telegram delivery tool name";
        }
        return false;
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
            *out_error = "Telegram delivery did not produce relayable summary text";
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
    if (!item.emotive_animation.is_null()) {
        payload["emotive_animation"] = item.emotive_animation;
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

struct server_rich_plan_response {
    bool valid = false;
    std::string format = "markdown";
    std::string title;
    std::string body;
    bool disable_web_page_preview = false;
    json reply_markup = nullptr;
    std::string delivery_hint = "reply";
    std::vector<std::string> stripped_fields;
};

static std::string trim_quotes_copy(const std::string & value) {
    std::string trimmed = trim_copy(value);
    if (trimmed.size() >= 2 &&
            ((trimmed.front() == '"' && trimmed.back() == '"') ||
             (trimmed.front() == '\'' && trimmed.back() == '\''))) {
        return trimmed.substr(1, trimmed.size() - 2);
    }
    return trimmed;
}

static json parse_rich_plan_reply_markup_block(const std::vector<std::string> & lines) {
    json rows = json::array();
    json current_row = json::array();
    json current_button = json::object();

    const auto flush_button = [&]() {
        if (!current_button.is_object() || current_button.empty()) {
            return;
        }
        if (!current_button.contains("text") || trim_copy(json_value(current_button, "text", std::string())).empty()) {
            current_button = json::object();
            return;
        }
        current_row.push_back(current_button);
        current_button = json::object();
    };
    const auto flush_row = [&]() {
        flush_button();
        if (!current_row.empty()) {
            rows.push_back(current_row);
            current_row = json::array();
        }
    };

    for (const std::string & raw_line : lines) {
        const std::string line = trim_copy(raw_line);
        if (line.empty() || line == "reply_markup:" || line == "inline_keyboard:") {
            continue;
        }
        if (line.rfind("- - ", 0) == 0) {
            flush_row();
            const std::string remainder = trim_copy(line.substr(4));
            const size_t separator = remainder.find(':');
            if (separator != std::string::npos) {
                current_button[trim_copy(remainder.substr(0, separator))] =
                        trim_quotes_copy(remainder.substr(separator + 1));
            }
            continue;
        }
        if (line.rfind("- ", 0) == 0) {
            flush_button();
            const std::string remainder = trim_copy(line.substr(2));
            const size_t separator = remainder.find(':');
            if (separator != std::string::npos) {
                current_button[trim_copy(remainder.substr(0, separator))] =
                        trim_quotes_copy(remainder.substr(separator + 1));
            }
            continue;
        }
        const size_t separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        current_button[trim_copy(line.substr(0, separator))] =
                trim_quotes_copy(line.substr(separator + 1));
    }
    flush_row();
    if (rows.empty()) {
        return nullptr;
    }
    return json{{"inline_keyboard", rows}};
}

static server_rich_plan_response parse_rich_plan_response(const std::string & raw_text) {
    server_rich_plan_response response = {};
    const std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return response;
    }

    response.valid = true;
    response.body = text;
    if (text.rfind("---\n", 0) != 0) {
        return response;
    }

    const size_t closing = text.find("\n---\n", 4);
    if (closing == std::string::npos) {
        return response;
    }

    std::istringstream meta(text.substr(4, closing - 4));
    std::string line;
    std::vector<std::string> reply_markup_lines;
    bool collecting_reply_markup = false;
    while (std::getline(meta, line)) {
        const std::string trimmed = trim_copy(line);
        if (trimmed.empty()) {
            continue;
        }
        if (collecting_reply_markup && (line.rfind(" ", 0) == 0 || line.rfind("\t", 0) == 0 || trimmed.rfind("- ", 0) == 0)) {
            reply_markup_lines.push_back(line);
            continue;
        }
        collecting_reply_markup = false;
        const size_t separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        const std::string key = trim_copy(line.substr(0, separator));
        const std::string value = trim_copy(line.substr(separator + 1));
        if (key == "format") {
            const std::string parsed = trim_quotes_copy(value);
            if (parsed == "plain_text" || parsed == "markdown" || parsed == "html") {
                response.format = parsed;
            } else {
                response.stripped_fields.push_back("format");
            }
        } else if (key == "title") {
            response.title = trim_quotes_copy(value);
        } else if (key == "disable_web_page_preview") {
            response.disable_web_page_preview = parse_truthy_header(value);
        } else if (key == "delivery_hint") {
            const std::string parsed = trim_quotes_copy(value);
            if (parsed == "reply" || parsed == "replace_prompt" || parsed == "silent") {
                response.delivery_hint = parsed;
            } else {
                response.stripped_fields.push_back("delivery_hint");
            }
        } else if (key == "reply_markup") {
            if (!value.empty()) {
                try {
                    response.reply_markup = json::parse(value);
                } catch (...) {
                    response.stripped_fields.push_back("reply_markup");
                }
            } else {
                collecting_reply_markup = true;
                reply_markup_lines.push_back("reply_markup:");
            }
        }
    }
    if (response.reply_markup.is_null() && !reply_markup_lines.empty()) {
        response.reply_markup = parse_rich_plan_reply_markup_block(reply_markup_lines);
        if (response.reply_markup.is_null()) {
            response.stripped_fields.push_back("reply_markup");
        }
    }

    response.body = trim_copy(text.substr(closing + 5));
    if (response.body.empty()) {
        response.valid = false;
    }
    return response;
}

static json rich_plan_response_to_json(const server_rich_plan_response & response) {
    if (!response.valid) {
        return nullptr;
    }
    json payload = {
        {"format", response.format},
        {"title", response.title.empty() ? json(nullptr) : json(response.title)},
        {"body", response.body},
        {"disable_web_page_preview", response.disable_web_page_preview},
        {"delivery_hint", response.delivery_hint},
        {"stripped_fields", response.stripped_fields},
    };
    if (!response.reply_markup.is_null()) {
        payload["reply_markup"] = response.reply_markup;
    }
    return payload;
}

static bool execute_telegram_delivery_for_bridge_request(
        const server_http_req & req,
        telegram_outbox_state * outbox,
        deepseek_chat_result * result,
        telegram_delivery_result * out_delivery,
        json * out_error,
        const runtime_request_trace_context * trace_context = nullptr) {
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
    if (result->tool_calls.size() == 1 && is_server_owned_telegram_delivery_tool_name(result->tool_calls.front().name)) {
        std::string parse_error;
        if (!parse_server_owned_telegram_delivery_tool_call(result->tool_calls.front(), context, &item, &parse_error)) {
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "telegram_delivery", "telegram_delivery_parse_failed", {
                    {"message", parse_error},
                    {"tool_name", result->tool_calls.front().name},
                });
            }
            if (out_error) {
                *out_error = format_error_response(parse_error, ERROR_TYPE_SERVER);
            }
            return false;
        }
        delivery_source = "tool_call";
    } else {
        if (!result->tool_calls.empty()) {
            return true;
        }
        const server_rich_plan_response rich_response = parse_rich_plan_response(result->content);
        if (!rich_response.valid) {
            return true;
        }
        result->rich_response = rich_plan_response_to_json(rich_response);
        item.chat_scope = context.chat_scope;
        item.reply_to_message_id = context.reply_to_message_id;
        item.text = rich_response.body;
        item.telegram_method = "sendMessage";
        item.telegram_payload = json{{"text", rich_response.body}};
        if (rich_response.format == "html") {
            item.telegram_payload["parse_mode"] = "HTML";
        } else if (rich_response.format == "markdown") {
            item.telegram_payload["parse_mode"] = "MarkdownV2";
        }
        if (rich_response.disable_web_page_preview) {
            item.telegram_payload["disable_web_page_preview"] = true;
        }
        if (!rich_response.reply_markup.is_null()) {
            item.telegram_payload["reply_markup"] = rich_response.reply_markup;
        }
        item.dedupe_key = build_telegram_bridge_dedupe_key(context, item.telegram_method);
        delivery_source = "rich_plan";
    }

    if (delivery_source == "rich_plan") {
        try {
            validate_telegram_payload_fields(item.telegram_method, item.telegram_payload);
        } catch (const std::exception & e) {
            if (item.telegram_payload.contains("reply_markup")) {
                item.telegram_payload.erase("reply_markup");
                item.text += "\n\n[reply markup omitted: invalid metadata]";
            } else {
                item.text += "\n\n[delivery metadata omitted]";
            }
            item.telegram_payload["text"] = item.text;
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "telegram_delivery", "rich_plan_metadata_stripped", {
                    {"message", e.what()},
                });
            }
        }
    }

    item.emotive_animation = build_telegram_emotive_animation_bundle(result->emotive_trace);

    telegram_outbox_enqueue_result enqueue = {};
    try {
        enqueue = telegram_outbox_enqueue(outbox, item);
    } catch (const std::exception & e) {
        if (trace_context) {
            runtime_request_trace_log(*trace_context, "telegram_delivery", "telegram_outbox_enqueue_failed", {
                {"telegram_method", item.telegram_method},
                {"message", e.what()},
            });
        }
        if (out_error) {
            *out_error = format_error_response(e.what(), ERROR_TYPE_SERVER);
        }
        return false;
    }
    if (trace_context) {
        runtime_request_trace_log(*trace_context, "telegram_delivery", "telegram_outbox_enqueued", {
            {"telegram_method", item.telegram_method},
            {"sequence_number", enqueue.sequence_number},
            {"queued", enqueue.queued},
            {"deduplicated", enqueue.deduplicated},
            {"source", delivery_source},
        });
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

static size_t find_leading_system_message_boundary(const json & messages) {
    if (!messages.is_array()) {
        return 0;
    }
    size_t index = 0;
    while (index < messages.size()) {
        const auto & item = messages.at(index);
        if (!item.is_object() || json_value(item, "role", std::string()) != "system") {
            break;
        }
        ++index;
    }
    return index;
}

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

static std::string build_metacognitive_control_guidance(const server_metacognitive_policy_decision & policy) {
    if (!policy.valid) {
        return std::string();
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "Metacognitive control policy: mode=" << policy.selected_mode
        << "; reasoning_depth=" << policy.reasoning_depth
        << "; tool_aggression=" << policy.tool_aggression
        << "; tool_parallelism_cap=" << policy.tool_parallelism_cap
        << "; interrupt_allowed=" << (policy.interrupt_allowed ? "true" : "false")
        << "; replan_required=" << (policy.replan_required ? "true" : "false")
        << "; early_stop_ok=" << (policy.early_stop_ok ? "true" : "false")
        << "; force_synthesis=" << (policy.force_synthesis ? "true" : "false")
        << ".";
    if (!policy.prompt_hints.empty()) {
        out << " Guidance: ";
        for (size_t i = 0; i < policy.prompt_hints.size(); ++i) {
            if (i > 0) {
                out << " ";
            }
            out << policy.prompt_hints[i];
        }
    }
    return out.str();
}

static int64_t metacognitive_token_budget_for_depth(const std::string & reasoning_depth) {
    if (reasoning_depth == "none") {
        return 256;
    }
    if (reasoning_depth == "short") {
        return 512;
    }
    if (reasoning_depth == "medium") {
        return 1024;
    }
    return 2048;
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
        const server_metacognitive_policy_decision * policy_decision,
        const server_heuristic_retrieval_decision * heuristic_decision,
        const std::string * heuristic_guidance_override,
        bool enable_heuristic_guidance,
        const runtime_request_trace_context * trace_context = nullptr) {
    if ((!builder || !builder->has_current_state()) && !emotive_runtime) {
        return body;
    }
    if (!body.contains("messages") || !body.at("messages").is_array()) {
        return body;
    }

    json messages = body.at("messages");
    const bool stage_followup_guidance =
            body.contains("x-vicuna-stage-followup-guidance") &&
            body.at("x-vicuna-stage-followup-guidance").is_boolean() &&
            body.at("x-vicuna-stage-followup-guidance").get<bool>();
    const active_tool_continuation_span span = find_active_tool_continuation_span(messages);
    const size_t insertion_index = span.valid ?
            span.last_tool_index + 1 :
            (stage_followup_guidance ? find_leading_system_message_boundary(messages) : messages.size());

    std::string vad_guidance;
    std::string vad_skip_reason;
    if (builder && builder->has_current_state() && (span.valid || stage_followup_guidance)) {
        if (span.valid) {
            const auto & assistant = messages.at(span.assistant_index);
            const std::string reasoning = json_value(assistant, "reasoning_content", std::string());
            if (!reasoning.empty()) {
                vad_guidance = format_interleaved_vad_guidance(builder->current_vad());
            } else {
                vad_skip_reason = "assistant_reasoning_missing";
            }
        } else {
            vad_guidance = format_interleaved_vad_guidance(builder->current_vad());
        }
    } else if (!builder || !builder->has_current_state()) {
        vad_skip_reason = "builder_state_missing";
    } else if (stage_followup_guidance) {
        vad_skip_reason = "stage_followup_disabled_by_builder";
    } else if (!span.valid) {
        vad_skip_reason = "no_active_tool_continuation_span";
    }

    std::future<std::string> heuristic_guidance_future;
    bool launched_heuristic_search = false;
    if (heuristic_guidance_override && !heuristic_guidance_override->empty()) {
        launched_heuristic_search = false;
    } else if (enable_heuristic_guidance && emotive_runtime) {
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

    std::string heuristic_guidance = heuristic_guidance_override ? *heuristic_guidance_override : std::string();
    if (launched_heuristic_search) {
        heuristic_guidance = heuristic_guidance_future.get();
    }
    const std::string policy_guidance = policy_decision ? build_metacognitive_control_guidance(*policy_decision) : std::string();
    if (trace_context) {
        json data = {
            {"insertion_index", static_cast<int64_t>(insertion_index)},
            {"span_valid", span.valid},
            {"stage_followup_guidance", stage_followup_guidance},
            {"builder_has_state", builder && builder->has_current_state()},
            {"heuristic_search_launched", launched_heuristic_search},
            {"heuristic_decision", heuristic_decision ? json{
                {"matched", heuristic_decision->matched},
                {"record_id", heuristic_decision->record_id},
                {"heuristic_id", heuristic_decision->heuristic_id},
                {"semantic_score", heuristic_decision->semantic_score},
                {"struct_score", heuristic_decision->struct_score},
                {"emotive_score", heuristic_decision->emotive_score},
                {"total_score", heuristic_decision->total_score},
                {"threshold", heuristic_decision->threshold},
                {"control_biases", heuristic_decision->control_biases},
            } : json(nullptr)},
            {"policy", policy_decision ? json{
                {"policy_version", policy_decision->policy_version},
                {"selected_mode", policy_decision->selected_mode},
                {"reasoning_depth", policy_decision->reasoning_depth},
                {"tool_aggression", policy_decision->tool_aggression},
                {"tool_parallelism_cap", policy_decision->tool_parallelism_cap},
                {"interrupt_allowed", policy_decision->interrupt_allowed},
                {"replan_required", policy_decision->replan_required},
                {"early_stop_ok", policy_decision->early_stop_ok},
                {"force_synthesis", policy_decision->force_synthesis},
                {"heuristic_biases", policy_decision->heuristic_biases},
            } : json(nullptr)},
            {"policy_guidance", policy_guidance.empty() ? json(nullptr) : json(policy_guidance)},
            {"vad_guidance", vad_guidance.empty() ? json(nullptr) : json(vad_guidance)},
            {"heuristic_guidance", heuristic_guidance.empty() ? json(nullptr) : json(heuristic_guidance)},
            {"policy_injected", !policy_guidance.empty()},
            {"vad_injected", !vad_guidance.empty()},
            {"heuristic_injected", !heuristic_guidance.empty()},
            {"vad_skip_reason", vad_skip_reason.empty() ? json(nullptr) : json(vad_skip_reason)},
        };
        if (builder && builder->has_current_state()) {
            const server_emotive_vad current_vad = builder->current_vad();
            data["current_vad"] = {
                {"valence", current_vad.valence},
                {"arousal", current_vad.arousal},
                {"dominance", current_vad.dominance},
                {"tone_label", current_vad.style_guide.tone_label},
            };
        }
        runtime_request_trace_log(*trace_context, "runtime_guidance", "guidance_evaluated", std::move(data));
    }
    if (heuristic_guidance.empty() && vad_guidance.empty() && policy_guidance.empty()) {
        return body;
    }

    auto insert_it = messages.begin() + static_cast<json::difference_type>(insertion_index);
    if (!policy_guidance.empty()) {
        insert_it = messages.insert(insert_it, json{
            {"role", "system"},
            {"content", policy_guidance},
        });
        ++insert_it;
    }
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
        float ongoing_task_due,
        const std::string & mode_label,
        const runtime_request_trace_context * trace_context) {
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
        turn_builder->mark_live_generation_start();
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

    server_heuristic_retrieval_decision heuristic_decision = {};
    std::string heuristic_guidance;
    if (enable_heuristic_guidance && !cognitive_replay && body.contains("messages") && body.at("messages").is_array()) {
        std::vector<std::string> struct_tags;
        const std::string query_text = build_live_heuristic_query_text(body.at("messages"), &struct_tags);
        if (!query_text.empty()) {
            const bool have_state = turn_builder && turn_builder->has_current_state();
            const server_emotive_vector heuristic_moment = have_state ? turn_builder->current_moment() : server_emotive_vector();
            const server_emotive_vad heuristic_vad = have_state ? turn_builder->current_vad() : server_emotive_vad();
            heuristic_decision = emotive_runtime.retrieve_matching_heuristic(
                    query_text,
                    struct_tags,
                    have_state ? &heuristic_moment : nullptr,
                    have_state ? &heuristic_vad : nullptr,
                    &heuristic_guidance);
        }
    }

    server_metacognitive_control_state control_state = {};
    if (turn_builder && turn_builder->has_current_state()) {
        control_state.moment = turn_builder->current_moment();
        control_state.vad = turn_builder->current_vad();
    }
    control_state.ongoing_task_due = ongoing_task_due;
    control_state.bridge_scoped = json_value(body, "x-vicuna-bridge-scoped", false);
    control_state.cognitive_replay = cognitive_replay;
    control_state.suppress_replay_admission = suppress_replay_admission;
    control_state.heuristic = heuristic_decision;
    const server_metacognitive_policy_decision policy_decision =
            emotive_runtime.compute_control_policy(control_state);
    if (trace_context) {
        runtime_request_trace_log(*trace_context, "control_policy", "policy_computed", {
            {"policy_version", policy_decision.policy_version},
            {"selected_mode", policy_decision.selected_mode},
            {"reasoning_depth", policy_decision.reasoning_depth},
            {"score_breakdown", {
                {"direct", policy_decision.direct_score},
                {"reflective", policy_decision.reflective_score},
                {"tool_light", policy_decision.tool_light_score},
                {"tool_heavy", policy_decision.tool_heavy_score},
                {"background_defer", policy_decision.background_defer_score},
            }},
            {"reasoning_score", policy_decision.reasoning_score},
            {"tool_aggression", policy_decision.tool_aggression},
            {"interrupt_score", policy_decision.interrupt_score},
            {"tool_parallelism_cap", policy_decision.tool_parallelism_cap},
            {"interrupt_allowed", policy_decision.interrupt_allowed},
            {"replan_required", policy_decision.replan_required},
            {"early_stop_ok", policy_decision.early_stop_ok},
            {"force_synthesis", policy_decision.force_synthesis},
            {"heuristic_biases", policy_decision.heuristic_biases},
            {"heuristic_matched", heuristic_decision.matched},
            {"heuristic_id", heuristic_decision.heuristic_id.empty() ? json(nullptr) : json(heuristic_decision.heuristic_id)},
        });
    }

    json adjusted_body = body;
    if (!adjusted_body.contains("parallel_tool_calls")) {
        adjusted_body["parallel_tool_calls"] = policy_decision.tool_parallelism_cap > 1;
    }
    if (!adjusted_body.contains("x-vicuna-provider-max-tokens-override") &&
            !adjusted_body.contains("max_tokens") &&
            !adjusted_body.contains("max_output_tokens") &&
            !adjusted_body.contains("max_completion_tokens")) {
        adjusted_body["x-vicuna-provider-max-tokens-override"] =
                metacognitive_token_budget_for_depth(policy_decision.reasoning_depth);
    }

    const json provider_body = inject_additive_runtime_guidance(
            adjusted_body,
            turn_builder.get(),
            &emotive_runtime,
            &policy_decision,
            &heuristic_decision,
            heuristic_guidance.empty() ? nullptr : &heuristic_guidance,
            enable_heuristic_guidance && !cognitive_replay,
            trace_context);
    if (turn_builder) {
        turn_builder->set_final_policy({
            {"policy_version", policy_decision.policy_version},
            {"selected_mode", policy_decision.selected_mode},
            {"reasoning_depth", policy_decision.reasoning_depth},
            {"score_breakdown", {
                {"direct", policy_decision.direct_score},
                {"reflective", policy_decision.reflective_score},
                {"tool_light", policy_decision.tool_light_score},
                {"tool_heavy", policy_decision.tool_heavy_score},
                {"background_defer", policy_decision.background_defer_score},
            }},
            {"reasoning_score", policy_decision.reasoning_score},
            {"tool_aggression", policy_decision.tool_aggression},
            {"interrupt_score", policy_decision.interrupt_score},
            {"tool_parallelism_cap", policy_decision.tool_parallelism_cap},
            {"interrupt_allowed", policy_decision.interrupt_allowed},
            {"replan_required", policy_decision.replan_required},
            {"early_stop_ok", policy_decision.early_stop_ok},
            {"force_synthesis", policy_decision.force_synthesis},
            {"heuristic_biases", policy_decision.heuristic_biases},
            {"prompt_hints", policy_decision.prompt_hints},
        });
        turn_builder->set_heuristic_retrieval({
            {"matched", heuristic_decision.matched},
            {"record_id", heuristic_decision.record_id},
            {"heuristic_id", heuristic_decision.heuristic_id},
            {"semantic_score", heuristic_decision.semantic_score},
            {"struct_score", heuristic_decision.struct_score},
            {"emotive_score", heuristic_decision.emotive_score},
            {"total_score", heuristic_decision.total_score},
            {"threshold", heuristic_decision.threshold},
            {"control_biases", heuristic_decision.control_biases},
        });
    }
    deepseek_request_trace provider_trace;
    if (trace_context) {
        provider_trace.request_id = trace_context->request_id;
        provider_trace.mode_label = mode_label.empty() ? trace_context->mode_label : mode_label;
        provider_trace.emit = [trace_context](const std::string & event, const json & data) {
            runtime_request_trace_log(*trace_context, "provider", event, data);
        };
    }
    if (!deepseek_complete_chat(
                config,
                provider_body,
                out_result,
                turn_builder ? &observer : nullptr,
                out_error,
                trace_context ? &provider_trace : nullptr)) {
        return false;
    }

    if (turn_builder) {
        turn_builder->observe_runtime_event(
                "provider_finish:" + (out_result->finish_reason.empty() ? std::string("stop") : out_result->finish_reason));
        server_emotive_trace trace = turn_builder->finalize();
        trace.final_policy = {
            {"policy_version", policy_decision.policy_version},
            {"selected_mode", policy_decision.selected_mode},
            {"reasoning_depth", policy_decision.reasoning_depth},
            {"score_breakdown", {
                {"direct", policy_decision.direct_score},
                {"reflective", policy_decision.reflective_score},
                {"tool_light", policy_decision.tool_light_score},
                {"tool_heavy", policy_decision.tool_heavy_score},
                {"background_defer", policy_decision.background_defer_score},
            }},
            {"reasoning_score", policy_decision.reasoning_score},
            {"tool_aggression", policy_decision.tool_aggression},
            {"interrupt_score", policy_decision.interrupt_score},
            {"tool_parallelism_cap", policy_decision.tool_parallelism_cap},
            {"interrupt_allowed", policy_decision.interrupt_allowed},
            {"replan_required", policy_decision.replan_required},
            {"early_stop_ok", policy_decision.early_stop_ok},
            {"force_synthesis", policy_decision.force_synthesis},
            {"heuristic_biases", policy_decision.heuristic_biases},
            {"prompt_hints", policy_decision.prompt_hints},
        };
        trace.heuristic_retrieval = {
            {"matched", heuristic_decision.matched},
            {"record_id", heuristic_decision.record_id},
            {"heuristic_id", heuristic_decision.heuristic_id},
            {"semantic_score", heuristic_decision.semantic_score},
            {"struct_score", heuristic_decision.struct_score},
            {"emotive_score", heuristic_decision.emotive_score},
            {"total_score", heuristic_decision.total_score},
            {"threshold", heuristic_decision.threshold},
            {"control_biases", heuristic_decision.control_biases},
        };
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
        json * out_error,
        const runtime_request_trace_context * trace_context = nullptr) {
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
    if (!load_server_owned_telegram_runtime_tools(adapter_config, &runtime_tools, &tool_error)) {
        if (trace_context) {
            runtime_request_trace_log(*trace_context, "runtime", "telegram_runtime_tools_load_failed", {
                {"message", tool_error},
            });
        }
        if (out_error) {
            *out_error = format_error_response(tool_error, ERROR_TYPE_SERVER);
        }
        return false;
    }
    if (trace_context) {
        runtime_request_trace_log(*trace_context, "runtime", "telegram_runtime_tools_loaded", {
            {"tool_count", runtime_tools.is_array() ? static_cast<int64_t>(runtime_tools.size()) : 0},
        });
    }

    json current_body = build_server_owned_bridge_request_body(body, context, runtime_tools);
    for (int32_t round = 0; round < std::max<int32_t>(1, adapter_config.max_rounds); ++round) {
        if (trace_context) {
            runtime_request_trace_log(*trace_context, "runtime", "bridge_round_started", {
                {"round", round + 1},
            });
        }
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
                        "telegram_bridge",
                        trace_context) :
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
                        0.0f,
                        "telegram_bridge",
                        trace_context);
        if (!executed) {
            return false;
        }

        if (round_result.tool_calls.empty()) {
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "runtime", "bridge_round_completed_without_tool_call", {
                    {"round", round + 1},
                    {"finish_reason", round_result.finish_reason},
                    {"content_chars", static_cast<int64_t>(round_result.content.size())},
                });
            }
            *out_result = std::move(round_result);
            return true;
        }

        bool contains_telegram_delivery = false;
        bool contains_runtime_tool = false;
        for (const auto & tool_call : round_result.tool_calls) {
            if (is_server_owned_telegram_delivery_tool_name(tool_call.name)) {
                contains_telegram_delivery = true;
            } else {
                contains_runtime_tool = true;
            }
        }
        if (contains_telegram_delivery && contains_runtime_tool) {
            if (out_error) {
                *out_error = format_error_response("bridge-scoped Telegram request returned mixed Telegram delivery and runtime tool calls", ERROR_TYPE_SERVER);
            }
            return false;
        }
        if (contains_telegram_delivery) {
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "runtime", "bridge_round_selected_telegram_delivery", {
                    {"round", round + 1},
                    {"tool_call_count", static_cast<int64_t>(round_result.tool_calls.size())},
                });
            }
            *out_result = std::move(round_result);
            return true;
        }

        json messages = current_body.at("messages");
        messages.push_back(build_bridge_assistant_tool_replay_message(round_result));
        for (const auto & tool_call : round_result.tool_calls) {
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "runtime_tool", "tool_invocation_started", {
                    {"round", round + 1},
                    {"tool_name", tool_call.name},
                    {"tool_call_id", tool_call.id},
                });
            }
            json observation;
            std::string observation_error;
            if (!invoke_server_owned_telegram_runtime_tool(adapter_config, tool_call, &observation, &observation_error)) {
                if (trace_context) {
                    runtime_request_trace_log(*trace_context, "runtime_tool", "tool_invocation_failed", {
                        {"round", round + 1},
                        {"tool_name", tool_call.name},
                        {"tool_call_id", tool_call.id},
                        {"message", observation_error},
                    });
                }
                if (out_error) {
                    *out_error = format_error_response(observation_error, ERROR_TYPE_SERVER);
                }
                return false;
            }
            if (trace_context) {
                runtime_request_trace_log(*trace_context, "runtime_tool", "tool_invocation_finished", {
                    {"round", round + 1},
                    {"tool_name", tool_call.name},
                    {"tool_call_id", tool_call.id},
                    {"observation", observation},
                });
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
    const std::string route = responses_api ? "/v1/responses" : (text_completion_api ? "/v1/completions" : "/v1/chat/completions");
    const runtime_request_trace_context trace_context =
            make_runtime_request_trace_context(req, route, telegram_context, telegram_context.active ? "telegram_bridge" : "foreground");
    runtime_request_trace_log(trace_context, "runtime", "request_received", {
        {"stream", json_value(body, "stream", false)},
        {"responses_api", responses_api},
        {"text_completion_api", text_completion_api},
        {"message_count", body.contains("messages") && body.at("messages").is_array() ?
                static_cast<int64_t>(body.at("messages").size()) : 0},
        {"tool_count", body.contains("tools") && body.at("tools").is_array() ?
                static_cast<int64_t>(body.at("tools").size()) : 0},
    });
    const bool executed = telegram_context.active ?
            execute_bridge_scoped_telegram_request(config, emotive_runtime, req, body, &result, &error, &trace_context) :
            (should_use_staged_tool_loop_for_request(body) ?
                    execute_deepseek_chat_with_staged_tools(config, emotive_runtime, body, &result, &error, false, "foreground", &trace_context) :
                    execute_deepseek_chat_with_emotive(config, emotive_runtime, body, &result, &error, nullptr, false, std::string(), true, false, 0.0f, "foreground", &trace_context));
    if (!executed) {
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", json_value(error, "message", std::string("request execution failed"))},
        });
        return make_error_response(error);
    }

    telegram_delivery_result telegram_delivery = {};
    if (!execute_telegram_delivery_for_bridge_request(req, telegram_outbox, &result, &telegram_delivery, &error, &trace_context)) {
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", json_value(error, "message", std::string("telegram delivery execution failed"))},
        });
        return make_error_response(error);
    }

    const std::string completion_id = gen_chatcmplid();
    if (responses_api) {
        json response = deepseek_format_responses_response(config, result);
        if (!result.rich_response.is_null()) {
            response["vicuna_rich_response"] = result.rich_response;
        }
        const json delivery_json = telegram_delivery_result_to_json(telegram_delivery);
        if (!delivery_json.is_null()) {
            response["vicuna_telegram_delivery"] = delivery_json;
        }
        return make_json_response(response);
    }
    if (text_completion_api) {
        json response = deepseek_format_text_completion_response(config, result, completion_id);
        if (!result.rich_response.is_null()) {
            response["vicuna_rich_response"] = result.rich_response;
        }
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
        runtime_request_trace_log(trace_context, "runtime", "request_completed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"stream", true},
            {"finish_reason", result.finish_reason},
            {"tool_call_count", static_cast<int64_t>(result.tool_calls.size())},
            {"queued_telegram_delivery", telegram_delivery.queued},
        });
        return res;
    }

    json response = deepseek_format_chat_completion_response(config, result, completion_id);
    if (!result.rich_response.is_null()) {
        response["vicuna_rich_response"] = result.rich_response;
    }
    const json delivery_json = telegram_delivery_result_to_json(telegram_delivery);
    if (!delivery_json.is_null()) {
        response["vicuna_telegram_delivery"] = delivery_json;
    }
    runtime_request_trace_log(trace_context, "runtime", "request_completed", {
        {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
        {"stream", false},
        {"finish_reason", result.finish_reason},
        {"tool_call_count", static_cast<int64_t>(result.tool_calls.size())},
        {"queued_telegram_delivery", telegram_delivery.queued},
        {"content_chars", static_cast<int64_t>(result.content.size())},
    });
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
    const runtime_request_trace_context trace_context =
            make_background_trace_context("/background/ongoing-task", "ongoing_task_decision");
    if (!out_registry) {
        ongoing_task_worker_set_error(worker_state, "ongoing-task registry output must not be null");
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", "ongoing-task registry output must not be null"},
        });
        return false;
    }
    if (out_selected_task) {
        *out_selected_task = server_ongoing_task_summary();
    }

    const int64_t poll_ms = current_time_ms_utc();
    runtime_request_trace_log(trace_context, "runtime", "request_received", {
        {"current_time_ms", poll_ms},
        {"ongoing_tasks_enabled", ongoing_config.enabled},
    });
    ongoing_task_worker_mark_poll(worker_state, poll_ms);
    ongoing_task_worker_set_mode(worker_state, "ongoing_task_poll", true);

    server_ongoing_task_registry registry = {};
    std::string registry_error;
    if (!load_ongoing_task_registry(ongoing_config, &registry, &registry_error)) {
        ongoing_task_worker_set_error(worker_state, registry_error);
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", registry_error},
        });
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
        runtime_request_trace_log(trace_context, "runtime", "request_completed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"task_count", 0},
            {"should_run", false},
            {"rationale", decision.rationale},
        });
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
                tasks.empty() ? 0.0f : 1.0f,
                "ongoing_task_decision",
                &trace_context)) {
        ongoing_task_worker_set_error(
                worker_state,
                json_value(error, "message", std::string("ongoing-task decision request failed")));
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", json_value(error, "message", std::string("ongoing-task decision request failed"))},
        });
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    std::string parse_error;
    if (!parse_ongoing_task_decision_response(result.content, &decision, &parse_error)) {
        ongoing_task_worker_set_error(worker_state, parse_error);
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", parse_error},
        });
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }
    decision.decided_at_ms = poll_ms;
    decision.current_time_iso = current_time_iso;
    decision.task_count = (int32_t) tasks.size();

    if (decision.should_run && !find_ongoing_task_summary(tasks, decision.selected_task_id)) {
        ongoing_task_worker_set_error(worker_state, "ongoing-task decision selected an unknown task_id");
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", "ongoing-task decision selected an unknown task_id"},
        });
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    ongoing_task_worker_set_decision(worker_state, decision);
    *out_registry = std::move(registry);
    runtime_request_trace_log(trace_context, "runtime", "request_completed", {
        {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
        {"task_count", static_cast<int64_t>(tasks.size())},
        {"should_run", decision.should_run},
        {"selected_task_id", decision.should_run ? json(decision.selected_task_id) : json(nullptr)},
        {"rationale", decision.rationale},
    });
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
    const runtime_request_trace_context trace_context =
            make_background_trace_context("/background/ongoing-task", "ongoing_task_execution");
    runtime_request_trace_log(trace_context, "runtime", "request_received", {
        {"task_id", task.task_id},
        {"task_text", task.task_text},
    });
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
                0.8f,
                "ongoing_task_execution",
                &trace_context)) {
        ongoing_task_worker_set_error(
                worker_state,
                json_value(error, "message", std::string("ongoing-task execution request failed")));
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", json_value(error, "message", std::string("ongoing-task execution request failed"))},
            {"task_id", task.task_id},
        });
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    const std::string completed_at_iso = iso_from_epoch_ms(current_time_ms_utc());
    std::string completion_error;
    if (!mark_ongoing_task_complete(&registry, task.task_id, completed_at_iso, &completion_error)) {
        ongoing_task_worker_set_error(worker_state, completion_error);
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", completion_error},
            {"task_id", task.task_id},
        });
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }
    if (!save_ongoing_task_registry(ongoing_config, std::move(registry), &completion_error)) {
        ongoing_task_worker_set_error(worker_state, completion_error);
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"message", completion_error},
            {"task_id", task.task_id},
        });
        ongoing_task_worker_mark_finished(worker_state);
        return false;
    }

    ongoing_task_worker_mark_completion(worker_state, task.task_id, completed_at_iso);
    runtime_request_trace_log(trace_context, "runtime", "request_completed", {
        {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
        {"task_id", task.task_id},
        {"completed_at", completed_at_iso},
    });
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
    const runtime_request_trace_context trace_context =
            make_background_trace_context("/background/cognitive-replay", "heuristic_compression");
    runtime_request_trace_log(trace_context, "runtime", "request_received", {
        {"entry_id", entry.entry_id},
        {"baseline_negative_mass", result.comparison.baseline_negative_mass},
        {"replay_negative_mass", result.comparison.replay_negative_mass},
    });
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
                false,
                false,
                0.0f,
                "heuristic_compression",
                &trace_context)) {
        if (worker_state) {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->last_error = json_value(error, "message", std::string("heuristic compression request failed"));
        }
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"entry_id", entry.entry_id},
            {"message", json_value(error, "message", std::string("heuristic compression request failed"))},
        });
        return;
    }

    try {
        const server_heuristic_object heuristic = parse_heuristic_object_response(compression_result.content);
        std::string persistence_error;
        if (!emotive_runtime.store_heuristic_memory_record(entry.entry_id, heuristic, &persistence_error)) {
            if (worker_state) {
                std::lock_guard<std::mutex> lock(worker_state->mutex);
                worker_state->last_error = persistence_error.empty() ?
                        std::string("heuristic persistence failed") :
                        persistence_error;
            }
            runtime_request_trace_log(trace_context, "runtime", "request_failed", {
                {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
                {"entry_id", entry.entry_id},
                {"message", persistence_error.empty() ? std::string("heuristic persistence failed") : persistence_error},
            });
            return;
        }
        runtime_request_trace_log(trace_context, "runtime", "request_completed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"entry_id", entry.entry_id},
            {"heuristic_id", heuristic.heuristic_id},
            {"title", heuristic.title},
        });
    } catch (const std::exception & e) {
        if (worker_state) {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->last_error = e.what();
        }
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"entry_id", entry.entry_id},
            {"message", e.what()},
        });
    }
}

static void run_cognitive_replay_once(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const server_cognitive_replay_entry & entry,
        cognitive_replay_worker_state * worker_state) {
    const runtime_request_trace_context trace_context =
            make_background_trace_context("/background/cognitive-replay", "cognitive_replay");
    runtime_request_trace_log(trace_context, "runtime", "request_received", {
        {"entry_id", entry.entry_id},
        {"negative_mass", entry.negative_mass},
        {"valence_drop", entry.valence_drop},
        {"dominance_drop", entry.dominance_drop},
    });
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
                entry.entry_id,
                true,
                false,
                0.0f,
                "cognitive_replay",
                &trace_context)) {
        emotive_runtime.fail_cognitive_replay_entry(
                entry.entry_id,
                json_value(error, "message", std::string("cognitive replay request failed")));
        if (worker_state) {
            std::lock_guard<std::mutex> lock(worker_state->mutex);
            worker_state->last_error = json_value(error, "message", std::string("cognitive replay request failed"));
        }
        runtime_request_trace_log(trace_context, "runtime", "request_failed", {
            {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
            {"entry_id", entry.entry_id},
            {"message", json_value(error, "message", std::string("cognitive replay request failed"))},
        });
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
    runtime_request_trace_log(trace_context, "runtime", "request_completed", {
        {"elapsed_ms", runtime_now_epoch_ms() - trace_context.started_at_ms},
        {"entry_id", entry.entry_id},
        {"resolved", resolved_entry.status == SERVER_COGNITIVE_REPLAY_RESOLVED},
        {"improved", resolved_result.comparison.improved},
    });

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
        payload["bridge_runtime"] = {
            {"telegram_runtime_tool_cache", telegram_runtime_tool_cache_health_json()},
            {"staged_prompt_cache", staged_tool_prompt_cache_health_json()},
            {"staged_fallback_enabled", parse_truthy_header(std::getenv("VICUNA_ENABLE_STAGED_TOOL_FALLBACK") ? std::getenv("VICUNA_ENABLE_STAGED_TOOL_FALLBACK") : "")},
            {"delivery_contract", "rich_plan_response_v1"},
        };
        payload["request_traces"] = runtime_request_trace_health_json();
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

    const auto get_request_traces = [](const server_http_req & req) {
        return make_json_response(runtime_request_trace_read_json(req));
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
    ctx_http.get("/v1/debug/request-traces", ex_wrapper(get_request_traces));
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
