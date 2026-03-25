#include "server-common.h"
#include "server-deepseek.h"
#include "server-emotive-runtime.h"
#include "server-http.h"
#include "server-runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <clocale>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <signal.h>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

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

static json telegram_outbox_item_to_json(const telegram_outbox_item & item) {
    json payload = {
        {"sequence_number", item.sequence_number},
        {"kind", item.kind},
        {"chat_scope", item.chat_scope},
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
    if (!outbox) {
        throw std::invalid_argument("telegram outbox state was not configured");
    }

    const std::string kind = trim_copy(json_value(body, "kind", std::string("message")));
    if (kind != "message") {
        throw std::invalid_argument("telegram outbox only supports kind=message on the provider-only path");
    }

    telegram_outbox_item request = {};
    request.kind = kind;
    request.chat_scope = trim_copy(json_value(body, "chat_scope", std::string()));
    request.text = trim_copy(json_value(body, "text", std::string()));
    request.reply_to_message_id = std::max<int64_t>(0, json_value(body, "reply_to_message_id", int64_t(0)));
    request.intent = trim_copy(json_value(body, "intent", std::string()));
    request.dedupe_key = trim_copy(json_value(body, "dedupe_key", std::string()));
    request.urgency = std::max(0.0, json_value(body, "urgency", 0.0));

    if (request.chat_scope.empty()) {
        throw std::invalid_argument("telegram outbox write requires non-empty chat_scope");
    }
    if (request.text.empty()) {
        throw std::invalid_argument("telegram outbox write requires non-empty text");
    }

    std::lock_guard<std::mutex> lock(outbox->mutex);

    if (!request.dedupe_key.empty()) {
        for (const auto & existing : outbox->items) {
            if (existing.kind == request.kind &&
                    existing.chat_scope == request.chat_scope &&
                    existing.dedupe_key == request.dedupe_key) {
                return {
                    {"ok", true},
                    {"queued", false},
                    {"deduplicated", true},
                    {"sequence_number", existing.sequence_number},
                    {"chat_scope", existing.chat_scope},
                    {"stored_items", outbox->items.size()},
                    {"next_sequence_number", outbox->next_sequence_number},
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
        {"ok", true},
        {"queued", true},
        {"deduplicated", false},
        {"sequence_number", request.sequence_number},
        {"chat_scope", request.chat_scope},
        {"stored_items", outbox->items.size()},
        {"next_sequence_number", outbox->next_sequence_number},
    };
}

static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
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

static void seed_emotive_turn_from_request(const json & body, server_emotive_turn_builder * builder) {
    if (!builder) {
        return;
    }

    if (body.contains("messages") && body.at("messages").is_array()) {
        for (const auto & item : body.at("messages")) {
            if (!item.is_object() || json_value(item, "role", std::string()) != "user") {
                continue;
            }

            const std::string text = item.contains("content") ?
                    request_content_to_text(item.at("content")) :
                    std::string();
            if (!text.empty()) {
                builder->add_user_message(text);
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

static server_http_res_ptr handle_deepseek_chat(
        const deepseek_runtime_config & config,
        server_emotive_runtime & emotive_runtime,
        const json & body,
        bool responses_api,
        bool text_completion_api) {
    json error;
    if (!deepseek_validate_runtime_config(config, &error)) {
        return make_error_response(error);
    }

    deepseek_chat_result result;
    std::unique_ptr<server_emotive_turn_builder> turn_builder;
    if (emotive_runtime.config().enabled) {
        turn_builder = std::make_unique<server_emotive_turn_builder>(emotive_runtime, config.model);
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

    if (!deepseek_complete_chat(config, body, &result, turn_builder ? &observer : nullptr, &error)) {
        return make_error_response(error);
    }
    if (turn_builder) {
        turn_builder->observe_runtime_event("provider_finish:" + (result.finish_reason.empty() ? std::string("stop") : result.finish_reason));
        result.emotive_trace = server_emotive_trace_to_json(turn_builder->finalize());
    }

    const std::string completion_id = gen_chatcmplid();
    if (responses_api) {
        return make_json_response(deepseek_format_responses_response(config, result));
    }
    if (text_completion_api) {
        return make_json_response(deepseek_format_text_completion_response(config, result, completion_id));
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

    return make_json_response(deepseek_format_chat_completion_response(config, result, completion_id));
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
    server_emotive_runtime emotive_runtime(emotive_config);
    json config_error;
    if (!deepseek_validate_runtime_config(deepseek_config, &config_error)) {
        LOG_ERR("%s: %s\n", __func__, json_value(config_error, "message", std::string("invalid provider config")).c_str());
        return 1;
    }

    bridge_compat_state bridge_state;
    telegram_outbox_state telegram_outbox;
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    const auto get_health = [&bridge_state, &deepseek_config, &emotive_runtime, &telegram_outbox](const server_http_req &) {
        json payload = deepseek_build_health_json(deepseek_config);
        payload["proactive_mailbox"] = {
            {"stored_responses", 0},
            {"publish_total", 0},
            {"live_stream_connected", bridge_state.live_stream_connected.load()},
        };
        payload["telegram_outbox"] = telegram_outbox_health_json(telegram_outbox);
        payload["emotive_runtime"] = emotive_runtime.health_json();
        return make_json_response(payload);
    };

    const auto get_models = [&deepseek_config](const server_http_req &) {
        return make_json_response(deepseek_build_models_json(deepseek_config));
    };

    const auto get_latest_emotive_trace = [&emotive_runtime](const server_http_req &) {
        return make_json_response(emotive_runtime.latest_trace_json());
    };

    const auto post_chat_completions = [&deepseek_config, &emotive_runtime](const server_http_req & req) {
        const json body = json::parse(req.body);
        return handle_deepseek_chat(deepseek_config, emotive_runtime, body, false, false);
    };

    const auto post_completions = [&deepseek_config, &emotive_runtime](const server_http_req & req) {
        const json body = json::parse(req.body);
        return handle_deepseek_chat(deepseek_config, emotive_runtime, body, false, true);
    };

    const auto post_responses = [&deepseek_config, &emotive_runtime](const server_http_req & req) {
        const json body = convert_responses_input_to_chat(json::parse(req.body));
        return handle_deepseek_chat(deepseek_config, emotive_runtime, body, true, false);
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
    ctx_http.post("/v1/chat/completions", ex_wrapper(post_chat_completions));
    ctx_http.post("/v1/completions", ex_wrapper(post_completions));
    ctx_http.post("/v1/responses", ex_wrapper(post_responses));
    ctx_http.get("/v1/responses/stream", ex_wrapper(get_responses_stream));
    ctx_http.get("/v1/telegram/outbox", ex_wrapper(get_telegram_outbox));
    ctx_http.post("/v1/telegram/outbox", ex_wrapper(post_telegram_outbox));
    ctx_http.post("/v1/telegram/approval", ex_wrapper(post_telegram_approval));
    ctx_http.post("/v1/telegram/interruption", ex_wrapper(post_telegram_interruption));

    if (params.api_surface != SERVER_API_SURFACE_OPENAI) {
        ctx_http.post("/chat/completions", ex_wrapper(post_chat_completions));
        ctx_http.post("/completions", ex_wrapper(post_completions));
        ctx_http.post("/responses", ex_wrapper(post_responses));
        ctx_http.get("/models", ex_wrapper(get_models));
    }

    if (!ctx_http.start()) {
        LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
        return 1;
    }
    ctx_http.is_ready.store(true);

    shutdown_handler = [&](int) {
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

    return 0;
}
