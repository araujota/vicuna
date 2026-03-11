#include "llama-hard-memory.h"

#include "llama-vocab.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

static std::string rstrip_slash(std::string value) {
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

template<size_t N>
static void copy_cstr(char (&dst)[N], const std::string & src) {
    std::memset(dst, 0, sizeof(dst));
    if (N == 0) {
        return;
    }
    const size_t copy_len = std::min(src.size(), N - 1);
    if (copy_len > 0) {
        std::memcpy(dst, src.data(), copy_len);
    }
}

template<size_t N>
static std::string read_cstr(const char (&src)[N]) {
    const size_t len = strnlen(src, N);
    return std::string(src, len);
}

static std::string trim_text(const std::string & text, size_t max_chars) {
    if (text.size() <= max_chars) {
        return text;
    }
    if (max_chars <= 3) {
        return text.substr(0, max_chars);
    }
    return text.substr(0, max_chars - 3) + "...";
}

static std::string join_lines(const json & value) {
    if (!value.is_array()) {
        return {};
    }
    std::ostringstream oss;
    bool first = true;
    for (const auto & item : value) {
        if (!item.is_string()) {
            continue;
        }
        if (!first) {
            oss << '\n';
        }
        first = false;
        oss << item.get<std::string>();
    }
    return oss.str();
}

static std::string event_text(const llama_vocab * vocab, const llama_self_state_event & event) {
    if (!vocab || !event.tokens || event.n_tokens == 0) {
        return {};
    }

    std::string out;
    for (size_t i = 0; i < event.n_tokens; ++i) {
        out += vocab->token_to_piece(event.tokens[i]);
    }
    return out;
}

static std::string register_delta_summary(const llama_self_state_delta_summary & delta) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    for (int32_t i = 0; i < delta.dimension_count; ++i) {
        if (i > 0) {
            oss << "; ";
        }
        oss << llama_self_state_register_name(delta.dimensions[i].register_id)
            << "=" << delta.dimensions[i].before_value
            << "->" << delta.dimensions[i].after_value;
    }
    return oss.str();
}

static const char * temporal_role_name(int32_t role) {
    switch (role) {
        case LLAMA_SERVING_LORA_LAYER_ACTIVE: return "active";
        case LLAMA_SERVING_LORA_LAYER_PAST_WEEK: return "past_week";
        case LLAMA_SERVING_LORA_LAYER_PAST_MONTH: return "past_month";
        case LLAMA_SERVING_LORA_LAYER_PAST_QUARTER: return "past_quarter";
        case LLAMA_SERVING_LORA_LAYER_PAST_YEAR: return "past_year";
        case LLAMA_SERVING_LORA_LAYER_ALL_TIME: return "all_time";
        default: return "unknown";
    }
}

static uint64_t fnv1a64(const std::string & text) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= (uint64_t) ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

static std::string effective_container_tag(
        const llama_hard_memory_config & config,
        const char * query_override) {
    if (query_override && query_override[0] != '\0') {
        return query_override;
    }
    if (config.container_tag[0] != '\0') {
        return config.container_tag;
    }
    if (config.runtime_identity[0] != '\0') {
        return config.runtime_identity;
    }
    return {};
}

static httplib::Headers build_headers(const llama_hard_memory_config & config) {
    httplib::Headers headers = {
        {"Content-Type", "application/json"},
    };
    const std::string token = read_cstr(config.auth_token);
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
        headers.emplace("x-supermemory-api-key", token);
    }
    return headers;
}

static void init_result(llama_hard_memory_result & out) {
    out = {};
    out.tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_QUERY;
    out.request_started_us = ggml_time_us();
}

static void init_archive(llama_hard_memory_archive_trace & out) {
    out = {};
    out.tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_WRITE;
    out.request_started_us = ggml_time_us();
}

static bool query_enabled(const llama_hard_memory_config & config) {
    return config.enabled && config.base_url[0] != '\0' && config.auth_token[0] != '\0';
}

} // namespace

llama_hard_memory::llama_hard_memory() {
    config = llama_hard_memory_default_config();
}

bool llama_hard_memory::configure(const llama_hard_memory_config & next) {
    config = next;
    const std::string normalized_url = rstrip_slash(read_cstr(config.base_url));
    copy_cstr(config.base_url, normalized_url);
    config.timeout_ms = std::max(100, config.timeout_ms);
    config.max_results = std::max<int32_t>(1, std::min<int32_t>(config.max_results, LLAMA_HARD_MEMORY_MAX_RESULTS));
    config.query_threshold = std::max(0.0f, std::min(1.0f, config.query_threshold));
    config.archival_delta_threshold = std::max(0.0f, std::min(4.0f, config.archival_delta_threshold));
    return true;
}

bool llama_hard_memory::get_config(llama_hard_memory_config * out_config) const {
    if (!out_config) {
        return false;
    }
    *out_config = config;
    return true;
}

bool llama_hard_memory::query(const llama_hard_memory_query_request & request, llama_hard_memory_result * out_result) {
    llama_hard_memory_result result = {};
    init_result(result);
    result.profile_included = request.include_profile || config.include_profile_by_default;

    std::string query_text_value = read_cstr(request.query);
    if (request.use_temporal_self_hint) {
        query_text_value += "\nTemporal self hint: ";
        query_text_value += temporal_role_name(request.temporal_adapter_role);
    }
    const std::string tag = effective_container_tag(config, request.container_tag);
    copy_cstr(result.effective_container_tag, tag);

    if (!query_enabled(config)) {
        copy_cstr(result.error, "hard memory is disabled or missing endpoint/auth configuration");
        result.request_completed_us = ggml_time_us();
        last_result = result;
        if (out_result) {
            *out_result = result;
        }
        return true;
    }

    if (query_text_value.empty()) {
        copy_cstr(result.error, "hard memory query text is empty");
        result.request_completed_us = ggml_time_us();
        last_result = result;
        if (out_result) {
            *out_result = result;
        }
        return true;
    }

    if (tag.empty()) {
        copy_cstr(result.error, "hard memory container tag is empty");
        result.request_completed_us = ggml_time_us();
        last_result = result;
        if (out_result) {
            *out_result = result;
        }
        return true;
    }

    httplib::Client cli(read_cstr(config.base_url));
    cli.set_connection_timeout(std::chrono::milliseconds(config.timeout_ms));
    cli.set_read_timeout(std::chrono::milliseconds(config.timeout_ms));
    cli.set_write_timeout(std::chrono::milliseconds(config.timeout_ms));

    json body = {
        {"containerTag", tag},
        {"q", query_text_value},
        {"threshold", request.threshold > 0.0f ? request.threshold : config.query_threshold},
    };

    const auto response = cli.Post("/v4/profile", build_headers(config), body.dump(), "application/json");
    result.request_completed_us = ggml_time_us();

    if (!response) {
        copy_cstr(result.error, "hard memory query request failed");
        last_result = result;
        if (out_result) {
            *out_result = result;
        }
        return true;
    }

    result.status_code = response->status;
    if (response->status < 200 || response->status >= 300) {
        copy_cstr(result.error, trim_text(response->body, sizeof(result.error) - 1));
        last_result = result;
        if (out_result) {
            *out_result = result;
        }
        return true;
    }

    try {
        const json parsed = json::parse(response->body);
        result.ok = true;
        copy_cstr(result.profile_static, join_lines(parsed.value("profile", json::object()).value("static", json::array())));
        copy_cstr(result.profile_dynamic, join_lines(parsed.value("profile", json::object()).value("dynamic", json::array())));

        const json search_results = parsed.contains("searchResults") ? parsed.at("searchResults") : json::object();
        const json items = search_results.value("results", json::array());
        const int32_t limit = std::max<int32_t>(1, std::min<int32_t>(request.limit > 0 ? request.limit : config.max_results, LLAMA_HARD_MEMORY_MAX_RESULTS));
        for (const auto & item : items) {
            if (result.result_count >= limit) {
                break;
            }
            auto & dst = result.results[result.result_count++];
            dst.memory_result = item.contains("memory") || !item.contains("chunk");
            dst.similarity = item.value("similarity", item.value("score", 0.0f));
            copy_cstr(dst.id, item.value("id", std::string()));
            copy_cstr(dst.title, item.value("title", std::string()));
            copy_cstr(dst.content, trim_text(item.value("memory", item.value("chunk", item.value("content", std::string()))), LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1));
        }
    } catch (const std::exception & e) {
        result.ok = false;
        copy_cstr(result.error, trim_text(e.what(), sizeof(result.error) - 1));
    }

    last_result = result;
    if (out_result) {
        *out_result = result;
    }
    return true;
}

bool llama_hard_memory::get_last_result(llama_hard_memory_result * out_result) const {
    if (!out_result) {
        return false;
    }
    *out_result = last_result;
    return true;
}

bool llama_hard_memory::archive_event(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        const llama_self_state_delta_summary & delta) {
    llama_hard_memory_archive_trace trace = {};
    init_archive(trace);
    trace.delta = delta;

    const std::string tag = effective_container_tag(config, nullptr);
    copy_cstr(trace.container_tag, tag);

    if (!query_enabled(config) || !config.archive_enabled) {
        copy_cstr(trace.error, "hard memory archival is disabled or missing endpoint/auth configuration");
        trace.request_completed_us = ggml_time_us();
        last_archive = trace;
        return true;
    }

    const std::string text = event_text(vocab, event);
    const std::string excerpt = trim_text(text, LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1);
    copy_cstr(trace.content_excerpt, excerpt);

    if (tag.empty() || text.empty()) {
        copy_cstr(trace.error, "hard memory archival requires non-empty container tag and event text");
        trace.request_completed_us = ggml_time_us();
        last_archive = trace;
        return true;
    }

    trace.attempted = true;

    const std::string identity = read_cstr(config.runtime_identity).empty() ? "vicuna" : read_cstr(config.runtime_identity);
    const std::string digest_input = identity + "|" + text + "|" + std::to_string(event.role) + "|" + std::to_string(event.channel) + "|" + std::to_string((unsigned long long) event.flags) + "|" + std::to_string(delta.total_delta);
    char custom_id[LLAMA_HARD_MEMORY_MAX_ID_CHARS] = {};
    std::snprintf(custom_id, sizeof(custom_id), "vicuna-%016llx",
            (unsigned long long) fnv1a64(digest_input));
    copy_cstr(trace.custom_id, custom_id);

    std::ostringstream content;
    content.setf(std::ios::fixed);
    content.precision(3);
    content << "vicuna self-state perturbation event\n"
            << "runtime: " << identity << "\n"
            << "role: " << event.role << "\n"
            << "channel: " << event.channel << "\n"
            << "flags: " << event.flags << "\n"
            << "total_delta: " << delta.total_delta << "\n"
            << "max_delta: " << delta.max_delta << "\n"
            << "changed_registers: " << register_delta_summary(delta) << "\n"
            << "message: " << text;

    json metadata = {
        {"source", "vicuna"},
        {"runtimeIdentity", identity},
        {"eventRole", event.role},
        {"eventChannel", event.channel},
        {"eventFlags", event.flags},
        {"totalDelta", delta.total_delta},
        {"maxDelta", delta.max_delta},
    };
    json changed = json::array();
    for (int32_t i = 0; i < delta.dimension_count; ++i) {
        changed.push_back({
            {"registerId", delta.dimensions[i].register_id},
            {"registerName", llama_self_state_register_name(delta.dimensions[i].register_id)},
            {"before", delta.dimensions[i].before_value},
            {"after", delta.dimensions[i].after_value},
            {"absDelta", delta.dimensions[i].abs_delta},
        });
    }
    metadata["changedRegisters"] = changed;

    json body = {
        {"containerTag", tag},
        {"memories", json::array({
            {
                {"content", trim_text(content.str(), 1024)},
                {"isStatic", false},
                {"metadata", metadata},
            }
        })}
    };

    httplib::Client cli(read_cstr(config.base_url));
    cli.set_connection_timeout(std::chrono::milliseconds(config.timeout_ms));
    cli.set_read_timeout(std::chrono::milliseconds(config.timeout_ms));
    cli.set_write_timeout(std::chrono::milliseconds(config.timeout_ms));

    const auto response = cli.Post("/v4/memories", build_headers(config), body.dump(), "application/json");
    trace.request_completed_us = ggml_time_us();
    if (!response) {
        copy_cstr(trace.error, "hard memory archival request failed");
        last_archive = trace;
        return true;
    }

    trace.status_code = response->status;
    if (response->status < 200 || response->status >= 300) {
        copy_cstr(trace.error, trim_text(response->body, sizeof(trace.error) - 1));
        last_archive = trace;
        return true;
    }

    trace.archived = true;
    last_archive = trace;
    return true;
}

bool llama_hard_memory::get_last_archive_trace(llama_hard_memory_archive_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = last_archive;
    return true;
}

llama_hard_memory_config llama_hard_memory_default_config(void) {
    llama_hard_memory_config config = {};
    config.enabled = false;
    config.archive_enabled = true;
    config.include_profile_by_default = true;
    config.archive_counterfactual_events = false;
    config.timeout_ms = 2500;
    config.max_results = 4;
    config.query_threshold = 0.45f;
    config.archival_delta_threshold = 0.65f;
    copy_cstr(config.base_url, "https://api.supermemory.ai");
    copy_cstr(config.runtime_identity, "vicuna");
    return config;
}
