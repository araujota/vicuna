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

static int32_t find_request_index(
        const llama_cognitive_hard_memory_request * requests,
        int32_t request_count,
        int32_t command_id) {
    for (int32_t i = 0; i < request_count; ++i) {
        if (requests[i].command_id == command_id) {
            return i;
        }
    }
    return -1;
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

static float clamp_unit(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

static const char * primitive_kind_name(int32_t kind) {
    switch (kind) {
        case LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT:      return "event_fragment";
        case LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY:          return "trajectory";
        case LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME:             return "outcome";
        case LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_OBSERVATION:    return "tool_observation";
        case LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL:          return "user_model";
        case LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT: return "self_model_fragment";
        default: return "event_fragment";
    }
}

static int32_t parse_primitive_kind(const json & value) {
    const std::string name = value.is_string() ? value.get<std::string>() : std::string();
    if (name == "trajectory") {
        return LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY;
    }
    if (name == "outcome") {
        return LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME;
    }
    if (name == "tool_observation") {
        return LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_OBSERVATION;
    }
    if (name == "user_model") {
        return LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL;
    }
    if (name == "self_model_fragment") {
        return LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT;
    }
    if (name == "event_fragment") {
        return LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT;
    }
    return value.is_number_integer() ? value.get<int32_t>() : LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT;
}

static const char * domain_name(int32_t domain) {
    switch (domain) {
        case LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS:    return "goal_progress";
        case LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME:     return "user_outcome";
        case LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC:        return "epistemic";
        case LLAMA_HARD_MEMORY_DOMAIN_EFFICIENCY:       return "efficiency";
        case LLAMA_HARD_MEMORY_DOMAIN_RECOVERY:         return "recovery";
        case LLAMA_HARD_MEMORY_DOMAIN_STRATEGY:         return "strategy";
        case LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT: return "self_improvement";
        default: return "epistemic";
    }
}

static int32_t parse_domain(const json & value) {
    const std::string name = value.is_string() ? value.get<std::string>() : std::string();
    if (name == "goal_progress") {
        return LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS;
    }
    if (name == "user_outcome") {
        return LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME;
    }
    if (name == "epistemic") {
        return LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
    }
    if (name == "efficiency") {
        return LLAMA_HARD_MEMORY_DOMAIN_EFFICIENCY;
    }
    if (name == "recovery") {
        return LLAMA_HARD_MEMORY_DOMAIN_RECOVERY;
    }
    if (name == "strategy") {
        return LLAMA_HARD_MEMORY_DOMAIN_STRATEGY;
    }
    if (name == "self_improvement") {
        return LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT;
    }
    return value.is_number_integer() ? value.get<int32_t>() : LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
}

static bool validate_primitive(llama_hard_memory_primitive * primitive) {
    if (!primitive ||
        primitive->kind < LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT ||
        primitive->kind > LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT ||
        primitive->domain < LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS ||
        primitive->domain > LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT ||
        primitive->key[0] == '\0' ||
        primitive->content[0] == '\0') {
        return false;
    }

    const float * bounded_values[] = {
        &primitive->importance,
        &primitive->confidence,
        &primitive->gain_bias,
        &primitive->allostatic_relevance,
    };
    for (const float * value : bounded_values) {
        if (!std::isfinite(*value)) {
            return false;
        }
    }

    primitive->importance = clamp_unit(primitive->importance);
    primitive->confidence = clamp_unit(primitive->confidence);
    primitive->gain_bias = clamp_unit(primitive->gain_bias);
    primitive->allostatic_relevance = clamp_unit(primitive->allostatic_relevance);
    return true;
}

static void ensure_primitive_key(llama_hard_memory_primitive * primitive) {
    if (!primitive || primitive->key[0] != '\0') {
        return;
    }
    const std::string digest_input =
            std::to_string(primitive->kind) + "|" +
            std::to_string(primitive->domain) + "|" +
            read_cstr(primitive->title) + "|" +
            read_cstr(primitive->content);
    char key[LLAMA_HARD_MEMORY_MAX_ID_CHARS] = {};
    std::snprintf(key, sizeof(key), "vicuna-%016llx",
            (unsigned long long) fnv1a64(digest_input));
    copy_cstr(primitive->key, key);
}

static bool validate_write_item(llama_hard_memory_write_item * item) {
    if (!item) {
        return false;
    }
    ensure_primitive_key(&item->primitive);
    return validate_primitive(&item->primitive);
}

static void copy_tags(
        char (&dst)[LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS][LLAMA_HARD_MEMORY_MAX_TAG_CHARS],
        const json & tags) {
    for (int32_t i = 0; i < LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS; ++i) {
        std::memset(dst[i], 0, sizeof(dst[i]));
    }
    if (!tags.is_array()) {
        return;
    }
    int32_t index = 0;
    for (const auto & item : tags) {
        if (index >= LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS || !item.is_string()) {
            break;
        }
        copy_cstr(dst[index++], item.get<std::string>());
    }
}

static json tags_to_json(
        const char (&tags)[LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS][LLAMA_HARD_MEMORY_MAX_TAG_CHARS]) {
    json out = json::array();
    for (int32_t i = 0; i < LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS; ++i) {
        if (tags[i][0] != '\0') {
            out.push_back(read_cstr(tags[i]));
        }
    }
    return out;
}

static void fill_primitive_summary(
        llama_hard_memory_primitive_summary & summary,
        const llama_hard_memory_primitive & primitive) {
    summary = {};
    summary.kind = primitive.kind;
    summary.domain = primitive.domain;
    summary.source_tool_kind = primitive.source_tool_kind;
    summary.flags = primitive.flags;
    summary.importance = primitive.importance;
    summary.confidence = primitive.confidence;
    summary.gain_bias = primitive.gain_bias;
    summary.allostatic_relevance = primitive.allostatic_relevance;
    copy_cstr(summary.key, read_cstr(primitive.key));
    copy_cstr(summary.title, read_cstr(primitive.title));
}

static json primitive_metadata_json(
        const llama_hard_memory_primitive & primitive,
        const std::string & runtime_identity,
        const llama_self_state_delta_summary * delta_summary) {
    json metadata = {
        {"source", "vicuna"},
        {"runtimeIdentity", runtime_identity},
        {"kind", primitive_kind_name(primitive.kind)},
        {"domain", domain_name(primitive.domain)},
        {"sourceRole", primitive.source_role},
        {"sourceChannel", primitive.source_channel},
        {"sourceToolKind", primitive.source_tool_kind},
        {"transactionId", primitive.transaction_id},
        {"flags", primitive.flags},
        {"importance", primitive.importance},
        {"confidence", primitive.confidence},
        {"gainBias", primitive.gain_bias},
        {"allostaticRelevance", primitive.allostatic_relevance},
        {"key", read_cstr(primitive.key)},
        {"title", read_cstr(primitive.title)},
        {"tags", tags_to_json(primitive.tags)},
    };

    if (delta_summary) {
        metadata["totalDelta"] = delta_summary->total_delta;
        metadata["maxDelta"] = delta_summary->max_delta;
        json changed = json::array();
        for (int32_t i = 0; i < delta_summary->dimension_count; ++i) {
            changed.push_back({
                {"registerId", delta_summary->dimensions[i].register_id},
                {"registerName", llama_self_state_register_name(delta_summary->dimensions[i].register_id)},
                {"before", delta_summary->dimensions[i].before_value},
                {"after", delta_summary->dimensions[i].after_value},
                {"absDelta", delta_summary->dimensions[i].abs_delta},
            });
        }
        metadata["changedRegisters"] = changed;
    }
    return metadata;
}

static bool archive_write_items_impl(
        const llama_hard_memory_config & config,
        const llama_hard_memory_write_item * input_items,
        int32_t item_count,
        const char * container_override,
        const llama_self_state_delta_summary * delta_summary,
        llama_hard_memory_archive_trace * io_last_archive) {
    if (!io_last_archive) {
        return false;
    }

    llama_hard_memory_archive_trace trace = {};
    init_archive(trace);
    if (delta_summary) {
        trace.delta = *delta_summary;
    }

    const std::string tag = effective_container_tag(config, container_override);
    copy_cstr(trace.container_tag, tag);

    if (!query_enabled(config) || !config.archive_enabled) {
        copy_cstr(trace.error, "hard memory archival is disabled or missing endpoint/auth configuration");
        trace.request_completed_us = ggml_time_us();
        *io_last_archive = trace;
        return true;
    }

    if (!input_items || item_count <= 0) {
        copy_cstr(trace.error, "hard memory archival requires at least one memory item");
        trace.request_completed_us = ggml_time_us();
        *io_last_archive = trace;
        return true;
    }

    if (tag.empty()) {
        copy_cstr(trace.error, "hard memory container tag is empty");
        trace.request_completed_us = ggml_time_us();
        *io_last_archive = trace;
        return true;
    }

    std::vector<llama_hard_memory_write_item> items;
    items.reserve((size_t) std::min<int32_t>(item_count, LLAMA_HARD_MEMORY_MAX_PRIMITIVES));
    for (int32_t i = 0; i < item_count; ++i) {
        llama_hard_memory_write_item item = input_items[i];
        if (!validate_write_item(&item)) {
            continue;
        }
        items.push_back(item);
    }
    if (items.empty()) {
        copy_cstr(trace.error, "hard memory archival rejected all memory items");
        trace.request_completed_us = ggml_time_us();
        *io_last_archive = trace;
        return true;
    }

    std::sort(items.begin(), items.end(), [](const llama_hard_memory_write_item & lhs, const llama_hard_memory_write_item & rhs) {
        if (lhs.primitive.importance == rhs.primitive.importance) {
            return std::strcmp(lhs.primitive.key, rhs.primitive.key) < 0;
        }
        return lhs.primitive.importance > rhs.primitive.importance;
    });
    if ((int32_t) items.size() > LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
        items.resize(LLAMA_HARD_MEMORY_MAX_PRIMITIVES);
    }

    trace.attempted = true;
    trace.primitive_count = (int32_t) items.size();
    const std::string identity = read_cstr(config.runtime_identity).empty() ? "vicuna" : read_cstr(config.runtime_identity);

    std::string digest_input = identity + "|" + tag;
    json memories = json::array();
    for (size_t i = 0; i < items.size(); ++i) {
        const auto & item = items[i];
        const auto & primitive = item.primitive;
        digest_input += "|" + read_cstr(primitive.key) + "|" + read_cstr(primitive.title) + "|" + read_cstr(primitive.content);
        fill_primitive_summary(trace.primitives[i], primitive);
        if (i == 0) {
            copy_cstr(trace.content_excerpt, trim_text(read_cstr(primitive.content), LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1));
        }
        memories.push_back({
            {"content", trim_text(read_cstr(primitive.content), 1024)},
            {"isStatic", item.is_static},
            {"metadata", primitive_metadata_json(primitive, identity, delta_summary)},
        });
    }

    char custom_id[LLAMA_HARD_MEMORY_MAX_ID_CHARS] = {};
    std::snprintf(custom_id, sizeof(custom_id), "vicuna-%016llx",
            (unsigned long long) fnv1a64(digest_input));
    copy_cstr(trace.custom_id, custom_id);

    json body = {
        {"containerTag", tag},
        {"memories", memories},
    };

    httplib::Client cli(read_cstr(config.base_url));
    cli.set_connection_timeout(std::chrono::milliseconds(config.timeout_ms));
    cli.set_read_timeout(std::chrono::milliseconds(config.timeout_ms));
    cli.set_write_timeout(std::chrono::milliseconds(config.timeout_ms));

    const auto response = cli.Post("/v4/memories", build_headers(config), body.dump(), "application/json");
    trace.request_completed_us = ggml_time_us();
    if (!response) {
        copy_cstr(trace.error, "hard memory archival request failed");
        *io_last_archive = trace;
        return true;
    }

    trace.status_code = response->status;
    if (response->status < 200 || response->status >= 300) {
        copy_cstr(trace.error, trim_text(response->body, sizeof(trace.error) - 1));
        *io_last_archive = trace;
        return true;
    }

    trace.archived = true;
    *io_last_archive = trace;
    return true;
}

static void accumulate_retrieval_summary(
        llama_hard_memory_retrieval_summary & summary,
        const llama_hard_memory_hit & hit) {
    const float weighted_similarity = clamp_unit(hit.similarity) * std::max(0.05f, hit.importance);
    switch (hit.kind) {
        case LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT:      ++summary.event_count; break;
        case LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY:          ++summary.trajectory_count; break;
        case LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME:             ++summary.outcome_count; break;
        case LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_OBSERVATION:    ++summary.tool_observation_count; break;
        case LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL:          ++summary.user_model_count; break;
        case LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT: ++summary.self_model_count; break;
        default: break;
    }

    summary.mean_similarity += clamp_unit(hit.similarity);
    summary.max_similarity = std::max(summary.max_similarity, clamp_unit(hit.similarity));
    summary.importance_signal += weighted_similarity;
    summary.confidence_signal += clamp_unit(hit.similarity) * std::max(0.05f, hit.confidence);
    summary.gain_support += clamp_unit(hit.similarity) * hit.gain_bias;
    summary.allostatic_support += clamp_unit(hit.similarity) * hit.allostatic_relevance;

    float * domain_support = nullptr;
    switch (hit.domain) {
        case LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS:    domain_support = &summary.goal_support; break;
        case LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME:     domain_support = &summary.user_support; break;
        case LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC:        domain_support = &summary.epistemic_support; break;
        case LLAMA_HARD_MEMORY_DOMAIN_EFFICIENCY:       domain_support = &summary.efficiency_support; break;
        case LLAMA_HARD_MEMORY_DOMAIN_RECOVERY:         domain_support = &summary.recovery_support; break;
        case LLAMA_HARD_MEMORY_DOMAIN_STRATEGY:         domain_support = &summary.strategy_support; break;
        case LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT: domain_support = &summary.self_improvement_support; break;
        default: break;
    }
    if (domain_support) {
        *domain_support = clamp_unit(*domain_support + weighted_similarity);
    }

    for (size_t i = 0; i < LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS; ++i) {
        if (std::strcmp(hit.tags[i], "preference") == 0) {
            summary.user_preference_support = clamp_unit(summary.user_preference_support + weighted_similarity);
        }
        if (std::strcmp(hit.tags[i], "rhetoric") == 0) {
            summary.user_rhetorical_support = clamp_unit(summary.user_rhetorical_support + weighted_similarity);
        }
    }
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

bool llama_hard_memory::set_request(const llama_cognitive_hard_memory_request & request) {
    if (request.command_id <= 0 || request.tool_job_id <= 0) {
        return false;
    }

    llama_cognitive_hard_memory_request normalized = request;
    normalized.operation =
            request.operation == LLAMA_COG_HARD_MEMORY_OPERATION_WRITE ?
                    LLAMA_COG_HARD_MEMORY_OPERATION_WRITE :
                    LLAMA_COG_HARD_MEMORY_OPERATION_QUERY;
    normalized.query.query[LLAMA_HARD_MEMORY_QUERY_MAX_CHARS - 1] = '\0';
    normalized.query.container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS - 1] = '\0';
    normalized.container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS - 1] = '\0';
    normalized.write_count = std::max<int32_t>(0, std::min<int32_t>(request.write_count, LLAMA_HARD_MEMORY_MAX_PRIMITIVES));
    for (int32_t i = 0; i < LLAMA_HARD_MEMORY_MAX_PRIMITIVES; ++i) {
        normalized.write_items[i].primitive.key[LLAMA_HARD_MEMORY_MAX_ID_CHARS - 1] = '\0';
        normalized.write_items[i].primitive.title[LLAMA_HARD_MEMORY_MAX_TITLE_CHARS - 1] = '\0';
        normalized.write_items[i].primitive.content[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1] = '\0';
        for (int32_t tag_idx = 0; tag_idx < LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS; ++tag_idx) {
            normalized.write_items[i].primitive.tags[tag_idx][LLAMA_HARD_MEMORY_MAX_TAG_CHARS - 1] = '\0';
        }
    }

    const int32_t index = find_request_index(requests, request_count, request.command_id);
    if (index >= 0) {
        requests[index] = normalized;
        return true;
    }

    if (request_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
        return false;
    }

    requests[request_count] = normalized;
    ++request_count;
    return true;
}

bool llama_hard_memory::get_request(int32_t command_id, llama_cognitive_hard_memory_request * out_request) const {
    if (!out_request) {
        return false;
    }

    const int32_t index = find_request_index(requests, request_count, command_id);
    if (index < 0) {
        return false;
    }

    *out_request = requests[index];
    return true;
}

bool llama_hard_memory::clear_request(int32_t command_id) {
    const int32_t index = find_request_index(requests, request_count, command_id);
    if (index < 0) {
        return false;
    }

    for (int32_t i = index + 1; i < request_count; ++i) {
        requests[i - 1] = requests[i];
    }
    requests[request_count - 1] = {};
    --request_count;
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
            const json metadata = item.value("metadata", json::object());
            dst.kind = parse_primitive_kind(metadata.contains("kind") ? metadata.at("kind") : json());
            dst.domain = parse_domain(metadata.contains("domain") ? metadata.at("domain") : json());
            dst.source_role = metadata.value("sourceRole", LLAMA_SELF_STATE_EVENT_SYSTEM);
            dst.source_channel = metadata.value("sourceChannel", LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY);
            dst.source_tool_kind = metadata.value("sourceToolKind", LLAMA_TOOL_KIND_NONE);
            dst.flags = metadata.value("flags", 0u);
            dst.importance = clamp_unit(metadata.value("importance", clamp_unit(dst.similarity)));
            dst.confidence = clamp_unit(metadata.value("confidence", clamp_unit(dst.similarity)));
            dst.gain_bias = clamp_unit(metadata.value("gainBias", clamp_unit(dst.similarity)));
            dst.allostatic_relevance = clamp_unit(metadata.value("allostaticRelevance", 0.0f));
            copy_cstr(dst.id, item.value("id", std::string()));
            copy_cstr(dst.title, item.value("title", metadata.value("title", std::string())));
            copy_cstr(dst.content, trim_text(item.value("memory", item.value("chunk", item.value("content", std::string()))), LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1));
            copy_tags(dst.tags, metadata.value("tags", item.value("tags", json::array())));
            accumulate_retrieval_summary(result.retrieval_summary, dst);
        }
        if (result.result_count > 0) {
            result.retrieval_summary.mean_similarity = clamp_unit(result.retrieval_summary.mean_similarity / (float) result.result_count);
            result.retrieval_summary.importance_signal = clamp_unit(result.retrieval_summary.importance_signal / (float) result.result_count);
            result.retrieval_summary.confidence_signal = clamp_unit(result.retrieval_summary.confidence_signal / (float) result.result_count);
            result.retrieval_summary.gain_support = clamp_unit(result.retrieval_summary.gain_support / (float) result.result_count);
            result.retrieval_summary.allostatic_support = clamp_unit(result.retrieval_summary.allostatic_support / (float) result.result_count);
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

bool llama_hard_memory::submit_result(const llama_hard_memory_result & result) {
    last_result = result;
    last_result.effective_container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS - 1] = '\0';
    last_result.profile_static[LLAMA_HARD_MEMORY_MAX_PROFILE_CHARS - 1] = '\0';
    last_result.profile_dynamic[LLAMA_HARD_MEMORY_MAX_PROFILE_CHARS - 1] = '\0';
    last_result.error[LLAMA_HARD_MEMORY_MAX_ERROR_CHARS - 1] = '\0';
    return true;
}

bool llama_hard_memory::submit_archive_trace(const llama_hard_memory_archive_trace & trace) {
    last_archive = trace;
    last_archive.custom_id[LLAMA_HARD_MEMORY_MAX_ID_CHARS - 1] = '\0';
    last_archive.container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS - 1] = '\0';
    last_archive.content_excerpt[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1] = '\0';
    last_archive.error[LLAMA_HARD_MEMORY_MAX_ERROR_CHARS - 1] = '\0';
    for (int32_t i = 0; i < LLAMA_HARD_MEMORY_MAX_PRIMITIVES; ++i) {
        last_archive.primitives[i].key[LLAMA_HARD_MEMORY_MAX_ID_CHARS - 1] = '\0';
        last_archive.primitives[i].title[LLAMA_HARD_MEMORY_MAX_TITLE_CHARS - 1] = '\0';
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

bool llama_hard_memory::archive_primitives(
        const llama_hard_memory_primitive * input_primitives,
        int32_t primitive_count,
        const llama_self_state_delta_summary * delta_summary) {
    if (!input_primitives || primitive_count <= 0) {
        return archive_write_items(nullptr, 0, nullptr, delta_summary);
    }

    std::vector<llama_hard_memory_write_item> items((size_t) std::min<int32_t>(primitive_count, LLAMA_HARD_MEMORY_MAX_PRIMITIVES));
    for (int32_t i = 0; i < primitive_count && i < LLAMA_HARD_MEMORY_MAX_PRIMITIVES; ++i) {
        items[(size_t) i].is_static = false;
        items[(size_t) i].primitive = input_primitives[i];
    }
    return archive_write_items(items.data(), std::min<int32_t>(primitive_count, LLAMA_HARD_MEMORY_MAX_PRIMITIVES), nullptr, delta_summary);
}

bool llama_hard_memory::archive_write_items(
        const llama_hard_memory_write_item * items,
        int32_t item_count,
        const char * container_override,
        const llama_self_state_delta_summary * delta_summary) {
    return archive_write_items_impl(config, items, item_count, container_override, delta_summary, &last_archive);
}

bool llama_hard_memory::archive_event(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        const llama_self_state_delta_summary & delta) {
    const std::string text = event_text(vocab, event);
    if (text.empty()) {
        llama_hard_memory_archive_trace trace = {};
        init_archive(trace);
        copy_cstr(trace.error, "hard memory archival requires non-empty event text");
        trace.request_completed_us = ggml_time_us();
        last_archive = trace;
        return true;
    }

    llama_hard_memory_primitive primitive = llama_hard_memory_default_primitive();
    primitive.kind = LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT;
    primitive.domain = LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
    primitive.source_role = event.role;
    primitive.source_channel = event.channel;
    primitive.source_tool_kind = event.role == LLAMA_SELF_STATE_EVENT_TOOL ? LLAMA_TOOL_KIND_GENERIC : LLAMA_TOOL_KIND_NONE;
    primitive.transaction_id = (int32_t) fnv1a64(text);
    primitive.importance = clamp_unit(0.25f + 0.35f * clamp_unit(delta.total_delta / 2.0f) + 0.40f * delta.max_delta);
    primitive.confidence = clamp_unit(0.55f + 0.30f * delta.max_delta);
    primitive.gain_bias = clamp_unit(0.20f + 0.60f * delta.max_delta);
    primitive.allostatic_relevance = 0.0f;
    copy_cstr(primitive.key, "event");
    copy_cstr(primitive.title, "self-state event");

    std::ostringstream content;
    content.setf(std::ios::fixed);
    content.precision(3);
    content << "vicuna self-state perturbation event\n"
            << "role: " << event.role << "\n"
            << "channel: " << event.channel << "\n"
            << "flags: " << event.flags << "\n"
            << "total_delta: " << delta.total_delta << "\n"
            << "max_delta: " << delta.max_delta << "\n"
            << "changed_registers: " << register_delta_summary(delta) << "\n"
            << "message: " << text;
    copy_cstr(primitive.content, trim_text(content.str(), LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1));
    copy_cstr(primitive.tags[0], "event");
    copy_cstr(primitive.tags[1], event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ? "counterfactual" : "primary");

    return archive_primitives(&primitive, 1, &delta);
}

bool llama_hard_memory::get_last_archive_trace(llama_hard_memory_archive_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = last_archive;
    return true;
}

llama_hard_memory_primitive llama_hard_memory_default_primitive(void) {
    llama_hard_memory_primitive primitive = {};
    primitive.kind = LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT;
    primitive.domain = LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
    primitive.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
    primitive.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
    primitive.source_tool_kind = LLAMA_TOOL_KIND_NONE;
    primitive.transaction_id = -1;
    primitive.flags = LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_GAIN;
    primitive.importance = 0.5f;
    primitive.confidence = 0.5f;
    primitive.gain_bias = 0.5f;
    primitive.allostatic_relevance = 0.0f;
    return primitive;
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
