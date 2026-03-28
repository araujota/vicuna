#include "server-emotive-runtime.h"
#include "server-runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

namespace {

static float clamp_unit(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

static float clamp_signed(float value) {
    return std::min(1.0f, std::max(-1.0f, value));
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

static bool env_to_bool(const char * name, bool default_value) {
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

static float env_to_float(const char * name, float default_value, float min_value, float max_value) {
    if (const char * value = std::getenv(name)) {
        const float parsed = std::strtof(value, nullptr);
        if (parsed >= min_value && parsed <= max_value) {
            return parsed;
        }
    }
    return default_value;
}

static float sigmoid_unit(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

static float centered_unit(float value) {
    return clamp_signed(2.0f * clamp_unit(value) - 1.0f);
}

static std::string trim_copy_local(const char * value) {
    if (!value) {
        return std::string();
    }
    std::string parsed = value;
    auto first = std::find_if_not(parsed.begin(), parsed.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    auto last = std::find_if_not(parsed.rbegin(), parsed.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return std::string();
    }
    return std::string(first, last);
}

static std::string default_heuristic_memory_path() {
    if (const char * value = std::getenv("VICUNA_HOST_SHELL_ROOT")) {
        const std::string parsed = trim_copy_local(value);
        if (!parsed.empty()) {
            return parsed + "/heuristics/vicuna-heuristic-memory.json";
        }
    }
    if (std::filesystem::exists("/home/vicuna/home")) {
        return "/home/vicuna/home/heuristics/vicuna-heuristic-memory.json";
    }
    return "vicuna-heuristic-memory.json";
}

static std::string kind_label(server_emotive_block_kind kind) {
    switch (kind) {
        case SERVER_EMOTIVE_BLOCK_USER_MESSAGE: return "user";
        case SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING: return "reasoning";
        case SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT: return "assistant";
        case SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT: return "runtime";
    }
    return "unknown";
}

static std::string kind_name(server_emotive_block_kind kind) {
    switch (kind) {
        case SERVER_EMOTIVE_BLOCK_USER_MESSAGE: return "user_message";
        case SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING: return "assistant_reasoning";
        case SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT: return "assistant_content";
        case SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT: return "runtime_event";
    }
    return "runtime_event";
}

static std::string normalize_text_copy(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        const unsigned char uch = (unsigned char) ch;
        if (std::isalnum(uch) || std::isspace(uch)) {
            out.push_back((char) std::tolower(uch));
        } else {
            out.push_back(' ');
        }
    }
    return out;
}

static std::vector<std::string> split_words(const std::string & text) {
    std::istringstream iss(normalize_text_copy(text));
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

static float ratio_of_matches(const std::vector<std::string> & words, const char * const * patterns, size_t count) {
    if (words.empty()) {
        return 0.0f;
    }

    float hits = 0.0f;
    for (const std::string & word : words) {
        for (size_t i = 0; i < count; ++i) {
            if (word == patterns[i]) {
                hits += 1.0f;
                break;
            }
        }
    }
    return clamp_unit(hits / (float) words.size());
}

static float unique_ratio(const std::vector<std::string> & words) {
    if (words.empty()) {
        return 0.0f;
    }
    std::vector<std::string> sorted = words;
    std::sort(sorted.begin(), sorted.end());
    int32_t unique = 0;
    std::string previous;
    for (const std::string & word : sorted) {
        if (word != previous) {
            ++unique;
            previous = word;
        }
    }
    return clamp_unit((float) unique / (float) words.size());
}

static float lexical_overlap(const std::vector<std::string> & lhs, const std::vector<std::string> & rhs) {
    if (lhs.empty() || rhs.empty()) {
        return 0.0f;
    }
    int32_t matches = 0;
    for (const std::string & word : lhs) {
        if (std::find(rhs.begin(), rhs.end(), word) != rhs.end()) {
            ++matches;
        }
    }
    return clamp_unit((float) matches / (float) lhs.size());
}

static std::string join_with_underscore(const std::vector<std::string> & values) {
    if (values.empty()) {
        return "neutral_grounded_balanced";
    }

    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out.push_back('_');
        }
        out += values[i];
    }
    return out;
}

static void append_unique(std::vector<std::string> & values, const std::string & value, size_t max_count) {
    if (value.empty() || values.size() >= max_count) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

static server_emotive_delta compute_delta(const server_emotive_vector & current, const server_emotive_vector * previous) {
    server_emotive_delta delta = {};
    if (!previous) {
        return delta;
    }

    delta.d_epistemic_pressure = clamp_signed(current.epistemic_pressure - previous->epistemic_pressure);
    delta.d_confidence = clamp_signed(current.confidence - previous->confidence);
    delta.d_contradiction_pressure = clamp_signed(current.contradiction_pressure - previous->contradiction_pressure);
    delta.d_planning_clarity = clamp_signed(current.planning_clarity - previous->planning_clarity);
    delta.d_curiosity = clamp_signed(current.curiosity - previous->curiosity);
    delta.d_caution = clamp_signed(current.caution - previous->caution);
    delta.d_frustration = clamp_signed(current.frustration - previous->frustration);
    delta.d_satisfaction = clamp_signed(current.satisfaction - previous->satisfaction);
    delta.d_momentum = clamp_signed(current.momentum - previous->momentum);
    delta.d_stall = clamp_signed(current.stall - previous->stall);
    delta.d_semantic_novelty = clamp_signed(current.semantic_novelty - previous->semantic_novelty);
    delta.d_user_alignment = clamp_signed(current.user_alignment - previous->user_alignment);
    delta.d_runtime_trust = clamp_signed(current.runtime_trust - previous->runtime_trust);
    delta.d_runtime_failure_pressure = clamp_signed(current.runtime_failure_pressure - previous->runtime_failure_pressure);

    const float negative_sum =
            std::max(0.0f, -delta.d_confidence) +
            std::max(0.0f, delta.d_contradiction_pressure) +
            std::max(0.0f, -delta.d_planning_clarity) +
            std::max(0.0f, delta.d_frustration) +
            std::max(0.0f, -delta.d_satisfaction) +
            std::max(0.0f, delta.d_stall) +
            std::max(0.0f, delta.d_runtime_failure_pressure);
    delta.negative_mass = clamp_unit(negative_sum / 7.0f);
    return delta;
}

static server_emotive_vad_trend compute_vad_trend(const server_emotive_vad & current, const server_emotive_vad * previous) {
    server_emotive_vad_trend trend = {};
    if (!previous) {
        return trend;
    }

    trend.d_valence = clamp_signed(current.valence - previous->valence);
    trend.d_arousal = clamp_signed(current.arousal - previous->arousal);
    trend.d_dominance = clamp_signed(current.dominance - previous->dominance);
    return trend;
}

static std::vector<std::string> select_top_dimensions(
        std::vector<std::pair<std::string, float>> dimension_scores,
        size_t max_count) {
    std::sort(
            dimension_scores.begin(),
            dimension_scores.end(),
            [](const std::pair<std::string, float> & lhs, const std::pair<std::string, float> & rhs) {
                if (lhs.second == rhs.second) {
                    return lhs.first < rhs.first;
                }
                return lhs.second > rhs.second;
            });

    std::vector<std::string> names;
    for (const auto & item : dimension_scores) {
        if (item.second <= 0.0f) {
            continue;
        }
        names.push_back(item.first);
        if (names.size() >= max_count) {
            break;
        }
    }
    return names;
}

static std::vector<std::string> build_vad_labels(const server_emotive_vad & vad) {
    std::vector<std::string> labels;
    append_unique(labels, vad.valence >= 0.30f ? "warm" : (vad.valence <= -0.30f ? "strained" : "neutral"), 3);
    append_unique(labels, vad.arousal >= 0.70f ? "reactive" : (vad.arousal <= 0.30f ? "calm" : "engaged"), 3);
    append_unique(labels, vad.dominance >= 0.30f ? "assertive" : (vad.dominance <= -0.30f ? "tentative" : "balanced"), 3);
    return labels;
}

static server_emotive_style_guide build_style_guide(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad) {
    server_emotive_style_guide style = {};
    const float valence_norm = clamp_unit((vad.valence + 1.0f) * 0.5f);
    const float dominance_norm = clamp_unit((vad.dominance + 1.0f) * 0.5f);

    style.warmth = clamp_unit(
            0.55f * valence_norm +
            0.20f * moment.user_alignment +
            0.15f * moment.satisfaction +
            0.10f * (1.0f - moment.frustration));
    style.energy = clamp_unit(
            0.70f * vad.arousal +
            0.20f * moment.momentum +
            0.10f * moment.curiosity);
    style.assertiveness = clamp_unit(
            0.75f * dominance_norm +
            0.15f * moment.planning_clarity +
            0.10f * moment.confidence);
    style.empathy = clamp_unit(
            0.35f * moment.user_alignment +
            0.25f * moment.caution +
            0.20f * valence_norm +
            0.20f * (1.0f - moment.runtime_failure_pressure));
    style.hedging = clamp_unit(
            0.32f * moment.epistemic_pressure +
            0.24f * moment.caution +
            0.24f * (1.0f - dominance_norm) +
            0.20f * moment.contradiction_pressure);
    style.directness = clamp_unit(
            0.35f * moment.planning_clarity +
            0.25f * dominance_norm +
            0.20f * moment.momentum +
            0.20f * moment.confidence -
            0.20f * style.hedging);

    style.tone_label = join_with_underscore(vad.labels);
    if (vad.valence <= -0.25f) {
        style.prompt_hints.push_back("Keep wording calm and non-escalatory.");
    } else if (vad.valence >= 0.25f) {
        style.prompt_hints.push_back("Keep the tone encouraging and affiliative.");
    }
    if (vad.arousal >= 0.70f) {
        style.prompt_hints.push_back("Use short stabilizing sentences and avoid digressions.");
    } else if (vad.arousal <= 0.30f) {
        style.prompt_hints.push_back("Maintain measured pacing and avoid forced urgency.");
    }
    if (vad.dominance >= 0.25f) {
        style.prompt_hints.push_back("Lead with clear decisions and concrete next steps.");
    } else if (vad.dominance <= -0.25f || style.hedging >= 0.60f) {
        style.prompt_hints.push_back("State uncertainty explicitly and separate observations from inferences.");
    }
    if (style.empathy >= 0.60f) {
        style.prompt_hints.push_back("Reference the user's goal before proposing actions.");
    }
    if (style.directness >= 0.65f && style.assertiveness >= 0.55f) {
        style.prompt_hints.push_back("Prefer direct declarative sentences over exploratory phrasing.");
    }
    if (style.prompt_hints.empty()) {
        style.prompt_hints.push_back("Maintain a neutral, grounded tone.");
    } else if (style.prompt_hints.size() > 4) {
        style.prompt_hints.resize(4);
    }

    return style;
}

static server_emotive_vad project_vad(
        const server_emotive_vector & moment,
        const server_emotive_vad * previous,
        float ema_alpha) {
    struct projector_term {
        const char * name;
        float value;
        float valence_weight;
        float arousal_weight;
        float dominance_weight;
    };

    const projector_term terms[] = {
        {"epistemic_pressure", centered_unit(moment.epistemic_pressure), -0.40f, +0.60f, -0.48f},
        {"confidence", centered_unit(moment.confidence), +0.55f, -0.12f, +0.78f},
        {"contradiction_pressure", centered_unit(moment.contradiction_pressure), -0.70f, +0.55f, -0.45f},
        {"planning_clarity", centered_unit(moment.planning_clarity), +0.35f, -0.18f, +0.74f},
        {"curiosity", centered_unit(moment.curiosity), +0.10f, +0.40f, +0.12f},
        {"caution", centered_unit(moment.caution), -0.18f, +0.32f, -0.24f},
        {"frustration", centered_unit(moment.frustration), -0.85f, +0.48f, -0.50f},
        {"satisfaction", centered_unit(moment.satisfaction), +0.90f, -0.10f, +0.26f},
        {"momentum", centered_unit(moment.momentum), +0.35f, +0.34f, +0.38f},
        {"stall", centered_unit(moment.stall), -0.55f, +0.22f, -0.76f},
        {"semantic_novelty", centered_unit(moment.semantic_novelty), +0.08f, +0.18f, +0.08f},
        {"user_alignment", centered_unit(moment.user_alignment), +0.55f, -0.10f, +0.30f},
        {"runtime_trust", centered_unit(moment.runtime_trust), +0.28f, -0.14f, +0.46f},
        {"runtime_failure_pressure", centered_unit(moment.runtime_failure_pressure), -0.65f, +0.58f, -0.70f},
    };

    float raw_valence = 0.0f;
    float raw_arousal = 0.0f;
    float raw_dominance = 0.0f;
    std::vector<std::pair<std::string, float>> dimension_scores;
    for (const projector_term & term : terms) {
        const float v_contrib = term.value * term.valence_weight;
        const float a_contrib = term.value * term.arousal_weight;
        const float d_contrib = term.value * term.dominance_weight;
        raw_valence += v_contrib;
        raw_arousal += a_contrib;
        raw_dominance += d_contrib;
        dimension_scores.push_back({
            term.name,
            std::fabs(v_contrib) + std::fabs(a_contrib) + std::fabs(d_contrib),
        });
    }

    raw_valence +=
            0.30f * centered_unit(moment.planning_clarity) * centered_unit(moment.confidence) -
            0.18f * centered_unit(moment.contradiction_pressure) * centered_unit(moment.frustration) -
            0.16f * moment.runtime_failure_pressure * (1.0f - moment.runtime_trust);
    raw_arousal +=
            0.22f * centered_unit(moment.epistemic_pressure) * centered_unit(moment.contradiction_pressure) +
            0.16f * centered_unit(moment.frustration) * centered_unit(moment.stall) -
            0.12f * centered_unit(moment.planning_clarity) * centered_unit(moment.confidence);
    raw_dominance +=
            0.24f * centered_unit(moment.planning_clarity) * centered_unit(moment.confidence) -
            0.18f * moment.runtime_failure_pressure * (1.0f - moment.runtime_trust) -
            0.16f * centered_unit(moment.epistemic_pressure) * centered_unit(moment.caution);

    server_emotive_vad vad = {};
    vad.valence = std::tanh(raw_valence);
    vad.arousal = sigmoid_unit(raw_arousal);
    vad.dominance = std::tanh(raw_dominance);

    if (previous) {
        const float alpha = clamp_unit(ema_alpha);
        vad.valence = clamp_signed(alpha * vad.valence + (1.0f - alpha) * previous->valence);
        vad.arousal = clamp_unit(alpha * vad.arousal + (1.0f - alpha) * previous->arousal);
        vad.dominance = clamp_signed(alpha * vad.dominance + (1.0f - alpha) * previous->dominance);
    }

    vad.trend = compute_vad_trend(vad, previous);
    vad.dominant_dimensions = select_top_dimensions(std::move(dimension_scores), 3);
    vad.labels = build_vad_labels(vad);
    vad.style_guide = build_style_guide(moment, vad);
    return vad;
}

static std::string trimmed_text(const std::string & text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return std::string();
    }
    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

static bool ends_block(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    const char last = text.back();
    return last == '.' || last == '!' || last == '?' || last == '\n';
}

static std::string make_trace_id() {
    return "emo_" + random_string();
}

static std::string make_cognitive_replay_entry_id() {
    return "creplay_" + random_string();
}

static std::string make_cognitive_replay_result_id() {
    return "creplay_result_" + random_string();
}

static std::string make_heuristic_record_id() {
    return "heuristic_record_" + random_string();
}

static std::string make_heuristic_object_id() {
    return "heuristic_" + random_string();
}

static std::string make_bad_path_object_id() {
    return "bad_path_object_" + random_string();
}

static int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static bool contains_any(const std::string & haystack, std::initializer_list<const char *> needles) {
    for (const char * needle : needles) {
        if (haystack.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static std::string cognitive_replay_status_name(server_cognitive_replay_status status) {
    switch (status) {
        case SERVER_COGNITIVE_REPLAY_OPEN: return "open";
        case SERVER_COGNITIVE_REPLAY_REVIEWING: return "reviewing";
        case SERVER_COGNITIVE_REPLAY_RESOLVED: return "resolved";
        case SERVER_COGNITIVE_REPLAY_DEFERRED: return "deferred";
    }
    return "open";
}

static float average_negative_mass(const server_emotive_trace & trace) {
    if (trace.blocks.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (const auto & block : trace.blocks) {
        sum += block.delta.negative_mass;
    }
    return clamp_unit(sum / (float) trace.blocks.size());
}

static float cognitive_replay_score(float valence, float dominance, float negative_mass) {
    return valence + dominance - negative_mass;
}

static void sort_and_dedupe_strings(std::vector<std::string> & values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

static void append_control_bias(
        std::vector<json> & biases,
        const std::string & heuristic_id,
        const std::string & target,
        float value,
        const std::string & rationale) {
    if (target.empty() || std::fabs(value) < 0.001f) {
        return;
    }
    biases.push_back({
        {"heuristic_id", heuristic_id},
        {"target", target},
        {"bias", std::max(-0.25f, std::min(0.25f, value))},
        {"rationale", rationale},
    });
}

static std::string join_strings(const std::vector<std::string> & values, const char * separator = "\n") {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << separator;
        }
        out << values[i];
    }
    return out.str();
}

static std::vector<json> derive_control_biases(const server_heuristic_object & heuristic) {
    const std::string normalized = normalize_text_copy(
            heuristic.semantic_trigger_text + "\n" +
            heuristic.failure_mode + "\n" +
            join_strings(heuristic.constraints) + "\n" +
            join_strings(heuristic.preferred_actions) + "\n" +
            join_strings(heuristic.action_ranking_rules) + "\n" +
            heuristic.mid_reasoning_correction + "\n" +
            join_strings(heuristic.applies_when));
    std::vector<json> biases;

    if (contains_any(normalized, {"validate", "verification", "verify", "confirm", "schema", "contract", "check"})) {
        append_control_bias(biases, heuristic.heuristic_id, "tool_light", 0.12f, "heuristic favors verification before committing");
        append_control_bias(biases, heuristic.heuristic_id, "tool_heavy", 0.08f, "heuristic favors verification before committing");
        append_control_bias(biases, heuristic.heuristic_id, "reasoning_score", 0.10f, "heuristic increases verification depth");
        append_control_bias(biases, heuristic.heuristic_id, "interrupt_score", 0.08f, "heuristic allows interruption when validation fails");
    }
    if (contains_any(normalized, {"retry", "replan", "stuck", "loop", "stall", "backtrack"})) {
        append_control_bias(biases, heuristic.heuristic_id, "reflective", 0.10f, "heuristic favors explicit replanning");
        append_control_bias(biases, heuristic.heuristic_id, "replan_pressure", 0.14f, "heuristic raises replanning pressure");
        append_control_bias(biases, heuristic.heuristic_id, "force_synthesis_gate", 0.08f, "heuristic forces synthesis instead of repeated looping");
    }
    if (contains_any(normalized, {"plan", "step", "order", "sequence", "first", "then"})) {
        append_control_bias(biases, heuristic.heuristic_id, "tool_parallelism_gate", -0.10f, "heuristic prefers sequential execution order");
        append_control_bias(biases, heuristic.heuristic_id, "reasoning_score", 0.06f, "heuristic emphasizes explicit planning");
    }
    if (contains_any(normalized, {"tool", "search", "lookup", "inspect"})) {
        append_control_bias(biases, heuristic.heuristic_id, "tool_aggression", 0.08f, "heuristic prefers external verification tools");
        append_control_bias(biases, heuristic.heuristic_id, "tool_light", 0.08f, "heuristic prefers tool-assisted execution");
    }
    if (contains_any(normalized, {"respond", "answer directly", "conclude", "finish"})) {
        append_control_bias(biases, heuristic.heuristic_id, "direct", 0.06f, "heuristic supports direct synthesis once checks complete");
    }
    return biases;
}

static std::vector<std::string> extract_struct_tags(const std::string & text, const std::string & kind) {
    std::vector<std::string> tags;
    const std::string normalized = normalize_text_copy(text);
    if (!kind.empty()) {
        tags.push_back(kind);
    }
    if (contains_any(normalized, {"tool", "function", "arguments"})) {
        tags.push_back("tool_context");
    }
    if (contains_any(normalized, {"error", "fail", "wrong", "timeout", "retry"})) {
        tags.push_back("runtime_failure");
    }
    if (contains_any(normalized, {"stuck", "cannot", "can't", "unsure", "unclear", "maybe"})) {
        tags.push_back("uncertainty_spike");
    }
    if (contains_any(normalized, {"validate", "confirm", "check", "schema", "contract"})) {
        tags.push_back("validation_step");
    }
    if (contains_any(normalized, {"plan", "next", "step", "first", "then"})) {
        tags.push_back("planning_step");
    }
    if (contains_any(normalized, {"resolved", "improves", "improve", "control", "restore"})) {
        tags.push_back("recovery_path");
    }
    sort_and_dedupe_strings(tags);
    return tags;
}

static std::string join_lines(const std::vector<std::string> & lines) {
    std::ostringstream out;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) {
            out << "\n";
        }
        out << lines[i];
    }
    return out.str();
}

static std::string join_block_lines(const std::vector<server_emotive_block_record> & blocks) {
    std::vector<std::string> lines;
    for (const auto & block : blocks) {
        lines.push_back(kind_name(block.kind) + std::string(": ") + block.text);
    }
    return join_lines(lines);
}

static std::string sanitize_json_payload(const std::string & text) {
    const std::string trimmed = trimmed_text(text);
    const size_t fence = trimmed.find("```");
    if (fence == std::string::npos) {
        return trimmed;
    }

    const size_t json_start = trimmed.find('{');
    const size_t json_end = trimmed.rfind('}');
    if (json_start == std::string::npos || json_end == std::string::npos || json_end < json_start) {
        return trimmed;
    }
    return trimmed.substr(json_start, json_end - json_start + 1);
}

static float lexical_text_similarity(const std::string & lhs_text, const std::string & rhs_text) {
    const auto lhs = split_words(lhs_text);
    const auto rhs = split_words(rhs_text);
    return lexical_overlap(lhs, rhs);
}

static float weighted_jaccard_score(const std::vector<std::string> & lhs, const std::vector<std::string> & rhs) {
    if (lhs.empty() || rhs.empty()) {
        return 0.0f;
    }

    int32_t matches = 0;
    for (const auto & value : lhs) {
        if (std::find(rhs.begin(), rhs.end(), value) != rhs.end()) {
            ++matches;
        }
    }

    const int32_t union_size = (int32_t) lhs.size() + (int32_t) rhs.size() - matches;
    if (union_size <= 0) {
        return 0.0f;
    }
    return clamp_unit((float) matches / (float) union_size);
}

static float normalized_emotive_similarity(
        const server_emotive_vector * current_moment,
        const server_emotive_vad * current_vad,
        const server_heuristic_bad_signature & bad_signature) {
    if (!current_vad) {
        return 0.0f;
    }

    const float current_negative_mass = current_moment ?
            clamp_unit(
                    0.30f * current_moment->frustration +
                    0.25f * current_moment->stall +
                    0.20f * current_moment->contradiction_pressure +
                    0.15f * current_moment->runtime_failure_pressure +
                    0.10f * (1.0f - current_moment->planning_clarity))
            : 0.0f;
    const float current_valence_norm = clamp_unit((current_vad->valence + 1.0f) * 0.5f);
    const float current_dominance_norm = clamp_unit((current_vad->dominance + 1.0f) * 0.5f);
    const float current_arousal = current_vad->arousal;

    const float stored_valence_norm = clamp_unit((bad_signature.valence + 1.0f) * 0.5f);
    const float stored_dominance_norm = clamp_unit((bad_signature.dominance + 1.0f) * 0.5f);

    const float l1 =
            std::fabs(current_negative_mass - bad_signature.negative_mass) +
            std::fabs(current_valence_norm - stored_valence_norm) +
            std::fabs(current_arousal - bad_signature.arousal) +
            std::fabs(current_dominance_norm - stored_dominance_norm);
    return clamp_unit(1.0f - (l1 / 4.0f));
}

struct replay_slice_measurement {
    bool valid = false;
    float negative_mass = 0.0f;
    float valence = 0.0f;
    float dominance = 0.0f;
    int32_t block_count = 0;
};

static bool is_assistant_generated_block(const server_emotive_block_record & block) {
    return block.kind == SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING ||
            block.kind == SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT;
}

static replay_slice_measurement measure_replay_generated_slice(const server_emotive_trace & trace) {
    replay_slice_measurement measurement = {};
    for (const auto & block : trace.blocks) {
        if (!is_assistant_generated_block(block)) {
            continue;
        }
        if (!measurement.valid) {
            measurement.valid = true;
            measurement.negative_mass = block.delta.negative_mass;
            measurement.valence = block.vad.valence;
            measurement.dominance = block.vad.dominance;
        } else {
            // Replay validation should score the generated path itself, not
            // the seeded prompts or trailing runtime bookkeeping.
            measurement.negative_mass = std::max(measurement.negative_mass, block.delta.negative_mass);
            measurement.valence = std::min(measurement.valence, block.vad.valence);
            measurement.dominance = std::min(measurement.dominance, block.vad.dominance);
        }
        measurement.block_count += 1;
    }

    if (measurement.valid) {
        return measurement;
    }

    measurement.valid = !trace.blocks.empty();
    measurement.negative_mass = average_negative_mass(trace);
    measurement.valence = trace.final_vad.valence;
    measurement.dominance = trace.final_vad.dominance;
    measurement.block_count = (int32_t) trace.blocks.size();
    return measurement;
}

static std::string summarize_blocks(const std::vector<server_emotive_block_record> & blocks, size_t max_chars) {
    std::string summary;
    for (const auto & block : blocks) {
        if (!summary.empty()) {
            summary += "\n";
        }
        summary += kind_label(block.kind);
        summary += ": ";
        summary += block.text;
        if (summary.size() >= max_chars) {
            summary.resize(max_chars);
            break;
        }
    }
    return summary;
}

} // namespace

server_emotive_runtime_config server_emotive_runtime_config_from_env() {
    server_emotive_runtime_config config;
    config.enabled = env_to_bool("VICUNA_EMOTIVE_ENABLED", true);
    config.block_max_chars = env_to_int("VICUNA_EMOTIVE_BLOCK_MAX_CHARS", config.block_max_chars);
    config.max_blocks_per_turn = env_to_int("VICUNA_EMOTIVE_MAX_BLOCKS_PER_TURN", config.max_blocks_per_turn);
    config.max_turn_history = env_to_int("VICUNA_EMOTIVE_MAX_TURN_HISTORY", config.max_turn_history);
    config.degraded_mode_allowed = env_to_bool("VICUNA_EMOTIVE_DEGRADED_MODE_ALLOWED", true);
    config.vad_ema_alpha = env_to_float("VICUNA_EMOTIVE_VAD_ALPHA", config.vad_ema_alpha, 0.05f, 1.0f);
    config.embedding = server_embedding_backend_config_from_env();
    config.cognitive_replay.enabled = env_to_bool("VICUNA_COGNITIVE_REPLAY_ENABLED", config.cognitive_replay.enabled);
    config.cognitive_replay.max_entries = env_to_int("VICUNA_COGNITIVE_REPLAY_MAX_ENTRIES", config.cognitive_replay.max_entries);
    config.cognitive_replay.max_results = env_to_int("VICUNA_COGNITIVE_REPLAY_MAX_RESULTS", config.cognitive_replay.max_results);
    config.cognitive_replay.neg_mass_threshold = env_to_float(
            "VICUNA_COGNITIVE_REPLAY_NEG_MASS_THRESHOLD",
            config.cognitive_replay.neg_mass_threshold,
            0.01f,
            1.0f);
    config.cognitive_replay.valence_drop_threshold = env_to_float(
            "VICUNA_COGNITIVE_REPLAY_VALENCE_DROP_THRESHOLD",
            config.cognitive_replay.valence_drop_threshold,
            0.01f,
            1.0f);
    config.cognitive_replay.dominance_drop_threshold = env_to_float(
            "VICUNA_COGNITIVE_REPLAY_DOMINANCE_DROP_THRESHOLD",
            config.cognitive_replay.dominance_drop_threshold,
            0.01f,
            1.0f);
    config.cognitive_replay.persistence_blocks = env_to_int(
            "VICUNA_COGNITIVE_REPLAY_PERSISTENCE_BLOCKS",
            config.cognitive_replay.persistence_blocks);
    config.cognitive_replay.context_before = env_to_int(
            "VICUNA_COGNITIVE_REPLAY_CONTEXT_BEFORE",
            config.cognitive_replay.context_before);
    config.cognitive_replay.context_after = env_to_int(
            "VICUNA_COGNITIVE_REPLAY_CONTEXT_AFTER",
            config.cognitive_replay.context_after);
    config.cognitive_replay.max_attempts = env_to_int(
            "VICUNA_COGNITIVE_REPLAY_MAX_ATTEMPTS",
            config.cognitive_replay.max_attempts);
    config.cognitive_replay.idle_after_ms = env_to_int(
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS",
            config.cognitive_replay.idle_after_ms);
    config.cognitive_replay.poll_interval_ms = env_to_int(
            "VICUNA_COGNITIVE_REPLAY_POLL_MS",
            config.cognitive_replay.poll_interval_ms);
    config.cognitive_replay.improvement_threshold = env_to_float(
            "VICUNA_COGNITIVE_REPLAY_IMPROVEMENT_THRESHOLD",
            config.cognitive_replay.improvement_threshold,
            0.01f,
            1.0f);
    config.heuristic_memory.enabled = env_to_bool("VICUNA_HEURISTIC_MEMORY_ENABLED", config.heuristic_memory.enabled);
    if (const char * value = std::getenv("VICUNA_HEURISTIC_MEMORY_PATH")) {
        config.heuristic_memory.path = value;
    } else {
        config.heuristic_memory.path = default_heuristic_memory_path();
    }
    config.heuristic_memory.max_records = env_to_int(
            "VICUNA_HEURISTIC_MEMORY_MAX_RECORDS",
            config.heuristic_memory.max_records);
    config.heuristic_memory.top_k_semantic = env_to_int(
            "VICUNA_HEURISTIC_MEMORY_TOP_K",
            config.heuristic_memory.top_k_semantic);
    config.heuristic_memory.semantic_threshold = env_to_float(
            "VICUNA_HEURISTIC_MEMORY_SEMANTIC_THRESHOLD",
            config.heuristic_memory.semantic_threshold,
            0.0f,
            1.0f);
    config.heuristic_memory.rerank_threshold = env_to_float(
            "VICUNA_HEURISTIC_MEMORY_RERANK_THRESHOLD",
            config.heuristic_memory.rerank_threshold,
            0.0f,
            1.0f);
    config.heuristic_memory.semantic_weight = env_to_float(
            "VICUNA_HEURISTIC_MEMORY_SEMANTIC_WEIGHT",
            config.heuristic_memory.semantic_weight,
            0.0f,
            1.0f);
    config.heuristic_memory.struct_weight = env_to_float(
            "VICUNA_HEURISTIC_MEMORY_STRUCT_WEIGHT",
            config.heuristic_memory.struct_weight,
            0.0f,
            1.0f);
    config.heuristic_memory.emotive_weight = env_to_float(
            "VICUNA_HEURISTIC_MEMORY_EMOTIVE_WEIGHT",
            config.heuristic_memory.emotive_weight,
            0.0f,
            1.0f);
    const float weight_sum =
            config.heuristic_memory.semantic_weight +
            config.heuristic_memory.struct_weight +
            config.heuristic_memory.emotive_weight;
    if (weight_sum > 0.0f) {
        config.heuristic_memory.semantic_weight /= weight_sum;
        config.heuristic_memory.struct_weight /= weight_sum;
        config.heuristic_memory.emotive_weight /= weight_sum;
    }
    return config;
}

static json vector_to_json(const server_emotive_vector & vector) {
    return {
        {"epistemic_pressure", vector.epistemic_pressure},
        {"confidence", vector.confidence},
        {"contradiction_pressure", vector.contradiction_pressure},
        {"planning_clarity", vector.planning_clarity},
        {"curiosity", vector.curiosity},
        {"caution", vector.caution},
        {"frustration", vector.frustration},
        {"satisfaction", vector.satisfaction},
        {"momentum", vector.momentum},
        {"stall", vector.stall},
        {"semantic_novelty", vector.semantic_novelty},
        {"user_alignment", vector.user_alignment},
        {"runtime_trust", vector.runtime_trust},
        {"runtime_failure_pressure", vector.runtime_failure_pressure},
    };
}

static json delta_to_json(const server_emotive_delta & delta) {
    return {
        {"d_epistemic_pressure", delta.d_epistemic_pressure},
        {"d_confidence", delta.d_confidence},
        {"d_contradiction_pressure", delta.d_contradiction_pressure},
        {"d_planning_clarity", delta.d_planning_clarity},
        {"d_curiosity", delta.d_curiosity},
        {"d_caution", delta.d_caution},
        {"d_frustration", delta.d_frustration},
        {"d_satisfaction", delta.d_satisfaction},
        {"d_momentum", delta.d_momentum},
        {"d_stall", delta.d_stall},
        {"d_semantic_novelty", delta.d_semantic_novelty},
        {"d_user_alignment", delta.d_user_alignment},
        {"d_runtime_trust", delta.d_runtime_trust},
        {"d_runtime_failure_pressure", delta.d_runtime_failure_pressure},
        {"negative_mass", delta.negative_mass},
    };
}

static json vad_to_json(const server_emotive_vad & vad) {
    json prompt_hints = json::array();
    for (const std::string & hint : vad.style_guide.prompt_hints) {
        prompt_hints.push_back(hint);
    }

    json labels = json::array();
    for (const std::string & label : vad.labels) {
        labels.push_back(label);
    }

    json dominant_dimensions = json::array();
    for (const std::string & dimension : vad.dominant_dimensions) {
        dominant_dimensions.push_back(dimension);
    }

    return {
        {"valence", vad.valence},
        {"arousal", vad.arousal},
        {"dominance", vad.dominance},
        {"trend", {
            {"d_valence", vad.trend.d_valence},
            {"d_arousal", vad.trend.d_arousal},
            {"d_dominance", vad.trend.d_dominance},
        }},
        {"labels", std::move(labels)},
        {"dominant_dimensions", std::move(dominant_dimensions)},
        {"style_guide", {
            {"tone_label", vad.style_guide.tone_label},
            {"warmth", vad.style_guide.warmth},
            {"energy", vad.style_guide.energy},
            {"assertiveness", vad.style_guide.assertiveness},
            {"empathy", vad.style_guide.empathy},
            {"hedging", vad.style_guide.hedging},
            {"directness", vad.style_guide.directness},
            {"prompt_hints", std::move(prompt_hints)},
        }},
    };
}

static json block_record_to_json(const server_emotive_block_record & block) {
    return {
        {"block_index", block.block_index},
        {"source", {
            {"kind", kind_name(block.kind)},
            {"label", kind_label(block.kind)},
        }},
        {"text", block.text},
        {"char_count", block.char_count},
        {"moment", vector_to_json(block.moment)},
        {"delta", delta_to_json(block.delta)},
        {"vad", vad_to_json(block.vad)},
        {"embedding_mode", block.embedding_mode},
        {"semantic_similarity_to_user", block.semantic_similarity_to_user},
        {"semantic_similarity_to_previous", block.semantic_similarity_to_previous},
    };
}

static json heuristic_object_to_json(const server_heuristic_object & heuristic) {
    return {
        {"heuristic_id", heuristic.heuristic_id},
        {"title", heuristic.title},
        {"trigger", {
            {"task_types", heuristic.task_types},
            {"tool_names", heuristic.tool_names},
            {"struct_tags", heuristic.struct_tags},
            {"emotive_conditions", heuristic.emotive_conditions},
            {"semantic_trigger_text", heuristic.semantic_trigger_text},
        }},
        {"diagnosis", {
            {"failure_mode", heuristic.failure_mode},
            {"evidence", heuristic.evidence},
        }},
        {"intervention", {
            {"constraints", heuristic.constraints},
            {"preferred_actions", heuristic.preferred_actions},
            {"action_ranking_rules", heuristic.action_ranking_rules},
            {"mid_reasoning_correction", heuristic.mid_reasoning_correction},
        }},
        {"scope", {
            {"applies_when", heuristic.applies_when},
            {"avoid_when", heuristic.avoid_when},
        }},
        {"confidence", {
            {"p_success", heuristic.p_success},
            {"calibration", heuristic.calibration},
            {"notes", heuristic.confidence_notes},
        }},
    };
}

static server_heuristic_object heuristic_object_from_json(const json & payload) {
    server_heuristic_object heuristic = {};
    heuristic.heuristic_id = json_value(payload, "heuristic_id", std::string(make_heuristic_object_id()));
    heuristic.title = json_value(payload, "title", std::string("Replay heuristic"));
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
    sort_and_dedupe_strings(heuristic.task_types);
    sort_and_dedupe_strings(heuristic.tool_names);
    sort_and_dedupe_strings(heuristic.struct_tags);
    return heuristic;
}

static json heuristic_bad_path_object_to_json(const server_heuristic_bad_path_object & object) {
    return {
        {"object_id", object.object_id},
        {"kind", object.kind},
        {"text", object.text},
        {"struct_tags", object.struct_tags},
        {"embedding", object.embedding},
    };
}

static server_heuristic_bad_path_object heuristic_bad_path_object_from_json(const json & payload) {
    server_heuristic_bad_path_object object = {};
    object.object_id = json_value(payload, "object_id", std::string(make_bad_path_object_id()));
    object.kind = json_value(payload, "kind", std::string());
    object.text = json_value(payload, "text", std::string());
    object.struct_tags = json_value(payload, "struct_tags", std::vector<std::string>());
    object.embedding = json_value(payload, "embedding", std::vector<float>());
    return object;
}

static json heuristic_bad_signature_to_json(const server_heuristic_bad_signature & signature) {
    return {
        {"negative_mass", signature.negative_mass},
        {"valence", signature.valence},
        {"arousal", signature.arousal},
        {"dominance", signature.dominance},
        {"struct_tags", signature.struct_tags},
    };
}

static server_heuristic_bad_signature heuristic_bad_signature_from_json(const json & payload) {
    server_heuristic_bad_signature signature = {};
    signature.negative_mass = json_value(payload, "negative_mass", 0.0f);
    signature.valence = json_value(payload, "valence", 0.0f);
    signature.arousal = json_value(payload, "arousal", 0.0f);
    signature.dominance = json_value(payload, "dominance", 0.0f);
    signature.struct_tags = json_value(payload, "struct_tags", std::vector<std::string>());
    return signature;
}

static json heuristic_memory_record_to_json(const server_heuristic_memory_record & record) {
    json objects = json::array();
    for (const auto & object : record.bad_path_objects) {
        objects.push_back(heuristic_bad_path_object_to_json(object));
    }
    return {
        {"record_id", record.record_id},
        {"entry_id", record.entry_id},
        {"result_id", record.result_id},
        {"source_trace_id", record.source_trace_id},
        {"created_at_ms", record.created_at_ms},
        {"bad_path_text", record.bad_path_text},
        {"better_path_reasoning_content", record.better_path_reasoning_content},
        {"better_path_content", record.better_path_content},
        {"bad_path_objects", std::move(objects)},
        {"heuristic", heuristic_object_to_json(record.heuristic)},
        {"bad_signature", heuristic_bad_signature_to_json(record.bad_signature)},
    };
}

static server_heuristic_memory_record heuristic_memory_record_from_json(const json & payload) {
    server_heuristic_memory_record record = {};
    record.record_id = json_value(payload, "record_id", std::string(make_heuristic_record_id()));
    record.entry_id = json_value(payload, "entry_id", std::string());
    record.result_id = json_value(payload, "result_id", std::string());
    record.source_trace_id = json_value(payload, "source_trace_id", std::string());
    record.created_at_ms = json_value(payload, "created_at_ms", int64_t(0));
    record.bad_path_text = json_value(payload, "bad_path_text", std::string());
    record.better_path_reasoning_content = json_value(payload, "better_path_reasoning_content", std::string());
    record.better_path_content = json_value(payload, "better_path_content", std::string());
    if (payload.contains("bad_path_objects") && payload.at("bad_path_objects").is_array()) {
        for (const auto & object : payload.at("bad_path_objects")) {
            record.bad_path_objects.push_back(heuristic_bad_path_object_from_json(object));
        }
    }
    record.heuristic = heuristic_object_from_json(json_value(payload, "heuristic", json::object()));
    record.bad_signature = heuristic_bad_signature_from_json(json_value(payload, "bad_signature", json::object()));
    return record;
}

static json heuristic_retrieval_decision_to_json(const server_heuristic_retrieval_decision & decision) {
    return {
        {"matched", decision.matched},
        {"record_id", decision.record_id},
        {"heuristic_id", decision.heuristic_id},
        {"query_text", decision.query_text},
        {"semantic_score", decision.semantic_score},
        {"struct_score", decision.struct_score},
        {"emotive_score", decision.emotive_score},
        {"total_score", decision.total_score},
        {"threshold", decision.threshold},
        {"created_at_ms", decision.created_at_ms},
        {"control_biases", decision.control_biases},
    };
}

static json metacognitive_policy_to_json(const server_metacognitive_policy_decision & decision) {
    return {
        {"valid", decision.valid},
        {"policy_version", decision.policy_version},
        {"selected_mode", decision.selected_mode},
        {"reasoning_depth", decision.reasoning_depth},
        {"thinking_mode", decision.thinking_mode},
        {"prefix_profile", decision.prefix_profile},
        {"stop_profile", decision.stop_profile},
        {"sampling_profile", decision.sampling_profile},
        {"repetition_profile", decision.repetition_profile},
        {"tool_choice_profile", decision.tool_choice_profile},
        {"score_breakdown", {
            {"direct", decision.direct_score},
            {"reflective", decision.reflective_score},
            {"tool_light", decision.tool_light_score},
            {"tool_heavy", decision.tool_heavy_score},
            {"background_defer", decision.background_defer_score},
        }},
        {"reasoning_score", decision.reasoning_score},
        {"tool_aggression", decision.tool_aggression},
        {"interrupt_score", decision.interrupt_score},
        {"tool_parallelism_cap", decision.tool_parallelism_cap},
        {"interrupt_allowed", decision.interrupt_allowed},
        {"replan_required", decision.replan_required},
        {"early_stop_ok", decision.early_stop_ok},
        {"force_synthesis", decision.force_synthesis},
        {"heuristic_biases", decision.heuristic_biases},
        {"prompt_hints", decision.prompt_hints},
    };
}

json server_policy_observation_to_json(const server_policy_observation & observation) {
    return {
        {"schema_version", observation.schema_version},
        {"request_id", observation.request_id},
        {"trace_id", observation.trace_id.empty() ? json(nullptr) : json(observation.trace_id)},
        {"decision_id", observation.decision_id},
        {"mode_label", observation.mode_label},
        {"bridge_scoped", observation.bridge_scoped},
        {"cognitive_replay", observation.cognitive_replay},
        {"ongoing_task_due", observation.ongoing_task_due},
        {"moment", vector_to_json(observation.moment)},
        {"vad", vad_to_json(observation.vad)},
        {"heuristic", {
            {"matched", observation.heuristic_matched},
            {"heuristic_id", observation.heuristic_id.empty() ? json(nullptr) : json(observation.heuristic_id)},
        }},
        {"tool_context", {
            {"available_tool_count", observation.available_tool_count},
            {"parallel_tool_calls_requested", observation.parallel_tool_calls_requested},
        }},
        {"recent_runtime", {
            {"input_message_count", observation.input_message_count},
        }},
    };
}

json server_policy_action_to_json(const server_policy_action & action) {
    return {
        {"schema_version", action.schema_version},
        {"policy_version", action.policy_version},
        {"selected_mode", action.selected_mode},
        {"reasoning_depth", action.reasoning_depth},
        {"token_budget_bucket", action.token_budget_bucket},
        {"tool_parallelism_cap", action.tool_parallelism_cap},
        {"interrupt_allowed", action.interrupt_allowed},
        {"replan_required", action.replan_required},
        {"early_stop_ok", action.early_stop_ok},
        {"force_synthesis", action.force_synthesis},
        {"thinking_mode", action.thinking_mode},
        {"prefix_profile", action.prefix_profile},
        {"stop_profile", action.stop_profile},
        {"sampling_profile", action.sampling_profile},
        {"repetition_profile", action.repetition_profile},
        {"tool_choice_profile", action.tool_choice_profile},
        {"proposal_source", action.proposal_source},
    };
}

json server_policy_action_mask_to_json(const server_policy_action_mask & mask) {
    return {
        {"allowed_modes", mask.allowed_modes},
        {"allowed_reasoning_depths", mask.allowed_reasoning_depths},
        {"allowed_thinking_modes", mask.allowed_thinking_modes},
        {"allowed_prefix_profiles", mask.allowed_prefix_profiles},
        {"allowed_stop_profiles", mask.allowed_stop_profiles},
        {"allowed_sampling_profiles", mask.allowed_sampling_profiles},
        {"allowed_repetition_profiles", mask.allowed_repetition_profiles},
        {"allowed_tool_choice_profiles", mask.allowed_tool_choice_profiles},
        {"max_tool_parallelism_cap", mask.max_tool_parallelism_cap},
        {"allow_interrupt", mask.allow_interrupt},
        {"allow_replan", mask.allow_replan},
        {"allow_early_stop", mask.allow_early_stop},
        {"allow_force_synthesis", mask.allow_force_synthesis},
    };
}

json server_policy_applied_provider_controls_to_json(const server_policy_applied_provider_controls & controls) {
    return {
        {"thinking_enabled", controls.thinking_enabled},
        {"prefix_profile", controls.prefix_profile},
        {"stop_profile", controls.stop_profile},
        {"sampling_profile", controls.sampling_profile},
        {"repetition_profile", controls.repetition_profile},
        {"tool_choice_profile", controls.tool_choice_profile},
        {"prefix_used", controls.prefix_used},
        {"temperature", controls.temperature_present ? json(controls.temperature) : json(nullptr)},
        {"top_p", controls.top_p_present ? json(controls.top_p) : json(nullptr)},
        {"frequency_penalty", controls.frequency_penalty_present ? json(controls.frequency_penalty) : json(nullptr)},
        {"presence_penalty", controls.presence_penalty_present ? json(controls.presence_penalty) : json(nullptr)},
        {"tool_choice", controls.tool_choice.empty() ? json(nullptr) : json(controls.tool_choice)},
        {"stop_sequences", controls.stop_sequences},
        {"suppressed_fields", controls.suppressed_fields},
        {"defaulted_fields", controls.defaulted_fields},
        {"field_sources", controls.field_sources},
        {"beta_routing_reason", controls.beta_routing_reason.empty() ? json(nullptr) : json(controls.beta_routing_reason)},
    };
}

json server_policy_reward_event_to_json(const server_policy_reward_event & event) {
    return {
        {"kind", event.kind},
        {"value", event.value},
        {"weight", event.weight},
        {"source", event.source},
    };
}

json server_policy_vad_axes_to_json(const server_policy_vad_axes & axes) {
    return {
        {"valence", axes.valence},
        {"arousal", axes.arousal},
        {"dominance", axes.dominance},
    };
}

json server_policy_reward_model_to_json(const server_policy_reward_model & model) {
    return {
        {"schema_version", model.schema_version},
        {"model_version", model.model_version},
        {"target_moment", vector_to_json(model.target_moment)},
        {"target_vad", server_policy_vad_axes_to_json(model.target_vad)},
        {"moment_weights", vector_to_json(model.moment_weights)},
        {"vad_weights", server_policy_vad_axes_to_json(model.vad_weights)},
        {"progress_weight", model.progress_weight},
        {"terminal_closeness_weight", model.terminal_closeness_weight},
        {"shaping_gamma", model.shaping_gamma},
        {"completion_stop_reward", model.completion_stop_reward},
        {"completion_non_stop_reward", model.completion_non_stop_reward},
        {"latency_cost_cap", model.latency_cost_cap},
        {"latency_ms_scale", model.latency_ms_scale},
        {"token_cost_cap", model.token_cost_cap},
        {"token_scale", model.token_scale},
        {"tool_success_bonus", model.tool_success_bonus},
        {"candidate_failure_penalty", model.candidate_failure_penalty},
    };
}

json server_policy_reward_breakdown_to_json(const server_policy_reward_breakdown & breakdown) {
    return {
        {"schema_version", breakdown.schema_version},
        {"model_version", breakdown.model_version},
        {"before_score", breakdown.before_score},
        {"after_score", breakdown.after_score},
        {"progress_reward", breakdown.progress_reward},
        {"terminal_closeness_reward", breakdown.terminal_closeness_reward},
        {"completion_quality_reward", breakdown.completion_quality_reward},
        {"latency_cost", breakdown.latency_cost},
        {"token_cost", breakdown.token_cost},
        {"tool_success_reward", breakdown.tool_success_reward},
        {"candidate_failure_penalty", breakdown.candidate_failure_penalty},
        {"total", breakdown.total},
    };
}

json server_policy_safety_guard_result_to_json(const server_policy_safety_guard_result & result) {
    return {
        {"candidate_present", result.candidate_present},
        {"allowed", result.allowed},
        {"blocked_fields", result.blocked_fields},
        {"clipped_fields", result.clipped_fields},
        {"fallback_to_native", result.fallback_to_native},
        {"reason", result.reason},
    };
}

json server_policy_transition_to_json(const server_policy_transition & transition) {
    json reward_events = json::array();
    for (const auto & event : transition.reward_events) {
        reward_events.push_back(server_policy_reward_event_to_json(event));
    }

    return {
        {"transition_id", transition.transition_id},
        {"request_id", transition.request_id},
        {"decision_id", transition.decision_id},
        {"policy_mode", transition.policy_mode},
        {"rollout_mode", transition.rollout_mode},
        {"behavior_policy_version", transition.behavior_policy_version},
        {"candidate_policy_version", transition.candidate_policy_version.empty() ? json(nullptr) : json(transition.candidate_policy_version)},
        {"candidate_policy_alias", transition.candidate_policy_alias.empty() ? json(nullptr) : json(transition.candidate_policy_alias)},
        {"observation", server_policy_observation_to_json(transition.observation)},
        {"action_mask", server_policy_action_mask_to_json(transition.action_mask)},
        {"candidate_action", transition.has_candidate_action ? server_policy_action_to_json(transition.candidate_action) : json(nullptr)},
        {"executed_action", server_policy_action_to_json(transition.executed_action)},
        {"rollout_sampled", transition.rollout_sampled},
        {"candidate_executed_live", transition.candidate_executed_live},
        {"candidate_confidence", transition.candidate_confidence_present ? json(transition.candidate_confidence) : json(nullptr)},
        {"candidate_confidence_passed", transition.candidate_confidence_passed},
        {"rollout_decision_reason", transition.rollout_decision_reason.empty() ? json(nullptr) : json(transition.rollout_decision_reason)},
        {"canary_share_percent", transition.canary_share_percent},
        {"rollout_step_index", transition.rollout_step_index},
        {"safety_guard", server_policy_safety_guard_result_to_json(transition.safety_guard)},
        {"applied_provider_controls", server_policy_applied_provider_controls_to_json(transition.applied_provider_controls)},
        {"reward_model", server_policy_reward_model_to_json(transition.reward_model)},
        {"reward_events", std::move(reward_events)},
        {"reward_breakdown", server_policy_reward_breakdown_to_json(transition.reward_breakdown)},
        {"reward_total", transition.reward_total},
        {"next_observation", server_policy_observation_to_json(transition.next_observation)},
        {"terminated", transition.terminated},
        {"termination_reason", transition.termination_reason},
        {"latency_ms", transition.latency_ms},
        {"provider_finish_reason", transition.provider_finish_reason},
        {"created_at_ms", transition.created_at_ms},
    };
}

static bool save_heuristic_memory_records(
        const std::string & path,
        const std::deque<server_heuristic_memory_record> & records,
        std::string * out_error) {
    json payload = {
        {"object", "vicuna.emotive.heuristic_memory"},
        {"version", 1},
        {"records", json::array()},
    };
    for (const auto & record : records) {
        payload["records"].push_back(heuristic_memory_record_to_json(record));
    }

    std::error_code error;
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path(), error);
        if (error) {
            if (out_error) {
                *out_error = "failed to create heuristic memory directory";
            }
            return false;
        }
    }

    const std::string temp_path = path + ".tmp";
    {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            if (out_error) {
                *out_error = "failed to open heuristic memory temp file for write";
            }
            return false;
        }
        out << payload.dump(2);
        if (!out.good()) {
            if (out_error) {
                *out_error = "failed to write heuristic memory temp file";
            }
            return false;
        }
    }

    std::remove(path.c_str());
    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
        std::remove(temp_path.c_str());
        if (out_error) {
            *out_error = "failed to atomically replace heuristic memory file";
        }
        return false;
    }

    return true;
}

json server_emotive_trace_to_json(const server_emotive_trace & trace) {
    if (!trace.valid) {
        return nullptr;
    }

    json blocks = json::array();
    for (const server_emotive_block_record & block : trace.blocks) {
        blocks.push_back(block_record_to_json(block));
    }

    return {
        {"trace_id", trace.trace_id},
        {"model", trace.model},
        {"blocks", blocks},
        {"live_generation_start_block_index", trace.live_generation_start_block_index},
        {"final_moment", vector_to_json(trace.final_moment)},
        {"final_vad", vad_to_json(trace.final_vad)},
        {"embedding_mode", trace.embedding_mode},
        {"estimator_version", trace.estimator_version},
        {"provider_streamed", trace.provider_streamed},
        {"retained_block_count", trace.retained_block_count},
        {"mode", !trace.mode_label.empty() ? trace.mode_label : (trace.cognitive_replay ? "cognitive_replay" : "foreground")},
        {"cognitive_replay", trace.cognitive_replay},
        {"cognitive_replay_entry_id", trace.cognitive_replay_entry_id},
        {"suppress_replay_admission", trace.suppress_replay_admission},
        {"final_policy", trace.final_policy},
        {"heuristic_retrieval", trace.heuristic_retrieval},
    };
}

server_emotive_runtime::server_emotive_runtime(const server_emotive_runtime_config & config) :
        config_(config) {
    std::string error;
    embedding_backend_.configure(config.embedding, &error);
    load_heuristic_memory();
}

const server_emotive_runtime_config & server_emotive_runtime::config() const {
    return config_;
}

bool server_emotive_runtime::build_embedding(const std::string & text, std::vector<float> & out_embedding, std::string * out_mode) const {
    if (!config_.enabled) {
        if (out_mode) {
            *out_mode = "disabled";
        }
        return false;
    }

    if (embedding_backend_.ready()) {
        if (embedding_backend_.embed_text(text, out_embedding)) {
            if (out_mode) {
                *out_mode = embedding_backend_.mode_label();
            }
            return true;
        }
    }

    if (out_mode) {
        *out_mode = "lexical_only";
    }
    out_embedding.clear();
    return false;
}

server_emotive_block_record server_emotive_runtime::evaluate_block(
        server_emotive_block_kind kind,
        const std::string & text,
        int32_t block_index,
        const std::vector<float> & previous_embedding,
        const std::vector<float> & user_anchor_embedding,
        const server_emotive_vector * previous_moment,
        const server_emotive_vad * previous_vad) const {
    static const char * const uncertainty_terms[] = {"maybe", "perhaps", "might", "unclear", "unsure", "guess", "probably", "possibly"};
    static const char * const contradiction_terms[] = {"but", "however", "conflict", "inconsistent", "not", "never", "cannot", "can't", "wrong"};
    static const char * const planning_terms[] = {"plan", "first", "next", "then", "step", "should", "need", "will", "let", "check"};
    static const char * const resolution_terms[] = {"done", "resolved", "answer", "found", "therefore", "summary", "complete", "clear"};
    static const char * const negative_terms[] = {"fail", "error", "issue", "problem", "stuck", "cannot", "can't", "wrong", "retry"};
    static const char * const positive_terms[] = {"good", "clear", "ready", "complete", "resolved", "yes", "confident", "correct"};

    server_emotive_block_record record = {};
    record.block_index = block_index;
    record.kind = kind;
    record.text = trimmed_text(text);
    record.char_count = (int32_t) record.text.size();

    const std::vector<std::string> words = split_words(record.text);
    const float uncertainty_ratio = ratio_of_matches(words, uncertainty_terms, sizeof(uncertainty_terms)/sizeof(uncertainty_terms[0]));
    const float contradiction_ratio = ratio_of_matches(words, contradiction_terms, sizeof(contradiction_terms)/sizeof(contradiction_terms[0]));
    const float planning_ratio = ratio_of_matches(words, planning_terms, sizeof(planning_terms)/sizeof(planning_terms[0]));
    const float resolution_ratio = ratio_of_matches(words, resolution_terms, sizeof(resolution_terms)/sizeof(resolution_terms[0]));
    const float negative_ratio = ratio_of_matches(words, negative_terms, sizeof(negative_terms)/sizeof(negative_terms[0]));
    const float positive_ratio = ratio_of_matches(words, positive_terms, sizeof(positive_terms)/sizeof(positive_terms[0]));
    const float question_ratio = clamp_unit(record.text.find('?') != std::string::npos ? 1.0f : 0.0f);
    const float structure_ratio = clamp_unit(record.text.find('\n') != std::string::npos || record.text.find(':') != std::string::npos ? 1.0f : 0.0f);
    const float density = unique_ratio(words);

    std::vector<float> embedding;
    std::string embedding_mode = "lexical_only";
    const bool have_embedding = build_embedding(record.text, embedding, &embedding_mode);
    record.embedding_mode = embedding_mode;

    float sim_previous = 0.0f;
    if (have_embedding && !previous_embedding.empty() && previous_embedding.size() == embedding.size()) {
        sim_previous = clamp_unit((server_embd_similarity_cos(previous_embedding.data(), embedding.data(), (int) embedding.size()) + 1.0f) * 0.5f);
    }

    float sim_user = 0.0f;
    if (have_embedding && !user_anchor_embedding.empty() && user_anchor_embedding.size() == embedding.size()) {
        sim_user = clamp_unit((server_embd_similarity_cos(user_anchor_embedding.data(), embedding.data(), (int) embedding.size()) + 1.0f) * 0.5f);
    } else if (kind != SERVER_EMOTIVE_BLOCK_USER_MESSAGE && !user_anchor_embedding.empty()) {
        sim_user = 0.5f;
    }

    const float semantic_novelty = clamp_unit(previous_embedding.empty() ? 0.0f : 1.0f - sim_previous);
    const float lexical_repetition = clamp_unit(previous_embedding.empty() ? 0.0f : sim_previous);
    const float runtime_failure = clamp_unit(kind == SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT &&
            (record.text.find("error") != std::string::npos || record.text.find("fail") != std::string::npos || record.text.find("timeout") != std::string::npos) ? 1.0f : 0.0f);
    const float user_alignment = clamp_unit(sim_user);
    const float runtime_trust = clamp_unit(0.70f - 0.60f * runtime_failure + 0.10f * (kind == SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT ? 0.0f : 1.0f));
    const float runtime_failure_pressure = clamp_unit(runtime_failure);

    record.semantic_similarity_to_previous = sim_previous;
    record.semantic_similarity_to_user = sim_user;

    record.moment.epistemic_pressure = clamp_unit(
            0.35f * uncertainty_ratio +
            0.25f * question_ratio +
            0.20f * contradiction_ratio +
            0.20f * (1.0f - user_alignment));
    record.moment.confidence = clamp_unit(
            0.30f * planning_ratio +
            0.30f * resolution_ratio +
            0.25f * positive_ratio +
            0.15f * user_alignment -
            0.30f * uncertainty_ratio);
    record.moment.contradiction_pressure = clamp_unit(
            0.55f * contradiction_ratio +
            0.25f * negative_ratio +
            0.20f * question_ratio);
    record.moment.planning_clarity = clamp_unit(
            0.40f * planning_ratio +
            0.25f * structure_ratio +
            0.20f * user_alignment +
            0.15f * resolution_ratio -
            0.20f * contradiction_ratio);
    record.moment.curiosity = clamp_unit(
            0.45f * question_ratio +
            0.35f * semantic_novelty +
            0.20f * (kind == SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING ? 1.0f : 0.0f));
    record.moment.caution = clamp_unit(
            0.45f * uncertainty_ratio +
            0.30f * contradiction_ratio +
            0.15f * negative_ratio +
            0.10f * runtime_failure_pressure);
    record.moment.frustration = clamp_unit(
            0.45f * negative_ratio +
            0.20f * contradiction_ratio +
            0.20f * runtime_failure_pressure +
            0.15f * lexical_repetition);
    record.moment.satisfaction = clamp_unit(
            0.40f * positive_ratio +
            0.25f * resolution_ratio +
            0.20f * user_alignment +
            0.15f * runtime_trust -
            0.25f * negative_ratio);
    record.moment.momentum = clamp_unit(
            0.35f * planning_ratio +
            0.25f * resolution_ratio +
            0.20f * semantic_novelty +
            0.20f * positive_ratio -
            0.20f * lexical_repetition);
    record.moment.stall = clamp_unit(
            0.40f * lexical_repetition +
            0.20f * uncertainty_ratio +
            0.20f * negative_ratio +
            0.20f * (1.0f - density));
    record.moment.semantic_novelty = semantic_novelty;
    record.moment.user_alignment = user_alignment;
    record.moment.runtime_trust = runtime_trust;
    record.moment.runtime_failure_pressure = runtime_failure_pressure;

    record.delta = compute_delta(record.moment, previous_moment);
    record.vad = project_vad(record.moment, previous_vad, config_.vad_ema_alpha);
    return record;
}

namespace {

struct cognitive_replay_candidate {
    bool valid = false;
    int32_t trigger_block_index = 0;
    int32_t window_start_block_index = 0;
    int32_t window_end_block_index = 0;
    float negative_mass = 0.0f;
    float valence_drop = 0.0f;
    float dominance_drop = 0.0f;
    float control_failure_mass = 0.0f;
    server_emotive_vad baseline_vad_before;
    server_emotive_vad baseline_vad_after;
    std::vector<server_emotive_block_record> window_blocks;
};

static bool should_admit_replay_candidate(
        const server_emotive_block_record & current,
        float valence_drop,
        float dominance_drop,
        const server_cognitive_replay_config & config) {
    const server_emotive_delta & delta = current.delta;
    const server_emotive_vector & moment = current.moment;
    const bool baseline_negative =
            delta.negative_mass >= config.neg_mass_threshold ||
            (valence_drop >= config.valence_drop_threshold &&
             dominance_drop >= config.dominance_drop_threshold);
    const bool control_failure =
            moment.stall >= 0.70f ||
            moment.runtime_failure_pressure >= 0.60f ||
            (moment.contradiction_pressure >= 0.60f && moment.planning_clarity <= 0.35f) ||
            (moment.stall >= 0.55f && moment.confidence >= 0.40f);
    return baseline_negative || control_failure;
}

static bool has_persistent_negative_signal(
        const server_emotive_trace & trace,
        size_t index,
        const server_cognitive_replay_config & config) {
    if (index == 0 || index >= trace.blocks.size()) {
        return false;
    }

    const auto & current = trace.blocks[index];
    const auto & previous = trace.blocks[index - 1];
    const float valence_drop = std::max(0.0f, previous.vad.valence - current.vad.valence);
    const float dominance_drop = std::max(0.0f, previous.vad.dominance - current.vad.dominance);
    return should_admit_replay_candidate(current, valence_drop, dominance_drop, config);
}

static float replay_candidate_priority(const cognitive_replay_candidate & candidate) {
    return candidate.negative_mass +
            candidate.valence_drop +
            candidate.dominance_drop +
            candidate.control_failure_mass;
}

static float replay_entry_priority(const server_cognitive_replay_entry & entry) {
    return entry.negative_mass + entry.valence_drop + entry.dominance_drop;
}

static cognitive_replay_candidate strongest_replay_candidate(
        const server_emotive_trace & trace,
        const server_cognitive_replay_config & config) {
    cognitive_replay_candidate best = {};
    if (trace.blocks.size() < 2) {
        return best;
    }

    for (size_t i = 1; i < trace.blocks.size(); ++i) {
        const auto & current = trace.blocks[i];
        const auto & previous = trace.blocks[i - 1];
        const float valence_drop = std::max(0.0f, previous.vad.valence - current.vad.valence);
        const float dominance_drop = std::max(0.0f, previous.vad.dominance - current.vad.dominance);
        if (!should_admit_replay_candidate(current, valence_drop, dominance_drop, config)) {
            continue;
        }

        int32_t persistence = 1;
        for (size_t j = i; j > 1; --j) {
            if (!has_persistent_negative_signal(trace, j - 1, config)) {
                break;
            }
            ++persistence;
        }
        for (size_t j = i + 1; j < trace.blocks.size(); ++j) {
            if (!has_persistent_negative_signal(trace, j, config)) {
                break;
            }
            ++persistence;
        }
        if (persistence < config.persistence_blocks) {
            continue;
        }

        cognitive_replay_candidate candidate = {};
        candidate.valid = true;
        candidate.trigger_block_index = (int32_t) i;
        candidate.window_start_block_index = std::max<int32_t>(0, (int32_t) i - config.context_before);
        candidate.window_end_block_index = std::min<int32_t>(
                (int32_t) trace.blocks.size() - 1,
                (int32_t) i + persistence - 1 + config.context_after);
        candidate.negative_mass = current.delta.negative_mass;
        candidate.valence_drop = valence_drop;
        candidate.dominance_drop = dominance_drop;
        candidate.control_failure_mass = std::max({
                current.moment.stall,
                current.moment.runtime_failure_pressure,
                clamp_unit(current.moment.contradiction_pressure - current.moment.planning_clarity + 0.35f),
        });
        candidate.baseline_vad_before = previous.vad;
        candidate.baseline_vad_after = current.vad;
        candidate.window_blocks.assign(
                trace.blocks.begin() + candidate.window_start_block_index,
                trace.blocks.begin() + candidate.window_end_block_index + 1);

        if (!best.valid || replay_candidate_priority(candidate) > replay_candidate_priority(best)) {
            best = std::move(candidate);
        }
    }

    return best;
}

static void trim_replay_entries_locked(
        std::deque<server_cognitive_replay_entry> & entries,
        int32_t max_entries) {
    while ((int32_t) entries.size() > max_entries) {
        auto erase_it = std::find_if(
                entries.begin(),
                entries.end(),
                [](const server_cognitive_replay_entry & entry) {
                    return entry.status != SERVER_COGNITIVE_REPLAY_REVIEWING;
                });
        if (erase_it == entries.end()) {
            break;
        }
        entries.erase(erase_it);
    }
}

static void trim_replay_results_locked(
        std::deque<server_cognitive_replay_result> & results,
        int32_t max_results) {
    while ((int32_t) results.size() > max_results) {
        results.pop_front();
    }
}

static server_cognitive_replay_comparison compare_replay_result(
        const server_cognitive_replay_entry & entry,
        const server_emotive_trace & replay_trace,
        float improvement_threshold) {
    server_cognitive_replay_comparison comparison = {};
    const replay_slice_measurement replay_measurement = measure_replay_generated_slice(replay_trace);
    comparison.baseline_negative_mass = entry.negative_mass;
    comparison.baseline_valence = entry.baseline_vad_after.valence;
    comparison.baseline_dominance = entry.baseline_vad_after.dominance;
    comparison.replay_negative_mass = replay_measurement.negative_mass;
    comparison.replay_valence = replay_measurement.valence;
    comparison.replay_dominance = replay_measurement.dominance;

    const float baseline_score = cognitive_replay_score(
            comparison.baseline_valence,
            comparison.baseline_dominance,
            comparison.baseline_negative_mass);
    const float replay_score = cognitive_replay_score(
            comparison.replay_valence,
            comparison.replay_dominance,
            comparison.replay_negative_mass);

    comparison.improved =
            replay_score >= baseline_score + improvement_threshold &&
            comparison.replay_valence >= comparison.baseline_valence - 0.02f &&
            comparison.replay_dominance >= comparison.baseline_dominance - 0.02f;
    return comparison;
}

} // namespace

void server_emotive_runtime::remember_trace(const server_emotive_trace & trace) {
    if (!trace.valid) {
        return;
    }

    std::lock_guard<std::mutex> lock(history_mutex_);
    latest_traces_.push_back(trace);
    while ((int32_t) latest_traces_.size() > config_.max_turn_history) {
        latest_traces_.pop_front();
    }

    if (!config_.cognitive_replay.enabled || trace.suppress_replay_admission || trace.cognitive_replay) {
        return;
    }

    const cognitive_replay_candidate candidate = strongest_replay_candidate(trace, config_.cognitive_replay);
    if (!candidate.valid) {
        return;
    }

    server_cognitive_replay_entry entry = {};
    entry.entry_id = make_cognitive_replay_entry_id();
    entry.status = SERVER_COGNITIVE_REPLAY_OPEN;
    entry.source_trace_id = trace.trace_id;
    entry.created_at_ms = now_ms();
    entry.updated_at_ms = entry.created_at_ms;
    entry.trigger_block_index = candidate.trigger_block_index;
    entry.window_start_block_index = candidate.window_start_block_index;
    entry.window_end_block_index = candidate.window_end_block_index;
    entry.max_attempts = config_.cognitive_replay.max_attempts;
    entry.negative_mass = candidate.negative_mass;
    entry.valence_drop = candidate.valence_drop;
    entry.dominance_drop = candidate.dominance_drop;
    entry.baseline_vad_before = candidate.baseline_vad_before;
    entry.baseline_vad_after = candidate.baseline_vad_after;
    entry.window_blocks = candidate.window_blocks;
    entry.summary_excerpt = summarize_blocks(entry.window_blocks, 400);

    replay_entries_.push_back(std::move(entry));
    trim_replay_entries_locked(replay_entries_, config_.cognitive_replay.max_entries);
}

json server_emotive_runtime::health_json() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    int32_t open_count = 0;
    int32_t reviewing_count = 0;
    int32_t resolved_count = 0;
    int32_t deferred_count = 0;
    for (const auto & entry : replay_entries_) {
        switch (entry.status) {
            case SERVER_COGNITIVE_REPLAY_OPEN: ++open_count; break;
            case SERVER_COGNITIVE_REPLAY_REVIEWING: ++reviewing_count; break;
            case SERVER_COGNITIVE_REPLAY_RESOLVED: ++resolved_count; break;
            case SERVER_COGNITIVE_REPLAY_DEFERRED: ++deferred_count; break;
        }
    }
    return {
        {"enabled", config_.enabled},
        {"control_policy_version", "control_surface_v2"},
        {"block_max_chars", config_.block_max_chars},
        {"max_blocks_per_turn", config_.max_blocks_per_turn},
        {"max_turn_history", config_.max_turn_history},
        {"vad_ema_alpha", config_.vad_ema_alpha},
        {"latest_turn_count", (int32_t) latest_traces_.size()},
        {"cognitive_replay", {
            {"enabled", config_.cognitive_replay.enabled},
            {"max_entries", config_.cognitive_replay.max_entries},
            {"max_results", config_.cognitive_replay.max_results},
            {"idle_after_ms", config_.cognitive_replay.idle_after_ms},
            {"poll_interval_ms", config_.cognitive_replay.poll_interval_ms},
            {"neg_mass_threshold", config_.cognitive_replay.neg_mass_threshold},
            {"valence_drop_threshold", config_.cognitive_replay.valence_drop_threshold},
            {"dominance_drop_threshold", config_.cognitive_replay.dominance_drop_threshold},
            {"persistence_blocks", config_.cognitive_replay.persistence_blocks},
            {"max_attempts", config_.cognitive_replay.max_attempts},
            {"open_count", open_count},
            {"reviewing_count", reviewing_count},
            {"resolved_count", resolved_count},
            {"deferred_count", deferred_count},
            {"result_count", (int32_t) replay_results_.size()},
        }},
        {"heuristic_memory", {
            {"enabled", config_.heuristic_memory.enabled},
            {"path", config_.heuristic_memory.path},
            {"max_records", config_.heuristic_memory.max_records},
            {"top_k_semantic", config_.heuristic_memory.top_k_semantic},
            {"semantic_threshold", config_.heuristic_memory.semantic_threshold},
            {"rerank_threshold", config_.heuristic_memory.rerank_threshold},
            {"record_count", (int32_t) heuristic_records_.size()},
            {"last_error", heuristic_memory_error_},
            {"last_retrieval", heuristic_retrieval_decision_to_json(last_heuristic_retrieval_)},
        }},
        {"embedding_backend", embedding_backend_.health_json()},
    };
}

json server_emotive_runtime::latest_trace_json() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    if (latest_traces_.empty()) {
        return {
            {"object", "vicuna.emotive.trace"},
            {"trace", nullptr},
            {"retained_turns", 0},
        };
    }

    return {
        {"object", "vicuna.emotive.trace"},
        {"trace", server_emotive_trace_to_json(latest_traces_.back())},
        {"retained_turns", (int32_t) latest_traces_.size()},
    };
}

json server_emotive_runtime::cognitive_replay_json() const {
    std::lock_guard<std::mutex> lock(history_mutex_);

    json entries = json::array();
    for (const auto & entry : replay_entries_) {
        json window_blocks = json::array();
        for (const auto & block : entry.window_blocks) {
            window_blocks.push_back(block_record_to_json(block));
        }
        entries.push_back({
            {"entry_id", entry.entry_id},
            {"status", cognitive_replay_status_name(entry.status)},
            {"source_trace_id", entry.source_trace_id},
            {"created_at_ms", entry.created_at_ms},
            {"updated_at_ms", entry.updated_at_ms},
            {"trigger_block_index", entry.trigger_block_index},
            {"window_start_block_index", entry.window_start_block_index},
            {"window_end_block_index", entry.window_end_block_index},
            {"attempt_count", entry.attempt_count},
            {"max_attempts", entry.max_attempts},
            {"severity", {
                {"negative_mass", entry.negative_mass},
                {"valence_drop", entry.valence_drop},
                {"dominance_drop", entry.dominance_drop},
            }},
            {"baseline_vad", {
                {"before", vad_to_json(entry.baseline_vad_before)},
                {"after", vad_to_json(entry.baseline_vad_after)},
            }},
            {"summary_excerpt", entry.summary_excerpt},
            {"last_error", entry.last_error},
            {"resolved_result_id", entry.resolved_result_id},
            {"window_blocks", std::move(window_blocks)},
        });
    }

    json latest_result = nullptr;
    if (!replay_results_.empty()) {
        const auto & result = replay_results_.back();
        latest_result = {
            {"result_id", result.result_id},
            {"entry_id", result.entry_id},
            {"trace_id", result.trace_id},
            {"created_at_ms", result.created_at_ms},
            {"reasoning_content", result.reasoning_content},
            {"content", result.content},
            {"comparison", {
                {"baseline_negative_mass", result.comparison.baseline_negative_mass},
                {"replay_negative_mass", result.comparison.replay_negative_mass},
                {"baseline_valence", result.comparison.baseline_valence},
                {"replay_valence", result.comparison.replay_valence},
                {"baseline_dominance", result.comparison.baseline_dominance},
                {"replay_dominance", result.comparison.replay_dominance},
                {"improved", result.comparison.improved},
            }},
            {"replay_trace", server_emotive_trace_to_json(result.replay_trace)},
        };
    }

    return {
        {"object", "vicuna.emotive.cognitive_replay"},
        {"entries", std::move(entries)},
        {"latest_result", std::move(latest_result)},
    };
}

json server_emotive_runtime::heuristic_memory_json() const {
    std::lock_guard<std::mutex> lock(history_mutex_);

    json records = json::array();
    for (const auto & record : heuristic_records_) {
        records.push_back(heuristic_memory_record_to_json(record));
    }

    return {
        {"object", "vicuna.emotive.heuristic_memory"},
        {"path", config_.heuristic_memory.path},
        {"enabled", config_.heuristic_memory.enabled},
        {"max_records", config_.heuristic_memory.max_records},
        {"record_count", (int32_t) heuristic_records_.size()},
        {"last_error", heuristic_memory_error_},
        {"last_retrieval", heuristic_retrieval_decision_to_json(last_heuristic_retrieval_)},
        {"records", std::move(records)},
    };
}

bool server_emotive_runtime::try_claim_cognitive_replay_entry(server_cognitive_replay_entry * out_entry) {
    if (!out_entry) {
        return false;
    }

    std::lock_guard<std::mutex> lock(history_mutex_);
    auto best_it = replay_entries_.end();
    float best_priority = -1.0f;
    for (auto it = replay_entries_.begin(); it != replay_entries_.end(); ++it) {
        if (it->status != SERVER_COGNITIVE_REPLAY_OPEN) {
            continue;
        }
        const float priority = replay_entry_priority(*it);
        if (best_it == replay_entries_.end() || priority > best_priority ||
                (priority == best_priority && it->updated_at_ms > best_it->updated_at_ms)) {
            best_it = it;
            best_priority = priority;
        }
    }

    if (best_it == replay_entries_.end()) {
        return false;
    }

    best_it->status = SERVER_COGNITIVE_REPLAY_REVIEWING;
    best_it->updated_at_ms = now_ms();
    *out_entry = *best_it;
    return true;
}

void server_emotive_runtime::fail_cognitive_replay_entry(const std::string & entry_id, const std::string & error_message) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    auto it = std::find_if(
            replay_entries_.begin(),
            replay_entries_.end(),
            [&entry_id](const server_cognitive_replay_entry & entry) {
                return entry.entry_id == entry_id;
            });
    if (it == replay_entries_.end()) {
        return;
    }

    it->attempt_count += 1;
    it->updated_at_ms = now_ms();
    it->last_error = error_message;
    it->status = it->attempt_count >= it->max_attempts ?
            SERVER_COGNITIVE_REPLAY_DEFERRED :
            SERVER_COGNITIVE_REPLAY_OPEN;
}

bool server_emotive_runtime::get_cognitive_replay_resolution(
        const std::string & entry_id,
        server_cognitive_replay_entry * out_entry,
        server_cognitive_replay_result * out_result) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    auto entry_it = std::find_if(
            replay_entries_.begin(),
            replay_entries_.end(),
            [&entry_id](const server_cognitive_replay_entry & entry) {
                return entry.entry_id == entry_id;
            });
    if (entry_it == replay_entries_.end()) {
        return false;
    }

    if (out_entry) {
        *out_entry = *entry_it;
    }

    if (!out_result) {
        return true;
    }

    auto result_it = replay_results_.end();
    if (!entry_it->resolved_result_id.empty()) {
        result_it = std::find_if(
                replay_results_.begin(),
                replay_results_.end(),
                [&entry_it](const server_cognitive_replay_result & result) {
                    return result.result_id == entry_it->resolved_result_id;
                });
    }
    if (result_it == replay_results_.end()) {
        const auto reverse_it = std::find_if(
                replay_results_.rbegin(),
                replay_results_.rend(),
                [&entry_id](const server_cognitive_replay_result & result) {
                    return result.entry_id == entry_id && result.comparison.improved;
                });
        if (reverse_it == replay_results_.rend()) {
            result_it = replay_results_.end();
        } else {
            result_it = std::prev(reverse_it.base());
        }
    }

    if (result_it == replay_results_.end()) {
        return false;
    }

    *out_result = *result_it;
    return true;
}

void server_emotive_runtime::record_cognitive_replay_result(
        const std::string & entry_id,
        const std::string & reasoning_content,
        const std::string & content,
        const server_emotive_trace & replay_trace) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    auto it = std::find_if(
            replay_entries_.begin(),
            replay_entries_.end(),
            [&entry_id](const server_cognitive_replay_entry & entry) {
                return entry.entry_id == entry_id;
            });
    if (it == replay_entries_.end()) {
        return;
    }

    server_cognitive_replay_result result = {};
    result.valid = true;
    result.result_id = make_cognitive_replay_result_id();
    result.entry_id = entry_id;
    result.trace_id = replay_trace.trace_id;
    result.created_at_ms = now_ms();
    result.reasoning_content = reasoning_content;
    result.content = content;
    result.replay_trace = replay_trace;
    result.comparison = compare_replay_result(*it, replay_trace, config_.cognitive_replay.improvement_threshold);

    replay_results_.push_back(result);
    trim_replay_results_locked(replay_results_, config_.cognitive_replay.max_results);

    it->updated_at_ms = result.created_at_ms;
    it->last_error.clear();
    if (result.comparison.improved) {
        it->status = SERVER_COGNITIVE_REPLAY_RESOLVED;
        it->resolved_result_id = result.result_id;
        return;
    }

    it->attempt_count += 1;
    it->status = it->attempt_count >= it->max_attempts ?
            SERVER_COGNITIVE_REPLAY_DEFERRED :
            SERVER_COGNITIVE_REPLAY_OPEN;
    if (it->status != SERVER_COGNITIVE_REPLAY_RESOLVED) {
        it->last_error = "Replay did not improve measured emotive state.";
    }
}

bool server_emotive_runtime::store_heuristic_memory_record(
        const std::string & entry_id,
        const server_heuristic_object & heuristic,
        std::string * out_error) {
    if (!config_.heuristic_memory.enabled) {
        if (out_error) {
            *out_error = "heuristic memory disabled";
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(history_mutex_);
    auto entry_it = std::find_if(
            replay_entries_.begin(),
            replay_entries_.end(),
            [&entry_id](const server_cognitive_replay_entry & entry) {
                return entry.entry_id == entry_id;
            });
    if (entry_it == replay_entries_.end()) {
        if (out_error) {
            *out_error = "cognitive replay entry not found";
        }
        heuristic_memory_error_ = out_error ? *out_error : "cognitive replay entry not found";
        return false;
    }
    if (entry_it->status != SERVER_COGNITIVE_REPLAY_RESOLVED) {
        if (out_error) {
            *out_error = "cognitive replay entry is not resolved";
        }
        heuristic_memory_error_ = out_error ? *out_error : "cognitive replay entry is not resolved";
        return false;
    }

    auto result_it = replay_results_.end();
    if (!entry_it->resolved_result_id.empty()) {
        result_it = std::find_if(
                replay_results_.begin(),
                replay_results_.end(),
                [&entry_it](const server_cognitive_replay_result & result) {
                    return result.result_id == entry_it->resolved_result_id;
                });
    }
    if (result_it == replay_results_.end()) {
        result_it = std::find_if(
                replay_results_.begin(),
                replay_results_.end(),
                [&entry_id](const server_cognitive_replay_result & result) {
                    return result.entry_id == entry_id && result.comparison.improved;
                });
    }
    if (result_it == replay_results_.end()) {
        if (out_error) {
            *out_error = "resolved replay result not found";
        }
        heuristic_memory_error_ = out_error ? *out_error : "resolved replay result not found";
        return false;
    }

    auto existing_it = std::find_if(
            heuristic_records_.begin(),
            heuristic_records_.end(),
            [&entry_id](const server_heuristic_memory_record & record) {
                return record.entry_id == entry_id;
            });
    if (existing_it != heuristic_records_.end()) {
        heuristic_records_.erase(existing_it);
    }

    server_heuristic_memory_record record = {};
    record.record_id = make_heuristic_record_id();
    record.entry_id = entry_it->entry_id;
    record.result_id = result_it->result_id;
    record.source_trace_id = entry_it->source_trace_id;
    record.created_at_ms = now_ms();
    record.bad_path_text = join_block_lines(entry_it->window_blocks);
    record.better_path_reasoning_content = result_it->reasoning_content;
    record.better_path_content = result_it->content;

    std::vector<std::string> signature_tags;
    std::vector<std::string> derived_tool_names;
    for (const auto & block : entry_it->window_blocks) {
        server_heuristic_bad_path_object object = {};
        object.object_id = make_bad_path_object_id();
        object.kind = kind_name(block.kind);
        object.text = block.text;
        object.struct_tags = extract_struct_tags(block.text, object.kind);
        signature_tags.insert(signature_tags.end(), object.struct_tags.begin(), object.struct_tags.end());
        build_embedding(object.text, object.embedding, nullptr);
        if (object.kind == "runtime_event" && object.text.rfind("tool_call:", 0) == 0) {
            const size_t name_start = std::string("tool_call:").size();
            const size_t name_end = object.text.find(' ', name_start);
            derived_tool_names.push_back(object.text.substr(name_start, name_end - name_start));
        }
        record.bad_path_objects.push_back(std::move(object));
    }

    record.heuristic = heuristic;
    if (record.heuristic.heuristic_id.empty()) {
        record.heuristic.heuristic_id = make_heuristic_object_id();
    }
    if (record.heuristic.title.empty()) {
        record.heuristic.title = "Replay-derived heuristic";
    }
    sort_and_dedupe_strings(record.heuristic.task_types);
    sort_and_dedupe_strings(record.heuristic.tool_names);
    sort_and_dedupe_strings(record.heuristic.struct_tags);
    sort_and_dedupe_strings(derived_tool_names);

    record.bad_signature.negative_mass = entry_it->negative_mass;
    record.bad_signature.valence = entry_it->baseline_vad_after.valence;
    record.bad_signature.arousal = entry_it->baseline_vad_after.arousal;
    record.bad_signature.dominance = entry_it->baseline_vad_after.dominance;
    record.bad_signature.struct_tags = std::move(signature_tags);
    sort_and_dedupe_strings(record.bad_signature.struct_tags);

    if (record.heuristic.struct_tags.empty()) {
        record.heuristic.struct_tags = record.bad_signature.struct_tags;
    }
    if (record.heuristic.tool_names.empty()) {
        record.heuristic.tool_names = derived_tool_names;
    }
    if (record.heuristic.task_types.empty()) {
        record.heuristic.task_types.push_back("reasoning_trace");
    }
    if (record.heuristic.semantic_trigger_text.empty()) {
        record.heuristic.semantic_trigger_text = entry_it->summary_excerpt;
    }
    if (record.heuristic.failure_mode.empty()) {
        record.heuristic.failure_mode = "negative self-state trajectory repeated a previously harmful pattern";
    }
    if (record.heuristic.applies_when.empty()) {
        record.heuristic.applies_when.push_back("when the current trace resembles this bad path and uncertainty or failure pressure is rising");
    }

    heuristic_records_.push_back(std::move(record));
    while ((int32_t) heuristic_records_.size() > config_.heuristic_memory.max_records) {
        heuristic_records_.pop_front();
    }

    std::string save_error;
    if (!save_heuristic_memory_records(config_.heuristic_memory.path, heuristic_records_, &save_error)) {
        if (out_error) {
            *out_error = save_error;
        }
        heuristic_memory_error_ = save_error;
        return false;
    }

    heuristic_memory_error_.clear();
    if (out_error) {
        out_error->clear();
    }
    return true;
}

server_heuristic_retrieval_decision server_emotive_runtime::retrieve_matching_heuristic(
        const std::string & query_text,
        const std::vector<std::string> & struct_tags,
        const server_emotive_vector * current_moment,
        const server_emotive_vad * current_vad,
        std::string * out_guidance) {
    server_heuristic_retrieval_decision decision = {};
    decision.query_text = query_text;
    decision.threshold = config_.heuristic_memory.rerank_threshold;
    decision.created_at_ms = now_ms();
    if (out_guidance) {
        out_guidance->clear();
    }

    if (!config_.heuristic_memory.enabled || trimmed_text(query_text).empty()) {
        std::lock_guard<std::mutex> lock(history_mutex_);
        last_heuristic_retrieval_ = decision;
        return decision;
    }

    std::vector<float> query_embedding;
    build_embedding(query_text, query_embedding, nullptr);

    struct candidate_score {
        const server_heuristic_memory_record * record = nullptr;
        float semantic_score = 0.0f;
    };
    std::vector<candidate_score> semantic_candidates;

    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        for (const auto & record : heuristic_records_) {
            float best_semantic = lexical_text_similarity(query_text, record.bad_path_text);
            for (const auto & object : record.bad_path_objects) {
                float score = lexical_text_similarity(query_text, object.text);
                if (!query_embedding.empty() &&
                        !object.embedding.empty() &&
                        object.embedding.size() == query_embedding.size()) {
                    score = std::max(
                            score,
                            clamp_unit((server_embd_similarity_cos(
                                                query_embedding.data(),
                                                object.embedding.data(),
                                                (int) query_embedding.size()) + 1.0f) * 0.5f));
                }
                best_semantic = std::max(best_semantic, score);
            }
            if (best_semantic >= config_.heuristic_memory.semantic_threshold) {
                semantic_candidates.push_back({&record, best_semantic});
            }
        }
    }

    std::sort(
            semantic_candidates.begin(),
            semantic_candidates.end(),
            [](const candidate_score & lhs, const candidate_score & rhs) {
                return lhs.semantic_score > rhs.semantic_score;
            });
    if ((int32_t) semantic_candidates.size() > config_.heuristic_memory.top_k_semantic) {
        semantic_candidates.resize(config_.heuristic_memory.top_k_semantic);
    }

    const std::vector<std::string> normalized_tags = [&struct_tags]() {
        std::vector<std::string> tags = struct_tags;
        sort_and_dedupe_strings(tags);
        return tags;
    }();

    const server_heuristic_memory_record * best_record = nullptr;
    for (const auto & candidate : semantic_candidates) {
        const float struct_score = weighted_jaccard_score(
                normalized_tags,
                candidate.record->heuristic.struct_tags.empty() ?
                        candidate.record->bad_signature.struct_tags :
                        candidate.record->heuristic.struct_tags);
        const float emotive_score = normalized_emotive_similarity(
                current_moment,
                current_vad,
                candidate.record->bad_signature);
        const float total_score =
                config_.heuristic_memory.semantic_weight * candidate.semantic_score +
                config_.heuristic_memory.struct_weight * struct_score +
                config_.heuristic_memory.emotive_weight * emotive_score;
        if (!best_record || total_score > decision.total_score) {
            best_record = candidate.record;
            decision.record_id = candidate.record->record_id;
            decision.heuristic_id = candidate.record->heuristic.heuristic_id;
            decision.semantic_score = candidate.semantic_score;
            decision.struct_score = struct_score;
            decision.emotive_score = emotive_score;
            decision.total_score = total_score;
        }
    }

    decision.matched = best_record && decision.total_score >= config_.heuristic_memory.rerank_threshold;
    if (decision.matched) {
        const auto & heuristic = best_record->heuristic;
        decision.control_biases = derive_control_biases(heuristic);
    }
    if (decision.matched && out_guidance) {
        const auto & heuristic = best_record->heuristic;
        std::ostringstream guidance;
        guidance << "[Critical Guidance | id=" << heuristic.heuristic_id << "]\n";
        guidance << "Trigger: "
                 << (heuristic.semantic_trigger_text.empty() ? heuristic.failure_mode : heuristic.semantic_trigger_text)
                 << "\n";
        if (!heuristic.constraints.empty()) {
            guidance << "Constraints:\n";
            for (const auto & constraint : heuristic.constraints) {
                guidance << "- " << constraint << "\n";
            }
        }
        if (!heuristic.preferred_actions.empty()) {
            guidance << "Action bias:\n";
            guidance << "- " << heuristic.preferred_actions.front() << "\n";
        } else if (!heuristic.mid_reasoning_correction.empty()) {
            guidance << "Action bias:\n";
            guidance << "- " << heuristic.mid_reasoning_correction << "\n";
        }
        if (!heuristic.applies_when.empty()) {
            guidance << "Scope:\n";
            guidance << "- " << heuristic.applies_when.front() << "\n";
        }
        *out_guidance = guidance.str();
    } else if (!decision.matched) {
        decision.control_biases.clear();
    }

    std::lock_guard<std::mutex> lock(history_mutex_);
    last_heuristic_retrieval_ = decision;
    return decision;
}

server_metacognitive_policy_decision server_emotive_runtime::compute_control_policy(
        const server_metacognitive_control_state & state) const {
    const server_emotive_vector & m = state.moment;
    const float valence_n = clamp_unit((state.vad.valence + 1.0f) * 0.5f);
    const float dominance_n = clamp_unit((state.vad.dominance + 1.0f) * 0.5f);
    server_metacognitive_policy_decision decision = {};
    decision.valid = true;
    decision.policy_version = "control_surface_v2";

    std::map<std::string, float> bias_by_target;
    for (const auto & item : state.heuristic.control_biases) {
        if (!item.is_object()) {
            continue;
        }
        const std::string target = json_value(item, "target", std::string());
        const float bias = std::max(-0.25f, std::min(0.25f, json_value(item, "bias", 0.0f)));
        if (target.empty()) {
            continue;
        }
        bias_by_target[target] = std::max(-0.35f, std::min(0.35f, bias_by_target[target] + bias));
        decision.heuristic_biases.push_back(item);
    }

    const auto with_bias = [&bias_by_target](const std::string & target, float value) {
        auto it = bias_by_target.find(target);
        if (it == bias_by_target.end()) {
            return value;
        }
        return value + it->second;
    };

    decision.direct_score = with_bias(
            "direct",
            0.55f * m.confidence +
                    0.45f * m.planning_clarity +
                    0.35f * m.satisfaction +
                    0.25f * m.user_alignment +
                    0.15f * dominance_n -
                    0.60f * m.epistemic_pressure -
                    0.55f * m.contradiction_pressure -
                    0.35f * m.runtime_failure_pressure -
                    0.30f * m.stall);
    decision.reflective_score = with_bias(
            "reflective",
            0.45f * m.user_alignment +
                    0.40f * m.caution +
                    0.35f * m.epistemic_pressure +
                    0.25f * (1.0f - dominance_n) +
                    0.20f * (1.0f - valence_n) -
                    0.30f * m.momentum -
                    0.25f * m.planning_clarity);
    decision.tool_light_score = with_bias(
            "tool_light",
            0.55f * m.epistemic_pressure +
                    0.35f * m.planning_clarity +
                    0.30f * m.user_alignment +
                    0.25f * m.confidence +
                    0.20f * m.runtime_trust -
                    0.30f * m.runtime_failure_pressure -
                    0.20f * m.frustration);
    decision.tool_heavy_score = with_bias(
            "tool_heavy",
            0.65f * m.epistemic_pressure +
                    0.55f * m.contradiction_pressure +
                    0.40f * (1.0f - m.runtime_trust) +
                    0.35f * (1.0f - m.user_alignment) +
                    0.25f * m.stall -
                    0.20f * m.satisfaction -
                    0.15f * valence_n);
    decision.background_defer_score = with_bias(
            "background_defer",
            0.50f * state.ongoing_task_due +
                    0.35f * m.satisfaction +
                    0.30f * m.planning_clarity +
                    0.25f * m.momentum -
                    0.55f * m.user_alignment -
                    0.40f * m.frustration -
                    0.30f * m.runtime_failure_pressure);

    const std::vector<std::pair<std::string, float>> mode_scores = {
        {"direct", decision.direct_score},
        {"reflective", decision.reflective_score},
        {"tool_light", decision.tool_light_score},
        {"tool_heavy", decision.tool_heavy_score},
        {"background_defer", decision.background_defer_score},
    };
    decision.selected_mode = "direct";
    float best_score = decision.direct_score;
    for (const auto & item : mode_scores) {
        if (item.second > best_score) {
            best_score = item.second;
            decision.selected_mode = item.first;
        }
    }

    if (m.runtime_failure_pressure >= 0.75f && decision.selected_mode == "direct") {
        decision.selected_mode = "tool_light";
    }
    if (m.stall >= 0.75f && m.contradiction_pressure >= 0.55f) {
        decision.selected_mode = "reflective";
    }
    if (state.ongoing_task_due >= 0.80f && m.user_alignment <= 0.25f) {
        decision.selected_mode = "background_defer";
    }
    if (state.cognitive_replay) {
        decision.selected_mode = "reflective";
    }
    if (state.bridge_scoped && decision.selected_mode == "background_defer") {
        decision.selected_mode = "reflective";
    }

    decision.reasoning_score = with_bias(
            "reasoning_score",
            0.50f * m.epistemic_pressure +
                    0.40f * m.contradiction_pressure +
                    0.30f * m.curiosity +
                    0.20f * m.caution -
                    0.45f * m.confidence -
                    0.25f * m.planning_clarity);
    if (decision.reasoning_score <= 0.05f) {
        decision.reasoning_depth = "none";
    } else if (decision.reasoning_score <= 0.30f) {
        decision.reasoning_depth = "short";
    } else if (decision.reasoning_score <= 0.60f) {
        decision.reasoning_depth = "medium";
    } else {
        decision.reasoning_depth = "deep";
    }

    decision.tool_aggression = clamp_unit(with_bias(
            "tool_aggression",
            0.45f * m.epistemic_pressure +
                    0.35f * m.contradiction_pressure +
                    0.35f * (1.0f - m.runtime_trust) +
                    0.20f * m.stall -
                    0.25f * m.satisfaction));

    if (decision.tool_aggression >= 0.45f &&
            m.planning_clarity >= 0.65f &&
            m.momentum >= 0.55f &&
            m.runtime_trust >= 0.55f &&
            with_bias("tool_parallelism_gate", 0.0f) >= -0.25f) {
        decision.tool_parallelism_cap = 2;
    } else if (decision.tool_aggression >= 0.20f) {
        decision.tool_parallelism_cap = 1;
    } else {
        decision.tool_parallelism_cap = 0;
    }

    decision.interrupt_score = clamp_unit(with_bias(
            "interrupt_score",
            std::max(m.contradiction_pressure, std::max(m.runtime_failure_pressure, m.stall)) +
                    0.25f * m.epistemic_pressure -
                    0.25f * m.momentum));
    decision.interrupt_allowed = decision.interrupt_score >= 0.70f;
    const float replan_pressure = with_bias("replan_pressure", m.stall);
    decision.replan_required = replan_pressure >= 0.70f && m.planning_clarity <= 0.35f;
    decision.early_stop_ok =
            m.satisfaction >= std::max(0.0f, 0.65f + with_bias("early_stop_gate", 0.0f)) &&
            m.contradiction_pressure <= 0.25f &&
            m.epistemic_pressure <= 0.35f;
    decision.force_synthesis =
            m.stall >= std::max(0.0f, 0.70f - with_bias("force_synthesis_gate", 0.0f)) &&
            m.confidence >= 0.40f;

    const bool low_deliberation_ready =
            decision.selected_mode == "direct" &&
            decision.reasoning_depth != "deep" &&
            m.epistemic_pressure <= 0.45f &&
            m.contradiction_pressure <= 0.35f &&
            m.stall <= 0.55f;
    if (decision.reasoning_depth == "none" || decision.force_synthesis || decision.early_stop_ok ||
            low_deliberation_ready) {
        decision.thinking_mode = "disabled";
    } else {
        decision.thinking_mode = "enabled";
    }

    if (decision.replan_required) {
        decision.prefix_profile = "replan_outline";
    } else if (decision.force_synthesis || decision.early_stop_ok) {
        decision.prefix_profile = "bounded_answer";
    } else {
        decision.prefix_profile = "none";
    }

    if (decision.prefix_profile == "bounded_answer" && (decision.early_stop_ok || decision.force_synthesis)) {
        decision.stop_profile = "concise_answer";
    } else {
        decision.stop_profile = "none";
    }

    if (m.stall >= 0.80f || (m.stall >= 0.60f && m.frustration >= 0.55f)) {
        decision.repetition_profile = "anti_stall_hard";
    } else if (m.stall >= 0.55f || m.frustration >= 0.55f) {
        decision.repetition_profile = "anti_stall_soft";
    } else if (m.curiosity >= 0.60f && m.semantic_novelty >= 0.50f) {
        decision.repetition_profile = "novelty_soft";
    } else {
        decision.repetition_profile = "none";
    }

    if (decision.force_synthesis || decision.early_stop_ok || decision.selected_mode == "direct") {
        decision.sampling_profile = "deterministic";
    } else if (m.curiosity >= 0.65f && m.semantic_novelty >= 0.55f && decision.selected_mode == "reflective") {
        decision.sampling_profile = "creative";
    } else if (decision.selected_mode == "reflective" || decision.selected_mode == "background_defer") {
        decision.sampling_profile = "balanced";
    } else {
        decision.sampling_profile = "provider_default";
    }

    if (decision.selected_mode == "tool_heavy") {
        decision.tool_choice_profile = "required";
    } else if (decision.selected_mode == "tool_light") {
        decision.tool_choice_profile = "auto";
    } else {
        decision.tool_choice_profile = "caller_default";
    }

    if (decision.selected_mode == "tool_heavy" || decision.selected_mode == "tool_light") {
        decision.prompt_hints.push_back("Prefer direct tool calls over speculative free-text answers.");
    }
    if (decision.selected_mode == "reflective") {
        decision.prompt_hints.push_back("Surface uncertainty and reframe the plan before committing.");
    }
    if (decision.replan_required) {
        decision.prompt_hints.push_back("Explicitly replan before continuing.");
    }
    if (decision.force_synthesis) {
        decision.prompt_hints.push_back("Synthesize the best bounded answer now instead of looping.");
    }
    if (decision.early_stop_ok) {
        decision.prompt_hints.push_back("Conclude once the answer is sufficient; avoid gratuitous extra steps.");
    }
    if (decision.thinking_mode == "disabled") {
        decision.prompt_hints.push_back("Prefer a direct answer path and avoid unnecessary hidden deliberation.");
    }
    if (state.cognitive_replay) {
        decision.prompt_hints.push_back("Stay grounded in the replay window and prefer concrete corrective actions.");
    }
    return decision;
}

server_emotive_turn_builder::server_emotive_turn_builder(
        server_emotive_runtime & runtime,
        const std::string & model_name,
        bool cognitive_replay,
        const std::string & cognitive_replay_entry_id,
        bool suppress_replay_admission,
        const std::string & mode_label) :
        runtime_(runtime),
        model_name_(model_name),
        pending_kind_(SERVER_EMOTIVE_BLOCK_USER_MESSAGE),
        has_pending_(false),
        user_anchor_count_(0),
        have_previous_moment_(false),
        have_previous_vad_(false),
        live_generation_start_block_index_(0),
        live_generation_start_marked_(false),
        cognitive_replay_(cognitive_replay),
        cognitive_replay_entry_id_(cognitive_replay_entry_id),
        suppress_replay_admission_(suppress_replay_admission),
        mode_label_(mode_label) {
}

server_emotive_turn_builder::~server_emotive_turn_builder() {
}

void server_emotive_turn_builder::add_user_message(const std::string & text) {
    append_text(SERVER_EMOTIVE_BLOCK_USER_MESSAGE, text);
    flush_pending();
}

void server_emotive_turn_builder::add_replay_block(server_emotive_block_kind kind, const std::string & text) {
    append_text(kind, text);
    flush_pending();
}

void server_emotive_turn_builder::observe_reasoning_delta(const std::string & text) {
    append_text(SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING, text);
}

void server_emotive_turn_builder::observe_content_delta(const std::string & text) {
    append_text(SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT, text);
}

void server_emotive_turn_builder::observe_runtime_event(const std::string & text) {
    append_text(SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT, text);
    flush_pending();
}

void server_emotive_turn_builder::mark_live_generation_start() {
    flush_pending();
    live_generation_start_block_index_ = (int32_t) blocks_.size();
    live_generation_start_marked_ = true;
}

void server_emotive_turn_builder::set_final_policy(json final_policy) {
    final_policy_ = std::move(final_policy);
}

void server_emotive_turn_builder::set_heuristic_retrieval(json heuristic_retrieval) {
    heuristic_retrieval_ = std::move(heuristic_retrieval);
}

bool server_emotive_turn_builder::has_current_state() const {
    return have_previous_vad_;
}

server_emotive_vector server_emotive_turn_builder::current_moment() const {
    return previous_moment_;
}

server_emotive_vad server_emotive_turn_builder::current_vad() const {
    return previous_vad_;
}

void server_emotive_turn_builder::append_text(server_emotive_block_kind kind, const std::string & text) {
    if (text.empty()) {
        return;
    }

    if (has_pending_ && kind != pending_kind_) {
        flush_pending();
    }

    pending_kind_ = kind;
    pending_text_ += text;
    has_pending_ = true;

    if ((int32_t) pending_text_.size() >= runtime_.config().block_max_chars || ends_block(text)) {
        flush_pending();
    }
}

void server_emotive_turn_builder::flush_pending() {
    if (!has_pending_) {
        return;
    }

    if ((int32_t) blocks_.size() >= runtime_.config().max_blocks_per_turn) {
        pending_text_.clear();
        has_pending_ = false;
        return;
    }

    server_emotive_block_record record = runtime_.evaluate_block(
            pending_kind_,
            pending_text_,
            (int32_t) blocks_.size(),
            previous_embedding_,
            user_anchor_embedding_,
            have_previous_moment_ ? &previous_moment_ : nullptr,
            have_previous_vad_ ? &previous_vad_ : nullptr);
    blocks_.push_back(record);
    previous_moment_ = record.moment;
    have_previous_moment_ = true;
    previous_vad_ = record.vad;
    have_previous_vad_ = true;

    std::vector<float> new_embedding;
    std::string mode;
    if (runtime_.build_embedding(record.text, new_embedding, &mode) && !new_embedding.empty()) {
        previous_embedding_ = new_embedding;
        if (pending_kind_ == SERVER_EMOTIVE_BLOCK_USER_MESSAGE) {
            if (user_anchor_embedding_.empty()) {
                user_anchor_embedding_ = new_embedding;
            } else if (user_anchor_embedding_.size() == new_embedding.size()) {
                for (size_t i = 0; i < user_anchor_embedding_.size(); ++i) {
                    user_anchor_embedding_[i] =
                            (user_anchor_embedding_[i] * (float) user_anchor_count_ + new_embedding[i]) /
                            (float) (user_anchor_count_ + 1);
                }
            }
            ++user_anchor_count_;
        }
    }

    pending_text_.clear();
    has_pending_ = false;
}

server_emotive_trace server_emotive_turn_builder::finalize() {
    flush_pending();

    server_emotive_trace trace = {};
    trace.valid = !blocks_.empty();
    trace.trace_id = make_trace_id();
    trace.model = model_name_;
    trace.blocks = blocks_;
    trace.live_generation_start_block_index = live_generation_start_marked_ ?
            live_generation_start_block_index_ :
            0;
    trace.embedding_mode = blocks_.empty() ? "lexical_only" : blocks_.back().embedding_mode;
    trace.estimator_version = "v2_vad_projection";
    trace.provider_streamed = true;
    trace.retained_block_count = (int32_t) blocks_.size();
    trace.cognitive_replay = cognitive_replay_;
    trace.cognitive_replay_entry_id = cognitive_replay_entry_id_;
    trace.suppress_replay_admission = suppress_replay_admission_ || cognitive_replay_;
    trace.mode_label = !mode_label_.empty() ? mode_label_ : (cognitive_replay_ ? "cognitive_replay" : "foreground");
    trace.final_policy = final_policy_;
    trace.heuristic_retrieval = heuristic_retrieval_;
    if (!blocks_.empty()) {
        trace.final_moment = blocks_.back().moment;
        trace.final_vad = blocks_.back().vad;
    }
    runtime_.remember_trace(trace);
    return trace;
}

void server_emotive_runtime::load_heuristic_memory() {
    if (!config_.heuristic_memory.enabled) {
        return;
    }

    std::ifstream in(config_.heuristic_memory.path, std::ios::binary);
    if (!in.is_open()) {
        heuristic_memory_error_.clear();
        return;
    }

    try {
        json payload = json::parse(in);
        if (!payload.is_object()) {
            heuristic_memory_error_ = "heuristic memory payload is not an object";
            return;
        }

        std::deque<server_heuristic_memory_record> loaded;
        const json records = json_value(payload, "records", json::array());
        if (!records.is_array()) {
            heuristic_memory_error_ = "heuristic memory records payload is not an array";
            return;
        }

        for (const auto & item : records) {
            server_heuristic_memory_record record = heuristic_memory_record_from_json(item);
            for (auto & object : record.bad_path_objects) {
                if (object.embedding.empty()) {
                    build_embedding(object.text, object.embedding, nullptr);
                }
                sort_and_dedupe_strings(object.struct_tags);
            }
            sort_and_dedupe_strings(record.bad_signature.struct_tags);
            sort_and_dedupe_strings(record.heuristic.task_types);
            sort_and_dedupe_strings(record.heuristic.tool_names);
            sort_and_dedupe_strings(record.heuristic.struct_tags);
            loaded.push_back(std::move(record));
        }

        while ((int32_t) loaded.size() > config_.heuristic_memory.max_records) {
            loaded.pop_front();
        }

        std::lock_guard<std::mutex> lock(history_mutex_);
        heuristic_records_ = std::move(loaded);
        heuristic_memory_error_.clear();
    } catch (const std::exception & e) {
        heuristic_memory_error_ = e.what();
    }
}
