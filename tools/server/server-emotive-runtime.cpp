#include "server-emotive-runtime.h"
#include "server-runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    server_emotive_vad baseline_vad_before;
    server_emotive_vad baseline_vad_after;
    std::vector<server_emotive_block_record> window_blocks;
};

static bool should_admit_replay_candidate(
        const server_emotive_delta & delta,
        float valence_drop,
        float dominance_drop,
        const server_cognitive_replay_config & config) {
    return delta.negative_mass >= config.neg_mass_threshold ||
            (valence_drop >= config.valence_drop_threshold &&
             dominance_drop >= config.dominance_drop_threshold);
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
    return should_admit_replay_candidate(current.delta, valence_drop, dominance_drop, config);
}

static float replay_candidate_priority(const cognitive_replay_candidate & candidate) {
    return candidate.negative_mass + candidate.valence_drop + candidate.dominance_drop;
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
        if (!should_admit_replay_candidate(current.delta, valence_drop, dominance_drop, config)) {
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
    }

    std::lock_guard<std::mutex> lock(history_mutex_);
    last_heuristic_retrieval_ = decision;
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
    trace.embedding_mode = blocks_.empty() ? "lexical_only" : blocks_.back().embedding_mode;
    trace.estimator_version = "v2_vad_projection";
    trace.provider_streamed = true;
    trace.retained_block_count = (int32_t) blocks_.size();
    trace.cognitive_replay = cognitive_replay_;
    trace.cognitive_replay_entry_id = cognitive_replay_entry_id_;
    trace.suppress_replay_admission = suppress_replay_admission_ || cognitive_replay_;
    trace.mode_label = !mode_label_.empty() ? mode_label_ : (cognitive_replay_ ? "cognitive_replay" : "foreground");
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
