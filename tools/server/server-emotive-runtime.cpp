#include "server-emotive-runtime.h"
#include "server-runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
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

json server_emotive_trace_to_json(const server_emotive_trace & trace) {
    if (!trace.valid) {
        return nullptr;
    }

    json blocks = json::array();
    for (const server_emotive_block_record & block : trace.blocks) {
        blocks.push_back({
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
        });
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
    };
}

server_emotive_runtime::server_emotive_runtime(const server_emotive_runtime_config & config) :
        config_(config) {
    std::string error;
    embedding_backend_.configure(config.embedding, &error);
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

void server_emotive_runtime::remember_trace(const server_emotive_trace & trace) {
    if (!trace.valid) {
        return;
    }

    std::lock_guard<std::mutex> lock(history_mutex_);
    latest_traces_.push_back(trace);
    while ((int32_t) latest_traces_.size() > config_.max_turn_history) {
        latest_traces_.pop_front();
    }
}

json server_emotive_runtime::health_json() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    return {
        {"enabled", config_.enabled},
        {"block_max_chars", config_.block_max_chars},
        {"max_blocks_per_turn", config_.max_blocks_per_turn},
        {"max_turn_history", config_.max_turn_history},
        {"vad_ema_alpha", config_.vad_ema_alpha},
        {"latest_turn_count", (int32_t) latest_traces_.size()},
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

server_emotive_turn_builder::server_emotive_turn_builder(server_emotive_runtime & runtime, const std::string & model_name) :
        runtime_(runtime),
        model_name_(model_name),
        pending_kind_(SERVER_EMOTIVE_BLOCK_USER_MESSAGE),
        has_pending_(false),
        user_anchor_count_(0),
        have_previous_moment_(false),
        have_previous_vad_(false) {
}

server_emotive_turn_builder::~server_emotive_turn_builder() {
}

void server_emotive_turn_builder::add_user_message(const std::string & text) {
    append_text(SERVER_EMOTIVE_BLOCK_USER_MESSAGE, text);
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
    if (!blocks_.empty()) {
        trace.final_moment = blocks_.back().moment;
        trace.final_vad = blocks_.back().vad;
    }
    runtime_.remember_trace(trace);
    return trace;
}
