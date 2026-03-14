#include "llama-self-state.h"

#include "llama-context.h"
#include "llama-hard-memory.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_set>

namespace {

static constexpr uint32_t LLAMA_SELF_UPDATER_VERSION = 2;
static constexpr float    LLAMA_SELF_DEFAULT_TOOL_SALIENCE_HALF_LIFE_MS = 300000.0f;
static constexpr float    LLAMA_SELF_EVOLUTION_PROGRESS_HORIZON_MS = 45.0f * 60.0f * 1000.0f;
static constexpr float    LLAMA_SELF_EVOLUTION_UPDATE_HORIZON_MS = 5.0f * 60.0f * 1000.0f;
static constexpr float    LLAMA_SELF_TWO_PI = 6.28318530717958647692f;
static constexpr size_t   LLAMA_SELF_MAX_WORKING_MEMORY_ITEMS = 32;
static constexpr size_t   LLAMA_SELF_MAX_GOALS = 16;
static constexpr size_t   LLAMA_SELF_MAX_COMMITMENTS = 32;
static constexpr size_t   LLAMA_SELF_MAX_MEMORY_HANDLES = 24;
static constexpr size_t   LLAMA_SELF_MAX_TRACE_ITEMS = 256;
static constexpr uint32_t LLAMA_SELF_TRACE_MAGIC = 0x4c535354u; // LSST
static constexpr uint32_t LLAMA_SELF_TRACE_VERSION = 2;
static constexpr size_t   LLAMA_SELF_BELIEF_SIGNATURE_DIM = 4;

static int64_t current_wall_clock_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static int64_t current_monotonic_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static bool localtime_safe(std::time_t t, std::tm * out_tm) {
#if defined(_WIN32)
    return localtime_s(out_tm, &t) == 0;
#else
    return localtime_r(&t, out_tm) != nullptr;
#endif
}

static bool gmtime_safe(std::time_t t, std::tm * out_tm) {
#if defined(_WIN32)
    return gmtime_s(out_tm, &t) == 0;
#else
    return gmtime_r(&t, out_tm) != nullptr;
#endif
}

static int32_t compute_timezone_offset_minutes(int64_t wall_clock_ms) {
    const std::time_t t = (std::time_t) (wall_clock_ms / 1000);

    std::tm local_tm = {};
    std::tm gm_tm = {};
    if (!localtime_safe(t, &local_tm) || !gmtime_safe(t, &gm_tm)) {
        return 0;
    }

    local_tm.tm_isdst = -1;
    gm_tm.tm_isdst = 0;

    const std::time_t local_seconds = std::mktime(&local_tm);
    const std::time_t gm_seconds = std::mktime(&gm_tm);

    return (int32_t) std::difftime(local_seconds, gm_seconds) / 60;
}

static float clamp_unit(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

static float clamp_range(float value, float lo, float hi) {
    return std::min(hi, std::max(lo, value));
}

static float clamp_signed_unit(float value) {
    return clamp_range(value, -1.0f, 1.0f);
}

static std::string trim_text(const std::string & text, size_t max_chars) {
    if (text.size() <= max_chars) {
        return text;
    }
    return text.substr(0, max_chars);
}

static float blend_value(float current, float target, float gain) {
    return current + clamp_unit(gain) * (target - current);
}

static float max4(float a, float b, float c, float d) {
    return std::max(std::max(a, b), std::max(c, d));
}

static std::vector<float> capture_self_register_scalars(const llama_context & ctx) {
    const int32_t count = ctx.self_state_register_count();
    std::vector<float> values(std::max(0, count), 0.0f);
    for (int32_t i = 0; i < count; ++i) {
        llama_self_register_info info = {};
        if (ctx.self_state_get_register(i, &info)) {
            values[i] = info.scalar_value;
        }
    }
    return values;
}

static llama_self_state_delta_summary summarize_self_state_delta(
        const std::vector<float> & before,
        const std::vector<float> & after,
        const llama_self_state_event & event) {
    llama_self_state_delta_summary summary = {};
    summary.role = event.role;
    summary.channel = event.channel;
    summary.flags = event.flags;

    struct delta_item {
        int32_t register_id = -1;
        float before_value = 0.0f;
        float after_value = 0.0f;
        float abs_delta = 0.0f;
    };

    std::vector<delta_item> deltas;
    const int32_t count = std::min((int32_t) before.size(), (int32_t) after.size());
    deltas.reserve(std::max(0, count));

    for (int32_t i = 0; i < count; ++i) {
        const float abs_delta = std::fabs(after[i] - before[i]);
        if (abs_delta <= 1.0e-6f) {
            continue;
        }
        summary.total_delta += abs_delta;
        summary.max_delta = std::max(summary.max_delta, abs_delta);
        deltas.push_back({
            /*.register_id =*/ i,
            /*.before_value =*/ before[i],
            /*.after_value =*/ after[i],
            /*.abs_delta =*/ abs_delta,
        });
    }

    std::sort(deltas.begin(), deltas.end(), [](const delta_item & lhs, const delta_item & rhs) {
        if (lhs.abs_delta == rhs.abs_delta) {
            return lhs.register_id < rhs.register_id;
        }
        return lhs.abs_delta > rhs.abs_delta;
    });

    summary.dimension_count = std::min<int32_t>(deltas.size(), LLAMA_SELF_STATE_MAX_DELTA_DIMS);
    for (int32_t i = 0; i < summary.dimension_count; ++i) {
        summary.dimensions[i].register_id = deltas[i].register_id;
        summary.dimensions[i].before_value = deltas[i].before_value;
        summary.dimensions[i].after_value = deltas[i].after_value;
        summary.dimensions[i].abs_delta = deltas[i].abs_delta;
    }

    return summary;
}

static float sigmoid_unit(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

template<size_t N>
static float dot_product(const std::array<float, N> & lhs, const std::array<float, N> & rhs) {
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

template<size_t N>
static float l1_norm(const std::array<float, N> & values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += std::fabs(value);
    }
    return sum;
}

template<size_t N>
static float normalized_entropy(const std::array<float, N> & values) {
    const float total = l1_norm(values);
    if (total <= 1.0e-6f) {
        return 0.0f;
    }

    float entropy = 0.0f;
    for (float value : values) {
        const float p = std::fabs(value) / total;
        if (p > 1.0e-6f) {
            entropy -= p * std::log(p);
        }
    }
    return clamp_unit(entropy / std::log((float) N));
}

template<size_t N>
static float linear_probe_score(float bias, const std::array<float, N> & weights, const std::array<float, N> & features) {
    float value = bias;
    for (size_t i = 0; i < N; ++i) {
        value += weights[i] * features[i];
    }
    return sigmoid_unit(value);
}

static int64_t elapsed_or_unset(int64_t now_ms, int64_t anchor_ms) {
    return anchor_ms < 0 ? -1 : std::max<int64_t>(0, now_ms - anchor_ms);
}

static float decay_to_unit(int64_t delta_ms, float half_life_ms) {
    if (delta_ms < 0 || half_life_ms <= 0.0f) {
        return 0.0f;
    }

    const float decay = std::exp(-std::log(2.0f) * (float) delta_ms / half_life_ms);
    return clamp_unit(decay);
}

static std::string normalize_piece(const std::string & piece) {
    std::string out;
    out.reserve(piece.size());

    for (char ch : piece) {
        const unsigned char uch = (unsigned char) ch;
        if (std::isalnum(uch) || ch == '\'' || ch == '?') {
            out.push_back((char) std::tolower(uch));
        }
    }

    return out;
}

static bool contains_any(const std::string & piece, const char * const * patterns, size_t n_patterns) {
    for (size_t i = 0; i < n_patterns; ++i) {
        if (piece.find(patterns[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static llama_self_register_updater_rule make_rule(
        int32_t register_id,
        uint32_t phase_mask,
        float baseline,
        float rise_gain,
        float fall_gain,
        float baseline_pull,
        std::initializer_list<std::pair<int32_t, float>> feature_terms,
        std::initializer_list<std::pair<int32_t, float>> source_terms = {}) {
    llama_self_register_updater_rule rule = {};
    rule.register_id = register_id;
    rule.phase_mask = phase_mask;
    rule.baseline = baseline;
    rule.rise_gain = rise_gain;
    rule.fall_gain = fall_gain;
    rule.baseline_pull = baseline_pull;

    for (size_t i = 0; i < LLAMA_SELF_MAX_UPDATER_RULE_TERMS; ++i) {
        rule.feature_ids[i] = LLAMA_SELF_UPDATER_FEATURE_NONE;
    }
    for (size_t i = 0; i < LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS; ++i) {
        rule.source_register_ids[i] = -1;
    }

    size_t idx = 0;
    for (const auto & term : feature_terms) {
        if (idx >= LLAMA_SELF_MAX_UPDATER_RULE_TERMS) {
            break;
        }
        rule.feature_ids[idx] = term.first;
        rule.feature_weights[idx] = term.second;
        ++idx;
    }

    idx = 0;
    for (const auto & term : source_terms) {
        if (idx >= LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS) {
            break;
        }
        rule.source_register_ids[idx] = term.first;
        rule.source_register_weights[idx] = term.second;
        ++idx;
    }

    return rule;
}

static std::array<float, 32> build_event_sketch(const llama_self_state_event & event) {
    std::array<float, 32> sketch = {};

    if (!event.tokens || event.n_tokens == 0) {
        return sketch;
    }

    for (size_t i = 0; i < event.n_tokens; ++i) {
        const uint32_t token = (uint32_t) event.tokens[i];
        const size_t dim = token % sketch.size();
        const float sign = (token & 1u) ? -1.0f : 1.0f;
        sketch[dim] += sign;
    }

    float norm = 0.0f;
    for (float value : sketch) {
        norm += value * value;
    }
    norm = std::sqrt(norm);

    if (norm > 0.0f) {
        for (float & value : sketch) {
            value /= norm;
        }
    }

    return sketch;
}

static std::array<float, 32> build_token_sketch(const llama_token * tokens, size_t n_tokens) {
    llama_self_state_event event = {
        /*.tokens =*/ tokens,
        /*.n_tokens =*/ n_tokens,
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ 0,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 0.0f,
    };
    return build_event_sketch(event);
}

static std::array<float, 32> build_text_sketch(const std::string & text) {
    std::array<float, 32> sketch = {};
    if (text.empty()) {
        return sketch;
    }

    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char ch = (unsigned char) text[i];
        const size_t dim = (size_t) ((ch + (unsigned char) (i * 17u)) % sketch.size());
        const float sign = (ch & 1u) ? -1.0f : 1.0f;
        sketch[dim] += sign;
    }

    float norm = 0.0f;
    for (float value : sketch) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float & value : sketch) {
            value /= norm;
        }
    }
    return sketch;
}

static float sketch_similarity(const std::array<float, 32> & lhs, const std::array<float, 32> & rhs) {
    float dot = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        dot += lhs[i] * rhs[i];
    }
    return std::min(1.0f, std::max(-1.0f, dot));
}

template<size_t N>
static void copy_bounded_cstr(char (&dst)[N], const char * src) {
    std::memset(dst, 0, sizeof(dst));
    if (!src || N == 0) {
        return;
    }
    const size_t copy_len = std::min(std::strlen(src), N - 1);
    if (copy_len > 0) {
        std::memcpy(dst, src, copy_len);
    }
}

template<size_t N>
static std::string read_bounded_cstr(const char (&src)[N]) {
    return std::string(src, strnlen(src, N));
}

static uint64_t fnv1a64_self_state(const std::string & text) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= (uint64_t) ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

static std::string register_delta_summary_self_state(const llama_self_state_delta_summary & delta) {
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

static bool sketch_is_zero(const std::array<float, 32> & sketch) {
    for (float value : sketch) {
        if (std::fabs(value) > 1.0e-6f) {
            return false;
        }
    }
    return true;
}

struct extension_domain_signal {
    float goal_progress = 0.0f;
    float user_outcome = 0.0f;
    float epistemic = 0.0f;
    float efficiency = 0.0f;
    float recovery = 0.0f;
    float strategy = 0.0f;
    float self_improvement = 0.0f;
};

static float extension_context_similarity(
        const std::array<float, 32> & lhs,
        const std::array<float, 32> & rhs) {
    if (sketch_is_zero(lhs) || sketch_is_zero(rhs)) {
        return 0.5f;
    }
    return clamp_unit(0.5f * (1.0f + sketch_similarity(lhs, rhs)));
}

static void apply_extension_signal_to_horizon(
        llama_self_model_horizon_info & horizon,
        const extension_domain_signal & signal) {
    if (signal.goal_progress > 0.0f) {
        horizon.goal_progress.goal_progress_estimate = clamp_unit(horizon.goal_progress.goal_progress_estimate + 0.08f * signal.goal_progress);
        horizon.goal_progress.blocker_severity = clamp_unit(horizon.goal_progress.blocker_severity - 0.08f * signal.goal_progress);
        horizon.goal_progress.dependency_readiness = clamp_unit(horizon.goal_progress.dependency_readiness + 0.07f * signal.goal_progress);
        horizon.goal_progress.expected_next_action_gain = clamp_unit(horizon.goal_progress.expected_next_action_gain + 0.10f * signal.goal_progress);
    }
    if (signal.user_outcome > 0.0f) {
        horizon.user_outcome.satisfaction_estimate = clamp_unit(horizon.user_outcome.satisfaction_estimate + 0.06f * signal.user_outcome);
        horizon.user_outcome.misunderstanding_risk = clamp_unit(horizon.user_outcome.misunderstanding_risk - 0.10f * signal.user_outcome);
        horizon.user_outcome.preference_uncertainty = clamp_unit(horizon.user_outcome.preference_uncertainty - 0.08f * signal.user_outcome);
        horizon.user_outcome.trust_repair_need = clamp_unit(horizon.user_outcome.trust_repair_need - 0.05f * signal.user_outcome);
    }
    if (signal.epistemic > 0.0f) {
        horizon.epistemic.answerability = clamp_unit(horizon.epistemic.answerability + 0.12f * signal.epistemic);
        horizon.epistemic.evidence_sufficiency = clamp_unit(horizon.epistemic.evidence_sufficiency + 0.14f * signal.epistemic);
        horizon.epistemic.ambiguity_concentration = clamp_unit(horizon.epistemic.ambiguity_concentration - 0.10f * signal.epistemic);
        horizon.epistemic.self_estimate_confidence = clamp_unit(horizon.epistemic.self_estimate_confidence + 0.10f * signal.epistemic);
    }
    if (signal.efficiency > 0.0f) {
        horizon.efficiency.expected_steps_remaining = clamp_unit(horizon.efficiency.expected_steps_remaining - 0.10f * signal.efficiency);
        horizon.efficiency.expected_inference_cost_remaining = clamp_unit(horizon.efficiency.expected_inference_cost_remaining - 0.10f * signal.efficiency);
        horizon.efficiency.loop_inefficiency = clamp_unit(horizon.efficiency.loop_inefficiency - 0.09f * signal.efficiency);
        horizon.efficiency.context_thrash_risk = clamp_unit(horizon.efficiency.context_thrash_risk - 0.07f * signal.efficiency);
    }
    if (signal.recovery > 0.0f) {
        horizon.recovery.recovery_momentum = clamp_unit(horizon.recovery.recovery_momentum + 0.08f * signal.recovery);
        horizon.recovery.regulation_debt = clamp_unit(horizon.recovery.regulation_debt - 0.08f * signal.recovery);
        horizon.recovery.unresolved_tension_load = clamp_unit(horizon.recovery.unresolved_tension_load - 0.08f * signal.recovery);
        horizon.recovery.recovery_cost_estimate = clamp_unit(horizon.recovery.recovery_cost_estimate - 0.08f * signal.recovery);
    }
    if (signal.strategy > 0.0f) {
        horizon.strategy.answer_bias = clamp_unit(horizon.strategy.answer_bias + 0.06f * signal.strategy);
        horizon.strategy.act_bias = clamp_unit(horizon.strategy.act_bias + 0.06f * signal.strategy);
        horizon.strategy.deliberate_bias = clamp_unit(horizon.strategy.deliberate_bias + 0.05f * signal.strategy);
        horizon.strategy.act_external_bias = clamp_unit(horizon.strategy.act_external_bias + 0.05f * signal.strategy);
    }
    if (signal.self_improvement > 0.0f) {
        horizon.self_improvement.expected_gain = clamp_unit(horizon.self_improvement.expected_gain + 0.07f * signal.self_improvement);
        horizon.self_improvement.evidence_deficit = clamp_unit(horizon.self_improvement.evidence_deficit - 0.08f * signal.self_improvement);
        horizon.self_improvement.readiness = clamp_unit(horizon.self_improvement.readiness + 0.08f * signal.self_improvement);
    }
}

static float extension_utility_score(const llama_self_model_horizon_info & horizon) {
    return clamp_unit(
            0.26f * horizon.epistemic.answerability +
            0.20f * horizon.epistemic.evidence_sufficiency +
            0.18f * horizon.goal_progress.expected_next_action_gain +
            0.14f * (1.0f - horizon.user_outcome.misunderstanding_risk) +
            0.12f * (1.0f - horizon.efficiency.expected_steps_remaining) +
            0.10f * (1.0f - horizon.efficiency.expected_inference_cost_remaining));
}

static const char * hard_memory_domain_tag(int32_t domain) {
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

static std::string render_event_text(const llama_vocab * vocab, const llama_self_state_event & event) {
    if (!vocab || !event.tokens || event.n_tokens == 0) {
        return {};
    }

    std::string out;
    for (size_t i = 0; i < event.n_tokens; ++i) {
        out += vocab->token_to_piece(event.tokens[i]);
    }
    return out;
}

static int32_t dominant_hard_memory_domain(const llama_self_state_delta_summary & delta) {
    float goal = 0.0f;
    float user = 0.0f;
    float epistemic = 0.0f;
    float efficiency = 0.0f;
    float recovery = 0.0f;
    for (int32_t i = 0; i < delta.dimension_count; ++i) {
        const float abs_delta = delta.dimensions[i].abs_delta;
        switch (delta.dimensions[i].register_id) {
            case LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE: goal += abs_delta; break;
            case LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK:
            case LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY: user += abs_delta; break;
            case LLAMA_SELF_REGISTER_ANSWERABILITY:
            case LLAMA_SELF_REGISTER_UNCERTAINTY:
            case LLAMA_SELF_REGISTER_CONTRADICTION: epistemic += abs_delta; break;
            case LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY: efficiency += abs_delta; break;
            case LLAMA_SELF_REGISTER_RECOVERY_URGENCY: recovery += abs_delta; break;
            default: break;
        }
    }

    int32_t domain = LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
    float best = epistemic;
    if (goal > best) {
        best = goal;
        domain = LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS;
    }
    if (user > best) {
        best = user;
        domain = LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME;
    }
    if (efficiency > best) {
        best = efficiency;
        domain = LLAMA_HARD_MEMORY_DOMAIN_EFFICIENCY;
    }
    if (recovery > best) {
        domain = LLAMA_HARD_MEMORY_DOMAIN_RECOVERY;
    }
    return domain;
}

static bool append_hard_memory_primitive(
        llama_hard_memory_primitive (&primitives)[LLAMA_HARD_MEMORY_MAX_PRIMITIVES],
        int32_t * primitive_count,
        const llama_hard_memory_primitive & primitive) {
    if (!primitive_count || *primitive_count < 0 || *primitive_count >= LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
        return false;
    }
    primitives[*primitive_count] = primitive;
    *primitive_count += 1;
    return true;
}

static int32_t build_postwrite_hard_memory_primitives(
        llama_context & ctx,
        const llama_self_state_event & event,
        const llama_self_state_delta_summary & delta,
        llama_hard_memory_primitive (&primitives)[LLAMA_HARD_MEMORY_MAX_PRIMITIVES]) {
    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
    const std::string text = render_event_text(vocab, event);
    if (text.empty()) {
        return 0;
    }

    const int32_t domain = dominant_hard_memory_domain(delta);
    const float delta_signal = clamp_unit(delta.total_delta / 2.0f);
    const uint64_t base_hash = fnv1a64_self_state(text + "|" + std::to_string(delta.dimension_count));
    int32_t primitive_count = 0;

    llama_hard_memory_primitive event_primitive = llama_hard_memory_default_primitive();
    event_primitive.kind = LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT;
    event_primitive.domain = domain;
    event_primitive.source_role = event.role;
    event_primitive.source_channel = event.channel;
    event_primitive.source_tool_kind = event.role == LLAMA_SELF_STATE_EVENT_TOOL ? LLAMA_TOOL_KIND_GENERIC : LLAMA_TOOL_KIND_NONE;
    event_primitive.transaction_id = (int32_t) (base_hash & 0x7fffffff);
    event_primitive.importance = clamp_unit(0.30f + 0.35f * delta_signal + 0.35f * delta.max_delta);
    event_primitive.confidence = clamp_unit(0.55f + 0.25f * delta.max_delta);
    event_primitive.gain_bias = clamp_unit(0.20f + 0.60f * delta.max_delta);
    event_primitive.allostatic_relevance = 0.0f;
    std::snprintf(event_primitive.key, sizeof(event_primitive.key), "event:%08llx",
            (unsigned long long) (base_hash & 0xffffffffull));
    copy_bounded_cstr(event_primitive.title, "self-state event fragment");
    std::ostringstream event_content;
    event_content.setf(std::ios::fixed);
    event_content.precision(3);
    event_content << "role=" << event.role
                  << " channel=" << event.channel
                  << " total_delta=" << delta.total_delta
                  << " max_delta=" << delta.max_delta
                  << " changed_registers=" << register_delta_summary_self_state(delta)
                  << " message=" << text;
    copy_bounded_cstr(event_primitive.content, trim_text(event_content.str(), LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1).c_str());
    copy_bounded_cstr(event_primitive.tags[0], "event");
    copy_bounded_cstr(event_primitive.tags[1], hard_memory_domain_tag(domain));
    copy_bounded_cstr(event_primitive.tags[2], event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ? "counterfactual" : "primary");
    (void) append_hard_memory_primitive(primitives, &primitive_count, event_primitive);

    llama_self_social_state_info social = {};
    if (ctx.self_state_get_social_state(&social)) {
        const float user_signal = clamp_unit(std::max(
                social.dissatisfaction,
                std::max(1.0f - social.trust, std::fabs(social.recent_user_valence))));
        if (user_signal > 0.25f && primitive_count < LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
            llama_self_model_state_info model_state = {};
            const bool have_model_state = ctx.self_state_get_model_state(&model_state);
            llama_hard_memory_primitive user_primitive = llama_hard_memory_default_primitive();
            user_primitive.kind = LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL;
            user_primitive.domain = LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME;
            user_primitive.source_role = event.role;
            user_primitive.source_channel = event.channel;
            user_primitive.transaction_id = event_primitive.transaction_id;
            user_primitive.importance = clamp_unit(0.25f + 0.55f * user_signal);
            user_primitive.confidence = clamp_unit(0.45f + 0.45f * social.familiarity);
            user_primitive.gain_bias = clamp_unit(0.20f + 0.45f * user_signal);
            user_primitive.allostatic_relevance = clamp_unit(0.20f + 0.50f * social.dissatisfaction);
            user_primitive.flags |= LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_ALLOSTASIS;
            std::snprintf(user_primitive.key, sizeof(user_primitive.key), "user_model:%08llx",
                    (unsigned long long) ((base_hash >> 8) & 0xffffffffull));
            copy_bounded_cstr(user_primitive.title, "user model fragment");
            std::ostringstream user_content;
            user_content.setf(std::ios::fixed);
            user_content.precision(3);
            user_content << "trust=" << social.trust
                         << " dissatisfaction=" << social.dissatisfaction
                         << " reciprocity=" << social.reciprocity
                         << " bond=" << social.bond_strength
                         << " recent_user_valence=" << social.recent_user_valence;
            if (have_model_state) {
                const auto & pref = model_state.horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT].user_preference;
                user_content << " directness=" << pref.directness_preference
                             << " verbosity=" << pref.verbosity_preference
                             << " structure=" << pref.structure_preference
                             << " clarification=" << pref.clarification_preference
                             << " autonomy=" << pref.autonomy_preference
                             << " disagreement_sensitivity=" << pref.disagreement_sensitivity
                             << " rhetorical_intensity=" << pref.rhetorical_intensity
                             << " pref_confidence=" << pref.preference_confidence
                             << " rhetoric_confidence=" << pref.rhetorical_confidence;
            }
            copy_bounded_cstr(user_primitive.content, trim_text(user_content.str(), LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1).c_str());
            copy_bounded_cstr(user_primitive.tags[0], "user_model");
            copy_bounded_cstr(user_primitive.tags[1], have_model_state ? "preference" : "social");
            copy_bounded_cstr(user_primitive.tags[2], have_model_state ? "rhetoric" : "user_outcome");
            (void) append_hard_memory_primitive(primitives, &primitive_count, user_primitive);
        }
    }

    llama_self_model_state_info model_state = {};
    if (ctx.self_state_get_model_state(&model_state) && primitive_count < LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
        const auto & instant = model_state.horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT];
        const float model_signal = clamp_unit(std::max(
                model_state.extension_summary.gain_signal_abs,
                std::max(model_state.extension_summary.context_activation,
                         std::fabs(model_state.forecast.valid ? model_state.forecast.predicted_satisfaction_delta : 0.0f))));
        if (model_signal > 0.18f) {
            llama_hard_memory_primitive self_primitive = llama_hard_memory_default_primitive();
            self_primitive.kind = LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT;
            self_primitive.domain = domain;
            self_primitive.source_role = event.role;
            self_primitive.source_channel = event.channel;
            self_primitive.transaction_id = event_primitive.transaction_id;
            self_primitive.importance = clamp_unit(0.20f + 0.60f * model_signal);
            self_primitive.confidence = clamp_unit(0.40f + 0.50f * instant.epistemic.self_estimate_confidence);
            self_primitive.gain_bias = clamp_unit(0.20f + 0.40f * model_state.extension_summary.gain_signal_abs);
            self_primitive.allostatic_relevance = clamp_unit(model_state.extension_summary.allostatic_divergence);
            std::snprintf(self_primitive.key, sizeof(self_primitive.key), "self_model:%08llx",
                    (unsigned long long) ((base_hash >> 16) & 0xffffffffull));
            copy_bounded_cstr(self_primitive.title, "self model fragment");
            std::ostringstream self_content;
            self_content.setf(std::ios::fixed);
            self_content.precision(3);
            self_content << "answerability=" << instant.epistemic.answerability
                         << " preference_uncertainty=" << instant.user_outcome.preference_uncertainty
                         << " loop_inefficiency=" << instant.efficiency.loop_inefficiency
                         << " recovery_cost=" << instant.recovery.recovery_cost_estimate
                         << " extension_gain_signal=" << model_state.extension_summary.gain_signal
                         << " extension_context_activation=" << model_state.extension_summary.context_activation;
            copy_bounded_cstr(self_primitive.content, trim_text(self_content.str(), LLAMA_HARD_MEMORY_MAX_TEXT_CHARS - 1).c_str());
            copy_bounded_cstr(self_primitive.tags[0], "self_model");
            copy_bounded_cstr(self_primitive.tags[1], hard_memory_domain_tag(domain));
            copy_bounded_cstr(self_primitive.tags[2], "control_state");
            (void) append_hard_memory_primitive(primitives, &primitive_count, self_primitive);
        }
    }

    return primitive_count;
}

static uint32_t source_mask_for_role(const llama_self_state_event & event) {
    uint32_t mask = 0;
    switch (event.role) {
        case LLAMA_SELF_STATE_EVENT_USER:   mask |= LLAMA_SELF_SOURCE_USER_EVENT; break;
        case LLAMA_SELF_STATE_EVENT_TOOL:   mask |= LLAMA_SELF_SOURCE_TOOL_EVENT; break;
        case LLAMA_SELF_STATE_EVENT_SYSTEM: mask |= LLAMA_SELF_SOURCE_EMIT_EVENT; break;
        default:                            mask = 0; break;
    }
    if (event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
        mask |= LLAMA_SELF_SOURCE_COUNTERFACTUAL;
    }
    return mask;
}

static llama_self_tool_state_info build_tool_state_info(const std::vector<llama_self_tool_job> & tool_jobs) {
    llama_self_tool_state_info info = {};
    info.active_status = LLAMA_SELF_TOOL_JOB_IDLE;
    info.readiness = 0.5f;
    info.last_update_monotonic_ms = -1;

    float pending_pressure = 0.0f;
    float running_pressure = 0.0f;
    float completed_signal = 0.0f;
    float failed_signal = 0.0f;

    for (const auto & job : tool_jobs) {
        info.last_update_monotonic_ms = std::max(info.last_update_monotonic_ms, job.last_update_monotonic_ms);

        switch (job.status) {
            case LLAMA_SELF_TOOL_JOB_PENDING:
                ++info.pending_jobs;
                pending_pressure = std::max(pending_pressure, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_PENDING);
                break;
            case LLAMA_SELF_TOOL_JOB_RUNNING:
                ++info.running_jobs;
                running_pressure = std::max(running_pressure, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_RUNNING);
                break;
            case LLAMA_SELF_TOOL_JOB_COMPLETED:
                ++info.completed_jobs;
                completed_signal = std::max(completed_signal, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_COMPLETED);
                break;
            case LLAMA_SELF_TOOL_JOB_FAILED:
                ++info.failed_jobs;
                failed_signal = std::max(failed_signal, job.importance);
                info.active_status = std::max(info.active_status, (int32_t) LLAMA_SELF_TOOL_JOB_FAILED);
                break;
            default:
                break;
        }
    }

    if (info.failed_jobs > 0) {
        info.readiness = clamp_unit(0.15f * (1.0f - failed_signal));
    } else if (info.running_jobs > 0) {
        info.readiness = clamp_unit(0.15f + 0.25f * (1.0f - running_pressure));
    } else if (info.pending_jobs > 0) {
        info.readiness = clamp_unit(0.45f - 0.20f * pending_pressure);
    } else if (info.completed_jobs > 0) {
        info.readiness = clamp_unit(0.75f + 0.20f * completed_signal);
    }

    return info;
}

static void accumulate_model_extensions(
        std::vector<llama_self_model_extension_entry> & model_extensions,
        const std::array<float, 32> & context_sketch,
        bool increment_activation_count,
        extension_domain_signal * out_signal,
        llama_self_model_extension_summary * out_summary) {
    extension_domain_signal signal = {};
    llama_self_model_extension_summary summary = {};
    float confidence_sum = 0.0f;
    float salience_sum = 0.0f;
    float gain_weight_sum = 0.0f;
    float allostatic_weight_sum = 0.0f;
    float context_sum = 0.0f;

    for (auto & extension : model_extensions) {
        if ((extension.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_ACTIVE) == 0) {
            continue;
        }

        ++summary.active_count;
        confidence_sum += extension.confidence;
        salience_sum += extension.salience;
        summary.max_salience = std::max(summary.max_salience, extension.salience);
        if (extension.source == LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_HARD_MEMORY) {
            ++summary.hard_memory_count;
        } else if (extension.source == LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_BASH_CLI ||
                   extension.source == LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_EXTERNAL) {
            ++summary.tool_count;
        }

        const float context_similarity = extension.kind == LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT ?
                extension_context_similarity(extension.sketch, context_sketch) :
                1.0f;
        const bool has_desired_state =
                (extension.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_HAS_DESIRED_STATE) != 0;
        const float lane_value = extension.kind == LLAMA_SELF_MODEL_EXTENSION_SCALAR_PARAM && has_desired_state ?
                clamp_unit(1.0f - std::fabs(extension.desired_value - extension.value)) :
                clamp_unit(extension.value);
        const float activation = clamp_unit(lane_value * extension.salience * extension.confidence * context_similarity);
        context_sum += activation;
        if (increment_activation_count && activation > 0.05f) {
            extension.activation_count += 1;
        }

        const float domain_signal = clamp_unit(
                activation * (extension.kind == LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT ? 1.0f : 0.85f));
        switch (extension.domain) {
            case LLAMA_SELF_MODEL_EXTENSION_DOMAIN_GOAL_PROGRESS:    signal.goal_progress = clamp_unit(signal.goal_progress + domain_signal); break;
            case LLAMA_SELF_MODEL_EXTENSION_DOMAIN_USER_OUTCOME:     signal.user_outcome = clamp_unit(signal.user_outcome + domain_signal); break;
            case LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EPISTEMIC:        signal.epistemic = clamp_unit(signal.epistemic + domain_signal); break;
            case LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EFFICIENCY:       signal.efficiency = clamp_unit(signal.efficiency + domain_signal); break;
            case LLAMA_SELF_MODEL_EXTENSION_DOMAIN_RECOVERY:         signal.recovery = clamp_unit(signal.recovery + domain_signal); break;
            case LLAMA_SELF_MODEL_EXTENSION_DOMAIN_STRATEGY:         signal.strategy = clamp_unit(signal.strategy + domain_signal); break;
            case LLAMA_SELF_MODEL_EXTENSION_DOMAIN_SELF_IMPROVEMENT: signal.self_improvement = clamp_unit(signal.self_improvement + domain_signal); break;
            default: break;
        }

        if ((extension.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN) != 0) {
            ++summary.gain_count;
            const float signed_gain = has_desired_state ?
                    clamp_signed_unit(extension.desired_value - extension.value) :
                    activation;
            gain_weight_sum += std::max(0.0f, extension.gain_weight);
            summary.gain_signal += extension.gain_weight * signed_gain;
            summary.gain_signal_abs += extension.gain_weight * std::fabs(signed_gain);
        }

        if ((extension.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS) != 0 && has_desired_state) {
            ++summary.allostatic_count;
            allostatic_weight_sum += std::max(0.0f, extension.allostatic_weight);
            summary.allostatic_divergence += extension.allostatic_weight * std::fabs(extension.desired_value - extension.value);
        }
    }

    if (summary.active_count > 0) {
        summary.mean_confidence = clamp_unit(confidence_sum / (float) summary.active_count);
        summary.mean_salience = clamp_unit(salience_sum / (float) summary.active_count);
        summary.context_activation = clamp_unit(context_sum / (float) summary.active_count);
    }
    if (gain_weight_sum > 0.0f) {
        summary.gain_signal = clamp_signed_unit(summary.gain_signal / gain_weight_sum);
        summary.gain_signal_abs = clamp_unit(summary.gain_signal_abs / gain_weight_sum);
    } else {
        summary.gain_signal = 0.0f;
        summary.gain_signal_abs = 0.0f;
    }
    if (allostatic_weight_sum > 0.0f) {
        summary.allostatic_divergence = clamp_unit(summary.allostatic_divergence / allostatic_weight_sum);
    } else {
        summary.allostatic_divergence = 0.0f;
    }

    if (out_signal) {
        *out_signal = signal;
    }
    if (out_summary) {
        *out_summary = summary;
    }
}

template<typename T>
static void append_bytes(std::vector<uint8_t> & out, const T & value) {
    const size_t offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

template<typename T>
static bool read_bytes(const uint8_t * src, size_t size, size_t * cursor, T * out_value) {
    if (!src || !cursor || !out_value || *cursor + sizeof(T) > size) {
        return false;
    }

    std::memcpy(out_value, src + *cursor, sizeof(T));
    *cursor += sizeof(T);
    return true;
}

static void repair_trace_item_pointers(std::vector<llama_self_trace_item> & items) {
    for (auto & item : items) {
        item.event.tokens = item.tokens.empty() ? nullptr : item.tokens.data();
        item.event.n_tokens = item.tokens.size();
    }
}

static std::array<float, 32> build_frozen_bucket_sketch(int32_t bucket_id, const llama_past_lora_bucket_stats & stats) {
    std::array<float, 32> sketch = {};
    const float values[] = {
        (float) bucket_id,
        (float) stats.version,
        stats.base_scale,
        stats.effective_scale,
        stats.gain_mean,
        stats.gain_max,
        stats.populated ? 1.0f : 0.0f,
    };

    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        const size_t dim = (bucket_id * 7 + (int32_t) i * 5) % sketch.size();
        sketch[dim] += values[i];
    }

    float norm = 0.0f;
    for (float value : sketch) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float & value : sketch) {
            value /= norm;
        }
    }

    return sketch;
}

} // namespace

std::array<llama_self_register_definition, LLAMA_SELF_REGISTER_COUNT> llama_self_state::build_definitions() {
    return {{
        { LLAMA_SELF_REGISTER_UNCERTAINTY,            "r_uncertainty",            LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_CONTRADICTION,         "r_contradiction",          LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_NOVELTY,               "r_novelty",                LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_TOPIC_SHIFT,           "r_topic_shift",            LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_GOAL_RELEVANCE,        "r_goal_relevance",         LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_SELF_RELEVANCE,        "r_self_relevance",         LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE,      "r_social_relevance",       LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_AFFORDANCE,            "r_affordance",             LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_BROADCAST_PRESSURE,    "r_broadcast_pressure",     LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_BROADCAST_INHIBITION,  "r_broadcast_inhibition",   LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION, "r_followup_continuation",  LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY, "r_memory_write_priority",  LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_TIME_PHASE,            "r_time_phase",             LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_TOOL_SALIENCE,         "r_tool_salience",          LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK,"r_user_satisfaction_risk",LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE,"r_goal_progress_pressure",LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY,     "r_loop_inefficiency",      LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_RECOVERY_URGENCY,      "r_recovery_urgency",       LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_ANSWERABILITY,         "r_answerability",          LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY,"r_preference_uncertainty", LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_USER_DIRECTNESS_PREFERENCE, "r_user_directness_preference", LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.5f, 0 },
        { LLAMA_SELF_REGISTER_USER_VERBOSITY_PREFERENCE, "r_user_verbosity_preference", LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.5f, 0 },
        { LLAMA_SELF_REGISTER_USER_STRUCTURE_PREFERENCE, "r_user_structure_preference", LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.5f, 0 },
        { LLAMA_SELF_REGISTER_USER_AUTONOMY_PREFERENCE, "r_user_autonomy_preference", LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.5f, 0 },
        { LLAMA_SELF_REGISTER_USER_CLARIFICATION_PREFERENCE, "r_user_clarification_preference", LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.5f, 0 },
        { LLAMA_SELF_REGISTER_USER_DISAGREEMENT_SENSITIVITY, "r_user_disagreement_sensitivity", LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.5f, 0 },
        { LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY, "r_evolution_uncertainty",  LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR, 0.0f, 1.0f, 0.0f, 0 },
        { LLAMA_SELF_REGISTER_CHANNEL_STATE,         "r_channel_state",          LLAMA_SELF_REGISTER_FAMILY_CATEGORICAL,    0.0f, 2.0f, 0.0f, LLAMA_SELF_STATE_CHANNEL_WAITING },
    }};
}

const llama_self_register_definition * llama_self_state::get_definition(int32_t register_id) {
    static const auto definitions = build_definitions();
    if (register_id < 0 || register_id >= LLAMA_SELF_REGISTER_COUNT) {
        return nullptr;
    }

    return &definitions[(size_t) register_id];
}

const char * llama_self_state::register_name(int32_t register_id) {
    const auto * def = get_definition(register_id);
    return def ? def->name : nullptr;
}

llama_self_state::llama_self_state() : definitions(build_definitions()) {
    params = llama_self_state_default_params();
    updater_program = llama_self_state_default_updater_program();

    for (size_t i = 0; i < definitions.size(); ++i) {
        const auto & def = definitions[i];
        auto & value = registers[i];
        value.scalar_value = def.default_scalar_value;
        value.categorical_value = def.default_categorical_value;
        value.confidence = 1.0f;
        value.updater_version = LLAMA_SELF_UPDATER_VERSION;
    }

    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_INIT);
    initialize_model_state();
    (void) refresh_time();
}

bool llama_self_state::configure(const llama_self_state_params & next_params) {
    params = next_params;
    if (params.tool_salience_half_life_ms <= 0.0f) {
        params.tool_salience_half_life_ms = LLAMA_SELF_DEFAULT_TOOL_SALIENCE_HALF_LIFE_MS;
    }

    params.prewrite_gain = clamp_unit(params.prewrite_gain);
    params.postwrite_gain = clamp_unit(params.postwrite_gain);
    params.belief_slot_count = std::max(0, std::min<int32_t>(params.belief_slot_count, LLAMA_SELF_BELIEF_MAX_SLOTS));
    params.belief_residual_decay = clamp_unit(params.belief_residual_decay);
    params.belief_pressure_clip = clamp_range(params.belief_pressure_clip, 0.0f, 1.0f);
    params.belief_confidence_floor = clamp_unit(params.belief_confidence_floor);
    params.belief_promotion_threshold = clamp_unit(params.belief_promotion_threshold);
    params.belief_max_update_step = clamp_unit(params.belief_max_update_step);
    params.belief_missing_observation_weight = clamp_unit(params.belief_missing_observation_weight);
    params.belief_unmodeled_care_weight = clamp_unit(params.belief_unmodeled_care_weight);
    params.belief_forecast_error_weight = clamp_unit(params.belief_forecast_error_weight);
    params.belief_counterfactual_miss_weight = clamp_unit(params.belief_counterfactual_miss_weight);
    params.belief_memory_residue_weight = clamp_unit(params.belief_memory_residue_weight);
    refresh_belief_summary();
    refresh_belief_promotion_candidates();
    return true;
}

bool llama_self_state::is_valid_time_point(const llama_self_state_time_point & time_point) const {
    if (time_point.wall_clock_ms < 0 || time_point.monotonic_ms < 0) {
        return false;
    }

    if (time_point.timezone_offset_minutes < -24 * 60 || time_point.timezone_offset_minutes > 24 * 60) {
        return false;
    }

    if (has_explicit_time && datetime.monotonic_ms > 0 && time_point.monotonic_ms < datetime.monotonic_ms) {
        return false;
    }

    return true;
}

bool llama_self_state::ensure_time_initialized() {
    return datetime.wall_clock_ms > 0 ? true : refresh_time();
}

bool llama_self_state::refresh_time() {
    const int64_t wall_clock_ms = current_wall_clock_ms();
    llama_self_state_time_point time_point = {
        /*.wall_clock_ms =*/ wall_clock_ms,
        /*.monotonic_ms =*/ current_monotonic_ms(),
        /*.timezone_offset_minutes =*/ compute_timezone_offset_minutes(wall_clock_ms),
    };

    return apply_time_point(time_point, LLAMA_SELF_SOURCE_TIME);
}

bool llama_self_state::set_time(llama_self_state_time_point time_point) {
    return apply_time_point(time_point, LLAMA_SELF_SOURCE_TIME | LLAMA_SELF_SOURCE_EXTERNAL_TIME);
}

bool llama_self_state::apply_time_point(const llama_self_state_time_point & time_point, uint32_t source_mask) {
    if (!is_valid_time_point(time_point)) {
        return false;
    }

    if ((source_mask & LLAMA_SELF_SOURCE_EXTERNAL_TIME) != 0 && !has_explicit_time) {
        has_explicit_time = true;
        session_start_wall_ms = time_point.wall_clock_ms;
        session_start_monotonic_ms = time_point.monotonic_ms;
        last_user_monotonic_ms = -1;
        last_tool_monotonic_ms = -1;
        last_emit_monotonic_ms = -1;
    }

    datetime.wall_clock_ms = time_point.wall_clock_ms;
    datetime.monotonic_ms = time_point.monotonic_ms;
    datetime.timezone_offset_minutes = time_point.timezone_offset_minutes;

    if (session_start_wall_ms < 0) {
        session_start_wall_ms = time_point.wall_clock_ms;
    }
    if (session_start_monotonic_ms < 0) {
        session_start_monotonic_ms = time_point.monotonic_ms;
    }
    if (last_validated_progress_monotonic_ms < 0) {
        last_validated_progress_monotonic_ms = time_point.monotonic_ms;
    }

    recompute_time_surface(source_mask);
    return true;
}

void llama_self_state::recompute_time_surface(uint32_t source_mask) {
    const int64_t local_wall_ms = datetime.wall_clock_ms + (int64_t) datetime.timezone_offset_minutes * 60000;
    const std::time_t local_time = (std::time_t) (local_wall_ms / 1000);

    std::tm local_tm = {};
    if (!gmtime_safe(local_time, &local_tm)) {
        return;
    }

    datetime.local_year   = local_tm.tm_year + 1900;
    datetime.local_month  = local_tm.tm_mon + 1;
    datetime.local_day    = local_tm.tm_mday;
    datetime.local_hour   = local_tm.tm_hour;
    datetime.local_minute = local_tm.tm_min;
    datetime.local_second = local_tm.tm_sec;
    datetime.day_of_week  = local_tm.tm_wday;
    datetime.day_of_year  = local_tm.tm_yday + 1;

    const float hour_of_day = (float) datetime.local_hour +
            (float) datetime.local_minute / 60.0f +
            (float) datetime.local_second / 3600.0f;
    const float weekday_phase = (float) datetime.day_of_week / 7.0f;
    const float year_day_phase = (float) (datetime.day_of_year - 1) / 365.0f;

    datetime.hour_sin     = std::sin(LLAMA_SELF_TWO_PI * hour_of_day / 24.0f);
    datetime.hour_cos     = std::cos(LLAMA_SELF_TWO_PI * hour_of_day / 24.0f);
    datetime.weekday_sin  = std::sin(LLAMA_SELF_TWO_PI * weekday_phase);
    datetime.weekday_cos  = std::cos(LLAMA_SELF_TWO_PI * weekday_phase);
    datetime.year_day_sin = std::sin(LLAMA_SELF_TWO_PI * year_day_phase);
    datetime.year_day_cos = std::cos(LLAMA_SELF_TWO_PI * year_day_phase);

    datetime.delta_since_last_user_ms = elapsed_or_unset(datetime.monotonic_ms, last_user_monotonic_ms);
    datetime.delta_since_last_tool_event_ms = elapsed_or_unset(datetime.monotonic_ms, last_tool_monotonic_ms);
    datetime.delta_since_last_emit_ms = elapsed_or_unset(datetime.monotonic_ms, last_emit_monotonic_ms);
    datetime.session_age_ms = elapsed_or_unset(datetime.monotonic_ms, session_start_monotonic_ms);

    const float minute_of_day = (float) datetime.local_hour * 60.0f + (float) datetime.local_minute + (float) datetime.local_second / 60.0f;
    update_scalar_register(LLAMA_SELF_REGISTER_TIME_PHASE, clamp_unit(minute_of_day / (24.0f * 60.0f)), source_mask);

    const float tool_salience = decay_to_unit(datetime.delta_since_last_tool_event_ms, params.tool_salience_half_life_ms);
    update_scalar_register(LLAMA_SELF_REGISTER_TOOL_SALIENCE, tool_salience, source_mask | (tool_salience > 0.0f ? LLAMA_SELF_SOURCE_TOOL_EVENT : 0));
    update_evolution_uncertainty(source_mask, 0.0f, 0.0f);
}

float llama_self_state::current_scalar_register(int32_t register_id) const {
    if (!get_definition(register_id)) {
        return 0.0f;
    }

    return registers[(size_t) register_id].scalar_value;
}

void llama_self_state::update_scalar_register(int32_t register_id, float value, uint32_t source_mask) {
    const auto * def = get_definition(register_id);
    if (!def) {
        return;
    }

    auto & reg = registers[(size_t) register_id];
    reg.scalar_value = clamp_range(value, def->value_min, def->value_max);
    reg.last_update_wall_ms = datetime.wall_clock_ms;
    reg.last_update_monotonic_ms = datetime.monotonic_ms;
    reg.source_mask = source_mask;
    reg.confidence = 1.0f;
    reg.updater_version = updater_program.version ? updater_program.version : LLAMA_SELF_UPDATER_VERSION;
    reg.dirty = true;
}

void llama_self_state::blend_scalar_register(int32_t register_id, float target, float gain, uint32_t source_mask) {
    const float current = current_scalar_register(register_id);
    const float blended = clamp_unit(current + gain * (clamp_unit(target) - current));
    update_scalar_register(register_id, blended, source_mask);
}

bool llama_self_state::validate_updater_program(const llama_self_updater_program & program) const {
    if (program.version == 0 || program.rule_count > LLAMA_SELF_MAX_UPDATER_RULES) {
        return false;
    }

    const float * scalar_fields[] = {
        &program.memory_novelty_weight,
        &program.memory_working_similarity_weight,
        &program.memory_handle_similarity_weight,
        &program.memory_uncertainty_weight,
        &program.memory_contradiction_weight,
        &program.memory_handle_variance_weight,
        &program.broadcast_social_weight,
        &program.broadcast_contradiction_weight,
        &program.broadcast_uncertainty_weight,
        &program.broadcast_tool_pending_weight,
        &program.broadcast_tool_unready_weight,
        &program.broadcast_failure_weight,
        &program.broadcast_question_weight,
        &program.broadcast_goal_weight,
        &program.repair_emit_threshold,
        &program.repair_dissatisfaction_floor,
        &program.repair_recent_user_valence_floor,
        &program.repair_inhibition_max,
        &program.repair_admission_floor,
        &program.repair_admission_weight,
    };
    for (const float * field : scalar_fields) {
        if (!std::isfinite(*field)) {
            return false;
        }
    }

    const uint32_t valid_phase_mask = LLAMA_SELF_UPDATER_PHASE_PREWRITE | LLAMA_SELF_UPDATER_PHASE_POSTWRITE;

    for (uint32_t i = 0; i < program.rule_count; ++i) {
        const auto & rule = program.rules[i];
        const auto * def = get_definition(rule.register_id);
        if (!def || def->family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR) {
            return false;
        }
        if ((rule.phase_mask & valid_phase_mask) == 0 || (rule.phase_mask & ~valid_phase_mask) != 0) {
            return false;
        }
        if (!std::isfinite(rule.baseline) || !std::isfinite(rule.rise_gain) ||
            !std::isfinite(rule.fall_gain) || !std::isfinite(rule.baseline_pull)) {
            return false;
        }
        if (rule.rise_gain < 0.0f || rule.rise_gain > 1.0f ||
            rule.fall_gain < 0.0f || rule.fall_gain > 1.0f ||
            rule.baseline_pull < 0.0f || rule.baseline_pull > 1.0f) {
            return false;
        }

        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_TERMS; ++term) {
            const int32_t feature_id = rule.feature_ids[term];
            if (feature_id == LLAMA_SELF_UPDATER_FEATURE_NONE) {
                continue;
            }
            if (feature_id < LLAMA_SELF_UPDATER_FEATURE_NOVELTY ||
                feature_id > LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_COMPLETED ||
                !std::isfinite(rule.feature_weights[term])) {
                return false;
            }
        }

        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS; ++term) {
            const int32_t source_register_id = rule.source_register_ids[term];
            if (source_register_id < 0) {
                continue;
            }
            const auto * source_def = get_definition(source_register_id);
            if (!source_def || source_def->family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR ||
                !std::isfinite(rule.source_register_weights[term])) {
                return false;
            }
        }
    }

    return true;
}

float llama_self_state::updater_feature_value(
        int32_t feature_id,
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) const {
    switch (feature_id) {
        case LLAMA_SELF_UPDATER_FEATURE_NOVELTY: return features.novelty;
        case LLAMA_SELF_UPDATER_FEATURE_TOPIC_SHIFT: return features.topic_shift;
        case LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY: return features.goal_top_similarity;
        case LLAMA_SELF_UPDATER_FEATURE_COMMITMENT_SIMILARITY: return features.commitment_top_similarity;
        case LLAMA_SELF_UPDATER_FEATURE_IDENTITY_SIMILARITY: return features.identity_similarity;
        case LLAMA_SELF_UPDATER_FEATURE_SELF_REFERENCE: return features.self_reference_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_NEGATION_RATIO: return features.negation_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY_LEXICAL: return features.uncertainty_lexical_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_ERROR_RATIO: return features.error_ratio;
        case LLAMA_SELF_UPDATER_FEATURE_RECENCY_USER: return features.recency_user;
        case LLAMA_SELF_UPDATER_FEATURE_RECENCY_TOOL: return features.recency_tool;
        case LLAMA_SELF_UPDATER_FEATURE_RECENCY_EMIT: return features.recency_emit;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_FAMILIARITY: return features.social_familiarity;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_TRUST: return features.social_trust;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_RECIPROCITY: return features.social_reciprocity;
        case LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND: return features.social_bond_strength;
        case LLAMA_SELF_UPDATER_FEATURE_TOOL_READINESS: return features.tool_readiness_score;
        case LLAMA_SELF_UPDATER_FEATURE_TOOL_PENDING_PRESSURE: return features.tool_pending_pressure;
        case LLAMA_SELF_UPDATER_FEATURE_DECODER_ENTROPY: return features.decoder_entropy;
        case LLAMA_SELF_UPDATER_FEATURE_DECODER_TOP_MARGIN: return features.decoder_top_margin;
        case LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION: return features.contradiction_score;
        case LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY: return features.uncertainty_score;
        case LLAMA_SELF_UPDATER_FEATURE_MEMORY_WRITE_PRESSURE: return features.memory_write_pressure;
        case LLAMA_SELF_UPDATER_FEATURE_BROADCAST_PRESSURE_HINT: return features.broadcast_pressure_hint;
        case LLAMA_SELF_UPDATER_FEATURE_BROADCAST_INHIBITION_HINT: return features.broadcast_inhibition_hint;
        case LLAMA_SELF_UPDATER_FEATURE_FOLLOWUP_HINT: return features.followup_hint;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_USER: return event.role == LLAMA_SELF_STATE_EVENT_USER ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_TOOL: return event.role == LLAMA_SELF_STATE_EVENT_TOOL ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_SYSTEM: return event.role == LLAMA_SELF_STATE_EVENT_SYSTEM ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_CHANNEL_PRIMARY: return event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_CHANNEL_COUNTERFACTUAL: return event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_ADMITTED: return (event.flags & LLAMA_SELF_STATE_EVENT_ADMITTED) ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_FAILED: return (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f;
        case LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_COMPLETED: return (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED) ? 1.0f : 0.0f;
        default: return 0.0f;
    }
}

bool llama_self_state::apply_register_update_rules(
        uint32_t phase_mask,
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features,
        uint32_t source_mask) {
    std::array<float, LLAMA_SELF_REGISTER_COUNT> snapshot = {};
    for (size_t i = 0; i < snapshot.size(); ++i) {
        snapshot[i] = registers[i].scalar_value;
    }

    for (uint32_t i = 0; i < updater_program.rule_count; ++i) {
        const auto & rule = updater_program.rules[i];
        if ((rule.phase_mask & phase_mask) == 0) {
            continue;
        }

        const auto * def = get_definition(rule.register_id);
        if (!def || def->family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR) {
            continue;
        }

        float target = rule.baseline;
        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_TERMS; ++term) {
            const int32_t feature_id = rule.feature_ids[term];
            if (feature_id == LLAMA_SELF_UPDATER_FEATURE_NONE) {
                continue;
            }
            target += rule.feature_weights[term] * updater_feature_value(feature_id, event, features);
        }
        for (size_t term = 0; term < LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS; ++term) {
            const int32_t source_register_id = rule.source_register_ids[term];
            if (source_register_id < 0 || (size_t) source_register_id >= snapshot.size()) {
                continue;
            }
            target += rule.source_register_weights[term] * snapshot[(size_t) source_register_id];
        }

        const float bounded_target = clamp_range(target, def->value_min, def->value_max);
        const float current = snapshot[(size_t) rule.register_id];
        const float delta = bounded_target - current;
        const float gain = delta >= 0.0f ? rule.rise_gain : rule.fall_gain;
        const float next = clamp_range(
                current + gain * delta + rule.baseline_pull * (rule.baseline - current),
                def->value_min,
                def->value_max);
        update_scalar_register(rule.register_id, next, source_mask);
    }

    return true;
}

void llama_self_state::update_categorical_register(int32_t register_id, int32_t value, uint32_t source_mask) {
    if (!get_definition(register_id)) {
        return;
    }

    auto & reg = registers[(size_t) register_id];
    reg.categorical_value = value;
    reg.last_update_wall_ms = datetime.wall_clock_ms;
    reg.last_update_monotonic_ms = datetime.monotonic_ms;
    reg.source_mask = source_mask;
    reg.confidence = 1.0f;
    reg.updater_version = updater_program.version ? updater_program.version : LLAMA_SELF_UPDATER_VERSION;
    reg.dirty = true;
}

bool llama_self_state::get_datetime(llama_self_state_datetime * out_info) const {
    if (!out_info) {
        return false;
    }

    *out_info = datetime;
    return true;
}

int32_t llama_self_state::register_count() const {
    return LLAMA_SELF_REGISTER_COUNT;
}

bool llama_self_state::get_register(int32_t register_id, llama_self_register_info * out_info) const {
    if (!out_info) {
        return false;
    }

    const auto * def = get_definition(register_id);
    if (!def) {
        return false;
    }

    const auto & reg = registers[(size_t) register_id];
    *out_info = {
        /*.register_id =*/ register_id,
        /*.family =*/ def->family,
        /*.scalar_value =*/ reg.scalar_value,
        /*.categorical_value =*/ reg.categorical_value,
        /*.value_min =*/ def->value_min,
        /*.value_max =*/ def->value_max,
        /*.confidence =*/ reg.confidence,
        /*.last_update_wall_ms =*/ reg.last_update_wall_ms,
        /*.last_update_monotonic_ms =*/ reg.last_update_monotonic_ms,
        /*.source_mask =*/ reg.source_mask,
        /*.updater_version =*/ reg.updater_version,
        /*.dirty =*/ reg.dirty,
    };
    return true;
}

bool llama_self_state::set_channel_state(int32_t next_channel_state) {
    if (next_channel_state != LLAMA_SELF_STATE_CHANNEL_WAITING &&
        next_channel_state != LLAMA_SELF_STATE_CHANNEL_ACTIVE &&
        next_channel_state != LLAMA_SELF_STATE_CHANNEL_DO_NOT_INTERRUPT) {
        return false;
    }

    if (!ensure_time_initialized()) {
        return false;
    }

    channel_state = next_channel_state;
    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_CHANNEL);
    return true;
}

bool llama_self_state::note_user_event() {
    if (!ensure_time_initialized()) {
        return false;
    }

    last_user_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_user_ms = 0;
    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_USER_EVENT | LLAMA_SELF_SOURCE_CHANNEL);
    return true;
}

bool llama_self_state::note_tool_event() {
    if (!ensure_time_initialized()) {
        return false;
    }

    last_tool_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_tool_event_ms = 0;
    update_scalar_register(LLAMA_SELF_REGISTER_TOOL_SALIENCE, 1.0f, LLAMA_SELF_SOURCE_TOOL_EVENT);
    return true;
}

bool llama_self_state::note_emit_event() {
    if (!ensure_time_initialized()) {
        return false;
    }

    last_emit_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_emit_ms = 0;
    return true;
}

bool llama_self_state::set_identity(const llama_token * tokens, size_t n_tokens) {
    if (!tokens || n_tokens == 0) {
        return false;
    }

    identity_sketch = build_token_sketch(tokens, n_tokens);
    has_identity_sketch = true;
    return true;
}

void llama_self_state::upsert_surface(
        std::vector<llama_self_sketch_surface> & surfaces,
        int32_t id,
        const std::array<float, 32> & sketch,
        float priority,
        bool unresolved) {
    auto it = std::find_if(surfaces.begin(), surfaces.end(), [id](const llama_self_sketch_surface & surface) {
        return surface.id == id;
    });

    if (it == surfaces.end()) {
        surfaces.push_back({});
        it = surfaces.end() - 1;
    }

    it->id = id;
    it->priority = clamp_unit(priority);
    it->unresolved = unresolved;
    it->last_update_monotonic_ms = datetime.monotonic_ms;
    it->sketch = sketch;
}

bool llama_self_state::upsert_goal(int32_t goal_id, const llama_token * tokens, size_t n_tokens, float priority) {
    if (goal_id < 0 || !tokens || n_tokens == 0) {
        return false;
    }

    upsert_surface(goals, goal_id, build_token_sketch(tokens, n_tokens), priority, true);
    if (goals.size() > LLAMA_SELF_MAX_GOALS) {
        std::sort(goals.begin(), goals.end(), [](const llama_self_sketch_surface & lhs, const llama_self_sketch_surface & rhs) {
            return lhs.priority > rhs.priority;
        });
        goals.resize(LLAMA_SELF_MAX_GOALS);
    }
    return true;
}

bool llama_self_state::upsert_commitment(
        int32_t commitment_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority,
        bool unresolved) {
    if (commitment_id < 0 || !tokens || n_tokens == 0) {
        return false;
    }

    upsert_surface(commitments, commitment_id, build_token_sketch(tokens, n_tokens), priority, unresolved);
    if (commitments.size() > LLAMA_SELF_MAX_COMMITMENTS) {
        std::sort(commitments.begin(), commitments.end(), [](const llama_self_sketch_surface & lhs, const llama_self_sketch_surface & rhs) {
            return lhs.priority > rhs.priority;
        });
        commitments.resize(LLAMA_SELF_MAX_COMMITMENTS);
    }
    return true;
}

int32_t llama_self_state::goal_count() const {
    return (int32_t) goals.size();
}

int32_t llama_self_state::commitment_count() const {
    return (int32_t) commitments.size();
}

int32_t llama_self_state::working_memory_count() const {
    return (int32_t) working_memory.size();
}

bool llama_self_state::upsert_memory_handle(
        int32_t handle_id,
        int32_t kind,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    if (handle_id < 0 || !tokens || n_tokens == 0 || !ensure_time_initialized()) {
        return false;
    }

    if (kind != LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER &&
        kind != LLAMA_SELF_MEMORY_HANDLE_FROZEN_BUCKET &&
        kind != LLAMA_SELF_MEMORY_HANDLE_ACTIVE_MEMORY &&
        kind != LLAMA_SELF_MEMORY_HANDLE_EXTERNAL) {
        return false;
    }

    const auto sketch = build_token_sketch(tokens, n_tokens);
    return upsert_memory_handle_sketch(handle_id, kind, sketch, priority, 1);
}

bool llama_self_state::upsert_memory_handle_sketch(
        int32_t handle_id,
        int32_t kind,
        const std::array<float, 32> & sketch,
        float priority,
        uint32_t member_count) {
    if (handle_id < 0 || !ensure_time_initialized()) {
        return false;
    }

    if (kind != LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER &&
        kind != LLAMA_SELF_MEMORY_HANDLE_FROZEN_BUCKET &&
        kind != LLAMA_SELF_MEMORY_HANDLE_ACTIVE_MEMORY &&
        kind != LLAMA_SELF_MEMORY_HANDLE_EXTERNAL) {
        return false;
    }

    auto it = std::find_if(memory_handles.begin(), memory_handles.end(), [handle_id](const llama_self_memory_handle & handle) {
        return handle.handle_id == handle_id;
    });

    if (it == memory_handles.end()) {
        memory_handles.push_back({});
        it = memory_handles.end() - 1;
    }

    it->handle_id = handle_id;
    it->kind = kind;
    it->priority = clamp_unit(priority);
    it->last_update_monotonic_ms = datetime.monotonic_ms;
    it->member_count = std::max<uint32_t>(1, member_count);
    it->centroid = sketch;

    if (memory_handles.size() > LLAMA_SELF_MAX_MEMORY_HANDLES) {
        std::sort(memory_handles.begin(), memory_handles.end(), [](const llama_self_memory_handle & lhs, const llama_self_memory_handle & rhs) {
            return lhs.priority > rhs.priority;
        });
        memory_handles.resize(LLAMA_SELF_MAX_MEMORY_HANDLES);
    }
    return true;
}

int32_t llama_self_state::memory_handle_count() const {
    return (int32_t) memory_handles.size();
}

int32_t llama_self_state::reactivation_count() const {
    return (int32_t) reactivation_priorities.size();
}

bool llama_self_state::get_reactivation(int32_t index, llama_self_reactivation_info * out_info) const {
    if (!out_info || index < 0 || (size_t) index >= reactivation_priorities.size()) {
        return false;
    }

    *out_info = reactivation_priorities[(size_t) index];
    return true;
}

bool llama_self_state::upsert_tool_job(int32_t job_id, int32_t status, float importance) {
    if (job_id < 0 || !ensure_time_initialized()) {
        return false;
    }

    if (status < LLAMA_SELF_TOOL_JOB_IDLE || status > LLAMA_SELF_TOOL_JOB_FAILED) {
        return false;
    }

    auto it = std::find_if(tool_jobs.begin(), tool_jobs.end(), [job_id](const llama_self_tool_job & job) {
        return job.job_id == job_id;
    });

    if (it == tool_jobs.end()) {
        tool_jobs.push_back({});
        it = tool_jobs.end() - 1;
        it->job_id = job_id;
    }

    it->status = status;
    it->importance = clamp_unit(importance);
    it->last_update_monotonic_ms = datetime.monotonic_ms;
    last_tool_monotonic_ms = datetime.monotonic_ms;
    datetime.delta_since_last_tool_event_ms = 0;
    refresh_tool_surface(LLAMA_SELF_SOURCE_TOOL_EVENT);
    return true;
}

bool llama_self_state::get_tool_state(llama_self_tool_state_info * out_info) const {
    if (!out_info) {
        return false;
    }

    *out_info = build_tool_state_info(tool_jobs);
    return true;
}

bool llama_self_state::get_social_state(llama_self_social_state_info * out_info) const {
    if (!out_info) {
        return false;
    }

    const float bond_strength = clamp_unit(
            0.40f * social_familiarity +
            0.35f * social_trust +
            0.25f * social_reciprocity);

    *out_info = {
        /*.familiarity =*/ social_familiarity,
        /*.trust =*/ social_trust,
        /*.reciprocity =*/ social_reciprocity,
        /*.bond_strength =*/ bond_strength,
        /*.recent_user_valence =*/ social_recent_user_valence,
        /*.dissatisfaction =*/ social_dissatisfaction,
        /*.user_turn_count =*/ social_user_turn_count,
        /*.system_turn_count =*/ social_system_turn_count,
        /*.last_update_monotonic_ms =*/ social_last_update_monotonic_ms,
    };
    return true;
}

bool llama_self_state::get_model_state(llama_self_model_state_info * out_info) const {
    if (!out_info) {
        return false;
    }

    out_info->horizon_count = LLAMA_SELF_HORIZON_COUNT;
    out_info->belief_slot_count = std::min<int32_t>(params.belief_slot_count, LLAMA_SELF_BELIEF_MAX_SLOTS);
    out_info->promotion_candidate_count = promotion_candidate_count;
    for (int32_t i = 0; i < LLAMA_SELF_HORIZON_COUNT; ++i) {
        out_info->horizons[i] = model_horizons[(size_t) i];
    }
    out_info->forecast = model_forecast;
    out_info->prediction_error = prediction_error;
    out_info->belief_summary = belief_summary;
    for (int32_t i = 0; i < LLAMA_SELF_BELIEF_MAX_SLOTS; ++i) {
        out_info->belief_slots[i] = belief_slots[(size_t) i];
    }
    for (int32_t i = 0; i < LLAMA_SELF_BELIEF_MAX_PROMOTION_CANDIDATES; ++i) {
        out_info->promotion_candidates[i] = promotion_candidates[(size_t) i];
    }
    out_info->extension_summary = extension_summary;
    out_info->last_extension_trace = extension_trace;
    return true;
}

int32_t llama_self_state::model_extension_count() const {
    return (int32_t) model_extensions.size();
}

bool llama_self_state::get_model_extension(int32_t index, llama_self_model_extension_info * out_info) const {
    if (!out_info || index < 0 || (size_t) index >= model_extensions.size()) {
        return false;
    }

    const auto & extension = model_extensions[(size_t) index];
    *out_info = {};
    out_info->slot = index;
    out_info->source = extension.source;
    out_info->source_tool_kind = extension.source_tool_kind;
    out_info->kind = extension.kind;
    out_info->domain = extension.domain;
    out_info->flags = extension.flags;
    out_info->last_update_monotonic_ms = extension.last_update_monotonic_ms;
    out_info->activation_count = extension.activation_count;
    out_info->value = extension.value;
    out_info->desired_value = extension.desired_value;
    out_info->confidence = extension.confidence;
    out_info->salience = extension.salience;
    out_info->gain_weight = extension.gain_weight;
    out_info->allostatic_weight = extension.allostatic_weight;
    copy_bounded_cstr(out_info->key, extension.key);
    copy_bounded_cstr(out_info->label, extension.label);
    copy_bounded_cstr(out_info->content, extension.content);
    return true;
}

bool llama_self_state::validate_model_extension_update(llama_self_model_extension_update * update) const {
    if (!update) {
        return false;
    }

    if (update->source < LLAMA_SELF_MODEL_EXTENSION_SOURCE_COUNTERFACTUAL ||
        update->source > LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_EXTERNAL ||
        update->kind < LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT ||
        update->kind > LLAMA_SELF_MODEL_EXTENSION_SCALAR_PARAM ||
        update->domain < LLAMA_SELF_MODEL_EXTENSION_DOMAIN_GOAL_PROGRESS ||
        update->domain > LLAMA_SELF_MODEL_EXTENSION_DOMAIN_SELF_IMPROVEMENT ||
        update->key[0] == '\0') {
        return false;
    }

    const float * values[] = {
        &update->value,
        &update->desired_value,
        &update->confidence,
        &update->salience,
        &update->gain_weight,
        &update->allostatic_weight,
    };
    for (const float * value : values) {
        if (!std::isfinite(*value)) {
            return false;
        }
    }

    update->flags |= LLAMA_SELF_MODEL_EXTENSION_FLAG_ACTIVE;
    if ((update->flags & (LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN | LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS)) == 0) {
        update->flags |= LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN;
    }

    update->value = clamp_unit(update->value);
    update->desired_value = clamp_unit(update->desired_value);
    update->confidence = clamp_unit(update->confidence);
    update->salience = clamp_unit(update->salience);
    update->gain_weight = clamp_unit(update->gain_weight <= 0.0f ? 1.0f : update->gain_weight);
    update->allostatic_weight = clamp_unit(update->allostatic_weight);

    if (update->kind == LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT ||
        update->source == LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_HARD_MEMORY) {
        update->flags &= ~LLAMA_SELF_MODEL_EXTENSION_FLAG_HAS_DESIRED_STATE;
        update->flags &= ~LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS;
        update->desired_value = 0.0f;
        update->allostatic_weight = 0.0f;
    }

    if ((update->flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_HAS_DESIRED_STATE) == 0) {
        update->flags &= ~LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS;
        update->desired_value = 0.0f;
        update->allostatic_weight = 0.0f;
    } else if (update->allostatic_weight <= 0.0f) {
        update->allostatic_weight = 1.0f;
    }

    return true;
}

void llama_self_state::refresh_model_extension_summary() {
    const std::array<float, 32> neutral_context = {};
    extension_domain_signal signal = {};
    accumulate_model_extensions(model_extensions, neutral_context, false, &signal, &extension_summary);
}

bool llama_self_state::upsert_model_extension(const llama_self_model_extension_update & input) {
    llama_self_model_extension_update update = input;
    if (!validate_model_extension_update(&update)) {
        return false;
    }

    auto it = std::find_if(model_extensions.begin(), model_extensions.end(), [&update](const llama_self_model_extension_entry & extension) {
        return std::strncmp(extension.key, update.key, sizeof(extension.key)) == 0;
    });

    if (it == model_extensions.end()) {
        if (model_extensions.size() >= LLAMA_SELF_MODEL_EXTENSION_MAX_ITEMS) {
            auto evict = std::min_element(model_extensions.begin(), model_extensions.end(), [](const llama_self_model_extension_entry & lhs, const llama_self_model_extension_entry & rhs) {
                const float lhs_score = lhs.salience * lhs.confidence;
                const float rhs_score = rhs.salience * rhs.confidence;
                if (lhs_score == rhs_score) {
                    return lhs.last_update_monotonic_ms < rhs.last_update_monotonic_ms;
                }
                return lhs_score < rhs_score;
            });
            if (evict == model_extensions.end()) {
                return false;
            }
            *evict = {};
            it = evict;
        } else {
            model_extensions.push_back({});
            it = model_extensions.end() - 1;
        }
    }

    it->source = update.source;
    it->source_tool_kind = update.source_tool_kind;
    it->kind = update.kind;
    it->domain = update.domain;
    it->flags = update.flags;
    it->value = update.value;
    it->desired_value = update.desired_value;
    it->confidence = update.confidence;
    it->salience = update.salience;
    it->gain_weight = update.gain_weight;
    it->allostatic_weight = update.allostatic_weight;
    it->last_update_monotonic_ms = datetime.monotonic_ms >= 0 ? datetime.monotonic_ms : current_monotonic_ms();
    copy_bounded_cstr(it->key, update.key);
    copy_bounded_cstr(it->label, update.label[0] != '\0' ? update.label : update.key);
    copy_bounded_cstr(it->content, update.content);
    const std::string sketch_source = update.content[0] != '\0' ?
            read_bounded_cstr(update.content) :
            (read_bounded_cstr(update.label) + " " + read_bounded_cstr(update.key));
    it->sketch = build_text_sketch(sketch_source);

    refresh_model_extension_summary();
    return true;
}

bool llama_self_state::remove_model_extension(const char * key) {
    if (!key || key[0] == '\0') {
        return false;
    }

    const std::string needle(key);
    auto it = std::find_if(model_extensions.begin(), model_extensions.end(), [&needle](const llama_self_model_extension_entry & extension) {
        return needle == read_bounded_cstr(extension.key);
    });
    if (it == model_extensions.end()) {
        return false;
    }

    model_extensions.erase(it);
    refresh_model_extension_summary();
    return true;
}

bool llama_self_state::promote_hard_memory_query(
        const llama_hard_memory_query_request & request,
        const llama_hard_memory_result & result) {
    extension_trace = {};
    if (!result.ok || result.result_count <= 0) {
        return true;
    }

    const std::array<float, 32> query_sketch = build_text_sketch(read_bounded_cstr(request.query));
    const float goal_signal = current_scalar_register(LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE);
    const float user_signal = current_scalar_register(LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK);
    const float epistemic_signal = clamp_unit(
            0.55f * current_scalar_register(LLAMA_SELF_REGISTER_UNCERTAINTY) +
            0.45f * (1.0f - current_scalar_register(LLAMA_SELF_REGISTER_ANSWERABILITY)));
    const float efficiency_signal = current_scalar_register(LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY);
    const float recovery_signal = current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY);

    int32_t default_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EPISTEMIC;
    float best_domain_score = epistemic_signal;
    if (goal_signal > best_domain_score) {
        best_domain_score = goal_signal;
        default_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_GOAL_PROGRESS;
    }
    if (user_signal > best_domain_score) {
        best_domain_score = user_signal;
        default_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_USER_OUTCOME;
    }
    if (efficiency_signal > best_domain_score) {
        best_domain_score = efficiency_signal;
        default_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EFFICIENCY;
    }
    if (recovery_signal > best_domain_score) {
        default_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_RECOVERY;
    }

    extension_trace.valid = true;
    extension_trace.winner_index = -1;
    float best_improvement = 0.0f;
    llama_self_model_extension_update winner = llama_self_model_extension_default_update();

    const llama_self_model_horizon_info base_horizon =
            model_horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT].last_update_monotonic_ms >= 0 ?
                    model_horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT] :
                    llama_self_model_horizon_info {};
    const float before_score = extension_utility_score(base_horizon);

    const int32_t candidate_limit = std::min<int32_t>(result.result_count, LLAMA_SELF_MODEL_EXTENSION_MAX_CANDIDATES);
    for (int32_t i = 0; i < candidate_limit; ++i) {
        const auto & hit = result.results[i];
        char key[LLAMA_HARD_MEMORY_MAX_ID_CHARS] = {};
        std::snprintf(key, sizeof(key), "hard_memory:%s", hit.id[0] != '\0' ? hit.id : "unknown");

        int32_t candidate_domain = default_domain;
        switch (hit.domain) {
            case LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS:    candidate_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_GOAL_PROGRESS; break;
            case LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME:     candidate_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_USER_OUTCOME; break;
            case LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC:        candidate_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EPISTEMIC; break;
            case LLAMA_HARD_MEMORY_DOMAIN_EFFICIENCY:       candidate_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EFFICIENCY; break;
            case LLAMA_HARD_MEMORY_DOMAIN_RECOVERY:         candidate_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_RECOVERY; break;
            case LLAMA_HARD_MEMORY_DOMAIN_STRATEGY:         candidate_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_STRATEGY; break;
            case LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT: candidate_domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_SELF_IMPROVEMENT; break;
            default: break;
        }

        llama_self_model_extension_update candidate = llama_self_model_extension_default_update();
        candidate.source = LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_HARD_MEMORY;
        candidate.source_tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_QUERY;
        candidate.kind = LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT;
        candidate.domain = candidate_domain;
        candidate.flags = LLAMA_SELF_MODEL_EXTENSION_FLAG_ACTIVE |
                LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN |
                LLAMA_SELF_MODEL_EXTENSION_FLAG_DISCOVERED;
        candidate.value = clamp_unit(std::max(hit.similarity, hit.importance));
        candidate.confidence = clamp_unit(std::max(hit.confidence, hit.similarity));
        candidate.salience = clamp_unit(std::max(hit.importance, hit.similarity));
        candidate.gain_weight = clamp_unit(0.35f + 0.35f * hit.similarity + 0.30f * hit.gain_bias);
        copy_bounded_cstr(candidate.key, key);
        copy_bounded_cstr(candidate.label, hit.title[0] != '\0' ? hit.title : key);
        copy_bounded_cstr(candidate.content, hit.content);

        extension_domain_signal signal = {};
        llama_self_model_extension_summary summary = {};
        std::vector<llama_self_model_extension_entry> probe(1);
        probe[0].source = candidate.source;
        probe[0].source_tool_kind = candidate.source_tool_kind;
        probe[0].kind = candidate.kind;
        probe[0].domain = candidate.domain;
        probe[0].flags = candidate.flags;
        probe[0].value = candidate.value;
        probe[0].desired_value = candidate.desired_value;
        probe[0].confidence = candidate.confidence;
        probe[0].salience = candidate.salience;
        probe[0].gain_weight = candidate.gain_weight;
        probe[0].allostatic_weight = candidate.allostatic_weight;
        copy_bounded_cstr(probe[0].key, candidate.key);
        copy_bounded_cstr(probe[0].label, candidate.label);
        copy_bounded_cstr(probe[0].content, candidate.content);
        probe[0].sketch = build_text_sketch(read_bounded_cstr(candidate.content));
        accumulate_model_extensions(probe, query_sketch, false, &signal, &summary);

        llama_self_model_horizon_info shadow = base_horizon;
        apply_extension_signal_to_horizon(shadow, signal);
        float kind_bonus = 0.0f;
        switch (hit.kind) {
            case LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY:          kind_bonus = 0.05f; break;
            case LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME:             kind_bonus = 0.06f; break;
            case LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_OBSERVATION:    kind_bonus = 0.04f; break;
            case LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL:          kind_bonus = 0.05f; break;
            case LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT: kind_bonus = 0.06f; break;
            default: break;
        }
        const float improvement = clamp_signed_unit(
                extension_utility_score(shadow) - before_score +
                0.08f * hit.importance +
                0.06f * hit.gain_bias +
                kind_bonus);

        auto & out_candidate = extension_trace.candidates[extension_trace.candidate_count++];
        out_candidate.promoted = false;
        out_candidate.source_tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_QUERY;
        out_candidate.domain = candidate.domain;
        out_candidate.expected_gain_improvement = improvement;
        out_candidate.expected_allostatic_delta = 0.0f;
        out_candidate.confidence = candidate.confidence;
        copy_bounded_cstr(out_candidate.key, candidate.key);
        copy_bounded_cstr(out_candidate.label, candidate.label);

        if (improvement > best_improvement) {
            best_improvement = improvement;
            winner = candidate;
            extension_trace.winner_index = extension_trace.candidate_count - 1;
        }
    }

    if (extension_trace.winner_index >= 0 && best_improvement > 0.01f) {
        if (upsert_model_extension(winner)) {
            extension_trace.promoted_count = 1;
            extension_trace.candidates[extension_trace.winner_index].promoted = true;
        }
    }

    return true;
}

int32_t llama_self_state::trace_count() const {
    return (int32_t) trace_items.size();
}

bool llama_self_state::clear_trace() {
    trace_items.clear();
    return true;
}

bool llama_self_state::set_updater_program(const llama_self_updater_program & program) {
    if (!validate_updater_program(program)) {
        return false;
    }

    updater_program = program;
    return true;
}

bool llama_self_state::get_updater_program(llama_self_updater_program * out_program) const {
    if (!out_program) {
        return false;
    }

    *out_program = updater_program;
    return true;
}

size_t llama_self_state::trace_export_size() const {
    size_t size = sizeof(uint32_t) * 3;

    for (const auto & item : trace_items) {
        size += sizeof(llama_self_state_time_point);
        size += sizeof(int32_t);
        size += sizeof(int32_t);
        size += sizeof(uint32_t);
        size += sizeof(float) * 2;
        size += sizeof(uint32_t);
        size += item.tokens.size() * sizeof(llama_token);
    }

    return size;
}

bool llama_self_state::trace_export(void * dst, size_t size) const {
    if (!dst || size < trace_export_size()) {
        return false;
    }

    std::vector<uint8_t> buffer;
    buffer.reserve(trace_export_size());
    append_bytes(buffer, LLAMA_SELF_TRACE_MAGIC);
    append_bytes(buffer, LLAMA_SELF_TRACE_VERSION);
    append_bytes(buffer, (uint32_t) trace_items.size());

    for (const auto & item : trace_items) {
        append_bytes(buffer, item.time_point);
        append_bytes(buffer, item.event.role);
        append_bytes(buffer, item.event.channel);
        append_bytes(buffer, item.event.flags);
        append_bytes(buffer, item.event.decoder_entropy);
        append_bytes(buffer, item.event.decoder_top_margin);
        append_bytes(buffer, (uint32_t) item.tokens.size());
        for (llama_token token : item.tokens) {
            append_bytes(buffer, token);
        }
    }

    if (buffer.size() > size) {
        return false;
    }

    std::memcpy(dst, buffer.data(), buffer.size());
    return true;
}

bool llama_self_state::trace_import(const void * src, size_t size, bool replace_existing) {
    if (!src) {
        return false;
    }

    const uint8_t * bytes = (const uint8_t *) src;
    size_t cursor = 0;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    if (!read_bytes(bytes, size, &cursor, &magic) ||
        !read_bytes(bytes, size, &cursor, &version) ||
        !read_bytes(bytes, size, &cursor, &count) ||
        magic != LLAMA_SELF_TRACE_MAGIC ||
        (version != 1 && version != LLAMA_SELF_TRACE_VERSION)) {
        return false;
    }

    std::vector<llama_self_trace_item> imported;
    imported.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        llama_self_state_time_point time_point = {};
        int32_t role = 0;
        int32_t channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
        uint32_t flags = 0;
        float decoder_entropy = 0.0f;
        float decoder_top_margin = 0.0f;
        uint32_t n_tokens = 0;

        if (!read_bytes(bytes, size, &cursor, &time_point) ||
            !read_bytes(bytes, size, &cursor, &role) ||
            (version >= 2 && !read_bytes(bytes, size, &cursor, &channel)) ||
            !read_bytes(bytes, size, &cursor, &flags) ||
            !read_bytes(bytes, size, &cursor, &decoder_entropy) ||
            !read_bytes(bytes, size, &cursor, &decoder_top_margin) ||
            !read_bytes(bytes, size, &cursor, &n_tokens)) {
            return false;
        }

        llama_self_trace_item item = {};
        item.time_point = time_point;
        item.tokens.resize(n_tokens);
        for (uint32_t j = 0; j < n_tokens; ++j) {
            if (!read_bytes(bytes, size, &cursor, &item.tokens[(size_t) j])) {
                return false;
            }
        }

        item.event = {
            /*.tokens =*/ item.tokens.data(),
            /*.n_tokens =*/ item.tokens.size(),
            /*.role =*/ role,
            /*.channel =*/ channel,
            /*.flags =*/ flags,
            /*.decoder_entropy =*/ decoder_entropy,
            /*.decoder_top_margin =*/ decoder_top_margin,
        };
        imported.push_back(std::move(item));
    }

    if (cursor != size) {
        return false;
    }

    if (replace_existing) {
        trace_items = std::move(imported);
    } else {
        trace_items.insert(trace_items.end(), imported.begin(), imported.end());
        if (trace_items.size() > LLAMA_SELF_MAX_TRACE_ITEMS) {
            trace_items.erase(trace_items.begin(), trace_items.end() - LLAMA_SELF_MAX_TRACE_ITEMS);
        }
    }

    repair_trace_item_pointers(trace_items);

    return true;
}

void llama_self_state::reset_dynamic_state_preserve_static() {
    registers = {};
    for (size_t i = 0; i < definitions.size(); ++i) {
        const auto & def = definitions[i];
        auto & value = registers[i];
        value.scalar_value = def.default_scalar_value;
        value.categorical_value = def.default_categorical_value;
        value.confidence = 1.0f;
        value.updater_version = LLAMA_SELF_UPDATER_VERSION;
    }

    working_memory.clear();
    tool_jobs.clear();
    reactivation_priorities.clear();
    next_working_memory_event_id = 1;
    has_previous_event_sketch = false;
    previous_event_sketch = {};
    last_user_monotonic_ms = -1;
    last_tool_monotonic_ms = -1;
    last_emit_monotonic_ms = -1;
    social_last_update_monotonic_ms = -1;
    social_user_turn_count = 0;
    social_system_turn_count = 0;
    social_familiarity = 0.0f;
    social_trust = 0.5f;
    social_reciprocity = 0.5f;
    social_recent_user_valence = 0.0f;
    social_dissatisfaction = 0.0f;
    initialize_model_state();
    session_start_wall_ms = -1;
    session_start_monotonic_ms = -1;
    has_explicit_time = false;
    channel_state = LLAMA_SELF_STATE_CHANNEL_WAITING;
    update_categorical_register(LLAMA_SELF_REGISTER_CHANNEL_STATE, channel_state, LLAMA_SELF_SOURCE_INIT);
    recompute_time_surface(LLAMA_SELF_SOURCE_TIME);
    refresh_tool_surface(LLAMA_SELF_SOURCE_INIT | LLAMA_SELF_SOURCE_TOOL_EVENT);
}

void llama_self_state::append_trace(const llama_self_state_event & event) {
    llama_self_trace_item item = {};
    item.time_point = {
        /*.wall_clock_ms =*/ datetime.wall_clock_ms,
        /*.monotonic_ms =*/ datetime.monotonic_ms,
        /*.timezone_offset_minutes =*/ datetime.timezone_offset_minutes,
    };
    item.tokens.assign(event.tokens, event.tokens + event.n_tokens);
    item.event = event;
    item.event.tokens = item.tokens.data();
    item.event.n_tokens = item.tokens.size();
    trace_items.push_back(std::move(item));

    if (trace_items.size() > LLAMA_SELF_MAX_TRACE_ITEMS) {
        trace_items.erase(trace_items.begin());
    }
}

bool llama_self_state::replay_trace(const llama_vocab * vocab, int32_t upto_count, int32_t override_channel) {
    if (!vocab) {
        return false;
    }

    if (override_channel != -1 &&
        override_channel != LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY &&
        override_channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
        return false;
    }

    if (upto_count < 0) {
        upto_count = (int32_t) trace_items.size();
    }

    const size_t limit = std::min(trace_items.size(), (size_t) upto_count);
    auto saved_trace = trace_items;
    repair_trace_item_pointers(saved_trace);
    trace_items.clear();
    reset_dynamic_state_preserve_static();

    for (size_t i = 0; i < limit; ++i) {
        auto item = saved_trace[i];
        if (override_channel != -1) {
            item.event.channel = override_channel;
        }
        if (!set_time(item.time_point)) {
            trace_items = saved_trace;
            return false;
        }

        llama_self_state_feature_vector prewrite = {};
        if (!build_prewrite_features(vocab, item.event, &prewrite) ||
            !apply_prewrite(item.event, prewrite)) {
            trace_items = saved_trace;
            return false;
        }

        llama_self_state_feature_vector postwrite = {};
        if (!build_postwrite_features(vocab, item.event, &postwrite) ||
            !apply_postwrite(item.event, postwrite)) {
            trace_items = saved_trace;
            return false;
        }
    }

    trace_items = saved_trace;
    return true;
}

bool llama_self_state::evaluate_counterfactual(
        const llama_vocab * vocab,
        const llama_self_updater_program & program,
        int32_t upto_count,
        int32_t replay_channel,
        llama_self_counterfactual_result * out_result) const {
    if (!out_result || !vocab || program.version == 0) {
        return false;
    }

    llama_self_state shadow = *this;
    repair_trace_item_pointers(shadow.trace_items);
    if (!shadow.set_updater_program(program) || !shadow.replay_trace(vocab, upto_count, replay_channel)) {
        return false;
    }

    *out_result = {
        /*.updater_version =*/ program.version,
        /*.replay_channel =*/ replay_channel,
        /*.replayed_events =*/ upto_count < 0 ? (int32_t) trace_items.size() : std::min((int32_t) trace_items.size(), upto_count),
        /*.working_memory_count =*/ shadow.working_memory_count(),
        /*.reactivation_count =*/ shadow.reactivation_count(),
        /*.uncertainty =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_UNCERTAINTY),
        /*.contradiction =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_CONTRADICTION),
        /*.memory_write_priority =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY),
        /*.broadcast_pressure =*/ shadow.current_scalar_register(LLAMA_SELF_REGISTER_BROADCAST_PRESSURE),
    };
    return true;
}

bool llama_self_state::evaluate_hypothetical_event(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        llama_self_state_delta_summary * out_delta,
        llama_self_model_state_info * out_model_state) const {
    if (!vocab || !out_delta || !out_model_state) {
        return false;
    }

    llama_self_state shadow = *this;
    repair_trace_item_pointers(shadow.trace_items);

    std::vector<float> before(registers.size(), 0.0f);
    for (size_t i = 0; i < registers.size(); ++i) {
        before[i] = registers[i].scalar_value;
    }

    llama_self_state_feature_vector prewrite = {};
    if (!shadow.build_prewrite_features(vocab, event, &prewrite) ||
            !shadow.apply_prewrite(event, prewrite)) {
        return false;
    }

    llama_self_state_feature_vector postwrite = {};
    if (!shadow.build_postwrite_features(vocab, event, &postwrite) ||
            !shadow.apply_postwrite(event, postwrite)) {
        return false;
    }

    std::vector<float> after(shadow.registers.size(), 0.0f);
    for (size_t i = 0; i < shadow.registers.size(); ++i) {
        after[i] = shadow.registers[i].scalar_value;
    }

    *out_delta = summarize_self_state_delta(before, after, event);
    shadow.get_model_state(out_model_state);
    return true;
}

float llama_self_state::max_similarity(
        const std::vector<llama_self_sketch_surface> & surfaces,
        const std::array<float, 32> & sketch,
        bool unresolved_only) const {
    float best = 0.0f;

    for (const auto & surface : surfaces) {
        if (unresolved_only && !surface.unresolved) {
            continue;
        }

        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, surface.sketch)));
        best = std::max(best, similarity * std::max(0.25f, surface.priority));
    }

    return clamp_unit(best);
}

void llama_self_state::working_memory_stats(
        const std::array<float, 32> & sketch,
        float * out_top_similarity,
        float * out_variance) const {
    if (!out_top_similarity || !out_variance) {
        return;
    }

    *out_top_similarity = 0.0f;
    *out_variance = 0.0f;

    if (working_memory.empty()) {
        return;
    }

    float top = 0.0f;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (const auto & item : working_memory) {
        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, item.sketch)));
        top = std::max(top, similarity);
        sum += similarity;
        sum_sq += similarity * similarity;
    }

    const float count = (float) working_memory.size();
    const float mean = sum / count;
    *out_top_similarity = top;
    *out_variance = clamp_unit(std::max(0.0f, sum_sq / count - mean * mean));
}

void llama_self_state::memory_handle_stats(
        const std::array<float, 32> & sketch,
        float * out_top_similarity,
        float * out_variance) const {
    if (!out_top_similarity || !out_variance) {
        return;
    }

    *out_top_similarity = 0.0f;
    *out_variance = 0.0f;

    if (memory_handles.empty()) {
        return;
    }

    float top = 0.0f;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (const auto & handle : memory_handles) {
        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, handle.centroid)));
        const float weighted = clamp_unit(similarity * std::max(0.25f, handle.priority));
        top = std::max(top, weighted);
        sum += weighted;
        sum_sq += weighted * weighted;
    }

    const float count = (float) memory_handles.size();
    const float mean = sum / count;
    *out_top_similarity = top;
    *out_variance = clamp_unit(std::max(0.0f, sum_sq / count - mean * mean));
}

void llama_self_state::admit_working_memory(
        const llama_self_state_event & event,
        const std::array<float, 32> & sketch,
        const llama_self_state_feature_vector & features) {
    llama_self_working_memory_item item = {};
    item.event_id = next_working_memory_event_id++;
    item.role = event.role;
    item.flags = event.flags;
    item.salience = clamp_unit(features.memory_write_pressure);
    item.unresolved_question = event.role == LLAMA_SELF_STATE_EVENT_USER &&
            (features.followup_hint > 0.25f || features.uncertainty_score > 0.45f);
    item.tool_affordance_hint = features.tool_pending_pressure > 0.15f ||
            features.tool_readiness_score < 0.55f ||
            current_scalar_register(LLAMA_SELF_REGISTER_AFFORDANCE) > 0.45f;
    item.admitted_monotonic_ms = datetime.monotonic_ms;
    item.sketch = sketch;
    working_memory.push_back(item);

    if (working_memory.size() > LLAMA_SELF_MAX_WORKING_MEMORY_ITEMS) {
        working_memory.erase(working_memory.begin());
    }

    bridge_working_memory_to_handles(sketch, item.salience);
}

float llama_self_state::run_contradiction_head(float analytic_score, const llama_self_state_feature_vector & features) const {
    if (params.enable_builtin_contradiction_probe) {
        static constexpr std::array<float, 12> weights = {{
            2.25f, 1.50f, 0.25f, 0.10f, 0.30f, 0.35f,
            0.90f, 0.35f, 0.40f, 0.20f, 0.10f, -0.20f,
        }};
        const std::array<float, 12> probe_features = {{
            features.negation_ratio,
            features.error_ratio,
            features.uncertainty_lexical_ratio,
            features.decoder_entropy,
            1.0f - features.decoder_top_margin,
            features.novelty,
            features.commitment_top_similarity,
            features.tool_pending_pressure,
            1.0f - features.tool_readiness_score,
            features.memory_handle_similarity_variance,
            analytic_score,
            features.social_trust,
        }};
        const float builtin_score = linear_probe_score(-1.15f, weights, probe_features);
        analytic_score = clamp_unit(0.55f * analytic_score + 0.45f * builtin_score);
    }

    if (!params.enable_learned_contradiction_head || !params.contradiction_head_callback) {
        return analytic_score;
    }

    float head_score = analytic_score;
    if (!params.contradiction_head_callback(&features, &head_score, params.contradiction_head_user_data) || !std::isfinite(head_score)) {
        return analytic_score;
    }

    return clamp_unit(head_score);
}

float llama_self_state::run_uncertainty_head(float analytic_score, const llama_self_state_feature_vector & features) const {
    if (params.enable_builtin_uncertainty_probe) {
        static constexpr std::array<float, 12> weights = {{
            0.20f, 0.55f, 2.30f, 1.85f, 1.40f, 0.30f,
            0.10f, 0.25f, 0.35f, 0.25f, 0.10f, -0.15f,
        }};
        const std::array<float, 12> probe_features = {{
            features.negation_ratio,
            features.error_ratio,
            features.uncertainty_lexical_ratio,
            features.decoder_entropy,
            1.0f - features.decoder_top_margin,
            features.novelty,
            features.commitment_top_similarity,
            features.tool_pending_pressure,
            1.0f - features.tool_readiness_score,
            features.memory_handle_similarity_variance,
            analytic_score,
            features.social_trust,
        }};
        const float builtin_score = linear_probe_score(-1.30f, weights, probe_features);
        analytic_score = clamp_unit(0.55f * analytic_score + 0.45f * builtin_score);
    }

    if (!params.enable_learned_uncertainty_head || !params.uncertainty_head_callback) {
        return analytic_score;
    }

    float head_score = analytic_score;
    if (!params.uncertainty_head_callback(&features, &head_score, params.uncertainty_head_user_data) || !std::isfinite(head_score)) {
        return analytic_score;
    }

    return clamp_unit(head_score);
}

float llama_self_state::run_broadcast_head(float analytic_score, const llama_self_state_feature_vector & features) const {
    if (params.enable_builtin_broadcast_probe) {
        static constexpr std::array<float, 12> weights = {{
            1.40f, -1.30f, 0.85f, 0.55f, 0.45f, 0.30f,
            0.40f, 0.20f, 0.18f, 0.12f, 0.10f, 0.35f,
        }};
        const std::array<float, 12> probe_features = {{
            analytic_score,
            features.broadcast_inhibition_hint,
            features.followup_hint,
            features.tool_pending_pressure,
            1.0f - features.tool_readiness_score,
            features.goal_top_similarity,
            features.social_bond_strength,
            features.contradiction_score,
            features.uncertainty_score,
            features.memory_handle_top_similarity,
            features.recency_user,
            features.recency_emit,
        }};
        const float builtin_score = linear_probe_score(-0.95f, weights, probe_features);
        analytic_score = clamp_unit(0.60f * analytic_score + 0.40f * builtin_score);
    }

    if (!params.enable_learned_broadcast_head || !params.broadcast_head_callback) {
        return analytic_score;
    }

    float head_score = analytic_score;
    if (!params.broadcast_head_callback(&features, &head_score, params.broadcast_head_user_data) || !std::isfinite(head_score)) {
        return analytic_score;
    }

    return clamp_unit(head_score);
}

bool llama_self_state::build_features(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        bool postwrite,
        llama_self_state_feature_vector * out_features) const {
    if (!vocab || !out_features) {
        return false;
    }

    static const char * const negation_terms[] = {"not", "no", "never", "n't", "cannot", "can't", "failed", "error", "wrong", "false"};
    static const char * const uncertainty_terms[] = {"maybe", "perhaps", "uncertain", "unknown", "unsure", "likely", "possibly"};
    static const char * const self_terms[] = {"i", "me", "my", "myself", "vicuna"};
    static const char * const error_terms[] = {"error", "fail", "failed", "cannot", "can't", "invalid", "denied", "timeout"};
    static const char * const imperative_terms[] = {"do", "please", "use", "make", "keep", "write", "show", "run", "fix", "give"};
    static const char * const negative_valence_terms[] = {
            "bad", "worse", "wrong", "frustrat", "annoy", "disappoint", "hate", "awful", "terrible", "useless", "upset"
    };

    const auto sketch = build_event_sketch(event);
    const float previous_similarity = has_previous_event_sketch ?
            clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, previous_event_sketch))) : 0.0f;
    float working_memory_top_similarity = 0.0f;
    float working_memory_similarity_variance = 0.0f;
    working_memory_stats(sketch, &working_memory_top_similarity, &working_memory_similarity_variance);
    float memory_handle_top_similarity = 0.0f;
    float memory_handle_similarity_variance = 0.0f;
    memory_handle_stats(sketch, &memory_handle_top_similarity, &memory_handle_similarity_variance);
    const float goal_top_similarity = max_similarity(goals, sketch, false);
    const float commitment_top_similarity = max_similarity(commitments, sketch, true);
    const float identity_similarity = has_identity_sketch ?
            clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, identity_sketch))) : 0.0f;
    const llama_self_tool_state_info tool_state = build_tool_state_info(tool_jobs);

    float negation_hits = 0.0f;
    float uncertainty_hits = 0.0f;
    float self_hits = 0.0f;
    float error_hits = 0.0f;
    float negative_valence_hits = 0.0f;
    float imperative_hits = 0.0f;
    uint32_t list_hits = 0;
    uint32_t newline_hits = 0;
    uint32_t emphasis_hits = 0;
    uint32_t question_hits = 0;
    std::unordered_set<llama_token> unique_tokens;

    for (size_t i = 0; i < event.n_tokens; ++i) {
        unique_tokens.insert(event.tokens[i]);
        const std::string raw_piece = vocab->token_to_piece(event.tokens[i]);
        const std::string piece = normalize_piece(raw_piece);

        if (contains_any(piece, negation_terms, sizeof(negation_terms)/sizeof(negation_terms[0]))) {
            negation_hits += 1.0f;
        }
        if (contains_any(piece, uncertainty_terms, sizeof(uncertainty_terms)/sizeof(uncertainty_terms[0]))) {
            uncertainty_hits += 1.0f;
        }
        if (contains_any(piece, self_terms, sizeof(self_terms)/sizeof(self_terms[0]))) {
            self_hits += 1.0f;
        }
        if (contains_any(piece, error_terms, sizeof(error_terms)/sizeof(error_terms[0]))) {
            error_hits += 1.0f;
        }
        if (contains_any(piece, negative_valence_terms, sizeof(negative_valence_terms)/sizeof(negative_valence_terms[0]))) {
            negative_valence_hits += 1.0f;
        }
        if (contains_any(piece, imperative_terms, sizeof(imperative_terms)/sizeof(imperative_terms[0]))) {
            imperative_hits += 1.0f;
        }
        if (piece.find('?') != std::string::npos) {
            ++question_hits;
        }
        if (raw_piece.find('\n') != std::string::npos) {
            ++newline_hits;
        }
        if (raw_piece.find("- ") != std::string::npos || raw_piece.find("* ") != std::string::npos || raw_piece.find("1.") != std::string::npos) {
            ++list_hits;
        }
        if (raw_piece.find('!') != std::string::npos || raw_piece.find("ALL") != std::string::npos) {
            ++emphasis_hits;
        }
    }

    const float token_count = (float) event.n_tokens;
    const float inv_tokens = token_count > 0.0f ? 1.0f / token_count : 0.0f;
    const float unique_ratio = token_count > 0.0f ? (float) unique_tokens.size() / token_count : 0.0f;

    const float continuity_similarity = std::max(previous_similarity, working_memory_top_similarity);
    const float novelty = token_count > 0.0f ? clamp_unit(1.0f - continuity_similarity) : 0.0f;
    const float topic_shift = clamp_unit(0.65f * novelty + 0.35f * (1.0f - previous_similarity));

    const float entropy_feature = event.decoder_entropy > 0.0f ? clamp_unit(event.decoder_entropy / 5.0f) : 0.0f;
    const float top_margin_feature = event.decoder_top_margin > 0.0f ? clamp_unit(event.decoder_top_margin) : 0.0f;

    const float negation_ratio = negation_hits * inv_tokens;
    const float uncertainty_ratio = uncertainty_hits * inv_tokens;
    const float self_ratio = self_hits * inv_tokens;
    const float error_ratio = error_hits * inv_tokens;
    const float negative_valence_ratio = negative_valence_hits * inv_tokens;
    const float imperative_ratio = imperative_hits * inv_tokens;
    const float list_ratio = token_count > 0.0f ? clamp_unit((float) list_hits / token_count) : 0.0f;
    const float newline_ratio = token_count > 0.0f ? clamp_unit((float) newline_hits / token_count) : 0.0f;
    const float emphasis_ratio = token_count > 0.0f ? clamp_unit((float) emphasis_hits / token_count) : 0.0f;

    const float contradiction_analytic = clamp_unit(
            0.35f * negation_ratio +
            0.20f * error_ratio +
            0.20f * commitment_top_similarity * std::max(negation_ratio, error_ratio) +
            0.15f * ((event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f) +
            0.10f * novelty);

    const float uncertainty_analytic = clamp_unit(std::max(
            uncertainty_ratio,
            0.60f * entropy_feature + 0.40f * (1.0f - top_margin_feature)));
    const float social_bond_strength = clamp_unit(
            0.40f * social_familiarity +
            0.35f * social_trust +
            0.25f * social_reciprocity);
    const bool counterfactual_channel = event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL;

    llama_self_state_feature_vector features = {};
    features.token_count_log = clamp_unit(std::log1pf(token_count) / 8.0f);
    features.unique_token_ratio = clamp_unit(unique_ratio);
    features.novelty = novelty;
    features.topic_shift = topic_shift;
    features.working_memory_top_similarity = working_memory_top_similarity;
    features.working_memory_similarity_variance = working_memory_similarity_variance;
    features.memory_handle_top_similarity = memory_handle_top_similarity;
    features.memory_handle_similarity_variance = memory_handle_similarity_variance;
    features.goal_top_similarity = goal_top_similarity;
    features.commitment_top_similarity = commitment_top_similarity;
    features.identity_similarity = identity_similarity;
    features.self_reference_ratio = clamp_unit(self_ratio);
    features.negation_ratio = clamp_unit(negation_ratio);
    features.uncertainty_lexical_ratio = clamp_unit(uncertainty_ratio);
    features.error_ratio = clamp_unit(error_ratio);
    features.recency_user = decay_to_unit(datetime.delta_since_last_user_ms, params.tool_salience_half_life_ms);
    features.recency_tool = decay_to_unit(datetime.delta_since_last_tool_event_ms, params.tool_salience_half_life_ms);
    features.recency_emit = decay_to_unit(datetime.delta_since_last_emit_ms, params.tool_salience_half_life_ms);
    features.social_familiarity = social_familiarity;
    features.social_trust = social_trust;
    features.social_reciprocity = social_reciprocity;
    features.social_bond_strength = social_bond_strength;
    features.tool_readiness_score = tool_state.readiness;
    features.tool_pending_pressure = clamp_unit(
            0.35f * std::min(1.0f, (float) tool_state.pending_jobs) +
            0.45f * std::min(1.0f, (float) tool_state.running_jobs) +
            0.20f * std::min(1.0f, (float) tool_state.failed_jobs));
    features.decoder_entropy = entropy_feature;
    features.decoder_top_margin = top_margin_feature;
    features.contradiction_score = run_contradiction_head(contradiction_analytic, features);
    features.uncertainty_score = run_uncertainty_head(uncertainty_analytic, features);
    features.memory_write_pressure = clamp_unit(
            updater_program.memory_novelty_weight * features.novelty +
            updater_program.memory_working_similarity_weight * (1.0f - features.working_memory_top_similarity) +
            updater_program.memory_handle_similarity_weight * (1.0f - features.memory_handle_top_similarity) +
            updater_program.memory_uncertainty_weight * features.uncertainty_score +
            updater_program.memory_contradiction_weight * features.contradiction_score +
            updater_program.memory_handle_variance_weight * features.memory_handle_similarity_variance);

    const float social_hint = clamp_unit(
            (event.role == LLAMA_SELF_STATE_EVENT_USER ? 0.55f : 0.20f) +
            0.25f * social_bond_strength +
            0.20f * social_reciprocity);
    const float question_hint = token_count > 0.0f ? clamp_unit((float) question_hits / token_count) : 0.0f;
    const float broadcast_pressure_base = postwrite ? clamp_unit(
            updater_program.broadcast_social_weight * social_hint +
            updater_program.broadcast_contradiction_weight * features.contradiction_score +
            updater_program.broadcast_uncertainty_weight * features.uncertainty_score +
            updater_program.broadcast_tool_pending_weight * features.tool_pending_pressure +
            updater_program.broadcast_tool_unready_weight * (1.0f - features.tool_readiness_score) +
            updater_program.broadcast_failure_weight * ((event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f) +
            updater_program.broadcast_question_weight * question_hint +
            updater_program.broadcast_goal_weight * features.goal_top_similarity) : 0.0f;
    features.broadcast_pressure_hint = counterfactual_channel ? 0.15f * broadcast_pressure_base : broadcast_pressure_base;

    const float interruption_guard = channel_state == LLAMA_SELF_STATE_CHANNEL_DO_NOT_INTERRUPT ? 1.0f :
            channel_state == LLAMA_SELF_STATE_CHANNEL_WAITING ? 0.55f : 0.20f;
    const float broadcast_inhibition_base = postwrite ? clamp_unit(
            0.60f * interruption_guard +
            0.20f * (1.0f - features.broadcast_pressure_hint) +
            0.10f * features.tool_pending_pressure +
            0.20f * (event.role == LLAMA_SELF_STATE_EVENT_SYSTEM ? 0.5f : 0.0f)) : 0.0f;
    features.broadcast_inhibition_hint = counterfactual_channel ?
            clamp_unit(std::max(0.80f, broadcast_inhibition_base)) :
            broadcast_inhibition_base;

    features.followup_hint = postwrite ? clamp_unit(
            0.45f * ((event.flags & LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP) ? 1.0f : 0.0f) +
            0.30f * ((event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? 1.0f : 0.0f) +
            0.15f * features.contradiction_score +
            0.10f * features.uncertainty_score) : 0.0f;
    features.negative_user_valence = clamp_unit(negative_valence_ratio);
    features.question_ratio = question_hint;
    features.imperative_ratio = clamp_unit(imperative_ratio);
    features.list_ratio = list_ratio;
    features.newline_ratio = newline_ratio;
    features.emphasis_ratio = emphasis_ratio;
    if (postwrite) {
        features.broadcast_pressure_hint = run_broadcast_head(features.broadcast_pressure_hint, features);
    }

    *out_features = features;
    return true;
}

bool llama_self_state::build_prewrite_features(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return build_features(vocab, event, false, out_features);
}

bool llama_self_state::apply_prewrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features) {
    if (!ensure_time_initialized()) {
        return false;
    }

    const uint32_t source_mask = source_mask_for_role(event);
    return apply_register_update_rules(LLAMA_SELF_UPDATER_PHASE_PREWRITE, event, features, source_mask);
}

bool llama_self_state::build_postwrite_features(
        const llama_vocab * vocab,
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return build_features(vocab, event, true, out_features);
}

bool llama_self_state::apply_postwrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features) {
    if (!ensure_time_initialized()) {
        return false;
    }

    const uint32_t source_mask = source_mask_for_role(event);
    (void) apply_register_update_rules(LLAMA_SELF_UPDATER_PHASE_POSTWRITE, event, features, source_mask);

    update_reactivation_priorities(build_event_sketch(event), features.memory_write_pressure);

    if ((event.flags & LLAMA_SELF_STATE_EVENT_ADMITTED) != 0) {
        admit_working_memory(event, build_event_sketch(event), features);

        if (event.channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
            if (event.role == LLAMA_SELF_STATE_EVENT_USER) {
                (void) note_user_event();
                (void) set_channel_state(LLAMA_SELF_STATE_CHANNEL_ACTIVE);
            } else if (event.role == LLAMA_SELF_STATE_EVENT_TOOL ||
                       (event.flags & (LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED | LLAMA_SELF_STATE_EVENT_TOOL_FAILED)) != 0) {
                (void) note_tool_event();
            } else if (event.role == LLAMA_SELF_STATE_EVENT_SYSTEM || (event.flags & LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP) != 0) {
                (void) note_emit_event();
            }
        }
    }

    if (event.channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) {
        update_social_state(event, features);
        update_expanded_model(event, features, source_mask);
    }
    previous_event_sketch = build_event_sketch(event);
    has_previous_event_sketch = event.n_tokens > 0;
    append_trace(event);
    return true;
}

bool llama_self_state::note_validated_progress(float signed_progress, float efficiency_advantage) {
    if (!ensure_time_initialized()) {
        return false;
    }

    update_evolution_uncertainty(
            LLAMA_SELF_SOURCE_COUNTERFACTUAL,
            clamp_signed_unit(signed_progress),
            clamp_signed_unit(efficiency_advantage));
    return true;
}

void llama_self_state::update_reactivation_priorities(
        const std::array<float, 32> & sketch,
        float memory_write_pressure) {
    reactivation_priorities.clear();

    for (const auto & handle : memory_handles) {
        const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, handle.centroid)));
        const float recency = decay_to_unit(elapsed_or_unset(datetime.monotonic_ms, handle.last_update_monotonic_ms), params.tool_salience_half_life_ms);
        const float priority = clamp_unit(
                0.50f * clamp_unit(memory_write_pressure * similarity) +
                0.30f * handle.priority +
                0.20f * recency);

        if (priority <= 0.0f) {
            continue;
        }

        reactivation_priorities.push_back({
            /*.handle_id =*/ handle.handle_id,
            /*.kind =*/ handle.kind,
            /*.priority =*/ priority,
            /*.top_similarity =*/ similarity,
            /*.last_update_monotonic_ms =*/ datetime.monotonic_ms,
        });
    }

    std::sort(reactivation_priorities.begin(), reactivation_priorities.end(), [](const llama_self_reactivation_info & lhs, const llama_self_reactivation_info & rhs) {
        if (lhs.priority == rhs.priority) {
            return lhs.handle_id < rhs.handle_id;
        }
        return lhs.priority > rhs.priority;
    });
}

void llama_self_state::refresh_tool_surface(uint32_t source_mask) {
    const llama_self_tool_state_info info = build_tool_state_info(tool_jobs);
    const float lifecycle_salience =
            info.failed_jobs > 0 ? 1.0f :
            info.running_jobs > 0 ? 0.92f :
            info.pending_jobs > 0 ? 0.72f :
            info.completed_jobs > 0 ? 0.38f :
            decay_to_unit(datetime.delta_since_last_tool_event_ms, params.tool_salience_half_life_ms);

    update_scalar_register(LLAMA_SELF_REGISTER_TOOL_SALIENCE, lifecycle_salience, source_mask);
}

void llama_self_state::bridge_working_memory_to_handles(
        const std::array<float, 32> & sketch,
        float salience) {
    auto best_it = memory_handles.end();
    float best_similarity = 0.0f;

    for (auto it = memory_handles.begin(); it != memory_handles.end(); ++it) {
        if (it->kind == LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER) {
            const float similarity = clamp_unit(0.5f * (1.0f + sketch_similarity(sketch, it->centroid)));
            if (best_it == memory_handles.end() || similarity > best_similarity) {
                best_it = it;
                best_similarity = similarity;
            }
        }
    }

    if (best_it == memory_handles.end()) {
        return;
    }

    auto & target = *best_it;
    const float blend = clamp_unit(0.20f + 0.60f * salience);
    const float existing_weight = (float) std::max<uint32_t>(1, target.member_count);
    for (size_t i = 0; i < target.centroid.size(); ++i) {
        target.centroid[i] = (1.0f - blend) * target.centroid[i] + blend * sketch[i];
    }

    float norm = 0.0f;
    for (float value : target.centroid) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float & value : target.centroid) {
            value /= norm;
        }
    }

    target.member_count = (uint32_t) std::min<float>(65535.0f, existing_weight + 1.0f);
    target.priority = clamp_unit(0.70f * target.priority + 0.30f * salience);
    target.last_update_monotonic_ms = datetime.monotonic_ms;
}

void llama_self_state::update_social_state(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) {
    social_last_update_monotonic_ms = datetime.monotonic_ms;

    if (event.role == LLAMA_SELF_STATE_EVENT_USER) {
        ++social_user_turn_count;
        social_familiarity = clamp_unit(social_familiarity + 0.18f * (1.0f - social_familiarity));
        social_recent_user_valence = clamp_unit(
                0.65f * social_recent_user_valence +
                0.35f * features.negative_user_valence);

        const float response_bonus = clamp_unit(0.45f * features.recency_emit + 0.55f * features.working_memory_top_similarity);
        social_reciprocity = clamp_unit(social_reciprocity + 0.15f * (response_bonus - social_reciprocity));

        const float trust_target = clamp_unit(
                0.55f +
                0.20f * (1.0f - features.contradiction_score) +
                0.15f * (1.0f - features.uncertainty_score) +
                0.10f * features.goal_top_similarity);
        social_trust = clamp_unit(social_trust + 0.10f * (trust_target - social_trust));
    } else {
        ++social_system_turn_count;
        const bool failed = (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) != 0;
        const float trust_target = failed ?
                clamp_unit(0.20f * (1.0f - features.contradiction_score)) :
                clamp_unit(0.45f + 0.25f * (1.0f - features.uncertainty_score) + 0.15f * features.social_bond_strength);
        social_trust = clamp_unit(social_trust + (failed ? 0.18f : 0.08f) * (trust_target - social_trust));

        const float reciprocity_target = clamp_unit(
                0.35f +
                0.25f * features.followup_hint +
                0.20f * features.recency_user +
                0.20f * features.goal_top_similarity);
        social_reciprocity = clamp_unit(social_reciprocity + 0.08f * (reciprocity_target - social_reciprocity));
    }

    social_dissatisfaction = clamp_unit(
            0.50f * social_recent_user_valence +
            0.30f * (1.0f - social_trust) +
            0.20f * (1.0f - social_reciprocity));
}

void llama_self_state::initialize_model_state() {
    for (int32_t i = 0; i < LLAMA_SELF_HORIZON_COUNT; ++i) {
        model_horizons[(size_t) i] = {};
        model_horizons[(size_t) i].horizon_id = i;
        model_horizons[(size_t) i].last_update_monotonic_ms = -1;
    }
    model_forecast = {};
    prediction_error = {};
    initialize_belief_state();
    extension_summary = {};
    extension_trace = {};
    refresh_model_extension_summary();
}

void llama_self_state::initialize_belief_state() {
    belief_summary = {};
    for (auto & slot : belief_slots) {
        slot = {};
        slot.last_update_monotonic_ms = -1;
    }
    for (auto & signature : belief_slot_signatures) {
        signature.fill(0.0f);
    }
    for (auto & candidate : promotion_candidates) {
        candidate = {};
    }
    promotion_candidate_count = 0;
}

void llama_self_state::refresh_belief_summary() {
    belief_summary = {};
    if (!params.enable_belief_state || params.belief_slot_count <= 0) {
        return;
    }

    const int32_t slot_count = std::min<int32_t>(params.belief_slot_count, LLAMA_SELF_BELIEF_MAX_SLOTS);
    std::array<float, LLAMA_SELF_BELIEF_MAX_SLOTS> pressures = {};
    float pressure_sum = 0.0f;
    float confidence_sum = 0.0f;
    float novelty_sum = 0.0f;
    float memory_sum = 0.0f;
    float forecast_sum = 0.0f;
    float weighted_slot_pressure = 0.0f;

    for (int32_t i = 0; i < slot_count; ++i) {
        const auto & slot = belief_slots[(size_t) i];
        pressures[(size_t) i] = clamp_unit(slot.pressure);
        pressure_sum += pressures[(size_t) i];
        confidence_sum += clamp_unit(slot.confidence);
        novelty_sum += clamp_unit(slot.novelty_support);
        memory_sum += clamp_unit(slot.memory_support);
        forecast_sum += clamp_unit(slot.forecast_error_support);
        weighted_slot_pressure += clamp_unit(slot.pressure) * clamp_unit(slot.confidence);
        belief_summary.max_slot_pressure = std::max(belief_summary.max_slot_pressure, clamp_unit(slot.pressure));
    }

    const float inv_count = slot_count > 0 ? 1.0f / (float) slot_count : 0.0f;
    const float forecast_error_mass = prediction_error.valid ?
            clamp_unit(
                    0.28f * prediction_error.steps_error +
                    0.20f * prediction_error.inference_cost_error +
                    0.18f * prediction_error.satisfaction_error +
                    0.18f * prediction_error.recovery_error +
                    0.16f * prediction_error.goal_progress_error) :
            0.0f;
    const float known_uncertainty = clamp_unit(
            0.45f * current_scalar_register(LLAMA_SELF_REGISTER_UNCERTAINTY) +
            0.35f * current_scalar_register(LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY) +
            0.20f * current_scalar_register(LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY));
    const float missing_uncertainty = clamp_unit(
            0.55f * novelty_sum * inv_count +
            0.25f * belief_summary.max_slot_pressure +
            0.20f * (1.0f - clamp_unit(extension_summary.context_activation)));
    const float unmodeled_uncertainty = clamp_unit(
            0.45f * forecast_error_mass +
            0.25f * weighted_slot_pressure * inv_count +
            0.15f * memory_sum * inv_count +
            0.15f * current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY));

    belief_summary.valid = true;
    belief_summary.known_care_uncertainty = known_uncertainty;
    belief_summary.missing_observation_uncertainty = missing_uncertainty;
    belief_summary.unmodeled_care_uncertainty = unmodeled_uncertainty;
    belief_summary.residual_allostatic_pressure = clamp_unit(
            0.45f * forecast_error_mass +
            0.30f * weighted_slot_pressure * inv_count +
            0.15f * extension_summary.allostatic_divergence +
            0.10f * current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY));
    belief_summary.promotion_readiness = clamp_unit(
            0.40f * belief_summary.max_slot_pressure +
            0.30f * memory_sum * inv_count +
            0.20f * forecast_sum * inv_count +
            0.10f * (1.0f - missing_uncertainty));
    belief_summary.belief_entropy = normalized_entropy(pressures);
    belief_summary.belief_confidence = clamp_unit(
            0.55f * confidence_sum * inv_count +
            0.25f * (1.0f - forecast_error_mass) +
            0.20f * (1.0f - missing_uncertainty));
    belief_summary.slot_pressure_mean = clamp_unit(pressure_sum * inv_count);
}

void llama_self_state::refresh_belief_promotion_candidates() {
    for (auto & candidate : promotion_candidates) {
        candidate = {};
    }
    promotion_candidate_count = 0;

    if (!params.enable_belief_state || params.belief_slot_count <= 0) {
        return;
    }

    struct scored_slot {
        int32_t index = -1;
        float score = 0.0f;
    };
    std::array<scored_slot, LLAMA_SELF_BELIEF_MAX_SLOTS> scored = {};
    const int32_t slot_count = std::min<int32_t>(params.belief_slot_count, LLAMA_SELF_BELIEF_MAX_SLOTS);
    for (int32_t i = 0; i < slot_count; ++i) {
        const auto & slot = belief_slots[(size_t) i];
        scored[(size_t) i] = {
            /*.index =*/ i,
            /*.score =*/ clamp_unit(
                    0.40f * slot.pressure +
                    0.25f * slot.confidence +
                    0.20f * slot.memory_support +
                    0.15f * slot.forecast_error_support),
        };
    }
    std::sort(scored.begin(), scored.begin() + slot_count, [](const scored_slot & lhs, const scored_slot & rhs) {
        return lhs.score > rhs.score;
    });

    for (int32_t i = 0; i < slot_count && promotion_candidate_count < LLAMA_SELF_BELIEF_MAX_PROMOTION_CANDIDATES; ++i) {
        const auto & choice = scored[(size_t) i];
        if (choice.index < 0 || choice.score < params.belief_promotion_threshold) {
            continue;
        }

        const auto & slot = belief_slots[(size_t) choice.index];
        auto & candidate = promotion_candidates[(size_t) promotion_candidate_count++];
        candidate.valid = true;
        candidate.slot_index = choice.index;
        candidate.support_score = choice.score;
        candidate.allostatic_relevance = clamp_unit(
                0.55f * slot.pressure +
                0.25f * belief_summary.residual_allostatic_pressure +
                0.20f * extension_summary.allostatic_divergence);
        candidate.suggested_desired_value = 0.0f;
        candidate.stability_score = clamp_unit(
                0.45f * slot.confidence +
                0.30f * slot.memory_support +
                0.25f * (1.0f - belief_summary.missing_observation_uncertainty));
        std::snprintf(
                candidate.suggested_label,
                sizeof(candidate.suggested_label),
                "latent_residue_slot_%d",
                choice.index);
    }
}

void llama_self_state::update_summary_registers(uint32_t source_mask) {
    const auto & instant = model_horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT];
    const float previous_satisfaction_risk = current_scalar_register(LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK);
    const float previous_loop_inefficiency = current_scalar_register(LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY);
    const float previous_recovery_urgency = current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY);
    const float previous_answerability = current_scalar_register(LLAMA_SELF_REGISTER_ANSWERABILITY);
    const float previous_preference_uncertainty = current_scalar_register(LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY);
    const float satisfaction_risk = clamp_unit(
            0.55f * instant.user_outcome.frustration_risk +
            0.25f * instant.user_outcome.misunderstanding_risk +
            0.20f * instant.user_outcome.trust_repair_need);
    const float goal_progress_pressure = clamp_unit(
            0.55f * (1.0f - instant.goal_progress.goal_progress_estimate) +
            0.25f * instant.goal_progress.blocker_severity +
            0.20f * instant.goal_progress.urgency);
    const float recovery_urgency = clamp_unit(
            0.30f * max4(
                    instant.recovery.favorable_divergence_goal,
                    instant.recovery.favorable_divergence_social,
                    instant.recovery.favorable_divergence_epistemic,
                    instant.recovery.favorable_divergence_action) +
            0.25f * instant.recovery.regulation_debt +
            0.20f * instant.recovery.unresolved_tension_load +
            0.15f * (1.0f - instant.recovery.recovery_momentum) +
            0.10f * extension_summary.allostatic_divergence);

    blend_scalar_register(LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK, satisfaction_risk, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE, goal_progress_pressure, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY, instant.efficiency.loop_inefficiency, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY, recovery_urgency, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_ANSWERABILITY, instant.epistemic.answerability, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY, instant.user_outcome.preference_uncertainty, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_USER_DIRECTNESS_PREFERENCE, instant.user_preference.directness_preference, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_USER_VERBOSITY_PREFERENCE, instant.user_preference.verbosity_preference, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_USER_STRUCTURE_PREFERENCE, instant.user_preference.structure_preference, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_USER_AUTONOMY_PREFERENCE, instant.user_preference.autonomy_preference, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_USER_CLARIFICATION_PREFERENCE, instant.user_preference.clarification_preference, params.postwrite_gain, source_mask);
    blend_scalar_register(LLAMA_SELF_REGISTER_USER_DISAGREEMENT_SENSITIVITY, instant.user_preference.disagreement_sensitivity, params.postwrite_gain, source_mask);

    const float signed_progress = clamp_signed_unit(
            0.28f * (previous_recovery_urgency - current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY)) +
            0.24f * (previous_loop_inefficiency - current_scalar_register(LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY)) +
            0.18f * (current_scalar_register(LLAMA_SELF_REGISTER_ANSWERABILITY) - previous_answerability) +
            0.15f * (previous_preference_uncertainty - current_scalar_register(LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY)) +
            0.15f * (previous_satisfaction_risk - current_scalar_register(LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK)));
    const float efficiency_advantage = clamp_signed_unit(
            0.60f * (previous_loop_inefficiency - current_scalar_register(LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY)) +
            0.40f * (previous_recovery_urgency - current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY)));
    update_evolution_uncertainty(source_mask, signed_progress, efficiency_advantage);
}

void llama_self_state::update_belief_state(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features,
        uint32_t /*source_mask*/) {
    if (!params.enable_belief_state || params.belief_slot_count <= 0) {
        belief_summary = {};
        promotion_candidate_count = 0;
        return;
    }

    const int32_t slot_count = std::min<int32_t>(params.belief_slot_count, LLAMA_SELF_BELIEF_MAX_SLOTS);
    const float forecast_error_mass = prediction_error.valid ?
            clamp_unit(
                    0.28f * prediction_error.steps_error +
                    0.20f * prediction_error.inference_cost_error +
                    0.18f * prediction_error.satisfaction_error +
                    0.18f * prediction_error.recovery_error +
                    0.16f * prediction_error.goal_progress_error) :
            0.0f;
    const float missing_obs = clamp_unit(
            0.45f * features.novelty +
            0.25f * features.topic_shift +
            0.15f * features.uncertainty_score +
            0.15f * (1.0f - model_horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT].epistemic.evidence_sufficiency));
    const float memory_residue = clamp_unit(
            0.45f * extension_summary.context_activation +
            0.30f * std::min(1.0f, 0.25f * extension_summary.hard_memory_count) +
            0.25f * features.memory_handle_top_similarity);
    const float counterfactual_miss = event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ?
            clamp_unit(0.45f * forecast_error_mass + 0.30f * features.uncertainty_score + 0.25f * features.contradiction_score) :
            0.0f;
    const float residual_mass = clamp_unit(
            params.belief_forecast_error_weight * forecast_error_mass +
            params.belief_missing_observation_weight * missing_obs +
            params.belief_memory_residue_weight * memory_residue +
            params.belief_counterfactual_miss_weight * counterfactual_miss);
    const float unmodeled_pressure = clamp_unit(
            residual_mass *
            clamp_unit(
                    0.55f +
                    params.belief_unmodeled_care_weight * 0.35f +
                    0.20f * current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY)));

    const std::array<float, LLAMA_SELF_BELIEF_SIGNATURE_DIM> evidence = {
        missing_obs,
        memory_residue,
        forecast_error_mass,
        clamp_unit(0.60f * unmodeled_pressure + 0.40f * current_scalar_register(LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY)),
    };

    int32_t slot_index = 0;
    float best_score = -1.0f;
    for (int32_t i = 0; i < slot_count; ++i) {
        const auto & slot = belief_slots[(size_t) i];
        const float signature_score = dot_product(belief_slot_signatures[(size_t) i], evidence);
        const float availability = slot.last_update_monotonic_ms < 0 ? 0.25f : 0.0f;
        const float score = signature_score + availability + 0.05f * (1.0f - clamp_unit(slot.pressure));
        if (score > best_score) {
            best_score = score;
            slot_index = i;
        }
    }

    const float step = clamp_unit(std::min(params.belief_max_update_step, residual_mass));
    auto & slot = belief_slots[(size_t) slot_index];
    auto & signature = belief_slot_signatures[(size_t) slot_index];
    if (slot.last_update_monotonic_ms < 0) {
        signature = evidence;
    } else {
        for (size_t i = 0; i < LLAMA_SELF_BELIEF_SIGNATURE_DIM; ++i) {
            signature[i] = clamp_unit(blend_value(signature[i], evidence[i], step));
        }
    }

    const float decay = clamp_unit(params.belief_residual_decay);
    for (int32_t i = 0; i < slot_count; ++i) {
        if (i == slot_index) {
            continue;
        }
        belief_slots[(size_t) i].pressure = clamp_unit(belief_slots[(size_t) i].pressure * (1.0f - 0.5f * decay));
        belief_slots[(size_t) i].confidence = clamp_unit(std::max(
                params.belief_confidence_floor,
                belief_slots[(size_t) i].confidence * (1.0f - 0.35f * decay)));
    }

    slot.pressure = clamp_unit(std::min(
            params.belief_pressure_clip,
            blend_value(slot.pressure * (1.0f - 0.30f * decay), unmodeled_pressure, step)));
    slot.confidence = clamp_unit(std::max(
            params.belief_confidence_floor,
            blend_value(slot.confidence, clamp_unit(0.50f * residual_mass + 0.25f * memory_residue + 0.25f * (1.0f - missing_obs)), step)));
    slot.novelty_support = clamp_unit(blend_value(slot.novelty_support, missing_obs, step));
    slot.memory_support = clamp_unit(blend_value(slot.memory_support, memory_residue, step));
    slot.forecast_error_support = clamp_unit(blend_value(slot.forecast_error_support, forecast_error_mass, step));
    slot.last_update_monotonic_ms = datetime.monotonic_ms;

    refresh_belief_summary();
    refresh_belief_promotion_candidates();
}

void llama_self_state::update_evolution_uncertainty(
        uint32_t source_mask,
        float signed_progress,
        float efficiency_advantage) {
    if (datetime.monotonic_ms < 0) {
        return;
    }

    if (last_validated_progress_monotonic_ms < 0) {
        last_validated_progress_monotonic_ms =
                session_start_monotonic_ms >= 0 ? session_start_monotonic_ms : datetime.monotonic_ms;
    }

    const int64_t now_ms = datetime.monotonic_ms;
    const int64_t since_progress_ms = elapsed_or_unset(now_ms, last_validated_progress_monotonic_ms);
    const int64_t since_update_ms = elapsed_or_unset(
            now_ms,
            registers[(size_t) LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY].last_update_monotonic_ms);
    const float time_pressure = clamp_unit((float) since_progress_ms / LLAMA_SELF_EVOLUTION_PROGRESS_HORIZON_MS);
    const float drift_pressure = clamp_unit(
            0.35f * current_scalar_register(LLAMA_SELF_REGISTER_RECOVERY_URGENCY) +
            0.25f * current_scalar_register(LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY) +
            0.20f * current_scalar_register(LLAMA_SELF_REGISTER_UNCERTAINTY) +
            0.20f * current_scalar_register(LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY));
    const float target = clamp_unit(0.58f * time_pressure + 0.42f * drift_pressure);
    const float current = current_scalar_register(LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY);

    float next = current;
    if (signed_progress > 0.05f || efficiency_advantage > 0.05f) {
        const float credit = clamp_unit(
                0.65f * std::max(0.0f, signed_progress) +
                0.35f * std::max(0.0f, efficiency_advantage));
        last_validated_progress_monotonic_ms = now_ms;
        last_validated_progress_score = credit;
        next = clamp_unit(current * (1.0f - 0.55f * credit));
    } else {
        const float update_gain = clamp_unit((float) since_update_ms / LLAMA_SELF_EVOLUTION_UPDATE_HORIZON_MS);
        const float growth_gain = clamp_unit((0.20f + 0.80f * drift_pressure) * update_gain);
        next = clamp_unit(current + growth_gain * (target - current));
        if (signed_progress < -0.05f || efficiency_advantage < -0.05f) {
            const float penalty = clamp_unit(
                    0.65f * std::max(0.0f, -signed_progress) +
                    0.35f * std::max(0.0f, -efficiency_advantage));
            next = clamp_unit(std::max(next, current + 0.12f * penalty));
        }
    }

    update_scalar_register(LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY, next, source_mask);
}

void llama_self_state::update_expanded_model(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features,
        uint32_t source_mask) {
    const llama_self_tool_state_info tool_state = build_tool_state_info(tool_jobs);
    const float contradiction = current_scalar_register(LLAMA_SELF_REGISTER_CONTRADICTION);
    const float uncertainty = current_scalar_register(LLAMA_SELF_REGISTER_UNCERTAINTY);
    const float goal_relevance = current_scalar_register(LLAMA_SELF_REGISTER_GOAL_RELEVANCE);
    const float self_relevance = current_scalar_register(LLAMA_SELF_REGISTER_SELF_RELEVANCE);
    const float social_relevance = current_scalar_register(LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE);
    const float affordance = current_scalar_register(LLAMA_SELF_REGISTER_AFFORDANCE);
    const float broadcast_pressure = current_scalar_register(LLAMA_SELF_REGISTER_BROADCAST_PRESSURE);
    const float broadcast_inhibition = current_scalar_register(LLAMA_SELF_REGISTER_BROADCAST_INHIBITION);
    const float continuation = current_scalar_register(LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION);
    const float memory_write_priority = current_scalar_register(LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY);
    const bool tool_failed = (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) != 0;
    const bool tool_completed = (event.flags & LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED) != 0;
    const bool is_tool = event.role == LLAMA_SELF_STATE_EVENT_TOOL;

    llama_self_model_horizon_info instant = {};
    instant.horizon_id = LLAMA_SELF_HORIZON_INSTANT;
    instant.last_update_monotonic_ms = datetime.monotonic_ms;

    instant.goal_progress.goal_progress_estimate = clamp_unit(
            0.38f * goal_relevance +
            0.18f * features.working_memory_top_similarity +
            0.12f * features.memory_handle_top_similarity +
            0.14f * (1.0f - contradiction) +
            0.10f * (1.0f - uncertainty) +
            0.08f * (tool_completed ? 1.0f : 0.0f));
    instant.goal_progress.blocker_severity = clamp_unit(
            0.35f * features.tool_pending_pressure +
            0.25f * contradiction +
            0.20f * uncertainty +
            0.20f * (tool_failed ? 1.0f : 0.0f));
    instant.goal_progress.dependency_readiness = clamp_unit(
            0.45f * tool_state.readiness +
            0.20f * (1.0f - features.tool_pending_pressure) +
            0.20f * features.memory_handle_top_similarity +
            0.15f * features.working_memory_top_similarity);
    instant.goal_progress.urgency = clamp_unit(
            0.38f * goal_relevance +
            0.18f * features.followup_hint +
            0.18f * social_dissatisfaction +
            0.14f * features.recency_user +
            0.12f * continuation);
    instant.goal_progress.expected_next_action_gain = clamp_unit(
            0.30f * (1.0f - uncertainty) +
            0.20f * affordance +
            0.20f * tool_state.readiness +
            0.20f * goal_relevance +
            0.10f * (tool_completed ? 1.0f : 0.0f));
    instant.goal_progress.commitment_slippage_risk = clamp_unit(
            0.45f * features.commitment_top_similarity * std::max(contradiction, uncertainty) +
            0.35f * instant.goal_progress.blocker_severity +
            0.20f * (1.0f - instant.goal_progress.goal_progress_estimate));
    instant.goal_progress.confidence = clamp_unit(
            0.45f * (1.0f - uncertainty) +
            0.20f * features.decoder_top_margin +
            0.20f * (1.0f - features.novelty) +
            0.15f * features.memory_handle_top_similarity);

    instant.user_outcome.satisfaction_estimate = clamp_unit(
            0.28f * social_trust +
            0.18f * social_reciprocity +
            0.16f * (1.0f - social_recent_user_valence) +
            0.14f * (1.0f - contradiction) +
            0.12f * (1.0f - uncertainty) +
            0.12f * instant.goal_progress.goal_progress_estimate);
    instant.user_outcome.frustration_risk = clamp_unit(
            0.40f * social_recent_user_valence +
            0.20f * social_dissatisfaction +
            0.15f * contradiction +
            0.10f * uncertainty +
            0.15f * (tool_failed ? 1.0f : 0.0f));
    instant.user_outcome.misunderstanding_risk = clamp_unit(
            0.35f * uncertainty +
            0.20f * features.question_ratio +
            0.15f * features.topic_shift +
            0.15f * (1.0f - features.working_memory_top_similarity) +
            0.15f * features.uncertainty_lexical_ratio);
    instant.user_outcome.trust_repair_need = clamp_unit(
            0.45f * (1.0f - social_trust) +
            0.30f * instant.user_outcome.frustration_risk +
            0.15f * contradiction +
            0.10f * (tool_failed ? 1.0f : 0.0f));
    instant.user_outcome.preference_uncertainty = clamp_unit(
            0.40f * features.topic_shift +
            0.20f * features.novelty +
            0.20f * instant.user_outcome.misunderstanding_risk +
            0.20f * (1.0f - features.working_memory_top_similarity));
    instant.user_outcome.cognitive_load_estimate = clamp_unit(
            0.25f * features.token_count_log +
            0.25f * instant.user_outcome.misunderstanding_risk +
            0.20f * contradiction +
            0.15f * uncertainty +
            0.15f * features.question_ratio);
    instant.user_outcome.autonomy_tolerance_estimate = clamp_unit(
            0.35f * social_trust +
            0.20f * social_reciprocity +
            0.20f * instant.goal_progress.goal_progress_estimate +
            0.15f * (1.0f - instant.user_outcome.frustration_risk) +
            0.10f * (1.0f - instant.user_outcome.preference_uncertainty));
    instant.user_outcome.confidence = clamp_unit(
            0.40f * (1.0f - instant.user_outcome.preference_uncertainty) +
            0.25f * (1.0f - uncertainty) +
            0.20f * features.working_memory_top_similarity +
            0.15f * features.decoder_top_margin);

    instant.user_preference.directness_preference = clamp_unit(
            0.34f * features.imperative_ratio +
            0.24f * (1.0f - features.uncertainty_lexical_ratio) +
            0.18f * (1.0f - features.question_ratio) +
            0.14f * (1.0f - features.token_count_log) +
            0.10f * social_trust);
    instant.user_preference.verbosity_preference = clamp_unit(
            0.45f * features.token_count_log +
            0.20f * features.followup_hint +
            0.20f * features.question_ratio +
            0.15f * features.newline_ratio);
    instant.user_preference.structure_preference = clamp_unit(
            0.42f * features.list_ratio +
            0.28f * features.newline_ratio +
            0.15f * features.goal_top_similarity +
            0.15f * features.social_familiarity);
    instant.user_preference.clarification_preference = clamp_unit(
            0.36f * features.question_ratio +
            0.24f * features.uncertainty_lexical_ratio +
            0.20f * instant.user_outcome.preference_uncertainty +
            0.20f * instant.user_outcome.misunderstanding_risk);
    instant.user_preference.autonomy_preference = clamp_unit(
            0.45f * instant.user_outcome.autonomy_tolerance_estimate +
            0.20f * social_trust +
            0.15f * features.imperative_ratio +
            0.10f * (1.0f - features.question_ratio) +
            0.10f * (1.0f - instant.user_outcome.frustration_risk));
    instant.user_preference.disagreement_sensitivity = clamp_unit(
            0.34f * features.negation_ratio +
            0.22f * features.negative_user_valence +
            0.20f * instant.user_outcome.trust_repair_need +
            0.14f * features.error_ratio +
            0.10f * (1.0f - social_trust));
    instant.user_preference.rhetorical_intensity = clamp_unit(
            0.34f * features.emphasis_ratio +
            0.20f * features.imperative_ratio +
            0.18f * features.negative_user_valence +
            0.14f * features.newline_ratio +
            0.14f * features.question_ratio);
    instant.user_preference.preference_confidence = clamp_unit(
            0.34f * (1.0f - instant.user_outcome.preference_uncertainty) +
            0.22f * features.working_memory_top_similarity +
            0.22f * social_familiarity +
            0.22f * (1.0f - uncertainty));
    instant.user_preference.rhetorical_confidence = clamp_unit(
            0.30f * features.working_memory_top_similarity +
            0.25f * social_familiarity +
            0.25f * (1.0f - features.topic_shift) +
            0.20f * (1.0f - uncertainty));
    instant.user_preference.simulator_readiness = clamp_unit(
            0.40f * instant.user_preference.preference_confidence +
            0.35f * instant.user_preference.rhetorical_confidence +
            0.25f * social_familiarity);

    instant.epistemic.answerability = clamp_unit(
            0.28f * (1.0f - uncertainty) +
            0.18f * (1.0f - contradiction) +
            0.18f * features.memory_handle_top_similarity +
            0.16f * features.working_memory_top_similarity +
            0.10f * goal_relevance +
            0.10f * tool_state.readiness);
    instant.epistemic.evidence_sufficiency = clamp_unit(
            0.32f * features.memory_handle_top_similarity +
            0.28f * features.working_memory_top_similarity +
            0.20f * goal_relevance +
            0.10f * (1.0f - features.novelty) +
            0.10f * self_relevance);
    instant.epistemic.ambiguity_concentration = clamp_unit(
            0.35f * features.uncertainty_lexical_ratio +
            0.25f * features.decoder_entropy +
            0.20f * features.question_ratio +
            0.20f * features.topic_shift);
    instant.epistemic.self_estimate_confidence = clamp_unit(
            0.35f * (1.0f - uncertainty) +
            0.20f * features.decoder_top_margin +
            0.20f * (1.0f - contradiction) +
            0.15f * (1.0f - features.novelty) +
            0.10f * features.memory_handle_top_similarity);
    instant.epistemic.tool_need_confidence = clamp_unit(
            0.40f * affordance +
            0.25f * instant.goal_progress.blocker_severity +
            0.20f * (1.0f - tool_state.readiness) +
            0.15f * features.tool_pending_pressure);
    instant.epistemic.contradiction_load = contradiction;
    instant.epistemic.uncertainty_load = uncertainty;

    instant.efficiency.expected_steps_remaining = clamp_unit(
            0.45f * (1.0f - instant.goal_progress.goal_progress_estimate) +
            0.20f * instant.goal_progress.blocker_severity +
            0.15f * uncertainty +
            0.10f * continuation +
            0.10f * instant.user_outcome.misunderstanding_risk);
    instant.efficiency.tool_roundtrip_cost = clamp_unit(
            0.45f * features.tool_pending_pressure +
            0.35f * (1.0f - tool_state.readiness) +
            0.20f * (is_tool ? 0.35f : 0.0f));
    instant.efficiency.context_thrash_risk = clamp_unit(
            0.40f * features.topic_shift +
            0.25f * features.novelty +
            0.20f * features.memory_handle_similarity_variance +
            0.15f * uncertainty);
    instant.efficiency.repetition_risk = clamp_unit(
            0.35f * (1.0f - features.novelty) +
            0.30f * features.working_memory_top_similarity +
            0.20f * continuation +
            0.15f * memory_write_priority);
    instant.efficiency.loop_inefficiency = clamp_unit(
            0.30f * continuation +
            0.25f * instant.efficiency.repetition_risk +
            0.20f * instant.efficiency.context_thrash_risk +
            0.15f * uncertainty +
            0.10f * (1.0f - instant.goal_progress.expected_next_action_gain));
    instant.efficiency.expected_inference_cost_remaining = clamp_unit(
            0.45f * instant.efficiency.expected_steps_remaining +
            0.30f * instant.efficiency.tool_roundtrip_cost +
            0.15f * instant.efficiency.loop_inefficiency +
            0.10f * instant.efficiency.context_thrash_risk);
    instant.efficiency.response_compaction_opportunity = clamp_unit(
            0.35f * instant.epistemic.answerability +
            0.25f * instant.epistemic.evidence_sufficiency +
            0.20f * (1.0f - instant.user_outcome.cognitive_load_estimate) +
            0.20f * (1.0f - instant.efficiency.expected_steps_remaining));

    instant.recovery.favorable_divergence_goal = clamp_unit(
            0.55f * (1.0f - instant.goal_progress.goal_progress_estimate) +
            0.25f * instant.goal_progress.blocker_severity +
            0.20f * continuation);
    instant.recovery.favorable_divergence_social = clamp_unit(
            0.45f * instant.user_outcome.frustration_risk +
            0.30f * instant.user_outcome.trust_repair_need +
            0.25f * social_dissatisfaction);
    instant.recovery.favorable_divergence_epistemic = clamp_unit(
            0.40f * uncertainty +
            0.30f * contradiction +
            0.30f * (1.0f - instant.epistemic.answerability));
    instant.recovery.favorable_divergence_action = clamp_unit(
            0.35f * instant.efficiency.loop_inefficiency +
            0.30f * instant.efficiency.tool_roundtrip_cost +
            0.20f * broadcast_pressure +
            0.15f * broadcast_inhibition);
    instant.recovery.regulation_debt = clamp_unit(
            0.30f * broadcast_pressure +
            0.25f * broadcast_inhibition +
            0.25f * continuation +
            0.20f * memory_write_priority);
    instant.recovery.unresolved_tension_load = clamp_unit(
            0.40f * instant.goal_progress.commitment_slippage_risk +
            0.25f * continuation +
            0.20f * contradiction +
            0.15f * instant.user_outcome.trust_repair_need);
    instant.recovery.recovery_cost_estimate = clamp_unit(
            0.35f * max4(
                    instant.recovery.favorable_divergence_goal,
                    instant.recovery.favorable_divergence_social,
                    instant.recovery.favorable_divergence_epistemic,
                    instant.recovery.favorable_divergence_action) +
            0.35f * instant.efficiency.expected_steps_remaining +
            0.30f * instant.efficiency.expected_inference_cost_remaining);

    instant.strategy.answer_bias = clamp_unit(
            0.45f * instant.epistemic.answerability +
            0.20f * instant.goal_progress.expected_next_action_gain +
            0.20f * (1.0f - instant.user_outcome.cognitive_load_estimate) +
            0.15f * (tool_completed ? 1.0f : 0.0f));
    instant.strategy.ask_bias = clamp_unit(
            0.40f * instant.user_outcome.misunderstanding_risk +
            0.30f * uncertainty +
            0.20f * instant.user_outcome.preference_uncertainty +
            0.10f * features.question_ratio);
    instant.strategy.act_bias = clamp_unit(
            0.45f * instant.epistemic.tool_need_confidence +
            0.25f * affordance +
            0.15f * instant.goal_progress.urgency +
            0.15f * tool_state.readiness);
    instant.strategy.wait_bias = clamp_unit(
            0.40f * broadcast_inhibition +
            0.25f * (is_tool && tool_state.running_jobs > 0 ? 1.0f : 0.0f) +
            0.20f * (1.0f - social_relevance) +
            0.15f * (1.0f - goal_relevance));
    instant.strategy.exploit_bias = clamp_unit(
            0.40f * instant.epistemic.evidence_sufficiency +
            0.35f * (1.0f - features.novelty) +
            0.25f * (1.0f - instant.user_outcome.preference_uncertainty));
    instant.strategy.deliberate_bias = clamp_unit(
            0.35f * uncertainty +
            0.25f * contradiction +
            0.20f * instant.goal_progress.blocker_severity +
            0.20f * instant.user_outcome.cognitive_load_estimate);
    instant.strategy.write_internal_bias = clamp_unit(
            0.45f * memory_write_priority +
            0.30f * instant.recovery.regulation_debt +
            0.25f * instant.goal_progress.commitment_slippage_risk);
    instant.strategy.act_external_bias = clamp_unit(
            0.40f * std::max(instant.strategy.answer_bias, instant.strategy.act_bias) +
            0.35f * instant.goal_progress.urgency +
            0.25f * (1.0f - instant.strategy.wait_bias));

    instant.self_improvement.update_worthiness = clamp_unit(
            0.30f * instant.recovery.recovery_cost_estimate +
            0.25f * instant.efficiency.loop_inefficiency +
            0.25f * instant.user_outcome.frustration_risk +
            0.20f * instant.goal_progress.commitment_slippage_risk);
    instant.self_improvement.expected_gain = clamp_unit(
            0.30f * instant.efficiency.expected_inference_cost_remaining +
            0.25f * instant.recovery.recovery_cost_estimate +
            0.25f * instant.goal_progress.blocker_severity +
            0.20f * instant.user_outcome.trust_repair_need);
    instant.self_improvement.evidence_deficit = clamp_unit(
            0.45f * (1.0f - instant.epistemic.self_estimate_confidence) +
            0.30f * instant.user_outcome.preference_uncertainty +
            0.25f * instant.epistemic.ambiguity_concentration);
    instant.self_improvement.blast_radius_risk = clamp_unit(
            0.30f * goal_relevance +
            0.20f * social_relevance +
            0.20f * instant.recovery.regulation_debt +
            0.15f * continuation +
            0.15f * self_relevance);
    instant.self_improvement.observability_deficit = clamp_unit(
            0.45f * instant.user_outcome.preference_uncertainty +
            0.30f * uncertainty +
            0.25f * features.novelty);
    instant.self_improvement.reversibility = clamp_unit(
            0.40f * (1.0f - instant.self_improvement.blast_radius_risk) +
            0.35f * (1.0f - instant.self_improvement.observability_deficit) +
            0.25f * (1.0f - continuation));
    instant.self_improvement.readiness = clamp_unit(
            instant.self_improvement.update_worthiness *
            (1.0f - instant.self_improvement.evidence_deficit) *
            instant.self_improvement.reversibility);

    const std::array<float, 32> context_sketch = build_event_sketch(event);
    extension_domain_signal extension_signal = {};
    accumulate_model_extensions(model_extensions, context_sketch, true, &extension_signal, &extension_summary);
    apply_extension_signal_to_horizon(instant, extension_signal);

    const auto prev_instant = model_horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT];
    const auto prev_short = model_horizons[(size_t) LLAMA_SELF_HORIZON_SHORT];
    const float prev_aggregate_div = max4(
            prev_instant.recovery.favorable_divergence_goal,
            prev_instant.recovery.favorable_divergence_social,
            prev_instant.recovery.favorable_divergence_epistemic,
            prev_instant.recovery.favorable_divergence_action);
    const float current_aggregate_div = max4(
            instant.recovery.favorable_divergence_goal,
            instant.recovery.favorable_divergence_social,
            instant.recovery.favorable_divergence_epistemic,
            instant.recovery.favorable_divergence_action);
    instant.recovery.recovery_momentum = clamp_unit(
            0.50f * clamp_unit(0.5f + 0.5f * (prev_aggregate_div - current_aggregate_div)) +
            0.30f * clamp_unit(0.5f + 0.5f * (prev_short.goal_progress.goal_progress_estimate - prev_instant.goal_progress.goal_progress_estimate + instant.goal_progress.goal_progress_estimate)) +
            0.20f * (1.0f - instant.efficiency.loop_inefficiency));

    if (model_forecast.valid) {
        const float prev_predicted_satisfaction = clamp_range(
                prev_instant.user_outcome.satisfaction_estimate + model_forecast.predicted_satisfaction_delta,
                0.0f, 1.0f);
        const float prev_predicted_recovery = clamp_range(
                prev_aggregate_div - model_forecast.predicted_recovery_delta,
                0.0f, 1.0f);
        const float prev_predicted_goal_progress = clamp_range(
                prev_instant.goal_progress.goal_progress_estimate + model_forecast.predicted_goal_progress_delta,
                0.0f, 1.0f);
        prediction_error.valid = true;
        prediction_error.observed_after_monotonic_ms = datetime.monotonic_ms;
        prediction_error.steps_error = std::fabs(model_forecast.predicted_steps_remaining - instant.efficiency.expected_steps_remaining);
        prediction_error.inference_cost_error = std::fabs(model_forecast.predicted_inference_cost_remaining - instant.efficiency.expected_inference_cost_remaining);
        prediction_error.satisfaction_error = std::fabs(prev_predicted_satisfaction - instant.user_outcome.satisfaction_estimate);
        prediction_error.recovery_error = std::fabs(prev_predicted_recovery - current_aggregate_div);
        prediction_error.goal_progress_error = std::fabs(prev_predicted_goal_progress - instant.goal_progress.goal_progress_estimate);
    }

    model_horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT] = instant;

    auto & short_horizon = model_horizons[(size_t) LLAMA_SELF_HORIZON_SHORT];
    if (short_horizon.last_update_monotonic_ms < 0) {
        short_horizon = instant;
        short_horizon.horizon_id = LLAMA_SELF_HORIZON_SHORT;
    } else {
        const float gain = 0.35f;
        short_horizon.last_update_monotonic_ms = datetime.monotonic_ms;
#define BLEND_FIELD(profile, field) short_horizon.profile.field = blend_value(short_horizon.profile.field, instant.profile.field, gain)
        BLEND_FIELD(goal_progress, goal_progress_estimate);
        BLEND_FIELD(goal_progress, blocker_severity);
        BLEND_FIELD(goal_progress, dependency_readiness);
        BLEND_FIELD(goal_progress, urgency);
        BLEND_FIELD(goal_progress, expected_next_action_gain);
        BLEND_FIELD(goal_progress, commitment_slippage_risk);
        BLEND_FIELD(goal_progress, confidence);
        BLEND_FIELD(user_outcome, satisfaction_estimate);
        BLEND_FIELD(user_outcome, frustration_risk);
        BLEND_FIELD(user_outcome, misunderstanding_risk);
        BLEND_FIELD(user_outcome, trust_repair_need);
        BLEND_FIELD(user_outcome, preference_uncertainty);
        BLEND_FIELD(user_outcome, cognitive_load_estimate);
        BLEND_FIELD(user_outcome, autonomy_tolerance_estimate);
        BLEND_FIELD(user_outcome, confidence);
        BLEND_FIELD(user_preference, directness_preference);
        BLEND_FIELD(user_preference, verbosity_preference);
        BLEND_FIELD(user_preference, structure_preference);
        BLEND_FIELD(user_preference, clarification_preference);
        BLEND_FIELD(user_preference, autonomy_preference);
        BLEND_FIELD(user_preference, disagreement_sensitivity);
        BLEND_FIELD(user_preference, rhetorical_intensity);
        BLEND_FIELD(user_preference, preference_confidence);
        BLEND_FIELD(user_preference, rhetorical_confidence);
        BLEND_FIELD(user_preference, simulator_readiness);
        BLEND_FIELD(epistemic, answerability);
        BLEND_FIELD(epistemic, evidence_sufficiency);
        BLEND_FIELD(epistemic, ambiguity_concentration);
        BLEND_FIELD(epistemic, self_estimate_confidence);
        BLEND_FIELD(epistemic, tool_need_confidence);
        BLEND_FIELD(epistemic, contradiction_load);
        BLEND_FIELD(epistemic, uncertainty_load);
        BLEND_FIELD(efficiency, expected_steps_remaining);
        BLEND_FIELD(efficiency, expected_inference_cost_remaining);
        BLEND_FIELD(efficiency, loop_inefficiency);
        BLEND_FIELD(efficiency, repetition_risk);
        BLEND_FIELD(efficiency, context_thrash_risk);
        BLEND_FIELD(efficiency, tool_roundtrip_cost);
        BLEND_FIELD(efficiency, response_compaction_opportunity);
        BLEND_FIELD(recovery, favorable_divergence_goal);
        BLEND_FIELD(recovery, favorable_divergence_social);
        BLEND_FIELD(recovery, favorable_divergence_epistemic);
        BLEND_FIELD(recovery, favorable_divergence_action);
        BLEND_FIELD(recovery, recovery_momentum);
        BLEND_FIELD(recovery, regulation_debt);
        BLEND_FIELD(recovery, unresolved_tension_load);
        BLEND_FIELD(recovery, recovery_cost_estimate);
        BLEND_FIELD(strategy, answer_bias);
        BLEND_FIELD(strategy, ask_bias);
        BLEND_FIELD(strategy, act_bias);
        BLEND_FIELD(strategy, wait_bias);
        BLEND_FIELD(strategy, exploit_bias);
        BLEND_FIELD(strategy, deliberate_bias);
        BLEND_FIELD(strategy, write_internal_bias);
        BLEND_FIELD(strategy, act_external_bias);
        BLEND_FIELD(self_improvement, update_worthiness);
        BLEND_FIELD(self_improvement, expected_gain);
        BLEND_FIELD(self_improvement, evidence_deficit);
        BLEND_FIELD(self_improvement, reversibility);
        BLEND_FIELD(self_improvement, blast_radius_risk);
        BLEND_FIELD(self_improvement, observability_deficit);
        BLEND_FIELD(self_improvement, readiness);
#undef BLEND_FIELD
    }

    auto & long_horizon = model_horizons[(size_t) LLAMA_SELF_HORIZON_LONG];
    if (long_horizon.last_update_monotonic_ms < 0) {
        long_horizon = instant;
        long_horizon.horizon_id = LLAMA_SELF_HORIZON_LONG;
    } else {
        const float gain = 0.12f;
        long_horizon.last_update_monotonic_ms = datetime.monotonic_ms;
#define BLEND_FIELD(profile, field) long_horizon.profile.field = blend_value(long_horizon.profile.field, instant.profile.field, gain)
        BLEND_FIELD(goal_progress, goal_progress_estimate);
        BLEND_FIELD(goal_progress, blocker_severity);
        BLEND_FIELD(goal_progress, dependency_readiness);
        BLEND_FIELD(goal_progress, urgency);
        BLEND_FIELD(goal_progress, expected_next_action_gain);
        BLEND_FIELD(goal_progress, commitment_slippage_risk);
        BLEND_FIELD(goal_progress, confidence);
        BLEND_FIELD(user_outcome, satisfaction_estimate);
        BLEND_FIELD(user_outcome, frustration_risk);
        BLEND_FIELD(user_outcome, misunderstanding_risk);
        BLEND_FIELD(user_outcome, trust_repair_need);
        BLEND_FIELD(user_outcome, preference_uncertainty);
        BLEND_FIELD(user_outcome, cognitive_load_estimate);
        BLEND_FIELD(user_outcome, autonomy_tolerance_estimate);
        BLEND_FIELD(user_outcome, confidence);
        BLEND_FIELD(user_preference, directness_preference);
        BLEND_FIELD(user_preference, verbosity_preference);
        BLEND_FIELD(user_preference, structure_preference);
        BLEND_FIELD(user_preference, clarification_preference);
        BLEND_FIELD(user_preference, autonomy_preference);
        BLEND_FIELD(user_preference, disagreement_sensitivity);
        BLEND_FIELD(user_preference, rhetorical_intensity);
        BLEND_FIELD(user_preference, preference_confidence);
        BLEND_FIELD(user_preference, rhetorical_confidence);
        BLEND_FIELD(user_preference, simulator_readiness);
        BLEND_FIELD(epistemic, answerability);
        BLEND_FIELD(epistemic, evidence_sufficiency);
        BLEND_FIELD(epistemic, ambiguity_concentration);
        BLEND_FIELD(epistemic, self_estimate_confidence);
        BLEND_FIELD(epistemic, tool_need_confidence);
        BLEND_FIELD(epistemic, contradiction_load);
        BLEND_FIELD(epistemic, uncertainty_load);
        BLEND_FIELD(efficiency, expected_steps_remaining);
        BLEND_FIELD(efficiency, expected_inference_cost_remaining);
        BLEND_FIELD(efficiency, loop_inefficiency);
        BLEND_FIELD(efficiency, repetition_risk);
        BLEND_FIELD(efficiency, context_thrash_risk);
        BLEND_FIELD(efficiency, tool_roundtrip_cost);
        BLEND_FIELD(efficiency, response_compaction_opportunity);
        BLEND_FIELD(recovery, favorable_divergence_goal);
        BLEND_FIELD(recovery, favorable_divergence_social);
        BLEND_FIELD(recovery, favorable_divergence_epistemic);
        BLEND_FIELD(recovery, favorable_divergence_action);
        BLEND_FIELD(recovery, recovery_momentum);
        BLEND_FIELD(recovery, regulation_debt);
        BLEND_FIELD(recovery, unresolved_tension_load);
        BLEND_FIELD(recovery, recovery_cost_estimate);
        BLEND_FIELD(strategy, answer_bias);
        BLEND_FIELD(strategy, ask_bias);
        BLEND_FIELD(strategy, act_bias);
        BLEND_FIELD(strategy, wait_bias);
        BLEND_FIELD(strategy, exploit_bias);
        BLEND_FIELD(strategy, deliberate_bias);
        BLEND_FIELD(strategy, write_internal_bias);
        BLEND_FIELD(strategy, act_external_bias);
        BLEND_FIELD(self_improvement, update_worthiness);
        BLEND_FIELD(self_improvement, expected_gain);
        BLEND_FIELD(self_improvement, evidence_deficit);
        BLEND_FIELD(self_improvement, reversibility);
        BLEND_FIELD(self_improvement, blast_radius_risk);
        BLEND_FIELD(self_improvement, observability_deficit);
        BLEND_FIELD(self_improvement, readiness);
#undef BLEND_FIELD
    }

    model_forecast.valid = true;
    model_forecast.issued_monotonic_ms = datetime.monotonic_ms;
    model_forecast.predicted_steps_remaining = clamp_unit(
            0.60f * instant.efficiency.expected_steps_remaining +
            0.25f * short_horizon.efficiency.expected_steps_remaining +
            0.15f * long_horizon.efficiency.expected_steps_remaining);
    model_forecast.predicted_inference_cost_remaining = clamp_unit(
            0.60f * instant.efficiency.expected_inference_cost_remaining +
            0.25f * short_horizon.efficiency.expected_inference_cost_remaining +
            0.15f * long_horizon.efficiency.expected_inference_cost_remaining);
    model_forecast.predicted_satisfaction_delta = clamp_range(
            0.35f * instant.goal_progress.expected_next_action_gain +
            0.25f * instant.epistemic.answerability +
            0.20f * instant.efficiency.response_compaction_opportunity -
            0.30f * instant.user_outcome.frustration_risk -
            0.20f * instant.user_outcome.misunderstanding_risk,
            -1.0f, 1.0f);
    model_forecast.predicted_recovery_delta = clamp_range(
            0.45f * instant.recovery.recovery_momentum +
            0.20f * instant.goal_progress.expected_next_action_gain -
            0.25f * instant.recovery.regulation_debt -
            0.20f * instant.efficiency.loop_inefficiency,
            -1.0f, 1.0f);
    model_forecast.predicted_goal_progress_delta = clamp_range(
            0.50f * instant.goal_progress.expected_next_action_gain +
            0.20f * instant.epistemic.answerability -
            0.20f * instant.goal_progress.blocker_severity -
            0.10f * instant.user_outcome.misunderstanding_risk,
            -1.0f, 1.0f);
    user_preference = instant.user_preference;
    model_forecast.confidence = clamp_unit(
            0.40f * instant.epistemic.self_estimate_confidence +
            0.25f * instant.goal_progress.confidence +
            0.20f * instant.user_outcome.confidence +
            0.15f * (1.0f - instant.self_improvement.evidence_deficit));

    update_summary_registers(source_mask);
    update_belief_state(event, features, source_mask);
}

bool llama_context::self_state_refresh_time() {
    return self_state && self_state->refresh_time();
}

bool llama_context::self_state_set_time(const llama_self_state_time_point & time_point) {
    return self_state && self_state->set_time(time_point);
}

bool llama_context::self_state_get_datetime(llama_self_state_datetime * out_info) const {
    return self_state && self_state->get_datetime(out_info);
}

bool llama_context::self_state_configure(const llama_self_state_params & params) {
    return self_state && self_state->configure(params);
}

int32_t llama_context::self_state_register_count() const {
    return self_state ? self_state->register_count() : 0;
}

bool llama_context::self_state_get_register(int32_t register_id, llama_self_register_info * out_info) const {
    return self_state && self_state->get_register(register_id, out_info);
}

bool llama_context::self_state_set_channel_state(int32_t next_channel_state) {
    return self_state && self_state->set_channel_state(next_channel_state);
}

bool llama_context::self_state_note_user_event() {
    return self_state && self_state->note_user_event();
}

bool llama_context::self_state_note_tool_event() {
    return self_state && self_state->note_tool_event();
}

bool llama_context::self_state_note_emit_event() {
    return self_state && self_state->note_emit_event();
}

bool llama_context::self_state_set_identity(const llama_token * tokens, size_t n_tokens) {
    return self_state && self_state->set_identity(tokens, n_tokens);
}

bool llama_context::self_state_upsert_goal(
        int32_t goal_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return self_state && self_state->upsert_goal(goal_id, tokens, n_tokens, priority);
}

bool llama_context::self_state_upsert_commitment(
        int32_t commitment_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority,
        bool unresolved) {
    return self_state && self_state->upsert_commitment(commitment_id, tokens, n_tokens, priority, unresolved);
}

int32_t llama_context::self_state_goal_count() const {
    return self_state ? self_state->goal_count() : 0;
}

int32_t llama_context::self_state_commitment_count() const {
    return self_state ? self_state->commitment_count() : 0;
}

int32_t llama_context::self_state_working_memory_count() const {
    return self_state ? self_state->working_memory_count() : 0;
}

bool llama_context::self_state_upsert_memory_handle(
        int32_t handle_id,
        int32_t kind,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return self_state && self_state->upsert_memory_handle(handle_id, kind, tokens, n_tokens, priority);
}

int32_t llama_context::self_state_memory_handle_count() const {
    return self_state ? self_state->memory_handle_count() : 0;
}

int32_t llama_context::self_state_reactivation_count() const {
    return self_state ? self_state->reactivation_count() : 0;
}

bool llama_context::self_state_get_reactivation(int32_t index, llama_self_reactivation_info * out_info) const {
    return self_state && self_state->get_reactivation(index, out_info);
}

bool llama_context::self_state_upsert_tool_job(int32_t job_id, int32_t status, float importance) {
    return self_state && self_state->upsert_tool_job(job_id, status, importance);
}

bool llama_context::self_state_get_tool_state(llama_self_tool_state_info * out_info) const {
    return self_state && self_state->get_tool_state(out_info);
}

bool llama_context::self_state_get_social_state(llama_self_social_state_info * out_info) const {
    return self_state && self_state->get_social_state(out_info);
}

bool llama_context::self_state_get_model_state(llama_self_model_state_info * out_info) const {
    return self_state && self_state->get_model_state(out_info);
}

int32_t llama_context::self_state_model_extension_count() const {
    return self_state ? self_state->model_extension_count() : 0;
}

bool llama_context::self_state_get_model_extension(int32_t index, llama_self_model_extension_info * out_info) const {
    return self_state && self_state->get_model_extension(index, out_info);
}

bool llama_context::self_state_upsert_model_extension(const llama_self_model_extension_update & update) {
    return self_state && self_state->upsert_model_extension(update);
}

bool llama_context::self_state_remove_model_extension(const char * key) {
    return self_state && self_state->remove_model_extension(key);
}

int32_t llama_context::self_state_trace_count() const {
    return self_state ? self_state->trace_count() : 0;
}

bool llama_context::self_state_clear_trace() {
    return self_state && self_state->clear_trace();
}

bool llama_context::self_state_replay_trace(int32_t upto_count) {
    return self_state && self_state->replay_trace(&get_model().vocab, upto_count, -1);
}

bool llama_context::self_state_replay_trace_on_channel(int32_t upto_count, int32_t replay_channel) {
    return self_state && self_state->replay_trace(&get_model().vocab, upto_count, replay_channel);
}

bool llama_context::self_state_set_updater_program(const llama_self_updater_program & program) {
    return self_state && self_state->set_updater_program(program);
}

bool llama_context::self_state_get_updater_program(llama_self_updater_program * out_program) const {
    return self_state && self_state->get_updater_program(out_program);
}

size_t llama_context::self_state_trace_export_size() const {
    return self_state ? self_state->trace_export_size() : 0;
}

bool llama_context::self_state_trace_export(void * dst, size_t size) const {
    return self_state && self_state->trace_export(dst, size);
}

bool llama_context::self_state_trace_import(const void * src, size_t size, bool replace_existing) {
    return self_state && self_state->trace_import(src, size, replace_existing);
}

bool llama_context::self_state_evaluate_counterfactual(
        const llama_self_updater_program & program,
        int32_t upto_count,
        llama_self_counterfactual_result * out_result) const {
    return self_state && self_state->evaluate_counterfactual(
            &get_model().vocab,
            program,
            upto_count,
            LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
            out_result);
}

bool llama_context::self_state_evaluate_counterfactual_on_channel(
        const llama_self_updater_program & program,
        int32_t upto_count,
        int32_t replay_channel,
        llama_self_counterfactual_result * out_result) const {
    return self_state && self_state->evaluate_counterfactual(&get_model().vocab, program, upto_count, replay_channel, out_result);
}

bool llama_context::self_state_evaluate_hypothetical_event(
        const llama_self_state_event & event,
        llama_self_state_delta_summary * out_delta,
        llama_self_model_state_info * out_model_state) const {
    return self_state && self_state->evaluate_hypothetical_event(&get_model().vocab, event, out_delta, out_model_state);
}

bool llama_context::self_state_build_prewrite_features(
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return self_state && self_state->build_prewrite_features(&get_model().vocab, event, out_features);
}

bool llama_context::self_state_apply_prewrite(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) {
    return self_state && self_state->apply_prewrite(event, features);
}

bool llama_context::self_state_build_postwrite_features(
        const llama_self_state_event & event,
        llama_self_state_feature_vector * out_features) const {
    return self_state && self_state->build_postwrite_features(&get_model().vocab, event, out_features);
}

bool llama_context::self_state_apply_postwrite(
        const llama_self_state_event & event,
        const llama_self_state_feature_vector & features) {
    if (!self_state) {
        return false;
    }

    const std::vector<float> before = capture_self_register_scalars(*this);
    if (!self_state->apply_postwrite(event, features)) {
        return false;
    }

    if (!hard_memory) {
        return true;
    }

    llama_hard_memory_config config = {};
    if (!hard_memory->get_config(&config) || !config.enabled || !config.archive_enabled) {
        return true;
    }
    if (event.channel == LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL && !config.archive_counterfactual_events) {
        return true;
    }
    if (!event.tokens || event.n_tokens == 0) {
        return true;
    }

    const std::vector<float> after = capture_self_register_scalars(*this);
    const llama_self_state_delta_summary delta = summarize_self_state_delta(before, after, event);
    if (delta.total_delta < config.archival_delta_threshold) {
        return true;
    }

    llama_hard_memory_primitive primitives[LLAMA_HARD_MEMORY_MAX_PRIMITIVES] = {};
    const int32_t primitive_count = build_postwrite_hard_memory_primitives(*this, event, delta, primitives);
    if (primitive_count > 0) {
        (void) hard_memory->archive_primitives(primitives, primitive_count, &delta);
    } else {
        (void) hard_memory->archive_event(&get_model().vocab, event, delta);
    }
    return true;
}

bool llama_context::self_state_note_validated_progress(float signed_progress, float efficiency_advantage) {
    return self_state && self_state->note_validated_progress(signed_progress, efficiency_advantage);
}

bool llama_context::hard_memory_configure(const llama_hard_memory_config & config) {
    return hard_memory && hard_memory->configure(config);
}

bool llama_context::hard_memory_get_config(llama_hard_memory_config * out_config) const {
    return hard_memory && hard_memory->get_config(out_config);
}

bool llama_context::hard_memory_query(const llama_hard_memory_query_request & query, llama_hard_memory_result * out_result) {
    if (!hard_memory) {
        return false;
    }
    const bool ok = hard_memory->query(query, out_result);
    if (ok && out_result && self_state) {
        (void) self_state->promote_hard_memory_query(query, *out_result);
    }
    return ok;
}

bool llama_context::hard_memory_archive_primitives(
        const llama_hard_memory_primitive * primitives,
        int32_t primitive_count,
        const llama_self_state_delta_summary * delta_summary) {
    return hard_memory && hard_memory->archive_primitives(primitives, primitive_count, delta_summary);
}

bool llama_context::hard_memory_get_last_result(llama_hard_memory_result * out_result) const {
    return hard_memory && hard_memory->get_last_result(out_result);
}

bool llama_context::hard_memory_get_last_archive_trace(llama_hard_memory_archive_trace * out_trace) const {
    return hard_memory && hard_memory->get_last_archive_trace(out_trace);
}

llama_self_model_extension_update llama_self_model_extension_default_update(void) {
    llama_self_model_extension_update update = {};
    update.source = LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_EXTERNAL;
    update.source_tool_kind = LLAMA_TOOL_KIND_NONE;
    update.kind = LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT;
    update.domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EPISTEMIC;
    update.flags = LLAMA_SELF_MODEL_EXTENSION_FLAG_ACTIVE | LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN;
    update.value = 1.0f;
    update.desired_value = 0.0f;
    update.confidence = 0.5f;
    update.salience = 0.5f;
    update.gain_weight = 1.0f;
    update.allostatic_weight = 0.0f;
    return update;
}

llama_self_state_params llama_self_state_default_params(void) {
    return {
        /*.enable_learned_contradiction_head =*/ false,
        /*.enable_learned_uncertainty_head =*/ false,
        /*.enable_learned_broadcast_head =*/ false,
        /*.enable_builtin_contradiction_probe =*/ true,
        /*.enable_builtin_uncertainty_probe =*/ true,
        /*.enable_builtin_broadcast_probe =*/ true,
        /*.tool_salience_half_life_ms =*/ LLAMA_SELF_DEFAULT_TOOL_SALIENCE_HALF_LIFE_MS,
        /*.prewrite_gain =*/ 0.65f,
        /*.postwrite_gain =*/ 0.55f,
        /*.enable_belief_state =*/ true,
        /*.belief_slot_count =*/ 3,
        /*.belief_residual_decay =*/ 0.18f,
        /*.belief_pressure_clip =*/ 1.0f,
        /*.belief_confidence_floor =*/ 0.05f,
        /*.belief_promotion_threshold =*/ 0.55f,
        /*.belief_max_update_step =*/ 0.35f,
        /*.belief_missing_observation_weight =*/ 0.35f,
        /*.belief_unmodeled_care_weight =*/ 0.55f,
        /*.belief_forecast_error_weight =*/ 0.45f,
        /*.belief_counterfactual_miss_weight =*/ 0.25f,
        /*.belief_memory_residue_weight =*/ 0.30f,
        /*.contradiction_head_callback =*/ nullptr,
        /*.contradiction_head_user_data =*/ nullptr,
        /*.uncertainty_head_callback =*/ nullptr,
        /*.uncertainty_head_user_data =*/ nullptr,
        /*.broadcast_head_callback =*/ nullptr,
        /*.broadcast_head_user_data =*/ nullptr,
    };
}

llama_self_updater_program llama_self_state_default_updater_program(void) {
    llama_self_updater_program program = {};
    program.version = 1;
    program.memory_novelty_weight = 0.26f;
    program.memory_working_similarity_weight = 0.14f;
    program.memory_handle_similarity_weight = 0.10f;
    program.memory_uncertainty_weight = 0.25f;
    program.memory_contradiction_weight = 0.20f;
    program.memory_handle_variance_weight = 0.05f;
    program.broadcast_social_weight = 0.24f;
    program.broadcast_contradiction_weight = 0.18f;
    program.broadcast_uncertainty_weight = 0.18f;
    program.broadcast_tool_pending_weight = 0.14f;
    program.broadcast_tool_unready_weight = 0.08f;
    program.broadcast_failure_weight = 0.15f;
    program.broadcast_question_weight = 0.08f;
    program.broadcast_goal_weight = 0.13f;
    program.repair_emit_threshold = 0.72f;
    program.repair_dissatisfaction_floor = 0.38f;
    program.repair_recent_user_valence_floor = 0.20f;
    program.repair_inhibition_max = 0.68f;
    program.repair_admission_floor = 0.34f;
    program.repair_admission_weight = 0.22f;

    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_UNCERTAINTY,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.08f, 0.58f, 0.42f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.72f},
                {LLAMA_SELF_UPDATER_FEATURE_DECODER_ENTROPY, 0.18f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_FAILED, 0.08f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_CONTRADICTION,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.60f, 0.40f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION, 0.74f},
                {LLAMA_SELF_UPDATER_FEATURE_NEGATION_RATIO, 0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_ERROR_RATIO, 0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_FAILED, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_NOVELTY,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.04f, 0.72f, 0.46f, 0.04f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_NOVELTY, 0.82f},
                {LLAMA_SELF_UPDATER_FEATURE_TOPIC_SHIFT, 0.10f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_TOPIC_SHIFT,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.04f, 0.70f, 0.44f, 0.04f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_TOPIC_SHIFT, 0.86f},
                {LLAMA_SELF_UPDATER_FEATURE_NOVELTY, 0.08f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_GOAL_RELEVANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.62f, 0.36f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.78f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_USER, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_SELF_RELEVANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.04f, 0.60f, 0.34f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_SELF_REFERENCE, 0.34f},
                {LLAMA_SELF_UPDATER_FEATURE_IDENTITY_SIMILARITY, 0.34f},
                {LLAMA_SELF_UPDATER_FEATURE_COMMITMENT_SIMILARITY, 0.22f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.08f, 0.58f, 0.32f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_USER, 0.24f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND, 0.42f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_RECIPROCITY, 0.14f},
                {LLAMA_SELF_UPDATER_FEATURE_RECENCY_USER, 0.10f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_AFFORDANCE,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.62f, 0.38f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_ERROR_RATIO, 0.34f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.26f},
                {LLAMA_SELF_UPDATER_FEATURE_TOOL_PENDING_PRESSURE, 0.14f},
                {LLAMA_SELF_UPDATER_FEATURE_TOOL_READINESS, -0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_SELF_REFERENCE, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY,
            LLAMA_SELF_UPDATER_PHASE_PREWRITE,
            0.05f, 0.64f, 0.40f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_MEMORY_WRITE_PRESSURE, 0.82f},
                {LLAMA_SELF_UPDATER_FEATURE_NOVELTY, 0.06f},
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.04f},
                {LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION, 0.04f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_BROADCAST_PRESSURE,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.03f, 0.58f, 0.34f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_BROADCAST_PRESSURE_HINT, 0.74f},
                {LLAMA_SELF_UPDATER_FEATURE_FOLLOWUP_HINT, 0.10f},
                {LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.06f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_BROADCAST_INHIBITION,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.06f, 0.60f, 0.36f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_BROADCAST_INHIBITION_HINT, 0.82f},
                {LLAMA_SELF_UPDATER_FEATURE_TOOL_PENDING_PRESSURE, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.06f},
            },
            {
                {LLAMA_SELF_REGISTER_BROADCAST_PRESSURE, -0.08f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.04f, 0.62f, 0.38f, 0.03f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_FOLLOWUP_HINT, 0.78f},
                {LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY, 0.04f},
            });
    program.rules[program.rule_count++] = make_rule(
            LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY,
            LLAMA_SELF_UPDATER_PHASE_POSTWRITE,
            0.05f, 0.60f, 0.38f, 0.02f,
            {
                {LLAMA_SELF_UPDATER_FEATURE_MEMORY_WRITE_PRESSURE, 0.78f},
                {LLAMA_SELF_UPDATER_FEATURE_EVENT_ADMITTED, 0.08f},
                {LLAMA_SELF_UPDATER_FEATURE_BROADCAST_PRESSURE_HINT, 0.04f},
            });

    return program;
}

int32_t llama_self_state_refresh_time(struct llama_context * ctx) {
    return ctx && ctx->self_state_refresh_time() ? 0 : -1;
}

int32_t llama_self_state_set_time(
        struct llama_context * ctx,
        struct llama_self_state_time_point time_point) {
    return ctx && ctx->self_state_set_time(time_point) ? 0 : -1;
}

int32_t llama_self_state_get_datetime(
        const struct llama_context * ctx,
        struct llama_self_state_datetime * out_info) {
    return ctx && ctx->self_state_get_datetime(out_info) ? 0 : -1;
}

int32_t llama_self_state_register_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_register_count() : -1;
}

int32_t llama_self_state_get_register(
        const struct llama_context * ctx,
        int32_t register_id,
        struct llama_self_register_info * out_info) {
    return ctx && ctx->self_state_get_register(register_id, out_info) ? 0 : -1;
}

const char * llama_self_state_register_name(int32_t register_id) {
    return llama_self_state::register_name(register_id);
}

int32_t llama_self_state_configure(
        struct llama_context * ctx,
        struct llama_self_state_params params) {
    return ctx && ctx->self_state_configure(params) ? 0 : -1;
}

int32_t llama_self_state_set_channel_state(
        struct llama_context * ctx,
        int32_t channel_state) {
    return ctx && ctx->self_state_set_channel_state(channel_state) ? 0 : -1;
}

int32_t llama_self_state_note_user_event(struct llama_context * ctx) {
    return ctx && ctx->self_state_note_user_event() ? 0 : -1;
}

int32_t llama_self_state_note_tool_event(struct llama_context * ctx) {
    return ctx && ctx->self_state_note_tool_event() ? 0 : -1;
}

int32_t llama_self_state_note_emit_event(struct llama_context * ctx) {
    return ctx && ctx->self_state_note_emit_event() ? 0 : -1;
}

int32_t llama_self_state_set_identity(
        struct llama_context * ctx,
        const llama_token * tokens,
        size_t n_tokens) {
    return ctx && ctx->self_state_set_identity(tokens, n_tokens) ? 0 : -1;
}

int32_t llama_self_state_upsert_goal(
        struct llama_context * ctx,
        int32_t goal_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return ctx && ctx->self_state_upsert_goal(goal_id, tokens, n_tokens, priority) ? 0 : -1;
}

int32_t llama_self_state_upsert_commitment(
        struct llama_context * ctx,
        int32_t commitment_id,
        const llama_token * tokens,
        size_t n_tokens,
        float priority,
        bool unresolved) {
    return ctx && ctx->self_state_upsert_commitment(commitment_id, tokens, n_tokens, priority, unresolved) ? 0 : -1;
}

int32_t llama_self_state_goal_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_goal_count() : -1;
}

int32_t llama_self_state_commitment_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_commitment_count() : -1;
}

int32_t llama_self_state_working_memory_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_working_memory_count() : -1;
}

int32_t llama_self_state_upsert_memory_handle(
        struct llama_context * ctx,
        int32_t handle_id,
        int32_t kind,
        const llama_token * tokens,
        size_t n_tokens,
        float priority) {
    return ctx && ctx->self_state_upsert_memory_handle(handle_id, kind, tokens, n_tokens, priority) ? 0 : -1;
}

int32_t llama_self_state_memory_handle_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_memory_handle_count() : -1;
}

int32_t llama_self_state_reactivation_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_reactivation_count() : -1;
}

int32_t llama_self_state_get_reactivation(
        const struct llama_context * ctx,
        int32_t index,
        struct llama_self_reactivation_info * out_info) {
    return ctx && ctx->self_state_get_reactivation(index, out_info) ? 0 : -1;
}

int32_t llama_self_state_upsert_tool_job(
        struct llama_context * ctx,
        int32_t job_id,
        int32_t status,
        float importance) {
    return ctx && ctx->self_state_upsert_tool_job(job_id, status, importance) ? 0 : -1;
}

int32_t llama_self_state_get_tool_state(
        const struct llama_context * ctx,
        struct llama_self_tool_state_info * out_info) {
    return ctx && ctx->self_state_get_tool_state(out_info) ? 0 : -1;
}

int32_t llama_self_state_get_social_state(
        const struct llama_context * ctx,
        struct llama_self_social_state_info * out_info) {
    return ctx && ctx->self_state_get_social_state(out_info) ? 0 : -1;
}

int32_t llama_self_state_get_model_state(
        const struct llama_context * ctx,
        struct llama_self_model_state_info * out_info) {
    return ctx && ctx->self_state_get_model_state(out_info) ? 0 : -1;
}

int32_t llama_self_state_model_extension_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_model_extension_count() : -1;
}

int32_t llama_self_state_get_model_extension(
        const struct llama_context * ctx,
        int32_t index,
        struct llama_self_model_extension_info * out_info) {
    return ctx && out_info && ctx->self_state_get_model_extension(index, out_info) ? 0 : -1;
}

int32_t llama_self_state_upsert_model_extension(
        struct llama_context * ctx,
        struct llama_self_model_extension_update update) {
    return ctx && ctx->self_state_upsert_model_extension(update) ? 0 : -1;
}

int32_t llama_self_state_remove_model_extension(
        struct llama_context * ctx,
        const char * key) {
    return ctx && key && ctx->self_state_remove_model_extension(key) ? 0 : -1;
}

int32_t llama_self_state_trace_count(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_trace_count() : -1;
}

int32_t llama_self_state_clear_trace(struct llama_context * ctx) {
    return ctx && ctx->self_state_clear_trace() ? 0 : -1;
}

int32_t llama_self_state_replay_trace(struct llama_context * ctx, int32_t upto_count) {
    return ctx && ctx->self_state_replay_trace(upto_count) ? 0 : -1;
}

int32_t llama_self_state_replay_trace_on_channel(
        struct llama_context * ctx,
        int32_t upto_count,
        int32_t replay_channel) {
    return ctx && ctx->self_state_replay_trace_on_channel(upto_count, replay_channel) ? 0 : -1;
}

int32_t llama_self_state_set_updater_program(
        struct llama_context * ctx,
        struct llama_self_updater_program program) {
    return ctx && ctx->self_state_set_updater_program(program) ? 0 : -1;
}

int32_t llama_self_state_get_updater_program(
        const struct llama_context * ctx,
        struct llama_self_updater_program * out_program) {
    return ctx && ctx->self_state_get_updater_program(out_program) ? 0 : -1;
}

size_t llama_self_state_trace_export_size(const struct llama_context * ctx) {
    return ctx ? ctx->self_state_trace_export_size() : 0;
}

int32_t llama_self_state_trace_export(
        const struct llama_context * ctx,
        void * dst,
        size_t size) {
    return ctx && ctx->self_state_trace_export(dst, size) ? 0 : -1;
}

int32_t llama_self_state_trace_import(
        struct llama_context * ctx,
        const void * src,
        size_t size,
        bool replace_existing) {
    return ctx && ctx->self_state_trace_import(src, size, replace_existing) ? 0 : -1;
}

int32_t llama_self_state_evaluate_counterfactual(
        const struct llama_context * ctx,
        struct llama_self_updater_program program,
        int32_t upto_count,
        struct llama_self_counterfactual_result * out_result) {
    return ctx && ctx->self_state_evaluate_counterfactual(program, upto_count, out_result) ? 0 : -1;
}

int32_t llama_self_state_evaluate_counterfactual_on_channel(
        const struct llama_context * ctx,
        struct llama_self_updater_program program,
        int32_t upto_count,
        int32_t replay_channel,
        struct llama_self_counterfactual_result * out_result) {
    return ctx && ctx->self_state_evaluate_counterfactual_on_channel(program, upto_count, replay_channel, out_result) ? 0 : -1;
}

int32_t llama_self_state_build_prewrite_features(
        const struct llama_context * ctx,
        const struct llama_self_state_event * event,
        struct llama_self_state_feature_vector * out_features) {
    return ctx && event && ctx->self_state_build_prewrite_features(*event, out_features) ? 0 : -1;
}

int32_t llama_self_state_apply_prewrite(
        struct llama_context * ctx,
        const struct llama_self_state_event * event,
        const struct llama_self_state_feature_vector * features) {
    return ctx && event && features && ctx->self_state_apply_prewrite(*event, *features) ? 0 : -1;
}

int32_t llama_self_state_build_postwrite_features(
        const struct llama_context * ctx,
        const struct llama_self_state_event * event,
        struct llama_self_state_feature_vector * out_features) {
    return ctx && event && ctx->self_state_build_postwrite_features(*event, out_features) ? 0 : -1;
}

int32_t llama_self_state_apply_postwrite(
        struct llama_context * ctx,
        const struct llama_self_state_event * event,
        const struct llama_self_state_feature_vector * features) {
    return ctx && event && features && ctx->self_state_apply_postwrite(*event, *features) ? 0 : -1;
}

int32_t llama_hard_memory_configure(
        struct llama_context * ctx,
        struct llama_hard_memory_config config) {
    return ctx && ctx->hard_memory_configure(config) ? 0 : -1;
}

int32_t llama_hard_memory_get_config(
        const struct llama_context * ctx,
        struct llama_hard_memory_config * out_config) {
    return ctx && ctx->hard_memory_get_config(out_config) ? 0 : -1;
}

int32_t llama_hard_memory_query(
        struct llama_context * ctx,
        const struct llama_hard_memory_query_request * query,
        struct llama_hard_memory_result * out_result) {
    return ctx && query && out_result && ctx->hard_memory_query(*query, out_result) ? 0 : -1;
}

int32_t llama_hard_memory_archive_primitives(
        struct llama_context * ctx,
        const struct llama_hard_memory_primitive * primitives,
        int32_t primitive_count) {
    return ctx && primitives && primitive_count > 0 &&
            ctx->hard_memory_archive_primitives(primitives, primitive_count) ? 0 : -1;
}

int32_t llama_hard_memory_get_last_result(
        const struct llama_context * ctx,
        struct llama_hard_memory_result * out_result) {
    return ctx && out_result && ctx->hard_memory_get_last_result(out_result) ? 0 : -1;
}

int32_t llama_hard_memory_get_last_archive_trace(
        const struct llama_context * ctx,
        struct llama_hard_memory_archive_trace * out_trace) {
    return ctx && out_trace && ctx->hard_memory_get_last_archive_trace(out_trace) ? 0 : -1;
}
