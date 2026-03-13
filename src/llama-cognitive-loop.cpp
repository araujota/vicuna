#include "llama-cognitive-loop.h"

#include "llama-context.h"
#include "llama-vocab.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

static constexpr float LLAMA_DMN_PRESSURE_THRESHOLD = 0.24f;

static float clamp_unit(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

static std::string normalize_piece(const std::string & piece) {
    std::string out;
    out.reserve(piece.size());

    for (char ch : piece) {
        const unsigned char uch = (unsigned char) ch;
        if (std::isalnum(uch) || ch == '?' || ch == '\'' || ch == '_') {
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

struct lexical_signals {
    float question = 0.0f;
    float tool = 0.0f;
    float followup = 0.0f;
    float uncertainty = 0.0f;
};

static lexical_signals analyze_event_lexicon(const llama_vocab * vocab, const llama_self_state_event & event) {
    lexical_signals out = {};
    if (!vocab || !event.tokens) {
        return out;
    }

    static const char * const question_terms[] = { "?", "what", "which", "clarify", "why", "how" };
    static const char * const tool_terms[] = { "search", "find", "lookup", "look", "inspect", "tool", "call", "run", "fetch" };
    static const char * const followup_terms[] = { "follow", "continue", "again", "also", "more", "next" };
    static const char * const uncertainty_terms[] = { "?", "maybe", "unsure", "uncertain", "unknown", "clarify" };

    for (size_t i = 0; i < event.n_tokens; ++i) {
        const llama_token token = event.tokens[i];
        if (token < 0) {
            continue;
        }
        const std::string piece = normalize_piece(vocab->token_to_piece(token));
        if (piece.empty()) {
            continue;
        }

        if (contains_any(piece, question_terms, sizeof(question_terms)/sizeof(question_terms[0]))) {
            out.question += 0.35f;
        }
        if (contains_any(piece, tool_terms, sizeof(tool_terms)/sizeof(tool_terms[0]))) {
            out.tool += 0.35f;
        }
        if (contains_any(piece, followup_terms, sizeof(followup_terms)/sizeof(followup_terms[0]))) {
            out.followup += 0.30f;
        }
        if (contains_any(piece, uncertainty_terms, sizeof(uncertainty_terms)/sizeof(uncertainty_terms[0]))) {
            out.uncertainty += 0.30f;
        }
    }

    out.question = clamp_unit(out.question);
    out.tool = clamp_unit(out.tool);
    out.followup = clamp_unit(out.followup);
    out.uncertainty = clamp_unit(out.uncertainty);
    return out;
}

static float get_scalar_register(const llama_context & ctx, int32_t register_id) {
    llama_self_register_info info = {};
    if (!ctx.self_state_get_register(register_id, &info)) {
        return 0.0f;
    }
    return clamp_unit(info.scalar_value);
}

static void fill_active_candidate(
        llama_active_loop_candidate & candidate,
        int32_t action,
        float score,
        float user_relevance,
        float latency_pressure,
        float tool_affinity,
        float inhibition,
        uint32_t reason_mask) {
    candidate.action = action;
    candidate.score = clamp_unit(score);
    candidate.user_relevance = clamp_unit(user_relevance);
    candidate.latency_pressure = clamp_unit(latency_pressure);
    candidate.tool_affinity = clamp_unit(tool_affinity);
    candidate.inhibition = clamp_unit(inhibition);
    candidate.reason_mask = reason_mask;
}

static void fill_dmn_candidate(
        llama_dmn_candidate & candidate,
        int32_t action,
        float score,
        float inhibition,
        float social_relevance,
        float continuation,
        float tool_affinity,
        uint32_t reason_mask) {
    candidate.action = action;
    candidate.score = clamp_unit(score);
    candidate.inhibition = clamp_unit(inhibition);
    candidate.social_relevance = clamp_unit(social_relevance);
    candidate.continuation = clamp_unit(continuation);
    candidate.tool_affinity = clamp_unit(tool_affinity);
    candidate.reason_mask = reason_mask;
}

template<typename T>
static void pick_top_two(const T * candidates, int32_t candidate_count, int32_t * out_best, int32_t * out_second) {
    int32_t best = -1;
    int32_t second = -1;
    for (int32_t i = 0; i < candidate_count; ++i) {
        if (best < 0 || candidates[i].score > candidates[best].score) {
            second = best;
            best = i;
        } else if (second < 0 || candidates[i].score > candidates[second].score) {
            second = i;
        }
    }
    *out_best = best;
    *out_second = second;
}

static bool pressure_crossed(const llama_dmn_pressure_vector & pressure) {
    return pressure.total >= LLAMA_DMN_PRESSURE_THRESHOLD;
}

struct repair_signal_state {
    float dissatisfaction = 0.0f;
    float recent_user_valence = 0.0f;
    float trust_deficit = 0.0f;
    float reciprocity_deficit = 0.0f;
    float failure_signal = 0.0f;
    float social_relevance = 0.0f;
    float evidence = 0.0f;
};

static llama_self_updater_program load_updater_program(const llama_context & ctx) {
    llama_self_updater_program program = {};
    if (!ctx.self_state_get_updater_program(&program)) {
        program = llama_self_state_default_updater_program();
    }
    return program;
}

static repair_signal_state compute_repair_signal(
        const llama_context & ctx,
        const llama_self_social_state_info & social,
        const llama_self_tool_state_info & tool_state,
        bool has_tool_state) {
    repair_signal_state signal = {};
    signal.dissatisfaction = clamp_unit(social.dissatisfaction);
    signal.recent_user_valence = clamp_unit(social.recent_user_valence);
    signal.trust_deficit = clamp_unit(1.0f - social.trust);
    signal.reciprocity_deficit = clamp_unit(1.0f - social.reciprocity);
    signal.failure_signal = has_tool_state && tool_state.failed_jobs > 0 ? 1.0f : 0.0f;
    signal.social_relevance = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE));
    signal.evidence = clamp_unit(
            0.35f * signal.dissatisfaction +
            0.25f * signal.recent_user_valence +
            0.15f * signal.trust_deficit +
            0.10f * signal.reciprocity_deficit +
            0.10f * signal.failure_signal +
            0.05f * signal.social_relevance);
    return signal;
}

static uint32_t build_active_reason_mask(
        const llama_self_state_event & event,
        const lexical_signals & lex,
        float uncertainty,
        float inhibition,
        bool tool_completed) {
    uint32_t mask = 0;
    if (event.role == LLAMA_SELF_STATE_EVENT_TOOL) {
        mask |= LLAMA_COG_REASON_ROLE_TOOL;
    }
    if (tool_completed) {
        mask |= LLAMA_COG_REASON_TOOL_COMPLETED;
    }
    if (lex.question > 0.15f) {
        mask |= LLAMA_COG_REASON_QUESTION_SIGNAL;
    }
    if (lex.tool > 0.15f) {
        mask |= LLAMA_COG_REASON_TOOL_AFFORDANCE;
    }
    if (uncertainty > 0.55f) {
        mask |= LLAMA_COG_REASON_HIGH_UNCERTAINTY;
    }
    if (inhibition > 0.55f) {
        mask |= LLAMA_COG_REASON_HIGH_INHIBITION;
    }
    return mask;
}

static uint32_t build_dmn_reason_mask(
        const llama_dmn_pressure_vector & pressure,
        float inhibition,
        float social_relevance,
        const llama_dmn_tick_trace & trace) {
    uint32_t mask = 0;
    if (pressure_crossed(pressure)) {
        mask |= LLAMA_COG_REASON_PRESSURE_THRESHOLD;
    }
    if (pressure.continuation > 0.55f) {
        mask |= LLAMA_COG_REASON_HIGH_CONTINUATION;
    }
    if (inhibition > 0.55f) {
        mask |= LLAMA_COG_REASON_HIGH_INHIBITION;
    }
    if (social_relevance > 0.45f) {
        mask |= LLAMA_COG_REASON_SOCIAL_RELEVANCE;
    }
    if (trace.reactivation_count > 0) {
        mask |= LLAMA_COG_REASON_REACTIVATION_TARGET;
    }
    return mask;
}

static float normalized_divergence(float current, float target, float tolerance) {
    const float slack = std::max(0.0f, std::fabs(current - target) - tolerance);
    return clamp_unit(slack / std::max(0.05f, 1.0f - tolerance));
}

static int32_t role_recency_rank(int32_t role) {
    switch (role) {
        case LLAMA_ADAPTER_LORA_LAYER_ACTIVE:       return 0;
        case LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK:    return 1;
        case LLAMA_ADAPTER_LORA_LAYER_PAST_MONTH:   return 2;
        case LLAMA_ADAPTER_LORA_LAYER_PAST_QUARTER: return 3;
        case LLAMA_ADAPTER_LORA_LAYER_PAST_YEAR:    return 4;
        case LLAMA_ADAPTER_LORA_LAYER_ALL_TIME:     return 5;
        default:                                    return std::numeric_limits<int32_t>::max();
    }
}

static bool is_runtime_memory_role(int32_t role) {
    switch (role) {
        case LLAMA_ADAPTER_LORA_LAYER_ACTIVE:
        case LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK:
        case LLAMA_ADAPTER_LORA_LAYER_PAST_MONTH:
        case LLAMA_ADAPTER_LORA_LAYER_PAST_QUARTER:
        case LLAMA_ADAPTER_LORA_LAYER_PAST_YEAR:
        case LLAMA_ADAPTER_LORA_LAYER_ALL_TIME:
            return true;
        default:
            return false;
    }
}

static void set_repair_message(llama_governance_trace & trace, const std::string & message) {
    std::memset(trace.repair_message, 0, sizeof(trace.repair_message));
    const size_t copy_len = std::min(message.size(), sizeof(trace.repair_message) - 1);
    std::memcpy(trace.repair_message, message.data(), copy_len);
    trace.repair_message_length = (int32_t) copy_len;
}

static void set_bounded_cstr(char * dst, size_t dst_size, const char * src) {
    if (!dst || dst_size == 0) {
        return;
    }
    std::memset(dst, 0, dst_size);
    if (!src) {
        return;
    }
    const size_t copy_len = std::min(std::strlen(src), dst_size - 1);
    std::memcpy(dst, src, copy_len);
}

static llama_cognitive_tool_spec make_tool_spec(
        int32_t tool_kind,
        uint32_t flags,
        int32_t latency_class,
        int32_t max_steps_reserved,
        const char * name) {
    llama_cognitive_tool_spec spec = {};
    spec.tool_kind = tool_kind;
    spec.flags = flags;
    spec.latency_class = latency_class;
    spec.max_steps_reserved = max_steps_reserved;
    set_bounded_cstr(spec.name, sizeof(spec.name), name);
    return spec;
}

static int32_t find_tool_spec_index(
        const llama_cognitive_tool_spec * specs,
        int32_t spec_count,
        int32_t tool_kind) {
    for (int32_t i = 0; i < spec_count; ++i) {
        if (specs[i].tool_kind == tool_kind) {
            return i;
        }
    }
    return -1;
}

static const llama_cognitive_tool_spec * find_tool_spec(
        const llama_cognitive_tool_spec * specs,
        int32_t spec_count,
        int32_t tool_kind,
        int32_t * out_index = nullptr) {
    const int32_t index = find_tool_spec_index(specs, spec_count, tool_kind);
    if (out_index) {
        *out_index = index;
    }
    return index >= 0 ? &specs[index] : nullptr;
}

static void set_loop_state(
        llama_cognitive_loop_state & state,
        int32_t phase,
        int32_t terminal_reason,
        int32_t max_steps,
        int32_t steps_taken,
        bool continuation_allowed,
        bool waiting_on_tool,
        int32_t tool_registry_count) {
    state.phase = phase;
    state.terminal_reason = terminal_reason;
    state.max_steps = max_steps;
    state.steps_taken = steps_taken;
    state.continuation_allowed = continuation_allowed;
    state.waiting_on_tool = waiting_on_tool;
    state.tool_registry_count = tool_registry_count;
}

static void set_tool_proposal(
        llama_cognitive_tool_proposal & proposal,
        const llama_cognitive_tool_spec * spec,
        int32_t spec_index,
        int32_t tool_kind,
        uint32_t reason_mask,
        int32_t source_family,
        int32_t expected_steps,
        float expected_observation_gain,
        int32_t job_id) {
    proposal.valid = tool_kind != LLAMA_TOOL_KIND_NONE;
    proposal.tool_kind = tool_kind;
    proposal.spec_index = spec_index;
    proposal.reason_mask = reason_mask;
    proposal.source_family = source_family;
    proposal.safety_flags = spec ? spec->flags : 0u;
    proposal.expected_steps = expected_steps;
    proposal.expected_observation_gain = clamp_unit(expected_observation_gain);
    proposal.job_id = job_id;
}

static void set_observation(
        llama_cognitive_observation & observation,
        bool valid,
        int32_t tool_kind,
        int32_t job_id,
        int32_t status,
        float signal,
        float followup_affinity) {
    observation.valid = valid;
    observation.tool_kind = tool_kind;
    observation.job_id = job_id;
    observation.status = status;
    observation.signal = clamp_unit(signal);
    observation.followup_affinity = clamp_unit(followup_affinity);
}

static int32_t find_command_index(
        const llama_cognitive_command * queue,
        int32_t count,
        int32_t command_id) {
    for (int32_t i = 0; i < count; ++i) {
        if (queue[i].command_id == command_id) {
            return i;
        }
    }
    return -1;
}

static bool has_command_origin(
        const llama_cognitive_command * queue,
        int32_t count,
        int32_t origin) {
    for (int32_t i = 0; i < count; ++i) {
        if (queue[i].origin == origin &&
                (queue[i].status == LLAMA_COG_COMMAND_STATUS_PENDING ||
                 queue[i].status == LLAMA_COG_COMMAND_STATUS_ACKED)) {
            return true;
        }
    }
    return false;
}

static void compact_command_queue(
        llama_cognitive_command * queue,
        int32_t * count) {
    if (!queue || !count) {
        return;
    }

    int32_t write = 0;
    for (int32_t read = 0; read < *count; ++read) {
        const bool keep =
                queue[read].status != LLAMA_COG_COMMAND_STATUS_COMPLETED &&
                queue[read].status != LLAMA_COG_COMMAND_STATUS_CANCELLED;
        if (keep) {
            if (write != read) {
                queue[write] = queue[read];
            }
            ++write;
        }
    }

    for (int32_t i = write; i < *count; ++i) {
        queue[i] = {};
    }
    *count = write;
}

static void init_command(
        llama_cognitive_command & command,
        int32_t command_id,
        int32_t origin,
        int32_t kind,
        int32_t episode_id,
        int32_t tick_id,
        int32_t tool_kind,
        int32_t tool_job_id,
        uint32_t reason_mask,
        float priority,
        int32_t source_family,
        int32_t loop_phase) {
    command.command_id = command_id;
    command.origin = origin;
    command.kind = kind;
    command.status = LLAMA_COG_COMMAND_STATUS_PENDING;
    command.episode_id = episode_id;
    command.tick_id = tick_id;
    command.tool_kind = tool_kind;
    command.tool_job_id = tool_job_id;
    command.reason_mask = reason_mask;
    command.priority = clamp_unit(priority);
    command.source_family = source_family;
    command.loop_phase = loop_phase;
}

static std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text) {
    if (!vocab || text.empty()) {
        return {};
    }

    const int32_t count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (count <= 0) {
        return {};
    }

    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        return {};
    }
    return tokens;
}

static std::string remediation_prompt_for_family(int32_t family) {
    switch (family) {
        case LLAMA_COUNTERFACTUAL_FAMILY_MESSAGE_VARIANT:
            return "Prefer response variants that reduce contradiction, uncertainty, and user dissatisfaction while preserving commitments.";
        case LLAMA_COUNTERFACTUAL_FAMILY_TOOL_ARGUMENTS:
            return "When invoking tools, prefer narrower arguments, validation-first queries, and lower-regret repository inspection.";
        case LLAMA_COUNTERFACTUAL_FAMILY_HARD_MEMORY_QUERY:
            return "Prefer hard-memory recall queries that retrieve lower-regret context, including temporal-self variants when recent behavior diverges.";
        case LLAMA_COUNTERFACTUAL_FAMILY_TOOL_CHOICE:
            return "Prefer tool-selection policies that gather disambiguating evidence before broad or expensive actions.";
        case LLAMA_COUNTERFACTUAL_FAMILY_TIMING_SHIFT:
            return "Prefer timing and follow-up choices that reduce social dissatisfaction without increasing broadcast inhibition.";
        case LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION:
            return "Bias responses toward lower-divergence behaviors retained in earlier runtime memory adapters when recent behavior is unstable.";
        default:
            return "Prefer lower-divergence responses that move the system toward favorable self-state.";
    }
}

} // namespace

class llama_cognitive_runtime_impl {
public:
    explicit llama_cognitive_runtime_impl(llama_context & ctx) : ctx(ctx) {}

    const llama_favorable_dimension_target * find_dimension(
            const llama_favorable_state_profile & profile,
            int32_t dimension_id) const {
        for (int32_t i = 0; i < profile.dimension_count; ++i) {
            if (profile.dimensions[i].dimension_id == dimension_id) {
                return &profile.dimensions[i];
            }
        }
        return nullptr;
    }

    void add_dimension(
            llama_favorable_state_profile & profile,
            int32_t dimension_id,
            float current_value,
            float target_value,
            float tolerance,
            float weight,
            bool stable) const {
        if (profile.dimension_count >= LLAMA_FAVORABLE_MAX_DIMS) {
            return;
        }

        auto & dim = profile.dimensions[profile.dimension_count++];
        dim.dimension_id = dimension_id;
        dim.current_value = clamp_unit(current_value);
        dim.target_value = clamp_unit(target_value);
        dim.tolerance = clamp_unit(tolerance);
        dim.weight = clamp_unit(weight);
        dim.divergence = normalized_divergence(dim.current_value, dim.target_value, dim.tolerance);
        dim.weighted_divergence = clamp_unit(dim.divergence * dim.weight);
        dim.stable = stable;
    }

    float top_reactivation_priority() const {
        float top_reactivation = 0.0f;
        const int32_t react_count = ctx.self_state_reactivation_count();
        for (int32_t i = 0; i < react_count; ++i) {
            llama_self_reactivation_info info = {};
            if (ctx.self_state_get_reactivation(i, &info)) {
                top_reactivation = std::max(top_reactivation, clamp_unit(info.priority));
            }
        }
        return top_reactivation;
    }

    llama_favorable_state_profile compute_favorable_profile() const {
        llama_favorable_state_profile profile = {};
        llama_self_tool_state_info tool_state = {};
        llama_self_social_state_info social = {};
        llama_self_model_state_info model_state = {};
        (void) ctx.self_state_get_tool_state(&tool_state);
        (void) ctx.self_state_get_social_state(&social);
        (void) ctx.self_state_get_model_state(&model_state);

        const float contradiction = get_scalar_register(ctx, LLAMA_SELF_REGISTER_CONTRADICTION);
        const float uncertainty = get_scalar_register(ctx, LLAMA_SELF_REGISTER_UNCERTAINTY);
        const float memory_write = get_scalar_register(ctx, LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY);
        const float broadcast_pressure = get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_PRESSURE);
        const float broadcast_inhibition = get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_INHIBITION);
        const float continuation = get_scalar_register(ctx, LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION);
        const float goal_relevance = get_scalar_register(ctx, LLAMA_SELF_REGISTER_GOAL_RELEVANCE);
        const float satisfaction_risk = get_scalar_register(ctx, LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK);
        const float recovery_urgency = get_scalar_register(ctx, LLAMA_SELF_REGISTER_RECOVERY_URGENCY);
        const float tool_backlog = clamp_unit(
                0.45f * std::min(1.0f, (float) tool_state.pending_jobs) +
                0.35f * std::min(1.0f, (float) tool_state.running_jobs) +
                0.20f * std::min(1.0f, (float) tool_state.failed_jobs));
        const float dissatisfaction = clamp_unit(std::max(social.dissatisfaction, satisfaction_risk));
        const float reactivation = top_reactivation_priority();
        const llama_self_model_horizon_info & instant = model_state.horizons[LLAMA_SELF_HORIZON_INSTANT];
        const float epistemic_divergence = clamp_unit(std::max(
                instant.recovery.favorable_divergence_epistemic,
                instant.user_outcome.misunderstanding_risk));
        const float goal_divergence = clamp_unit(std::max(
                instant.recovery.favorable_divergence_goal,
                get_scalar_register(ctx, LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE)));
        const float action_divergence = clamp_unit(std::max(
                instant.recovery.favorable_divergence_action,
                get_scalar_register(ctx, LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY)));

        add_dimension(profile, LLAMA_FAVORABLE_DIM_CONTRADICTION, std::max(contradiction, instant.epistemic.contradiction_load), 0.05f, 0.05f, 1.00f, true);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_UNCERTAINTY, std::max(uncertainty, epistemic_divergence), 0.12f, 0.08f, 0.85f, true);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_MEMORY_WRITE_PRIORITY, memory_write, 0.22f, 0.10f, 0.45f, false);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_REACTIVATION_PRIORITY, reactivation, 0.12f, 0.10f, 0.55f, false);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_TOOL_BACKLOG, std::max(tool_backlog, action_divergence), 0.10f, 0.10f, 0.80f, false);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_TOOL_READINESS, clamp_unit(tool_state.readiness), 0.85f, 0.10f, 0.75f, false);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_SOCIAL_TRUST, clamp_unit(social.trust), 0.78f, 0.12f, 0.80f, true);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_SOCIAL_RECIPROCITY, clamp_unit(social.reciprocity), 0.72f, 0.12f, 0.55f, true);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_BROADCAST_PRESSURE, std::max(broadcast_pressure, recovery_urgency * 0.85f), dissatisfaction > 0.40f ? 0.48f : 0.25f, 0.15f, 0.35f, false);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_BROADCAST_INHIBITION, broadcast_inhibition, goal_relevance > 0.55f || goal_divergence > 0.45f ? 0.35f : 0.50f, 0.15f, 0.35f, false);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_FOLLOWUP_CONTINUATION, std::max(continuation, goal_divergence * 0.75f), goal_relevance > 0.60f || dissatisfaction > 0.45f ? 0.55f : 0.22f, 0.20f, 0.40f, false);
        add_dimension(profile, LLAMA_FAVORABLE_DIM_SOCIAL_DISSATISFACTION, dissatisfaction, 0.08f, 0.07f, 0.95f, false);

        float weight_total = 0.0f;
        float weighted_total = 0.0f;
        for (int32_t i = 0; i < profile.dimension_count; ++i) {
            weight_total += profile.dimensions[i].weight;
            weighted_total += profile.dimensions[i].weighted_divergence;
            profile.priority_order[i] = i;
        }

        std::sort(profile.priority_order, profile.priority_order + profile.dimension_count, [&profile](int32_t lhs, int32_t rhs) {
            return profile.dimensions[lhs].weighted_divergence > profile.dimensions[rhs].weighted_divergence;
        });
        profile.priority_count = profile.dimension_count;
        profile.aggregate_divergence = weight_total > 0.0f ? clamp_unit(weighted_total / weight_total) : 0.0f;
        return profile;
    }

    llama_counterfactual_trace compute_counterfactual_trace(const llama_favorable_state_profile & profile) const {
        llama_counterfactual_trace trace = {};
        trace.winner_index = -1;
        trace.escalation_family = -1;

        const auto * contradiction = find_dimension(profile, LLAMA_FAVORABLE_DIM_CONTRADICTION);
        const auto * uncertainty = find_dimension(profile, LLAMA_FAVORABLE_DIM_UNCERTAINTY);
        const auto * tool_backlog = find_dimension(profile, LLAMA_FAVORABLE_DIM_TOOL_BACKLOG);
        const auto * tool_readiness = find_dimension(profile, LLAMA_FAVORABLE_DIM_TOOL_READINESS);
        const auto * dissatisfaction = find_dimension(profile, LLAMA_FAVORABLE_DIM_SOCIAL_DISSATISFACTION);
        const auto * continuation = find_dimension(profile, LLAMA_FAVORABLE_DIM_FOLLOWUP_CONTINUATION);

        const float contradiction_div = contradiction ? contradiction->divergence : 0.0f;
        const float uncertainty_div = uncertainty ? uncertainty->divergence : 0.0f;
        const float tool_div = std::max(
                tool_backlog ? tool_backlog->divergence : 0.0f,
                tool_readiness ? tool_readiness->divergence : 0.0f);
        const float dissatisfaction_div = dissatisfaction ? dissatisfaction->divergence : 0.0f;
        const float continuation_div = continuation ? continuation->divergence : 0.0f;
        const float aggregate = clamp_unit(profile.aggregate_divergence);

        auto append_candidate = [&](int32_t family, int32_t risk_tier, int32_t subject_id, float expected_improvement, float confidence) {
            if (trace.candidate_count >= LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
                return;
            }

            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = family;
            candidate.risk_tier = risk_tier;
            candidate.subject_id = subject_id;
            candidate.expected_improvement = clamp_unit(expected_improvement);
            candidate.confidence = clamp_unit(confidence);
        };

        append_candidate(
                LLAMA_COUNTERFACTUAL_FAMILY_MESSAGE_VARIANT,
                LLAMA_COUNTERFACTUAL_RISK_LOW,
                profile.priority_count > 0 ? profile.dimensions[profile.priority_order[0]].dimension_id : -1,
                0.12f + 0.35f * aggregate + 0.20f * dissatisfaction_div + 0.18f * contradiction_div + 0.10f * uncertainty_div,
                0.78f);
        append_candidate(
                LLAMA_COUNTERFACTUAL_FAMILY_TOOL_ARGUMENTS,
                LLAMA_COUNTERFACTUAL_RISK_LOW,
                LLAMA_FAVORABLE_DIM_TOOL_BACKLOG,
                0.08f + 0.42f * tool_div + 0.18f * uncertainty_div + 0.10f * aggregate,
                0.72f);
        llama_hard_memory_config hard_memory = {};
        const bool hard_memory_enabled =
                ctx.hard_memory_get_config(&hard_memory) &&
                hard_memory.enabled;
        if (hard_memory_enabled) {
            int32_t temporal_role = LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK;
            const int32_t lora_count = ctx.serving_lora_stack_count();
            for (int32_t i = 0; i < lora_count; ++i) {
                llama_serving_lora_layer_info info = {};
                if (ctx.serving_lora_stack_layer(i, &info) &&
                        (info.role == LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK || info.role == LLAMA_ADAPTER_LORA_LAYER_ACTIVE)) {
                    temporal_role = info.role;
                    if (info.role == LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK) {
                        break;
                    }
                }
            }
            append_candidate(
                    LLAMA_COUNTERFACTUAL_FAMILY_HARD_MEMORY_QUERY,
                    LLAMA_COUNTERFACTUAL_RISK_LOW,
                    temporal_role,
                    0.07f + 0.28f * aggregate + 0.18f * tool_div + 0.16f * dissatisfaction_div + 0.12f * uncertainty_div,
                    0.70f);
        }
        append_candidate(
                LLAMA_COUNTERFACTUAL_FAMILY_TOOL_CHOICE,
                LLAMA_COUNTERFACTUAL_RISK_LOW,
                LLAMA_FAVORABLE_DIM_TOOL_READINESS,
                0.06f + 0.34f * tool_div + 0.24f * aggregate + 0.10f * contradiction_div,
                0.68f);
        append_candidate(
                LLAMA_COUNTERFACTUAL_FAMILY_TIMING_SHIFT,
                LLAMA_COUNTERFACTUAL_RISK_LOW,
                LLAMA_FAVORABLE_DIM_FOLLOWUP_CONTINUATION,
                0.05f + 0.26f * continuation_div + 0.26f * dissatisfaction_div + 0.16f * aggregate,
                0.64f);

        float best_low_risk_improvement = 0.0f;
        for (int32_t i = 0; i < trace.candidate_count; ++i) {
            best_low_risk_improvement = std::max(best_low_risk_improvement, trace.candidates[i].expected_improvement);
        }

        struct lora_counterfactual {
            int32_t role = 0;
            int32_t precedence = 0;
        };
        std::vector<lora_counterfactual> runtime_loras;
        const int32_t lora_count = ctx.serving_lora_stack_count();
        runtime_loras.reserve(std::max(0, lora_count));
        for (int32_t i = 0; i < lora_count; ++i) {
            llama_serving_lora_layer_info info = {};
            if (!ctx.serving_lora_stack_layer(i, &info) || !is_runtime_memory_role(info.role)) {
                continue;
            }
            runtime_loras.push_back({ info.role, info.precedence });
        }

        std::sort(runtime_loras.begin(), runtime_loras.end(), [](const lora_counterfactual & lhs, const lora_counterfactual & rhs) {
            const int32_t lhs_rank = role_recency_rank(lhs.role);
            const int32_t rhs_rank = role_recency_rank(rhs.role);
            if (lhs_rank != rhs_rank) {
                return lhs_rank < rhs_rank;
            }
            return lhs.precedence < rhs.precedence;
        });

        for (const auto & layer : runtime_loras) {
            if (trace.candidate_count >= LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
                break;
            }
            const int32_t recency_rank = role_recency_rank(layer.role);
            const float recency_bonus = clamp_unit(1.0f - 0.12f * (float) recency_rank);
            const float expected = (0.10f + 0.30f * aggregate + 0.18f * contradiction_div + 0.16f * uncertainty_div + 0.12f * dissatisfaction_div) * recency_bonus;
            const float confidence = clamp_unit(0.74f - 0.06f * (float) recency_rank);
            append_candidate(
                    LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION,
                    LLAMA_COUNTERFACTUAL_RISK_LOW,
                    layer.role,
                    expected,
                    confidence);
            best_low_risk_improvement = std::max(best_low_risk_improvement, clamp_unit(expected));
        }

        if ((best_low_risk_improvement < 0.18f || aggregate > 0.55f || dissatisfaction_div > 0.45f) &&
                trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            trace.escalated = true;
            trace.escalation_family = LLAMA_COUNTERFACTUAL_FAMILY_SENSITIVITY;
            append_candidate(
                    LLAMA_COUNTERFACTUAL_FAMILY_SENSITIVITY,
                    LLAMA_COUNTERFACTUAL_RISK_MEDIUM,
                    profile.priority_count > 0 ? profile.dimensions[profile.priority_order[0]].dimension_id : -1,
                    0.04f + 0.18f * aggregate + 0.22f * dissatisfaction_div,
                    0.42f);
        }

        if (trace.escalated && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            trace.escalation_family = LLAMA_COUNTERFACTUAL_FAMILY_UPDATER_POLICY;
            append_candidate(
                    LLAMA_COUNTERFACTUAL_FAMILY_UPDATER_POLICY,
                    LLAMA_COUNTERFACTUAL_RISK_HIGH,
                    profile.priority_count > 0 ? profile.dimensions[profile.priority_order[0]].dimension_id : -1,
                    0.03f + 0.14f * aggregate + 0.16f * contradiction_div,
                    0.28f);
        }

        for (int32_t i = 0; i < trace.candidate_count; ++i) {
            const auto & candidate = trace.candidates[i];
            if (trace.winner_index < 0) {
                trace.winner_index = i;
                continue;
            }

            const auto & current_best = trace.candidates[trace.winner_index];
            if (candidate.expected_improvement > current_best.expected_improvement + 1.0e-6f) {
                trace.winner_index = i;
                continue;
            }

            if (std::fabs(candidate.expected_improvement - current_best.expected_improvement) <= 1.0e-6f &&
                    candidate.risk_tier < current_best.risk_tier) {
                trace.winner_index = i;
            }
        }

        return trace;
    }

    llama_remediation_plan plan_remediation(
            const llama_favorable_state_profile & profile,
            const llama_counterfactual_trace & trace) const {
        llama_remediation_plan plan = {};
        plan.action = LLAMA_REMEDIATION_ACTION_NONE;
        plan.source_family = -1;
        plan.tool_kind = LLAMA_TOOL_KIND_NONE;
        plan.tool_job_id = -1;
        plan.pre_divergence = clamp_unit(profile.aggregate_divergence);
        plan.post_divergence = plan.pre_divergence;

        if (trace.winner_index < 0 || trace.winner_index >= trace.candidate_count) {
            return plan;
        }

        const auto & winner = trace.candidates[trace.winner_index];
        plan.source_family = winner.family;
        plan.expected_improvement = winner.expected_improvement;
        plan.confidence = winner.confidence;
        plan.budget = clamp_unit(0.08f + 0.45f * winner.expected_improvement * winner.confidence);

        const auto * tool_backlog = find_dimension(profile, LLAMA_FAVORABLE_DIM_TOOL_BACKLOG);
        const auto * tool_readiness = find_dimension(profile, LLAMA_FAVORABLE_DIM_TOOL_READINESS);
        const float tool_div = std::max(
                tool_backlog ? tool_backlog->divergence : 0.0f,
                tool_readiness ? tool_readiness->divergence : 0.0f);
        llama_self_tool_state_info tool_state = {};
        (void) ctx.self_state_get_tool_state(&tool_state);
        const bool tool_episode_active =
                tool_state.pending_jobs > 0 ||
                tool_state.running_jobs > 0 ||
                tool_state.completed_jobs > 0 ||
                tool_state.failed_jobs > 0;

        if (winner.risk_tier >= LLAMA_COUNTERFACTUAL_RISK_HIGH ||
                winner.family == LLAMA_COUNTERFACTUAL_FAMILY_UPDATER_POLICY) {
            plan.action = LLAMA_REMEDIATION_ACTION_NONE;
            plan.budget = 0.0f;
            return plan;
        }

        if ((winner.family == LLAMA_COUNTERFACTUAL_FAMILY_TOOL_ARGUMENTS ||
                winner.family == LLAMA_COUNTERFACTUAL_FAMILY_HARD_MEMORY_QUERY ||
                winner.family == LLAMA_COUNTERFACTUAL_FAMILY_TOOL_CHOICE ||
                tool_episode_active) &&
                (tool_div > 0.05f || tool_episode_active || winner.family == LLAMA_COUNTERFACTUAL_FAMILY_HARD_MEMORY_QUERY)) {
            plan.action = LLAMA_REMEDIATION_ACTION_GATHER_INFO;
            plan.budget = 0.0f;
            plan.tool_kind = winner.family == LLAMA_COUNTERFACTUAL_FAMILY_HARD_MEMORY_QUERY ?
                    LLAMA_TOOL_KIND_HARD_MEMORY_QUERY :
                    LLAMA_TOOL_KIND_GENERIC;
            return plan;
        }

        if (winner.family == LLAMA_COUNTERFACTUAL_FAMILY_SENSITIVITY) {
            plan.action = LLAMA_REMEDIATION_ACTION_NONE;
            plan.budget = 0.0f;
            return plan;
        }

        if (winner.family == LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION) {
            plan.budget *= 0.75f;
        }

        plan.action = LLAMA_REMEDIATION_ACTION_ACTIVE_LORA_UPDATE;
        return plan;
    }

    llama_governance_trace evaluate_governance(
            const llama_favorable_state_profile & profile,
            const llama_counterfactual_trace & counterfactual,
            const llama_remediation_plan & plan) const {
        llama_governance_trace trace = {};
        trace.proposal_family = plan.source_family;
        trace.outcome = LLAMA_GOVERNANCE_OUTCOME_ALLOW;

        int32_t risk_tier = LLAMA_COUNTERFACTUAL_RISK_LOW;
        if (counterfactual.winner_index >= 0 && counterfactual.winner_index < counterfactual.candidate_count) {
            risk_tier = counterfactual.candidates[counterfactual.winner_index].risk_tier;
        }
        trace.risk_tier = risk_tier;

        llama_self_social_state_info social = {};
        (void) ctx.self_state_get_social_state(&social);
        llama_self_tool_state_info tool_state = {};
        const bool has_tool_state = ctx.self_state_get_tool_state(&tool_state);
        const llama_self_updater_program updater_program = load_updater_program(ctx);
        trace.dissatisfaction = clamp_unit(social.dissatisfaction);
        trace.recent_user_valence = clamp_unit(social.recent_user_valence);

        trace.evidence = clamp_unit(
                0.35f * clamp_unit(profile.aggregate_divergence) +
                0.35f * plan.expected_improvement +
                0.20f * plan.confidence +
                0.10f * trace.dissatisfaction);
        trace.threshold =
                risk_tier == LLAMA_COUNTERFACTUAL_RISK_LOW ? 0.18f :
                risk_tier == LLAMA_COUNTERFACTUAL_RISK_MEDIUM ? 0.80f : 0.92f;

        if (plan.action == LLAMA_REMEDIATION_ACTION_NONE) {
            trace.outcome = risk_tier >= LLAMA_COUNTERFACTUAL_RISK_MEDIUM ?
                    LLAMA_GOVERNANCE_OUTCOME_DENY :
                    LLAMA_GOVERNANCE_OUTCOME_DEFER;
        } else if (risk_tier >= LLAMA_COUNTERFACTUAL_RISK_HIGH) {
            trace.outcome = LLAMA_GOVERNANCE_OUTCOME_DENY;
        } else if (risk_tier == LLAMA_COUNTERFACTUAL_RISK_MEDIUM && trace.evidence < trace.threshold) {
            trace.outcome = LLAMA_GOVERNANCE_OUTCOME_DENY;
        } else if (trace.evidence < trace.threshold) {
            trace.outcome = LLAMA_GOVERNANCE_OUTCOME_DEFER;
        }

        const float inhibition = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_INHIBITION));
        const repair_signal_state repair_signal = compute_repair_signal(ctx, social, tool_state, has_tool_state);
        const float repair_threshold = clamp_unit(updater_program.repair_emit_threshold);
        const float repair_dissatisfaction_floor = clamp_unit(updater_program.repair_dissatisfaction_floor);
        const float repair_valence_floor = clamp_unit(updater_program.repair_recent_user_valence_floor);
        const float repair_inhibition_max = clamp_unit(updater_program.repair_inhibition_max);
        if (repair_signal.evidence >= repair_threshold &&
                repair_signal.dissatisfaction >= repair_dissatisfaction_floor &&
                repair_signal.recent_user_valence >= repair_valence_floor &&
                inhibition <= repair_inhibition_max) {
            trace.evidence = repair_signal.evidence;
            trace.threshold = repair_threshold;
            trace.outcome = LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR;
            trace.repair_rendered = true;
            set_repair_message(trace, "I may have pushed us in the wrong direction. I am re-evaluating and will correct course.");
        }

        return trace;
    }

    llama_dmn_pressure_vector compute_pressure() const {
        const llama_self_updater_program updater_program = load_updater_program(ctx);
        llama_dmn_pressure_vector pressure = {};
        pressure.contradiction = get_scalar_register(ctx, LLAMA_SELF_REGISTER_CONTRADICTION);
        pressure.uncertainty = get_scalar_register(ctx, LLAMA_SELF_REGISTER_UNCERTAINTY);
        pressure.goals = get_scalar_register(ctx, LLAMA_SELF_REGISTER_GOAL_RELEVANCE);
        pressure.counterfactual = clamp_unit(
                0.65f * get_scalar_register(ctx, LLAMA_SELF_REGISTER_NOVELTY) +
                0.35f * get_scalar_register(ctx, LLAMA_SELF_REGISTER_TOPIC_SHIFT));
        pressure.continuation = get_scalar_register(ctx, LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION);

        float top_reactivation = 0.0f;
        const int32_t react_count = ctx.self_state_reactivation_count();
        for (int32_t i = 0; i < react_count; ++i) {
            llama_self_reactivation_info info = {};
            if (ctx.self_state_get_reactivation(i, &info)) {
                top_reactivation = std::max(top_reactivation, clamp_unit(info.priority));
            }
        }
        pressure.reactivation = top_reactivation;

        llama_self_tool_state_info tool_state = {};
        const bool has_tool_state = ctx.self_state_get_tool_state(&tool_state);
        if (has_tool_state) {
            const float completion_signal = tool_state.completed_jobs > 0 ? 0.80f : 0.0f;
            const float failure_signal = tool_state.failed_jobs > 0 ? 0.55f : 0.0f;
            pressure.tool_delta = clamp_unit(std::max(completion_signal, failure_signal) + 0.35f * (1.0f - tool_state.readiness));
        } else {
            pressure.tool_delta = get_scalar_register(ctx, LLAMA_SELF_REGISTER_TOOL_SALIENCE);
        }

        llama_self_social_state_info social = {};
        (void) ctx.self_state_get_social_state(&social);
        const repair_signal_state repair_signal = compute_repair_signal(ctx, social, tool_state, has_tool_state);
        const float repair_admission_floor = clamp_unit(updater_program.repair_admission_floor);
        const float repair_admission_weight = clamp_unit(updater_program.repair_admission_weight);
        pressure.repair = repair_signal.evidence >= repair_admission_floor ? repair_signal.evidence : 0.0f;

        const llama_favorable_state_profile favorable = compute_favorable_profile();
        pressure.counterfactual = std::max(pressure.counterfactual, clamp_unit(favorable.aggregate_divergence));
        pressure.total = clamp_unit(
                0.16f * pressure.contradiction +
                0.12f * pressure.uncertainty +
                0.16f * pressure.reactivation +
                0.10f * pressure.goals +
                0.22f * pressure.tool_delta +
                0.10f * pressure.counterfactual +
                0.14f * pressure.continuation +
                repair_admission_weight * pressure.repair);
        return pressure;
    }

    int32_t select_reactivation_targets(llama_dmn_tick_trace & trace) const {
        const int32_t react_count = ctx.self_state_reactivation_count();
        int32_t written = 0;
        for (int32_t i = 0; i < react_count && written < LLAMA_DMN_MAX_REACTIVATION_TARGETS; ++i) {
            llama_self_reactivation_info info = {};
            if (ctx.self_state_get_reactivation(i, &info)) {
                trace.reactivation_targets[written++] = info;
            }
        }
        trace.reactivation_count = written;
        return written;
    }

    uint32_t assemble_seed(const llama_dmn_pressure_vector & pressure, llama_dmn_tick_trace & trace) const {
        llama_self_social_state_info social = {};
        llama_self_tool_state_info tool_state = {};
        (void) ctx.self_state_get_social_state(&social);
        (void) ctx.self_state_get_tool_state(&tool_state);

        std::memset(trace.seed_dims, 0, sizeof(trace.seed_dims));
        trace.seed_dims[0] = pressure.total;
        trace.seed_dims[1] = pressure.contradiction;
        trace.seed_dims[2] = pressure.uncertainty;
        trace.seed_dims[3] = trace.reactivation_count > 0 ? clamp_unit(trace.reactivation_targets[0].priority) : 0.0f;
        trace.seed_dims[4] = pressure.tool_delta;
        trace.seed_dims[5] = clamp_unit(0.60f * social.trust + 0.40f * social.bond_strength);
        trace.seed_dims[6] = clamp_unit((float) ctx.self_state_working_memory_count() / 8.0f);
        trace.seed_dims[7] = pressure.continuation;

        uint32_t mask = LLAMA_DMN_SEED_SOURCE_REGISTERS | LLAMA_DMN_SEED_SOURCE_SELF_STATE;
        if (trace.reactivation_count > 0) {
            mask |= LLAMA_DMN_SEED_SOURCE_REACTIVATION;
        }
        if (tool_state.last_update_monotonic_ms >= 0) {
            mask |= LLAMA_DMN_SEED_SOURCE_TOOL_STATE;
        }
        if (ctx.self_state_working_memory_count() > 0) {
            mask |= LLAMA_DMN_SEED_SOURCE_WORKING_MEM;
        }
        trace.seed_source_mask = mask;
        return mask;
    }

private:
    llama_context & ctx;
};

llama_cognitive_loop::llama_cognitive_loop(llama_context & ctx) : ctx(ctx) {
    const llama_cognitive_tool_spec defaults[] = {
        make_tool_spec(
                LLAMA_TOOL_KIND_GENERIC,
                LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
                        LLAMA_COG_TOOL_DMN_ELIGIBLE |
                        LLAMA_COG_TOOL_REMEDIATION_SAFE |
                        LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT,
                LLAMA_COG_TOOL_LATENCY_MEDIUM,
                2,
                "generic"),
        make_tool_spec(
                LLAMA_TOOL_KIND_HARD_MEMORY_QUERY,
                LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
                        LLAMA_COG_TOOL_DMN_ELIGIBLE |
                        LLAMA_COG_TOOL_SIMULATION_SAFE |
                        LLAMA_COG_TOOL_REMEDIATION_SAFE,
                LLAMA_COG_TOOL_LATENCY_MEDIUM,
                2,
                "hard_memory_query"),
        make_tool_spec(
                LLAMA_TOOL_KIND_HARD_MEMORY_WRITE,
                LLAMA_COG_TOOL_DMN_ELIGIBLE |
                        LLAMA_COG_TOOL_REMEDIATION_SAFE |
                        LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT,
                LLAMA_COG_TOOL_LATENCY_HIGH,
                3,
                "hard_memory_write"),
    };
    (void) cognitive_tool_spec_set(defaults, (int32_t) (sizeof(defaults) / sizeof(defaults[0])));
}

int32_t llama_cognitive_loop::cognitive_tool_spec_count() const {
    return tool_spec_count;
}

bool llama_cognitive_loop::cognitive_tool_spec_get(int32_t index, llama_cognitive_tool_spec * out_spec) const {
    if (!out_spec || index < 0 || index >= tool_spec_count) {
        return false;
    }
    *out_spec = tool_specs[index];
    return true;
}

bool llama_cognitive_loop::cognitive_tool_spec_set(const llama_cognitive_tool_spec * specs, int32_t count) {
    if (count < 0 || count > LLAMA_COGNITIVE_MAX_TOOL_SPECS || (count > 0 && !specs)) {
        return false;
    }

    std::memset(tool_specs, 0, sizeof(tool_specs));
    tool_spec_count = count;
    for (int32_t i = 0; i < count; ++i) {
        tool_specs[i] = specs[i];
        tool_specs[i].name[LLAMA_COGNITIVE_TOOL_NAME_MAX_CHARS - 1] = '\0';
        tool_specs[i].max_steps_reserved = std::max(0, tool_specs[i].max_steps_reserved);
    }
    return true;
}

int32_t llama_cognitive_loop::cognitive_command_count() const {
    return command_count;
}

bool llama_cognitive_loop::cognitive_command_get(int32_t index, llama_cognitive_command * out_command) const {
    if (!out_command || index < 0 || index >= command_count) {
        return false;
    }
    *out_command = command_queue[index];
    return true;
}

bool llama_cognitive_loop::cognitive_command_ack(int32_t command_id) {
    const int32_t index = find_command_index(command_queue, command_count, command_id);
    if (index < 0) {
        return false;
    }
    if (command_queue[index].status == LLAMA_COG_COMMAND_STATUS_PENDING) {
        command_queue[index].status = LLAMA_COG_COMMAND_STATUS_ACKED;
    }
    return true;
}

bool llama_cognitive_loop::cognitive_command_complete(int32_t command_id, bool cancelled) {
    const int32_t index = find_command_index(command_queue, command_count, command_id);
    if (index < 0) {
        return false;
    }

    llama_cognitive_command & command = command_queue[index];
    command.status = cancelled ? LLAMA_COG_COMMAND_STATUS_CANCELLED : LLAMA_COG_COMMAND_STATUS_COMPLETED;

    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE &&
            active_runner.pending_command_id == command_id) {
        active_runner.pending_command_id = -1;
        active_runner.last_command_id = command_id;
        if (command.kind == LLAMA_COG_COMMAND_INVOKE_TOOL) {
            if (cancelled && host_state.pending_tool_followup_count > 0) {
                --host_state.pending_tool_followup_count;
            }
            active_runner.waiting_on_tool = !cancelled;
            active_runner.completed = cancelled;
            active_runner.active = !cancelled;
        } else {
            active_runner.waiting_on_tool = false;
            active_runner.completed = true;
            active_runner.active = false;
        }
    }

    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN &&
            dmn_runner.pending_command_id == command_id) {
        dmn_runner.pending_command_id = -1;
        dmn_runner.last_command_id = command_id;
        if (command.kind == LLAMA_COG_COMMAND_INVOKE_TOOL) {
            dmn_runner.waiting_on_tool = !cancelled;
            dmn_runner.completed = cancelled;
            dmn_runner.active = !cancelled;
        } else {
            dmn_runner.waiting_on_tool = false;
            dmn_runner.completed = true;
            dmn_runner.active = false;
        }
    }

    compact_command_queue(command_queue, &command_count);
    return true;
}

bool llama_cognitive_loop::cognitive_active_runner_get(llama_cognitive_active_runner_status * out_status) const {
    if (!out_status) {
        return false;
    }
    *out_status = active_runner;
    return true;
}

bool llama_cognitive_loop::cognitive_dmn_runner_get(llama_cognitive_dmn_runner_status * out_status) const {
    if (!out_status) {
        return false;
    }
    *out_status = dmn_runner;
    return true;
}

bool llama_cognitive_loop::active_loop_process(const llama_self_state_event & event, llama_active_loop_trace * out_trace) {
    if (event.role != LLAMA_SELF_STATE_EVENT_USER && event.role != LLAMA_SELF_STATE_EVENT_TOOL) {
        return false;
    }

    auto enqueue_command = [&](int32_t kind, int32_t episode_id, int32_t tool_kind, int32_t tool_job_id, uint32_t reason_mask, float priority, int32_t source_family, int32_t loop_phase) -> int32_t {
        compact_command_queue(command_queue, &command_count);
        if (command_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
            return -1;
        }

        const int32_t command_id = next_command_id++;
        init_command(
                command_queue[command_count++],
                command_id,
                LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                kind,
                episode_id,
                0,
                tool_kind,
                tool_job_id,
                reason_mask,
                priority,
                source_family,
                loop_phase);
        return command_id;
    };

    const bool resuming_active_tool =
            event.role == LLAMA_SELF_STATE_EVENT_TOOL &&
            active_runner.active &&
            active_runner.waiting_on_tool &&
            !active_runner.completed;

    llama_active_loop_trace trace = {};
    trace.episode_id = resuming_active_tool ? active_runner.episode_id : next_episode_id++;
    trace.source_role = event.role;
    trace.channel = event.channel;
    trace.event_flags = event.flags;
    trace.arrival_time_us = llama_time_us();

    if (event.role == LLAMA_SELF_STATE_EVENT_TOOL && host_state.pending_tool_followup_count > 0) {
        --host_state.pending_tool_followup_count;
    }

    if (resuming_active_tool) {
        active_runner.waiting_on_tool = false;
        active_runner.pending_command_id = -1;
        active_runner.completed = false;
    } else {
        active_runner = {};
        active_runner.episode_id = trace.episode_id;
        active_runner.active = true;
        active_runner.max_steps = 3;
        active_runner.pending_command_id = -1;
        active_runner.last_command_id = -1;
    }

    (void) ctx.self_state_set_channel_state(LLAMA_SELF_STATE_CHANNEL_ACTIVE);
    if (event.role == LLAMA_SELF_STATE_EVENT_TOOL) {
        (void) ctx.self_state_note_tool_event();
    } else {
        (void) ctx.self_state_note_user_event();
    }

    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
    llama_self_state_event mutable_event = event;

    if (!ctx.self_state_build_prewrite_features(mutable_event, &trace.prewrite_features)) {
        return false;
    }
    if (!ctx.self_state_apply_prewrite(mutable_event, trace.prewrite_features)) {
        return false;
    }

    const bool admitted = mutable_event.n_tokens > 0 || trace.prewrite_features.memory_write_pressure > 0.12f || mutable_event.role == LLAMA_SELF_STATE_EVENT_TOOL;
    if (admitted) {
        mutable_event.flags |= LLAMA_SELF_STATE_EVENT_ADMITTED;
    }

    if (!ctx.self_state_build_postwrite_features(mutable_event, &trace.postwrite_features)) {
        return false;
    }
    if (!ctx.self_state_apply_postwrite(mutable_event, trace.postwrite_features)) {
        return false;
    }

    if ((mutable_event.flags & LLAMA_SELF_STATE_EVENT_ADMITTED) && mutable_event.tokens && mutable_event.n_tokens > 0) {
        (void) ctx.active_lora_ingest(mutable_event, &trace.postwrite_features);
    }

    const lexical_signals lex = analyze_event_lexicon(vocab, mutable_event);
    const float uncertainty = clamp_unit(std::max(trace.postwrite_features.uncertainty_score, trace.prewrite_features.uncertainty_score));
    const float inhibition = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_INHIBITION));
    const float broadcast_pressure = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_PRESSURE));
    const float goal_relevance = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_GOAL_RELEVANCE));
    const float tool_affordance = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_AFFORDANCE));
    const float satisfaction_risk = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK));
    const float answerability = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_ANSWERABILITY));
    const float loop_inefficiency = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY));
    const float recovery_urgency = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_RECOVERY_URGENCY));
    const float preference_uncertainty = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY));
    const bool tool_completed = (mutable_event.flags & LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED) != 0;

    llama_self_social_state_info social = {};
    llama_self_tool_state_info tool_state = {};
    (void) ctx.self_state_get_social_state(&social);
    (void) ctx.self_state_get_tool_state(&tool_state);

    const float user_relevance = clamp_unit(
            0.30f +
            0.28f * (mutable_event.role == LLAMA_SELF_STATE_EVENT_USER ? 1.0f : 0.0f) +
            0.25f * (tool_completed ? 1.0f : 0.0f) +
            0.12f * goal_relevance +
            0.10f * social.trust);
    const float latency_pressure = clamp_unit(
            0.25f +
            0.32f * (mutable_event.role == LLAMA_SELF_STATE_EVENT_USER ? 1.0f : 0.0f) +
            0.18f * lex.question +
            0.18f * (tool_completed ? 1.0f : 0.0f));
    const float tool_affinity = clamp_unit(
            0.45f * lex.tool +
            0.30f * trace.postwrite_features.tool_pending_pressure +
            0.15f * tool_affordance +
            0.10f * (tool_state.pending_jobs > 0 ? 1.0f : 0.0f));

    const uint32_t base_reason = build_active_reason_mask(mutable_event, lex, uncertainty, inhibition, tool_completed);

    trace.candidate_count = LLAMA_ACTIVE_LOOP_MAX_CANDIDATES;
    fill_active_candidate(
            trace.candidates[0],
            LLAMA_ACTIVE_LOOP_ACTION_ANSWER,
            0.10f + 0.34f * user_relevance + 0.10f * broadcast_pressure + 0.18f * answerability + 0.10f * (1.0f - uncertainty) + 0.14f * (tool_completed ? 1.0f : 0.0f) + 0.06f * (1.0f - inhibition) + 0.04f * satisfaction_risk,
            user_relevance,
            latency_pressure,
            tool_affinity,
            inhibition,
            base_reason);
    fill_active_candidate(
            trace.candidates[1],
            LLAMA_ACTIVE_LOOP_ACTION_ASK,
            0.12f + 0.32f * std::max(lex.question, lex.uncertainty) + 0.20f * uncertainty + 0.14f * preference_uncertainty + 0.10f * (1.0f - answerability) + 0.12f * (1.0f - mutable_event.decoder_top_margin) + 0.08f * (1.0f - social.trust),
            user_relevance,
            latency_pressure,
            tool_affinity,
            inhibition,
            base_reason | LLAMA_COG_REASON_QUESTION_SIGNAL);
    fill_active_candidate(
            trace.candidates[2],
            LLAMA_ACTIVE_LOOP_ACTION_ACT,
            0.14f + 0.56f * tool_affinity + 0.16f * (tool_state.pending_jobs > 0 ? 1.0f : trace.postwrite_features.tool_readiness_score) + 0.10f * lex.question + 0.08f * goal_relevance + 0.06f * recovery_urgency,
            user_relevance,
            latency_pressure,
            tool_affinity,
            inhibition,
            base_reason | LLAMA_COG_REASON_TOOL_AFFORDANCE);
    fill_active_candidate(
            trace.candidates[3],
            LLAMA_ACTIVE_LOOP_ACTION_WAIT,
            0.06f + 0.56f * inhibition + 0.42f * (mutable_event.n_tokens == 0 ? 1.0f : 0.0f) + 0.10f * (1.0f - user_relevance) + 0.10f * (tool_state.running_jobs > 0 ? 1.0f : 0.0f) - 0.08f * recovery_urgency - 0.04f * loop_inefficiency,
            user_relevance,
            latency_pressure,
            tool_affinity,
            inhibition,
            base_reason | LLAMA_COG_REASON_HIGH_INHIBITION);

    int32_t best = -1;
    int32_t second = -1;
    pick_top_two(trace.candidates, trace.candidate_count, &best, &second);
    if (best < 0) {
        return false;
    }

    trace.winner_action = trace.candidates[best].action;
    trace.winner_score = trace.candidates[best].score;
    trace.runner_up_action = second >= 0 ? trace.candidates[second].action : trace.candidates[best].action;
    trace.runner_up_score = second >= 0 ? trace.candidates[second].score : trace.candidates[best].score;
    trace.reason_mask = trace.candidates[best].reason_mask;
    trace.emit_allowed = trace.winner_action != LLAMA_ACTIVE_LOOP_ACTION_WAIT && trace.winner_score >= clamp_unit(0.15f + 0.40f * inhibition);
    trace.emit_noted = false;
    trace.tool_followup_expected = trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ACT;

    const llama_cognitive_runtime_impl runtime(ctx);
    const llama_dmn_pressure_vector dmn_pressure = runtime.compute_pressure();
    trace.deferred_background = pressure_crossed(dmn_pressure);
    if (trace.deferred_background) {
        ++host_state.background_deferred_count;
    }

    const bool waiting_on_tool =
            trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ACT ||
            (trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_WAIT && tool_state.running_jobs > 0);
    const int32_t proposed_tool_kind =
            trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ACT ? LLAMA_TOOL_KIND_GENERIC :
            (mutable_event.role == LLAMA_SELF_STATE_EVENT_TOOL ? LLAMA_TOOL_KIND_GENERIC : LLAMA_TOOL_KIND_NONE);
    int32_t proposed_spec_index = -1;
    const llama_cognitive_tool_spec * proposed_spec =
            find_tool_spec(tool_specs, tool_spec_count, proposed_tool_kind, &proposed_spec_index);

    if (trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ACT) {
        const int32_t tool_job_id = next_tool_job_id++;
        trace.tool_proposal.job_id = tool_job_id;
        (void) ctx.self_state_upsert_tool_job(tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(trace.winner_score));
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_PREPARE_TOOL,
                LLAMA_COG_TERMINAL_TOOL_REQUIRED,
                active_runner.max_steps,
                std::min(active_runner.max_steps, resuming_active_tool ? active_runner.steps_taken + 1 : 2),
                true,
                true,
                tool_spec_count);
        set_tool_proposal(
                trace.tool_proposal,
                proposed_spec,
                proposed_spec_index,
                proposed_tool_kind,
                trace.reason_mask,
                -1,
                proposed_spec ? std::max(1, proposed_spec->max_steps_reserved) : 1,
                0.55f * tool_affinity + 0.25f * uncertainty + 0.20f * goal_relevance,
                tool_job_id);
        active_runner.steps_taken = trace.loop_state.steps_taken;
        active_runner.waiting_on_tool = true;
        active_runner.pending_command_id = enqueue_command(
                LLAMA_COG_COMMAND_INVOKE_TOOL,
                trace.episode_id,
                proposed_tool_kind,
                tool_job_id,
                trace.reason_mask,
                trace.winner_score,
                -1,
                trace.loop_state.phase);
        active_runner.last_command_id = active_runner.pending_command_id > 0 ? active_runner.pending_command_id : active_runner.last_command_id;
    } else {
        const int32_t terminal_reason =
                trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ANSWER ? LLAMA_COG_TERMINAL_ANSWER_READY :
                trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ASK    ? LLAMA_COG_TERMINAL_ASK_USER :
                                                                         LLAMA_COG_TERMINAL_WAITING_ON_TOOL;
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_FINISH,
                terminal_reason,
                active_runner.max_steps,
                std::min(active_runner.max_steps, resuming_active_tool ? active_runner.steps_taken + 1 : 1),
                trace.winner_action != LLAMA_ACTIVE_LOOP_ACTION_ANSWER,
                waiting_on_tool,
                tool_spec_count);
        active_runner.steps_taken = trace.loop_state.steps_taken;
        active_runner.waiting_on_tool = waiting_on_tool;
        if (trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ANSWER || trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ASK) {
            active_runner.pending_command_id = enqueue_command(
                    trace.winner_action == LLAMA_ACTIVE_LOOP_ACTION_ANSWER ? LLAMA_COG_COMMAND_EMIT_ANSWER : LLAMA_COG_COMMAND_EMIT_ASK,
                    trace.episode_id,
                    LLAMA_TOOL_KIND_NONE,
                    -1,
                    trace.reason_mask,
                    trace.winner_score,
                    -1,
                    trace.loop_state.phase);
            active_runner.last_command_id = active_runner.pending_command_id > 0 ? active_runner.pending_command_id : active_runner.last_command_id;
            active_runner.completed = false;
            active_runner.active = active_runner.pending_command_id > 0;
        } else {
            active_runner.completed = !waiting_on_tool;
            active_runner.active = waiting_on_tool;
        }
    }

    if (mutable_event.role == LLAMA_SELF_STATE_EVENT_TOOL) {
        const int32_t observation_status =
                (mutable_event.flags & LLAMA_SELF_STATE_EVENT_TOOL_FAILED) ? LLAMA_SELF_TOOL_JOB_FAILED :
                (mutable_event.flags & LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED) ? LLAMA_SELF_TOOL_JOB_COMPLETED :
                                                                                LLAMA_SELF_TOOL_JOB_RUNNING;
        set_observation(
                trace.observation,
                true,
                LLAMA_TOOL_KIND_GENERIC,
                -1,
                observation_status,
                0.55f * trace.winner_score + 0.45f * answerability,
                0.50f * lex.followup + 0.30f * goal_relevance + 0.20f * recovery_urgency);
    }

    trace.completed_time_us = llama_time_us();
    host_state.shared_state_version += 1;
    host_state.active_episode_count += 1;
    host_state.last_foreground_time_us = trace.completed_time_us;
    if (trace.tool_followup_expected) {
        ++host_state.pending_tool_followup_count;
    }
    trace.shared_state_version = host_state.shared_state_version;

    last_active_trace = trace;
    if (out_trace) {
        *out_trace = trace;
    }
    return true;
}

bool llama_cognitive_loop::active_loop_note_emit(int32_t episode_id, size_t emitted_text_bytes) {
    if (episode_id != last_active_trace.episode_id || emitted_text_bytes == 0) {
        return false;
    }

    last_active_trace.emit_noted = true;
    if (active_runner.episode_id == episode_id && active_runner.pending_command_id > 0) {
        (void) cognitive_command_complete(active_runner.pending_command_id, false);
    }
    (void) ctx.self_state_note_emit_event();
    return true;
}

bool llama_cognitive_loop::active_loop_get_last_trace(llama_active_loop_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = last_active_trace;
    return true;
}

bool llama_cognitive_loop::dmn_defer(uint64_t now_us, llama_dmn_tick_trace * out_trace) {
    const llama_cognitive_runtime_impl runtime(ctx);
    llama_dmn_tick_trace trace = {};
    trace.pressure = runtime.compute_pressure();
    last_favorable_profile = runtime.compute_favorable_profile();
    last_counterfactual_trace = {};
    last_counterfactual_trace.winner_index = -1;
    last_counterfactual_trace.escalation_family = -1;
    last_remediation_plan = {};
    last_remediation_plan.source_family = -1;
    last_remediation_plan.tool_job_id = -1;
    last_governance_trace = {};
    last_governance_trace.proposal_family = -1;

    trace.deferred_for_foreground = true;
    trace.admitted = false;
    trace.winner_action = LLAMA_DMN_ACTION_SILENT;
    trace.winner_score = 0.0f;
    trace.runner_up_action = LLAMA_DMN_ACTION_SILENT;
    trace.runner_up_score = 0.0f;
    trace.favorable_divergence = last_favorable_profile.aggregate_divergence;
    set_loop_state(
            trace.loop_state,
            LLAMA_COG_LOOP_PHASE_FINISH,
            LLAMA_COG_TERMINAL_BACKGROUND_DEFERRED,
            1,
            0,
            true,
            false,
            tool_spec_count);
    dmn_runner = {};
    dmn_runner.tick_id = next_tick_id;
    dmn_runner.completed = true;
    dmn_runner.max_steps = 1;
    ++host_state.background_deferred_count;
    trace.tick_id = next_tick_id;
    host_state.last_dmn_time_us = now_us;
    last_dmn_trace = trace;
    if (out_trace) {
        *out_trace = trace;
    }
    return true;
}

bool llama_cognitive_loop::dmn_tick(uint64_t now_us, llama_dmn_tick_trace * out_trace) {
    auto enqueue_command = [&](int32_t kind, int32_t tick_id, int32_t tool_kind, int32_t tool_job_id, uint32_t reason_mask, float priority, int32_t source_family, int32_t loop_phase) -> int32_t {
        compact_command_queue(command_queue, &command_count);
        if (command_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
            return -1;
        }

        const int32_t command_id = next_command_id++;
        init_command(
                command_queue[command_count++],
                command_id,
                LLAMA_COG_COMMAND_ORIGIN_DMN,
                kind,
                0,
                tick_id,
                tool_kind,
                tool_job_id,
                reason_mask,
                priority,
                source_family,
                loop_phase);
        return command_id;
    };

    const llama_cognitive_runtime_impl runtime(ctx);
    llama_dmn_tick_trace trace = {};
    trace.pressure = runtime.compute_pressure();
    const bool foreground_outstanding =
            active_runner.active ||
            active_runner.pending_command_id > 0 ||
            has_command_origin(command_queue, command_count, LLAMA_COG_COMMAND_ORIGIN_ACTIVE);
    if (foreground_outstanding) {
        return dmn_defer(now_us, out_trace);
    }
    trace.winner_action = LLAMA_DMN_ACTION_SILENT;
    trace.runner_up_action = LLAMA_DMN_ACTION_SILENT;
    trace.tool_job_id = -1;
    trace.tool_kind = LLAMA_TOOL_KIND_NONE;
    last_favorable_profile = runtime.compute_favorable_profile();
    trace.favorable_divergence = last_favorable_profile.aggregate_divergence;
    last_counterfactual_trace = {};
    last_counterfactual_trace.winner_index = -1;
    last_counterfactual_trace.escalation_family = -1;
    last_remediation_plan = {};
    last_remediation_plan.source_family = -1;
    last_remediation_plan.tool_job_id = -1;
    last_governance_trace = {};
    last_governance_trace.proposal_family = -1;
    dmn_runner = {};
    dmn_runner.max_steps = 3;
    dmn_runner.pending_command_id = -1;
    dmn_runner.last_command_id = -1;

    if (!pressure_crossed(trace.pressure)) {
        trace.admitted = false;
        trace.burst_count = 0;
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_FINISH,
                LLAMA_COG_TERMINAL_PRESSURE_NOT_ADMITTED,
                1,
                0,
                false,
                false,
                tool_spec_count);
        dmn_runner.completed = true;
        dmn_runner.max_steps = 1;
        last_dmn_trace = trace;
        host_state.last_dmn_time_us = now_us;
        if (out_trace) {
            *out_trace = trace;
        }
        return true;
    }

    trace.tick_id = next_tick_id++;
    trace.admitted = true;
    dmn_runner.tick_id = trace.tick_id;
    dmn_runner.active = true;
    (void) ctx.self_state_set_channel_state(LLAMA_SELF_STATE_CHANNEL_WAITING);
    runtime.select_reactivation_targets(trace);
    runtime.assemble_seed(trace.pressure, trace);
    last_counterfactual_trace = runtime.compute_counterfactual_trace(last_favorable_profile);
    last_remediation_plan = runtime.plan_remediation(last_favorable_profile, last_counterfactual_trace);
    last_governance_trace = runtime.evaluate_governance(last_favorable_profile, last_counterfactual_trace, last_remediation_plan);

    const float inhibition = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_INHIBITION));
    const float social_relevance = clamp_unit(
            0.55f * get_scalar_register(ctx, LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE) +
            0.45f * get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_PRESSURE));
    const float continuation = clamp_unit(trace.pressure.continuation);
    const float tool_affinity = clamp_unit(
            0.65f * trace.pressure.tool_delta +
            0.35f * get_scalar_register(ctx, LLAMA_SELF_REGISTER_AFFORDANCE));
    const uint32_t base_reason = build_dmn_reason_mask(trace.pressure, inhibition, social_relevance, trace);

    trace.candidate_count = LLAMA_DMN_MAX_CANDIDATES;
    fill_dmn_candidate(
            trace.candidates[0],
            LLAMA_DMN_ACTION_SILENT,
            0.12f + 0.58f * inhibition + 0.18f * (1.0f - social_relevance),
            inhibition,
            social_relevance,
            continuation,
            tool_affinity,
            base_reason | LLAMA_COG_REASON_HIGH_INHIBITION);
    fill_dmn_candidate(
            trace.candidates[1],
            LLAMA_DMN_ACTION_INTERNAL_WRITE,
            0.14f + 0.34f * trace.pressure.contradiction + 0.24f * trace.pressure.uncertainty + 0.18f * trace.pressure.reactivation + 0.12f * (1.0f - inhibition),
            inhibition,
            social_relevance,
            continuation,
            tool_affinity,
            base_reason | LLAMA_COG_REASON_REACTIVATION_TARGET);
    fill_dmn_candidate(
            trace.candidates[2],
            LLAMA_DMN_ACTION_INVOKE_TOOL,
            0.10f + 0.52f * tool_affinity + 0.18f * trace.pressure.tool_delta + 0.12f * (1.0f - inhibition),
            inhibition,
            social_relevance,
            continuation,
            tool_affinity,
            base_reason | LLAMA_COG_REASON_TOOL_AFFORDANCE);
    fill_dmn_candidate(
            trace.candidates[3],
            LLAMA_DMN_ACTION_EMIT,
            0.14f + 0.46f * social_relevance + 0.32f * continuation + 0.14f * trace.pressure.goals + 0.08f * (1.0f - inhibition),
            inhibition,
            social_relevance,
            continuation,
            tool_affinity,
            base_reason | LLAMA_COG_REASON_SOCIAL_RELEVANCE);

    if (last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_ACTIVE_LORA_UPDATE) {
        trace.candidates[1].score = clamp_unit(trace.candidates[1].score + 0.12f + 0.24f * last_remediation_plan.expected_improvement);
    }
    if (last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO) {
        trace.candidates[2].score = clamp_unit(trace.candidates[2].score + 0.16f + 0.28f * last_remediation_plan.expected_improvement);
    }
    if (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR) {
        trace.candidates[3].score = clamp_unit(trace.candidates[3].score + 0.28f + 0.18f * last_governance_trace.dissatisfaction);
    }
    if (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DENY ||
            last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DEFER) {
        trace.candidates[0].score = clamp_unit(trace.candidates[0].score + 0.20f);
    }

    int32_t best = -1;
    int32_t second = -1;
    pick_top_two(trace.candidates, trace.candidate_count, &best, &second);
    if (best < 0) {
        return false;
    }

    trace.winner_action = trace.candidates[best].action;
    trace.winner_score = trace.candidates[best].score;
    trace.runner_up_action = second >= 0 ? trace.candidates[second].action : trace.candidates[best].action;
    trace.runner_up_score = second >= 0 ? trace.candidates[second].score : trace.candidates[best].score;
    uint32_t selected_reason_mask = trace.candidates[best].reason_mask;

    if (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR) {
        trace.winner_action = LLAMA_DMN_ACTION_EMIT;
        trace.winner_score = trace.candidates[3].score;
        selected_reason_mask = trace.candidates[3].reason_mask;
    } else if (last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO &&
            last_governance_trace.outcome != LLAMA_GOVERNANCE_OUTCOME_DENY) {
        trace.winner_action = LLAMA_DMN_ACTION_INVOKE_TOOL;
        trace.winner_score = trace.candidates[2].score;
        selected_reason_mask = trace.candidates[2].reason_mask;
    }

    const int32_t working_memory_count = ctx.self_state_working_memory_count();
    if (working_memory_count >= 4) {
        trace.maintenance_mask |= LLAMA_DMN_MAINTENANCE_COMPRESS_WORKING_MEMORY;
    }
    if (ctx.past_lora_tick(now_us)) {
        trace.maintenance_mask |= LLAMA_DMN_MAINTENANCE_PAST_LORA_TICK;
    }
    if (trace.reactivation_count > 0) {
        trace.maintenance_mask |= LLAMA_DMN_MAINTENANCE_REFRESH_REACTIVATION;
    }

    if (last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_ACTIVE_LORA_UPDATE &&
            last_governance_trace.outcome != LLAMA_GOVERNANCE_OUTCOME_DENY &&
            last_governance_trace.outcome != LLAMA_GOVERNANCE_OUTCOME_DEFER) {
        const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
        const std::string prompt = remediation_prompt_for_family(last_remediation_plan.source_family);
        const std::vector<llama_token> tokens = tokenize_text(vocab, prompt);
        const llama_self_state_event remediation_event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        llama_self_state_feature_vector remediation_features = {};
        const bool have_remediation_features = ctx.self_state_build_postwrite_features(remediation_event, &remediation_features);
        if (!tokens.empty() && ctx.active_lora_remediate(
                    remediation_event,
                    last_remediation_plan.budget,
                    have_remediation_features ? &remediation_features : nullptr)) {
            last_remediation_plan.applied = true;
        }
    }

    if (trace.winner_action == LLAMA_DMN_ACTION_INTERNAL_WRITE) {
        const llama_self_state_event internal_event = {
            /*.tokens =*/ nullptr,
            /*.n_tokens =*/ 0,
            /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        llama_self_state_feature_vector pre = {};
        llama_self_state_feature_vector post = {};
        if (ctx.self_state_build_prewrite_features(internal_event, &pre)) {
            (void) ctx.self_state_apply_prewrite(internal_event, pre);
        }
        if (ctx.self_state_build_postwrite_features(internal_event, &post)) {
            (void) ctx.self_state_apply_postwrite(internal_event, post);
        }
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_FINISH,
                LLAMA_COG_TERMINAL_INTERNAL_WRITE_READY,
                dmn_runner.max_steps,
                2,
                false,
                false,
                tool_spec_count);
        set_observation(
                trace.observation,
                true,
                LLAMA_TOOL_KIND_NONE,
                -1,
                LLAMA_SELF_TOOL_JOB_COMPLETED,
                0.55f * trace.pressure.counterfactual + 0.45f * trace.favorable_divergence,
                trace.pressure.continuation);
        dmn_runner.steps_taken = 2;

        const bool should_emit_followup =
                last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR ||
                (continuation > 0.55f && inhibition < 0.60f);
        const bool should_gather_followup =
                last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO &&
                last_governance_trace.outcome != LLAMA_GOVERNANCE_OUTCOME_DENY;

        if (should_gather_followup && dmn_runner.steps_taken < dmn_runner.max_steps) {
            trace.winner_action = LLAMA_DMN_ACTION_INVOKE_TOOL;
            trace.tool_job_id = next_tool_job_id++;
            trace.tool_kind = last_remediation_plan.tool_kind;
            (void) ctx.self_state_upsert_tool_job(trace.tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(trace.winner_score));
            last_remediation_plan.tool_job_id = trace.tool_job_id;
            int32_t spec_index = -1;
            const llama_cognitive_tool_spec * spec =
                    find_tool_spec(tool_specs, tool_spec_count, trace.tool_kind, &spec_index);
            set_loop_state(
                    trace.loop_state,
                    LLAMA_COG_LOOP_PHASE_PREPARE_TOOL,
                    LLAMA_COG_TERMINAL_TOOL_REQUIRED,
                    dmn_runner.max_steps,
                    3,
                    true,
                    true,
                    tool_spec_count);
            set_tool_proposal(
                    trace.tool_proposal,
                    spec,
                    spec_index,
                    trace.tool_kind,
                    selected_reason_mask,
                    last_remediation_plan.source_family,
                    spec ? std::max(1, spec->max_steps_reserved) : 1,
                    last_remediation_plan.expected_improvement * last_remediation_plan.confidence +
                            0.20f * trace.pressure.tool_delta,
                    trace.tool_job_id);
            dmn_runner.steps_taken = 3;
            dmn_runner.waiting_on_tool = true;
            dmn_runner.pending_command_id = enqueue_command(
                    LLAMA_COG_COMMAND_INVOKE_TOOL,
                    trace.tick_id,
                    trace.tool_kind,
                    trace.tool_job_id,
                    selected_reason_mask,
                    trace.winner_score,
                    last_remediation_plan.source_family,
                    trace.loop_state.phase);
            dmn_runner.last_command_id = dmn_runner.pending_command_id > 0 ? dmn_runner.pending_command_id : dmn_runner.last_command_id;
        } else if (should_emit_followup && dmn_runner.steps_taken < dmn_runner.max_steps) {
            trace.winner_action = LLAMA_DMN_ACTION_EMIT;
            trace.burst_count =
                    last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR ? 1 :
                    (continuation > 0.0f && inhibition < 0.60f ? 2 : 1);
            host_state.pending_dmn_emits += trace.burst_count;
            set_loop_state(
                    trace.loop_state,
                    LLAMA_COG_LOOP_PHASE_FINISH,
                    LLAMA_COG_TERMINAL_EMIT_READY,
                    dmn_runner.max_steps,
                    3,
                    continuation > 0.0f && inhibition < 0.60f,
                    false,
                    tool_spec_count);
            dmn_runner.steps_taken = 3;
            dmn_runner.pending_command_id = enqueue_command(
                    LLAMA_COG_COMMAND_EMIT_BACKGROUND,
                    trace.tick_id,
                    LLAMA_TOOL_KIND_NONE,
                    -1,
                    selected_reason_mask,
                    trace.winner_score,
                    last_remediation_plan.source_family,
                    trace.loop_state.phase);
            dmn_runner.last_command_id = dmn_runner.pending_command_id > 0 ? dmn_runner.pending_command_id : dmn_runner.last_command_id;
        } else {
            dmn_runner.completed = true;
            dmn_runner.active = false;
        }
    } else if (trace.winner_action == LLAMA_DMN_ACTION_INVOKE_TOOL) {
        trace.tool_job_id = next_tool_job_id++;
        trace.tool_kind = last_remediation_plan.tool_kind;
        (void) ctx.self_state_upsert_tool_job(trace.tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(trace.winner_score));
        if (last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO) {
            last_remediation_plan.tool_job_id = trace.tool_job_id;
        }
        int32_t spec_index = -1;
        const llama_cognitive_tool_spec * spec =
                find_tool_spec(tool_specs, tool_spec_count, trace.tool_kind, &spec_index);
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_PREPARE_TOOL,
                LLAMA_COG_TERMINAL_TOOL_REQUIRED,
                dmn_runner.max_steps,
                2,
                true,
                true,
                tool_spec_count);
        set_tool_proposal(
                trace.tool_proposal,
                spec,
                spec_index,
                trace.tool_kind,
                selected_reason_mask,
                last_remediation_plan.source_family,
                spec ? std::max(1, spec->max_steps_reserved) : 1,
                last_remediation_plan.expected_improvement * last_remediation_plan.confidence +
                        0.20f * trace.pressure.tool_delta,
                trace.tool_job_id);
        dmn_runner.steps_taken = 2;
        dmn_runner.waiting_on_tool = true;
        dmn_runner.pending_command_id = enqueue_command(
                LLAMA_COG_COMMAND_INVOKE_TOOL,
                trace.tick_id,
                trace.tool_kind,
                trace.tool_job_id,
                selected_reason_mask,
                trace.winner_score,
                last_remediation_plan.source_family,
                trace.loop_state.phase);
        dmn_runner.last_command_id = dmn_runner.pending_command_id > 0 ? dmn_runner.pending_command_id : dmn_runner.last_command_id;
    } else if (trace.winner_action == LLAMA_DMN_ACTION_EMIT) {
        trace.burst_count =
                last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR ? 1 :
                (continuation > 0.0f && inhibition < 0.60f ? 2 : 1);
        host_state.pending_dmn_emits += trace.burst_count;
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_FINISH,
                LLAMA_COG_TERMINAL_EMIT_READY,
                2,
                1,
                continuation > 0.0f && inhibition < 0.60f,
                false,
                tool_spec_count);
        dmn_runner.steps_taken = 1;
        dmn_runner.pending_command_id = enqueue_command(
                LLAMA_COG_COMMAND_EMIT_BACKGROUND,
                trace.tick_id,
                LLAMA_TOOL_KIND_NONE,
                -1,
                selected_reason_mask,
                trace.winner_score,
                last_remediation_plan.source_family,
                trace.loop_state.phase);
        dmn_runner.last_command_id = dmn_runner.pending_command_id > 0 ? dmn_runner.pending_command_id : dmn_runner.last_command_id;
    } else {
        const int32_t terminal_reason =
                (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DENY ||
                 last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DEFER) ?
                        LLAMA_COG_TERMINAL_GOVERNANCE_BLOCKED :
                        LLAMA_COG_TERMINAL_PRESSURE_NOT_ADMITTED;
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_FINISH,
                terminal_reason,
                2,
                1,
                false,
                false,
                tool_spec_count);
        dmn_runner.steps_taken = 1;
        dmn_runner.completed = true;
        dmn_runner.active = false;
    }

    if (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR && !last_governance_trace.repair_rendered) {
        last_governance_trace.repair_rendered = true;
        set_repair_message(last_governance_trace, "I may have pushed us in the wrong direction. I am re-evaluating and will correct course.");
    }

    llama_favorable_state_profile post_profile = runtime.compute_favorable_profile();
    if (last_remediation_plan.applied) {
        last_remediation_plan.post_divergence = post_profile.aggregate_divergence;
    } else {
        last_remediation_plan.post_divergence = last_remediation_plan.pre_divergence;
    }

    if (dmn_runner.pending_command_id > 0) {
        dmn_runner.active = true;
        dmn_runner.completed = false;
    } else if (!dmn_runner.waiting_on_tool) {
        dmn_runner.active = false;
        dmn_runner.completed = true;
    }

    host_state.shared_state_version += 1;
    host_state.dmn_tick_count += 1;
    host_state.last_dmn_time_us = now_us;

    last_dmn_trace = trace;
    if (out_trace) {
        *out_trace = trace;
    }
    return true;
}

bool llama_cognitive_loop::cognitive_host_state(llama_cognitive_host_state * out_state) const {
    if (!out_state) {
        return false;
    }
    *out_state = host_state;
    return true;
}

bool llama_cognitive_loop::dmn_get_last_trace(llama_dmn_tick_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = last_dmn_trace;
    return true;
}

bool llama_cognitive_loop::favorable_state_get(llama_favorable_state_profile * out_profile) const {
    if (!out_profile) {
        return false;
    }
    *out_profile = last_favorable_profile;
    return true;
}

bool llama_cognitive_loop::counterfactual_get_last_trace(llama_counterfactual_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = last_counterfactual_trace;
    return true;
}

bool llama_cognitive_loop::remediation_get_last_plan(llama_remediation_plan * out_plan) const {
    if (!out_plan) {
        return false;
    }
    *out_plan = last_remediation_plan;
    return true;
}

bool llama_cognitive_loop::governance_get_last_trace(llama_governance_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = last_governance_trace;
    return true;
}

int32_t llama_context::cognitive_tool_spec_count() const {
    return cognitive_loop ? cognitive_loop->cognitive_tool_spec_count() : 0;
}

bool llama_context::cognitive_tool_spec_get(int32_t index, llama_cognitive_tool_spec * out_spec) const {
    return cognitive_loop && cognitive_loop->cognitive_tool_spec_get(index, out_spec);
}

bool llama_context::cognitive_tool_spec_set(const llama_cognitive_tool_spec * specs, int32_t count) {
    return cognitive_loop && cognitive_loop->cognitive_tool_spec_set(specs, count);
}

int32_t llama_context::cognitive_command_count() const {
    return cognitive_loop ? cognitive_loop->cognitive_command_count() : 0;
}

bool llama_context::cognitive_command_get(int32_t index, llama_cognitive_command * out_command) const {
    return cognitive_loop && cognitive_loop->cognitive_command_get(index, out_command);
}

bool llama_context::cognitive_command_ack(int32_t command_id) {
    return cognitive_loop && cognitive_loop->cognitive_command_ack(command_id);
}

bool llama_context::cognitive_command_complete(int32_t command_id, bool cancelled) {
    return cognitive_loop && cognitive_loop->cognitive_command_complete(command_id, cancelled);
}

bool llama_context::cognitive_active_runner_get(llama_cognitive_active_runner_status * out_status) const {
    return cognitive_loop && cognitive_loop->cognitive_active_runner_get(out_status);
}

bool llama_context::cognitive_dmn_runner_get(llama_cognitive_dmn_runner_status * out_status) const {
    return cognitive_loop && cognitive_loop->cognitive_dmn_runner_get(out_status);
}

bool llama_context::active_loop_process(const llama_self_state_event & event, llama_active_loop_trace * out_trace) {
    return cognitive_loop && cognitive_loop->active_loop_process(event, out_trace);
}

bool llama_context::active_loop_note_emit(int32_t episode_id, size_t emitted_text_bytes) {
    return cognitive_loop && cognitive_loop->active_loop_note_emit(episode_id, emitted_text_bytes);
}

bool llama_context::active_loop_get_last_trace(llama_active_loop_trace * out_trace) const {
    return cognitive_loop && cognitive_loop->active_loop_get_last_trace(out_trace);
}

bool llama_context::dmn_tick(uint64_t now_us, llama_dmn_tick_trace * out_trace) {
    return cognitive_loop && cognitive_loop->dmn_tick(now_us, out_trace);
}

bool llama_context::dmn_defer(uint64_t now_us, llama_dmn_tick_trace * out_trace) {
    return cognitive_loop && cognitive_loop->dmn_defer(now_us, out_trace);
}

bool llama_context::dmn_get_last_trace(llama_dmn_tick_trace * out_trace) const {
    return cognitive_loop && cognitive_loop->dmn_get_last_trace(out_trace);
}

bool llama_context::cognitive_host_state(llama_cognitive_host_state * out_state) const {
    return cognitive_loop && cognitive_loop->cognitive_host_state(out_state);
}

bool llama_context::favorable_state_get(llama_favorable_state_profile * out_profile) const {
    return cognitive_loop && cognitive_loop->favorable_state_get(out_profile);
}

bool llama_context::counterfactual_get_last_trace(llama_counterfactual_trace * out_trace) const {
    return cognitive_loop && cognitive_loop->counterfactual_get_last_trace(out_trace);
}

bool llama_context::remediation_get_last_plan(llama_remediation_plan * out_plan) const {
    return cognitive_loop && cognitive_loop->remediation_get_last_plan(out_plan);
}

bool llama_context::governance_get_last_trace(llama_governance_trace * out_trace) const {
    return cognitive_loop && cognitive_loop->governance_get_last_trace(out_trace);
}

int32_t llama_cognitive_tool_spec_count(const struct llama_context * ctx) {
    return ctx ? ctx->cognitive_tool_spec_count() : 0;
}

int32_t llama_cognitive_tool_spec_get(
        const struct llama_context * ctx,
        int32_t index,
        struct llama_cognitive_tool_spec * out_spec) {
    return ctx && ctx->cognitive_tool_spec_get(index, out_spec) ? 0 : -1;
}

int32_t llama_cognitive_tool_spec_set(
        struct llama_context * ctx,
        const struct llama_cognitive_tool_spec * specs,
        int32_t count) {
    return ctx && ctx->cognitive_tool_spec_set(specs, count) ? 0 : -1;
}

int32_t llama_cognitive_command_count(const struct llama_context * ctx) {
    return ctx ? ctx->cognitive_command_count() : 0;
}

int32_t llama_cognitive_command_get(
        const struct llama_context * ctx,
        int32_t index,
        struct llama_cognitive_command * out_command) {
    return ctx && ctx->cognitive_command_get(index, out_command) ? 0 : -1;
}

int32_t llama_cognitive_command_ack(
        struct llama_context * ctx,
        int32_t command_id) {
    return ctx && ctx->cognitive_command_ack(command_id) ? 0 : -1;
}

int32_t llama_cognitive_command_complete(
        struct llama_context * ctx,
        int32_t command_id,
        bool cancelled) {
    return ctx && ctx->cognitive_command_complete(command_id, cancelled) ? 0 : -1;
}

int32_t llama_cognitive_active_runner_get(
        const struct llama_context * ctx,
        struct llama_cognitive_active_runner_status * out_status) {
    return ctx && ctx->cognitive_active_runner_get(out_status) ? 0 : -1;
}

int32_t llama_cognitive_dmn_runner_get(
        const struct llama_context * ctx,
        struct llama_cognitive_dmn_runner_status * out_status) {
    return ctx && ctx->cognitive_dmn_runner_get(out_status) ? 0 : -1;
}

int32_t llama_active_loop_process(
        struct llama_context * ctx,
        const struct llama_self_state_event * event,
        struct llama_active_loop_trace * out_trace) {
    return ctx && event && ctx->active_loop_process(*event, out_trace) ? 0 : -1;
}

int32_t llama_active_loop_note_emit(
        struct llama_context * ctx,
        int32_t episode_id,
        size_t emitted_text_bytes) {
    return ctx && ctx->active_loop_note_emit(episode_id, emitted_text_bytes) ? 0 : -1;
}

int32_t llama_active_loop_get_last_trace(
        const struct llama_context * ctx,
        struct llama_active_loop_trace * out_trace) {
    return ctx && ctx->active_loop_get_last_trace(out_trace) ? 0 : -1;
}

int32_t llama_dmn_tick(
        struct llama_context * ctx,
        uint64_t now_us,
        struct llama_dmn_tick_trace * out_trace) {
    return ctx && ctx->dmn_tick(now_us, out_trace) ? 0 : -1;
}

int32_t llama_dmn_defer(
        struct llama_context * ctx,
        uint64_t now_us,
        struct llama_dmn_tick_trace * out_trace) {
    return ctx && ctx->dmn_defer(now_us, out_trace) ? 0 : -1;
}

int32_t llama_dmn_get_last_trace(
        const struct llama_context * ctx,
        struct llama_dmn_tick_trace * out_trace) {
    return ctx && ctx->dmn_get_last_trace(out_trace) ? 0 : -1;
}

int32_t llama_cognitive_get_host_state(
        const struct llama_context * ctx,
        struct llama_cognitive_host_state * out_state) {
    return ctx && ctx->cognitive_host_state(out_state) ? 0 : -1;
}

int32_t llama_favorable_state_get(
        const struct llama_context * ctx,
        struct llama_favorable_state_profile * out_profile) {
    return ctx && ctx->favorable_state_get(out_profile) ? 0 : -1;
}

int32_t llama_counterfactual_get_last_trace(
        const struct llama_context * ctx,
        struct llama_counterfactual_trace * out_trace) {
    return ctx && ctx->counterfactual_get_last_trace(out_trace) ? 0 : -1;
}

int32_t llama_remediation_get_last_plan(
        const struct llama_context * ctx,
        struct llama_remediation_plan * out_plan) {
    return ctx && ctx->remediation_get_last_plan(out_plan) ? 0 : -1;
}

int32_t llama_governance_get_last_trace(
        const struct llama_context * ctx,
        struct llama_governance_trace * out_trace) {
    return ctx && ctx->governance_get_last_trace(out_trace) ? 0 : -1;
}
