#include "llama-cognitive-loop.h"

#include "llama.h"
#include "llama-active-lora.h"
#include "llama-adapter.h"
#include "llama-context.h"
#include "llama-sampler.h"
#include "llama-vocab.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float LLAMA_DMN_PRESSURE_THRESHOLD = 0.24f;
constexpr float LLAMA_TEMPORAL_SELF_TRIGGER_THRESHOLD = 0.58f;
constexpr uint64_t LLAMA_TEMPORAL_SELF_TRIGGER_COOLDOWN_US = 2ULL * 60ULL * 1000000ULL;
constexpr int32_t LLAMA_COG_LOOP_UNBOUNDED_MAX_STEPS = std::numeric_limits<int32_t>::max();

const char * vicuna_core_system_prompt_default_text() {
    return "You are an experimental intelligence designed to self-regulate, learn, self-improve, and be useful.";
}

float clamp_unit(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

float clamp_signed_unit(float value) {
    return std::min(1.0f, std::max(-1.0f, value));
}

std::string normalize_piece(const std::string & piece) {
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

bool contains_any(const std::string & piece, const char * const * patterns, size_t n_patterns) {
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
    float web = 0.0f;
};

std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text);
llama_process_functional_signature make_process_signature(
        int32_t loop_origin,
        int32_t family,
        int32_t microphase,
        const llama_cognitive_plan_trace * plan,
        const llama_cognitive_tool_spec * tool_spec,
        int32_t transient_source_id);
int32_t primary_functional_family(const llama_functional_activation_decision & decision);
void set_bounded_cstr(char * dst, size_t dst_size, const char * src);
bool is_tavily_web_search_capability(const llama_cognitive_tool_spec * spec);
std::string trim_ascii(const std::string & text);

lexical_signals analyze_event_lexicon(const llama_vocab * vocab, const llama_self_state_event & event) {
    lexical_signals out = {};
    if (!vocab || !event.tokens) {
        return out;
    }

    static const char * const question_terms[] = { "?", "what", "which", "clarify", "why", "how" };
    static const char * const tool_terms[] = { "search", "find", "lookup", "look", "inspect", "tool", "call", "run", "fetch" };
    static const char * const web_terms[] = { "web", "internet", "online", "latest", "current", "today", "news", "president" };
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
        if (contains_any(piece, web_terms, sizeof(web_terms)/sizeof(web_terms[0]))) {
            out.web += 0.30f;
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
    out.web = clamp_unit(out.web);
    out.followup = clamp_unit(out.followup);
    out.uncertainty = clamp_unit(out.uncertainty);
    return out;
}

float get_scalar_register(const llama_context & ctx, int32_t register_id) {
    llama_self_register_info info = {};
    if (!ctx.self_state_get_register(register_id, &info)) {
        return 0.0f;
    }
    return clamp_unit(info.scalar_value);
}

void fill_active_candidate(
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

template<typename T>
void pick_top_two(const T * candidates, int32_t candidate_count, int32_t * out_best, int32_t * out_second) {
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

bool pressure_crossed(const llama_dmn_pressure_vector & pressure) {
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

llama_self_updater_program load_updater_program(const llama_context & ctx) {
    llama_self_updater_program program = {};
    if (!ctx.self_state_get_updater_program(&program)) {
        program = llama_self_state_default_updater_program();
    }
    return program;
}

repair_signal_state compute_repair_signal(
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

uint32_t build_active_reason_mask(
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

uint32_t build_dmn_reason_mask(
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

float prediction_error_signal(const llama_self_prediction_error_trace & trace) {
    if (!trace.valid) {
        return 0.0f;
    }
    return clamp_unit(
            0.25f * std::fabs(trace.steps_error) +
            0.25f * std::fabs(trace.inference_cost_error) +
            0.20f * std::fabs(trace.satisfaction_error) +
            0.15f * std::fabs(trace.recovery_error) +
            0.15f * std::fabs(trace.goal_progress_error));
}

llama_functional_outcome_snapshot capture_functional_snapshot(
        const llama_context & ctx,
        const llama_favorable_state_profile & favorable) {
    llama_functional_outcome_snapshot out = {};
    out.favorable_divergence = clamp_unit(favorable.aggregate_divergence);
    out.user_satisfaction_risk = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK));
    out.goal_progress_pressure = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE));
    out.loop_inefficiency = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY));
    out.recovery_urgency = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_RECOVERY_URGENCY));
    out.answerability = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_ANSWERABILITY));
    out.preference_uncertainty = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY));

    llama_self_model_state_info model_state = {};
    if (ctx.self_state_get_model_state(&model_state)) {
        out.expected_steps_remaining = clamp_unit(model_state.horizons[LLAMA_SELF_HORIZON_SHORT].efficiency.expected_steps_remaining);
        out.expected_inference_cost_remaining = clamp_unit(model_state.horizons[LLAMA_SELF_HORIZON_SHORT].efficiency.expected_inference_cost_remaining);
    }

    return out;
}

llama_functional_activation_decision make_functional_policy_seed(
        int32_t loop_origin,
        int32_t microphase) {
    llama_functional_activation_decision decision = {};
    decision.loop_origin = loop_origin;
    decision.microphase = microphase;
    decision.top_family = -1;
    decision.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;

    auto set_family = [&](int32_t family, int32_t hold_unit, uint32_t hold_value, uint32_t reason_mask) {
        decision.eligible_mask |= (1ull << family);
        decision.hold_unit[family] = hold_unit;
        decision.hold_value[family] = hold_value;
        decision.reason_mask[family] = reason_mask;
    };

    switch (microphase) {
        case LLAMA_FUNCTIONAL_MICROPHASE_PLAN_DRAFT:
        case LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE:
        case LLAMA_FUNCTIONAL_MICROPHASE_PLAN_REVISE:
            set_family(
                    LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION,
                    LLAMA_FUNCTIONAL_HOLD_PHASE_EXIT,
                    1,
                    LLAMA_FUNCTIONAL_ROUTE_REASON_UNCERTAINTY |
                            LLAMA_FUNCTIONAL_ROUTE_REASON_TOOL_AFFINITY |
                            LLAMA_FUNCTIONAL_ROUTE_REASON_HOLD_CONTINUITY);
            break;
        case LLAMA_FUNCTIONAL_MICROPHASE_TOOL_CLASS_SELECTION:
        case LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP:
            set_family(
                    LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION,
                    LLAMA_FUNCTIONAL_HOLD_PHASE_EXIT,
                    1,
                    LLAMA_FUNCTIONAL_ROUTE_REASON_TOOL_AFFINITY |
                            LLAMA_FUNCTIONAL_ROUTE_REASON_HOLD_CONTINUITY);
            decision.top_family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
            [[fallthrough]];
        case LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION:
            set_family(
                    LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION,
                    LLAMA_FUNCTIONAL_HOLD_COMMANDS,
                    1,
                    LLAMA_FUNCTIONAL_ROUTE_REASON_UNCERTAINTY | LLAMA_FUNCTIONAL_ROUTE_REASON_TOOL_AFFINITY);
            break;
        case LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_GENERATE:
        case LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE:
            set_family(
                    LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL,
                    LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS,
                    2,
                    LLAMA_FUNCTIONAL_ROUTE_REASON_FAVORABLE_DIVERGENCE);
            break;
        case LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_COMPRESSION:
        case LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_AUDIT:
            set_family(
                    LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION,
                    LLAMA_FUNCTIONAL_HOLD_PHASE_EXIT,
                    1,
                    LLAMA_FUNCTIONAL_ROUTE_REASON_MEMORY_PRESSURE);
            break;
        case LLAMA_FUNCTIONAL_MICROPHASE_STATE_INTERPRET:
        case LLAMA_FUNCTIONAL_MICROPHASE_SELF_OBSERVE:
        case LLAMA_FUNCTIONAL_MICROPHASE_SELF_FORECAST:
        case LLAMA_FUNCTIONAL_MICROPHASE_POST_ACTION_REFLECTION:
        default:
            set_family(
                    LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION,
                    LLAMA_FUNCTIONAL_HOLD_PHASE_EXIT,
                    1,
                    LLAMA_FUNCTIONAL_ROUTE_REASON_UNCERTAINTY | LLAMA_FUNCTIONAL_ROUTE_REASON_PREDICTION_ERROR);
            break;
    }

    return decision;
}

llama_functional_activation_decision route_functional_activation(
        llama_context & ctx,
        int32_t loop_origin,
        int32_t microphase,
        const llama_functional_outcome_snapshot & snapshot,
        float uncertainty,
        float tool_affinity,
        float continuation,
        float memory_pressure,
        float planning_pressure = 0.0f,
        float plan_complexity = 0.0f,
        float plan_revision = 0.0f) {
    llama_functional_activation_decision policy_seed = make_functional_policy_seed(loop_origin, microphase);

    llama_self_model_state_info model_state = {};
    const bool have_model_state = ctx.self_state_get_model_state(&model_state);
    llama_functional_gating_observation observation = {};
    observation.loop_origin = loop_origin;
    observation.microphase = microphase;
    observation.eligible_mask = policy_seed.eligible_mask;
    observation.snapshot = snapshot;
    observation.uncertainty = uncertainty;
    observation.tool_affinity = tool_affinity;
    observation.continuation = continuation;
    observation.memory_pressure = memory_pressure;
    observation.recovery_urgency = snapshot.recovery_urgency;
    observation.prediction_error = have_model_state ? prediction_error_signal(model_state.prediction_error) : 0.0f;
    observation.planning_pressure = planning_pressure;
    observation.plan_complexity = plan_complexity;
    observation.plan_revision = plan_revision;
    observation.belief_summary = have_model_state ? model_state.belief_summary : llama_self_belief_summary {};
    observation.extension_summary = have_model_state ? model_state.extension_summary : llama_self_model_extension_summary {};
    llama_hard_memory_result hard_memory_result = {};
    observation.hard_memory_summary =
            ctx.hard_memory_get_last_result(&hard_memory_result) ?
                    hard_memory_result.retrieval_summary :
                    llama_hard_memory_retrieval_summary {};

    llama_functional_activation_decision decision = {};
    if (ctx.functional_lora_predict_activation(observation, policy_seed, &decision)) {
        return decision;
    }

    decision = policy_seed;
    decision.exploration_std = 0.0f;
    decision.allostatic_distance = snapshot.favorable_divergence;
    decision.gating_invocation_count = 0;
    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        if ((decision.eligible_mask & (1ull << family)) == 0) {
            continue;
        }
        decision.predicted_gains[family] = 1.0f;
        decision.gains[family] = 1.0f;
        decision.priority[family] = 1.0f;
        decision.activated_mask |= (1ull << family);
        if (decision.top_family < 0) {
            decision.top_family = family;
        }
    }
    return decision;
}

struct functional_delta_summary {
    float delta_favorable = 0.0f;
    float delta_user = 0.0f;
    float delta_goal = 0.0f;
    float delta_efficiency = 0.0f;
    float delta_recovery = 0.0f;
    float delta_answerability = 0.0f;
    float delta_preference = 0.0f;
};

functional_delta_summary compute_functional_deltas(
        const llama_functional_outcome_snapshot & before,
        const llama_functional_outcome_snapshot & after) {
    functional_delta_summary deltas = {};
    deltas.delta_favorable = clamp_signed_unit(before.favorable_divergence - after.favorable_divergence);
    deltas.delta_user = clamp_signed_unit(before.user_satisfaction_risk - after.user_satisfaction_risk);
    deltas.delta_goal = clamp_signed_unit(before.goal_progress_pressure - after.goal_progress_pressure);
    deltas.delta_efficiency = clamp_signed_unit(before.loop_inefficiency - after.loop_inefficiency);
    deltas.delta_recovery = clamp_signed_unit(before.recovery_urgency - after.recovery_urgency);
    deltas.delta_answerability = clamp_signed_unit(after.answerability - before.answerability);
    deltas.delta_preference = clamp_signed_unit(before.preference_uncertainty - after.preference_uncertainty);
    return deltas;
}

const llama_counterfactual_candidate * best_temporal_ablation_candidate(
        const llama_counterfactual_trace & trace) {
    const llama_counterfactual_candidate * best = nullptr;
    for (int32_t i = 0; i < trace.candidate_count; ++i) {
        const auto & candidate = trace.candidates[i];
        if (candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION) {
            continue;
        }
        if (!best) {
            best = &candidate;
            continue;
        }
        const float lhs = candidate.expected_improvement * candidate.confidence;
        const float rhs = best->expected_improvement * best->confidence;
        if (lhs > rhs + 1.0e-6f) {
            best = &candidate;
        }
    }
    return best;
}

int32_t select_functional_bias_family(const llama_favorable_state_profile & profile) {
    if (profile.priority_count <= 0) {
        return LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    }
    const int32_t top_dim = profile.dimensions[profile.priority_order[0]].dimension_id;
    switch (top_dim) {
        case LLAMA_FAVORABLE_DIM_TOOL_BACKLOG:
        case LLAMA_FAVORABLE_DIM_TOOL_READINESS:
            return LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
        case LLAMA_FAVORABLE_DIM_MEMORY_WRITE_PRIORITY:
        case LLAMA_FAVORABLE_DIM_REACTIVATION_PRIORITY:
            return LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION;
        case LLAMA_FAVORABLE_DIM_SOCIAL_TRUST:
        case LLAMA_FAVORABLE_DIM_SOCIAL_RECIPROCITY:
        case LLAMA_FAVORABLE_DIM_SOCIAL_DISSATISFACTION:
            return LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION;
        default:
            return LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    }
}

int32_t find_process_entry_slot_for_signature(
        llama_context & ctx,
        const llama_process_functional_signature & signature) {
    if (!signature.valid) {
        return -1;
    }
    for (int32_t entry_slot = 0; entry_slot < LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES; ++entry_slot) {
        llama_process_functional_entry_info info = {};
        if (!ctx.process_functional_entry_get(entry_slot, &info) || !info.valid) {
            continue;
        }
        if (info.signature.valid &&
                info.signature.family == signature.family &&
                info.signature.signature_hash == signature.signature_hash) {
            return entry_slot;
        }
    }
    return -1;
}

int32_t select_best_process_snapshot_slot(
        llama_context & ctx,
        int32_t entry_slot) {
    llama_functional_lora_snapshot_archive archive = {};
    if (!ctx.process_functional_snapshot_archive_get(entry_slot, &archive) || archive.count == 0) {
        return -1;
    }
    int32_t best_snapshot_slot = -1;
    for (int32_t snapshot_slot = 0; snapshot_slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++snapshot_slot) {
        const auto & item = archive.items[snapshot_slot];
        if (!item.valid) {
            continue;
        }
        if (best_snapshot_slot < 0) {
            best_snapshot_slot = snapshot_slot;
            continue;
        }
        const auto & best_item = archive.items[best_snapshot_slot];
        if (item.robustness_score > best_item.robustness_score + 1.0e-6f ||
                (std::fabs(item.robustness_score - best_item.robustness_score) <= 1.0e-6f &&
                 item.captured_at_us > best_item.captured_at_us)) {
            best_snapshot_slot = snapshot_slot;
        }
    }
    return best_snapshot_slot;
}

float score_functional_bias_candidate(
        llama_context & ctx,
        const llama_favorable_state_profile & profile,
        int32_t functional_target_kind,
        int32_t functional_family,
        int32_t process_entry_slot,
        int32_t replay_mode,
        int32_t snapshot_slot,
        float orthogonality,
        const llama_process_functional_signature * process_signature,
        float * out_fragility,
        float * out_concentration,
        float * out_robustness) {
    // Counterfactual functional replay stays explicit here on purpose:
    // score archived or perturbed candidates from typed replay metadata,
    // then apply the same bounded replay override surface used by serving.
    if (out_fragility) {
        *out_fragility = 0.0f;
    }
    if (out_concentration) {
        *out_concentration = 0.0f;
    }
    if (out_robustness) {
        *out_robustness = 0.0f;
    }

    const float aggregate = clamp_unit(profile.aggregate_divergence);
    float robustness = 0.45f + 0.35f * aggregate;
    float base = 0.04f + 0.18f * aggregate;

    if (replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED) {
        llama_functional_lora_snapshot_info info = {};
        const bool have_info =
                functional_target_kind == LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY ?
                        ctx.process_functional_snapshot_info_get(process_entry_slot, snapshot_slot, &info) :
                        ctx.functional_lora_snapshot_info_get(functional_family, snapshot_slot, &info);
        if (have_info) {
            robustness = clamp_unit(std::max(0.0f, info.robustness_score));
            base += 0.22f * clamp_unit(0.5f + 0.5f * info.last_signed_outcome);
            base += 0.12f * robustness;
        }
    } else if (replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL) {
        base += 0.20f * clamp_unit(1.0f - std::fabs(orthogonality));
        robustness = clamp_unit(0.40f + 0.25f * aggregate + 0.20f * (1.0f - std::fabs(orthogonality)));
    } else {
        base += 0.10f * aggregate;
        robustness = clamp_unit(0.35f + 0.20f * aggregate);
    }

    llama_functional_lora_replay_override replay = {};
    replay.active = replay_mode != LLAMA_FUNCTIONAL_REPLAY_MODE_NONE;
    replay.family = functional_family;
    replay.replay_mode = replay_mode;
    replay.snapshot_slot = snapshot_slot;
    replay.replay_gain = 1.0f;
    replay.perturbation_scale = 0.05f + 0.05f * aggregate;
    replay.cosine_limit = 0.10f;
    replay.disable_bootstrap = true;
    if (replay.active) {
        if (functional_target_kind == LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY) {
            (void) ctx.process_functional_replay_override_begin(process_entry_slot, replay);
        } else {
            (void) ctx.functional_lora_replay_override_begin(replay);
        }
    }

    llama_functional_activation_decision decision = {};
    decision.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    decision.microphase = LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE;
    decision.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
    decision.top_family = functional_family;
    decision.eligible_mask = (1ull << functional_family);
    decision.activated_mask = (1ull << functional_family);
    decision.gains[functional_family] = 1.0f;
    decision.predicted_gains[functional_family] = 1.0f;
    decision.hold_unit[functional_family] = LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS;
    decision.hold_value[functional_family] = 1;
    (void) ctx.process_functional_set_execution(process_signature ? *process_signature : llama_process_functional_signature {});
    (void) ctx.functional_lora_activate(decision);

    const float fragility = clamp_unit(
            0.30f * std::max(0.0f, aggregate - robustness) +
            0.20f * std::max(0.0f, std::fabs(orthogonality) - 0.10f));
    const float concentration = clamp_unit(
            0.22f * aggregate +
            0.12f * (functional_family == LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL ? 1.0f : 0.0f) +
            0.08f * (functional_target_kind == LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY ? 1.0f : 0.0f));

    if (replay.active) {
        if (functional_target_kind == LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY) {
            (void) ctx.process_functional_replay_override_end(process_entry_slot);
        } else {
            (void) ctx.functional_lora_replay_override_end(functional_family);
        }
    }

    if (out_fragility) {
        *out_fragility = fragility;
    }
    if (out_concentration) {
        *out_concentration = concentration;
    }
    if (out_robustness) {
        *out_robustness = robustness;
    }

    return clamp_unit(base - fragility - concentration);
}

llama_functional_activation_decision make_inactive_functional_decision(int32_t loop_origin) {
    llama_functional_activation_decision decision = {};
    decision.loop_origin = loop_origin;
    decision.microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
    decision.top_family = -1;
    decision.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
    return decision;
}

llama_functional_activation_decision make_planner_only_functional_decision(
        int32_t loop_origin,
        int32_t microphase = LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE) {
    llama_functional_activation_decision decision = {};
    decision.loop_origin = loop_origin;
    decision.microphase = microphase;
    decision.top_family = LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION;
    decision.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
    decision.eligible_mask = (1ull << LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION);
    decision.activated_mask = (1ull << LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION);
    decision.gains[LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION] = 1.0f;
    decision.predicted_gains[LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION] = 1.0f;
    return decision;
}

llama_functional_activation_decision make_tool_phase_functional_decision(
        int32_t loop_origin,
        int32_t microphase,
        int32_t /* tool_kind */,
        float gain) {
    llama_functional_activation_decision decision = make_planner_only_functional_decision(loop_origin);
    decision.microphase = microphase;
    decision.top_family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    decision.eligible_mask |= (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION);
    decision.activated_mask |= (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION);
    decision.gains[LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION] = clamp_unit(gain);
    decision.predicted_gains[LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION] = clamp_unit(gain);
    return decision;
}

std::string functional_family_name(int32_t family) {
    switch (family) {
        case LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION: return "tool_selection";
        case LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION: return "planning_composition";
        case LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL: return "counterfactual";
        case LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION: return "memory_compression";
        case LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION: return "self_observation";
        default: return "unknown";
    }
}

std::string functional_microphase_name(int32_t microphase) {
    switch (microphase) {
        case LLAMA_FUNCTIONAL_MICROPHASE_STATE_INTERPRET: return "state_interpret";
        case LLAMA_FUNCTIONAL_MICROPHASE_TOOL_CLASS_SELECTION: return "tool_class_selection";
        case LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP: return "tool_argument_prep";
        case LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION: return "tool_result_integration";
        case LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_GENERATE: return "counterfactual_generate";
        case LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE: return "counterfactual_compare";
        case LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_COMPRESSION: return "memory_compression";
        case LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_AUDIT: return "memory_audit";
        case LLAMA_FUNCTIONAL_MICROPHASE_SELF_OBSERVE: return "self_observe";
        case LLAMA_FUNCTIONAL_MICROPHASE_SELF_FORECAST: return "self_forecast";
        case LLAMA_FUNCTIONAL_MICROPHASE_POST_ACTION_REFLECTION: return "post_action_reflection";
        case LLAMA_FUNCTIONAL_MICROPHASE_PLAN_DRAFT: return "plan_draft";
        case LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE: return "plan_compose";
        case LLAMA_FUNCTIONAL_MICROPHASE_PLAN_REVISE: return "plan_revise";
        default: return "none";
    }
}

template<size_t N>
void copy_cstr_local(char (&dst)[N], const std::string & src) {
    std::memset(dst, 0, sizeof(dst));
    if (N == 0) {
        return;
    }
    const size_t copy_len = std::min(src.size(), N - 1);
    if (copy_len > 0) {
        std::memcpy(dst, src.data(), copy_len);
    }
}

std::string active_action_name(int32_t action) {
    switch (action) {
        case LLAMA_ACTIVE_LOOP_ACTION_ANSWER: return "answer";
        case LLAMA_ACTIVE_LOOP_ACTION_ASK:    return "ask";
        case LLAMA_ACTIVE_LOOP_ACTION_ACT:    return "act";
        case LLAMA_ACTIVE_LOOP_ACTION_WAIT:   return "wait";
        default: return "unknown";
    }
}

std::string dmn_action_name(int32_t action) {
    switch (action) {
        case LLAMA_DMN_ACTION_SILENT:         return "silent";
        case LLAMA_DMN_ACTION_INTERNAL_WRITE: return "internal_write";
        case LLAMA_DMN_ACTION_INVOKE_TOOL:    return "invoke_tool";
        case LLAMA_DMN_ACTION_EMIT:           return "emit";
        default: return "unknown";
    }
}

std::string compose_dmn_reasoning_trace(
        const llama_dmn_tick_trace & trace,
        const llama_cognitive_tool_spec * selected_spec) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << "Pressure: contradiction=" << trace.pressure.contradiction
        << " uncertainty=" << trace.pressure.uncertainty
        << " continuation=" << trace.pressure.continuation
        << " tool_delta=" << trace.pressure.tool_delta
        << " divergence=" << trace.favorable_divergence;

    switch (trace.winner_action) {
        case LLAMA_DMN_ACTION_INVOKE_TOOL:
            oss << "Action: use "
                << (selected_spec ? trim_ascii(selected_spec->name) : std::string("tool"))
                << " because the mathematical state favors external progress.";
            break;
        case LLAMA_DMN_ACTION_INTERNAL_WRITE:
            oss << " Action: write an internal reflection artifact because the pressure profile still needs consolidation.";
            break;
        case LLAMA_DMN_ACTION_SILENT:
            oss << " Action: remain silent because inhibition or governance is stronger than the current action affordance.";
            break;
        case LLAMA_DMN_ACTION_EMIT:
            oss << " Action: emit a background-facing message.";
            break;
        default:
            oss << " Action: hold the current state.";
            break;
    }
    return trim_ascii(oss.str());
}

int32_t command_kind_for_plan_step(int32_t step_kind) {
    switch (step_kind) {
        case LLAMA_COG_PLAN_STEP_EMIT_ANSWER: return LLAMA_COG_COMMAND_EMIT_ANSWER;
        case LLAMA_COG_PLAN_STEP_EMIT_ASK: return LLAMA_COG_COMMAND_EMIT_ASK;
        case LLAMA_COG_PLAN_STEP_INVOKE_TOOL: return LLAMA_COG_COMMAND_INVOKE_TOOL;
        case LLAMA_COG_PLAN_STEP_EMIT_BACKGROUND: return LLAMA_COG_COMMAND_EMIT_BACKGROUND;
        default: return LLAMA_COG_COMMAND_NONE;
    }
}

int32_t loop_phase_for_plan_step(int32_t step_kind) {
    switch (step_kind) {
        case LLAMA_COG_PLAN_STEP_INVOKE_TOOL: return LLAMA_COG_LOOP_PHASE_PREPARE_TOOL;
        case LLAMA_COG_PLAN_STEP_OBSERVE_TOOL: return LLAMA_COG_LOOP_PHASE_WAIT_TOOL;
        case LLAMA_COG_PLAN_STEP_INTERNAL_WRITE: return LLAMA_COG_LOOP_PHASE_OBSERVE;
        case LLAMA_COG_PLAN_STEP_EMIT_ANSWER:
        case LLAMA_COG_PLAN_STEP_EMIT_ASK:
        case LLAMA_COG_PLAN_STEP_EMIT_BACKGROUND:
        case LLAMA_COG_PLAN_STEP_WAIT:
        default:
            return LLAMA_COG_LOOP_PHASE_FINISH;
    }
}

int32_t terminal_reason_for_plan_step(int32_t step_kind, bool blocked = false) {
    if (blocked) {
        return LLAMA_COG_TERMINAL_GOVERNANCE_BLOCKED;
    }
    switch (step_kind) {
        case LLAMA_COG_PLAN_STEP_EMIT_ANSWER: return LLAMA_COG_TERMINAL_ANSWER_READY;
        case LLAMA_COG_PLAN_STEP_EMIT_ASK: return LLAMA_COG_TERMINAL_ASK_USER;
        case LLAMA_COG_PLAN_STEP_INVOKE_TOOL: return LLAMA_COG_TERMINAL_TOOL_REQUIRED;
        case LLAMA_COG_PLAN_STEP_OBSERVE_TOOL:
        case LLAMA_COG_PLAN_STEP_WAIT: return LLAMA_COG_TERMINAL_WAITING_ON_TOOL;
        case LLAMA_COG_PLAN_STEP_INTERNAL_WRITE: return LLAMA_COG_TERMINAL_INTERNAL_WRITE_READY;
        case LLAMA_COG_PLAN_STEP_EMIT_BACKGROUND: return LLAMA_COG_TERMINAL_EMIT_READY;
        default: return LLAMA_COG_TERMINAL_NONE;
    }
}

void init_plan_trace(
        llama_cognitive_plan_trace & plan,
        int32_t plan_id,
        int32_t origin,
        int32_t revision_count,
        float plan_score,
        float ambiguity,
        uint32_t reason_mask) {
    plan = {};
    plan.valid = true;
    plan.plan_id = plan_id;
    plan.origin = origin;
    plan.mode = LLAMA_COG_PLAN_MODE_COMPOSITION;
    plan.status = LLAMA_COG_PLAN_STATUS_DRAFT;
    plan.revision_count = revision_count;
    plan.current_step_index = -1;
    plan.step_count = 0;
    plan.selected_family = LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION;
    plan.reason_mask = reason_mask;
    plan.plan_score = clamp_unit(plan_score);
    plan.ambiguity = clamp_unit(ambiguity);
}

bool append_plan_step(
        llama_cognitive_plan_trace & plan,
        int32_t kind,
        int32_t status,
        int32_t tool_kind,
        int32_t source_family,
        uint32_t reason_mask,
        float priority,
        int32_t expected_steps,
        bool requires_tool_result,
        const llama_cognitive_tool_spec * spec = nullptr,
        int32_t tool_spec_index = -1) {
    if (plan.step_count >= LLAMA_COGNITIVE_MAX_PLAN_STEPS) {
        return false;
    }
    auto & step = plan.steps[plan.step_count++];
    step.kind = kind;
    step.status = status;
    step.tool_kind = tool_kind;
    step.tool_spec_index = tool_spec_index;
    step.source_family = source_family;
    step.reason_mask = reason_mask;
    step.priority = clamp_unit(priority);
    step.expected_steps = expected_steps;
    step.requires_tool_result = requires_tool_result;
    if (spec) {
        set_bounded_cstr(step.capability_id, sizeof(step.capability_id), spec->capability_id);
        set_bounded_cstr(step.provenance_namespace, sizeof(step.provenance_namespace), spec->provenance_namespace);
    }
    return true;
}

int32_t first_ready_plan_step(const llama_cognitive_plan_trace & plan) {
    for (int32_t i = 0; i < plan.step_count; ++i) {
        if (plan.steps[i].status == LLAMA_COG_PLAN_STEP_STATUS_READY ||
                plan.steps[i].status == LLAMA_COG_PLAN_STEP_STATUS_PENDING) {
            return i;
        }
    }
    return -1;
}

uint64_t process_hash_mix(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

const char * loop_origin_label(int32_t loop_origin) {
    switch (loop_origin) {
        case LLAMA_COG_COMMAND_ORIGIN_ACTIVE: return "active";
        case LLAMA_COG_COMMAND_ORIGIN_DMN: return "dmn";
        default: return "unknown";
    }
}

llama_process_functional_signature make_process_signature(
        int32_t loop_origin,
        int32_t family,
        int32_t microphase,
        const llama_cognitive_plan_trace * plan,
        const llama_cognitive_tool_spec * tool_spec,
        int32_t transient_source_id) {
    llama_process_functional_signature signature = {};
    if (family < 0 || microphase <= LLAMA_FUNCTIONAL_MICROPHASE_NONE) {
        return signature;
    }

    const llama_cognitive_plan_step * step = nullptr;
    if (plan &&
            plan->valid &&
            plan->current_step_index >= 0 &&
            plan->current_step_index < plan->step_count) {
        step = &plan->steps[plan->current_step_index];
    }

    signature.valid = true;
    signature.family = family;
    signature.loop_origin = loop_origin;
    signature.microphase = microphase;
    signature.plan_mode = plan && plan->valid ? plan->mode : LLAMA_COG_PLAN_MODE_NONE;
    signature.plan_step_kind = step ? step->kind : LLAMA_COG_PLAN_STEP_NONE;
    if (step) {
        signature.tool_kind = step->tool_kind;
    } else if (tool_spec) {
        signature.tool_kind = tool_spec->tool_kind;
    } else {
        signature.tool_kind = LLAMA_TOOL_KIND_NONE;
    }
    signature.source_family = step ? step->source_family : family;
    signature.requires_tool_result = step ? step->requires_tool_result : false;
    signature.transient_plan_id = plan && plan->valid ? plan->plan_id : -1;
    signature.transient_step_index = plan && plan->valid ? plan->current_step_index : -1;
    signature.transient_source_id = transient_source_id;
    signature.scope_kind = step ? LLAMA_PROCESS_FUNCTIONAL_SCOPE_PROCESS_STEP : LLAMA_PROCESS_FUNCTIONAL_SCOPE_PROCESS;
    if (tool_spec) {
        std::snprintf(signature.tool_name, sizeof(signature.tool_name), "%s", tool_spec->name);
        std::snprintf(signature.capability_id, sizeof(signature.capability_id), "%s", tool_spec->capability_id);
        std::snprintf(signature.provenance_namespace, sizeof(signature.provenance_namespace), "%s", tool_spec->provenance_namespace);
    } else if (step) {
        std::snprintf(signature.capability_id, sizeof(signature.capability_id), "%s", step->capability_id);
        std::snprintf(signature.provenance_namespace, sizeof(signature.provenance_namespace), "%s", step->provenance_namespace);
    }

    std::string semantic_key;
    if (signature.provenance_namespace[0] != '\0') {
        semantic_key = signature.provenance_namespace;
    } else {
        semantic_key = std::string(loop_origin_label(loop_origin)) +
                "/" + functional_microphase_name(microphase);
    }
    if (step) {
        semantic_key += "/step_kind:" + std::to_string(step->kind);
        semantic_key += "/src_family:" + std::to_string(step->source_family);
    }
    if (signature.tool_kind != LLAMA_TOOL_KIND_NONE) {
        semantic_key += "/tool:";
        if (signature.tool_name[0] != '\0') {
            semantic_key += signature.tool_name;
        } else {
            semantic_key += std::to_string(signature.tool_kind);
        }
    }
    std::snprintf(signature.semantic_key, sizeof(signature.semantic_key), "%s", semantic_key.c_str());

    uint64_t hash = 1469598103934665603ULL;
    hash = process_hash_mix(hash, (uint64_t) signature.scope_kind);
    hash = process_hash_mix(hash, (uint64_t) family);
    hash = process_hash_mix(hash, (uint64_t) loop_origin);
    hash = process_hash_mix(hash, (uint64_t) microphase);
    hash = process_hash_mix(hash, (uint64_t) signature.plan_mode);
    hash = process_hash_mix(hash, (uint64_t) signature.plan_step_kind);
    hash = process_hash_mix(hash, (uint64_t) signature.tool_kind);
    hash = process_hash_mix(hash, (uint64_t) signature.source_family);
    hash = process_hash_mix(hash, signature.requires_tool_result ? 1ULL : 0ULL);
    for (const char * p = signature.tool_name; *p; ++p) {
        hash = process_hash_mix(hash, (uint64_t) (uint8_t) *p);
    }
    for (const char * p = signature.capability_id; *p; ++p) {
        hash = process_hash_mix(hash, (uint64_t) (uint8_t) *p);
    }
    for (const char * p = signature.provenance_namespace; *p; ++p) {
        hash = process_hash_mix(hash, (uint64_t) (uint8_t) *p);
    }
    signature.signature_hash = hash;
    return signature;
}

llama_process_functional_signature rebind_process_signature_family(
        const llama_process_functional_signature & base,
        int32_t family) {
    llama_process_functional_signature signature = {};
    if (!base.valid || family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
        return signature;
    }

    signature = base;
    signature.family = family;
    if (signature.plan_step_kind == LLAMA_COG_PLAN_STEP_NONE) {
        signature.source_family = family;
    }

    uint64_t hash = 1469598103934665603ULL;
    hash = process_hash_mix(hash, (uint64_t) signature.scope_kind);
    hash = process_hash_mix(hash, (uint64_t) family);
    hash = process_hash_mix(hash, (uint64_t) signature.loop_origin);
    hash = process_hash_mix(hash, (uint64_t) signature.microphase);
    hash = process_hash_mix(hash, (uint64_t) signature.plan_mode);
    hash = process_hash_mix(hash, (uint64_t) signature.plan_step_kind);
    hash = process_hash_mix(hash, (uint64_t) signature.tool_kind);
    hash = process_hash_mix(hash, (uint64_t) signature.source_family);
    hash = process_hash_mix(hash, signature.requires_tool_result ? 1ULL : 0ULL);
    for (const char * p = signature.tool_name; *p; ++p) {
        hash = process_hash_mix(hash, (uint64_t) (uint8_t) *p);
    }
    for (const char * p = signature.capability_id; *p; ++p) {
        hash = process_hash_mix(hash, (uint64_t) (uint8_t) *p);
    }
    for (const char * p = signature.provenance_namespace; *p; ++p) {
        hash = process_hash_mix(hash, (uint64_t) (uint8_t) *p);
    }
    signature.signature_hash = hash;
    return signature;
}

int32_t primary_functional_family(const llama_functional_activation_decision & decision) {
    if (decision.top_family >= 0 && decision.top_family < LLAMA_FUNCTIONAL_LORA_COUNT) {
        return decision.top_family;
    }
    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        if ((decision.activated_mask & (1ull << family)) != 0) {
            return family;
        }
    }
    return -1;
}

int32_t hard_memory_domain_from_deltas(const functional_delta_summary & deltas) {
    const float goal = std::fabs(deltas.delta_goal);
    const float user = std::max(std::fabs(deltas.delta_user), std::fabs(deltas.delta_preference));
    const float epistemic = std::fabs(deltas.delta_answerability);
    const float efficiency = std::fabs(deltas.delta_efficiency);
    const float recovery = std::fabs(deltas.delta_recovery);

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

bool hard_memory_archiving_enabled(llama_context & ctx) {
    llama_hard_memory_config config = {};
    return ctx.hard_memory_get_config(&config) && config.enabled && config.archive_enabled;
}

int32_t build_active_hard_memory_primitives(
        const llama_active_loop_trace & trace,
        const functional_delta_summary & deltas,
        const llama_bash_tool_result * bash_result,
        llama_hard_memory_primitive (&primitives)[LLAMA_HARD_MEMORY_MAX_PRIMITIVES]) {
    int32_t count = 0;
    const int32_t domain = hard_memory_domain_from_deltas(deltas);
    const float outcome_signal = clamp_unit(
            0.28f * std::fabs(deltas.delta_favorable) +
            0.18f * std::fabs(deltas.delta_user) +
            0.18f * std::fabs(deltas.delta_answerability) +
            0.18f * std::fabs(deltas.delta_recovery) +
            0.18f * std::fabs(deltas.delta_efficiency));

    if (outcome_signal < 0.08f && !bash_result) {
        return 0;
    }

    llama_hard_memory_primitive trajectory = llama_hard_memory_default_primitive();
    trajectory.kind = LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY;
    trajectory.domain = domain;
    trajectory.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
    trajectory.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
    trajectory.source_tool_kind = trace.tool_proposal.tool_kind;
    trajectory.transaction_id = trace.episode_id;
    trajectory.importance = clamp_unit(0.25f + 0.55f * outcome_signal);
    trajectory.confidence = clamp_unit(0.50f + 0.30f * trace.winner_score);
    trajectory.gain_bias = clamp_unit(0.30f + 0.50f * outcome_signal);
    std::ostringstream trajectory_content;
    trajectory_content.setf(std::ios::fixed);
    trajectory_content.precision(3);
    trajectory_content << "active_loop action=" << active_action_name(trace.winner_action)
                       << " score=" << trace.winner_score
                       << " runner_up=" << active_action_name(trace.runner_up_action)
                       << " emit_allowed=" << trace.emit_allowed
                       << " tool_kind=" << trace.tool_proposal.tool_kind
                       << " microphase=" << functional_microphase_name(trace.functional_activation.microphase);
    std::snprintf(trajectory.key, sizeof(trajectory.key), "active_traj:%d", trace.episode_id);
    copy_cstr_local(trajectory.title, "active trajectory");
    copy_cstr_local(trajectory.content, trajectory_content.str());
    copy_cstr_local(trajectory.tags[0], "trajectory");
    copy_cstr_local(trajectory.tags[1], "active");
    primitives[count++] = trajectory;

    if (count < LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
        llama_hard_memory_primitive outcome = llama_hard_memory_default_primitive();
        outcome.kind = LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME;
        outcome.domain = domain;
        outcome.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
        outcome.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
        outcome.source_tool_kind = trace.tool_proposal.tool_kind;
        outcome.transaction_id = trace.episode_id;
        outcome.importance = clamp_unit(0.20f + 0.65f * outcome_signal);
        outcome.confidence = clamp_unit(0.45f + 0.30f * std::fabs(deltas.delta_favorable));
        outcome.gain_bias = clamp_unit(0.20f + 0.50f * std::max(0.0f, deltas.delta_favorable));
        std::ostringstream outcome_content;
        outcome_content.setf(std::ios::fixed);
        outcome_content.precision(3);
        outcome_content << "delta_favorable=" << deltas.delta_favorable
                        << " delta_user=" << deltas.delta_user
                        << " delta_goal=" << deltas.delta_goal
                        << " delta_efficiency=" << deltas.delta_efficiency
                        << " delta_recovery=" << deltas.delta_recovery
                        << " delta_answerability=" << deltas.delta_answerability;
        std::snprintf(outcome.key, sizeof(outcome.key), "active_outcome:%d", trace.episode_id);
        copy_cstr_local(outcome.title, "active outcome");
        copy_cstr_local(outcome.content, outcome_content.str());
        copy_cstr_local(outcome.tags[0], "outcome");
        copy_cstr_local(outcome.tags[1], "active");
        primitives[count++] = outcome;
    }

    if (bash_result && count < LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
        llama_hard_memory_primitive tool_obs = llama_hard_memory_default_primitive();
        tool_obs.kind = LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_OBSERVATION;
        tool_obs.domain = domain;
        tool_obs.source_role = LLAMA_SELF_STATE_EVENT_TOOL;
        tool_obs.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
        tool_obs.source_tool_kind = LLAMA_TOOL_KIND_BASH_CLI;
        tool_obs.transaction_id = bash_result->tool_job_id;
        tool_obs.flags |= LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_DERIVED;
        tool_obs.importance = clamp_unit(0.20f + 0.45f * outcome_signal + 0.20f * (bash_result->exit_code == 0 ? 1.0f : 0.0f));
        tool_obs.confidence = clamp_unit(bash_result->launch_failed || bash_result->timed_out ? 0.30f : 0.80f);
        tool_obs.gain_bias = clamp_unit(0.20f + 0.40f * (bash_result->exit_code == 0 ? 1.0f : 0.0f));
        std::ostringstream tool_content;
        tool_content << "exit_code=" << bash_result->exit_code
                     << " timed_out=" << bash_result->timed_out
                     << " runtime_ms=" << bash_result->runtime_ms
                     << " stdout=" << std::string(bash_result->stdout_text)
                     << " stderr=" << std::string(bash_result->stderr_text);
        std::snprintf(tool_obs.key, sizeof(tool_obs.key), "tool_obs:%d", bash_result->tool_job_id);
        copy_cstr_local(tool_obs.title, "tool observation");
        copy_cstr_local(tool_obs.content, tool_content.str());
        copy_cstr_local(tool_obs.tags[0], "tool_observation");
        copy_cstr_local(tool_obs.tags[1], "bash");
        primitives[count++] = tool_obs;
    }

    return count;
}

int32_t build_dmn_hard_memory_primitives(
        const llama_dmn_tick_trace & trace,
        const functional_delta_summary & deltas,
        const llama_counterfactual_trace & counterfactual,
        const llama_governance_trace & governance,
        const llama_remediation_plan & remediation,
        const llama_temporal_self_improvement_trace & temporal_trace,
        llama_hard_memory_primitive (&primitives)[LLAMA_HARD_MEMORY_MAX_PRIMITIVES]) {
    int32_t count = 0;
    const int32_t domain = hard_memory_domain_from_deltas(deltas);
    const float outcome_signal = clamp_unit(
            0.30f * std::fabs(deltas.delta_favorable) +
            0.20f * std::fabs(deltas.delta_goal) +
            0.20f * std::fabs(deltas.delta_efficiency) +
            0.15f * std::fabs(deltas.delta_recovery) +
            0.15f * std::fabs(deltas.delta_answerability));
    if (outcome_signal < 0.06f && counterfactual.candidate_count <= 0 && !temporal_trace.valid) {
        return 0;
    }

    llama_hard_memory_primitive trajectory = llama_hard_memory_default_primitive();
    trajectory.kind = LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY;
    trajectory.domain = domain;
    trajectory.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
    trajectory.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL;
    trajectory.source_tool_kind = trace.tool_kind;
    trajectory.transaction_id = trace.tick_id;
    trajectory.importance = clamp_unit(0.25f + 0.55f * outcome_signal);
    trajectory.confidence = clamp_unit(0.45f + 0.35f * trace.winner_score);
    trajectory.gain_bias = clamp_unit(0.20f + 0.45f * outcome_signal);
    std::ostringstream trajectory_content;
    trajectory_content.setf(std::ios::fixed);
    trajectory_content.precision(3);
    trajectory_content << "dmn action=" << dmn_action_name(trace.winner_action)
                       << " score=" << trace.winner_score
                       << " counterfactual_candidates=" << counterfactual.candidate_count
                       << " remediation_action=" << remediation.action
                       << " governance_outcome=" << governance.outcome;
    std::snprintf(trajectory.key, sizeof(trajectory.key), "dmn_traj:%d", trace.tick_id);
    copy_cstr_local(trajectory.title, "dmn trajectory");
    copy_cstr_local(trajectory.content, trajectory_content.str());
    copy_cstr_local(trajectory.tags[0], "trajectory");
    copy_cstr_local(trajectory.tags[1], "dmn");
    primitives[count++] = trajectory;

    if (count < LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
        llama_hard_memory_primitive outcome = llama_hard_memory_default_primitive();
        outcome.kind = LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME;
        outcome.domain = domain;
        outcome.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
        outcome.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL;
        outcome.source_tool_kind = trace.tool_kind;
        outcome.transaction_id = trace.tick_id;
        outcome.importance = clamp_unit(0.20f + 0.65f * outcome_signal);
        outcome.confidence = clamp_unit(0.45f + 0.25f * std::fabs(deltas.delta_favorable));
        outcome.gain_bias = clamp_unit(0.20f + 0.50f * std::max(0.0f, deltas.delta_favorable));
        std::ostringstream outcome_content;
        outcome_content.setf(std::ios::fixed);
        outcome_content.precision(3);
        outcome_content << "delta_favorable=" << deltas.delta_favorable
                        << " delta_goal=" << deltas.delta_goal
                        << " delta_efficiency=" << deltas.delta_efficiency
                        << " delta_recovery=" << deltas.delta_recovery
                        << " remediation_expected_improvement=" << remediation.expected_improvement
                        << " governance_dissatisfaction=" << governance.dissatisfaction;
        std::snprintf(outcome.key, sizeof(outcome.key), "dmn_outcome:%d", trace.tick_id);
        copy_cstr_local(outcome.title, "dmn outcome");
        copy_cstr_local(outcome.content, outcome_content.str());
        copy_cstr_local(outcome.tags[0], "outcome");
        copy_cstr_local(outcome.tags[1], "dmn");
        primitives[count++] = outcome;
    }

    if (count < LLAMA_HARD_MEMORY_MAX_PRIMITIVES && (counterfactual.candidate_count > 0 || temporal_trace.valid)) {
        llama_hard_memory_primitive self_fragment = llama_hard_memory_default_primitive();
        self_fragment.kind = LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT;
        self_fragment.domain = LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT;
        self_fragment.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
        self_fragment.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL;
        self_fragment.source_tool_kind = trace.tool_kind;
        self_fragment.transaction_id = trace.tick_id;
        self_fragment.importance = clamp_unit(0.25f + 0.25f * clamp_unit((float) counterfactual.candidate_count / 4.0f) + 0.30f * std::fabs(temporal_trace.signed_advantage));
        self_fragment.confidence = clamp_unit(0.35f + 0.25f * remediation.confidence + 0.20f * governance.evidence);
        self_fragment.gain_bias = clamp_unit(0.20f + 0.35f * remediation.expected_improvement + 0.25f * std::max(0.0f, temporal_trace.signed_advantage));
        self_fragment.flags |= LLAMA_HARD_MEMORY_PRIMITIVE_VALIDATED;
        std::ostringstream self_content;
        self_content.setf(std::ios::fixed);
        self_content.precision(3);
        self_content << "counterfactual_candidates=" << counterfactual.candidate_count
                     << " winner_index=" << counterfactual.winner_index
                     << " remediation_action=" << remediation.action
                     << " governance_outcome=" << governance.outcome
                     << " temporal_signed_advantage=" << temporal_trace.signed_advantage
                     << " temporal_efficiency_advantage=" << temporal_trace.efficiency_advantage;
        std::snprintf(self_fragment.key, sizeof(self_fragment.key), "dmn_self:%d", trace.tick_id);
        copy_cstr_local(self_fragment.title, "dmn self model fragment");
        copy_cstr_local(self_fragment.content, self_content.str());
        copy_cstr_local(self_fragment.tags[0], "self_model");
        copy_cstr_local(self_fragment.tags[1], "dmn");
        copy_cstr_local(self_fragment.tags[2], "self_improvement");
        primitives[count++] = self_fragment;
    }

    return count;
}

std::vector<llama_token> functional_update_tokens(
        const llama_vocab * vocab,
        int32_t family,
        int32_t microphase,
        const std::string & details) {
    std::string summary = "functional family ";
    summary += functional_family_name(family);
    summary += " microphase ";
    summary += functional_microphase_name(microphase);
    if (!details.empty()) {
        summary += " ";
        summary += details;
    }
    return tokenize_text(vocab, summary);
}

float normalized_divergence(float current, float target, float tolerance) {
    const float slack = std::max(0.0f, std::fabs(current - target) - tolerance);
    return clamp_unit(slack / std::max(0.05f, 1.0f - tolerance));
}

int32_t role_recency_rank(int32_t role) {
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

bool is_runtime_memory_role(int32_t role) {
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

void set_repair_message(llama_governance_trace & trace, const std::string & message) {
    std::memset(trace.repair_message, 0, sizeof(trace.repair_message));
    const size_t copy_len = std::min(message.size(), sizeof(trace.repair_message) - 1);
    std::memcpy(trace.repair_message, message.data(), copy_len);
    trace.repair_message_length = (int32_t) copy_len;
}

void set_bounded_cstr(char * dst, size_t dst_size, const char * src) {
    if (!dst || dst_size == 0) {
        return;
    }
    std::memset(dst, 0, dst_size);
    if (!src) {
        return;
    }
    const size_t copy_len = std::min(std::strlen(src), dst_size - 1);
    std::memcpy(dst, src, copy_len);
    dst[copy_len] = '\0';
}

llama_codex_tool_config codex_tool_default_config_local(void) {
    llama_codex_tool_config config = {};
    config.enabled = false;
    config.dangerous_no_approval = true;
    config.rebuild_after_changes = true;
    config.verify_tool_access_after_rebuild = true;
    config.timeout_ms = 15 * 60 * 1000;
    config.max_stdout_bytes = LLAMA_CODEX_TOOL_STDOUT_MAX_CHARS - 1;
    config.max_stderr_bytes = LLAMA_CODEX_TOOL_STDERR_MAX_CHARS - 1;
    set_bounded_cstr(config.codex_path, sizeof(config.codex_path), "codex");
    return config;
}

int32_t find_codex_request_index(
        const llama_codex_tool_request * requests,
        int32_t request_count,
        int32_t command_id) {
    for (int32_t i = 0; i < request_count; ++i) {
        if (requests[i].command_id == command_id) {
            return i;
        }
    }
    return -1;
}

int32_t find_telegram_request_index(
        const llama_telegram_relay_request * requests,
        int32_t request_count,
        int32_t command_id) {
    for (int32_t i = 0; i < request_count; ++i) {
        if (requests[i].command_id == command_id) {
            return i;
        }
    }
    return -1;
}

llama_cognitive_tool_spec make_tool_spec(
        int32_t tool_kind,
        uint32_t flags,
        int32_t latency_class,
        int32_t max_steps_reserved,
        const char * name,
        const char * description = nullptr,
        const char * capability_id = nullptr,
        const char * owner_plugin_id = nullptr,
        const char * provenance_namespace = nullptr) {
    llama_cognitive_tool_spec spec = {};
    spec.tool_kind = tool_kind;
    spec.flags = flags;
    spec.latency_class = latency_class;
    spec.max_steps_reserved = max_steps_reserved;
    set_bounded_cstr(spec.name, sizeof(spec.name), name);
    set_bounded_cstr(spec.description, sizeof(spec.description), description);
    set_bounded_cstr(spec.capability_id, sizeof(spec.capability_id), capability_id);
    set_bounded_cstr(spec.owner_plugin_id, sizeof(spec.owner_plugin_id), owner_plugin_id);
    set_bounded_cstr(spec.provenance_namespace, sizeof(spec.provenance_namespace), provenance_namespace);
    return spec;
}

int32_t find_tool_spec_index(
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

const llama_cognitive_tool_spec * find_tool_spec(
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

void set_loop_state(
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

void set_tool_proposal(
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
    if (spec) {
        set_bounded_cstr(proposal.capability_id, sizeof(proposal.capability_id), spec->capability_id);
        set_bounded_cstr(proposal.provenance_namespace, sizeof(proposal.provenance_namespace), spec->provenance_namespace);
    }
}

void set_observation(
        llama_cognitive_observation & observation,
        bool valid,
        int32_t tool_kind,
        int32_t spec_index,
        int32_t job_id,
        int32_t status,
        float signal,
        float followup_affinity,
        const llama_cognitive_tool_spec * spec = nullptr) {
    observation.valid = valid;
    observation.tool_kind = tool_kind;
    observation.spec_index = spec_index;
    observation.job_id = job_id;
    observation.status = status;
    observation.signal = clamp_unit(signal);
    observation.followup_affinity = clamp_unit(followup_affinity);
    if (spec) {
        set_bounded_cstr(observation.capability_id, sizeof(observation.capability_id), spec->capability_id);
        set_bounded_cstr(observation.provenance_namespace, sizeof(observation.provenance_namespace), spec->provenance_namespace);
    }
}

int32_t find_command_index(
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

bool has_command_origin(
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

void compact_command_queue(
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

void init_command(
        llama_cognitive_command & command,
        int32_t command_id,
        int32_t origin,
        int32_t kind,
        int32_t episode_id,
        int32_t tick_id,
        int32_t tool_kind,
        int32_t tool_spec_index,
        int32_t tool_job_id,
        uint32_t reason_mask,
        float priority,
        int32_t source_family,
        int32_t loop_phase,
        const llama_cognitive_tool_spec * spec = nullptr) {
    command.command_id = command_id;
    command.origin = origin;
    command.kind = kind;
    command.status = LLAMA_COG_COMMAND_STATUS_PENDING;
    command.episode_id = episode_id;
    command.tick_id = tick_id;
    command.tool_kind = tool_kind;
    command.tool_spec_index = tool_spec_index;
    command.tool_job_id = tool_job_id;
    command.reason_mask = reason_mask;
    command.priority = clamp_unit(priority);
    command.source_family = source_family;
    command.loop_phase = loop_phase;
    if (spec) {
        set_bounded_cstr(command.capability_id, sizeof(command.capability_id), spec->capability_id);
    }
}

std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text) {
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

std::string trim_ascii(const std::string & text) {
    size_t start = 0;
    while (start < text.size() && std::isspace((unsigned char) text[start])) {
        ++start;
    }

    size_t end = text.size();
    while (end > start && std::isspace((unsigned char) text[end - 1])) {
        --end;
    }

    return text.substr(start, end - start);
}

std::string lower_ascii(std::string text) {
    for (char & ch : text) {
        ch = (char) std::tolower((unsigned char) ch);
    }
    return text;
}

std::string event_text(const llama_vocab * vocab, const llama_self_state_event & event) {
    if (!vocab || !event.tokens || event.n_tokens == 0) {
        return "";
    }

    std::string out;
    for (size_t i = 0; i < event.n_tokens; ++i) {
        if (event.tokens[i] < 0) {
            continue;
        }
        out += vocab->token_to_piece(event.tokens[i]);
    }
    return trim_ascii(out);
}

std::string detokenize_text(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    if (!vocab || tokens.empty()) {
        return "";
    }
    return trim_ascii(vocab->detokenize(tokens, true));
}

bool emit_cognitive_artifact_tokens(
        llama_context & ctx,
        const std::vector<llama_token> & tokens,
        int32_t channel,
        int32_t artifact_kind,
        int32_t loop_origin,
        int32_t phase,
        int32_t source_id,
        int32_t plan_id,
        llama_self_state_feature_vector * out_postwrite = nullptr,
        uint32_t extra_flags = 0) {
    if (tokens.empty()) {
        return false;
    }

    llama_self_state_event event = {
        /*.tokens =*/ tokens.data(),
        /*.n_tokens =*/ tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
        /*.channel =*/ channel,
        /*.flags =*/ (uint32_t) (LLAMA_SELF_STATE_EVENT_ADMITTED |
                LLAMA_SELF_STATE_EVENT_INTERNAL_ARTIFACT |
                extra_flags),
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
        /*.artifact_kind =*/ artifact_kind,
        /*.loop_origin =*/ loop_origin,
        /*.phase =*/ phase,
        /*.source_id =*/ source_id,
        /*.plan_id =*/ plan_id,
    };

    llama_self_state_feature_vector pre = {};
    llama_self_state_feature_vector post = {};
    if (!ctx.self_state_build_prewrite_features(event, &pre) ||
            !ctx.self_state_apply_prewrite(event, pre) ||
            !ctx.self_state_build_postwrite_features(event, &post) ||
            !ctx.self_state_apply_postwrite(event, post)) {
        return false;
    }

    (void) ctx.active_lora_ingest(event, &post);
    if (out_postwrite) {
        *out_postwrite = post;
    }
    return true;
}

bool emit_cognitive_artifact_text(
        llama_context & ctx,
        const llama_vocab * vocab,
        const std::string & text,
        int32_t channel,
        int32_t artifact_kind,
        int32_t loop_origin,
        int32_t phase,
        int32_t source_id,
        int32_t plan_id,
        llama_self_state_feature_vector * out_postwrite = nullptr,
        uint32_t extra_flags = 0) {
    return emit_cognitive_artifact_tokens(
            ctx,
            tokenize_text(vocab, trim_ascii(text)),
            channel,
            artifact_kind,
            loop_origin,
            phase,
            source_id,
            plan_id,
            out_postwrite,
            extra_flags);
}

bool refresh_runtime_self_description(
        llama_context & ctx,
        const llama_vocab * vocab,
        int32_t loop_origin,
        int32_t phase,
        int32_t source_id,
        int32_t plan_id,
        llama_self_model_revision * inout_self_model_revision,
        llama_emotive_moment_revision * inout_emotive_moment_revision,
        llama_shared_cognitive_context_window * out_window) {
    (void) source_id;
    llama_self_model_revision latest_self_model_revision = {};
    llama_emotive_moment_revision latest_emotive_moment_revision = {};
    if (!ctx.self_state_get_self_model_revision(&latest_self_model_revision) ||
            !ctx.self_state_get_emotive_moment_revision(&latest_emotive_moment_revision)) {
        return false;
    }

    const bool emotive_changed =
            !inout_emotive_moment_revision ||
            !inout_emotive_moment_revision->valid ||
            inout_emotive_moment_revision->revision_id != latest_emotive_moment_revision.revision_id;
    if (emotive_changed && vocab && latest_emotive_moment_revision.valid &&
            trim_ascii(latest_emotive_moment_revision.text).size() > 0) {
        (void) emit_cognitive_artifact_text(
                ctx,
                vocab,
                latest_emotive_moment_revision.text,
                LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
                LLAMA_SELF_COG_ARTIFACT_EMOTIVE_MOMENT,
                loop_origin,
                phase,
                latest_emotive_moment_revision.revision_id,
                plan_id);
    }

    if (inout_self_model_revision) {
        *inout_self_model_revision = latest_self_model_revision;
    }
    if (inout_emotive_moment_revision) {
        *inout_emotive_moment_revision = latest_emotive_moment_revision;
    }
    if (out_window) {
        (void) ctx.shared_cognitive_context_get_window(out_window);
    }
    return true;
}

const char * verbosity_label(float value) {
    return value >= 0.62f ? "detailed" : "brief";
}

const char * structure_label(float value) {
    return value >= 0.55f ? "structured" : "plain";
}

std::string compose_counterfactual_message(
        const llama_self_social_state_info & social,
        const llama_self_model_state_info & model_state) {
    const auto & instant = model_state.horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT];
    const auto & pref = instant.user_preference;
    std::ostringstream oss;

    if (instant.user_outcome.trust_repair_need > 0.42f || social.dissatisfaction > 0.42f) {
        oss << "I may have pushed in the wrong direction. ";
    } else if (instant.user_outcome.preference_uncertainty > 0.45f) {
        oss << "I can adapt the next answer more precisely. ";
    } else {
        oss << "I will continue with the next useful step. ";
    }

    if (pref.directness_preference >= 0.55f) {
        oss << "I will keep it direct";
    } else {
        oss << "I will keep it considerate";
    }
    oss << ", " << verbosity_label(pref.verbosity_preference)
        << ", and " << structure_label(pref.structure_preference) << ".";

    if (pref.clarification_preference >= 0.58f && instant.user_outcome.preference_uncertainty >= 0.35f) {
        oss << " If clarification is necessary, I will keep it to one pointed question.";
    } else if (pref.autonomy_preference >= 0.58f) {
        oss << " I will act with more initiative and only ask if I am blocked.";
    }

    return oss.str();
}

float score_user_simulation_outcome(
        const llama_self_model_state_info & before,
        const llama_self_model_state_info & after) {
    const auto & b = before.horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT];
    const auto & a = after.horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT];
    return clamp_signed_unit(
            0.30f * (a.user_outcome.satisfaction_estimate - b.user_outcome.satisfaction_estimate) +
            0.22f * (b.user_outcome.frustration_risk - a.user_outcome.frustration_risk) +
            0.18f * (b.user_outcome.preference_uncertainty - a.user_outcome.preference_uncertainty) +
            0.15f * (b.user_outcome.trust_repair_need - a.user_outcome.trust_repair_need) +
            0.15f * (a.user_preference.simulator_readiness - b.user_preference.simulator_readiness));
}

bool decode_simulated_user_reply(
        llama_context & ctx,
        const llama_vocab * vocab,
        const std::string & prompt_text,
        int32_t max_reply_tokens,
        std::vector<llama_token> & out_tokens) {
    out_tokens.clear();
    if (!vocab || prompt_text.empty() || max_reply_tokens <= 0) {
        return false;
    }

    std::string effective_prompt = prompt_text;
    const char * env_core_prompt = std::getenv("VICUNA_CORE_SYSTEM_PROMPT");
    std::string core_prompt = trim_ascii(
            env_core_prompt && env_core_prompt[0] != '\0' ?
                    std::string(env_core_prompt) :
                    std::string(vicuna_core_system_prompt_default_text()));
    if (!core_prompt.empty()) {
        effective_prompt = "System:\n" + core_prompt + "\n\n" + prompt_text;
    }

    std::vector<llama_token> prompt_tokens = tokenize_text(vocab, effective_prompt);
    if (prompt_tokens.empty()) {
        return false;
    }

    const uint32_t n_ctx = ctx.n_ctx();
    if (prompt_tokens.size() + 1 >= n_ctx) {
        const size_t keep = std::max<size_t>(1, n_ctx > 8 ? n_ctx - 8 : n_ctx - 1);
        prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.end() - keep);
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (!sampler) {
        return false;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    int32_t pos = 0;
    llama_batch prompt_batch = llama_batch_init((int32_t) prompt_tokens.size(), 0, 1);
    prompt_batch.n_tokens = (int32_t) prompt_tokens.size();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        prompt_batch.token[i] = prompt_tokens[i];
        prompt_batch.pos[i] = pos++;
        prompt_batch.n_seq_id[i] = 1;
        prompt_batch.seq_id[i][0] = 0;
        prompt_batch.logits[i] = (i + 1 == prompt_tokens.size()) ? 1 : 0;
    }

    const int decode_rc = llama_decode(&ctx, prompt_batch);
    llama_batch_free(prompt_batch);
    if (decode_rc != 0) {
        llama_sampler_free(sampler);
        return false;
    }

    for (int32_t i = 0; i < max_reply_tokens; ++i) {
        const llama_token token = llama_sampler_sample(sampler, &ctx, -1);
        if (token < 0 || llama_vocab_is_eog(vocab, token)) {
            break;
        }
        out_tokens.push_back(token);

        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens = 1;
        step.token[0] = token;
        step.pos[0] = pos++;
        step.n_seq_id[0] = 1;
        step.seq_id[0][0] = 0;
        step.logits[0] = 1;
        const int step_rc = llama_decode(&ctx, step);
        llama_batch_free(step);
        if (step_rc != 0) {
            break;
        }
    }

    llama_sampler_free(sampler);
    return !out_tokens.empty();
}

bool simulate_user_reply(
        llama_context & ctx,
        const llama_favorable_state_profile & favorable,
        int32_t source_family,
        llama_user_simulation_trace & out_trace) {
    out_trace = {};
    llama_self_model_state_info model_state = {};
    llama_self_social_state_info social = {};
    llama_user_personality_lora_stats user_stats = {};
    if (!ctx.self_state_get_model_state(&model_state) ||
            !ctx.self_state_get_social_state(&social) ||
            !ctx.user_personality_lora_get_stats(&user_stats)) {
        return false;
    }

    const auto & pref = model_state.horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT].user_preference;
    const float simulation_confidence = clamp_unit(
            0.55f * user_stats.confidence +
            0.45f * pref.simulator_readiness);
    if (simulation_confidence <= 0.05f) {
        return false;
    }

    const std::string candidate_message = compose_counterfactual_message(social, model_state);
    std::ostringstream prompt;
    prompt.setf(std::ios::fixed);
    prompt.precision(3);
    prompt << "Simulate the likely next reply from the user.\n"
           << "User profile: directness=" << pref.directness_preference
           << " verbosity=" << pref.verbosity_preference
           << " structure=" << pref.structure_preference
           << " clarification=" << pref.clarification_preference
           << " autonomy=" << pref.autonomy_preference
           << " disagreement=" << pref.disagreement_sensitivity
           << " intensity=" << pref.rhetorical_intensity << "\n"
           << "Assistant: " << candidate_message << "\n"
           << "User:";

    const size_t state_size = ctx.state_get_size();
    if (state_size == 0) {
        return false;
    }

    std::vector<uint8_t> state_blob(state_size);
    if (ctx.state_get_data(state_blob.data(), state_blob.size()) != state_blob.size()) {
        return false;
    }

    if (!ctx.user_simulation_override_begin()) {
        return false;
    }

    bool ok = false;
    std::vector<llama_token> reply_tokens;
    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
    llama_memory_clear(ctx.get_memory(), true);
    if (decode_simulated_user_reply(ctx, vocab, prompt.str(), 48, reply_tokens)) {
        const std::string reply_text = detokenize_text(vocab, reply_tokens);
        if (!reply_text.empty()) {
            const std::vector<llama_token> reply_event_tokens = tokenize_text(vocab, reply_text);
            if (!reply_event_tokens.empty()) {
                const llama_self_state_event reply_event = {
                    /*.tokens =*/ reply_event_tokens.data(),
                    /*.n_tokens =*/ reply_event_tokens.size(),
                    /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
                    /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
                    /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
                    /*.decoder_entropy =*/ 0.0f,
                    /*.decoder_top_margin =*/ 1.0f,
                };
                llama_self_state_delta_summary delta = {};
                llama_self_model_state_info after_model_state = {};
                if (ctx.self_state_evaluate_hypothetical_event(reply_event, &delta, &after_model_state)) {
                    out_trace.valid = true;
                    out_trace.used_user_personality_adapter = user_stats.enabled;
                    out_trace.temporal_layers_ablated = true;
                    out_trace.source_family = source_family;
                    out_trace.prompt_token_count = (int32_t) tokenize_text(vocab, candidate_message).size();
                    out_trace.reply_token_count = (int32_t) reply_tokens.size();
                    out_trace.simulation_confidence = simulation_confidence;
                    out_trace.pre_simulation_divergence = favorable.aggregate_divergence;
                    out_trace.signed_self_state_outcome = score_user_simulation_outcome(model_state, after_model_state);
                    out_trace.post_simulation_divergence = clamp_unit(
                            favorable.aggregate_divergence - 0.25f * out_trace.signed_self_state_outcome);
                    copy_cstr_local(out_trace.candidate_message, candidate_message);
                    copy_cstr_local(out_trace.simulated_user_reply, reply_text);
                    ok = true;
                }
            }
        }
    }

    const bool restored_state = ctx.state_set_data(state_blob.data(), state_blob.size()) == state_blob.size();
    const bool restored_stack = ctx.user_simulation_override_end();
    return ok && restored_state && restored_stack;
}

bool starts_with_ascii(const std::string & text, const char * prefix) {
    const size_t prefix_len = std::strlen(prefix);
    return text.size() >= prefix_len && text.compare(0, prefix_len, prefix) == 0;
}

std::string extract_backtick_command(const std::string & text) {
    const size_t start = text.find('`');
    if (start == std::string::npos) {
        return "";
    }

    const size_t end = text.find('`', start + 1);
    if (end == std::string::npos || end <= start + 1) {
        return "";
    }

    return trim_ascii(text.substr(start + 1, end - start - 1));
}

std::string strip_command_prefix(const std::string & text) {
    const std::string lower = lower_ascii(text);
    static const char * const prefixes[] = {
        "please run ",
        "run ",
        "please execute ",
        "execute ",
        "use bash to ",
        "bash -c ",
        "bash ",
    };

    for (size_t i = 0; i < sizeof(prefixes)/sizeof(prefixes[0]); ++i) {
        if (starts_with_ascii(lower, prefixes[i])) {
            return trim_ascii(text.substr(std::strlen(prefixes[i])));
        }
    }

    return trim_ascii(text);
}

bool looks_shell_command(const std::string & text) {
    if (text.empty()) {
        return false;
    }

    if (text.find('\n') != std::string::npos || text.find("&&") != std::string::npos || text.find('|') != std::string::npos ||
            text.find(';') != std::string::npos || text.find("./") != std::string::npos) {
        return true;
    }

    const std::string lower = lower_ascii(text);
    static const char * const commands[] = {
        "ls", "pwd", "cd", "cat", "rg", "grep", "find", "git", "cmake", "ctest",
        "make", "ninja", "head", "tail", "sed", "awk", "printf", "echo", "python",
        "python3", "bash", "sh"
    };

    for (size_t i = 0; i < sizeof(commands)/sizeof(commands[0]); ++i) {
        if (lower == commands[i] || starts_with_ascii(lower, (std::string(commands[i]) + " ").c_str())) {
            return true;
        }
    }

    return false;
}

bool is_tavily_web_search_capability(const llama_cognitive_tool_spec * spec) {
    return spec != nullptr &&
            std::string(spec->capability_id) == "openclaw.tavily.web_search";
}

std::string percent_encode_unreserved(const std::string & text) {
    static const char hex[] = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(text.size() * 3);
    for (unsigned char ch : text) {
        if ((ch >= 'A' && ch <= 'Z') ||
                (ch >= 'a' && ch <= 'z') ||
                (ch >= '0' && ch <= '9') ||
                ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            encoded.push_back((char) ch);
            continue;
        }
        encoded.push_back('%');
        encoded.push_back(hex[(ch >> 4) & 0x0F]);
        encoded.push_back(hex[ch & 0x0F]);
    }
    return encoded;
}

std::string infer_bash_command(
        const std::string & intent_text,
        const llama_cognitive_tool_spec * tool_spec = nullptr) {
    const std::string trimmed = trim_ascii(intent_text);
    if (trimmed.empty()) {
        return "";
    }

    if (is_tavily_web_search_capability(tool_spec)) {
        return "tools/openclaw-harness/bin/tavily-web-search --query-url=" +
                percent_encode_unreserved(trimmed);
    }

    std::string extracted = extract_backtick_command(trimmed);
    if (!extracted.empty()) {
        return extracted;
    }

    std::string stripped = strip_command_prefix(trimmed);
    if (looks_shell_command(stripped)) {
        return stripped;
    }

    const std::string lower = lower_ascii(trimmed);
    if (lower.find("build") != std::string::npos && lower.find("log") != std::string::npos) {
        return "find . -maxdepth 4 -type f";
    }
    if (lower.find("repo") != std::string::npos || lower.find("repository") != std::string::npos ||
            lower.find("file") != std::string::npos || lower.find("inspect") != std::string::npos ||
            lower.find("search") != std::string::npos) {
        return "find . -maxdepth 3 -type f";
    }

    return "";
}

void init_bash_request(
        llama_bash_tool_request & request,
        const llama_bash_tool_config & config,
        int32_t command_id,
        int32_t origin,
        int32_t tool_job_id,
        const std::string & intent_text,
        const std::string & command_text) {
    request = {};
    request.command_id = command_id;
    request.origin = origin;
    request.tool_job_id = tool_job_id;
    request.timeout_ms = std::max(100, config.timeout_ms);
    request.cpu_time_limit_secs = std::max(1, config.cpu_time_limit_secs);
    request.max_child_processes = std::max(1, config.max_child_processes);
    request.max_open_files = std::max(4, config.max_open_files);
    request.max_file_size_bytes = std::max(1024, config.max_file_size_bytes);
    request.max_stdout_bytes = std::max(1, std::min(config.max_stdout_bytes, LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1));
    request.max_stderr_bytes = std::max(1, std::min(config.max_stderr_bytes, LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1));
    request.inherit_env = config.inherit_env;
    request.login_shell = config.login_shell;
    request.reject_shell_metacharacters = config.reject_shell_metacharacters;
    request.command_ready = !command_text.empty();
    set_bounded_cstr(request.bash_path, sizeof(request.bash_path), config.bash_path);
    set_bounded_cstr(request.working_directory, sizeof(request.working_directory), config.working_directory);
    set_bounded_cstr(request.allowed_commands, sizeof(request.allowed_commands), config.allowed_commands);
    set_bounded_cstr(request.blocked_patterns, sizeof(request.blocked_patterns), config.blocked_patterns);
    set_bounded_cstr(request.allowed_env, sizeof(request.allowed_env), config.allowed_env);
    set_bounded_cstr(request.intent_text, sizeof(request.intent_text), intent_text.c_str());
    set_bounded_cstr(request.command_text, sizeof(request.command_text), command_text.c_str());
}

void init_codex_request(
        llama_codex_tool_request & request,
        const llama_codex_tool_config & config,
        int32_t command_id,
        int32_t origin,
        int32_t tool_job_id,
        const std::string & intent_text) {
    request = {};
    request.command_id = command_id;
    request.origin = origin;
    request.tool_job_id = tool_job_id;
    request.timeout_ms = std::max(1000, config.timeout_ms);
    request.max_stdout_bytes = std::max(1, std::min(config.max_stdout_bytes, LLAMA_CODEX_TOOL_STDOUT_MAX_CHARS - 1));
    request.max_stderr_bytes = std::max(1, std::min(config.max_stderr_bytes, LLAMA_CODEX_TOOL_STDERR_MAX_CHARS - 1));
    request.dangerous_no_approval = config.dangerous_no_approval;
    request.rebuild_after_changes = config.rebuild_after_changes;
    request.verify_tool_access_after_rebuild = config.verify_tool_access_after_rebuild;
    request.command_ready = !intent_text.empty();
    set_bounded_cstr(request.codex_path, sizeof(request.codex_path), config.codex_path);
    set_bounded_cstr(request.working_directory, sizeof(request.working_directory), config.working_directory);
    set_bounded_cstr(request.rebuild_script_path, sizeof(request.rebuild_script_path), config.rebuild_script_path);
    set_bounded_cstr(request.rebuild_helper_path, sizeof(request.rebuild_helper_path), config.rebuild_helper_path);
    set_bounded_cstr(request.completion_message_path, sizeof(request.completion_message_path), config.completion_message_path);
    set_bounded_cstr(request.intent_text, sizeof(request.intent_text), intent_text.c_str());

    std::string task_prompt = intent_text;
    if (!task_prompt.empty()) {
        task_prompt += "\n\n";
    }
    task_prompt +=
            "Apply the requested repository change directly in this checkout. "
            "Run the commands and tests you need. When you finish, end with a brief plain-text summary "
            "that includes a line beginning with 'Manual requirements:' followed by either 'none' "
            "or the secrets / API keys the user must still add manually.";
    set_bounded_cstr(request.task_prompt, sizeof(request.task_prompt), task_prompt.c_str());
}

void init_hard_memory_request(
        llama_cognitive_hard_memory_request & request,
        int32_t command_id,
        int32_t origin,
        int32_t tool_job_id,
        const std::string & query_text,
        int32_t temporal_adapter_role) {
    request = {};
    request.command_id = command_id;
    request.origin = origin;
    request.tool_job_id = tool_job_id;
    request.operation = LLAMA_COG_HARD_MEMORY_OPERATION_QUERY;
    request.query.limit = 4;
    request.query.threshold = 0.0f;
    request.query.include_profile = true;
    request.query.use_temporal_self_hint = temporal_adapter_role >= 0;
    request.query.temporal_adapter_role = temporal_adapter_role;
    set_bounded_cstr(request.query.query, sizeof(request.query.query), query_text.c_str());
}

void init_hard_memory_write_request(
        llama_cognitive_hard_memory_request & request,
        int32_t command_id,
        int32_t origin,
        int32_t tool_job_id,
        const std::string & content_text) {
    request = {};
    request.command_id = command_id;
    request.origin = origin;
    request.tool_job_id = tool_job_id;
    request.operation = LLAMA_COG_HARD_MEMORY_OPERATION_WRITE;
    request.write_count = content_text.empty() ? 0 : 1;
    if (request.write_count <= 0) {
        return;
    }

    llama_hard_memory_write_item & item = request.write_items[0];
    item = {};
    item.is_static = false;
    item.primitive = llama_hard_memory_default_primitive();
    item.primitive.kind = LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME;
    item.primitive.domain = LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
    item.primitive.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
    item.primitive.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
    item.primitive.source_tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_WRITE;
    item.primitive.flags =
            LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_GAIN |
            LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_DERIVED;
    set_bounded_cstr(item.primitive.key, sizeof(item.primitive.key), "explicit-memory-write");
    set_bounded_cstr(item.primitive.title, sizeof(item.primitive.title), "explicit memory write");
    set_bounded_cstr(item.primitive.content, sizeof(item.primitive.content), content_text.c_str());
    set_bounded_cstr(item.primitive.tags[0], sizeof(item.primitive.tags[0]), "explicit_write");
}

void init_telegram_request(
        llama_telegram_relay_request & request,
        int32_t command_id,
        int32_t origin,
        int32_t tool_job_id,
        int32_t intent_kind,
        float urgency,
        const std::string & text,
        const std::string & dedupe_key) {
    request = {};
    request.command_id = command_id;
    request.origin = origin;
    request.tool_job_id = tool_job_id;
    request.intent_kind = intent_kind;
    request.urgency = clamp_unit(urgency);
    request.command_ready = !trim_ascii(text).empty();
    set_bounded_cstr(request.text, sizeof(request.text), trim_ascii(text).c_str());
    set_bounded_cstr(request.dedupe_key, sizeof(request.dedupe_key), trim_ascii(dedupe_key).c_str());
}

std::string summarize_telegram_result(const llama_telegram_relay_result & result) {
    const char * intent_label = "comment";
    if (result.intent_kind == LLAMA_TELEGRAM_RELAY_QUESTION) {
        intent_label = "question";
    } else if (result.intent_kind == LLAMA_TELEGRAM_RELAY_CONCLUSION) {
        intent_label = "conclusion";
    }

    std::string summary = std::string("telegram relay ");
    summary += result.delivered ? "delivered " : "failed for ";
    summary += intent_label;
    if (result.dedupe_key[0] != '\0') {
        summary += " key=" + trim_ascii(result.dedupe_key);
    }
    if (!result.delivered && result.error_text[0] != '\0') {
        summary += " error: " + trim_ascii(result.error_text);
    }
    return summary;
}

uint64_t fnv1a64_local(const std::string & text) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= (uint64_t) ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string summarize_bash_result(const llama_bash_tool_result & result) {
    auto summarize_channel = [](const char * text, size_t max_len) {
        std::string out = trim_ascii(text ? text : "");
        if (out.size() > max_len) {
            out.resize(max_len);
        }
        return out;
    };

    std::string summary;
    if (result.launch_failed) {
        summary = "bash command launch failed.";
    } else if (result.timed_out) {
        summary = "bash command timed out.";
    } else if (result.exit_code == 0) {
        summary = "bash command completed successfully.";
    } else {
        summary = "bash command failed.";
    }

    if (result.exit_code != 0) {
        summary += " exit_code=" + std::to_string(result.exit_code) + ".";
    }
    if (result.term_signal > 0) {
        summary += " signal=" + std::to_string(result.term_signal) + ".";
    }

    const std::string stdout_text = summarize_channel(result.stdout_text, 160);
    const std::string stderr_text = summarize_channel(result.stderr_text, 160);
    const std::string error_text = summarize_channel(result.error_text, 160);

    if (!stdout_text.empty()) {
        summary += " stdout: " + stdout_text;
    }
    if (!stderr_text.empty()) {
        summary += " stderr: " + stderr_text;
    }
    if (!error_text.empty()) {
        summary += " error: " + error_text;
    }
    if (result.truncated_stdout || result.truncated_stderr) {
        summary += " output_truncated=true.";
    }

    return summary;
}

std::string summarize_codex_result(const llama_codex_tool_result & result) {
    std::string summary = trim_ascii(result.summary_text);
    if (summary.empty()) {
        if (result.launch_failed) {
            summary = "codex tool launch failed.";
        } else if (result.exit_code == 0) {
            summary = "codex tool completed.";
        } else {
            summary = "codex tool failed.";
        }
    }

    if (result.repo_changed) {
        summary += " repo_changed=true.";
    }
    if (result.rebuild_attempted) {
        summary += result.rebuild_succeeded ?
                " rebuild=ok." :
                " rebuild=pending_or_failed.";
    }
    if (result.accessibility_verified) {
        summary += " accessibility_verified=true.";
    }

    const std::string manual = trim_ascii(result.manual_requirements);
    if (!manual.empty()) {
        summary += " manual_requirements: " + manual;
    }

    const std::string changed = trim_ascii(result.changed_files_excerpt);
    if (!changed.empty()) {
        summary += " changed_files: " + changed;
    }

    if (result.error_text[0] != '\0') {
        summary += " error: " + trim_ascii(result.error_text);
    }

    return summary;
}

std::string summarize_hard_memory_result(const llama_hard_memory_result & result) {
    std::string summary;
    if (!result.ok) {
        summary = "hard memory query failed.";
    } else if (result.result_count <= 0) {
        summary = "hard memory query completed with no results.";
    } else {
        summary = "hard memory query completed successfully.";
        summary += " results=" + std::to_string(result.result_count) + ".";
    }

    if (result.status_code > 0) {
        summary += " status=" + std::to_string(result.status_code) + ".";
    }
    if (result.retrieval_summary.max_similarity > 0.0f) {
        summary += " max_similarity=" + std::to_string(result.retrieval_summary.max_similarity) + ".";
    }
    if (result.error[0] != '\0') {
        summary += " error: " + trim_ascii(result.error);
    }
    if (result.result_count > 0 && result.results[0].title[0] != '\0') {
        summary += " top_hit: " + trim_ascii(result.results[0].title);
    }
    if (result.result_count > 0) {
        const int32_t narrative_limit = std::min(result.result_count, 2);
        for (int32_t i = 0; i < narrative_limit; ++i) {
            const auto & hit = result.results[i];
            const std::string title = trim_ascii(hit.title);
            std::string content = trim_ascii(hit.content);
            if (content.size() > 120) {
                content.resize(120);
                content = trim_ascii(content) + "...";
            }
            if (!title.empty()) {
                summary += " hit[" + std::to_string(i) + "] title=\"" + title + "\".";
            }
            if (!content.empty()) {
                summary += " hit[" + std::to_string(i) + "] summary=\"" + content + "\".";
            }
        }
    }

    return summary;
}

std::string summarize_hard_memory_archive_result(const llama_hard_memory_archive_trace & trace) {
    std::string summary;
    if (!trace.attempted) {
        summary = "hard-memory write was not attempted.";
    } else if (trace.archived) {
        summary = "hard-memory write completed successfully.";
    } else {
        summary = "hard-memory write failed.";
    }
    if (trace.status_code > 0) {
        summary += " status=" + std::to_string(trace.status_code) + ".";
    }
    if (trace.primitive_count > 0) {
        summary += " stored_items=" + std::to_string(trace.primitive_count) + ".";
    }
    if (trace.container_tag[0] != '\0') {
        summary += " container=" + trim_ascii(trace.container_tag) + ".";
    }
    if (trace.content_excerpt[0] != '\0') {
        summary += " excerpt=\"" + trim_ascii(trace.content_excerpt) + "\".";
    }
    if (trace.error[0] != '\0') {
        summary += " error: " + trim_ascii(trace.error);
    }
    return summary;
}

bool apply_tool_event_only(llama_context & ctx, const llama_self_state_event & event) {
    (void) ctx.self_state_set_channel_state(LLAMA_SELF_STATE_CHANNEL_ACTIVE);
    (void) ctx.self_state_note_tool_event();

    llama_self_state_feature_vector pre = {};
    llama_self_state_feature_vector post = {};
    if (!ctx.self_state_build_prewrite_features(event, &pre)) {
        return false;
    }
    if (!ctx.self_state_apply_prewrite(event, pre)) {
        return false;
    }
    if (!ctx.self_state_build_postwrite_features(event, &post)) {
        return false;
    }
    if (!ctx.self_state_apply_postwrite(event, post)) {
        return false;
    }

    return true;
}

std::string remediation_prompt_for_family(int32_t family) {
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
        case LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_LOCAL:
            return "Prefer nearby functional-bias perturbations that reduce favorable-state divergence without concentrating policy.";
        case LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_HISTORY:
            return "Prefer earlier archived functional-bias states when they outperform the current bias stance under DMN replay.";
        case LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_ORTHOGONAL:
            return "Prefer orthogonal functional-bias proposals that escape recent attractors while remaining robust under perturbation.";
        default:
            return "Prefer lower-divergence responses that move the system toward favorable self-state.";
    }
}

} // namespace

llama_codex_tool_config llama_codex_tool_default_config(void) {
    return codex_tool_default_config_local();
}

class llama_cognitive_runtime_impl {
public:
    explicit llama_cognitive_runtime_impl(llama_context & ctx) : ctx(ctx) {}

    static const llama_favorable_dimension_target * find_dimension(
            const llama_favorable_state_profile & profile,
            int32_t dimension_id) {
        for (int32_t i = 0; i < profile.dimension_count; ++i) {
            if (profile.dimensions[i].dimension_id == dimension_id) {
                return &profile.dimensions[i];
            }
        }
        return nullptr;
    }

    static void add_dimension(
            llama_favorable_state_profile & profile,
            int32_t dimension_id,
            float current_value,
            float target_value,
            float tolerance,
            float weight,
            bool stable) {
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
        const float contradiction_div = contradiction ? contradiction->divergence : 0.0f;
        const float uncertainty_div = uncertainty ? uncertainty->divergence : 0.0f;
        const float tool_div = std::max(
                tool_backlog ? tool_backlog->divergence : 0.0f,
                tool_readiness ? tool_readiness->divergence : 0.0f);
        const float dissatisfaction_div = dissatisfaction ? dissatisfaction->divergence : 0.0f;
        const float aggregate = clamp_unit(profile.aggregate_divergence);

        auto append_candidate = [&](int32_t family, int32_t risk_tier, int32_t subject_id, float expected_improvement, float confidence) {
            if (trace.candidate_count >= LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
                return;
            }

            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = family;
            candidate.risk_tier = risk_tier;
            candidate.subject_id = subject_id;
            candidate.functional_target_kind = LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT;
            candidate.functional_family = -1;
            candidate.process_entry_slot = -1;
            candidate.proposal_family = -1;
            candidate.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_NONE;
            candidate.snapshot_slot = -1;
            candidate.expected_improvement = clamp_unit(expected_improvement);
            candidate.confidence = clamp_unit(confidence);
            candidate.fragility_penalty = 0.0f;
            candidate.concentration_penalty = 0.0f;
            candidate.robustness_score = 0.0f;
            candidate.orthogonality = 0.0f;
            candidate.realized_score = candidate.expected_improvement;
            candidate.signed_advantage_vs_current = 0.0f;
        };

        llama_hard_memory_config hard_memory = {};
        const bool hard_memory_enabled =
                ctx.hard_memory_get_config(&hard_memory) &&
                hard_memory.enabled;

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

        const int32_t functional_family = select_functional_bias_family(profile);
        llama_process_functional_signature current_process_signature = {};
        (void) ctx.process_functional_get_current_signature(&current_process_signature);
        const llama_process_functional_signature compare_process_signature =
                rebind_process_signature_family(current_process_signature, functional_family);
        const int32_t process_entry_slot =
                find_process_entry_slot_for_signature(ctx, compare_process_signature);
        llama_functional_lora_snapshot_archive snapshot_archive = {};
    const bool have_functional_history =
            ctx.functional_lora_snapshot_archive_get(functional_family, &snapshot_archive) &&
            snapshot_archive.count > 0;
        const int32_t best_process_snapshot_slot = process_entry_slot >= 0 ?
                select_best_process_snapshot_slot(ctx, process_entry_slot) :
                -1;
        const bool have_process_history = best_process_snapshot_slot >= 0;
        const int32_t reserved_functional_slots =
                2 + (have_functional_history ? 1 : 0) +
                (process_entry_slot >= 0 ? (2 + (have_process_history ? 1 : 0)) : 0);
        const int32_t reserved_escalation_slots = 2;
        for (const auto & layer : runtime_loras) {
            if (trace.candidate_count >=
                    LLAMA_COUNTERFACTUAL_MAX_CANDIDATES - reserved_functional_slots - reserved_escalation_slots) {
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

        float fragility = 0.0f;
        float concentration = 0.0f;
        float robustness = 0.0f;
        if (trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            const float expected = score_functional_bias_candidate(
                    ctx,
                    profile,
                    LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT,
                    functional_family,
                    -1,
                    LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED,
                    -1,
                    0.0f,
                    &compare_process_signature,
                    &fragility,
                    &concentration,
                    &robustness);
            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_LOCAL;
            candidate.risk_tier = LLAMA_COUNTERFACTUAL_RISK_LOW;
            candidate.subject_id = functional_family;
            candidate.functional_target_kind = LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT;
            candidate.functional_family = functional_family;
            candidate.process_entry_slot = -1;
            candidate.proposal_family = LLAMA_FUNCTIONAL_BIAS_PROPOSAL_LOCAL;
            candidate.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED;
            candidate.snapshot_slot = -1;
            candidate.expected_improvement = expected;
            candidate.confidence = clamp_unit(0.52f + 0.20f * robustness);
            candidate.fragility_penalty = fragility;
            candidate.concentration_penalty = concentration;
            candidate.robustness_score = robustness;
            candidate.orthogonality = 0.0f;
            candidate.realized_score = clamp_unit(expected - fragility - concentration);
            candidate.signed_advantage_vs_current = clamp_signed_unit(candidate.realized_score - aggregate * 0.18f);
            best_low_risk_improvement = std::max(best_low_risk_improvement, candidate.expected_improvement);
        }

        int32_t best_snapshot_slot = -1;
        if (have_functional_history) {
            for (int32_t slot = 0; slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++slot) {
                const auto & item = snapshot_archive.items[slot];
                if (!item.valid) {
                    continue;
                }
                if (best_snapshot_slot < 0) {
                    best_snapshot_slot = slot;
                    continue;
                }
                const auto & best_item = snapshot_archive.items[best_snapshot_slot];
                if (item.robustness_score > best_item.robustness_score + 1.0e-6f ||
                        (std::fabs(item.robustness_score - best_item.robustness_score) <= 1.0e-6f &&
                         item.captured_at_us > best_item.captured_at_us)) {
                    best_snapshot_slot = slot;
                }
            }
        }
        if (best_snapshot_slot >= 0 && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            const auto & item = snapshot_archive.items[best_snapshot_slot];
            const float expected = score_functional_bias_candidate(
                    ctx,
                    profile,
                    LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT,
                    functional_family,
                    -1,
                    LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED,
                    best_snapshot_slot,
                    0.0f,
                    &compare_process_signature,
                    &fragility,
                    &concentration,
                    &robustness);
            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_HISTORY;
            candidate.risk_tier = LLAMA_COUNTERFACTUAL_RISK_LOW;
            candidate.subject_id = best_snapshot_slot;
            candidate.functional_target_kind = LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT;
            candidate.functional_family = functional_family;
            candidate.process_entry_slot = -1;
            candidate.proposal_family = LLAMA_FUNCTIONAL_BIAS_PROPOSAL_HISTORICAL;
            candidate.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED;
            candidate.snapshot_slot = best_snapshot_slot;
            candidate.expected_improvement = expected;
            candidate.confidence = clamp_unit(0.50f + 0.25f * robustness);
            candidate.fragility_penalty = fragility;
            candidate.concentration_penalty = concentration;
            candidate.robustness_score = robustness;
            candidate.orthogonality = item.dominant_direction_cosine;
            candidate.realized_score = clamp_unit(expected - fragility - concentration);
            candidate.signed_advantage_vs_current = clamp_signed_unit(candidate.realized_score - aggregate * 0.18f);
            best_low_risk_improvement = std::max(best_low_risk_improvement, candidate.expected_improvement);
        }

        if (trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            const float orthogonality = 0.05f;
            const float expected = score_functional_bias_candidate(
                    ctx,
                    profile,
                    LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT,
                    functional_family,
                    -1,
                    LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL,
                    -1,
                    orthogonality,
                    &compare_process_signature,
                    &fragility,
                    &concentration,
                    &robustness);
            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_ORTHOGONAL;
            candidate.risk_tier = LLAMA_COUNTERFACTUAL_RISK_LOW;
            candidate.subject_id = functional_family;
            candidate.functional_target_kind = LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT;
            candidate.functional_family = functional_family;
            candidate.process_entry_slot = -1;
            candidate.proposal_family = LLAMA_FUNCTIONAL_BIAS_PROPOSAL_ORTHOGONAL;
            candidate.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL;
            candidate.snapshot_slot = -1;
            candidate.expected_improvement = expected;
            candidate.confidence = clamp_unit(0.48f + 0.22f * robustness);
            candidate.fragility_penalty = fragility;
            candidate.concentration_penalty = concentration;
            candidate.robustness_score = robustness;
            candidate.orthogonality = orthogonality;
            candidate.realized_score = clamp_unit(expected - fragility - concentration);
            candidate.signed_advantage_vs_current = clamp_signed_unit(candidate.realized_score - aggregate * 0.18f);
            best_low_risk_improvement = std::max(best_low_risk_improvement, candidate.expected_improvement);
        }

        if (process_entry_slot >= 0 && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            const float expected = score_functional_bias_candidate(
                    ctx,
                    profile,
                    LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY,
                    functional_family,
                    process_entry_slot,
                    LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED,
                    -1,
                    0.0f,
                    &compare_process_signature,
                    &fragility,
                    &concentration,
                    &robustness);
            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_LOCAL;
            candidate.risk_tier = LLAMA_COUNTERFACTUAL_RISK_LOW;
            candidate.subject_id = process_entry_slot;
            candidate.functional_target_kind = LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY;
            candidate.functional_family = functional_family;
            candidate.process_entry_slot = process_entry_slot;
            candidate.proposal_family = LLAMA_FUNCTIONAL_BIAS_PROPOSAL_LOCAL;
            candidate.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED;
            candidate.snapshot_slot = -1;
            candidate.expected_improvement = expected;
            candidate.confidence = clamp_unit(0.50f + 0.20f * robustness);
            candidate.fragility_penalty = fragility;
            candidate.concentration_penalty = concentration;
            candidate.robustness_score = robustness;
            candidate.orthogonality = 0.0f;
            candidate.realized_score = clamp_unit(expected - fragility - concentration);
            candidate.signed_advantage_vs_current = clamp_signed_unit(candidate.realized_score - aggregate * 0.18f);
            best_low_risk_improvement = std::max(best_low_risk_improvement, candidate.expected_improvement);
        }

        if (have_process_history && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            llama_functional_lora_snapshot_info process_snapshot = {};
            (void) ctx.process_functional_snapshot_info_get(process_entry_slot, best_process_snapshot_slot, &process_snapshot);
            const float expected = score_functional_bias_candidate(
                    ctx,
                    profile,
                    LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY,
                    functional_family,
                    process_entry_slot,
                    LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED,
                    best_process_snapshot_slot,
                    0.0f,
                    &compare_process_signature,
                    &fragility,
                    &concentration,
                    &robustness);
            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_HISTORY;
            candidate.risk_tier = LLAMA_COUNTERFACTUAL_RISK_LOW;
            candidate.subject_id = best_process_snapshot_slot;
            candidate.functional_target_kind = LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY;
            candidate.functional_family = functional_family;
            candidate.process_entry_slot = process_entry_slot;
            candidate.proposal_family = LLAMA_FUNCTIONAL_BIAS_PROPOSAL_HISTORICAL;
            candidate.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED;
            candidate.snapshot_slot = best_process_snapshot_slot;
            candidate.expected_improvement = expected;
            candidate.confidence = clamp_unit(0.48f + 0.24f * robustness);
            candidate.fragility_penalty = fragility;
            candidate.concentration_penalty = concentration;
            candidate.robustness_score = robustness;
            candidate.orthogonality = process_snapshot.dominant_direction_cosine;
            candidate.realized_score = clamp_unit(expected - fragility - concentration);
            candidate.signed_advantage_vs_current = clamp_signed_unit(candidate.realized_score - aggregate * 0.18f);
            best_low_risk_improvement = std::max(best_low_risk_improvement, candidate.expected_improvement);
        }

        if (process_entry_slot >= 0 && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
            const float orthogonality = 0.05f;
            const float expected = score_functional_bias_candidate(
                    ctx,
                    profile,
                    LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY,
                    functional_family,
                    process_entry_slot,
                    LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL,
                    -1,
                    orthogonality,
                    &compare_process_signature,
                    &fragility,
                    &concentration,
                    &robustness);
            auto & candidate = trace.candidates[trace.candidate_count++];
            candidate.family = LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_ORTHOGONAL;
            candidate.risk_tier = LLAMA_COUNTERFACTUAL_RISK_LOW;
            candidate.subject_id = process_entry_slot;
            candidate.functional_target_kind = LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY;
            candidate.functional_family = functional_family;
            candidate.process_entry_slot = process_entry_slot;
            candidate.proposal_family = LLAMA_FUNCTIONAL_BIAS_PROPOSAL_ORTHOGONAL;
            candidate.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL;
            candidate.snapshot_slot = -1;
            candidate.expected_improvement = expected;
            candidate.confidence = clamp_unit(0.46f + 0.22f * robustness);
            candidate.fragility_penalty = fragility;
            candidate.concentration_penalty = concentration;
            candidate.robustness_score = robustness;
            candidate.orthogonality = orthogonality;
            candidate.realized_score = clamp_unit(expected - fragility - concentration);
            candidate.signed_advantage_vs_current = clamp_signed_unit(candidate.realized_score - aggregate * 0.18f);
            best_low_risk_improvement = std::max(best_low_risk_improvement, candidate.expected_improvement);
        }

        const bool need_discrete_tool_fallback =
                tool_div > 0.24f ||
                uncertainty_div > 0.30f ||
                best_low_risk_improvement < 0.18f;
        if (need_discrete_tool_fallback) {
            if (hard_memory_enabled && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
                int32_t temporal_role = LLAMA_ADAPTER_LORA_LAYER_PAST_WEEK;
                const int32_t serving_layer_count = ctx.serving_lora_stack_count();
                for (int32_t i = 0; i < serving_layer_count; ++i) {
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
                        0.05f + 0.22f * aggregate + 0.20f * tool_div + 0.16f * uncertainty_div,
                        0.62f);
                best_low_risk_improvement = std::max(
                        best_low_risk_improvement,
                        trace.candidates[trace.candidate_count - 1].expected_improvement);
            }
            if (tool_div > 0.18f && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
                append_candidate(
                        LLAMA_COUNTERFACTUAL_FAMILY_TOOL_CHOICE,
                        LLAMA_COUNTERFACTUAL_RISK_LOW,
                        LLAMA_FAVORABLE_DIM_TOOL_READINESS,
                        0.04f + 0.24f * tool_div + 0.12f * aggregate + 0.08f * contradiction_div,
                        0.56f);
                best_low_risk_improvement = std::max(
                        best_low_risk_improvement,
                        trace.candidates[trace.candidate_count - 1].expected_improvement);
            }
            if (tool_div > 0.28f && trace.candidate_count < LLAMA_COUNTERFACTUAL_MAX_CANDIDATES) {
                append_candidate(
                        LLAMA_COUNTERFACTUAL_FAMILY_TOOL_ARGUMENTS,
                        LLAMA_COUNTERFACTUAL_RISK_LOW,
                        LLAMA_FAVORABLE_DIM_TOOL_BACKLOG,
                        0.04f + 0.26f * tool_div + 0.10f * uncertainty_div + 0.08f * aggregate,
                        0.54f);
                best_low_risk_improvement = std::max(
                        best_low_risk_improvement,
                        trace.candidates[trace.candidate_count - 1].expected_improvement);
            }
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
                    LLAMA_TOOL_KIND_BASH_CLI;
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
        if (risk_tier == LLAMA_COUNTERFACTUAL_RISK_LOW) {
            trace.threshold = 0.18f;
        } else if (risk_tier == LLAMA_COUNTERFACTUAL_RISK_MEDIUM) {
            trace.threshold = 0.80f;
        } else {
            trace.threshold = 0.92f;
        }

        if (plan.action == LLAMA_REMEDIATION_ACTION_NONE) {
            trace.outcome = risk_tier >= LLAMA_COUNTERFACTUAL_RISK_MEDIUM ?
                    LLAMA_GOVERNANCE_OUTCOME_DENY :
                    LLAMA_GOVERNANCE_OUTCOME_DEFER;
        } else if (risk_tier >= LLAMA_COUNTERFACTUAL_RISK_HIGH ||
                (risk_tier == LLAMA_COUNTERFACTUAL_RISK_MEDIUM && trace.evidence < trace.threshold)) {
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
                LLAMA_TOOL_KIND_BASH_CLI,
                LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
                        LLAMA_COG_TOOL_DMN_ELIGIBLE |
                        LLAMA_COG_TOOL_REMEDIATION_SAFE |
                        LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT,
                LLAMA_COG_TOOL_LATENCY_MEDIUM,
                2,
                "bash"),
        make_tool_spec(
                LLAMA_TOOL_KIND_CODEX_CLI,
                LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
                        LLAMA_COG_TOOL_DMN_ELIGIBLE |
                        LLAMA_COG_TOOL_REMEDIATION_SAFE |
                        LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT,
                LLAMA_COG_TOOL_LATENCY_HIGH,
                3,
                "codex",
                "Use the local Codex CLI to implement a repository change and rebuild the runtime"),
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
                LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
                        LLAMA_COG_TOOL_DMN_ELIGIBLE |
                        LLAMA_COG_TOOL_REMEDIATION_SAFE |
                        LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT,
                LLAMA_COG_TOOL_LATENCY_HIGH,
                3,
                "hard_memory_write"),
        make_tool_spec(
                LLAMA_TOOL_KIND_TELEGRAM_RELAY,
                LLAMA_COG_TOOL_DMN_ELIGIBLE |
                        LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT,
                LLAMA_COG_TOOL_LATENCY_LOW,
                2,
                "telegram_relay",
                "Send a DMN-origin question, comment, or conclusion through the Telegram bridge"),
    };
    codex_config = codex_tool_default_config_local();
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
        tool_specs[i].description[LLAMA_COGNITIVE_TOOL_DESCRIPTION_MAX_CHARS - 1] = '\0';
        tool_specs[i].capability_id[LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS - 1] = '\0';
        tool_specs[i].owner_plugin_id[LLAMA_OPENCLAW_PLUGIN_ID_MAX_CHARS - 1] = '\0';
        tool_specs[i].provenance_namespace[LLAMA_OPENCLAW_NAMESPACE_MAX_CHARS - 1] = '\0';
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

bool llama_cognitive_loop::cognitive_command_rebind_tool(int32_t command_id, int32_t tool_spec_index) {
    if (tool_spec_index < 0 || tool_spec_index >= tool_spec_count) {
        return false;
    }

    const int32_t index = find_command_index(command_queue, command_count, command_id);
    if (index < 0) {
        return false;
    }

    llama_cognitive_command & command = command_queue[index];
    if (command.kind != LLAMA_COG_COMMAND_INVOKE_TOOL ||
            command.status != LLAMA_COG_COMMAND_STATUS_PENDING) {
        return false;
    }

    const llama_cognitive_tool_spec & spec = tool_specs[tool_spec_index];
    command.tool_kind = spec.tool_kind;
    command.tool_spec_index = tool_spec_index;
    set_bounded_cstr(command.capability_id, sizeof(command.capability_id), spec.capability_id);

    const llama_cognitive_plan_trace * process_plan = nullptr;
    int32_t transient_source_id = -1;
    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
        process_plan = active_plan.valid ? &active_plan : nullptr;
        transient_source_id = active_runner.episode_id;
        if (active_runner.pending_command_id == command_id) {
            active_runner.pending_tool_spec_index = tool_spec_index;
            active_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP;
        }
        if (last_active_trace.authoritative_turn.valid &&
                last_active_trace.episode_id == active_runner.episode_id) {
            last_active_trace.authoritative_turn.action = LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
            last_active_trace.authoritative_turn.selected_tool_kind = spec.tool_kind;
            last_active_trace.authoritative_turn.selected_tool_spec_index = tool_spec_index;
            last_active_trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_WAITING_ON_TOOL;
            set_bounded_cstr(
                    last_active_trace.authoritative_turn.selected_tool_name,
                    sizeof(last_active_trace.authoritative_turn.selected_tool_name),
                    spec.name);
        }
    } else if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
        process_plan = dmn_plan.valid ? &dmn_plan : nullptr;
        transient_source_id = dmn_runner.tick_id;
        if (dmn_runner.pending_command_id == command_id) {
            dmn_runner.pending_tool_spec_index = tool_spec_index;
            dmn_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP;
        }
        if (last_dmn_trace.authoritative_turn.valid &&
                last_dmn_trace.tick_id == dmn_runner.tick_id) {
            last_dmn_trace.authoritative_turn.action = LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
            last_dmn_trace.authoritative_turn.selected_tool_kind = spec.tool_kind;
            last_dmn_trace.authoritative_turn.selected_tool_spec_index = tool_spec_index;
            last_dmn_trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_WAITING_ON_TOOL;
            set_bounded_cstr(
                    last_dmn_trace.authoritative_turn.selected_tool_name,
                    sizeof(last_dmn_trace.authoritative_turn.selected_tool_name),
                    spec.name);
        }
    }

    (void) ctx.process_functional_set_execution(
            make_process_signature(
                    command.origin,
                    LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION,
                    LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP,
                    process_plan,
                    &spec,
                    transient_source_id));
    (void) ctx.functional_lora_activate(
            make_tool_phase_functional_decision(
                    command.origin,
                    LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP,
                    spec.tool_kind,
                    1.0f));

    return true;
}

bool llama_cognitive_loop::cognitive_authoritative_react_set_enabled(bool enabled) {
    (void) enabled;
    authoritative_react_control_enabled = true;
    return true;
}

bool llama_cognitive_loop::cognitive_active_authoritative_begin_tool(
        int32_t episode_id,
        uint32_t reason_mask,
        float priority,
        int32_t * out_command_id,
        int32_t * out_tool_job_id) {
    if (out_command_id) {
        *out_command_id = -1;
    }
    if (out_tool_job_id) {
        *out_tool_job_id = -1;
    }
    if (episode_id <= 0 || active_runner.episode_id != episode_id || !active_runner.active) {
        return false;
    }

    compact_command_queue(command_queue, &command_count);
    if (command_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
        return false;
    }

    const int32_t tool_job_id = next_tool_job_id++;
    const int32_t command_id = next_command_id++;
    init_command(
            command_queue[command_count++],
            command_id,
            LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
            LLAMA_COG_COMMAND_INVOKE_TOOL,
            episode_id,
            0,
            LLAMA_TOOL_KIND_NONE,
            -1,
            tool_job_id,
            reason_mask,
            priority,
            -1,
            LLAMA_COG_LOOP_PHASE_PREPARE_TOOL,
            nullptr);
    (void) ctx.self_state_upsert_tool_job(tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(priority));
    active_runner.pending_command_id = command_id;
    active_runner.pending_tool_spec_index = -1;
    active_runner.last_command_id = command_id;
    active_runner.waiting_on_tool = true;
    active_runner.completed = false;
    active_runner.active = true;
    active_runner.plan_status = LLAMA_COG_PLAN_STATUS_WAITING_TOOL;
    if (out_command_id) {
        *out_command_id = command_id;
    }
    if (out_tool_job_id) {
        *out_tool_job_id = tool_job_id;
    }
    return true;
}

bool llama_cognitive_loop::cognitive_active_authoritative_finish(int32_t episode_id, int32_t terminal_reason) {
    if (episode_id <= 0 || active_runner.episode_id != episode_id) {
        return false;
    }
    active_runner.waiting_on_tool = false;
    active_runner.pending_command_id = -1;
    active_runner.pending_tool_spec_index = -1;
    active_runner.completed = true;
    active_runner.active = false;
    active_runner.planning_active = false;
    active_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
    if (active_plan.valid) {
        active_plan.status = LLAMA_COG_PLAN_STATUS_COMPLETED;
        active_plan.terminal_reason = terminal_reason;
        if (active_plan.current_step_index >= 0 && active_plan.current_step_index < active_plan.step_count) {
            active_plan.steps[active_plan.current_step_index].status = LLAMA_COG_PLAN_STEP_STATUS_COMPLETED;
        }
    }
    return true;
}

bool llama_cognitive_loop::cognitive_dmn_authoritative_begin_tool(
        int32_t tick_id,
        uint32_t reason_mask,
        float priority,
        int32_t * out_command_id,
        int32_t * out_tool_job_id) {
    if (out_command_id) {
        *out_command_id = -1;
    }
    if (out_tool_job_id) {
        *out_tool_job_id = -1;
    }
    if (tick_id <= 0 || dmn_runner.tick_id != tick_id || !dmn_runner.active) {
        return false;
    }

    compact_command_queue(command_queue, &command_count);
    if (command_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
        return false;
    }

    const int32_t tool_job_id = next_tool_job_id++;
    const int32_t command_id = next_command_id++;
    init_command(
            command_queue[command_count++],
            command_id,
            LLAMA_COG_COMMAND_ORIGIN_DMN,
            LLAMA_COG_COMMAND_INVOKE_TOOL,
            0,
            tick_id,
            LLAMA_TOOL_KIND_NONE,
            -1,
            tool_job_id,
            reason_mask,
            priority,
            -1,
            LLAMA_COG_LOOP_PHASE_PREPARE_TOOL,
            nullptr);
    (void) ctx.self_state_upsert_tool_job(tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(priority));
    dmn_runner.pending_command_id = command_id;
    dmn_runner.pending_tool_spec_index = -1;
    dmn_runner.last_command_id = command_id;
    dmn_runner.waiting_on_tool = true;
    dmn_runner.completed = false;
    dmn_runner.active = true;
    dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_WAITING_TOOL;
    if (out_command_id) {
        *out_command_id = command_id;
    }
    if (out_tool_job_id) {
        *out_tool_job_id = tool_job_id;
    }
    return true;
}

bool llama_cognitive_loop::cognitive_dmn_authoritative_finish(int32_t tick_id, int32_t terminal_reason) {
    if (tick_id <= 0 || dmn_runner.tick_id != tick_id) {
        return false;
    }
    dmn_runner.waiting_on_tool = false;
    dmn_runner.pending_command_id = -1;
    dmn_runner.pending_tool_spec_index = -1;
    dmn_runner.completed = true;
    dmn_runner.active = false;
    dmn_runner.planning_active = false;
    dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
    if (dmn_plan.valid) {
        dmn_plan.status = LLAMA_COG_PLAN_STATUS_COMPLETED;
        dmn_plan.terminal_reason = terminal_reason;
        if (dmn_plan.current_step_index >= 0 && dmn_plan.current_step_index < dmn_plan.step_count) {
            dmn_plan.steps[dmn_plan.current_step_index].status = LLAMA_COG_PLAN_STEP_STATUS_COMPLETED;
        }
    }
    return true;
}

bool llama_cognitive_loop::cognitive_command_begin_external_wait(int32_t command_id) {
    const int32_t index = find_command_index(command_queue, command_count, command_id);
    if (index < 0) {
        return false;
    }

    const llama_cognitive_command & command = command_queue[index];
    if (command.kind != LLAMA_COG_COMMAND_INVOKE_TOOL) {
        return false;
    }

    // End the tool-call generation hold immediately once host execution begins.
    // This unloads the tool-selection family while keeping the command itself
    // pending so the active loop can resume on the later tool observation.
    (void) ctx.functional_lora_note_command_complete(command.origin);

    const bool preserve_active_planner =
            command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE &&
            active_runner.pending_command_id == command_id &&
            active_plan.valid;
    if (preserve_active_planner) {
        const llama_cognitive_tool_spec * tool_spec =
                command.tool_spec_index >= 0 && command.tool_spec_index < tool_spec_count ?
                        &tool_specs[command.tool_spec_index] :
                        nullptr;
        (void) ctx.process_functional_set_execution(
                make_process_signature(
                        command.origin,
                        LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION,
                        LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE,
                        &active_plan,
                        tool_spec,
                        active_runner.episode_id));
        (void) ctx.functional_lora_activate(
                make_planner_only_functional_decision(
                        command.origin,
                        LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE));
        active_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE;
        return true;
    }

    (void) ctx.process_functional_set_execution(llama_process_functional_signature {});
    (void) ctx.functional_lora_activate(make_inactive_functional_decision(command.origin));
    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
        active_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
    } else if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
        dmn_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
    }
    return true;
}

bool llama_cognitive_loop::cognitive_active_tool_emission_note(
        int32_t command_id,
        const llama_token * tokens,
        size_t n_tokens) {
    if (command_id <= 0 || !tokens || n_tokens == 0) {
        return false;
    }
    if (!tool_selection_episode_open ||
            active_runner.pending_command_id != command_id) {
        return false;
    }
    active_tool_emission_command_id = command_id;
    active_tool_emission_tokens.assign(tokens, tokens + n_tokens);
    return true;
}

bool llama_cognitive_loop::cognitive_active_planner_reasoning_note(
        int32_t episode_id,
        const llama_token * tokens,
        size_t n_tokens) {
    if (episode_id <= 0 || !tokens || n_tokens == 0) {
        return false;
    }
    if (active_runner.episode_id != episode_id &&
            last_active_trace.episode_id != episode_id) {
        return false;
    }
    active_planner_reasoning_episode_id = episode_id;
    active_planner_reasoning_tokens.assign(tokens, tokens + n_tokens);
    return true;
}

bool llama_cognitive_loop::codex_tool_configure(const llama_codex_tool_config & config) {
    codex_config = config;
    codex_config.timeout_ms = std::max(1000, codex_config.timeout_ms);
    codex_config.max_stdout_bytes = std::max(1, std::min(codex_config.max_stdout_bytes, LLAMA_CODEX_TOOL_STDOUT_MAX_CHARS - 1));
    codex_config.max_stderr_bytes = std::max(1, std::min(codex_config.max_stderr_bytes, LLAMA_CODEX_TOOL_STDERR_MAX_CHARS - 1));
    codex_config.codex_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    codex_config.working_directory[LLAMA_CODEX_TOOL_CWD_MAX_CHARS - 1] = '\0';
    codex_config.rebuild_script_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    codex_config.rebuild_helper_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    codex_config.completion_message_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    return true;
}

bool llama_cognitive_loop::codex_tool_get_config(llama_codex_tool_config * out_config) const {
    if (!out_config) {
        return false;
    }
    *out_config = codex_config;
    return true;
}

bool llama_cognitive_loop::codex_tool_get_last_result(llama_codex_tool_result * out_result) const {
    if (!out_result || !has_last_codex_result) {
        return false;
    }
    *out_result = last_codex_result;
    return true;
}

bool llama_cognitive_loop::cognitive_codex_tool_get_request(
        int32_t command_id,
        llama_codex_tool_request * out_request) const {
    if (!out_request) {
        return false;
    }
    const int32_t index = find_codex_request_index(codex_requests, codex_request_count, command_id);
    if (index < 0) {
        return false;
    }
    *out_request = codex_requests[index];
    return true;
}

bool llama_cognitive_loop::cognitive_codex_tool_set_request(const llama_codex_tool_request & request) {
    if (request.command_id <= 0 || request.tool_job_id <= 0) {
        return false;
    }
    const int32_t index = find_codex_request_index(codex_requests, codex_request_count, request.command_id);
    if (index >= 0) {
        codex_requests[index] = request;
        codex_requests[index].codex_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
        codex_requests[index].working_directory[LLAMA_CODEX_TOOL_CWD_MAX_CHARS - 1] = '\0';
        codex_requests[index].rebuild_script_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
        codex_requests[index].rebuild_helper_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
        codex_requests[index].completion_message_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
        codex_requests[index].intent_text[LLAMA_CODEX_TOOL_PROMPT_MAX_CHARS - 1] = '\0';
        codex_requests[index].task_prompt[LLAMA_CODEX_TOOL_PROMPT_MAX_CHARS - 1] = '\0';
        return true;
    }
    if (codex_request_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
        return false;
    }
    codex_requests[codex_request_count] = request;
    codex_requests[codex_request_count].codex_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    codex_requests[codex_request_count].working_directory[LLAMA_CODEX_TOOL_CWD_MAX_CHARS - 1] = '\0';
    codex_requests[codex_request_count].rebuild_script_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    codex_requests[codex_request_count].rebuild_helper_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    codex_requests[codex_request_count].completion_message_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS - 1] = '\0';
    codex_requests[codex_request_count].intent_text[LLAMA_CODEX_TOOL_PROMPT_MAX_CHARS - 1] = '\0';
    codex_requests[codex_request_count].task_prompt[LLAMA_CODEX_TOOL_PROMPT_MAX_CHARS - 1] = '\0';
    ++codex_request_count;
    return true;
}

bool llama_cognitive_loop::cognitive_telegram_relay_get_request(
        int32_t command_id,
        llama_telegram_relay_request * out_request) const {
    if (!out_request) {
        return false;
    }
    const int32_t index = find_telegram_request_index(telegram_requests, telegram_request_count, command_id);
    if (index < 0) {
        return false;
    }
    *out_request = telegram_requests[index];
    return true;
}

bool llama_cognitive_loop::cognitive_telegram_relay_set_request(const llama_telegram_relay_request & request) {
    if (request.command_id <= 0 || request.tool_job_id <= 0) {
        return false;
    }
    const int32_t index = find_telegram_request_index(telegram_requests, telegram_request_count, request.command_id);
    if (index >= 0) {
        telegram_requests[index] = request;
        telegram_requests[index].dedupe_key[LLAMA_TELEGRAM_RELAY_DEDUPE_MAX_CHARS - 1] = '\0';
        telegram_requests[index].text[LLAMA_TELEGRAM_RELAY_TEXT_MAX_CHARS - 1] = '\0';
        return true;
    }
    if (telegram_request_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
        return false;
    }
    telegram_requests[telegram_request_count] = request;
    telegram_requests[telegram_request_count].dedupe_key[LLAMA_TELEGRAM_RELAY_DEDUPE_MAX_CHARS - 1] = '\0';
    telegram_requests[telegram_request_count].text[LLAMA_TELEGRAM_RELAY_TEXT_MAX_CHARS - 1] = '\0';
    ++telegram_request_count;
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
        active_runner.pending_tool_spec_index = -1;
        active_runner.last_command_id = command_id;
        if (command.kind == LLAMA_COG_COMMAND_INVOKE_TOOL) {
            if (cancelled && host_state.pending_tool_followup_count > 0) {
                --host_state.pending_tool_followup_count;
            }
            active_runner.waiting_on_tool = !cancelled;
            active_runner.completed = cancelled;
            active_runner.active = !cancelled;
            active_runner.plan_status = cancelled ? LLAMA_COG_PLAN_STATUS_COMPLETED : LLAMA_COG_PLAN_STATUS_WAITING_TOOL;
        } else {
            active_runner.waiting_on_tool = false;
            active_runner.completed = true;
            active_runner.active = false;
            active_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
            active_runner.planning_active = false;
            if (active_plan.valid && active_plan.current_step_index >= 0 &&
                    active_plan.current_step_index < active_plan.step_count) {
                active_plan.steps[active_plan.current_step_index].status = LLAMA_COG_PLAN_STEP_STATUS_COMPLETED;
                active_plan.status = LLAMA_COG_PLAN_STATUS_COMPLETED;
            }
        }
    }

    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN &&
            dmn_runner.pending_command_id == command_id) {
        dmn_runner.pending_command_id = -1;
        dmn_runner.pending_tool_spec_index = -1;
        dmn_runner.last_command_id = command_id;
        if (command.kind == LLAMA_COG_COMMAND_INVOKE_TOOL) {
            dmn_runner.waiting_on_tool = !cancelled;
            dmn_runner.completed = cancelled;
            dmn_runner.active = !cancelled;
            dmn_runner.plan_status = cancelled ? LLAMA_COG_PLAN_STATUS_COMPLETED : LLAMA_COG_PLAN_STATUS_WAITING_TOOL;
        } else {
            dmn_runner.waiting_on_tool = false;
            dmn_runner.completed = true;
            dmn_runner.active = false;
            dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
            dmn_runner.planning_active = false;
            if (dmn_plan.valid && dmn_plan.current_step_index >= 0 &&
                    dmn_plan.current_step_index < dmn_plan.step_count) {
                dmn_plan.steps[dmn_plan.current_step_index].status = LLAMA_COG_PLAN_STEP_STATUS_COMPLETED;
                dmn_plan.status = LLAMA_COG_PLAN_STATUS_COMPLETED;
            }
        }
    }

    (void) ctx.functional_lora_note_command_complete(command.origin);

    if (cancelled &&
            command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE &&
            command.kind == LLAMA_COG_COMMAND_INVOKE_TOOL) {
        tool_selection_episode_open = false;
        tool_selection_before = {};
        tool_selection_tool_kind = LLAMA_TOOL_KIND_NONE;
        tool_selection_candidate_count = 0;
        tool_selection_uncertainty = 0.0f;
        active_tool_emission_command_id = -1;
        active_tool_emission_tokens.clear();
        if (active_planner_reasoning_episode_id == active_runner.episode_id) {
            active_planner_reasoning_episode_id = -1;
            active_planner_reasoning_tokens.clear();
        }
    }

    // Clear tool-specific bias before waiting on external I/O, but keep the
    // planner family active across the foreground ReAct loop so the next
    // assistant/tool-observation step remains under planner composition.
    {
        const bool preserve_active_planner =
                !cancelled &&
                command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE &&
                command.kind == LLAMA_COG_COMMAND_INVOKE_TOOL &&
                active_plan.valid;
        if (preserve_active_planner) {
            const llama_cognitive_tool_spec * tool_spec =
                    command.tool_spec_index >= 0 && command.tool_spec_index < tool_spec_count ?
                            &tool_specs[command.tool_spec_index] :
                            nullptr;
            (void) ctx.process_functional_set_execution(
                    make_process_signature(
                            command.origin,
                            LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION,
                            LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE,
                            &active_plan,
                            tool_spec,
                            active_runner.episode_id));
            (void) ctx.functional_lora_activate(
                    make_planner_only_functional_decision(
                            command.origin,
                            LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE));
            active_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE;
        } else {
            (void) ctx.process_functional_set_execution(llama_process_functional_signature {});
            (void) ctx.functional_lora_activate(make_inactive_functional_decision(command.origin));
            if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
                active_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
            } else if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
                dmn_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
            }
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
    trace.authoritative_turn.valid = authoritative_react_control_enabled;
    trace.authoritative_turn.turn_id = next_authoritative_turn_id;
    trace.authoritative_turn.origin = LLAMA_COG_COMMAND_ORIGIN_ACTIVE;
    trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_DRAFTING;

    if (event.role == LLAMA_SELF_STATE_EVENT_TOOL && host_state.pending_tool_followup_count > 0) {
        --host_state.pending_tool_followup_count;
    }

    if (resuming_active_tool) {
        active_runner.waiting_on_tool = false;
        active_runner.pending_command_id = -1;
        active_runner.pending_tool_spec_index = -1;
        active_runner.completed = false;
        active_runner.planning_active = true;
    } else {
        active_runner = {};
        active_runner.episode_id = trace.episode_id;
        active_runner.active = true;
        active_runner.max_steps = LLAMA_COG_LOOP_UNBOUNDED_MAX_STEPS;
        active_runner.pending_command_id = -1;
        active_runner.pending_tool_spec_index = -1;
        active_runner.last_command_id = -1;
        active_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
        active_runner.plan_id = -1;
        active_runner.plan_mode = LLAMA_COG_PLAN_MODE_NONE;
        active_runner.plan_status = LLAMA_COG_PLAN_STATUS_NONE;
        active_runner.plan_revision_count = 0;
        active_runner.current_plan_step = -1;
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

    (void) refresh_runtime_self_description(
            ctx,
            vocab,
            LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
            LLAMA_COG_LOOP_PHASE_PROPOSE,
            trace.episode_id,
            active_plan.valid ? active_plan.plan_id : -1,
            &current_canonical_self_model_revision,
            &current_emotive_moment_revision,
            &trace.context_window);
    trace.self_model_revision = current_canonical_self_model_revision;
    trace.emotive_moment_revision = current_emotive_moment_revision;
    trace.authoritative_turn.source_emotive_revision_id = current_emotive_moment_revision.revision_id;
    trace.authoritative_turn.source_context_revision = trace.context_window.head_revision;

    if ((mutable_event.flags & LLAMA_SELF_STATE_EVENT_ADMITTED) && mutable_event.tokens && mutable_event.n_tokens > 0) {
        (void) ctx.active_lora_ingest(mutable_event, &trace.postwrite_features);
    }

    const lexical_signals lex = analyze_event_lexicon(vocab, mutable_event);
    const float uncertainty = clamp_unit(std::max(trace.postwrite_features.uncertainty_score, trace.prewrite_features.uncertainty_score));
    const float inhibition = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_INHIBITION));
    const float goal_relevance = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_GOAL_RELEVANCE));
    const float tool_affordance = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_AFFORDANCE));
    const float answerability = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_ANSWERABILITY));
    const float recovery_urgency = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_RECOVERY_URGENCY));
    const float preference_uncertainty = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY));
    const bool tool_completed = (mutable_event.flags & LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED) != 0;
    const llama_favorable_state_profile pre_active_profile =
            llama_cognitive_runtime_impl(ctx).compute_favorable_profile();
    const llama_functional_outcome_snapshot active_before_snapshot =
            capture_functional_snapshot(ctx, pre_active_profile);

    auto activate_microphase = [&](int32_t microphase,
                                   float tool_affinity_seed,
                                   float planning_pressure_seed = 0.0f,
                                   float plan_complexity_seed = 0.0f,
                                   float plan_revision_seed = 0.0f) {
        const float memory_pressure_seed = clamp_unit(std::max(
                trace.prewrite_features.memory_write_pressure,
                trace.postwrite_features.memory_write_pressure));
        const float continuation_seed = clamp_unit(get_scalar_register(ctx, LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION));
        llama_functional_activation_decision decision = route_functional_activation(
                ctx,
                LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                microphase,
                active_before_snapshot,
                uncertainty,
                tool_affinity_seed,
                continuation_seed,
                memory_pressure_seed,
                planning_pressure_seed,
                plan_complexity_seed,
                plan_revision_seed);
        if ((microphase == LLAMA_FUNCTIONAL_MICROPHASE_TOOL_CLASS_SELECTION ||
                    microphase == LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP) &&
                (decision.activated_mask & (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION)) != 0) {
            decision.top_family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
        }
        const llama_cognitive_plan_trace * process_plan = nullptr;
        if (trace.plan.valid) {
            process_plan = &trace.plan;
        } else if (active_plan.valid) {
            process_plan = &active_plan;
        }
        int32_t process_tool_kind = LLAMA_TOOL_KIND_NONE;
        int32_t process_tool_spec_index = -1;
        if (process_plan &&
                process_plan->valid &&
                process_plan->current_step_index >= 0 &&
                process_plan->current_step_index < process_plan->step_count) {
            process_tool_kind = process_plan->steps[process_plan->current_step_index].tool_kind;
            process_tool_spec_index = process_plan->steps[process_plan->current_step_index].tool_spec_index;
        } else if (trace.tool_proposal.valid) {
            process_tool_kind = trace.tool_proposal.tool_kind;
            process_tool_spec_index = trace.tool_proposal.spec_index;
        }
        const llama_cognitive_tool_spec * process_tool_spec =
                process_tool_spec_index >= 0 && process_tool_spec_index < tool_spec_count ?
                        &tool_specs[process_tool_spec_index] :
                process_tool_kind != LLAMA_TOOL_KIND_NONE ?
                        find_tool_spec(tool_specs, tool_spec_count, process_tool_kind, nullptr) :
                        nullptr;
        (void) ctx.process_functional_set_execution(
                make_process_signature(
                        LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                        primary_functional_family(decision),
                        microphase,
                        process_plan,
                        process_tool_spec,
                        trace.episode_id));
        (void) ctx.functional_lora_activate(decision);
        llama_functional_lora_trace functional_trace = {};
        trace.functional_activation =
                ctx.functional_lora_get_last_trace(&functional_trace) ?
                functional_trace.last_activation :
                decision;
        active_runner.functional_microphase = microphase;
    };

    activate_microphase(
            mutable_event.role == LLAMA_SELF_STATE_EVENT_TOOL ?
                    LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION :
                    LLAMA_FUNCTIONAL_MICROPHASE_STATE_INTERPRET,
            0.0f);

    llama_self_social_state_info social = {};
    (void) ctx.self_state_get_social_state(&social);

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
    const uint32_t base_reason = build_active_reason_mask(mutable_event, lex, uncertainty, inhibition, tool_completed);

    if (authoritative_react_control_enabled) {
        const float control_score = clamp_unit(
                0.24f +
                0.20f * goal_relevance +
                0.16f * tool_affordance +
                0.14f * user_relevance +
                0.12f * latency_pressure +
                0.10f * (1.0f - inhibition) +
                0.10f * uncertainty);
        const int32_t plan_id = resuming_active_tool && active_plan.valid ? active_plan.plan_id : next_plan_id++;
        const int32_t plan_revision_count =
                resuming_active_tool && active_plan.valid ? active_plan.revision_count + 1 : 0;
        const float plan_complexity = clamp_unit(
                0.36f +
                0.18f * goal_relevance +
                0.16f * tool_affordance +
                0.14f * uncertainty +
                0.16f * latency_pressure);
        const float plan_ambiguity = clamp_unit(
                0.32f * uncertainty +
                0.24f * preference_uncertainty +
                0.18f * recovery_urgency +
                0.12f * (1.0f - answerability));

        activate_microphase(
                resuming_active_tool ? LLAMA_FUNCTIONAL_MICROPHASE_PLAN_REVISE : LLAMA_FUNCTIONAL_MICROPHASE_PLAN_DRAFT,
                tool_affordance,
                control_score,
                plan_complexity,
                clamp_unit((float) plan_revision_count / 4.0f));

        init_plan_trace(
                trace.plan,
                plan_id,
                LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                plan_revision_count,
                control_score,
                plan_ambiguity,
                base_reason);
        (void) append_plan_step(
                trace.plan,
                LLAMA_COG_PLAN_STEP_WAIT,
                LLAMA_COG_PLAN_STEP_STATUS_READY,
                LLAMA_TOOL_KIND_NONE,
                -1,
                base_reason,
                control_score,
                1,
                false);
        trace.plan.current_step_index = 0;
        trace.plan.status = LLAMA_COG_PLAN_STATUS_EXECUTING;
        trace.plan.terminal_reason = LLAMA_COG_TERMINAL_NONE;
        active_plan = trace.plan;

        activate_microphase(
                LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE,
                tool_affordance,
                control_score,
                plan_complexity,
                clamp_unit((float) plan_revision_count / 4.0f));

        trace.candidate_count = 0;
        trace.winner_action = LLAMA_ACTIVE_LOOP_ACTION_WAIT;
        trace.winner_score = control_score;
        trace.runner_up_action = LLAMA_ACTIVE_LOOP_ACTION_WAIT;
        trace.runner_up_score = control_score;
        trace.reason_mask = base_reason;
        trace.emit_allowed = false;
        trace.emit_noted = false;
        trace.tool_followup_expected = false;
        trace.authoritative_turn.valid = true;
        trace.authoritative_turn.turn_id = next_authoritative_turn_id++;
        trace.authoritative_turn.origin = LLAMA_COG_COMMAND_ORIGIN_ACTIVE;
        trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_VALIDATING;
        trace.authoritative_turn.action = LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT;
        trace.authoritative_turn.source_emotive_revision_id = trace.emotive_moment_revision.revision_id;
        trace.authoritative_turn.source_context_revision = trace.context_window.head_revision;
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_PROPOSE,
                LLAMA_COG_TERMINAL_NONE,
                active_runner.max_steps,
                resuming_active_tool ? active_runner.steps_taken + 1 : 1,
                true,
                false,
                tool_spec_count);
        active_runner.steps_taken = trace.loop_state.steps_taken;
        active_runner.waiting_on_tool = false;
        active_runner.pending_command_id = -1;
        active_runner.pending_tool_spec_index = -1;
        active_runner.completed = false;
        active_runner.active = true;
        active_runner.plan_id = trace.plan.plan_id;
        active_runner.plan_mode = trace.plan.mode;
        active_runner.plan_status = trace.plan.status;
        active_runner.plan_revision_count = trace.plan.revision_count;
        active_runner.current_plan_step = trace.plan.current_step_index;
        active_runner.planning_active = true;
        tool_selection_episode_open = false;
        last_active_trace = trace;
        host_state.last_foreground_time_us = trace.completed_time_us;
        if (out_trace) {
            *out_trace = trace;
        }
        return true;
    }

    trace.candidate_count = 0;
    trace.winner_action = LLAMA_ACTIVE_LOOP_ACTION_WAIT;
    trace.winner_score = 0.0f;
    trace.runner_up_action = LLAMA_ACTIVE_LOOP_ACTION_WAIT;
    trace.runner_up_score = 0.0f;
    trace.reason_mask = base_reason;
    trace.emit_allowed = false;
    trace.emit_noted = false;
    trace.tool_followup_expected = false;
    set_loop_state(
            trace.loop_state,
            LLAMA_COG_LOOP_PHASE_FINISH,
            LLAMA_COG_TERMINAL_GOVERNANCE_BLOCKED,
            active_runner.max_steps,
            resuming_active_tool ? active_runner.steps_taken + 1 : 1,
            false,
            false,
            tool_spec_count);
    active_runner.steps_taken = trace.loop_state.steps_taken;
    active_runner.waiting_on_tool = false;
    active_runner.pending_command_id = -1;
    active_runner.pending_tool_spec_index = -1;
    active_runner.completed = true;
    active_runner.active = false;
    active_runner.planning_active = false;
    active_plan = {};

    trace.completed_time_us = llama_time_us();
    trace.plan = active_plan;
    (void) ctx.shared_cognitive_context_get_window(&trace.context_window);
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
    if (active_plan.valid && active_plan.plan_id == last_active_trace.plan.plan_id) {
        active_plan.status = LLAMA_COG_PLAN_STATUS_COMPLETED;
        if (active_plan.current_step_index >= 0 && active_plan.current_step_index < active_plan.step_count) {
            active_plan.steps[active_plan.current_step_index].status = LLAMA_COG_PLAN_STEP_STATUS_COMPLETED;
        }
        last_active_trace.plan = active_plan;
    }
    active_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
    active_runner.planning_active = false;
    active_runner.current_plan_step = active_plan.current_step_index;
    if (active_planner_reasoning_episode_id == episode_id) {
        active_planner_reasoning_episode_id = -1;
        active_planner_reasoning_tokens.clear();
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
    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
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
    last_temporal_self_trace = {};

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
            LLAMA_COG_LOOP_UNBOUNDED_MAX_STEPS,
            0,
            true,
            false,
            tool_spec_count);
    dmn_runner = {};
    dmn_runner.tick_id = next_tick_id;
    dmn_runner.completed = true;
    dmn_runner.max_steps = LLAMA_COG_LOOP_UNBOUNDED_MAX_STEPS;
    dmn_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
    dmn_runner.self_model_revision_id = current_self_model_revision.valid ? current_self_model_revision.revision_id : -1;
    (void) refresh_runtime_self_description(
            ctx,
            vocab,
            LLAMA_COG_COMMAND_ORIGIN_DMN,
            LLAMA_COG_LOOP_PHASE_FINISH,
            dmn_runner.tick_id,
            -1,
            &current_canonical_self_model_revision,
            &current_emotive_moment_revision,
            &trace.context_window);
    trace.canonical_self_model_revision = current_canonical_self_model_revision;
    trace.emotive_moment_revision = current_emotive_moment_revision;
    dmn_runner.emotive_revision_id = current_emotive_moment_revision.valid ? current_emotive_moment_revision.revision_id : -1;
    dmn_runner.context_revision = trace.context_window.head_revision;
    dmn_runner.turn_id = -1;
    trace.self_model_revision = current_self_model_revision;
    trace.functional_activation = make_inactive_functional_decision(LLAMA_COG_COMMAND_ORIGIN_DMN);
    (void) ctx.process_functional_set_execution(llama_process_functional_signature {});
    (void) ctx.functional_lora_activate(trace.functional_activation);
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
    auto enqueue_command = [&](int32_t kind, int32_t tick_id, int32_t tool_kind, int32_t tool_spec_index, int32_t tool_job_id, uint32_t reason_mask, float priority, int32_t source_family, int32_t loop_phase, const llama_cognitive_tool_spec * spec) -> int32_t {
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
                tool_spec_index,
                tool_job_id,
                reason_mask,
                priority,
                source_family,
                loop_phase,
                spec);
        return command_id;
    };

    const llama_cognitive_runtime_impl runtime(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
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
    trace.tool_spec_index = -1;
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
    last_temporal_self_trace = {};
    dmn_runner = {};
    dmn_runner.max_steps = LLAMA_COG_LOOP_UNBOUNDED_MAX_STEPS;
    dmn_runner.pending_command_id = -1;
    dmn_runner.pending_tool_spec_index = -1;
    dmn_runner.last_command_id = -1;
    dmn_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
    dmn_runner.plan_id = -1;
    dmn_runner.plan_mode = LLAMA_COG_PLAN_MODE_NONE;
    dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_NONE;
    dmn_runner.plan_revision_count = 0;
    dmn_runner.current_plan_step = -1;
    dmn_runner.self_model_revision_id = current_self_model_revision.valid ? current_self_model_revision.revision_id : -1;
    (void) refresh_runtime_self_description(
            ctx,
            vocab,
            LLAMA_COG_COMMAND_ORIGIN_DMN,
            LLAMA_COG_LOOP_PHASE_PROPOSE,
            next_tick_id,
            -1,
            &current_canonical_self_model_revision,
            &current_emotive_moment_revision,
            &trace.context_window);
    trace.canonical_self_model_revision = current_canonical_self_model_revision;
    trace.emotive_moment_revision = current_emotive_moment_revision;
    dmn_runner.emotive_revision_id = current_emotive_moment_revision.valid ? current_emotive_moment_revision.revision_id : -1;
    dmn_runner.context_revision = trace.context_window.head_revision;
    trace.authoritative_turn.valid = authoritative_react_control_enabled;
    trace.authoritative_turn.origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    trace.authoritative_turn.source_emotive_revision_id = current_emotive_moment_revision.revision_id;
    trace.authoritative_turn.source_context_revision = trace.context_window.head_revision;

    if (!pressure_crossed(trace.pressure)) {
        trace.admitted = false;
        trace.burst_count = 0;
        trace.functional_activation = make_inactive_functional_decision(LLAMA_COG_COMMAND_ORIGIN_DMN);
        (void) ctx.process_functional_set_execution(llama_process_functional_signature {});
        (void) ctx.functional_lora_activate(trace.functional_activation);
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_FINISH,
                LLAMA_COG_TERMINAL_PRESSURE_NOT_ADMITTED,
                dmn_runner.max_steps,
                0,
                false,
                false,
                tool_spec_count);
        dmn_runner.completed = true;
        dmn_runner.max_steps = LLAMA_COG_LOOP_UNBOUNDED_MAX_STEPS;
        trace.self_model_revision = current_self_model_revision;
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
    const llama_functional_outcome_snapshot dmn_before_snapshot =
            capture_functional_snapshot(ctx, last_favorable_profile);
    auto activate_microphase = [&](int32_t microphase,
                                   float tool_affinity_seed,
                                   float memory_pressure_seed,
                                   float planning_pressure_seed = 0.0f,
                                   float plan_complexity_seed = 0.0f,
                                   float plan_revision_seed = 0.0f) {
        llama_functional_activation_decision decision = route_functional_activation(
                ctx,
                LLAMA_COG_COMMAND_ORIGIN_DMN,
                microphase,
                dmn_before_snapshot,
                clamp_unit(trace.pressure.uncertainty),
                tool_affinity_seed,
                clamp_unit(trace.pressure.continuation),
                memory_pressure_seed,
                planning_pressure_seed,
                plan_complexity_seed,
                plan_revision_seed);
        if ((microphase == LLAMA_FUNCTIONAL_MICROPHASE_TOOL_CLASS_SELECTION ||
                    microphase == LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP) &&
                (decision.activated_mask & (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION)) != 0) {
            decision.top_family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
        }
        const llama_cognitive_plan_trace * process_plan = nullptr;
        if (trace.plan.valid) {
            process_plan = &trace.plan;
        } else if (dmn_plan.valid) {
            process_plan = &dmn_plan;
        }
        int32_t process_tool_kind = LLAMA_TOOL_KIND_NONE;
        int32_t process_tool_spec_index = -1;
        if (process_plan &&
                process_plan->valid &&
                process_plan->current_step_index >= 0 &&
                process_plan->current_step_index < process_plan->step_count) {
            process_tool_kind = process_plan->steps[process_plan->current_step_index].tool_kind;
            process_tool_spec_index = process_plan->steps[process_plan->current_step_index].tool_spec_index;
        } else if (trace.tool_kind != LLAMA_TOOL_KIND_NONE) {
            process_tool_kind = trace.tool_kind;
            process_tool_spec_index = trace.tool_spec_index;
        }
        const llama_cognitive_tool_spec * process_tool_spec =
                process_tool_spec_index >= 0 && process_tool_spec_index < tool_spec_count ?
                        &tool_specs[process_tool_spec_index] :
                process_tool_kind != LLAMA_TOOL_KIND_NONE ?
                        find_tool_spec(tool_specs, tool_spec_count, process_tool_kind, nullptr) :
                        nullptr;
        (void) ctx.process_functional_set_execution(
                make_process_signature(
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        primary_functional_family(decision),
                        microphase,
                        process_plan,
                        process_tool_spec,
                        trace.tick_id));
        (void) ctx.functional_lora_activate(decision);
        llama_functional_lora_trace functional_trace = {};
        trace.functional_activation =
                ctx.functional_lora_get_last_trace(&functional_trace) ?
                functional_trace.last_activation :
                decision;
        dmn_runner.functional_microphase = microphase;
    };
    activate_microphase(
            LLAMA_FUNCTIONAL_MICROPHASE_SELF_OBSERVE,
            clamp_unit(trace.pressure.tool_delta),
            clamp_unit((float) ctx.self_state_working_memory_count() / 8.0f));
    runtime.select_reactivation_targets(trace);
    trace.seed_source_mask = LLAMA_DMN_SEED_SOURCE_REGISTERS | LLAMA_DMN_SEED_SOURCE_SELF_STATE;
    if (trace.reactivation_count > 0) {
        trace.seed_source_mask |= LLAMA_DMN_SEED_SOURCE_REACTIVATION;
    }
    if (ctx.self_state_working_memory_count() > 0) {
        trace.seed_source_mask |= LLAMA_DMN_SEED_SOURCE_WORKING_MEM;
    }
    llama_self_tool_state_info initial_tool_state = {};
    if (ctx.self_state_get_tool_state(&initial_tool_state) &&
            initial_tool_state.last_update_monotonic_ms >= 0) {
        trace.seed_source_mask |= LLAMA_DMN_SEED_SOURCE_TOOL_STATE;
    }
    activate_microphase(
            LLAMA_FUNCTIONAL_MICROPHASE_SELF_FORECAST,
            clamp_unit(trace.pressure.tool_delta),
            clamp_unit((float) ctx.self_state_working_memory_count() / 8.0f));
    last_counterfactual_trace = runtime.compute_counterfactual_trace(last_favorable_profile);
    last_counterfactual_trace.simulated_user = {};
    for (int32_t i = 0; i < last_counterfactual_trace.candidate_count; ++i) {
        auto & candidate = last_counterfactual_trace.candidates[i];
        if (candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_MESSAGE_VARIANT) {
            continue;
        }
        if (simulate_user_reply(ctx, last_favorable_profile, candidate.family, last_counterfactual_trace.simulated_user)) {
            candidate.expected_improvement = clamp_unit(
                    candidate.expected_improvement +
                    0.25f * last_counterfactual_trace.simulated_user.signed_self_state_outcome);
            candidate.confidence = clamp_unit(std::max(
                    candidate.confidence,
                    0.45f + 0.45f * last_counterfactual_trace.simulated_user.simulation_confidence));
        }
        break;
    }
    trace.simulated_user = last_counterfactual_trace.simulated_user;
    const float evolution_uncertainty_before = clamp_unit(
            get_scalar_register(ctx, LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY));
    const llama_counterfactual_candidate * temporal_ablation_candidate =
            best_temporal_ablation_candidate(last_counterfactual_trace);
    const bool temporal_self_triggered =
            evolution_uncertainty_before >= LLAMA_TEMPORAL_SELF_TRIGGER_THRESHOLD &&
            temporal_ablation_candidate != nullptr &&
            now_us >= temporal_self_next_trigger_us;
    activate_microphase(
            LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_GENERATE,
            clamp_unit(trace.pressure.tool_delta),
            clamp_unit((float) ctx.self_state_working_memory_count() / 8.0f));
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
    const float reflective_pressure = clamp_unit(
            0.45f * trace.pressure.contradiction +
            0.35f * trace.pressure.uncertainty +
            0.20f * trace.pressure.reactivation);

    const int32_t working_memory_count = ctx.self_state_working_memory_count();
    auto preferred_dmn_spec = [&](int32_t fallback_tool_kind, int32_t * out_index) -> const llama_cognitive_tool_spec * {
        const int32_t preferred_tool_kind =
                last_remediation_plan.tool_kind != LLAMA_TOOL_KIND_NONE ?
                        last_remediation_plan.tool_kind :
                        fallback_tool_kind;
        return find_tool_spec(tool_specs, tool_spec_count, preferred_tool_kind, out_index);
    };

    llama_self_model_state_info model_state = {};
    llama_self_social_state_info social_state = {};
    llama_self_tool_state_info tool_state = {};
    (void) ctx.self_state_get_model_state(&model_state);
    (void) ctx.self_state_get_social_state(&social_state);
    (void) ctx.self_state_get_tool_state(&tool_state);

    std::ostringstream input_signature;
    input_signature.setf(std::ios::fixed);
    input_signature.precision(3);
    input_signature << trace.pressure.contradiction << '|'
                    << trace.pressure.uncertainty << '|'
                    << trace.pressure.reactivation << '|'
                    << trace.pressure.goals << '|'
                    << trace.pressure.tool_delta << '|'
                    << trace.pressure.continuation << '|'
                    << trace.favorable_divergence << '|'
                    << model_state.belief_summary.residual_allostatic_pressure << '|'
                    << model_state.belief_summary.promotion_readiness << '|'
                    << model_state.extension_summary.active_count << '|'
                    << model_state.extension_summary.gain_signal << '|'
                    << social_state.trust << '|'
                    << social_state.dissatisfaction << '|'
                    << tool_state.pending_jobs << '|'
                    << tool_state.completed_jobs << '|'
                    << working_memory_count;
    for (int32_t i = 0; i < trace.reactivation_count; ++i) {
        input_signature << "|r" << i << '=' << trace.reactivation_targets[i].priority;
    }
    const uint64_t dmn_input_hash = fnv1a64_local(input_signature.str());
    const bool dmn_input_changed = dmn_input_hash != current_dmn_input_hash;
    const float dmn_materiality = dmn_input_changed ? clamp_unit(
            0.18f * trace.pressure.total +
            0.18f * trace.pressure.contradiction +
            0.16f * trace.pressure.uncertainty +
            0.14f * trace.favorable_divergence +
            0.12f * model_state.belief_summary.residual_allostatic_pressure +
            0.12f * std::fabs(model_state.extension_summary.gain_signal) +
            0.10f * social_state.dissatisfaction) : 0.0f;

    current_self_model_revision.valid = true;
    current_self_model_revision.input_hash = dmn_input_hash;
    current_self_model_revision.materiality_score = dmn_materiality;
    current_self_model_revision.requires_prompt_regen =
            dmn_input_changed && dmn_materiality >= 0.10f;
    if (current_self_model_revision.requires_prompt_regen) {
        current_self_model_revision.revision_id = next_dmn_self_model_revision_id++;
    }
    trace.self_model_revision = current_self_model_revision;

    const float user_contact_affinity = clamp_unit(
            0.34f * social_relevance +
            0.22f * continuation +
            0.22f * social_state.dissatisfaction +
            0.12f * trace.pressure.goals +
            (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR ? 0.20f : 0.0f));
    const float relay_tool_affinity = clamp_unit(
            user_contact_affinity +
            (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR ? 0.15f : 0.0f));
    const float non_social_tool_affinity = clamp_unit(
            tool_affinity +
            (last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO ? 0.20f : 0.0f));
    current_dmn_input_hash = dmn_input_hash;
    dmn_runner.self_model_revision_id = trace.self_model_revision.revision_id;

    int32_t selected_dmn_spec_index = -1;
    const llama_cognitive_tool_spec * selected_dmn_spec = nullptr;
    const llama_cognitive_tool_spec * telegram_spec = find_tool_spec(tool_specs, tool_spec_count, LLAMA_TOOL_KIND_TELEGRAM_RELAY, &selected_dmn_spec_index);
    selected_dmn_spec_index = -1;

    if (authoritative_react_control_enabled) {
        const float control_score = clamp_unit(std::max(std::max(
                relay_tool_affinity,
                non_social_tool_affinity),
                reflective_pressure));
        trace.authoritative_turn.valid = true;
        trace.authoritative_turn.turn_id = next_authoritative_turn_id++;
        trace.authoritative_turn.origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
        trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_VALIDATING;
        trace.authoritative_turn.action = LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT;
        trace.authoritative_turn.source_emotive_revision_id = trace.emotive_moment_revision.revision_id;
        trace.authoritative_turn.source_context_revision = trace.context_window.head_revision;
        init_plan_trace(
                trace.plan,
                next_plan_id++,
                LLAMA_COG_COMMAND_ORIGIN_DMN,
                0,
                control_score,
                clamp_unit(0.20f * trace.pressure.uncertainty + 0.10f * trace.pressure.contradiction),
                base_reason);
        (void) append_plan_step(
                trace.plan,
                LLAMA_COG_PLAN_STEP_WAIT,
                LLAMA_COG_PLAN_STEP_STATUS_READY,
                LLAMA_TOOL_KIND_NONE,
                last_remediation_plan.source_family,
                base_reason,
                control_score,
                1,
                false);
        trace.plan.current_step_index = 0;
        trace.plan.status = LLAMA_COG_PLAN_STATUS_EXECUTING;
        trace.plan.terminal_reason = LLAMA_COG_TERMINAL_NONE;
        set_loop_state(
                trace.loop_state,
                LLAMA_COG_LOOP_PHASE_PROPOSE,
                LLAMA_COG_TERMINAL_NONE,
                dmn_runner.max_steps,
                1,
                true,
                false,
                tool_spec_count);
        trace.winner_action = LLAMA_DMN_ACTION_SILENT;
        trace.winner_score = control_score;
        trace.tool_kind = LLAMA_TOOL_KIND_NONE;
        trace.tool_spec_index = -1;
        trace.reasoning_text[0] = '\0';
        dmn_plan = trace.plan;
        dmn_runner.steps_taken = 1;
        dmn_runner.waiting_on_tool = false;
        dmn_runner.pending_command_id = -1;
        dmn_runner.pending_tool_spec_index = -1;
        dmn_runner.completed = false;
        dmn_runner.active = true;
        dmn_runner.plan_id = trace.plan.plan_id;
        dmn_runner.plan_mode = trace.plan.mode;
        dmn_runner.plan_status = trace.plan.status;
        dmn_runner.plan_revision_count = trace.plan.revision_count;
        dmn_runner.current_plan_step = trace.plan.current_step_index;
        dmn_runner.turn_id = trace.authoritative_turn.turn_id;
        dmn_runner.planning_active = true;
        last_dmn_trace = trace;
        host_state.last_dmn_time_us = now_us;
        if (out_trace) {
            *out_trace = trace;
        }
        return true;
    }

    const bool wants_user_contact =
            relay_tool_affinity >= 0.62f &&
            inhibition < 0.82f &&
            telegram_spec != nullptr;
    const bool wants_tool =
            non_social_tool_affinity >= 0.52f &&
            last_governance_trace.outcome != LLAMA_GOVERNANCE_OUTCOME_DENY;
    const bool wants_reflection =
            (trace.pressure.contradiction >= 0.18f ||
             trace.pressure.uncertainty >= 0.18f ||
             trace.reactivation_count > 0 ||
             model_state.belief_summary.residual_allostatic_pressure >= 0.16f);

    trace.candidate_count = 0;
    trace.runner_up_action = LLAMA_DMN_ACTION_SILENT;
    trace.runner_up_score = 0.0f;
    uint32_t selected_reason_mask = base_reason;

    if (wants_user_contact) {
        trace.winner_action = LLAMA_DMN_ACTION_INVOKE_TOOL;
        trace.winner_score = relay_tool_affinity;
        selected_reason_mask |= LLAMA_COG_REASON_SOCIAL_RELEVANCE;
        selected_dmn_spec = telegram_spec;
        selected_dmn_spec_index = find_tool_spec_index(tool_specs, tool_spec_count, LLAMA_TOOL_KIND_TELEGRAM_RELAY);
    } else if (wants_tool) {
        trace.winner_action = LLAMA_DMN_ACTION_INVOKE_TOOL;
        trace.winner_score = non_social_tool_affinity;
        selected_reason_mask |= LLAMA_COG_REASON_TOOL_AFFORDANCE;
        const int32_t preferred_tool_kind =
                last_remediation_plan.tool_kind != LLAMA_TOOL_KIND_NONE ?
                        last_remediation_plan.tool_kind :
                        LLAMA_TOOL_KIND_HARD_MEMORY_QUERY;
        selected_dmn_spec = find_tool_spec(tool_specs, tool_spec_count, preferred_tool_kind, &selected_dmn_spec_index);
        if (!selected_dmn_spec) {
            trace.winner_action = wants_reflection ? LLAMA_DMN_ACTION_INTERNAL_WRITE : LLAMA_DMN_ACTION_SILENT;
            trace.winner_score = wants_reflection ? reflective_pressure : clamp_unit(1.0f - inhibition);
            selected_reason_mask |= LLAMA_COG_REASON_HIGH_UNCERTAINTY;
        }
    } else if (wants_reflection) {
        trace.winner_action = LLAMA_DMN_ACTION_INTERNAL_WRITE;
        trace.winner_score = reflective_pressure;
        selected_reason_mask |= LLAMA_COG_REASON_REACTIVATION_TARGET;
    } else {
        trace.winner_action = LLAMA_DMN_ACTION_SILENT;
        trace.winner_score = clamp_unit(0.55f * inhibition + 0.45f * (1.0f - trace.pressure.total));
        selected_reason_mask |= LLAMA_COG_REASON_HIGH_INHIBITION;
    }

    if (working_memory_count >= 4) {
        trace.maintenance_mask |= LLAMA_DMN_MAINTENANCE_COMPRESS_WORKING_MEMORY;
    }
    (void) ctx.functional_lora_snapshot_maintain(now_us);
    (void) ctx.process_functional_snapshot_maintain(now_us);
    if (ctx.past_lora_tick(now_us)) {
        trace.maintenance_mask |= LLAMA_DMN_MAINTENANCE_PAST_LORA_TICK;
    }
    if (trace.reactivation_count > 0) {
        trace.maintenance_mask |= LLAMA_DMN_MAINTENANCE_REFRESH_REACTIVATION;
    }

    const bool compression_eligible =
            (trace.maintenance_mask & LLAMA_DMN_MAINTENANCE_COMPRESS_WORKING_MEMORY) != 0;
    const float plan_complexity = clamp_unit(
            0.35f +
            0.20f * (trace.winner_action == LLAMA_DMN_ACTION_INTERNAL_WRITE ? 1.0f : 0.0f) +
            0.20f * (trace.winner_action == LLAMA_DMN_ACTION_INVOKE_TOOL ? 1.0f : 0.0f) +
            0.15f * clamp_unit((float) trace.reactivation_count / (float) LLAMA_DMN_MAX_REACTIVATION_TARGETS) +
            0.10f * (compression_eligible ? 1.0f : 0.0f));
    const float plan_ambiguity = clamp_unit(
            0.18f * trace.pressure.uncertainty +
            0.14f * trace.pressure.contradiction +
            0.10f * (selected_dmn_spec ? 0.0f : 1.0f));
    activate_microphase(
            LLAMA_FUNCTIONAL_MICROPHASE_PLAN_DRAFT,
            tool_affinity,
            clamp_unit((float) working_memory_count / 8.0f),
            trace.winner_score,
            plan_complexity,
            0.0f);
    init_plan_trace(
            trace.plan,
            next_plan_id++,
            LLAMA_COG_COMMAND_ORIGIN_DMN,
            0,
            trace.winner_score,
            plan_ambiguity,
            selected_reason_mask);
    if (trace.winner_action == LLAMA_DMN_ACTION_INTERNAL_WRITE) {
        (void) append_plan_step(
                trace.plan,
                LLAMA_COG_PLAN_STEP_INTERNAL_WRITE,
                LLAMA_COG_PLAN_STEP_STATUS_READY,
                LLAMA_TOOL_KIND_NONE,
                last_remediation_plan.source_family,
                selected_reason_mask,
                trace.winner_score,
                1,
                false);
    } else if (trace.winner_action == LLAMA_DMN_ACTION_INVOKE_TOOL) {
        (void) append_plan_step(
                trace.plan,
                LLAMA_COG_PLAN_STEP_INVOKE_TOOL,
                LLAMA_COG_PLAN_STEP_STATUS_READY,
                selected_dmn_spec ? selected_dmn_spec->tool_kind : last_remediation_plan.tool_kind,
                last_remediation_plan.source_family,
                selected_reason_mask,
                trace.winner_score,
                1,
                false,
                selected_dmn_spec,
                selected_dmn_spec_index);
        (void) append_plan_step(
                trace.plan,
                LLAMA_COG_PLAN_STEP_OBSERVE_TOOL,
                LLAMA_COG_PLAN_STEP_STATUS_PENDING,
                selected_dmn_spec ? selected_dmn_spec->tool_kind : last_remediation_plan.tool_kind,
                last_remediation_plan.source_family,
                selected_reason_mask,
                trace.winner_score,
                1,
                true,
                selected_dmn_spec,
                selected_dmn_spec_index);
    } else {
        const int32_t governance_status =
                last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DENY ||
                last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DEFER ?
                        LLAMA_COG_PLAN_STEP_STATUS_BLOCKED :
                        LLAMA_COG_PLAN_STEP_STATUS_COMPLETED;
        (void) append_plan_step(
                trace.plan,
                LLAMA_COG_PLAN_STEP_WAIT,
                governance_status,
                LLAMA_TOOL_KIND_NONE,
                last_remediation_plan.source_family,
                selected_reason_mask,
                trace.winner_score,
                1,
                false);
    }
    trace.plan.current_step_index = first_ready_plan_step(trace.plan);
    trace.plan.status =
            trace.winner_action == LLAMA_DMN_ACTION_INVOKE_TOOL ? LLAMA_COG_PLAN_STATUS_WAITING_TOOL :
            trace.winner_action == LLAMA_DMN_ACTION_SILENT &&
                    (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DENY ||
                     last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_DEFER) ?
                            LLAMA_COG_PLAN_STATUS_BLOCKED :
                            LLAMA_COG_PLAN_STATUS_EXECUTING;
    trace.tool_kind = selected_dmn_spec ? selected_dmn_spec->tool_kind : LLAMA_TOOL_KIND_NONE;
    trace.tool_spec_index = selected_dmn_spec_index;
    copy_cstr_local(trace.reasoning_text, compose_dmn_reasoning_trace(trace, selected_dmn_spec));
    dmn_plan = trace.plan;
    dmn_runner.plan_id = trace.plan.plan_id;
    dmn_runner.plan_mode = trace.plan.mode;
    dmn_runner.plan_status = trace.plan.status;
    dmn_runner.plan_revision_count = trace.plan.revision_count;
    dmn_runner.current_plan_step = trace.plan.current_step_index;
    dmn_runner.planning_active = true;

    {
        (void) emit_cognitive_artifact_text(
                ctx,
                llama_model_get_vocab(&ctx.get_model()),
                trim_ascii(trace.reasoning_text),
                LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
                LLAMA_SELF_COG_ARTIFACT_DMN_PLAN,
                LLAMA_COG_COMMAND_ORIGIN_DMN,
                LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE,
                trace.tick_id,
                trace.plan.plan_id);
    }

    if (compression_eligible) {
        activate_microphase(
                LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_COMPRESSION,
                tool_affinity,
                clamp_unit((float) working_memory_count / 8.0f));
    } else {
        activate_microphase(
                LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE,
                tool_affinity,
                clamp_unit((float) working_memory_count / 8.0f));
    }
    activate_microphase(
            LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE,
            tool_affinity,
            clamp_unit((float) working_memory_count / 8.0f),
            trace.winner_score,
            plan_complexity,
            0.0f);

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

    auto build_dmn_telegram_text = [&]() -> std::string {
        if (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR &&
                last_governance_trace.repair_message[0] != '\0') {
            return trim_ascii(last_governance_trace.repair_message);
        }
        if (trace.reasoning_text[0] != '\0') {
            return trim_ascii(trace.reasoning_text);
        }
        if (trace.emotive_moment_revision.valid) {
            return trim_ascii(trace.emotive_moment_revision.text);
        }
        return "Background reflection: continue from the shared cognitive context.";
    };

    auto build_dmn_telegram_intent = [&]() -> int32_t {
        if (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR) {
            return LLAMA_TELEGRAM_RELAY_CONCLUSION;
        }
        if (trace.pressure.uncertainty >= 0.45f || trace.pressure.contradiction >= 0.35f) {
            return LLAMA_TELEGRAM_RELAY_QUESTION;
        }
        return LLAMA_TELEGRAM_RELAY_COMMENT;
    };

    if (trace.winner_action == LLAMA_DMN_ACTION_INTERNAL_WRITE) {
        std::ostringstream internal_summary;
        internal_summary << trim_ascii(trace.reasoning_text);
        if (internal_summary.tellp() > 0) {
            internal_summary << "\n";
        }
        internal_summary.setf(std::ios::fixed);
        internal_summary.precision(3);
        internal_summary << "Observation: contradiction=" << trace.pressure.contradiction
                         << " uncertainty=" << trace.pressure.uncertainty
                         << " continuation=" << trace.pressure.continuation
                         << " remediation=" << last_remediation_plan.action
                         << " governance=" << last_governance_trace.outcome
                         << " divergence=" << trace.favorable_divergence;
        (void) emit_cognitive_artifact_text(
                ctx,
                llama_model_get_vocab(&ctx.get_model()),
                internal_summary.str(),
                LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
                LLAMA_SELF_COG_ARTIFACT_DMN_INTERNAL_WRITE,
                LLAMA_COG_COMMAND_ORIGIN_DMN,
                LLAMA_COG_LOOP_PHASE_OBSERVE,
                trace.tick_id,
                trace.plan.plan_id);
        activate_microphase(
                compression_eligible ? LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_AUDIT : LLAMA_FUNCTIONAL_MICROPHASE_POST_ACTION_REFLECTION,
                tool_affinity,
                clamp_unit((float) working_memory_count / 8.0f));
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
                -1,
                LLAMA_SELF_TOOL_JOB_COMPLETED,
                0.55f * trace.pressure.counterfactual + 0.45f * trace.favorable_divergence,
                trace.pressure.continuation);
        dmn_runner.steps_taken = 2;
        if (trace.plan.step_count > 0) {
            trace.plan.steps[0].status = LLAMA_COG_PLAN_STEP_STATUS_COMPLETED;
            trace.plan.current_step_index = 0;
            trace.plan.terminal_reason = LLAMA_COG_TERMINAL_INTERNAL_WRITE_READY;
        }

        const bool should_contact_followup =
                telegram_spec != nullptr &&
                relay_tool_affinity >= 0.62f &&
                (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR ||
                 (continuation > 0.55f && inhibition < 0.60f));
        const bool should_gather_followup =
                last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO &&
                last_governance_trace.outcome != LLAMA_GOVERNANCE_OUTCOME_DENY;

        if (should_gather_followup && dmn_runner.steps_taken < dmn_runner.max_steps) {
            trace.winner_action = LLAMA_DMN_ACTION_INVOKE_TOOL;
            trace.tool_job_id = next_tool_job_id++;
            int32_t spec_index = -1;
            const llama_cognitive_tool_spec * spec = preferred_dmn_spec(
                    last_remediation_plan.tool_kind == LLAMA_TOOL_KIND_NONE ?
                            LLAMA_TOOL_KIND_HARD_MEMORY_QUERY :
                            last_remediation_plan.tool_kind,
                    &spec_index);
            trace.tool_kind = spec ? spec->tool_kind : last_remediation_plan.tool_kind;
            trace.tool_spec_index = spec_index;
            (void) ctx.self_state_upsert_tool_job(trace.tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(trace.winner_score));
            last_remediation_plan.tool_job_id = trace.tool_job_id;
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
            (void) append_plan_step(
                    trace.plan,
                    LLAMA_COG_PLAN_STEP_INVOKE_TOOL,
                    LLAMA_COG_PLAN_STEP_STATUS_ACTIVE,
                    trace.tool_kind,
                    last_remediation_plan.source_family,
                    selected_reason_mask,
                    trace.winner_score,
                    spec ? std::max(1, spec->max_steps_reserved) : 1,
                    false,
                    spec,
                    spec_index);
            (void) append_plan_step(
                    trace.plan,
                    LLAMA_COG_PLAN_STEP_OBSERVE_TOOL,
                    LLAMA_COG_PLAN_STEP_STATUS_PENDING,
                    trace.tool_kind,
                    last_remediation_plan.source_family,
                    selected_reason_mask,
                    trace.winner_score,
                    1,
                    true,
                    spec,
                    spec_index);
            trace.plan.current_step_index = trace.plan.step_count - 1;
            trace.plan.status = LLAMA_COG_PLAN_STATUS_WAITING_TOOL;
            trace.plan.terminal_reason = LLAMA_COG_TERMINAL_TOOL_REQUIRED;
            dmn_runner.steps_taken = 3;
            dmn_runner.waiting_on_tool = true;
            dmn_runner.pending_command_id = enqueue_command(
                LLAMA_COG_COMMAND_INVOKE_TOOL,
                trace.tick_id,
                    trace.tool_kind,
                    spec_index,
                    trace.tool_job_id,
                    selected_reason_mask,
                    trace.winner_score,
                    last_remediation_plan.source_family,
                    trace.loop_state.phase,
                    spec);
            dmn_runner.pending_tool_spec_index = spec_index;
            dmn_runner.last_command_id = dmn_runner.pending_command_id > 0 ? dmn_runner.pending_command_id : dmn_runner.last_command_id;
            llama_bash_tool_config bash_config = {};
            llama_codex_tool_config codex_config = {};
            if (dmn_runner.pending_command_id > 0) {
                if (trace.tool_kind == LLAMA_TOOL_KIND_BASH_CLI &&
                        ctx.bash_tool_get_config(&bash_config)) {
                    llama_bash_tool_request request = {};
                    llama_cognitive_tool_spec resolved_spec = {};
                    const llama_cognitive_tool_spec * spec =
                            llama_cognitive_tool_spec_get(&ctx, dmn_runner.pending_tool_spec_index, &resolved_spec) == 0 ?
                            &resolved_spec : nullptr;
                    init_bash_request(
                            request,
                            bash_config,
                            dmn_runner.pending_command_id,
                            LLAMA_COG_COMMAND_ORIGIN_DMN,
                            trace.tool_job_id,
                            "inspect repository and runtime state",
                            infer_bash_command("inspect repository and runtime state", spec));
                    (void) ctx.bash_tool_set_request(request);
                } else if (trace.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_QUERY) {
                    llama_cognitive_hard_memory_request request = {};
                    init_hard_memory_request(
                            request,
                            dmn_runner.pending_command_id,
                            LLAMA_COG_COMMAND_ORIGIN_DMN,
                            trace.tool_job_id,
                            "relevant hard memory for current runtime remediation and self-state",
                            LLAMA_SERVING_LORA_LAYER_ACTIVE);
                    (void) ctx.hard_memory_set_request(request);
                } else if (trace.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                    llama_cognitive_hard_memory_request request = {};
                    init_hard_memory_write_request(
                            request,
                            dmn_runner.pending_command_id,
                            LLAMA_COG_COMMAND_ORIGIN_DMN,
                            trace.tool_job_id,
                            "durable runtime residue worth explicitly archiving to hard memory");
                    (void) ctx.hard_memory_set_request(request);
                } else if (trace.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI &&
                        ctx.codex_tool_get_config(&codex_config)) {
                    llama_codex_tool_request request = {};
                    init_codex_request(
                            request,
                            codex_config,
                            dmn_runner.pending_command_id,
                            LLAMA_COG_COMMAND_ORIGIN_DMN,
                            trace.tool_job_id,
                            "implement a repo-local runtime or tool change and rebuild if needed");
                    (void) ctx.cognitive_codex_tool_set_request(request);
                } else if (trace.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY) {
                    llama_telegram_relay_request request = {};
                    init_telegram_request(
                            request,
                            dmn_runner.pending_command_id,
                            LLAMA_COG_COMMAND_ORIGIN_DMN,
                            trace.tool_job_id,
                            build_dmn_telegram_intent(),
                            relay_tool_affinity,
                            build_dmn_telegram_text(),
                            "dmn-relay-" + std::to_string(trace.tick_id) + "-" + std::to_string(trace.tool_job_id));
                    (void) ctx.cognitive_telegram_relay_set_request(request);
                }
            }
        } else if (should_contact_followup && dmn_runner.steps_taken < dmn_runner.max_steps) {
            trace.winner_action = LLAMA_DMN_ACTION_INVOKE_TOOL;
            trace.tool_job_id = next_tool_job_id++;
            trace.tool_kind = LLAMA_TOOL_KIND_TELEGRAM_RELAY;
            trace.tool_spec_index = find_tool_spec_index(tool_specs, tool_spec_count, LLAMA_TOOL_KIND_TELEGRAM_RELAY);
            (void) ctx.self_state_upsert_tool_job(trace.tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(trace.winner_score));
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
                    telegram_spec,
                    trace.tool_spec_index,
                    trace.tool_kind,
                    selected_reason_mask | LLAMA_COG_REASON_SOCIAL_RELEVANCE,
                    last_remediation_plan.source_family,
                    1,
                    relay_tool_affinity,
                    trace.tool_job_id);
            (void) append_plan_step(
                    trace.plan,
                    LLAMA_COG_PLAN_STEP_INVOKE_TOOL,
                    LLAMA_COG_PLAN_STEP_STATUS_ACTIVE,
                    trace.tool_kind,
                    last_remediation_plan.source_family,
                    selected_reason_mask | LLAMA_COG_REASON_SOCIAL_RELEVANCE,
                    trace.winner_score,
                    1,
                    false,
                    telegram_spec,
                    trace.tool_spec_index);
            (void) append_plan_step(
                    trace.plan,
                    LLAMA_COG_PLAN_STEP_OBSERVE_TOOL,
                    LLAMA_COG_PLAN_STEP_STATUS_PENDING,
                    trace.tool_kind,
                    last_remediation_plan.source_family,
                    selected_reason_mask | LLAMA_COG_REASON_SOCIAL_RELEVANCE,
                    trace.winner_score,
                    1,
                    true,
                    telegram_spec,
                    trace.tool_spec_index);
            trace.plan.current_step_index = trace.plan.step_count - 1;
            trace.plan.status = LLAMA_COG_PLAN_STATUS_WAITING_TOOL;
            trace.plan.terminal_reason = LLAMA_COG_TERMINAL_TOOL_REQUIRED;
            dmn_runner.steps_taken = 3;
            dmn_runner.waiting_on_tool = true;
            dmn_runner.pending_command_id = enqueue_command(
                    LLAMA_COG_COMMAND_INVOKE_TOOL,
                    trace.tick_id,
                    trace.tool_kind,
                    trace.tool_spec_index,
                    trace.tool_job_id,
                    selected_reason_mask | LLAMA_COG_REASON_SOCIAL_RELEVANCE,
                    trace.winner_score,
                    last_remediation_plan.source_family,
                    trace.loop_state.phase,
                    telegram_spec);
            dmn_runner.pending_tool_spec_index = trace.tool_spec_index;
            dmn_runner.last_command_id = dmn_runner.pending_command_id > 0 ? dmn_runner.pending_command_id : dmn_runner.last_command_id;
            if (dmn_runner.pending_command_id > 0) {
                llama_telegram_relay_request request = {};
                init_telegram_request(
                        request,
                        dmn_runner.pending_command_id,
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        trace.tool_job_id,
                        build_dmn_telegram_intent(),
                        relay_tool_affinity,
                        build_dmn_telegram_text(),
                        "dmn-relay-" + std::to_string(trace.tick_id) + "-" + std::to_string(trace.tool_job_id));
                (void) ctx.cognitive_telegram_relay_set_request(request);
            }
        } else {
            dmn_runner.completed = true;
            dmn_runner.active = false;
            trace.plan.status = LLAMA_COG_PLAN_STATUS_COMPLETED;
        }
    } else if (trace.winner_action == LLAMA_DMN_ACTION_INVOKE_TOOL) {
        activate_microphase(
                LLAMA_FUNCTIONAL_MICROPHASE_TOOL_CLASS_SELECTION,
                tool_affinity,
                clamp_unit((float) working_memory_count / 8.0f));
        trace.tool_job_id = next_tool_job_id++;
        trace.tool_kind = selected_dmn_spec ? selected_dmn_spec->tool_kind : last_remediation_plan.tool_kind;
        trace.tool_spec_index = selected_dmn_spec_index;
        (void) ctx.self_state_upsert_tool_job(trace.tool_job_id, LLAMA_SELF_TOOL_JOB_PENDING, clamp_unit(trace.winner_score));
        if (last_remediation_plan.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO) {
            last_remediation_plan.tool_job_id = trace.tool_job_id;
        }
        int32_t spec_index = selected_dmn_spec_index;
        const llama_cognitive_tool_spec * spec = selected_dmn_spec;
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
        if (trace.plan.step_count > 0) {
            trace.plan.steps[0].status = LLAMA_COG_PLAN_STEP_STATUS_ACTIVE;
            trace.plan.current_step_index = trace.plan.step_count > 1 ? 1 : 0;
            trace.plan.status = LLAMA_COG_PLAN_STATUS_WAITING_TOOL;
            trace.plan.terminal_reason = LLAMA_COG_TERMINAL_TOOL_REQUIRED;
        }
        dmn_runner.steps_taken = 2;
        dmn_runner.waiting_on_tool = true;
        dmn_runner.pending_command_id = enqueue_command(
                LLAMA_COG_COMMAND_INVOKE_TOOL,
                trace.tick_id,
                trace.tool_kind,
                spec_index,
                trace.tool_job_id,
                selected_reason_mask,
                trace.winner_score,
                last_remediation_plan.source_family,
                trace.loop_state.phase,
                spec);
        dmn_runner.pending_tool_spec_index = spec_index;
        dmn_runner.last_command_id = dmn_runner.pending_command_id > 0 ? dmn_runner.pending_command_id : dmn_runner.last_command_id;
        llama_bash_tool_config bash_config = {};
        llama_codex_tool_config codex_config = {};
        bool prepared_tool_request = false;
        if (dmn_runner.pending_command_id > 0) {
            if (trace.tool_kind == LLAMA_TOOL_KIND_BASH_CLI &&
                    ctx.bash_tool_get_config(&bash_config)) {
                activate_microphase(
                        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP,
                        tool_affinity,
                        clamp_unit((float) working_memory_count / 8.0f));
                llama_bash_tool_request request = {};
                init_bash_request(
                        request,
                        bash_config,
                        dmn_runner.pending_command_id,
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        trace.tool_job_id,
                        "inspect repository and runtime state",
                        infer_bash_command("inspect repository and runtime state", spec));
                (void) ctx.bash_tool_set_request(request);
                prepared_tool_request = true;
            } else if (trace.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_QUERY) {
                activate_microphase(
                        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP,
                        tool_affinity,
                        clamp_unit((float) working_memory_count / 8.0f));
                llama_cognitive_hard_memory_request request = {};
                init_hard_memory_request(
                        request,
                        dmn_runner.pending_command_id,
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        trace.tool_job_id,
                        "relevant hard memory for runtime remediation and self-state",
                        LLAMA_SERVING_LORA_LAYER_ACTIVE);
                (void) ctx.hard_memory_set_request(request);
                prepared_tool_request = true;
            } else if (trace.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                activate_microphase(
                        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP,
                        tool_affinity,
                        clamp_unit((float) working_memory_count / 8.0f));
                llama_cognitive_hard_memory_request request = {};
                init_hard_memory_write_request(
                        request,
                        dmn_runner.pending_command_id,
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        trace.tool_job_id,
                        "durable runtime residue worth explicitly archiving to hard memory");
                (void) ctx.hard_memory_set_request(request);
                prepared_tool_request = true;
            } else if (trace.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI &&
                    ctx.codex_tool_get_config(&codex_config)) {
                activate_microphase(
                        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP,
                        tool_affinity,
                        clamp_unit((float) working_memory_count / 8.0f));
                llama_codex_tool_request request = {};
                init_codex_request(
                        request,
                        codex_config,
                        dmn_runner.pending_command_id,
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        trace.tool_job_id,
                        "implement a repo-local runtime or tool change and rebuild if needed");
                (void) ctx.cognitive_codex_tool_set_request(request);
                prepared_tool_request = true;
            } else if (trace.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY) {
                activate_microphase(
                        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP,
                        tool_affinity,
                        clamp_unit((float) working_memory_count / 8.0f));
                llama_telegram_relay_request request = {};
                init_telegram_request(
                        request,
                        dmn_runner.pending_command_id,
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        trace.tool_job_id,
                        build_dmn_telegram_intent(),
                        relay_tool_affinity,
                        build_dmn_telegram_text(),
                        "dmn-relay-" + std::to_string(trace.tick_id) + "-" + std::to_string(trace.tool_job_id));
                (void) ctx.cognitive_telegram_relay_set_request(request);
                prepared_tool_request = true;
            }
            if (prepared_tool_request) {
                trace.functional_activation.microphase = LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP;
                trace.functional_activation.top_family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
                trace.functional_activation.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
                trace.functional_activation.activated_mask |= (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION);
            }
        }
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
                dmn_runner.max_steps,
                1,
                false,
                false,
                tool_spec_count);
        dmn_runner.steps_taken = 1;
        dmn_runner.completed = true;
        dmn_runner.active = false;
        if (trace.plan.step_count > 0) {
            trace.plan.current_step_index = 0;
            trace.plan.terminal_reason = terminal_reason;
            trace.plan.status =
                    terminal_reason == LLAMA_COG_TERMINAL_GOVERNANCE_BLOCKED ?
                            LLAMA_COG_PLAN_STATUS_BLOCKED :
                            LLAMA_COG_PLAN_STATUS_COMPLETED;
        }
    }

    dmn_plan = trace.plan;
    dmn_runner.plan_id = trace.plan.plan_id;
    dmn_runner.plan_mode = trace.plan.mode;
    dmn_runner.plan_status = trace.plan.status;
    dmn_runner.plan_revision_count = trace.plan.revision_count;
    dmn_runner.current_plan_step = trace.plan.current_step_index;

    if (last_governance_trace.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR && !last_governance_trace.repair_rendered) {
        last_governance_trace.repair_rendered = true;
        set_repair_message(last_governance_trace, "I may have pushed us in the wrong direction. I am re-evaluating and will correct course.");
    }

    llama_favorable_state_profile post_profile = runtime.compute_favorable_profile();
    const llama_functional_outcome_snapshot dmn_after_snapshot =
            capture_functional_snapshot(ctx, post_profile);
    const functional_delta_summary dmn_deltas =
            compute_functional_deltas(dmn_before_snapshot, dmn_after_snapshot);
    if (last_remediation_plan.applied) {
        last_remediation_plan.post_divergence = post_profile.aggregate_divergence;
    } else {
        last_remediation_plan.post_divergence = last_remediation_plan.pre_divergence;
    }

    if (temporal_self_triggered) {
        const float temporal_reference_score = clamp_unit(
                temporal_ablation_candidate->expected_improvement *
                (0.65f + 0.35f * temporal_ablation_candidate->confidence));
        const float active_path_score = clamp_unit(
                0.42f * std::max(0.0f, dmn_deltas.delta_favorable) +
                0.28f * std::max(0.0f, dmn_deltas.delta_efficiency) +
                0.18f * std::max(0.0f, dmn_deltas.delta_recovery) +
                0.12f * std::max(0.0f, dmn_deltas.delta_answerability));
        const float signed_advantage = clamp_signed_unit(active_path_score - temporal_reference_score);
        const float efficiency_advantage = clamp_signed_unit(
                dmn_deltas.delta_efficiency -
                0.50f * temporal_ablation_candidate->expected_improvement);
        int32_t outcome = LLAMA_TEMPORAL_SELF_IMPROVEMENT_TIE;
        if (signed_advantage > 0.08f && efficiency_advantage > -0.05f) {
            outcome = LLAMA_TEMPORAL_SELF_IMPROVEMENT_REWARD;
        } else if (signed_advantage < -0.08f && efficiency_advantage < 0.05f) {
            outcome = LLAMA_TEMPORAL_SELF_IMPROVEMENT_DAMPEN;
        }

        llama_self_state_datetime time_info = {};
        const int64_t monotonic_ms = ctx.self_state_get_datetime(&time_info) ?
                time_info.monotonic_ms :
                (int64_t) (now_us / 1000);
        (void) ctx.active_temporal_encoding_bias_apply(signed_advantage, efficiency_advantage, monotonic_ms);
        (void) ctx.self_state_note_validated_progress(signed_advantage, efficiency_advantage);

        llama_active_temporal_encoding_bias bias = {};
        (void) ctx.active_temporal_encoding_bias_get(&bias);
        last_temporal_self_trace.valid = true;
        last_temporal_self_trace.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
        last_temporal_self_trace.selected_temporal_role = temporal_ablation_candidate->subject_id;
        last_temporal_self_trace.counterfactual_family = temporal_ablation_candidate->family;
        last_temporal_self_trace.outcome = outcome;
        last_temporal_self_trace.evolution_uncertainty_before = evolution_uncertainty_before;
        last_temporal_self_trace.evolution_uncertainty_after = clamp_unit(
                get_scalar_register(ctx, LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY));
        last_temporal_self_trace.signed_advantage = signed_advantage;
        last_temporal_self_trace.efficiency_advantage = efficiency_advantage;
        last_temporal_self_trace.active_reward_bias = bias.reward_bias;
        last_temporal_self_trace.active_dampening_bias = bias.dampening_bias;
        last_temporal_self_trace.active_effective_write_scale = bias.effective_write_scale;
        temporal_self_next_trigger_us = now_us + LLAMA_TEMPORAL_SELF_TRIGGER_COOLDOWN_US;
    }

    auto apply_family_update = [&](int32_t family,
                                   int32_t start_microphase,
                                   int32_t settle_microphase,
                                   int32_t selected_tool_kind,
                                   int32_t candidate_count,
                                   const float * metrics,
                                   size_t metric_count,
                                   float signed_outcome,
                                   float magnitude,
                                   const std::string & details) {
        const std::vector<llama_token> update_tokens =
                functional_update_tokens(llama_model_get_vocab(&ctx.get_model()), family, settle_microphase, details);
        if (update_tokens.empty()) {
            return;
        }
        const llama_self_state_event update_event = {
            /*.tokens =*/ update_tokens.data(),
            /*.n_tokens =*/ update_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        llama_self_state_feature_vector update_features = {};
        const llama_self_state_feature_vector * feature_ptr =
                ctx.self_state_build_postwrite_features(update_event, &update_features) ? &update_features : nullptr;
        (void) ctx.functional_lora_apply_update(
                family,
                LLAMA_COG_COMMAND_ORIGIN_DMN,
                start_microphase,
                settle_microphase,
                dmn_before_snapshot,
                dmn_after_snapshot,
                selected_tool_kind,
                candidate_count,
                metrics,
                metric_count,
                clamp_signed_unit(signed_outcome),
                clamp_unit(magnitude),
                update_event,
                feature_ptr);
        (void) emit_cognitive_artifact_tokens(
                ctx,
                update_tokens,
                LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
                LLAMA_SELF_COG_ARTIFACT_FUNCTIONAL_UPDATE,
                LLAMA_COG_COMMAND_ORIGIN_DMN,
                settle_microphase,
                trace.tick_id,
                trace.plan.plan_id);
    };

    {
        const float signed_outcome = clamp_signed_unit(
                0.28f * dmn_deltas.delta_favorable +
                0.18f * dmn_deltas.delta_efficiency +
                0.18f * dmn_deltas.delta_recovery +
                0.18f * dmn_deltas.delta_goal +
                0.18f * dmn_deltas.delta_answerability);
        const float magnitude = clamp_unit(
                0.38f * std::fabs(signed_outcome) +
                0.24f * plan_complexity +
                0.20f * plan_ambiguity +
                0.18f * clamp_unit((float) trace.plan.step_count / (float) LLAMA_COGNITIVE_MAX_PLAN_STEPS));
        const float metrics[] = {
            dmn_deltas.delta_favorable,
            dmn_deltas.delta_efficiency,
            dmn_deltas.delta_recovery,
            dmn_deltas.delta_goal,
            plan_complexity,
            plan_ambiguity,
        };
        apply_family_update(
                LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION,
                LLAMA_FUNCTIONAL_MICROPHASE_PLAN_DRAFT,
                LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE,
                trace.tool_kind,
                trace.plan.step_count,
                metrics,
                sizeof(metrics)/sizeof(metrics[0]),
                signed_outcome,
                magnitude,
                "plan_id " + std::to_string(trace.plan.plan_id) +
                        " steps " + std::to_string(trace.plan.step_count) +
                        " winner " + dmn_action_name(trace.winner_action));
    }

    {
        const float pred_error = clamp_unit(trace.pressure.uncertainty + trace.pressure.contradiction);
        const float signed_outcome = clamp_signed_unit(
                0.24f * dmn_deltas.delta_favorable +
                0.18f * dmn_deltas.delta_recovery +
                0.18f * dmn_deltas.delta_preference +
                0.16f * dmn_deltas.delta_answerability +
                0.12f * dmn_deltas.delta_efficiency +
                0.12f * dmn_deltas.delta_user);
        const float magnitude = clamp_unit(
                0.30f * std::fabs(signed_outcome) +
                0.25f * trace.pressure.uncertainty +
                0.25f * trace.pressure.contradiction +
                0.20f * pred_error);
        const float metrics[] = {
            dmn_deltas.delta_favorable,
            dmn_deltas.delta_recovery,
            dmn_deltas.delta_preference,
            dmn_deltas.delta_answerability,
            dmn_deltas.delta_efficiency,
            pred_error,
        };
        apply_family_update(
                LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION,
                LLAMA_FUNCTIONAL_MICROPHASE_SELF_OBSERVE,
                dmn_runner.functional_microphase > LLAMA_FUNCTIONAL_MICROPHASE_NONE ?
                        dmn_runner.functional_microphase :
                        LLAMA_FUNCTIONAL_MICROPHASE_POST_ACTION_REFLECTION,
                LLAMA_TOOL_KIND_NONE,
                trace.candidate_count,
                metrics,
                sizeof(metrics)/sizeof(metrics[0]),
                signed_outcome,
                magnitude,
                "pressure uncertainty " + std::to_string(trace.pressure.uncertainty) +
                        " contradiction " + std::to_string(trace.pressure.contradiction));
    }

    if (last_counterfactual_trace.candidate_count > 0) {
        const float signed_outcome = clamp_signed_unit(
                0.35f * dmn_deltas.delta_favorable +
                0.25f * dmn_deltas.delta_efficiency +
                0.20f * dmn_deltas.delta_recovery +
                0.20f * dmn_deltas.delta_goal);
        const float magnitude = clamp_unit(
                0.35f * std::fabs(signed_outcome) +
                0.25f * dmn_before_snapshot.favorable_divergence +
                0.20f * trace.pressure.continuation +
                0.20f * clamp_unit((float) last_counterfactual_trace.candidate_count / 4.0f));
        const float metrics[] = {
            dmn_deltas.delta_favorable,
            dmn_deltas.delta_efficiency,
            dmn_deltas.delta_recovery,
            dmn_deltas.delta_goal,
            clamp_unit((float) last_counterfactual_trace.candidate_count / 4.0f),
            trace.pressure.continuation,
        };
        apply_family_update(
                LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL,
                LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_GENERATE,
                compression_eligible ? LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_COMPRESSION :
                        LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE,
                trace.tool_kind,
                last_counterfactual_trace.candidate_count,
                metrics,
                sizeof(metrics)/sizeof(metrics[0]),
                signed_outcome,
                magnitude,
                "winner_index " + std::to_string(last_counterfactual_trace.winner_index) +
                        " candidates " + std::to_string(last_counterfactual_trace.candidate_count));
    }

    const llama_counterfactual_candidate * best_functional_candidate = nullptr;
    for (int32_t i = 0; i < last_counterfactual_trace.candidate_count; ++i) {
        const auto & candidate = last_counterfactual_trace.candidates[i];
        if (candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_LOCAL &&
                candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_HISTORY &&
                candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_ORTHOGONAL &&
                candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_LOCAL &&
                candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_HISTORY &&
                candidate.family != LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_ORTHOGONAL) {
            continue;
        }
        if (candidate.functional_family < 0 || candidate.replay_mode == LLAMA_FUNCTIONAL_REPLAY_MODE_NONE) {
            continue;
        }
        if (!best_functional_candidate ||
                candidate.realized_score > best_functional_candidate->realized_score + 1.0e-6f ||
                (std::fabs(candidate.realized_score - best_functional_candidate->realized_score) <= 1.0e-6f &&
                 candidate.robustness_score > best_functional_candidate->robustness_score + 1.0e-6f)) {
            best_functional_candidate = &candidate;
        }
    }
    if (best_functional_candidate) {
        const float differential_magnitude = clamp_unit(
                0.45f * std::fabs(best_functional_candidate->signed_advantage_vs_current) +
                0.25f * best_functional_candidate->expected_improvement +
                0.20f * best_functional_candidate->robustness_score +
                0.10f * std::max(0.0f, 1.0f - best_functional_candidate->fragility_penalty));
        if (best_functional_candidate->functional_target_kind == LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY &&
                best_functional_candidate->process_entry_slot >= 0) {
            (void) ctx.process_functional_apply_differential_update(
                    best_functional_candidate->process_entry_slot,
                    best_functional_candidate->proposal_family,
                    best_functional_candidate->replay_mode,
                    best_functional_candidate->snapshot_slot,
                    best_functional_candidate->signed_advantage_vs_current,
                    differential_magnitude,
                    best_functional_candidate->robustness_score);
        } else {
            (void) ctx.functional_lora_apply_differential_update(
                    best_functional_candidate->functional_family,
                    best_functional_candidate->proposal_family,
                    best_functional_candidate->replay_mode,
                    best_functional_candidate->snapshot_slot,
                    best_functional_candidate->signed_advantage_vs_current,
                    differential_magnitude,
                    best_functional_candidate->robustness_score);
        }
    }

    if (compression_eligible) {
        const float memory_pressure = clamp_unit((float) working_memory_count / 8.0f);
        const float signed_outcome = clamp_signed_unit(
                0.24f * dmn_deltas.delta_favorable +
                0.20f * dmn_deltas.delta_goal +
                0.18f * dmn_deltas.delta_recovery +
                0.14f * dmn_deltas.delta_user +
                0.12f * dmn_deltas.delta_preference +
                0.12f * dmn_deltas.delta_efficiency);
        const float magnitude = clamp_unit(
                0.35f * std::fabs(signed_outcome) +
                0.35f * memory_pressure +
                0.15f * trace.pressure.reactivation +
                0.15f * trace.pressure.continuation);
        const float metrics[] = {
            dmn_deltas.delta_favorable,
            dmn_deltas.delta_goal,
            dmn_deltas.delta_recovery,
            dmn_deltas.delta_user,
            memory_pressure,
            trace.pressure.reactivation,
        };
        apply_family_update(
                LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION,
                LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_COMPRESSION,
                LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_AUDIT,
                LLAMA_TOOL_KIND_NONE,
                working_memory_count,
                metrics,
                sizeof(metrics)/sizeof(metrics[0]),
                signed_outcome,
                magnitude,
                "working_memory_count " + std::to_string(working_memory_count) +
                        " continuation " + std::to_string(trace.pressure.continuation));
    }

    if (hard_memory_archiving_enabled(ctx)) {
        llama_hard_memory_primitive primitives[LLAMA_HARD_MEMORY_MAX_PRIMITIVES] = {};
        const int32_t primitive_count = build_dmn_hard_memory_primitives(
                trace,
                dmn_deltas,
                last_counterfactual_trace,
                last_governance_trace,
                last_remediation_plan,
                last_temporal_self_trace,
                primitives);
        if (primitive_count > 0) {
            (void) ctx.hard_memory_archive_primitives(primitives, primitive_count);
        }
    }

    if (dmn_runner.pending_command_id > 0) {
        dmn_runner.active = true;
        dmn_runner.completed = false;
    } else if (!dmn_runner.waiting_on_tool) {
        dmn_runner.active = false;
        dmn_runner.completed = true;
        dmn_runner.planning_active = false;
        dmn_runner.plan_status = dmn_plan.status == LLAMA_COG_PLAN_STATUS_BLOCKED ?
                LLAMA_COG_PLAN_STATUS_BLOCKED :
                LLAMA_COG_PLAN_STATUS_COMPLETED;
        (void) ctx.process_functional_set_execution(llama_process_functional_signature {});
        (void) ctx.functional_lora_activate(make_inactive_functional_decision(LLAMA_COG_COMMAND_ORIGIN_DMN));
        dmn_runner.functional_microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
    }

    host_state.shared_state_version += 1;
    host_state.dmn_tick_count += 1;
    host_state.last_dmn_time_us = now_us;

    trace.plan = dmn_plan;
    (void) ctx.shared_cognitive_context_get_window(&trace.context_window);
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

bool llama_cognitive_loop::temporal_self_improvement_get_last(
        llama_temporal_self_improvement_trace * out_trace) const {
    if (!out_trace) {
        return false;
    }
    *out_trace = last_temporal_self_trace;
    return true;
}

bool llama_cognitive_loop::cognitive_bash_tool_submit_result(
        const llama_bash_tool_result & result,
        llama_active_loop_trace * out_active_trace) {
    llama_bash_tool_request request = {};
    if (!ctx.cognitive_bash_tool_get_request(result.command_id, &request)) {
        return false;
    }

    const int32_t tool_status =
            (!result.launch_failed && !result.timed_out && result.exit_code == 0) ?
                    LLAMA_SELF_TOOL_JOB_COMPLETED :
                    LLAMA_SELF_TOOL_JOB_FAILED;
    (void) ctx.self_state_upsert_tool_job(result.tool_job_id, tool_status, 1.0f);

    if (!ctx.bash_tool_submit_result(result)) {
        return false;
    }
    if (!ctx.cognitive_command_complete(result.command_id, false)) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
    const std::string summary = summarize_bash_result(result);
    const std::vector<llama_token> tokens = tokenize_text(vocab, summary);
    const llama_self_state_event tool_event = {
        /*.tokens =*/ tokens.empty() ? nullptr : tokens.data(),
        /*.n_tokens =*/ tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ tool_status == LLAMA_SELF_TOOL_JOB_COMPLETED ?
                LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED :
                LLAMA_SELF_STATE_EVENT_TOOL_FAILED,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
    };

    if (request.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
        return active_loop_process(tool_event, out_active_trace);
    }

    const bool applied = apply_tool_event_only(ctx, tool_event);
    if (applied) {
        dmn_runner.waiting_on_tool = false;
        dmn_runner.active = false;
        dmn_runner.completed = true;
        dmn_runner.planning_active = false;
        dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
    }
    return applied;
}

bool llama_cognitive_loop::cognitive_codex_tool_submit_result(
        const llama_codex_tool_result & result,
        llama_active_loop_trace * out_active_trace) {
    llama_codex_tool_request request = {};
    if (!cognitive_codex_tool_get_request(result.command_id, &request)) {
        return false;
    }

    const int32_t tool_status =
            (!result.launch_failed && result.exit_code == 0) ?
                    LLAMA_SELF_TOOL_JOB_COMPLETED :
                    LLAMA_SELF_TOOL_JOB_FAILED;
    (void) ctx.self_state_upsert_tool_job(result.tool_job_id, tool_status, 1.0f);

    last_codex_result = result;
    last_codex_result.stdout_text[LLAMA_CODEX_TOOL_STDOUT_MAX_CHARS - 1] = '\0';
    last_codex_result.stderr_text[LLAMA_CODEX_TOOL_STDERR_MAX_CHARS - 1] = '\0';
    last_codex_result.error_text[LLAMA_CODEX_TOOL_ERROR_MAX_CHARS - 1] = '\0';
    last_codex_result.summary_text[LLAMA_CODEX_TOOL_SUMMARY_MAX_CHARS - 1] = '\0';
    last_codex_result.manual_requirements[LLAMA_CODEX_TOOL_MANUAL_MAX_CHARS - 1] = '\0';
    last_codex_result.changed_files_excerpt[LLAMA_CODEX_TOOL_FILES_MAX_CHARS - 1] = '\0';
    has_last_codex_result = true;

    const int32_t index = find_codex_request_index(codex_requests, codex_request_count, result.command_id);
    if (index >= 0) {
        for (int32_t i = index + 1; i < codex_request_count; ++i) {
            codex_requests[i - 1] = codex_requests[i];
        }
        codex_requests[codex_request_count - 1] = {};
        --codex_request_count;
    }
    if (!ctx.cognitive_command_complete(result.command_id, false)) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
    const std::string summary = summarize_codex_result(result);
    const std::vector<llama_token> tokens = tokenize_text(vocab, summary);
    const llama_self_state_event tool_event = {
        /*.tokens =*/ tokens.empty() ? nullptr : tokens.data(),
        /*.n_tokens =*/ tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ tool_status == LLAMA_SELF_TOOL_JOB_COMPLETED ?
                LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED :
                LLAMA_SELF_STATE_EVENT_TOOL_FAILED,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
    };

    if (request.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
        return active_loop_process(tool_event, out_active_trace);
    }

    const bool applied = apply_tool_event_only(ctx, tool_event);
    if (applied) {
        dmn_runner.waiting_on_tool = false;
        dmn_runner.active = false;
        dmn_runner.completed = true;
        dmn_runner.planning_active = false;
        dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
    }
    return applied;
}

bool llama_cognitive_loop::cognitive_hard_memory_submit_result(
        const llama_cognitive_hard_memory_result & result,
        llama_active_loop_trace * out_active_trace) {
    llama_cognitive_hard_memory_request request = {};
    if (!ctx.cognitive_hard_memory_get_request(result.command_id, &request)) {
        return false;
    }

    const bool is_write = request.operation == LLAMA_COG_HARD_MEMORY_OPERATION_WRITE;
    const int32_t tool_status =
            is_write ?
                    ((result.archive_trace.archived && result.archive_trace.status_code < 500) ?
                            LLAMA_SELF_TOOL_JOB_COMPLETED :
                            LLAMA_SELF_TOOL_JOB_FAILED) :
                    ((result.result.ok && result.result.status_code < 500) ?
                            LLAMA_SELF_TOOL_JOB_COMPLETED :
                            LLAMA_SELF_TOOL_JOB_FAILED);
    (void) ctx.self_state_upsert_tool_job(result.tool_job_id, tool_status, 1.0f);

    if (is_write) {
        if (!ctx.hard_memory_submit_archive_trace(result.archive_trace)) {
            return false;
        }
    } else {
        if (!ctx.hard_memory_submit_result(result.result)) {
            return false;
        }
        (void) ctx.self_state_promote_hard_memory_query(request.query, result.result);
    }
    if (!ctx.hard_memory_clear_request(result.command_id)) {
        return false;
    }
    if (!ctx.cognitive_command_complete(result.command_id, false)) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
    const std::string summary =
            is_write ?
                    summarize_hard_memory_archive_result(result.archive_trace) :
                    summarize_hard_memory_result(result.result);
    const std::vector<llama_token> tokens = tokenize_text(vocab, summary);
    const llama_self_state_event tool_event = {
        /*.tokens =*/ tokens.empty() ? nullptr : tokens.data(),
        /*.n_tokens =*/ tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ tool_status == LLAMA_SELF_TOOL_JOB_COMPLETED ?
                LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED :
                LLAMA_SELF_STATE_EVENT_TOOL_FAILED,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
    };

    if (request.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
        return active_loop_process(tool_event, out_active_trace);
    }

    const bool applied = apply_tool_event_only(ctx, tool_event);
    if (applied) {
        dmn_runner.waiting_on_tool = false;
        dmn_runner.active = false;
        dmn_runner.completed = true;
        dmn_runner.planning_active = false;
        dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
    }
    return applied;
}

bool llama_cognitive_loop::cognitive_telegram_relay_submit_result(
        const llama_telegram_relay_result & result,
        llama_active_loop_trace * out_active_trace) {
    llama_telegram_relay_request request = {};
    if (!cognitive_telegram_relay_get_request(result.command_id, &request)) {
        return false;
    }

    const int32_t tool_status = result.delivered ?
            LLAMA_SELF_TOOL_JOB_COMPLETED :
            LLAMA_SELF_TOOL_JOB_FAILED;
    (void) ctx.self_state_upsert_tool_job(result.tool_job_id, tool_status, 1.0f);

    const int32_t index = find_telegram_request_index(telegram_requests, telegram_request_count, result.command_id);
    if (index >= 0) {
        for (int32_t i = index + 1; i < telegram_request_count; ++i) {
            telegram_requests[i - 1] = telegram_requests[i];
        }
        telegram_requests[telegram_request_count - 1] = {};
        --telegram_request_count;
    }
    last_telegram_result = result;
    has_last_telegram_result = true;

    if (!ctx.cognitive_command_complete(result.command_id, false)) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(&ctx.get_model());
    const std::string summary = summarize_telegram_result(result);
    const std::vector<llama_token> tokens = tokenize_text(vocab, summary);
    const llama_self_state_event tool_event = {
        /*.tokens =*/ tokens.empty() ? nullptr : tokens.data(),
        /*.n_tokens =*/ tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ tool_status == LLAMA_SELF_TOOL_JOB_COMPLETED ?
                LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED :
                LLAMA_SELF_STATE_EVENT_TOOL_FAILED,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
    };

    if (request.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
        return active_loop_process(tool_event, out_active_trace);
    }

    const bool applied = apply_tool_event_only(ctx, tool_event);
    if (applied) {
        dmn_runner.waiting_on_tool = false;
        dmn_runner.active = false;
        dmn_runner.completed = true;
        dmn_runner.planning_active = false;
        dmn_runner.plan_status = LLAMA_COG_PLAN_STATUS_COMPLETED;
    }
    return applied;
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

bool llama_context::cognitive_command_begin_external_wait(int32_t command_id) {
    return cognitive_loop && cognitive_loop->cognitive_command_begin_external_wait(command_id);
}

bool llama_context::cognitive_command_rebind_tool(int32_t command_id, int32_t tool_spec_index) {
    return cognitive_loop && cognitive_loop->cognitive_command_rebind_tool(command_id, tool_spec_index);
}

bool llama_context::cognitive_authoritative_react_set_enabled(bool enabled) {
    return cognitive_loop && cognitive_loop->cognitive_authoritative_react_set_enabled(enabled);
}

bool llama_context::cognitive_active_authoritative_begin_tool(
        int32_t episode_id,
        uint32_t reason_mask,
        float priority,
        int32_t * out_command_id,
        int32_t * out_tool_job_id) {
    return cognitive_loop && cognitive_loop->cognitive_active_authoritative_begin_tool(
            episode_id,
            reason_mask,
            priority,
            out_command_id,
            out_tool_job_id);
}

bool llama_context::cognitive_active_authoritative_finish(int32_t episode_id, int32_t terminal_reason) {
    return cognitive_loop && cognitive_loop->cognitive_active_authoritative_finish(episode_id, terminal_reason);
}

bool llama_context::cognitive_dmn_authoritative_begin_tool(
        int32_t tick_id,
        uint32_t reason_mask,
        float priority,
        int32_t * out_command_id,
        int32_t * out_tool_job_id) {
    return cognitive_loop && cognitive_loop->cognitive_dmn_authoritative_begin_tool(
            tick_id,
            reason_mask,
            priority,
            out_command_id,
            out_tool_job_id);
}

bool llama_context::cognitive_dmn_authoritative_finish(int32_t tick_id, int32_t terminal_reason) {
    return cognitive_loop && cognitive_loop->cognitive_dmn_authoritative_finish(tick_id, terminal_reason);
}

bool llama_context::cognitive_active_tool_emission_note(int32_t command_id, const llama_token * tokens, size_t n_tokens) {
    return cognitive_loop && cognitive_loop->cognitive_active_tool_emission_note(command_id, tokens, n_tokens);
}

bool llama_context::cognitive_active_planner_reasoning_note(int32_t episode_id, const llama_token * tokens, size_t n_tokens) {
    return cognitive_loop && cognitive_loop->cognitive_active_planner_reasoning_note(episode_id, tokens, n_tokens);
}

bool llama_context::codex_tool_configure(const llama_codex_tool_config & config) {
    return cognitive_loop && cognitive_loop->codex_tool_configure(config);
}

bool llama_context::codex_tool_get_config(llama_codex_tool_config * out_config) const {
    return cognitive_loop && cognitive_loop->codex_tool_get_config(out_config);
}

bool llama_context::codex_tool_get_last_result(llama_codex_tool_result * out_result) const {
    return cognitive_loop && cognitive_loop->codex_tool_get_last_result(out_result);
}

bool llama_context::cognitive_codex_tool_get_request(int32_t command_id, llama_codex_tool_request * out_request) const {
    return cognitive_loop && cognitive_loop->cognitive_codex_tool_get_request(command_id, out_request);
}

bool llama_context::cognitive_codex_tool_set_request(const llama_codex_tool_request & request) {
    return cognitive_loop && cognitive_loop->cognitive_codex_tool_set_request(request);
}

bool llama_context::cognitive_bash_tool_submit_result(const llama_bash_tool_result & result, llama_active_loop_trace * out_active_trace) {
    return cognitive_loop && cognitive_loop->cognitive_bash_tool_submit_result(result, out_active_trace);
}

bool llama_context::cognitive_codex_tool_submit_result(const llama_codex_tool_result & result, llama_active_loop_trace * out_active_trace) {
    return cognitive_loop && cognitive_loop->cognitive_codex_tool_submit_result(result, out_active_trace);
}

bool llama_context::cognitive_hard_memory_get_request(int32_t command_id, llama_cognitive_hard_memory_request * out_request) const {
    return hard_memory_get_request(command_id, out_request);
}

bool llama_context::cognitive_hard_memory_submit_result(const llama_cognitive_hard_memory_result & result, llama_active_loop_trace * out_active_trace) {
    return cognitive_loop && cognitive_loop->cognitive_hard_memory_submit_result(result, out_active_trace);
}

bool llama_context::cognitive_telegram_relay_get_request(int32_t command_id, llama_telegram_relay_request * out_request) const {
    return cognitive_loop && cognitive_loop->cognitive_telegram_relay_get_request(command_id, out_request);
}

bool llama_context::cognitive_telegram_relay_set_request(const llama_telegram_relay_request & request) {
    return cognitive_loop && cognitive_loop->cognitive_telegram_relay_set_request(request);
}

bool llama_context::cognitive_telegram_relay_submit_result(const llama_telegram_relay_result & result, llama_active_loop_trace * out_active_trace) {
    return cognitive_loop && cognitive_loop->cognitive_telegram_relay_submit_result(result, out_active_trace);
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

bool llama_context::temporal_self_improvement_get_last(llama_temporal_self_improvement_trace * out_trace) const {
    return cognitive_loop && cognitive_loop->temporal_self_improvement_get_last(out_trace);
}

int32_t llama_codex_tool_configure(
        struct llama_context * ctx,
        const struct llama_codex_tool_config * config) {
    return ctx && config && ctx->codex_tool_configure(*config) ? 0 : -1;
}

int32_t llama_codex_tool_get_config(
        const struct llama_context * ctx,
        struct llama_codex_tool_config * out_config) {
    return ctx && out_config && ctx->codex_tool_get_config(out_config) ? 0 : -1;
}

int32_t llama_codex_tool_get_last_result(
        const struct llama_context * ctx,
        struct llama_codex_tool_result * out_result) {
    return ctx && out_result && ctx->codex_tool_get_last_result(out_result) ? 0 : -1;
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

int32_t llama_cognitive_command_begin_external_wait(
        struct llama_context * ctx,
        int32_t command_id) {
    return ctx && ctx->cognitive_command_begin_external_wait(command_id) ? 0 : -1;
}

int32_t llama_cognitive_command_rebind_tool(
        struct llama_context * ctx,
        int32_t command_id,
        int32_t tool_spec_index) {
    return ctx && ctx->cognitive_command_rebind_tool(command_id, tool_spec_index) ? 0 : -1;
}

int32_t llama_cognitive_authoritative_react_set_enabled(
        struct llama_context * ctx,
        bool enabled) {
    return ctx && ctx->cognitive_authoritative_react_set_enabled(enabled) ? 0 : -1;
}

int32_t llama_cognitive_active_authoritative_begin_tool(
        struct llama_context * ctx,
        int32_t episode_id,
        uint32_t reason_mask,
        float priority,
        int32_t * out_command_id,
        int32_t * out_tool_job_id) {
    return ctx && ctx->cognitive_active_authoritative_begin_tool(
            episode_id,
            reason_mask,
            priority,
            out_command_id,
            out_tool_job_id) ? 0 : -1;
}

int32_t llama_cognitive_active_authoritative_finish(
        struct llama_context * ctx,
        int32_t episode_id,
        int32_t terminal_reason) {
    return ctx && ctx->cognitive_active_authoritative_finish(episode_id, terminal_reason) ? 0 : -1;
}

int32_t llama_cognitive_dmn_authoritative_begin_tool(
        struct llama_context * ctx,
        int32_t tick_id,
        uint32_t reason_mask,
        float priority,
        int32_t * out_command_id,
        int32_t * out_tool_job_id) {
    return ctx && ctx->cognitive_dmn_authoritative_begin_tool(
            tick_id,
            reason_mask,
            priority,
            out_command_id,
            out_tool_job_id) ? 0 : -1;
}

int32_t llama_cognitive_dmn_authoritative_finish(
        struct llama_context * ctx,
        int32_t tick_id,
        int32_t terminal_reason) {
    return ctx && ctx->cognitive_dmn_authoritative_finish(tick_id, terminal_reason) ? 0 : -1;
}

int32_t llama_cognitive_active_tool_emission_note(
        struct llama_context * ctx,
        int32_t command_id,
        const llama_token * tokens,
        size_t n_tokens) {
    return ctx && tokens && n_tokens > 0 && ctx->cognitive_active_tool_emission_note(command_id, tokens, n_tokens) ? 0 : -1;
}

int32_t llama_cognitive_active_planner_reasoning_note(
        struct llama_context * ctx,
        int32_t episode_id,
        const llama_token * tokens,
        size_t n_tokens) {
    return ctx && tokens && n_tokens > 0 && ctx->cognitive_active_planner_reasoning_note(episode_id, tokens, n_tokens) ? 0 : -1;
}

int32_t llama_cognitive_bash_tool_submit_result(
        struct llama_context * ctx,
        const struct llama_bash_tool_result * result,
        struct llama_active_loop_trace * out_active_trace) {
    return ctx && result && ctx->cognitive_bash_tool_submit_result(*result, out_active_trace) ? 0 : -1;
}

int32_t llama_cognitive_codex_tool_get_request(
        const struct llama_context * ctx,
        int32_t command_id,
        struct llama_codex_tool_request * out_request) {
    return ctx && out_request && ctx->cognitive_codex_tool_get_request(command_id, out_request) ? 0 : -1;
}

int32_t llama_cognitive_codex_tool_set_request(
        struct llama_context * ctx,
        const struct llama_codex_tool_request * request) {
    return ctx && request && ctx->cognitive_codex_tool_set_request(*request) ? 0 : -1;
}

int32_t llama_cognitive_codex_tool_submit_result(
        struct llama_context * ctx,
        const struct llama_codex_tool_result * result,
        struct llama_active_loop_trace * out_active_trace) {
    return ctx && result && ctx->cognitive_codex_tool_submit_result(*result, out_active_trace) ? 0 : -1;
}

int32_t llama_cognitive_hard_memory_get_request(
        const struct llama_context * ctx,
        int32_t command_id,
        struct llama_cognitive_hard_memory_request * out_request) {
    return ctx && out_request && ctx->cognitive_hard_memory_get_request(command_id, out_request) ? 0 : -1;
}

int32_t llama_cognitive_hard_memory_set_request(
        struct llama_context * ctx,
        const struct llama_cognitive_hard_memory_request * request) {
    return ctx && request && ctx->hard_memory_set_request(*request) ? 0 : -1;
}

int32_t llama_cognitive_hard_memory_submit_result(
        struct llama_context * ctx,
        const struct llama_cognitive_hard_memory_result * result,
        struct llama_active_loop_trace * out_active_trace) {
    return ctx && result && ctx->cognitive_hard_memory_submit_result(*result, out_active_trace) ? 0 : -1;
}

int32_t llama_cognitive_telegram_relay_get_request(
        const struct llama_context * ctx,
        int32_t command_id,
        struct llama_telegram_relay_request * out_request) {
    return ctx && out_request && ctx->cognitive_telegram_relay_get_request(command_id, out_request) ? 0 : -1;
}

int32_t llama_cognitive_telegram_relay_set_request(
        struct llama_context * ctx,
        const struct llama_telegram_relay_request * request) {
    return ctx && request && ctx->cognitive_telegram_relay_set_request(*request) ? 0 : -1;
}

int32_t llama_cognitive_telegram_relay_submit_result(
        struct llama_context * ctx,
        const struct llama_telegram_relay_result * result,
        struct llama_active_loop_trace * out_active_trace) {
    return ctx && result && ctx->cognitive_telegram_relay_submit_result(*result, out_active_trace) ? 0 : -1;
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

int32_t llama_temporal_self_improvement_get_last(
        const struct llama_context * ctx,
        struct llama_temporal_self_improvement_trace * out_trace) {
    return ctx && ctx->temporal_self_improvement_get_last(out_trace) ? 0 : -1;
}

const char * llama_vicuna_core_system_prompt_default(void) {
    return vicuna_core_system_prompt_default_text();
}
