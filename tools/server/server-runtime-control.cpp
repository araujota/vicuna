#include "server-runtime-control.h"

namespace {

static json action_mask_to_json(const llama_runtime_control_action_mask & mask) {
    return {
        {"allow_sampling", mask.allow_sampling},
        {"allow_repetition", mask.allow_repetition},
        {"allow_structure", mask.allow_structure},
        {"allow_branch", mask.allow_branch},
        {"allow_steering", mask.allow_steering},
        {"max_branch_sample_count", mask.max_branch_sample_count},
    };
}

static json limits_to_json(const llama_runtime_control_limits & limits) {
    return {
        {"min_temperature", limits.min_temperature},
        {"max_temperature", limits.max_temperature},
        {"min_top_p", limits.min_top_p},
        {"max_top_p", limits.max_top_p},
        {"min_min_p", limits.min_min_p},
        {"max_min_p", limits.max_min_p},
        {"min_typical_p", limits.min_typical_p},
        {"max_typical_p", limits.max_typical_p},
        {"max_top_n_sigma", limits.max_top_n_sigma},
        {"min_repeat_penalty", limits.min_repeat_penalty},
        {"max_repeat_penalty", limits.max_repeat_penalty},
        {"max_frequency_penalty", limits.max_frequency_penalty},
        {"max_presence_penalty", limits.max_presence_penalty},
        {"max_dry_multiplier", limits.max_dry_multiplier},
        {"max_penalty_last_n", limits.max_penalty_last_n},
        {"max_dry_allowed_length", limits.max_dry_allowed_length},
        {"max_dry_penalty_last_n", limits.max_dry_penalty_last_n},
    };
}

static json action_to_json_local(const llama_runtime_control_action & action) {
    return {
        {"valid", action.valid},
        {"sampling", {
            {"enabled", action.sampling.enabled},
            {"has_temperature", action.sampling.has_temperature},
            {"has_top_k", action.sampling.has_top_k},
            {"has_top_p", action.sampling.has_top_p},
            {"has_min_p", action.sampling.has_min_p},
            {"has_typical_p", action.sampling.has_typical_p},
            {"has_top_n_sigma", action.sampling.has_top_n_sigma},
            {"temperature", action.sampling.temperature},
            {"top_k", action.sampling.top_k},
            {"top_p", action.sampling.top_p},
            {"min_p", action.sampling.min_p},
            {"typical_p", action.sampling.typical_p},
            {"top_n_sigma", action.sampling.top_n_sigma},
            {"min_keep", action.sampling.min_keep},
        }},
        {"repetition", {
            {"enabled", action.repetition.enabled},
            {"has_repeat_penalty", action.repetition.has_repeat_penalty},
            {"has_frequency_penalty", action.repetition.has_frequency_penalty},
            {"has_presence_penalty", action.repetition.has_presence_penalty},
            {"has_penalty_last_n", action.repetition.has_penalty_last_n},
            {"repeat_penalty", action.repetition.repeat_penalty},
            {"frequency_penalty", action.repetition.frequency_penalty},
            {"presence_penalty", action.repetition.presence_penalty},
            {"penalty_last_n", action.repetition.penalty_last_n},
        }},
        {"structure", {
            {"enabled", action.structure.enabled},
            {"clear_grammar", action.structure.clear_grammar},
            {"clear_logit_bias", action.structure.clear_logit_bias},
        }},
        {"branch", {
            {"enabled", action.branch.enabled},
            {"checkpoint_now", action.branch.checkpoint_now},
            {"request_verify", action.branch.request_verify},
            {"checkpoint_slot", action.branch.checkpoint_slot},
            {"restore_slot", action.branch.restore_slot},
            {"branch_sample_count", action.branch.branch_sample_count},
        }},
        {"steering", {
            {"enabled", action.steering.enabled},
            {"clear_cvec", action.steering.clear_cvec},
        }},
    };
}

static void action_from_json_local(const json & payload, llama_runtime_control_action * out_action) {
    if (!out_action || !payload.is_object()) {
        return;
    }
    out_action->valid = json_value(payload, "valid", false);
    const json sampling = payload.value("sampling", json::object());
    out_action->sampling.enabled = json_value(sampling, "enabled", false);
    out_action->sampling.has_temperature = json_value(sampling, "has_temperature", false);
    out_action->sampling.has_top_k = json_value(sampling, "has_top_k", false);
    out_action->sampling.has_top_p = json_value(sampling, "has_top_p", false);
    out_action->sampling.has_min_p = json_value(sampling, "has_min_p", false);
    out_action->sampling.has_typical_p = json_value(sampling, "has_typical_p", false);
    out_action->sampling.has_top_n_sigma = json_value(sampling, "has_top_n_sigma", false);
    out_action->sampling.temperature = json_value(sampling, "temperature", 0.0f);
    out_action->sampling.top_k = json_value(sampling, "top_k", int32_t(0));
    out_action->sampling.top_p = json_value(sampling, "top_p", 0.0f);
    out_action->sampling.min_p = json_value(sampling, "min_p", 0.0f);
    out_action->sampling.typical_p = json_value(sampling, "typical_p", 0.0f);
    out_action->sampling.top_n_sigma = json_value(sampling, "top_n_sigma", 0.0f);
    out_action->sampling.min_keep = json_value(sampling, "min_keep", int32_t(0));

    const json repetition = payload.value("repetition", json::object());
    out_action->repetition.enabled = json_value(repetition, "enabled", false);
    out_action->repetition.has_repeat_penalty = json_value(repetition, "has_repeat_penalty", false);
    out_action->repetition.has_frequency_penalty = json_value(repetition, "has_frequency_penalty", false);
    out_action->repetition.has_presence_penalty = json_value(repetition, "has_presence_penalty", false);
    out_action->repetition.has_penalty_last_n = json_value(repetition, "has_penalty_last_n", false);
    out_action->repetition.repeat_penalty = json_value(repetition, "repeat_penalty", 0.0f);
    out_action->repetition.frequency_penalty = json_value(repetition, "frequency_penalty", 0.0f);
    out_action->repetition.presence_penalty = json_value(repetition, "presence_penalty", 0.0f);
    out_action->repetition.penalty_last_n = json_value(repetition, "penalty_last_n", int32_t(0));

    const json structure = payload.value("structure", json::object());
    out_action->structure.enabled = json_value(structure, "enabled", false);
    out_action->structure.clear_grammar = json_value(structure, "clear_grammar", false);
    out_action->structure.clear_logit_bias = json_value(structure, "clear_logit_bias", false);

    const json branch = payload.value("branch", json::object());
    out_action->branch.enabled = json_value(branch, "enabled", false);
    out_action->branch.checkpoint_now = json_value(branch, "checkpoint_now", false);
    out_action->branch.request_verify = json_value(branch, "request_verify", false);
    out_action->branch.checkpoint_slot = json_value(branch, "checkpoint_slot", int32_t(0));
    out_action->branch.restore_slot = json_value(branch, "restore_slot", int32_t(0));
    out_action->branch.branch_sample_count = json_value(branch, "branch_sample_count", int32_t(0));

    const json steering = payload.value("steering", json::object());
    out_action->steering.enabled = json_value(steering, "enabled", false);
    out_action->steering.clear_cvec = json_value(steering, "clear_cvec", false);
}

static json candidate_metadata_to_json(const server_decode_control_candidate_metadata & metadata) {
    return {
        {"available", metadata.available},
        {"controller_version", metadata.controller_version},
        {"controller_alias", metadata.controller_alias},
        {"confidence_present", metadata.confidence_present},
        {"confidence", metadata.confidence},
        {"executed_live", metadata.executed_live},
        {"proposal_source", metadata.proposal_source},
    };
}

static void candidate_metadata_from_json_local(const json & payload, server_decode_control_candidate_metadata * out_metadata) {
    if (!out_metadata || !payload.is_object()) {
        return;
    }
    out_metadata->available = json_value(payload, "available", false);
    out_metadata->controller_version = json_value(payload, "controller_version", std::string());
    out_metadata->controller_alias = json_value(payload, "controller_alias", std::string());
    out_metadata->confidence_present = json_value(payload, "confidence_present", false);
    out_metadata->confidence = json_value(payload, "confidence", 0.0f);
    out_metadata->executed_live = json_value(payload, "executed_live", false);
    out_metadata->proposal_source = json_value(payload, "proposal_source", std::string("decode_controller"));
}

} // namespace

server_runtime_signal_summary server_runtime_signal_summary_from_telemetry_event(
        const llama_runtime_telemetry_event & event) {
    (void) event;
    return {};
}

llama_runtime_control_signal_bundles server_llama_runtime_control_bundles_from_state(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        const server_runtime_signal_summary & runtime_signals) {
    (void) moment;
    (void) vad;
    (void) runtime_signals;
    return {};
}

llama_runtime_control_action server_llama_runtime_control_action_from_state(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        const server_runtime_signal_summary & runtime_signals,
        const llama_runtime_control_action_mask * mask) {
    (void) moment;
    (void) vad;
    (void) runtime_signals;
    (void) mask;
    return {};
}

server_llama_runtime_control_plan server_llama_runtime_control_plan_from_state(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        const server_runtime_signal_summary & runtime_signals,
        const llama_runtime_control_action_mask * mask) {
    server_llama_runtime_control_plan plan = {};
    plan.bundles = server_llama_runtime_control_bundles_from_state(moment, vad, runtime_signals);
    plan.action = server_llama_runtime_control_action_from_state(moment, vad, runtime_signals, mask);
    return plan;
}

bool server_llama_runtime_control_generate_cvec_profile(
        const server_llama_runtime_control_state & state,
        server_llama_runtime_generated_cvec_profile * out_profile,
        std::string * out_error) {
    (void) state;
    if (out_profile) {
        *out_profile = {};
    }
    if (out_error) {
        *out_error = "host runtime does not generate local cvec profiles";
    }
    return false;
}

void server_llama_runtime_control_begin_request(
        server_llama_runtime_control_state * state,
        const std::string & request_id,
        const std::string & model_name) {
    if (!state) {
        return;
    }
    state->request_id = request_id;
    state->resolved_model = model_name;
    state->active_decode_trace = {};
    state->active_decode_trace.request_id = request_id;
    state->active_decode_trace.model = model_name;
}

server_decode_control_trace server_llama_runtime_control_finalize_trace(
        server_llama_runtime_control_state * state,
        const std::string & emotive_trace_id) {
    if (!state) {
        return {};
    }
    state->active_decode_trace.emotive_trace_id = emotive_trace_id;
    return state->active_decode_trace;
}

json server_llama_runtime_control_action_to_json(const llama_runtime_control_action & action) {
    return action_to_json_local(action);
}

json server_llama_runtime_control_signal_bundles_to_json(const llama_runtime_control_signal_bundles & bundles) {
    return {
        {"uncertainty_regulation", bundles.uncertainty_regulation},
        {"anti_repetition_recovery", bundles.anti_repetition_recovery},
        {"structural_validity", bundles.structural_validity},
        {"verification_pressure", bundles.verification_pressure},
        {"commit_efficiency", bundles.commit_efficiency},
        {"steering_pressure", bundles.steering_pressure},
    };
}

json server_decode_policy_config_to_json(const server_decode_policy_config & config) {
    return {
        {"schema_version", config.schema_version},
        {"base_temperature", config.base_temperature},
        {"base_top_k", config.base_top_k},
        {"base_top_p", config.base_top_p},
        {"base_min_p", config.base_min_p},
        {"action_mask", action_mask_to_json(config.action_mask)},
        {"control_limits", limits_to_json(config.control_limits)},
    };
}

bool server_decode_policy_config_from_json(
        const json & payload,
        server_decode_policy_config * out_config,
        std::string * out_error) {
    if (!out_config || !payload.is_object()) {
        if (out_error) {
            *out_error = "decode policy config must be an object";
        }
        return false;
    }
    out_config->schema_version = json_value(payload, "schema_version", std::string("decode_policy_config_v1"));
    out_config->base_temperature = json_value(payload, "base_temperature", 0.0f);
    out_config->base_top_k = json_value(payload, "base_top_k", int32_t(0));
    out_config->base_top_p = json_value(payload, "base_top_p", 0.0f);
    out_config->base_min_p = json_value(payload, "base_min_p", 0.0f);
    const json action_mask = payload.value("action_mask", json::object());
    out_config->action_mask.allow_sampling = json_value(action_mask, "allow_sampling", false);
    out_config->action_mask.allow_repetition = json_value(action_mask, "allow_repetition", false);
    out_config->action_mask.allow_structure = json_value(action_mask, "allow_structure", false);
    out_config->action_mask.allow_branch = json_value(action_mask, "allow_branch", false);
    out_config->action_mask.allow_steering = json_value(action_mask, "allow_steering", false);
    out_config->action_mask.max_branch_sample_count = json_value(action_mask, "max_branch_sample_count", int32_t(1));
    const json limits = payload.value("control_limits", json::object());
    out_config->control_limits.min_temperature = json_value(limits, "min_temperature", 0.0f);
    out_config->control_limits.max_temperature = json_value(limits, "max_temperature", 2.0f);
    out_config->control_limits.min_top_p = json_value(limits, "min_top_p", 0.0f);
    out_config->control_limits.max_top_p = json_value(limits, "max_top_p", 1.0f);
    out_config->control_limits.min_min_p = json_value(limits, "min_min_p", 0.0f);
    out_config->control_limits.max_min_p = json_value(limits, "max_min_p", 1.0f);
    out_config->control_limits.min_typical_p = json_value(limits, "min_typical_p", 0.0f);
    out_config->control_limits.max_typical_p = json_value(limits, "max_typical_p", 1.0f);
    out_config->control_limits.max_top_n_sigma = json_value(limits, "max_top_n_sigma", 8.0f);
    out_config->control_limits.min_repeat_penalty = json_value(limits, "min_repeat_penalty", 0.0f);
    out_config->control_limits.max_repeat_penalty = json_value(limits, "max_repeat_penalty", 2.0f);
    out_config->control_limits.max_frequency_penalty = json_value(limits, "max_frequency_penalty", 2.0f);
    out_config->control_limits.max_presence_penalty = json_value(limits, "max_presence_penalty", 2.0f);
    out_config->control_limits.max_dry_multiplier = json_value(limits, "max_dry_multiplier", 2.0f);
    out_config->control_limits.max_penalty_last_n = json_value(limits, "max_penalty_last_n", int32_t(256));
    out_config->control_limits.max_dry_allowed_length = json_value(limits, "max_dry_allowed_length", int32_t(256));
    out_config->control_limits.max_dry_penalty_last_n = json_value(limits, "max_dry_penalty_last_n", int32_t(256));
    return true;
}

json server_decode_control_trace_to_json(const server_decode_control_trace & trace) {
    json steps = json::array();
    for (const auto & step : trace.steps) {
        steps.push_back({
            {"timestamp_ms", step.timestamp_ms},
            {"seq_id", step.seq_id},
            {"step_index", step.step_index},
            {"output_index", step.output_index},
            {"moment", server_emotive_vector_to_json(step.moment)},
            {"vad", server_emotive_vad_to_json(step.vad)},
            {"runtime_signals", server_runtime_signal_summary_to_json(step.runtime_signals)},
            {"bundles", server_llama_runtime_control_signal_bundles_to_json(step.bundles)},
            {"decode_policy", server_decode_policy_config_to_json(step.decode_policy)},
            {"previous_executed_action_available", step.previous_executed_action_available},
            {"previous_executed_action", action_to_json_local(step.previous_executed_action)},
            {"teacher_action", action_to_json_local(step.teacher_action)},
            {"has_candidate_action", step.has_candidate_action},
            {"candidate_action", action_to_json_local(step.candidate_action)},
            {"candidate_metadata", candidate_metadata_to_json(step.candidate_metadata)},
            {"executed_action", action_to_json_local(step.executed_action)},
            {"generated_cvec_applied", step.generated_cvec_applied},
            {"generated_cvec_profile_id", step.generated_cvec_profile_id},
            {"generated_cvec_norm", step.generated_cvec_norm},
            {"generated_cvec_clipped", step.generated_cvec_clipped},
            {"next_outcome", {
                {"available", step.next_outcome.available},
                {"d_mean_entropy", step.next_outcome.d_mean_entropy},
                {"d_repeat_hit_rate", step.next_outcome.d_repeat_hit_rate},
                {"d_branch_disagreement", step.next_outcome.d_branch_disagreement},
                {"d_verifier_disagreement", step.next_outcome.d_verifier_disagreement},
                {"d_confidence", step.next_outcome.d_confidence},
                {"d_epistemic_pressure", step.next_outcome.d_epistemic_pressure},
                {"d_contradiction_pressure", step.next_outcome.d_contradiction_pressure},
                {"d_runtime_trust", step.next_outcome.d_runtime_trust},
                {"d_stall", step.next_outcome.d_stall},
            }},
        });
    }
    return {
        {"request_id", trace.request_id},
        {"emotive_trace_id", trace.emotive_trace_id},
        {"model", trace.model},
        {"behavior_policy_version", trace.behavior_policy_version},
        {"controller_mode", trace.controller_mode},
        {"candidate_policy_version", trace.candidate_policy_version},
        {"candidate_policy_alias", trace.candidate_policy_alias},
        {"created_at_ms", trace.created_at_ms},
        {"steps", std::move(steps)},
    };
}

bool server_decode_control_trace_from_json(
        const json & payload,
        server_decode_control_trace * out_trace,
        std::string * out_error) {
    if (!out_trace || !payload.is_object()) {
        if (out_error) {
            *out_error = "decode trace must be an object";
        }
        return false;
    }
    server_decode_control_trace trace = {};
    trace.request_id = json_value(payload, "request_id", std::string());
    trace.emotive_trace_id = json_value(payload, "emotive_trace_id", std::string());
    trace.model = json_value(payload, "model", std::string());
    trace.behavior_policy_version = json_value(payload, "behavior_policy_version", std::string("native_decode_control_v1"));
    trace.controller_mode = json_value(payload, "controller_mode", std::string("capture"));
    trace.candidate_policy_version = json_value(payload, "candidate_policy_version", std::string());
    trace.candidate_policy_alias = json_value(payload, "candidate_policy_alias", std::string());
    trace.created_at_ms = json_value(payload, "created_at_ms", int64_t(0));
    const json steps = payload.value("steps", json::array());
    if (steps.is_array()) {
        for (const auto & item : steps) {
            if (!item.is_object()) {
                continue;
            }
            server_decode_control_step_trace step = {};
            step.timestamp_ms = json_value(item, "timestamp_ms", int64_t(0));
            step.seq_id = json_value(item, "seq_id", int32_t(-1));
            step.step_index = json_value(item, "step_index", int32_t(-1));
            step.output_index = json_value(item, "output_index", int32_t(-1));
            server_emotive_vector_from_json(item.value("moment", json::object()), &step.moment, nullptr);
            server_emotive_vad_from_json(item.value("vad", json::object()), &step.vad, nullptr);
            step.runtime_signals = server_runtime_signal_summary_from_json(item.value("runtime_signals", json::object()));
            const json bundles = item.value("bundles", json::object());
            step.bundles.uncertainty_regulation = json_value(bundles, "uncertainty_regulation", 0.0f);
            step.bundles.anti_repetition_recovery = json_value(bundles, "anti_repetition_recovery", 0.0f);
            step.bundles.structural_validity = json_value(bundles, "structural_validity", 0.0f);
            step.bundles.verification_pressure = json_value(bundles, "verification_pressure", 0.0f);
            step.bundles.commit_efficiency = json_value(bundles, "commit_efficiency", 0.0f);
            step.bundles.steering_pressure = json_value(bundles, "steering_pressure", 0.0f);
            server_decode_policy_config_from_json(item.value("decode_policy", json::object()), &step.decode_policy, nullptr);
            step.previous_executed_action_available = json_value(item, "previous_executed_action_available", false);
            action_from_json_local(item.value("previous_executed_action", json::object()), &step.previous_executed_action);
            action_from_json_local(item.value("teacher_action", json::object()), &step.teacher_action);
            step.has_candidate_action = json_value(item, "has_candidate_action", false);
            action_from_json_local(item.value("candidate_action", json::object()), &step.candidate_action);
            candidate_metadata_from_json_local(item.value("candidate_metadata", json::object()), &step.candidate_metadata);
            action_from_json_local(item.value("executed_action", json::object()), &step.executed_action);
            step.generated_cvec_applied = json_value(item, "generated_cvec_applied", false);
            step.generated_cvec_profile_id = json_value(item, "generated_cvec_profile_id", std::string());
            step.generated_cvec_norm = json_value(item, "generated_cvec_norm", 0.0f);
            step.generated_cvec_clipped = json_value(item, "generated_cvec_clipped", false);
            const json next_outcome = item.value("next_outcome", json::object());
            step.next_outcome.available = json_value(next_outcome, "available", false);
            step.next_outcome.d_mean_entropy = json_value(next_outcome, "d_mean_entropy", 0.0f);
            step.next_outcome.d_repeat_hit_rate = json_value(next_outcome, "d_repeat_hit_rate", 0.0f);
            step.next_outcome.d_branch_disagreement = json_value(next_outcome, "d_branch_disagreement", 0.0f);
            step.next_outcome.d_verifier_disagreement = json_value(next_outcome, "d_verifier_disagreement", 0.0f);
            step.next_outcome.d_confidence = json_value(next_outcome, "d_confidence", 0.0f);
            step.next_outcome.d_epistemic_pressure = json_value(next_outcome, "d_epistemic_pressure", 0.0f);
            step.next_outcome.d_contradiction_pressure = json_value(next_outcome, "d_contradiction_pressure", 0.0f);
            step.next_outcome.d_runtime_trust = json_value(next_outcome, "d_runtime_trust", 0.0f);
            step.next_outcome.d_stall = json_value(next_outcome, "d_stall", 0.0f);
            trace.steps.push_back(std::move(step));
        }
    }
    *out_trace = std::move(trace);
    return true;
}

bool server_llama_runtime_control_callback(
        const llama_runtime_control_tick * tick,
        llama_runtime_control_action * action,
        void * user_data) {
    (void) tick;
    (void) action;
    (void) user_data;
    return false;
}
