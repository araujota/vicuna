#include "tools/server/server-runtime-control.h"
#include "tools/server/server-decode-controller.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

static bool streq(const char * lhs, const char * rhs) {
    return std::strcmp(lhs ? lhs : "", rhs ? rhs : "") == 0;
}

static bool test_decode_candidate_callback(
        const server_decode_control_observation * observation,
        llama_runtime_control_action * out_action,
        server_decode_control_candidate_metadata * out_metadata,
        void * user_data) {
    (void) user_data;
    if (observation == nullptr || out_action == nullptr || out_metadata == nullptr) {
        return false;
    }
    *out_action = observation->teacher_action;
    out_action->valid = true;
    std::snprintf(out_action->summary, sizeof(out_action->summary), "%s", "candidate_override");
    out_metadata->available = true;
    out_metadata->controller_version = "decode-gru-shadow-v1";
    out_metadata->controller_alias = "candidate";
    out_metadata->confidence_present = true;
    out_metadata->confidence = 0.81f;
    return true;
}

int main() {
    {
        server_decode_policy_config config = {};
        config.base_temperature = 0.7f;
        config.base_top_k = 24;
        config.base_top_p = 0.91f;
        config.base_min_p = 0.05f;
        config.action_mask.allow_sampling = true;
        config.action_mask.allow_repetition = false;
        config.action_mask.allow_structure = true;
        config.action_mask.allow_branch = false;
        config.action_mask.allow_steering = true;
        config.action_mask.max_branch_sample_count = 3;
        config.control_limits.max_temperature = 1.5f;
        config.control_limits.max_top_p = 0.98f;

        const auto payload = server_decode_policy_config_to_json(config);
        server_decode_policy_config parsed = {};
        assert(server_decode_policy_config_from_json(payload, &parsed, nullptr));
        assert(parsed.base_top_k == 24);
        assert(parsed.action_mask.allow_repetition == false);
        assert(parsed.control_limits.max_temperature == 1.5f);
    }

    {
        llama_runtime_telemetry_event event = {};
        event.schema_version = 1;
        event.type = LLAMA_RUNTIME_TELEMETRY_EVENT_MEMORY_RAIL_PLANNED;
        event.memory_strategy = LLAMA_RUNTIME_MEMORY_STRATEGY_HYBRID_RECURRENT;
        event.sink_materialization_mode = LLAMA_RUNTIME_MEMORY_SINK_MATERIALIZATION_METADATA_ONLY;
        std::snprintf(event.memory_strategy_label, sizeof(event.memory_strategy_label), "%s", "hybrid_recurrent");
        std::snprintf(event.sink_materialization_label, sizeof(event.sink_materialization_label), "%s", "metadata_only");
        event.attention_budget_ratio = 0.93f;
        event.recurrent_budget_ratio = 1.0f;
        event.attention_pos_min = 128;
        event.attention_pos_max = 4096;
        event.recurrent_pos_min = 4096;
        event.recurrent_pos_max = 4096;

        const auto summary = server_runtime_signal_summary_from_telemetry_event(event);
        assert(summary.available);
        assert(summary.memory_strategy_label == "hybrid_recurrent");
        assert(summary.sink_materialization_label == "metadata_only");
        assert(summary.attention_budget_ratio > 0.9f);
        assert(summary.recurrent_pos_min == 4096);
    }

    {
        server_emotive_vector moment = {};
        moment.epistemic_pressure = 0.88f;
        moment.caution = 0.74f;
        moment.contradiction_pressure = 0.69f;
        moment.confidence = 0.18f;
        moment.runtime_trust = 0.22f;
        moment.runtime_failure_pressure = 0.61f;

        server_emotive_vad vad = {};
        vad.valence = -0.35f;
        vad.arousal = 0.64f;
        vad.dominance = -0.56f;

        server_runtime_signal_summary signals = {};
        signals.prompt_section_label = "answer";
        signals.branch_disagreement = 0.72f;
        signals.verifier_disagreement = 0.66f;
        signals.verifier_active = true;

        const auto plan = server_llama_runtime_control_plan_from_state(moment, vad, signals);
        assert(plan.bundles.uncertainty_regulation > 0.6f);
        assert(plan.bundles.verification_pressure > 0.6f);
        assert(plan.action.stage == LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY);
        assert(plan.action.sampling.enabled);
        assert(plan.action.sampling.has_temperature);
        assert(plan.action.sampling.has_top_k);
        assert(plan.action.sampling.has_top_p);
        assert(plan.action.sampling.has_min_p);
        assert(plan.action.sampling.has_typical_p);
        assert(plan.action.sampling.has_top_n_sigma);
        assert(plan.action.branch.enabled);
        assert(plan.action.branch.checkpoint_now);
        assert(plan.action.branch.request_verify);
        assert(plan.action.branch.branch_sample_count >= 2);
        assert(plan.action.steering.enabled);
        assert(streq(plan.action.steering.cvec_profile_id, "truthfulness"));
    }

    {
        server_emotive_vector moment = {};
        moment.stall = 0.86f;
        moment.frustration = 0.78f;
        moment.semantic_novelty = 0.10f;
        moment.momentum = 0.19f;
        moment.planning_clarity = 0.28f;

        server_emotive_vad vad = {};
        vad.dominance = -0.14f;

        server_runtime_signal_summary signals = {};
        signals.repeat_hit_rate = 0.67f;

        const auto plan = server_llama_runtime_control_plan_from_state(moment, vad, signals);
        assert(plan.bundles.anti_repetition_recovery > 0.6f);
        assert(plan.action.repetition.enabled);
        assert(plan.action.repetition.has_repeat_penalty);
        assert(plan.action.repetition.has_frequency_penalty);
        assert(plan.action.repetition.has_presence_penalty);
        assert(plan.action.repetition.has_dry_multiplier);
        assert(plan.action.repetition.penalty_last_n == 96);
        assert(streq(plan.action.summary, "anti_repetition_overlay"));
    }

    {
        server_emotive_vector moment = {};
        moment.confidence = 0.89f;
        moment.planning_clarity = 0.91f;
        moment.runtime_trust = 0.86f;
        moment.satisfaction = 0.73f;
        moment.user_alignment = 0.82f;
        moment.caution = 0.62f;

        server_emotive_vad vad = {};
        vad.valence = 0.22f;
        vad.dominance = 0.58f;

        server_runtime_signal_summary signals = {};
        signals.prompt_section_label = "json";

        const auto plan = server_llama_runtime_control_plan_from_state(moment, vad, signals);
        assert(plan.bundles.commit_efficiency > 0.7f);
        assert(plan.action.sampling.enabled);
        assert(plan.action.sampling.has_temperature);
        assert(plan.action.sampling.has_top_k);
        assert(plan.action.sampling.temperature <= 0.55f);
        assert(plan.action.sampling.has_top_p);
        assert(plan.action.structure.enabled);
        assert(plan.action.steering.enabled);
        assert(streq(plan.action.steering.cvec_profile_id, "concise_direct"));
        assert(streq(plan.action.structure.grammar_profile_id, "json"));
    }

    {
        server_emotive_vector moment = {};
        moment.confidence = 0.34f;
        moment.curiosity = 0.28f;
        moment.frustration = 0.08f;
        moment.satisfaction = 0.22f;
        moment.momentum = 0.31f;
        moment.caution = 0.18f;
        moment.stall = 0.09f;
        moment.epistemic_pressure = 0.19f;
        moment.planning_clarity = 0.42f;
        moment.user_alignment = 0.41f;
        moment.semantic_novelty = 0.30f;
        moment.runtime_trust = 0.52f;
        moment.runtime_failure_pressure = 0.05f;
        moment.contradiction_pressure = 0.04f;

        server_emotive_vad vad = {};
        vad.valence = -0.12f;
        vad.arousal = 0.11f;
        vad.dominance = 0.08f;

        server_runtime_signal_summary signals = {};
        signals.prompt_section_label = "answer";

        const auto action = server_llama_runtime_control_action_from_state(moment, vad, signals, nullptr);
        assert(action.steering.enabled);
        assert(streq(action.steering.cvec_profile_id, ""));
    }

    {
        server_emotive_vector moment = {};
        moment.epistemic_pressure = 0.92f;
        moment.contradiction_pressure = 0.80f;
        moment.confidence = 0.12f;
        moment.runtime_trust = 0.14f;
        moment.runtime_failure_pressure = 0.84f;

        server_emotive_vad vad = {};
        vad.dominance = -0.78f;

        server_runtime_signal_summary signals = {};
        signals.prompt_section_label = "xml";
        signals.branch_disagreement = 0.91f;
        signals.verifier_disagreement = 0.74f;
        signals.verifier_active = true;

        llama_runtime_control_action_mask mask = llama_runtime_control_action_mask_default();
        mask.allow_branch = false;
        mask.allow_steering = false;

        const auto action = server_llama_runtime_control_action_from_state(moment, vad, signals, &mask);
        assert(action.sampling.enabled);
        assert(action.structure.enabled);
        assert(streq(action.structure.grammar_profile_id, "xml"));
        assert(!action.branch.enabled);
        assert(!action.steering.enabled);
    }

    {
        auto artifact = std::make_shared<server_cvec_generator_artifact>();
        artifact->schema_version = "vicuna.cvec_generator_artifact.v1";
        artifact->generator_version = "test-gen-1";
        artifact->activation = "tanh";
        artifact->output_mode = "none";
        artifact->target_embedding_dim = 4;
        artifact->target_layer_start = 2;
        artifact->target_layer_end = 6;
        artifact->output_norm_cap = 8.0f;
        artifact->input_mean.assign(17, 0.0f);
        artifact->input_std.assign(17, 1.0f);
        server_cvec_generator_layer hidden = {};
        hidden.weights = {
            std::vector<float>(17, 0.10f),
            std::vector<float>(17, -0.05f),
            std::vector<float>(17, 0.02f),
        };
        hidden.bias = {0.01f, -0.02f, 0.03f};
        server_cvec_generator_layer output = {};
        output.weights = {
            {0.50f, 0.10f, -0.20f},
            {-0.30f, 0.20f, 0.40f},
            {0.25f, -0.35f, 0.15f},
            {0.05f, 0.45f, 0.30f},
        };
        output.bias = {0.01f, 0.02f, -0.01f, 0.03f};
        artifact->layers = {hidden, output};

        server_llama_runtime_control_state state = {};
        state.moment.confidence = 0.62f;
        state.moment.curiosity = 0.48f;
        state.moment.frustration = 0.12f;
        state.moment.satisfaction = 0.55f;
        state.moment.momentum = 0.44f;
        state.moment.caution = 0.39f;
        state.moment.stall = 0.08f;
        state.moment.epistemic_pressure = 0.28f;
        state.moment.planning_clarity = 0.73f;
        state.moment.user_alignment = 0.81f;
        state.moment.semantic_novelty = 0.33f;
        state.moment.runtime_trust = 0.77f;
        state.moment.runtime_failure_pressure = 0.06f;
        state.moment.contradiction_pressure = 0.15f;
        state.vad.valence = 0.18f;
        state.vad.arousal = 0.29f;
        state.vad.dominance = 0.41f;
        state.cvec_generator = artifact;
        state.generated_cvec_profile_id = "generated_test";

        server_llama_runtime_generated_cvec_profile generated = {};
        std::string error;
        const bool ok = server_llama_runtime_control_generate_cvec_profile(state, &generated, &error);
        assert(ok);
        assert(generated.available);
        assert(generated.profile_id == "generated_test");
        assert(generated.data.size() == 4);
        assert(generated.n_embd == 4);
        assert(generated.il_start == 2);
        assert(generated.il_end == 6);
        assert(generated.norm > 0.0f);
    }

    {
        json payload = {
            {"schema_version", "vicuna.decode_controller_artifact.v1"},
            {"controller_version", "decode-gru-test"},
            {"input_schema", {{"input_dimension", 133}}},
            {"normalization", {{"input_mean", json::array()}, {"input_std", json::array()}}},
            {"architecture", {{"hidden_dim", 4}}},
            {"action_schema", {
                {"boolean_fields", {"valid","sampling.enabled","sampling.has_temperature","sampling.has_top_k","sampling.has_top_p","sampling.has_min_p","sampling.has_typical_p","sampling.has_top_n_sigma","repetition.enabled","repetition.has_repeat_penalty","repetition.has_frequency_penalty","repetition.has_presence_penalty","repetition.has_penalty_last_n","structure.enabled","structure.clear_grammar","structure.clear_logit_bias","branch.enabled","branch.checkpoint_now","branch.request_verify","steering.enabled","steering.clear_cvec"}},
                {"numeric_fields", {"sampling.temperature","sampling.top_k","sampling.top_p","sampling.min_p","sampling.typical_p","sampling.top_n_sigma","sampling.min_keep","repetition.repeat_penalty","repetition.frequency_penalty","repetition.presence_penalty","repetition.penalty_last_n","branch.checkpoint_slot","branch.restore_slot","branch.branch_sample_count"}},
                {"numeric_ranges", {
                    {"sampling.temperature", {0,2}}, {"sampling.top_k",{0,200}}, {"sampling.top_p",{0,1}}, {"sampling.min_p",{0,1}}, {"sampling.typical_p",{0,1}}, {"sampling.top_n_sigma",{0,8}}, {"sampling.min_keep",{0,16}},
                    {"repetition.repeat_penalty",{0,2}}, {"repetition.frequency_penalty",{0,2}}, {"repetition.presence_penalty",{0,2}}, {"repetition.penalty_last_n",{0,512}},
                    {"branch.checkpoint_slot",{0,8}}, {"branch.restore_slot",{0,8}}, {"branch.branch_sample_count",{0,8}},
                }},
                {"profile_fields", {"structure.grammar_profile_id","structure.logit_bias_profile_id","steering.cvec_profile_id"}},
                {"profile_vocabs", {
                    {"structure.grammar_profile_id", {"", "json"}},
                    {"structure.logit_bias_profile_id", {""}},
                    {"steering.cvec_profile_id", {"", "truthfulness"}},
                }},
            }},
            {"state_dict", json::object()},
            {"training_metrics", {{"record_count", 2}, {"total", 0.1}}},
        };
        for (int i = 0; i < 133; ++i) {
            payload["normalization"]["input_mean"].push_back(0.0);
            payload["normalization"]["input_std"].push_back(1.0);
        }
        auto zero_matrix = [](int rows, int cols) {
            json matrix = json::array();
            for (int r = 0; r < rows; ++r) {
                json row = json::array();
                for (int c = 0; c < cols; ++c) {
                    row.push_back(0.0);
                }
                matrix.push_back(std::move(row));
            }
            return matrix;
        };
        payload["state_dict"]["gru.weight_ih_l0"] = zero_matrix(12, 133);
        payload["state_dict"]["gru.weight_hh_l0"] = zero_matrix(12, 4);
        payload["state_dict"]["gru.bias_ih_l0"] = std::vector<float>(12, 0.0f);
        payload["state_dict"]["gru.bias_hh_l0"] = std::vector<float>(12, 0.0f);
        payload["state_dict"]["bool_head.weight"] = zero_matrix(21, 4);
        payload["state_dict"]["bool_head.bias"] = {1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1};
        payload["state_dict"]["numeric_head.weight"] = zero_matrix(14, 4);
        payload["state_dict"]["numeric_head.bias"] = {0.7,24,0.9,0,0,0,1,0,0,0,0,0,0,0};
        payload["state_dict"]["profile_heads.structure.grammar_profile_id.weight"] = zero_matrix(2, 4);
        payload["state_dict"]["profile_heads.structure.grammar_profile_id.bias"] = {1,0};
        payload["state_dict"]["profile_heads.structure.logit_bias_profile_id.weight"] = zero_matrix(1, 4);
        payload["state_dict"]["profile_heads.structure.logit_bias_profile_id.bias"] = {1};
        payload["state_dict"]["profile_heads.steering.cvec_profile_id.weight"] = zero_matrix(2, 4);
        payload["state_dict"]["profile_heads.steering.cvec_profile_id.bias"] = {0,1};

        const std::string path = "/tmp/vicuna_decode_controller_test.json";
        std::ofstream out(path);
        out << payload.dump();
        out.close();

        server_decode_controller_artifact artifact = {};
        std::string error;
        const bool loaded = server_decode_controller_load_artifact(path, &artifact, &error);
        assert(loaded);
        assert(artifact.controller_version == "decode-gru-test");

        server_llama_runtime_control_state state = {};
        state.decode_controller = std::make_shared<server_decode_controller_artifact>(artifact);
        state.moment.confidence = 0.5f;
        state.vad.valence = 0.1f;
        state.runtime_signals.prompt_section_label = "answer";

        server_decode_control_observation observation = {};
        observation.moment = state.moment;
        observation.vad = state.vad;
        observation.runtime_signals = state.runtime_signals;
        observation.mask = llama_runtime_control_action_mask_default();
        observation.decode_policy = {};
        observation.decode_policy.base_temperature = 0.7f;
        observation.decode_policy.base_top_k = 24;
        observation.decode_policy.base_top_p = 0.9f;
        observation.decode_policy.base_min_p = 0.05f;
        observation.decode_policy.action_mask = observation.mask;
        observation.teacher_action = {};

        llama_runtime_control_action action = {};
        server_decode_control_candidate_metadata meta = {};
        const bool ok = server_decode_controller_predict(
                artifact,
                observation,
                &state.decode_controller_hidden,
                &action,
                &meta,
                &error);
        assert(ok);
        assert(action.valid);
        assert(action.sampling.enabled);
        assert(action.sampling.has_temperature);
        assert(action.steering.enabled);
        assert(streq(action.steering.cvec_profile_id, "truthfulness"));
        assert(meta.available);
        assert(meta.controller_version == "decode-gru-test");
    }

    {
        server_llama_runtime_control_state state = {};
        state.moment.confidence = 0.85f;
        state.moment.planning_clarity = 0.90f;
        state.moment.runtime_trust = 0.88f;
        state.vad.dominance = 0.48f;
        state.runtime_signals.prompt_section_label = "json";

        llama_runtime_control_tick tick = {};
        tick.schema_version = 1;
        tick.stage = LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY;

        llama_runtime_control_action action = {};
        const bool ok = server_llama_runtime_control_callback(&tick, &action, &state);
        assert(ok);
        assert(action.valid);
        assert(action.stage == LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY);
        assert(action.sampling.enabled);
    }

    {
        server_llama_runtime_control_state state = {};
        state.moment.confidence = 0.48f;
        state.moment.epistemic_pressure = 0.31f;
        state.moment.runtime_trust = 0.58f;
        state.vad.valence = 0.05f;
        state.runtime_signals.prompt_section_label = "answer";
        state.decode_controller_mode = "shadow";
        server_llama_runtime_control_begin_request(&state, "req-gru-1", "qwen-local");

        llama_runtime_control_tick first_tick = {};
        first_tick.schema_version = 1;
        first_tick.stage = LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY;
        first_tick.seq_id = 7;
        first_tick.output_index = 0;
        first_tick.latest_event.schema_version = 1;
        first_tick.latest_event.step_index = 1;
        first_tick.latest_event.timestamp_us = 1000;
        first_tick.latest_event.mean_entropy = 0.60f;
        first_tick.latest_event.repeat_hit_rate = 0.05f;
        first_tick.latest_event.branch_disagreement = 0.10f;
        first_tick.latest_event.verifier_disagreement = 0.08f;

        llama_runtime_control_action first_action = {};
        const bool first_ok = server_llama_runtime_control_callback(&first_tick, &first_action, &state);
        assert(first_ok);
        assert(state.active_decode_trace.steps.size() == 1);
        assert(!state.active_decode_trace.steps[0].next_outcome.available);

        state.moment.confidence = 0.56f;
        state.moment.epistemic_pressure = 0.22f;
        state.moment.runtime_trust = 0.66f;
        state.moment.stall = 0.03f;

        llama_runtime_control_tick second_tick = {};
        second_tick.schema_version = 1;
        second_tick.stage = LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY;
        second_tick.seq_id = 7;
        second_tick.output_index = 1;
        second_tick.latest_event.schema_version = 1;
        second_tick.latest_event.step_index = 2;
        second_tick.latest_event.timestamp_us = 2000;
        second_tick.latest_event.mean_entropy = 0.42f;
        second_tick.latest_event.repeat_hit_rate = 0.02f;
        second_tick.latest_event.branch_disagreement = 0.04f;
        second_tick.latest_event.verifier_disagreement = 0.02f;

        llama_runtime_control_action second_action = {};
        const bool second_ok = server_llama_runtime_control_callback(&second_tick, &second_action, &state);
        assert(second_ok);
        assert(state.active_decode_trace.steps.size() == 2);
        assert(state.active_decode_trace.steps[0].next_outcome.available);
        assert(state.active_decode_trace.steps[0].next_outcome.d_mean_entropy < 0.0f);
        assert(state.active_decode_trace.steps[1].previous_executed_action_available);

        const auto trace = server_llama_runtime_control_finalize_trace(&state, "trace-gru-1");
        assert(trace.request_id == "req-gru-1");
        assert(trace.emotive_trace_id == "trace-gru-1");
        assert(trace.steps.size() == 2);
    }

    {
        server_llama_runtime_control_state state = {};
        state.moment.confidence = 0.42f;
        state.moment.epistemic_pressure = 0.38f;
        state.runtime_signals.prompt_section_label = "answer";
        state.decode_controller_mode = "shadow";
        state.decode_controller_callback = test_decode_candidate_callback;
        server_llama_runtime_control_begin_request(&state, "req-gru-2", "qwen-local");

        llama_runtime_control_tick tick = {};
        tick.schema_version = 1;
        tick.stage = LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY;
        tick.seq_id = 3;
        tick.output_index = 0;
        tick.latest_event.schema_version = 1;
        tick.latest_event.step_index = 5;
        tick.latest_event.timestamp_us = 5000;

        llama_runtime_control_action action = {};
        const bool ok = server_llama_runtime_control_callback(&tick, &action, &state);
        assert(ok);
        assert(state.active_decode_trace.steps.size() == 1);
        assert(state.active_decode_trace.steps[0].has_candidate_action);
        assert(!state.active_decode_trace.steps[0].candidate_metadata.executed_live);
        assert(state.active_decode_trace.steps[0].candidate_metadata.controller_version == "decode-gru-shadow-v1");
    }

    return 0;
}
