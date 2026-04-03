#pragma once

#include "server-cvec-generator.h"
#include "server-emotive-runtime.h"

#include <memory>

struct llama_context {};
struct llama_runtime_telemetry_event {};
struct llama_runtime_control_tick {};

enum llama_runtime_control_stage : int32_t {
    LLAMA_RUNTIME_CONTROL_STAGE_UNKNOWN = 0,
    LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY = 1,
};

struct llama_runtime_control_signal_bundles {
    float uncertainty_regulation = 0.0f;
    float anti_repetition_recovery = 0.0f;
    float structural_validity = 0.0f;
    float verification_pressure = 0.0f;
    float commit_efficiency = 0.0f;
    float steering_pressure = 0.0f;
};

struct llama_runtime_control_action_mask {
    bool allow_sampling = false;
    bool allow_repetition = false;
    bool allow_structure = false;
    bool allow_branch = false;
    bool allow_steering = false;
    int32_t max_branch_sample_count = 1;
};

inline llama_runtime_control_action_mask llama_runtime_control_action_mask_default() {
    return {};
}

struct llama_runtime_control_limits {
    float min_temperature = 0.0f;
    float max_temperature = 2.0f;
    float min_top_p = 0.0f;
    float max_top_p = 1.0f;
    float min_min_p = 0.0f;
    float max_min_p = 1.0f;
    float min_typical_p = 0.0f;
    float max_typical_p = 1.0f;
    float max_top_n_sigma = 8.0f;
    float min_repeat_penalty = 0.0f;
    float max_repeat_penalty = 2.0f;
    float max_frequency_penalty = 2.0f;
    float max_presence_penalty = 2.0f;
    float max_dry_multiplier = 2.0f;
    int32_t max_penalty_last_n = 256;
    int32_t max_dry_allowed_length = 256;
    int32_t max_dry_penalty_last_n = 256;
};

inline llama_runtime_control_limits llama_runtime_control_limits_default() {
    return {};
}

struct llama_runtime_control_sampling_action {
    bool enabled = false;
    bool has_temperature = false;
    bool has_top_k = false;
    bool has_top_p = false;
    bool has_min_p = false;
    bool has_typical_p = false;
    bool has_top_n_sigma = false;
    float temperature = 0.0f;
    int32_t top_k = 0;
    float top_p = 0.0f;
    float min_p = 0.0f;
    float typical_p = 0.0f;
    float top_n_sigma = 0.0f;
    int32_t min_keep = 0;
};

struct llama_runtime_control_repetition_action {
    bool enabled = false;
    bool has_repeat_penalty = false;
    bool has_frequency_penalty = false;
    bool has_presence_penalty = false;
    bool has_penalty_last_n = false;
    float repeat_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int32_t penalty_last_n = 0;
};

struct llama_runtime_control_structure_action {
    bool enabled = false;
    bool clear_grammar = false;
    bool clear_logit_bias = false;
    char grammar_profile_id[128] = {};
    char logit_bias_profile_id[128] = {};
};

struct llama_runtime_control_branch_action {
    bool enabled = false;
    bool checkpoint_now = false;
    bool request_verify = false;
    int32_t checkpoint_slot = 0;
    int32_t restore_slot = 0;
    int32_t branch_sample_count = 0;
};

struct llama_runtime_control_steering_action {
    bool enabled = false;
    bool clear_cvec = false;
    char cvec_profile_id[128] = {};
};

struct llama_runtime_control_action {
    int32_t schema_version = 1;
    int32_t stage = LLAMA_RUNTIME_CONTROL_STAGE_UNKNOWN;
    char proposal_source[64] = {};
    char summary[128] = {};
    bool valid = false;
    llama_runtime_control_sampling_action sampling = {};
    llama_runtime_control_repetition_action repetition = {};
    llama_runtime_control_structure_action structure = {};
    llama_runtime_control_branch_action branch = {};
    llama_runtime_control_steering_action steering = {};
};

struct server_llama_runtime_control_plan {
    llama_runtime_control_signal_bundles bundles = {};
    llama_runtime_control_action action = {};
};

struct server_decode_policy_config {
    std::string schema_version = "decode_policy_config_v1";
    float base_temperature = 0.0f;
    int32_t base_top_k = 0;
    float base_top_p = 0.0f;
    float base_min_p = 0.0f;
    llama_runtime_control_action_mask action_mask = llama_runtime_control_action_mask_default();
    llama_runtime_control_limits control_limits = llama_runtime_control_limits_default();
};

struct server_decode_control_outcome_proxy {
    bool available = false;
    float d_mean_entropy = 0.0f;
    float d_repeat_hit_rate = 0.0f;
    float d_branch_disagreement = 0.0f;
    float d_verifier_disagreement = 0.0f;
    float d_confidence = 0.0f;
    float d_epistemic_pressure = 0.0f;
    float d_contradiction_pressure = 0.0f;
    float d_runtime_trust = 0.0f;
    float d_stall = 0.0f;
};

struct server_decode_control_candidate_metadata {
    bool available = false;
    std::string controller_version;
    std::string controller_alias;
    bool confidence_present = false;
    float confidence = 0.0f;
    bool executed_live = false;
    std::string proposal_source = "decode_controller";
};

struct server_decode_control_observation {
    server_emotive_vector moment = {};
    server_emotive_vad vad = {};
    server_runtime_signal_summary runtime_signals = {};
    llama_runtime_control_signal_bundles bundles = {};
    server_decode_policy_config decode_policy = {};
    llama_runtime_control_action_mask mask = llama_runtime_control_action_mask_default();
    llama_runtime_control_action teacher_action = {};
    bool previous_executed_action_available = false;
    llama_runtime_control_action previous_executed_action = {};
    int64_t timestamp_ms = 0;
    int32_t seq_id = -1;
    int32_t step_index = -1;
    int32_t output_index = -1;
};

typedef bool (*server_decode_controller_callback)(
        const server_decode_control_observation * observation,
        llama_runtime_control_action * out_action,
        server_decode_control_candidate_metadata * out_metadata,
        void * user_data);

struct server_cvec_rollout_metadata {
    bool active_available = false;
    std::string active_generator_version;
    std::string active_generator_alias;
    bool candidate_available = false;
    std::string candidate_generator_version;
    std::string candidate_generator_alias;
    bool candidate_executed_live = false;
    bool candidate_compared = false;
    float candidate_cosine_similarity = 0.0f;
    float candidate_norm_delta = 0.0f;
};

struct server_decode_control_step_trace {
    int64_t timestamp_ms = 0;
    int32_t seq_id = -1;
    int32_t step_index = -1;
    int32_t output_index = -1;
    server_emotive_vector moment = {};
    server_emotive_vad vad = {};
    server_runtime_signal_summary runtime_signals = {};
    llama_runtime_control_signal_bundles bundles = {};
    server_decode_policy_config decode_policy = {};
    bool previous_executed_action_available = false;
    llama_runtime_control_action previous_executed_action = {};
    llama_runtime_control_action teacher_action = {};
    bool has_candidate_action = false;
    llama_runtime_control_action candidate_action = {};
    server_decode_control_candidate_metadata candidate_metadata = {};
    llama_runtime_control_action executed_action = {};
    bool generated_cvec_applied = false;
    std::string generated_cvec_profile_id;
    float generated_cvec_norm = 0.0f;
    bool generated_cvec_clipped = false;
    server_cvec_rollout_metadata cvec_rollout = {};
    server_decode_control_outcome_proxy next_outcome = {};
};

struct server_decode_control_trace {
    std::string request_id;
    std::string emotive_trace_id;
    std::string model;
    std::string behavior_policy_version = "native_decode_control_v1";
    std::string controller_mode = "capture";
    std::string candidate_policy_version;
    std::string candidate_policy_alias;
    int64_t created_at_ms = 0;
    std::vector<server_decode_control_step_trace> steps;
};

struct server_llama_runtime_generated_cvec_profile {
    bool available = false;
    std::string profile_id;
    std::vector<float> data;
    int32_t n_embd = 0;
    int32_t il_start = 0;
    int32_t il_end = -1;
    float norm = 0.0f;
    bool clipped = false;
};

struct server_decode_controller_artifact;

struct server_llama_runtime_control_state {
    server_emotive_vector moment = {};
    server_emotive_vad vad = {};
    server_runtime_signal_summary runtime_signals = {};
    server_decode_policy_config decode_policy = {};
    llama_runtime_control_action_mask mask = llama_runtime_control_action_mask_default();
    llama_context * runtime_ctx = nullptr;
    std::shared_ptr<server_cvec_generator_artifact> cvec_generator;
    std::shared_ptr<server_cvec_generator_artifact> candidate_cvec_generator;
    std::shared_ptr<server_decode_controller_artifact> decode_controller;
    std::shared_ptr<server_decode_controller_artifact> candidate_decode_controller;
    std::vector<float> decode_controller_hidden;
    std::vector<float> candidate_decode_controller_hidden;
    std::string generated_cvec_profile_id = "emvad_generated";
    std::string model_architecture;
    float base_temperature = 0.0f;
    int32_t base_top_k = 0;
    float base_top_p = 0.0f;
    float base_min_p = 0.0f;
    std::string request_id;
    std::string resolved_model;
    std::string decode_controller_mode = "capture";
    bool decode_candidate_execute_live = false;
    std::string decode_controller_alias;
    std::string decode_controller_version;
    std::string candidate_decode_controller_alias;
    std::string candidate_decode_controller_version;
    std::string cvec_generator_mode = "capture";
    bool cvec_candidate_execute_live = false;
    std::string cvec_generator_alias;
    std::string cvec_generator_version;
    std::string candidate_cvec_generator_alias;
    std::string candidate_cvec_generator_version;
    server_decode_controller_callback decode_controller_callback = nullptr;
    void * decode_controller_user_data = nullptr;
    server_decode_control_trace active_decode_trace = {};
    bool has_last_executed_action = false;
    llama_runtime_control_action last_executed_action = {};
};

server_runtime_signal_summary server_runtime_signal_summary_from_telemetry_event(
        const llama_runtime_telemetry_event & event);

llama_runtime_control_signal_bundles server_llama_runtime_control_bundles_from_state(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        const server_runtime_signal_summary & runtime_signals);

llama_runtime_control_action server_llama_runtime_control_action_from_state(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        const server_runtime_signal_summary & runtime_signals,
        const llama_runtime_control_action_mask * mask = nullptr);

server_llama_runtime_control_plan server_llama_runtime_control_plan_from_state(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        const server_runtime_signal_summary & runtime_signals,
        const llama_runtime_control_action_mask * mask = nullptr);

bool server_llama_runtime_control_generate_cvec_profile(
        const server_llama_runtime_control_state & state,
        server_llama_runtime_generated_cvec_profile * out_profile,
        std::string * out_error);

void server_llama_runtime_control_begin_request(
        server_llama_runtime_control_state * state,
        const std::string & request_id,
        const std::string & model_name);

server_decode_control_trace server_llama_runtime_control_finalize_trace(
        server_llama_runtime_control_state * state,
        const std::string & emotive_trace_id);

json server_llama_runtime_control_action_to_json(const llama_runtime_control_action & action);
json server_llama_runtime_control_signal_bundles_to_json(const llama_runtime_control_signal_bundles & bundles);
json server_decode_policy_config_to_json(const server_decode_policy_config & config);
bool server_decode_policy_config_from_json(
        const json & payload,
        server_decode_policy_config * out_config,
        std::string * out_error = nullptr);
json server_decode_control_trace_to_json(const server_decode_control_trace & trace);
bool server_decode_control_trace_from_json(
        const json & payload,
        server_decode_control_trace * out_trace,
        std::string * out_error = nullptr);

bool server_llama_runtime_control_callback(
        const llama_runtime_control_tick * tick,
        llama_runtime_control_action * action,
        void * user_data);
