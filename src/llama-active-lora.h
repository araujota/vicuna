#pragma once

#include "llama.h"

#include <array>
#include <memory>

struct llama_functional_gating_observation {
    int32_t loop_origin = 0;
    int32_t microphase = 0;
    uint64_t eligible_mask = 0;
    llama_functional_outcome_snapshot snapshot = {};
    float uncertainty = 0.0f;
    float tool_affinity = 0.0f;
    float continuation = 0.0f;
    float memory_pressure = 0.0f;
    float recovery_urgency = 0.0f;
    float prediction_error = 0.0f;
    float planning_pressure = 0.0f;
    float plan_complexity = 0.0f;
    float plan_revision = 0.0f;
    llama_self_belief_summary belief_summary = {};
    llama_self_model_extension_summary extension_summary = {};
    llama_hard_memory_retrieval_summary hard_memory_summary = {};
};

class llama_active_lora_manager {
public:
    explicit llama_active_lora_manager(llama_context & owner);
    ~llama_active_lora_manager();

    bool init(const llama_active_lora_params & params);
    bool init_past(const llama_past_lora_params & params);
    bool ingest(const llama_token * tokens, size_t n_tokens);
    bool ingest(const llama_self_state_event & event, const llama_self_state_feature_vector * features);
    bool remediate(const llama_token * tokens, size_t n_tokens, float budget_scale);
    bool remediate(const llama_self_state_event & event, float budget_scale, const llama_self_state_feature_vector * features);
    bool get_stats(llama_active_lora_stats * out_stats) const;
    bool user_personality_get_stats(llama_user_personality_lora_stats * out_stats) const;
    bool tick_past(uint64_t now_us);
    bool get_past_stats(llama_past_lora_stats * out_stats) const;
    static int32_t functional_family_count();
    bool functional_family_config_get(int32_t family, llama_functional_lora_family_config * out_config) const;
    bool functional_family_state_get(int32_t family, llama_functional_lora_family_state * out_state) const;
    bool functional_get_last_trace(llama_functional_lora_trace * out_trace) const;
    bool functional_get_last_update(int32_t family, llama_functional_lora_update_info * out_update) const;
    bool functional_snapshot_archive_get(int32_t family, llama_functional_lora_snapshot_archive * out_archive) const;
    bool functional_snapshot_info_get(int32_t family, int32_t slot, llama_functional_lora_snapshot_info * out_info) const;
    bool functional_get_last_snapshot_maintenance(llama_functional_snapshot_maintenance_trace * out_trace) const;
    bool functional_set_ablation(const llama_functional_lora_ablation_config & config);
    bool functional_get_ablation(llama_functional_lora_ablation_config * out_config) const;
    bool functional_replay_override_begin(const llama_functional_lora_replay_override & config);
    bool functional_replay_override_end(int32_t family);
    bool functional_get_last_differential_update(int32_t family, llama_functional_lora_differential_update * out_update) const;
    size_t functional_snapshot_blob_size(int32_t family, int32_t slot) const;
    bool functional_snapshot_blob_export(int32_t family, int32_t slot, void * dst, size_t size) const;
    bool functional_snapshot_blob_import(int32_t family, int32_t slot, const llama_functional_lora_snapshot_info & info, const void * src, size_t size);
    bool functional_snapshot_maintain(uint64_t now_us);
    bool process_functional_get_params(llama_process_functional_params * out_params) const;
    bool process_functional_set_params(const llama_process_functional_params & params);
    int32_t process_functional_entry_count() const;
    bool process_functional_entry_get(int32_t index, llama_process_functional_entry_info * out_info) const;
    int32_t process_functional_ledger_count() const;
    bool process_functional_ledger_get(int32_t index, llama_process_functional_ledger_info * out_info) const;
    bool process_functional_get_last_trace(llama_process_functional_trace * out_trace) const;
    bool process_functional_get_current_signature(llama_process_functional_signature * out_signature) const;
    bool process_functional_snapshot_archive_get(int32_t entry_slot, llama_functional_lora_snapshot_archive * out_archive) const;
    bool process_functional_snapshot_info_get(int32_t entry_slot, int32_t snapshot_slot, llama_functional_lora_snapshot_info * out_info) const;
    bool process_functional_get_last_snapshot_maintenance(llama_functional_snapshot_maintenance_trace * out_trace) const;
    bool process_functional_replay_override_begin(int32_t entry_slot, const llama_functional_lora_replay_override & config);
    bool process_functional_replay_override_end(int32_t entry_slot);
    bool process_functional_get_last_differential_update(int32_t entry_slot, llama_functional_lora_differential_update * out_update) const;
    bool process_functional_apply_differential_update(int32_t entry_slot, int32_t proposal_family, int32_t replay_mode, int32_t snapshot_slot, float signed_score_delta, float magnitude, float robustness_score);
    size_t process_functional_entry_blob_size(int32_t index) const;
    bool process_functional_entry_blob_export(int32_t index, void * dst, size_t size) const;
    bool process_functional_entry_blob_import(int32_t index, const llama_process_functional_entry_info & info, const void * src, size_t size);
    size_t process_functional_snapshot_blob_size(int32_t entry_slot, int32_t snapshot_slot) const;
    bool process_functional_snapshot_blob_export(int32_t entry_slot, int32_t snapshot_slot, void * dst, size_t size) const;
    bool process_functional_snapshot_blob_import(int32_t entry_slot, int32_t snapshot_slot, const llama_functional_lora_snapshot_info & info, const void * src, size_t size);
    bool process_functional_snapshot_maintain(uint64_t now_us);
    bool process_functional_set_execution(const llama_process_functional_signature & signature);
    bool temporal_encoding_bias_get(llama_active_temporal_encoding_bias * out_bias) const;
    bool temporal_encoding_bias_apply(float signed_advantage, float efficiency_advantage, int64_t monotonic_ms);
    bool functional_predict_activation(
            const llama_functional_gating_observation & observation,
            const llama_functional_activation_decision & policy_seed,
            llama_functional_activation_decision * out_decision);
    bool functional_activate(const llama_functional_activation_decision & decision);
    bool functional_note_command_complete(int32_t origin);
    bool functional_apply_update(
            int32_t family,
            int32_t loop_origin,
            int32_t start_microphase,
            int32_t settle_microphase,
            const llama_functional_outcome_snapshot & before,
            const llama_functional_outcome_snapshot & after,
            int32_t selected_tool_kind,
            int32_t candidate_count,
            const float * metrics,
            size_t metric_count,
            float signed_outcome,
            float magnitude,
            const llama_self_state_event & event,
            const llama_self_state_feature_vector * features);
    bool functional_apply_differential_update(
            int32_t family,
            int32_t proposal_family,
            int32_t replay_mode,
            int32_t snapshot_slot,
            float signed_score_delta,
            float magnitude,
            float robustness_score);
    llama_adapter_lora * user_personality_adapter() const;
    float user_personality_scale() const;
    bool user_personality_set_attached(bool attached);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
