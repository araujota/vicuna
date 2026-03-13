#pragma once

#include "llama.h"

#include <memory>

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
    bool tick_past(uint64_t now_us);
    bool get_past_stats(llama_past_lora_stats * out_stats) const;
    static int32_t functional_family_count();
    bool functional_family_config_get(int32_t family, llama_functional_lora_family_config * out_config) const;
    bool functional_family_state_get(int32_t family, llama_functional_lora_family_state * out_state) const;
    bool functional_get_last_trace(llama_functional_lora_trace * out_trace) const;
    bool functional_get_last_update(int32_t family, llama_functional_lora_update_info * out_update) const;
    bool functional_set_ablation(const llama_functional_lora_ablation_config & config);
    bool functional_get_ablation(llama_functional_lora_ablation_config * out_config) const;
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

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
