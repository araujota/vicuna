#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

struct llama_context;

class llama_cognitive_loop {
public:
    explicit llama_cognitive_loop(llama_context & ctx);

    int32_t cognitive_tool_spec_count() const;
    bool cognitive_tool_spec_get(int32_t index, llama_cognitive_tool_spec * out_spec) const;
    bool cognitive_tool_spec_set(const llama_cognitive_tool_spec * specs, int32_t count);
    int32_t cognitive_command_count() const;
    bool cognitive_command_get(int32_t index, llama_cognitive_command * out_command) const;
    bool cognitive_command_ack(int32_t command_id);
    bool cognitive_command_complete(int32_t command_id, bool cancelled);
    bool cognitive_bash_tool_submit_result(const llama_bash_tool_result & result, llama_active_loop_trace * out_active_trace);
    bool cognitive_active_runner_get(llama_cognitive_active_runner_status * out_status) const;
    bool cognitive_dmn_runner_get(llama_cognitive_dmn_runner_status * out_status) const;

    bool active_loop_process(const llama_self_state_event & event, llama_active_loop_trace * out_trace);
    bool active_loop_note_emit(int32_t episode_id, size_t emitted_text_bytes);
    bool active_loop_get_last_trace(llama_active_loop_trace * out_trace) const;

    bool dmn_tick(uint64_t now_us, llama_dmn_tick_trace * out_trace);
    bool dmn_defer(uint64_t now_us, llama_dmn_tick_trace * out_trace);
    bool dmn_get_last_trace(llama_dmn_tick_trace * out_trace) const;
    bool cognitive_host_state(llama_cognitive_host_state * out_state) const;
    bool favorable_state_get(llama_favorable_state_profile * out_profile) const;
    bool counterfactual_get_last_trace(llama_counterfactual_trace * out_trace) const;
    bool remediation_get_last_plan(llama_remediation_plan * out_plan) const;
    bool governance_get_last_trace(llama_governance_trace * out_trace) const;
    bool temporal_self_improvement_get_last(llama_temporal_self_improvement_trace * out_trace) const;

private:
    llama_context & ctx;
    llama_cognitive_tool_spec tool_specs[LLAMA_COGNITIVE_MAX_TOOL_SPECS] = {};
    int32_t tool_spec_count = 0;
    llama_cognitive_command command_queue[LLAMA_COGNITIVE_MAX_PENDING_COMMANDS] = {};
    int32_t command_count = 0;
    llama_cognitive_active_runner_status active_runner = {};
    llama_cognitive_dmn_runner_status dmn_runner = {};
    llama_active_loop_trace last_active_trace = {};
    llama_dmn_tick_trace last_dmn_trace = {};
    llama_favorable_state_profile last_favorable_profile = {};
    llama_counterfactual_trace last_counterfactual_trace = {};
    llama_remediation_plan last_remediation_plan = {};
    llama_governance_trace last_governance_trace = {};
    llama_temporal_self_improvement_trace last_temporal_self_trace = {};
    llama_cognitive_host_state host_state = {};
    bool tool_selection_episode_open = false;
    llama_functional_outcome_snapshot tool_selection_before = {};
    int32_t tool_selection_tool_kind = LLAMA_TOOL_KIND_NONE;
    int32_t tool_selection_candidate_count = 0;
    float tool_selection_uncertainty = 0.0f;
    uint64_t temporal_self_next_trigger_us = 0;
    int32_t next_episode_id = 1;
    int32_t next_tick_id = 1;
    int32_t next_tool_job_id = 1;
    int32_t next_command_id = 1;
};
