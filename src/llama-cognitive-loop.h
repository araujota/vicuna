#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

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
    bool cognitive_command_begin_external_wait(int32_t command_id);
    bool cognitive_command_rebind_tool(int32_t command_id, int32_t tool_spec_index);
    bool cognitive_authoritative_react_set_enabled(bool enabled);
    bool cognitive_active_authoritative_begin_tool(int32_t episode_id, uint32_t reason_mask, float priority, int32_t * out_command_id, int32_t * out_tool_job_id);
    bool cognitive_active_authoritative_finish(int32_t episode_id, int32_t terminal_reason);
    bool cognitive_dmn_authoritative_begin_tool(int32_t tick_id, uint32_t reason_mask, float priority, int32_t * out_command_id, int32_t * out_tool_job_id);
    bool cognitive_dmn_authoritative_finish(int32_t tick_id, int32_t terminal_reason);
    bool cognitive_active_tool_emission_note(int32_t command_id, const llama_token * tokens, size_t n_tokens);
    bool cognitive_active_planner_reasoning_note(int32_t episode_id, const llama_token * tokens, size_t n_tokens);
    bool codex_tool_configure(const llama_codex_tool_config & config);
    bool codex_tool_get_config(llama_codex_tool_config * out_config) const;
    bool codex_tool_get_last_result(llama_codex_tool_result * out_result) const;
    bool cognitive_codex_tool_get_request(int32_t command_id, llama_codex_tool_request * out_request) const;
    bool cognitive_codex_tool_set_request(const llama_codex_tool_request & request);
    bool cognitive_bash_tool_submit_result(const llama_bash_tool_result & result, llama_active_loop_trace * out_active_trace);
    bool cognitive_codex_tool_submit_result(const llama_codex_tool_result & result, llama_active_loop_trace * out_active_trace);
    bool cognitive_hard_memory_submit_result(const llama_cognitive_hard_memory_result & result, llama_active_loop_trace * out_active_trace);
    bool cognitive_telegram_relay_get_request(int32_t command_id, llama_telegram_relay_request * out_request) const;
    bool cognitive_telegram_relay_set_request(const llama_telegram_relay_request & request);
    bool cognitive_telegram_relay_submit_result(const llama_telegram_relay_result & result, llama_active_loop_trace * out_active_trace);
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
    llama_cognitive_plan_trace active_plan = {};
    llama_cognitive_plan_trace dmn_plan = {};
    llama_favorable_state_profile last_favorable_profile = {};
    llama_counterfactual_trace last_counterfactual_trace = {};
    llama_remediation_plan last_remediation_plan = {};
    llama_governance_trace last_governance_trace = {};
    llama_temporal_self_improvement_trace last_temporal_self_trace = {};
    llama_cognitive_host_state host_state = {};
    bool authoritative_react_control_enabled = true;
    bool tool_selection_episode_open = false;
    llama_functional_outcome_snapshot tool_selection_before = {};
    int32_t tool_selection_tool_kind = LLAMA_TOOL_KIND_NONE;
    int32_t tool_selection_candidate_count = 0;
    float tool_selection_uncertainty = 0.0f;
    int32_t active_tool_emission_command_id = -1;
    std::vector<llama_token> active_tool_emission_tokens;
    int32_t active_planner_reasoning_episode_id = -1;
    std::vector<llama_token> active_planner_reasoning_tokens;
    llama_codex_tool_config codex_config = {};
    llama_codex_tool_request codex_requests[LLAMA_COGNITIVE_MAX_PENDING_COMMANDS] = {};
    int32_t codex_request_count = 0;
    llama_telegram_relay_request telegram_requests[LLAMA_COGNITIVE_MAX_PENDING_COMMANDS] = {};
    int32_t telegram_request_count = 0;
    llama_codex_tool_result last_codex_result = {};
    bool has_last_codex_result = false;
    llama_telegram_relay_result last_telegram_result = {};
    bool has_last_telegram_result = false;
    llama_dmn_self_model_revision current_self_model_revision = {};
    llama_self_model_revision current_canonical_self_model_revision = {};
    llama_emotive_moment_revision current_emotive_moment_revision = {};
    uint64_t current_dmn_input_hash = 0;
    uint64_t temporal_self_next_trigger_us = 0;
    int32_t next_episode_id = 1;
    int32_t next_tick_id = 1;
    int32_t next_tool_job_id = 1;
    int32_t next_command_id = 1;
    int32_t next_plan_id = 1;
    int32_t next_dmn_self_model_revision_id = 1;
    int32_t next_authoritative_turn_id = 1;
};
