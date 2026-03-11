#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

struct llama_context;

class llama_cognitive_loop {
public:
    explicit llama_cognitive_loop(llama_context & ctx);

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

private:
    llama_context & ctx;
    llama_active_loop_trace last_active_trace = {};
    llama_dmn_tick_trace last_dmn_trace = {};
    llama_favorable_state_profile last_favorable_profile = {};
    llama_counterfactual_trace last_counterfactual_trace = {};
    llama_remediation_plan last_remediation_plan = {};
    llama_governance_trace last_governance_trace = {};
    llama_cognitive_host_state host_state = {};
    int32_t next_episode_id = 1;
    int32_t next_tick_id = 1;
    int32_t next_tool_job_id = 1;
};
