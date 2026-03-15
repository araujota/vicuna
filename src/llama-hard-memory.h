#pragma once

#include "llama.h"

struct llama_vocab;

class llama_hard_memory {
public:
    llama_hard_memory();

    bool configure(const llama_hard_memory_config & config);
    bool get_config(llama_hard_memory_config * out_config) const;
    bool set_request(const llama_cognitive_hard_memory_request & request);
    bool get_request(int32_t command_id, llama_cognitive_hard_memory_request * out_request) const;
    bool clear_request(int32_t command_id);
    bool query(const llama_hard_memory_query_request & query, llama_hard_memory_result * out_result);
    bool submit_result(const llama_hard_memory_result & result);
    bool archive_primitives(
            const llama_hard_memory_primitive * primitives,
            int32_t primitive_count,
            const llama_self_state_delta_summary * delta_summary = nullptr);
    bool get_last_result(llama_hard_memory_result * out_result) const;
    bool archive_event(
            const llama_vocab * vocab,
            const llama_self_state_event & event,
            const llama_self_state_delta_summary & delta);
    bool get_last_archive_trace(llama_hard_memory_archive_trace * out_trace) const;

private:
    llama_hard_memory_config config = {};
    llama_cognitive_hard_memory_request requests[LLAMA_COGNITIVE_MAX_PENDING_COMMANDS] = {};
    int32_t request_count = 0;
    llama_hard_memory_result last_result = {};
    llama_hard_memory_archive_trace last_archive = {};
};
