#pragma once

#include "llama.h"

struct llama_vocab;

class llama_hard_memory {
public:
    llama_hard_memory();

    bool configure(const llama_hard_memory_config & config);
    bool get_config(llama_hard_memory_config * out_config) const;
    bool query(const llama_hard_memory_query_request & query, llama_hard_memory_result * out_result);
    bool get_last_result(llama_hard_memory_result * out_result) const;
    bool archive_event(
            const llama_vocab * vocab,
            const llama_self_state_event & event,
            const llama_self_state_delta_summary & delta);
    bool get_last_archive_trace(llama_hard_memory_archive_trace * out_trace) const;

private:
    llama_hard_memory_config config = {};
    llama_hard_memory_result last_result = {};
    llama_hard_memory_archive_trace last_archive = {};
};
