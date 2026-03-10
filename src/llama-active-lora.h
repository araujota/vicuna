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
    bool get_stats(llama_active_lora_stats * out_stats) const;
    bool tick_past(uint64_t now_us);
    bool get_past_stats(llama_past_lora_stats * out_stats) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
