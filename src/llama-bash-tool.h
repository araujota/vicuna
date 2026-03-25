#pragma once

#include "llama.h"

class llama_bash_tool {
public:
    llama_bash_tool();

    bool configure(const llama_bash_tool_config & next);
    bool get_config(llama_bash_tool_config * out_config) const;

    bool set_request(const llama_bash_tool_request & request);
    bool get_request(int32_t command_id, llama_bash_tool_request * out_request) const;
    bool clear_request(int32_t command_id);

    bool submit_result(const llama_bash_tool_result & result);
    bool get_last_result(llama_bash_tool_result * out_result) const;

private:
    llama_bash_tool_config config = {};
    llama_bash_tool_request requests[LLAMA_BASH_TOOL_MAX_PENDING_REQUESTS] = {};
    int32_t request_count = 0;
    llama_bash_tool_result last_result = {};
    bool has_last_result = false;
};
