#pragma once

#include "llama.h"

bool server_bash_tool_execute(
        const llama_bash_tool_request & request,
        llama_bash_tool_result * out_result);
