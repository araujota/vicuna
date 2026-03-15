#include "llama-bash-tool.h"

#include "llama-context.h"

#include <algorithm>
#include <cstring>

namespace {

static void copy_bounded_cstr(char * dst, size_t dst_size, const char * src) {
    if (!dst || dst_size == 0) {
        return;
    }

    std::memset(dst, 0, dst_size);
    if (!src) {
        return;
    }

    const size_t copy_len = std::min(std::strlen(src), dst_size - 1);
    std::memcpy(dst, src, copy_len);
}

static int32_t find_request_index(
        const llama_bash_tool_request * requests,
        int32_t request_count,
        int32_t command_id) {
    for (int32_t i = 0; i < request_count; ++i) {
        if (requests[i].command_id == command_id) {
            return i;
        }
    }
    return -1;
}

} // namespace

llama_bash_tool_config llama_bash_tool_default_config(void) {
    llama_bash_tool_config config = {};
    config.enabled = false;
    config.inherit_env = false;
    config.login_shell = false;
    config.reject_shell_metacharacters = true;
    config.timeout_ms = 5000;
    config.cpu_time_limit_secs = 5;
    config.max_child_processes = 8;
    config.max_open_files = 32;
    config.max_file_size_bytes = 1 << 20;
    config.max_stdout_bytes = LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1;
    config.max_stderr_bytes = LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1;
    copy_bounded_cstr(config.bash_path, sizeof(config.bash_path), "/bin/bash");
    copy_bounded_cstr(config.allowed_commands, sizeof(config.allowed_commands), "pwd,ls,find,rg,cat,head,tail,grep,git");
    copy_bounded_cstr(config.blocked_patterns, sizeof(config.blocked_patterns), "rm -rf,:(){:|:&};:,mkfs,dd if=,chmod -R,chown -R,shutdown,reboot");
    copy_bounded_cstr(config.allowed_env, sizeof(config.allowed_env), "PATH,HOME,LANG,LC_ALL,LC_CTYPE");
    return config;
}

llama_bash_tool::llama_bash_tool() {
    config = llama_bash_tool_default_config();
}

bool llama_bash_tool::configure(const llama_bash_tool_config & next) {
    config = next;
    config.timeout_ms = std::max(100, config.timeout_ms);
    config.cpu_time_limit_secs = std::max(1, config.cpu_time_limit_secs);
    config.max_child_processes = std::max(1, config.max_child_processes);
    config.max_open_files = std::max(4, config.max_open_files);
    config.max_file_size_bytes = std::max(1024, config.max_file_size_bytes);
    config.max_stdout_bytes = std::max(1, std::min(config.max_stdout_bytes, LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1));
    config.max_stderr_bytes = std::max(1, std::min(config.max_stderr_bytes, LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1));
    config.bash_path[LLAMA_BASH_TOOL_PATH_MAX_CHARS - 1] = '\0';
    config.working_directory[LLAMA_BASH_TOOL_CWD_MAX_CHARS - 1] = '\0';
    config.allowed_commands[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
    config.blocked_patterns[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
    config.allowed_env[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
    return true;
}

bool llama_bash_tool::get_config(llama_bash_tool_config * out_config) const {
    if (!out_config) {
        return false;
    }
    *out_config = config;
    return true;
}

bool llama_bash_tool::set_request(const llama_bash_tool_request & request) {
    if (request.command_id <= 0 || request.tool_job_id <= 0) {
        return false;
    }

    const int32_t index = find_request_index(requests, request_count, request.command_id);
    if (index >= 0) {
        requests[index] = request;
        requests[index].bash_path[LLAMA_BASH_TOOL_PATH_MAX_CHARS - 1] = '\0';
        requests[index].working_directory[LLAMA_BASH_TOOL_CWD_MAX_CHARS - 1] = '\0';
        requests[index].allowed_commands[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
        requests[index].blocked_patterns[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
        requests[index].allowed_env[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
        requests[index].intent_text[LLAMA_BASH_TOOL_INTENT_MAX_CHARS - 1] = '\0';
        requests[index].command_text[LLAMA_BASH_TOOL_COMMAND_MAX_CHARS - 1] = '\0';
        return true;
    }

    if (request_count >= LLAMA_COGNITIVE_MAX_PENDING_COMMANDS) {
        return false;
    }

    requests[request_count] = request;
    requests[request_count].bash_path[LLAMA_BASH_TOOL_PATH_MAX_CHARS - 1] = '\0';
    requests[request_count].working_directory[LLAMA_BASH_TOOL_CWD_MAX_CHARS - 1] = '\0';
    requests[request_count].allowed_commands[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
    requests[request_count].blocked_patterns[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
    requests[request_count].allowed_env[LLAMA_BASH_TOOL_POLICY_MAX_CHARS - 1] = '\0';
    requests[request_count].intent_text[LLAMA_BASH_TOOL_INTENT_MAX_CHARS - 1] = '\0';
    requests[request_count].command_text[LLAMA_BASH_TOOL_COMMAND_MAX_CHARS - 1] = '\0';
    ++request_count;
    return true;
}

bool llama_bash_tool::get_request(int32_t command_id, llama_bash_tool_request * out_request) const {
    if (!out_request) {
        return false;
    }

    const int32_t index = find_request_index(requests, request_count, command_id);
    if (index < 0) {
        return false;
    }

    *out_request = requests[index];
    return true;
}

bool llama_bash_tool::clear_request(int32_t command_id) {
    const int32_t index = find_request_index(requests, request_count, command_id);
    if (index < 0) {
        return false;
    }

    for (int32_t i = index + 1; i < request_count; ++i) {
        requests[i - 1] = requests[i];
    }
    requests[request_count - 1] = {};
    --request_count;
    return true;
}

bool llama_bash_tool::submit_result(const llama_bash_tool_result & result) {
    last_result = result;
    last_result.stdout_text[LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1] = '\0';
    last_result.stderr_text[LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1] = '\0';
    last_result.error_text[LLAMA_BASH_TOOL_ERROR_MAX_CHARS - 1] = '\0';
    has_last_result = true;
    return clear_request(result.command_id);
}

bool llama_bash_tool::get_last_result(llama_bash_tool_result * out_result) const {
    if (!out_result || !has_last_result) {
        return false;
    }
    *out_result = last_result;
    return true;
}

bool llama_context::bash_tool_configure(const llama_bash_tool_config & config) {
    return bash_tool && bash_tool->configure(config);
}

bool llama_context::bash_tool_get_config(llama_bash_tool_config * out_config) const {
    return bash_tool && bash_tool->get_config(out_config);
}

bool llama_context::bash_tool_set_request(const llama_bash_tool_request & request) {
    return bash_tool && bash_tool->set_request(request);
}

bool llama_context::bash_tool_clear_request(int32_t command_id) {
    return bash_tool && bash_tool->clear_request(command_id);
}

bool llama_context::bash_tool_submit_result(const llama_bash_tool_result & result) {
    return bash_tool && bash_tool->submit_result(result);
}

bool llama_context::bash_tool_get_last_result(llama_bash_tool_result * out_result) const {
    return bash_tool && bash_tool->get_last_result(out_result);
}

bool llama_context::cognitive_bash_tool_get_request(int32_t command_id, llama_bash_tool_request * out_request) const {
    return bash_tool && bash_tool->get_request(command_id, out_request);
}

int32_t llama_bash_tool_configure(
        struct llama_context * ctx,
        const struct llama_bash_tool_config * config) {
    return ctx && config && ctx->bash_tool_configure(*config) ? 0 : -1;
}

int32_t llama_bash_tool_get_config(
        const struct llama_context * ctx,
        struct llama_bash_tool_config * out_config) {
    return ctx && out_config && ctx->bash_tool_get_config(out_config) ? 0 : -1;
}

int32_t llama_bash_tool_get_last_result(
        const struct llama_context * ctx,
        struct llama_bash_tool_result * out_result) {
    return ctx && out_result && ctx->bash_tool_get_last_result(out_result) ? 0 : -1;
}

int32_t llama_cognitive_bash_tool_get_request(
        const struct llama_context * ctx,
        int32_t command_id,
        struct llama_bash_tool_request * out_request) {
    return ctx && out_request && ctx->cognitive_bash_tool_get_request(command_id, out_request) ? 0 : -1;
}
