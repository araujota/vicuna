#include "server-bash-tool.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
extern "C" char ** environ;
#endif

namespace {

static void copy_bounded_text(char * dst, size_t dst_size, const std::string & src) {
    if (!dst || dst_size == 0) {
        return;
    }

    std::memset(dst, 0, dst_size);
    const size_t copy_len = std::min(src.size(), dst_size - 1);
    if (copy_len > 0) {
        std::memcpy(dst, src.data(), copy_len);
    }
}

static int64_t monotonic_ms_now() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = (char) (ch - 'A' + 'a');
        }
    }
    return value;
}

static std::string trim_ascii(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace((unsigned char) value[begin])) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace((unsigned char) value[end - 1])) {
        --end;
    }
    return value.substr(begin, end - begin);
}

static std::vector<std::string> split_csv_lower(const char * text) {
    std::vector<std::string> values;
    if (!text) {
        return values;
    }

    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = lower_ascii(trim_ascii(item));
        if (!item.empty()) {
            values.push_back(item);
        }
    }
    return values;
}

static std::string first_command_token(const std::string & command_text) {
    const std::string trimmed = trim_ascii(command_text);
    if (trimmed.empty()) {
        return {};
    }

    size_t end = 0;
    while (end < trimmed.size() && !std::isspace((unsigned char) trimmed[end])) {
        ++end;
    }
    std::string token = trimmed.substr(0, end);
    const size_t slash = token.find_last_of('/');
    if (slash != std::string::npos) {
        token = token.substr(slash + 1);
    }
    return lower_ascii(token);
}

static bool command_contains_forbidden_meta(const std::string & command_text) {
    static const char * const blocked_tokens[] = {
        "&&", "||", ";", "|", ">", "<", "`", "$("
    };
    for (size_t i = 0; i < sizeof(blocked_tokens)/sizeof(blocked_tokens[0]); ++i) {
        if (command_text.find(blocked_tokens[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool validate_bash_request(
        const llama_bash_tool_request & request,
        std::string * out_error) {
    const std::string command_text = trim_ascii(request.command_text);
    if (command_text.empty()) {
        if (out_error) {
            *out_error = "bash tool request did not include a command";
        }
        return false;
    }

    const std::string lowered = lower_ascii(command_text);
    const std::vector<std::string> blocked = split_csv_lower(request.blocked_patterns);
    for (const std::string & pattern : blocked) {
        if (!pattern.empty() && lowered.find(pattern) != std::string::npos) {
            if (out_error) {
                *out_error = "bash command blocked by production policy";
            }
            return false;
        }
    }

    if (request.reject_shell_metacharacters && command_contains_forbidden_meta(command_text)) {
        if (out_error) {
            *out_error = "bash command contains disallowed shell metacharacters";
        }
        return false;
    }

    const std::vector<std::string> allowed = split_csv_lower(request.allowed_commands);
    if (!allowed.empty()) {
        const std::string head = first_command_token(command_text);
        const bool found = std::find(allowed.begin(), allowed.end(), head) != allowed.end();
        if (!found) {
            if (out_error) {
                *out_error = "bash command is not on the allowlist";
            }
            return false;
        }
    }

    return true;
}

static void init_result(const llama_bash_tool_request & request, llama_bash_tool_result * out_result) {
    *out_result = {};
    out_result->command_id = request.command_id;
    out_result->tool_job_id = request.tool_job_id;
}

#if !defined(_WIN32)

static bool set_nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return false;
    }
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static bool set_cloexec(int fd) {
    const int flags = fcntl(fd, F_GETFD, 0);
    if (flags < 0) {
        return false;
    }
    return fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

static void append_capped(
        std::string & dst,
        const char * data,
        size_t size,
        size_t cap,
        bool * out_truncated) {
    if (size == 0) {
        return;
    }

    if (dst.size() < cap) {
        const size_t remaining = cap - dst.size();
        const size_t keep = std::min(size, remaining);
        dst.append(data, keep);
        if (keep < size && out_truncated) {
            *out_truncated = true;
        }
    } else if (out_truncated) {
        *out_truncated = true;
    }
}

static bool drain_pipe_into_buffer(
        int fd,
        std::string & dst,
        size_t cap,
        bool * out_truncated,
        bool * out_open) {
    std::array<char, 1024> buffer = {};

    while (true) {
        const ssize_t n_read = read(fd, buffer.data(), buffer.size());
        if (n_read > 0) {
            append_capped(dst, buffer.data(), (size_t) n_read, cap, out_truncated);
            continue;
        }

        if (n_read == 0) {
            close(fd);
            if (out_open) {
                *out_open = false;
            }
            return true;
        }

        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return true;
        }

        return false;
    }
}

static void write_child_error(int fd, const char * stage) {
    if (fd < 0) {
        return;
    }

    char buffer[LLAMA_BASH_TOOL_ERROR_MAX_CHARS] = {};
    const int len = std::snprintf(buffer, sizeof(buffer), "%s: %s", stage, std::strerror(errno));
    if (len > 0) {
        (void) write(fd, buffer, (size_t) std::min<int>(len, sizeof(buffer) - 1));
    }
}

static void apply_rlimit(int resource, rlim_t soft_limit, rlim_t hard_limit) {
    struct rlimit limit = {};
    limit.rlim_cur = soft_limit;
    limit.rlim_max = hard_limit;
    (void) setrlimit(resource, &limit);
}

static void clear_environment_portable() {
    if (!environ) {
        return;
    }

    std::vector<std::string> keys;
    for (char ** entry = environ; *entry != nullptr; ++entry) {
        const std::string env_entry(*entry);
        const size_t equals = env_entry.find('=');
        if (equals != std::string::npos && equals > 0) {
            keys.push_back(env_entry.substr(0, equals));
        }
    }

    for (const std::string & key : keys) {
        (void) unsetenv(key.c_str());
    }
}

static void seed_allowed_env(const llama_bash_tool_request & request) {
    clear_environment_portable();
    const std::vector<std::string> allowed = split_csv_lower(request.allowed_env);
    for (const std::string & key_lower : allowed) {
        std::string lookup = key_lower;
        for (char & ch : lookup) {
            if (ch >= 'a' && ch <= 'z') {
                ch = (char) (ch - 'a' + 'A');
            }
        }
        const char * value = std::getenv(lookup.c_str());
        if (value) {
            (void) setenv(lookup.c_str(), value, 1);
        }
    }
}

#endif

} // namespace

bool server_bash_tool_execute(
        const llama_bash_tool_request & request,
        llama_bash_tool_result * out_result) {
    if (!out_result) {
        return false;
    }

    init_result(request, out_result);

    if (!request.command_ready || request.command_text[0] == '\0') {
        out_result->launch_failed = true;
        out_result->exit_code = 127;
        copy_bounded_text(out_result->error_text, sizeof(out_result->error_text), "bash tool request did not include a command");
        return true;
    }

    if (request.bash_path[0] == '\0') {
        out_result->launch_failed = true;
        out_result->exit_code = 127;
        copy_bounded_text(out_result->error_text, sizeof(out_result->error_text), "bash tool request did not include a bash path");
        return true;
    }

    std::string validation_error;
    if (!validate_bash_request(request, &validation_error)) {
        out_result->launch_failed = true;
        out_result->exit_code = 126;
        copy_bounded_text(out_result->error_text, sizeof(out_result->error_text), validation_error);
        return true;
    }

#if defined(_WIN32)
    out_result->launch_failed = true;
    out_result->exit_code = 127;
    copy_bounded_text(out_result->error_text, sizeof(out_result->error_text), "bash tool execution is not supported on this host");
    return true;
#else
    const int64_t started_ms = monotonic_ms_now();

    int stdout_pipe[2] = { -1, -1 };
    int stderr_pipe[2] = { -1, -1 };
    int error_pipe[2] = { -1, -1 };
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0 || pipe(error_pipe) != 0) {
        out_result->launch_failed = true;
        out_result->exit_code = 127;
        copy_bounded_text(out_result->error_text, sizeof(out_result->error_text), std::string("pipe setup failed: ") + std::strerror(errno));
        if (stdout_pipe[0] >= 0) close(stdout_pipe[0]);
        if (stdout_pipe[1] >= 0) close(stdout_pipe[1]);
        if (stderr_pipe[0] >= 0) close(stderr_pipe[0]);
        if (stderr_pipe[1] >= 0) close(stderr_pipe[1]);
        if (error_pipe[0] >= 0) close(error_pipe[0]);
        if (error_pipe[1] >= 0) close(error_pipe[1]);
        return true;
    }

    (void) set_cloexec(error_pipe[1]);

    const pid_t pid = fork();
    if (pid < 0) {
        out_result->launch_failed = true;
        out_result->exit_code = 127;
        copy_bounded_text(out_result->error_text, sizeof(out_result->error_text), std::string("fork failed: ") + std::strerror(errno));
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        close(error_pipe[0]);
        close(error_pipe[1]);
        return true;
    }

    if (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        close(error_pipe[0]);

        if (setpgid(0, 0) != 0) {
            write_child_error(error_pipe[1], "setpgid");
            _exit(127);
        }

        if (dup2(stdout_pipe[1], STDOUT_FILENO) < 0) {
            write_child_error(error_pipe[1], "dup2 stdout");
            _exit(127);
        }
        if (dup2(stderr_pipe[1], STDERR_FILENO) < 0) {
            write_child_error(error_pipe[1], "dup2 stderr");
            _exit(127);
        }

        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        if (request.working_directory[0] != '\0' && chdir(request.working_directory) != 0) {
            write_child_error(error_pipe[1], "chdir");
            _exit(127);
        }

        apply_rlimit(RLIMIT_CPU, (rlim_t) request.cpu_time_limit_secs, (rlim_t) request.cpu_time_limit_secs);
        apply_rlimit(RLIMIT_NOFILE, (rlim_t) request.max_open_files, (rlim_t) request.max_open_files);
#ifdef RLIMIT_NPROC
        apply_rlimit(RLIMIT_NPROC, (rlim_t) request.max_child_processes, (rlim_t) request.max_child_processes);
#endif
#ifdef RLIMIT_FSIZE
        apply_rlimit(RLIMIT_FSIZE, (rlim_t) request.max_file_size_bytes, (rlim_t) request.max_file_size_bytes);
#endif

        const char * argv[] = {
            request.bash_path,
            request.login_shell ? "-lc" : "-c",
            request.command_text,
            nullptr,
        };

        if (request.inherit_env) {
            execve(request.bash_path, const_cast<char * const *>(argv), environ);
        } else {
            seed_allowed_env(request);
            execve(request.bash_path, const_cast<char * const *>(argv), environ);
        }

        write_child_error(error_pipe[1], "execve");
        _exit(127);
    }

    close(stdout_pipe[1]);
    close(stderr_pipe[1]);
    close(error_pipe[1]);

    (void) set_nonblocking(stdout_pipe[0]);
    (void) set_nonblocking(stderr_pipe[0]);
    (void) set_nonblocking(error_pipe[0]);

    std::string stdout_text;
    std::string stderr_text;
    std::string error_text;
    stdout_text.reserve((size_t) std::max(1, request.max_stdout_bytes));
    stderr_text.reserve((size_t) std::max(1, request.max_stderr_bytes));

    bool stdout_open = true;
    bool stderr_open = true;
    bool process_exited = false;
    int status = 0;

    const int max_stdout = std::max(1, request.max_stdout_bytes);
    const int max_stderr = std::max(1, request.max_stderr_bytes);
    const int timeout_ms = std::max(100, request.timeout_ms);

    while (stdout_open || stderr_open || !process_exited) {
        if (!process_exited) {
            const pid_t wait_rc = waitpid(pid, &status, WNOHANG);
            if (wait_rc == pid) {
                process_exited = true;
            }
        }

        const int64_t now_ms = monotonic_ms_now();
        if (!process_exited && now_ms - started_ms >= timeout_ms) {
            out_result->timed_out = true;
            (void) killpg(pid, SIGKILL);
            (void) kill(pid, SIGKILL);
            (void) waitpid(pid, &status, 0);
            process_exited = true;
        }

        std::array<pollfd, 2> poll_fds = {};
        nfds_t nfds = 0;
        if (stdout_open) {
            poll_fds[nfds++] = { stdout_pipe[0], POLLIN | POLLHUP, 0 };
        }
        if (stderr_open) {
            poll_fds[nfds++] = { stderr_pipe[0], POLLIN | POLLHUP, 0 };
        }

        const int poll_timeout_ms = process_exited ? 0 : 25;
        if (nfds > 0) {
            (void) poll(poll_fds.data(), nfds, poll_timeout_ms);
        }

        if (stdout_open && !drain_pipe_into_buffer(stdout_pipe[0], stdout_text, (size_t) max_stdout, &out_result->truncated_stdout, &stdout_open)) {
            out_result->launch_failed = true;
            error_text = std::string("failed to read bash stdout: ") + std::strerror(errno);
            break;
        }
        if (stderr_open && !drain_pipe_into_buffer(stderr_pipe[0], stderr_text, (size_t) max_stderr, &out_result->truncated_stderr, &stderr_open)) {
            out_result->launch_failed = true;
            error_text = std::string("failed to read bash stderr: ") + std::strerror(errno);
            break;
        }
    }

    if (stdout_open) {
        close(stdout_pipe[0]);
    }
    if (stderr_open) {
        close(stderr_pipe[0]);
    }

    {
        std::array<char, LLAMA_BASH_TOOL_ERROR_MAX_CHARS> buffer = {};
        const ssize_t n_read = read(error_pipe[0], buffer.data(), buffer.size() - 1);
        if (n_read > 0) {
            error_text.assign(buffer.data(), (size_t) n_read);
            out_result->launch_failed = true;
        }
        close(error_pipe[0]);
    }

    out_result->runtime_ms = (int32_t) std::max<int64_t>(0, monotonic_ms_now() - started_ms);

    if (!out_result->launch_failed) {
        if (WIFEXITED(status)) {
            out_result->exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            out_result->term_signal = WTERMSIG(status);
            out_result->exit_code = 128 + out_result->term_signal;
        }
    } else if (out_result->exit_code == 0) {
        out_result->exit_code = 127;
    }

    if (out_result->timed_out && error_text.empty()) {
        error_text = "bash command timed out";
    }

    copy_bounded_text(out_result->stdout_text, sizeof(out_result->stdout_text), stdout_text);
    copy_bounded_text(out_result->stderr_text, sizeof(out_result->stderr_text), stderr_text);
    copy_bounded_text(out_result->error_text, sizeof(out_result->error_text), error_text);
    return true;
#endif
}
