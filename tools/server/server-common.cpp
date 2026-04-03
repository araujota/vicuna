#include "server-common.h"

#include <chrono>
#include <random>
#include <sstream>

namespace {

std::mutex & vicuna_log_mutex() {
    static std::mutex mutex;
    return mutex;
}

int & vicuna_log_level_storage() {
    static int level = VICUNA_LOG_LEVEL_INFO;
    return level;
}

FILE * vicuna_log_stream_for_level(int level) {
    return level <= VICUNA_LOG_LEVEL_WARN ? stderr : stdout;
}

} // namespace

void vicuna_log_set_verbosity(int level) {
    vicuna_log_level_storage() = level;
}

void vicuna_log_printf(int level, const char * fmt, ...) {
    if (level > vicuna_log_level_storage()) {
        return;
    }
    std::lock_guard<std::mutex> lock(vicuna_log_mutex());
    FILE * stream = vicuna_log_stream_for_level(level);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stream, fmt, ap);
    va_end(ap);
    std::fflush(stream);
}

json format_error_response(const std::string & message, enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
            type_str = "exceed_context_size_error";
            code = 400;
            break;
        case ERROR_TYPE_SERVER:
        default:
            type_str = "server_error";
            code = 500;
            break;
    }

    return {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

std::string random_string() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static constexpr char alphabet[] =
            "0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<size_t> dist(0, sizeof(alphabet) - 2);

    std::string value;
    value.reserve(24);
    for (int i = 0; i < 24; ++i) {
        value.push_back(alphabet[dist(rng)]);
    }
    return value;
}

std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

std::string safe_json_to_str(const json & data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

std::string format_oai_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & item) {
        ss << "data: " << safe_json_to_str(item) << "\n\n";
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}
