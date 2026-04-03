#pragma once

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::ordered_json;

enum vicuna_log_level {
    VICUNA_LOG_LEVEL_ERROR = 1,
    VICUNA_LOG_LEVEL_WARN  = 2,
    VICUNA_LOG_LEVEL_INFO  = 3,
    VICUNA_LOG_LEVEL_DEBUG = 4,
};

void vicuna_log_set_verbosity(int level);
void vicuna_log_printf(int level, const char * fmt, ...);

#define LOG_INF(...) vicuna_log_printf(VICUNA_LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WRN(...) vicuna_log_printf(VICUNA_LOG_LEVEL_WARN, __VA_ARGS__)
#define LOG_ERR(...) vicuna_log_printf(VICUNA_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_DBG(...) vicuna_log_printf(VICUNA_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_CNT(...) vicuna_log_printf(VICUNA_LOG_LEVEL_INFO, __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            return default_value;
        }
    }
    return default_value;
}

enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE,
    ERROR_TYPE_NOT_SUPPORTED,
    ERROR_TYPE_EXCEED_CONTEXT_SIZE,
};

json format_error_response(const std::string & message, enum error_type type);

std::string random_string();
std::string gen_chatcmplid();

std::string safe_json_to_str(const json & data);
std::string format_oai_sse(const json & data);
