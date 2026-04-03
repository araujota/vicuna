#pragma once

#include "server-common.h"

#include <cpp-httplib/httplib.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

enum server_api_surface {
    SERVER_API_SURFACE_DEFAULT = 0,
    SERVER_API_SURFACE_OPENAI  = 1,
};

struct server_runtime_params {
    std::string hostname = "127.0.0.1";
    int32_t port = 8080;
    std::string api_prefix;
    server_api_surface api_surface = SERVER_API_SURFACE_DEFAULT;
    bool webui = false;
    std::string public_path;
    std::string ssl_file_key;
    std::string ssl_file_cert;
    int32_t timeout_read = 600;
    int32_t timeout_write = 600;
    std::vector<std::string> api_keys;
    int32_t n_threads_http = -1;
    int32_t n_parallel = 4;
    int32_t n_threads = -1;
    int32_t n_threads_batch = -1;
    bool verbose = false;
};

struct server_http_url {
    std::string scheme;
    std::string user;
    std::string password;
    std::string host;
    int port = 0;
    std::string path = "/";
};

bool server_runtime_params_parse(
        int argc,
        char ** argv,
        server_runtime_params & params,
        bool * out_show_help = nullptr,
        std::string * out_error = nullptr);
void server_runtime_print_usage(const char * prog_name);
void server_runtime_init(bool verbose);
std::string server_runtime_system_info(const server_runtime_params & params);
const std::string & server_runtime_build_info();

server_http_url server_http_parse_url(const std::string & url);
std::pair<httplib::Client, server_http_url> server_http_client(const std::string & url);
std::string server_http_show_masked_url(const server_http_url & parts);

std::string server_string_format(const char * fmt, ...);
std::vector<std::string> server_string_split(const std::string & str, char delim);
bool server_string_ends_with(std::string_view str, std::string_view suffix);
float server_embd_similarity_cos(const float * a, const float * b, int n);
