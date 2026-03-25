#include "server-runtime.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

static bool parse_int32(const std::string & value, int32_t * out_value) {
    if (!out_value || value.empty()) {
        return false;
    }

    char * end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }

    *out_value = (int32_t) parsed;
    return true;
}

static bool parse_bool_string(const std::string & value, bool * out_value) {
    if (!out_value) {
        return false;
    }
    if (value == "1" || value == "true" || value == "TRUE" || value == "on") {
        *out_value = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "off") {
        *out_value = false;
        return true;
    }
    return false;
}

static bool parse_api_surface_string(const std::string & value, server_api_surface * out_surface) {
    if (!out_surface) {
        return false;
    }
    if (value == "default") {
        *out_surface = SERVER_API_SURFACE_DEFAULT;
        return true;
    }
    if (value == "openai") {
        *out_surface = SERVER_API_SURFACE_OPENAI;
        return true;
    }
    return false;
}

static bool option_requires_value(const std::string & arg) {
    static const std::set<std::string> valued = {
        "--host",
        "--port",
        "--api-prefix",
        "--api-key",
        "--api-surface",
        "--public-path",
        "--ssl-file-key",
        "--ssl-file-cert",
        "--timeout-read",
        "--timeout-write",
        "--threads-http",
        "--parallel",
        "--threads",
        "--threads-batch",
        "--temp",
        "--seed",
        "--model",
        "--model-url",
        "--model-draft",
        "--hf-repo",
        "--hf-file",
        "--models-dir",
        "--models-max",
        "--batch-size",
        "--ubatch-size",
        "--n-gpu-layers",
        "--draft",
        "--pooling",
        "--alias",
        "--tags",
        "--ctx-size",
        "-ctk",
        "-ctv",
        "-fa",
        "--n-predict",
        "--slot-save-path",
        "--grp-attn-n",
        "--grp-attn-w",
        "--draft-max",
        "--draft-min",
        "--reasoning-format",
        "--reasoning-budget",
        "--chat-template",
        "--chat-template-file",
        "--mmproj-url",
        "--media-path",
        "--sleep-idle-seconds",
    };
    return valued.find(arg) != valued.end();
}

static bool option_is_ignored_flag(const std::string & arg) {
    static const std::set<std::string> ignored = {
        "--offline",
        "--cont-batching",
        "--embedding",
        "--reranking",
        "--metrics",
        "--kv-unified",
        "--slots",
        "--no-slots",
        "--context-shift",
        "--jinja",
        "--no-jinja",
        "--webui-mcp-proxy",
    };
    return ignored.find(arg) != ignored.end();
}

static const char * server_api_surface_to_string(server_api_surface surface) {
    switch (surface) {
        case SERVER_API_SURFACE_OPENAI: return "openai";
        case SERVER_API_SURFACE_DEFAULT:
        default:                        return "default";
    }
}

} // namespace

bool server_runtime_params_parse(
        int argc,
        char ** argv,
        server_runtime_params & params,
        bool * out_show_help,
        std::string * out_error) {
    if (out_show_help) {
        *out_show_help = false;
    }

    auto fail = [&](const std::string & message) {
        if (out_error) {
            *out_error = message;
        }
        return false;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            if (out_show_help) {
                *out_show_help = true;
            }
            return true;
        }
        if (arg == "--version") {
            std::fprintf(stdout, "%s\n", server_runtime_build_info().c_str());
            return false;
        }
        if (arg == "--no-webui") {
            params.webui = false;
            continue;
        }
        if (arg == "--webui") {
            params.webui = true;
            continue;
        }
        if (arg == "--verbose") {
            params.verbose = true;
            continue;
        }
        if (option_is_ignored_flag(arg)) {
            continue;
        }

        if (!option_requires_value(arg)) {
            return fail("unsupported argument: " + arg);
        }
        if (i + 1 >= argc) {
            return fail("missing value for argument: " + arg);
        }

        const std::string value = argv[++i];

        if (arg == "--host") {
            params.hostname = value;
            continue;
        }
        if (arg == "--port") {
            if (!parse_int32(value, &params.port) || params.port < 0) {
                return fail("invalid port: " + value);
            }
            continue;
        }
        if (arg == "--api-prefix") {
            params.api_prefix = value;
            continue;
        }
        if (arg == "--api-key") {
            params.api_keys.push_back(value);
            continue;
        }
        if (arg == "--api-surface") {
            if (!parse_api_surface_string(value, &params.api_surface)) {
                return fail("invalid api surface: " + value);
            }
            continue;
        }
        if (arg == "--public-path") {
            params.public_path = value;
            continue;
        }
        if (arg == "--ssl-file-key") {
            params.ssl_file_key = value;
            continue;
        }
        if (arg == "--ssl-file-cert") {
            params.ssl_file_cert = value;
            continue;
        }
        if (arg == "--timeout-read") {
            if (!parse_int32(value, &params.timeout_read) || params.timeout_read <= 0) {
                return fail("invalid read timeout: " + value);
            }
            continue;
        }
        if (arg == "--timeout-write") {
            if (!parse_int32(value, &params.timeout_write) || params.timeout_write <= 0) {
                return fail("invalid write timeout: " + value);
            }
            continue;
        }
        if (arg == "--threads-http") {
            if (!parse_int32(value, &params.n_threads_http)) {
                return fail("invalid HTTP thread count: " + value);
            }
            continue;
        }
        if (arg == "--parallel") {
            if (!parse_int32(value, &params.n_parallel)) {
                return fail("invalid parallel count: " + value);
            }
            continue;
        }
        if (arg == "--threads") {
            if (!parse_int32(value, &params.n_threads)) {
                return fail("invalid thread count: " + value);
            }
            continue;
        }
        if (arg == "--threads-batch") {
            if (!parse_int32(value, &params.n_threads_batch)) {
                return fail("invalid batch thread count: " + value);
            }
            continue;
        }

        // Provider-first compatibility: accept and ignore legacy local-runtime flags.
        if (option_requires_value(arg)) {
            continue;
        }
    }

    if (params.n_parallel < 1) {
        params.n_parallel = 4;
    }
    if (params.timeout_read <= 0) {
        params.timeout_read = 600;
    }
    if (params.timeout_write <= 0) {
        params.timeout_write = 600;
    }

    return true;
}

void server_runtime_print_usage(const char * prog_name) {
    std::fprintf(stdout,
            "Usage: %s [options]\n"
            "  --host <addr>             Bind address or unix socket path\n"
            "  --port <port>             Listen port, 0 for ephemeral\n"
            "  --api-surface <name>      default | openai\n"
            "  --api-key <token>         Require bearer/API key\n"
            "  --api-prefix <prefix>     HTTP route prefix\n"
            "  --no-webui                Disable embedded Web UI\n"
            "  --public-path <dir>       Serve static assets from a directory\n"
            "  --ssl-file-key <path>     TLS private key\n"
            "  --ssl-file-cert <path>    TLS certificate\n"
            "  --timeout-read <sec>      HTTP read timeout\n"
            "  --timeout-write <sec>     HTTP write timeout\n"
            "  --threads-http <n>        HTTP worker thread count\n"
            "  --parallel <n>            Parallel request budget for thread sizing\n"
            "  --threads <n>             Reported CPU thread count\n"
            "  --verbose                 Enable verbose logging\n",
            prog_name ? prog_name : "llama-server");
}

void server_runtime_init(bool verbose) {
    common_log_set_verbosity_thold(verbose ? LOG_LEVEL_DEBUG : LOG_LEVEL_INFO);
    llama_log_set(common_log_default_callback, nullptr);
    LOG_INF("build: %s\n", server_runtime_build_info().c_str());
}

std::string server_runtime_system_info(const server_runtime_params & params) {
    std::ostringstream os;
    os << "system_info: n_threads = " << params.n_threads;
    if (params.n_threads_batch >= 0) {
        os << " (n_threads_batch = " << params.n_threads_batch << ")";
    }
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
    os << " | api_surface = " << server_api_surface_to_string(params.api_surface);
    return os.str();
}

const std::string & server_runtime_build_info() {
    static const std::string build =
#ifdef VICUNA_SERVER_BUILD_INFO
            VICUNA_SERVER_BUILD_INFO;
#else
            "provider-runtime";
#endif
    return build;
}

server_http_url server_http_parse_url(const std::string & url) {
    server_http_url parts;
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw std::runtime_error("invalid URL: no scheme");
    }

    parts.scheme = url.substr(0, scheme_end);
    if (parts.scheme != "http" && parts.scheme != "https") {
        throw std::runtime_error("unsupported URL scheme: " + parts.scheme);
    }

    std::string rest = url.substr(scheme_end + 3);
    const size_t at_pos = rest.find('@');
    if (at_pos != std::string::npos) {
        const std::string auth = rest.substr(0, at_pos);
        const size_t colon_pos = auth.find(':');
        if (colon_pos != std::string::npos) {
            parts.user = auth.substr(0, colon_pos);
            parts.password = auth.substr(colon_pos + 1);
        } else {
            parts.user = auth;
        }
        rest = rest.substr(at_pos + 1);
    }

    const size_t slash_pos = rest.find('/');
    if (slash_pos == std::string::npos) {
        parts.host = rest;
    } else {
        parts.host = rest.substr(0, slash_pos);
        parts.path = rest.substr(slash_pos);
    }

    const size_t colon_pos = parts.host.rfind(':');
    if (colon_pos != std::string::npos) {
        int32_t parsed_port = 0;
        if (!parse_int32(parts.host.substr(colon_pos + 1), &parsed_port) || parsed_port <= 0) {
            throw std::runtime_error("invalid URL port");
        }
        parts.port = parsed_port;
        parts.host = parts.host.substr(0, colon_pos);
    } else {
        parts.port = parts.scheme == "https" ? 443 : 80;
    }

    if (parts.host.empty()) {
        throw std::runtime_error("invalid URL: missing host");
    }

    return parts;
}

std::pair<httplib::Client, server_http_url> server_http_client(const std::string & url) {
    server_http_url parts = server_http_parse_url(url);

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (parts.scheme == "https") {
        throw std::runtime_error("HTTPS is not supported by this build");
    }
#endif

    httplib::Client client(parts.scheme + "://" + parts.host + ":" + std::to_string(parts.port));
    if (!parts.user.empty()) {
        client.set_basic_auth(parts.user, parts.password);
    }
    client.set_follow_location(true);
    return {std::move(client), std::move(parts)};
}

std::string server_http_show_masked_url(const server_http_url & parts) {
    return parts.scheme + "://" + (parts.user.empty() ? "" : "****:****@") + parts.host + parts.path;
}

std::string server_string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap_copy;
    va_start(ap, fmt);
    va_copy(ap_copy, ap);
    const int size = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    GGML_ASSERT(size >= 0);
    std::vector<char> buffer((size_t) size + 1);
    const int written = std::vsnprintf(buffer.data(), buffer.size(), fmt, ap_copy);
    va_end(ap_copy);
    GGML_ASSERT(written == size);
    return std::string(buffer.data(), (size_t) size);
}

std::vector<std::string> server_string_split(const std::string & str, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        out.push_back(item);
    }
    if (!str.empty() && str.back() == delim) {
        out.emplace_back();
    }
    return out;
}

bool server_string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix;
}

void server_batch_add(
        llama_batch & batch,
        llama_token id,
        llama_pos pos,
        const std::vector<llama_seq_id> & seq_ids,
        bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

std::vector<llama_token> server_tokenize(
        const llama_context * ctx,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int32_t n_tokens = (int32_t) text.size() + (add_special ? 2 : 0);
    std::vector<llama_token> result((size_t) std::max(1, n_tokens));
    n_tokens = llama_tokenize(vocab, text.data(), (int32_t) text.size(), result.data(), (int32_t) result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenization failed: text too large");
    }
    if (n_tokens < 0) {
        result.resize((size_t) -n_tokens);
        const int32_t check = llama_tokenize(vocab, text.data(), (int32_t) text.size(), result.data(), (int32_t) result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize((size_t) n_tokens);
    }
    return result;
}

void server_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1:
            sum = 1.0;
            break;
        case 0:
            for (int i = 0; i < n; ++i) {
                sum = std::max(sum, (double) std::abs(inp[i]));
            }
            sum /= 32760.0;
            break;
        case 2:
            for (int i = 0; i < n; ++i) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default:
            for (int i = 0; i < n; ++i) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = inp[i] * norm;
    }
}

float server_embd_similarity_cos(const float * embd1, const float * embd2, int n) {
    double sum = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; ++i) {
        sum += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    if (sum1 == 0.0 || sum2 == 0.0) {
        return (sum1 == 0.0 && sum2 == 0.0) ? 1.0f : 0.0f;
    }

    return (float) (sum / (std::sqrt(sum1) * std::sqrt(sum2)));
}
