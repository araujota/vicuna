#include "server-openclaw-fabric.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {

struct builtin_capability_registration {
    const char * tool_id;
    server_openclaw_dispatch_backend backend;
    bool (*is_available)(bool bash_enabled, bool hard_memory_enabled);
    openclaw_tool_capability_descriptor (*build_descriptor)();
};

bool env_flag_enabled(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }
    const std::string normalized = [&]() {
        std::string out(value);
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
            return (char) std::tolower(ch);
        });
        return out;
    }();
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool env_list_contains(const char * name, const char * needle, bool default_value) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());
        std::transform(item.begin(), item.end(), item.begin(), [](unsigned char ch) {
            return (char) std::tolower(ch);
        });
        if (item == needle) {
            return true;
        }
    }
    return false;
}

void set_bounded(char * dst, size_t dst_size, const std::string & src) {
    if (!dst || dst_size == 0) {
        return;
    }
    std::memset(dst, 0, dst_size);
    const size_t copy_len = std::min(dst_size - 1, src.size());
    std::memcpy(dst, src.data(), copy_len);
    dst[copy_len] = '\0';
}

void set_error(std::string * out_error, const std::string & value) {
    if (out_error) {
        *out_error = value;
    }
}

server_openclaw_capability_runtime make_runtime(
        const openclaw_tool_capability_descriptor & descriptor,
        server_openclaw_dispatch_backend backend) {
    server_openclaw_capability_runtime runtime;
    runtime.descriptor = descriptor;
    runtime.backend = backend;
    runtime.tool_spec.tool_kind = descriptor.tool_kind;
    runtime.tool_spec.flags = descriptor.tool_flags;
    runtime.tool_spec.latency_class = descriptor.latency_class;
    runtime.tool_spec.max_steps_reserved = descriptor.max_steps_reserved;
    set_bounded(runtime.tool_spec.name, sizeof(runtime.tool_spec.name), descriptor.tool_name);
    set_bounded(runtime.tool_spec.description, sizeof(runtime.tool_spec.description), descriptor.description);
    set_bounded(runtime.tool_spec.capability_id, sizeof(runtime.tool_spec.capability_id), descriptor.capability_id);
    set_bounded(runtime.tool_spec.owner_plugin_id, sizeof(runtime.tool_spec.owner_plugin_id), descriptor.owner_plugin_id);
    set_bounded(runtime.tool_spec.provenance_namespace, sizeof(runtime.tool_spec.provenance_namespace), descriptor.provenance_namespace);
    return runtime;
}

bool dispatch_backend_from_string(
        const std::string & value,
        server_openclaw_dispatch_backend * out_backend,
        std::string * out_error) {
    if (value == "legacy_bash") {
        if (out_backend) {
            *out_backend = SERVER_OPENCLAW_DISPATCH_LEGACY_BASH;
        }
        return true;
    }
    if (value == "legacy_hard_memory") {
        if (out_backend) {
            *out_backend = SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY;
        }
        return true;
    }
    set_error(out_error, "unsupported dispatch_backend: " + value);
    return false;
}

bool exec_available(bool bash_enabled, bool /*hard_memory_enabled*/) {
    return bash_enabled;
}

bool hard_memory_available(bool /*bash_enabled*/, bool hard_memory_enabled) {
    return hard_memory_enabled;
}

openclaw_tool_capability_descriptor build_exec_descriptor() {
    openclaw_tool_capability_descriptor exec = {};
    exec.capability_id = "openclaw.exec.command";
    exec.tool_surface_id = "vicuna.exec.main";
    exec.capability_kind = "tool";
    exec.owner_plugin_id = "openclaw-core";
    exec.tool_name = "exec";
    exec.description = "Run a command through the bounded execution policy";
    exec.input_schema_json = R"({"type":"object","required":["command"],"properties":{"command":{"type":"string"},"workdir":{"type":"string"}}})";
    exec.output_contract = "pending_then_result";
    exec.side_effect_class = "system_exec";
    exec.approval_mode = "policy_driven";
    exec.execution_modes = {"sync", "background", "approval_gated"};
    exec.provenance_namespace = "openclaw/openclaw-core/tool/exec";
    exec.tool_kind = LLAMA_TOOL_KIND_BASH_CLI;
    exec.tool_flags =
            LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
            LLAMA_COG_TOOL_DMN_ELIGIBLE |
            LLAMA_COG_TOOL_REMEDIATION_SAFE |
            LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT;
    exec.latency_class = LLAMA_COG_TOOL_LATENCY_MEDIUM;
    exec.max_steps_reserved = 2;
    exec.dispatch_backend = "legacy_bash";
    return exec;
}

openclaw_tool_capability_descriptor build_hard_memory_descriptor() {
    openclaw_tool_capability_descriptor memory = {};
    memory.capability_id = "openclaw.vicuna.hard_memory_query";
    memory.tool_surface_id = "vicuna.memory.hard_query";
    memory.capability_kind = "memory_adapter";
    memory.owner_plugin_id = "vicuna-memory";
    memory.tool_name = "hard_memory_query";
    memory.description = "Query Vicuña hard memory and return typed retrieval results";
    memory.input_schema_json = R"({"type":"object","required":["query"],"properties":{"query":{"type":"string"}}})";
    memory.output_contract = "completed_result";
    memory.side_effect_class = "memory_read";
    memory.approval_mode = "none";
    memory.execution_modes = {"sync"};
    memory.provenance_namespace = "openclaw/vicuna-memory/memory_adapter/hard_memory_query";
    memory.tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_QUERY;
    memory.tool_flags =
            LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
            LLAMA_COG_TOOL_DMN_ELIGIBLE |
            LLAMA_COG_TOOL_SIMULATION_SAFE |
            LLAMA_COG_TOOL_REMEDIATION_SAFE;
    memory.latency_class = LLAMA_COG_TOOL_LATENCY_MEDIUM;
    memory.max_steps_reserved = 2;
    memory.dispatch_backend = "legacy_hard_memory";
    return memory;
}

const builtin_capability_registration * builtin_capability_registrations(size_t * out_count) {
    static const builtin_capability_registration registrations[] = {
        {
            "exec",
            SERVER_OPENCLAW_DISPATCH_LEGACY_BASH,
            exec_available,
            build_exec_descriptor,
        },
        {
            "hard_memory_query",
            SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY,
            hard_memory_available,
            build_hard_memory_descriptor,
        },
    };
    if (out_count) {
        *out_count = sizeof(registrations) / sizeof(registrations[0]);
    }
    return registrations;
}

}  // namespace

bool server_openclaw_fabric::configure(bool bash_enabled, bool hard_memory_enabled, std::string * out_error) {
    configured_enabled = env_flag_enabled("VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED", false);
    catalog_state = {};
    capability_state.clear();
    external_catalog_path.clear();
    catalog_source_signature.clear();

    if (!configured_enabled) {
        return true;
    }

    const char * catalog_path_env = std::getenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH");
    if (catalog_path_env && catalog_path_env[0] != '\0') {
        external_catalog_path = catalog_path_env;
    }

    openclaw_tool_capability_catalog next_catalog = {};
    std::vector<server_openclaw_capability_runtime> next_capabilities;
    if (!rebuild_catalog(
                bash_enabled,
                hard_memory_enabled,
                &next_catalog,
                &next_capabilities,
                out_error)) {
        configured_enabled = false;
        return false;
    }

    if (next_capabilities.empty()) {
        configured_enabled = false;
        set_error(out_error, "OpenClaw tool fabric enabled without any routable capabilities");
        return false;
    }
    catalog_state = std::move(next_catalog);
    capability_state = std::move(next_capabilities);
    catalog_source_signature = current_external_catalog_signature();
    return true;
}

bool server_openclaw_fabric::maybe_reload(
        bool bash_enabled,
        bool hard_memory_enabled,
        bool * out_reloaded,
        std::string * out_error) {
    if (out_reloaded) {
        *out_reloaded = false;
    }
    if (!configured_enabled || external_catalog_path.empty()) {
        return true;
    }

    const std::string next_signature = current_external_catalog_signature();
    if (next_signature == catalog_source_signature) {
        return true;
    }

    openclaw_tool_capability_catalog next_catalog = {};
    std::vector<server_openclaw_capability_runtime> next_capabilities;
    if (!rebuild_catalog(
                bash_enabled,
                hard_memory_enabled,
                &next_catalog,
                &next_capabilities,
                out_error)) {
        catalog_source_signature = next_signature;
        return false;
    }

    catalog_state = std::move(next_catalog);
    capability_state = std::move(next_capabilities);
    catalog_source_signature = next_signature;
    if (out_reloaded) {
        *out_reloaded = true;
    }
    return true;
}

bool server_openclaw_fabric::enabled() const {
    return configured_enabled;
}

const openclaw_tool_capability_catalog & server_openclaw_fabric::catalog() const {
    return catalog_state;
}

const std::vector<server_openclaw_capability_runtime> & server_openclaw_fabric::capabilities() const {
    return capability_state;
}

bool server_openclaw_fabric::build_cognitive_specs(std::vector<llama_cognitive_tool_spec> * out_specs) const {
    if (!out_specs) {
        return false;
    }
    out_specs->clear();
    out_specs->reserve(capability_state.size());
    for (const auto & capability : capability_state) {
        out_specs->push_back(capability.tool_spec);
    }
    return true;
}

bool server_openclaw_fabric::build_chat_tools(
        std::vector<common_chat_tool> * out_tools,
        const std::vector<int32_t> * spec_indexes) const {
    if (!out_tools) {
        return false;
    }
    out_tools->clear();

    auto append_tool = [&](const server_openclaw_capability_runtime & capability) {
        common_chat_tool tool;
        tool.name = capability.descriptor.tool_name;
        tool.description = capability.descriptor.description;
        tool.parameters = capability.descriptor.input_schema_json;
        out_tools->push_back(std::move(tool));
    };

    if (!spec_indexes || spec_indexes->empty()) {
        out_tools->reserve(capability_state.size());
        for (const auto & capability : capability_state) {
            append_tool(capability);
        }
        return true;
    }

    out_tools->reserve(spec_indexes->size());
    for (int32_t spec_index : *spec_indexes) {
        const server_openclaw_capability_runtime * capability = capability_by_spec_index(spec_index);
        if (!capability) {
            continue;
        }
        append_tool(*capability);
    }
    return true;
}

const server_openclaw_capability_runtime * server_openclaw_fabric::capability_by_spec_index(int32_t spec_index) const {
    if (spec_index < 0 || spec_index >= (int32_t) capability_state.size()) {
        return nullptr;
    }
    return &capability_state[(size_t) spec_index];
}

const server_openclaw_capability_runtime * server_openclaw_fabric::capability_by_tool_name(const std::string & tool_name) const {
    if (tool_name.empty()) {
        return nullptr;
    }
    for (const auto & capability : capability_state) {
        if (capability.descriptor.tool_name == tool_name) {
            return &capability;
        }
    }
    return nullptr;
}

bool server_openclaw_fabric::rebuild_catalog(
        bool bash_enabled,
        bool hard_memory_enabled,
        openclaw_tool_capability_catalog * out_catalog,
        std::vector<server_openclaw_capability_runtime> * out_capabilities,
        std::string * out_error) const {
    if (!out_catalog || !out_capabilities) {
        set_error(out_error, "catalog rebuild requires output storage");
        return false;
    }

    out_catalog->catalog_version = 1;
    out_catalog->capabilities.clear();
    out_capabilities->clear();

    size_t registration_count = 0;
    const builtin_capability_registration * registrations =
            builtin_capability_registrations(&registration_count);
    for (size_t i = 0; i < registration_count; ++i) {
        const auto & registration = registrations[i];
        if (!env_list_contains("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS", registration.tool_id, true)) {
            continue;
        }
        if (!registration.is_available(bash_enabled, hard_memory_enabled)) {
            continue;
        }
        out_capabilities->push_back(make_runtime(
                registration.build_descriptor(),
                registration.backend));
    }

    if (!external_catalog_path.empty() &&
            std::filesystem::exists(external_catalog_path)) {
        if (!load_external_catalog(
                    external_catalog_path,
                    bash_enabled,
                    hard_memory_enabled,
                    out_capabilities,
                    out_error)) {
            return false;
        }
    }

    for (const auto & capability : *out_capabilities) {
        out_catalog->capabilities.push_back(capability.descriptor);
    }

    std::string validation_error;
    if (!openclaw_tool_capability_catalog_validate(*out_catalog, &validation_error)) {
        set_error(out_error, validation_error);
        return false;
    }
    return true;
}

bool server_openclaw_fabric::load_external_catalog(
        const std::string & path,
        bool bash_enabled,
        bool hard_memory_enabled,
        std::vector<server_openclaw_capability_runtime> * out_capabilities,
        std::string * out_error) const {
    if (!out_capabilities) {
        set_error(out_error, "external catalog load requires capability storage");
        return false;
    }

    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open external catalog");
        }
        nlohmann::json data;
        in >> data;

        openclaw_tool_capability_catalog external_catalog = {};
        std::string parse_error;
        if (!openclaw_tool_capability_catalog_from_json(data, &external_catalog, &parse_error)) {
            set_error(out_error, parse_error);
            return false;
        }

        for (const auto & descriptor : external_catalog.capabilities) {
            server_openclaw_dispatch_backend backend = SERVER_OPENCLAW_DISPATCH_NONE;
            std::string backend_error;
            if (!dispatch_backend_from_string(descriptor.dispatch_backend, &backend, &backend_error)) {
                set_error(out_error, backend_error);
                return false;
            }
            if (backend == SERVER_OPENCLAW_DISPATCH_LEGACY_BASH && !bash_enabled) {
                continue;
            }
            if (backend == SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY && !hard_memory_enabled) {
                continue;
            }
            out_capabilities->push_back(make_runtime(descriptor, backend));
        }
    } catch (const std::exception & err) {
        set_error(out_error, err.what());
        return false;
    }

    return true;
}

std::string server_openclaw_fabric::current_external_catalog_signature() const {
    if (external_catalog_path.empty()) {
        return "builtin-only";
    }
    if (!std::filesystem::exists(external_catalog_path)) {
        return external_catalog_path + "|missing";
    }
    const auto write_time = std::filesystem::last_write_time(external_catalog_path);
    const auto ticks = write_time.time_since_epoch().count();
    return external_catalog_path + "|" + std::to_string((long long) ticks);
}

const server_openclaw_capability_runtime * server_openclaw_fabric::resolve_command(
        const llama_cognitive_command & command,
        std::string * out_error) const {
    if (!configured_enabled) {
        set_error(out_error, "fabric is disabled");
        return nullptr;
    }
    if (command.tool_spec_index < 0 ||
            command.tool_spec_index >= (int32_t) capability_state.size()) {
        set_error(out_error, "tool_spec_index is out of range");
        return nullptr;
    }

    const auto & capability = capability_state[(size_t) command.tool_spec_index];
    if (capability.tool_spec.tool_kind != command.tool_kind) {
        set_error(out_error, "tool_kind does not match registered capability");
        return nullptr;
    }
    if (command.capability_id[0] == '\0') {
        set_error(out_error, "command is missing capability_id");
        return nullptr;
    }
    if (capability.descriptor.capability_id != std::string(command.capability_id)) {
        set_error(out_error, "capability_id does not match registered capability");
        return nullptr;
    }
    if (capability.backend == SERVER_OPENCLAW_DISPATCH_NONE) {
        set_error(out_error, "registered capability has no executor backend");
        return nullptr;
    }
    return &capability;
}
