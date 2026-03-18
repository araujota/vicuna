#include "server-openclaw-fabric.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
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

void set_error(std::string * out_error, const std::string & value) {
    if (out_error) {
        *out_error = value;
    }
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

    if (!configured_enabled) {
        return true;
    }

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
        capability_state.push_back(make_runtime(
                registration.build_descriptor(),
                registration.backend));
    }

    catalog_state.catalog_version = 1;
    for (const auto & capability : capability_state) {
        catalog_state.capabilities.push_back(capability.descriptor);
    }

    std::string validation_error;
    if (!openclaw_tool_capability_catalog_validate(catalog_state, &validation_error)) {
        configured_enabled = false;
        set_error(out_error, validation_error);
        return false;
    }

    if (capability_state.empty()) {
        configured_enabled = false;
        set_error(out_error, "OpenClaw tool fabric enabled without any routable capabilities");
        return false;
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
