#include "common/openclaw-session-binding.h"
#include "common/openclaw-tool-fabric-events.h"
#include "common/openclaw-tool-fabric.h"
#include "../tools/server/server-openclaw-fabric.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <vector>

static bool expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        return false;
    }
    return true;
}

int main() {
    openclaw_tool_capability_descriptor descriptor = {};
    descriptor.capability_id = "openclaw.exec.command";
    descriptor.tool_surface_id = "vicuna.exec.main";
    descriptor.capability_kind = "tool";
    descriptor.owner_plugin_id = "openclaw-core";
    descriptor.tool_name = "exec";
    descriptor.description = "Run a command";
    descriptor.input_schema_json = R"({"type":"object","properties":{"command":{"type":"string"}}})";
    descriptor.output_contract = "pending_then_result";
    descriptor.side_effect_class = "system_exec";
    descriptor.approval_mode = "policy_driven";
    descriptor.execution_modes = {"sync", "approval_gated"};
    descriptor.provenance_namespace = "openclaw/openclaw-core/tool/exec";
    descriptor.tool_kind = LLAMA_TOOL_KIND_BASH_CLI;
    descriptor.tool_flags = LLAMA_COG_TOOL_ACTIVE_ELIGIBLE | LLAMA_COG_TOOL_DMN_ELIGIBLE;
    descriptor.latency_class = LLAMA_COG_TOOL_LATENCY_MEDIUM;
    descriptor.max_steps_reserved = 2;
    descriptor.dispatch_backend = "legacy_bash";

    std::string error;
    if (!expect(openclaw_tool_capability_descriptor_validate(descriptor, &error), error.c_str())) {
        return 1;
    }

    const auto descriptor_json = openclaw_tool_capability_descriptor_to_json(descriptor);
    openclaw_tool_capability_descriptor parsed_descriptor = {};
    if (!expect(openclaw_tool_capability_descriptor_from_json(descriptor_json, &parsed_descriptor, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed_descriptor.capability_id == descriptor.capability_id, "descriptor round-trip lost capability_id")) {
        return 1;
    }
    if (!expect(parsed_descriptor.tool_surface_id == descriptor.tool_surface_id, "descriptor round-trip lost tool_surface_id")) {
        return 1;
    }

    openclaw_tool_invocation invocation = {};
    invocation.invocation_id = "ocinv_1";
    invocation.capability_id = descriptor.capability_id;
    invocation.origin_phase = "active";
    invocation.arguments_json = R"({"command":"pwd"})";
    if (!expect(openclaw_tool_invocation_validate(invocation, &error), error.c_str())) {
        return 1;
    }

    openclaw_tool_observation observation = {};
    observation.invocation_id = invocation.invocation_id;
    observation.status = OPENCLAW_TOOL_OBSERVATION_COMPLETED;
    observation.summary_text = "completed";
    observation.structured_payload_json = R"({"stdout":"/workspace\n","exit_code":0})";
    if (!expect(openclaw_tool_observation_validate(observation, &error), error.c_str())) {
        return 1;
    }

    openclaw_session_binding binding = {};
    binding.vicuna_session_id = "sess_1";
    binding.openclaw_binding_id = "bind_1";
    if (!expect(openclaw_session_binding_validate(binding, &error), error.c_str())) {
        return 1;
    }

    setenv("VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED", "1", 1);
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS");

    server_openclaw_fabric fabric;
    if (!expect(fabric.configure(true, true, &error), error.c_str())) {
        return 1;
    }
    if (!expect(fabric.enabled(), "fabric should be enabled")) {
        return 1;
    }

    std::vector<llama_cognitive_tool_spec> specs;
    if (!expect(fabric.build_cognitive_specs(&specs), "failed to build cognitive specs")) {
        return 1;
    }
    if (!expect(specs.size() == 2, "expected exec and hard-memory capabilities")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("exec") != nullptr, "expected exec tool lookup to succeed")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("missing") == nullptr, "expected unknown tool lookup to fail")) {
        return 1;
    }

    llama_cognitive_command valid_command = {};
    valid_command.kind = LLAMA_COG_COMMAND_INVOKE_TOOL;
    valid_command.tool_kind = specs[0].tool_kind;
    valid_command.tool_spec_index = 0;
    std::snprintf(valid_command.capability_id, sizeof(valid_command.capability_id), "%s", specs[0].capability_id);
    if (!expect(fabric.resolve_command(valid_command, &error) != nullptr, error.c_str())) {
        return 1;
    }

    llama_cognitive_command invalid_command = valid_command;
    std::snprintf(invalid_command.capability_id, sizeof(invalid_command.capability_id), "%s", "openclaw.exec.fake");
    if (!expect(fabric.resolve_command(invalid_command, &error) == nullptr, "expected invalid capability id to be rejected")) {
        return 1;
    }

    const std::string catalog_path = "/tmp/vicuna-openclaw-catalog-test.json";
    std::remove(catalog_path.c_str());
    setenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH", catalog_path.c_str(), 1);

    const char * initial_catalog = R"({
        "catalog_version": 1,
        "capabilities": [
            {
                "capability_id": "openclaw.exec.initial",
                "tool_surface_id": "vicuna.exec.initial",
                "capability_kind": "tool",
                "owner_plugin_id": "openclaw-initial",
                "tool_name": "exec_initial",
                "description": "Run an initial bounded command",
                "input_schema_json": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": { "type": "string" }
                    }
                },
                "output_contract": "pending_then_result",
                "side_effect_class": "system_exec",
                "approval_mode": "policy_driven",
                "execution_modes": ["sync"],
                "provenance_namespace": "openclaw/openclaw-initial/tool/exec_initial",
                "tool_kind": 4,
                "tool_flags": 0,
                "latency_class": 1,
                "max_steps_reserved": 2,
                "dispatch_backend": "legacy_bash"
            }
        ]
    })";
    {
        std::ofstream out(catalog_path, std::ios::binary | std::ios::trunc);
        out << initial_catalog;
    }

    server_openclaw_fabric reloadable_fabric;
    if (!expect(reloadable_fabric.configure(true, true, &error), error.c_str())) {
        return 1;
    }

    const char * updated_catalog = R"({
        "catalog_version": 1,
        "capabilities": [
            {
                "capability_id": "openclaw.exec.extra",
                "tool_surface_id": "vicuna.exec.extra",
                "capability_kind": "tool",
                "owner_plugin_id": "openclaw-extra",
                "tool_name": "exec_extra",
                "description": "Run an extra bounded command",
                "input_schema_json": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": { "type": "string" }
                    }
                },
                "output_contract": "pending_then_result",
                "side_effect_class": "system_exec",
                "approval_mode": "policy_driven",
                "execution_modes": ["sync"],
                "provenance_namespace": "openclaw/openclaw-extra/tool/exec_extra",
                "tool_kind": 4,
                "tool_flags": 0,
                "latency_class": 1,
                "max_steps_reserved": 2,
                "dispatch_backend": "legacy_bash"
            }
        ]
    })";

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    {
        std::ofstream out(catalog_path, std::ios::binary | std::ios::trunc);
        out << updated_catalog;
    }

    bool reloaded = false;
    if (!expect(reloadable_fabric.maybe_reload(true, true, &reloaded, &error), error.c_str())) {
        return 1;
    }
    if (!expect(reloaded, "expected external catalog reload to occur")) {
        return 1;
    }

    specs.clear();
    if (!expect(reloadable_fabric.build_cognitive_specs(&specs), "failed to build reloaded cognitive specs")) {
        return 1;
    }
    if (!expect(specs.size() == 3, "expected built-ins plus one externally loaded capability")) {
        return 1;
    }

    llama_cognitive_command external_command = {};
    external_command.kind = LLAMA_COG_COMMAND_INVOKE_TOOL;
    external_command.tool_kind = specs[2].tool_kind;
    external_command.tool_spec_index = 2;
    std::snprintf(external_command.capability_id, sizeof(external_command.capability_id), "%s", specs[2].capability_id);
    if (!expect(reloadable_fabric.resolve_command(external_command, &error) != nullptr, error.c_str())) {
        return 1;
    }

    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH");
    std::remove(catalog_path.c_str());
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED");
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS");
    return 0;
}
