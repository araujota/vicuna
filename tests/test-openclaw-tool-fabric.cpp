#include "common/openclaw-session-binding.h"
#include "common/openclaw-tool-fabric-events.h"
#include "common/openclaw-tool-fabric.h"
#include "../tools/server/server-openclaw-fabric.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED");
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS");
    return 0;
}
