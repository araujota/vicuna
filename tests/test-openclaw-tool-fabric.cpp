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
    if (!expect(fabric.configure(true, true, true, &error), error.c_str())) {
        return 1;
    }
    if (!expect(fabric.enabled(), "fabric should be enabled")) {
        return 1;
    }

    std::vector<llama_cognitive_tool_spec> specs;
    if (!expect(fabric.build_cognitive_specs(&specs), "failed to build cognitive specs")) {
        return 1;
    }
    if (!expect(specs.size() == 5, "expected exec, hard-memory query, hard-memory write, codex, and telegram relay capabilities")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("exec") != nullptr, "expected exec tool lookup to succeed")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("hard_memory_write") != nullptr, "expected hard-memory write tool lookup to succeed")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("codex") != nullptr, "expected codex tool lookup to succeed")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("telegram_relay") != nullptr, "expected telegram relay tool lookup to succeed")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("missing") == nullptr, "expected unknown tool lookup to fail")) {
        return 1;
    }

    setenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS", "exec,hard_memory_query", 1);
    server_openclaw_fabric legacy_filtered_fabric;
    if (!expect(legacy_filtered_fabric.configure(true, true, true, &error), error.c_str())) {
        return 1;
    }
    std::vector<llama_cognitive_tool_spec> legacy_specs;
    if (!expect(legacy_filtered_fabric.build_cognitive_specs(&legacy_specs), "failed to build legacy-filtered cognitive specs")) {
        return 1;
    }
    if (!expect(legacy_specs.size() == 4, "expected legacy allowlist to preserve exec, telegram relay, and the full hard-memory tool layer")) {
        return 1;
    }
    if (!expect(legacy_filtered_fabric.capability_by_tool_name("hard_memory_write") != nullptr,
                "expected legacy allowlist compatibility to retain hard-memory write")) {
        return 1;
    }
    if (!expect(legacy_filtered_fabric.capability_by_tool_name("telegram_relay") != nullptr,
                "expected legacy allowlist compatibility to retain telegram relay")) {
        return 1;
    }
    if (!expect(legacy_filtered_fabric.capability_by_tool_name("codex") == nullptr,
                "expected codex to remain filtered when omitted from explicit allowlist")) {
        return 1;
    }
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS");

    std::vector<int32_t> exec_only = { 0 };
    std::vector<server_openclaw_xml_tool_contract> contracts;
    if (!expect(fabric.build_xml_tool_contracts(&contracts, &exec_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(contracts.size() == 1, "expected one XML contract for the selected tool")) {
        return 1;
    }
    if (!expect(contracts[0].tool_name == "exec", "expected exec XML contract")) {
        return 1;
    }
    if (!expect(!contracts[0].args.empty(), "expected schema-derived XML args")) {
        return 1;
    }

    std::string guidance;
    if (!expect(fabric.render_tool_call_xml_guidance(&guidance, &exec_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(guidance.find("<vicuna_tool_call tool=\"exec\">") != std::string::npos,
                "expected canonical XML guidance example")) {
        return 1;
    }
    if (!expect(guidance.find("single command only; no pipes, redirects, chaining, or substitution") != std::string::npos,
                "expected exec guidance to document bounded single-command policy")) {
        return 1;
    }

    server_openclaw_parsed_tool_call parsed = {};
    const std::string valid_tool_xml =
            "<think>I should inspect the filesystem first.</think>\n"
            "<vicuna_tool_call tool=\"exec\">\n"
            "  <arg name=\"command\" type=\"string\">pwd</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(fabric.parse_tool_call_xml(valid_tool_xml, &parsed, &exec_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls.size() == 1, "expected one parsed tool call")) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls[0].name == "exec", "expected exec parsed tool name")) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls[0].arguments == "{\"command\":\"pwd\"}",
                "expected typed JSON arguments from XML")) {
        return 1;
    }
    if (!expect(parsed.message.content.empty(),
                "expected hidden reasoning to be removed from visible assistant content")) {
        return 1;
    }
    if (!expect(parsed.message.reasoning_content == "I should inspect the filesystem first.",
                "expected hidden reasoning to populate assistant reasoning_content")) {
        return 1;
    }
    if (!expect(parsed.captured_planner_reasoning == "I should inspect the filesystem first.",
                "expected planner reasoning capture to preserve hidden reasoning text")) {
        return 1;
    }
    if (!expect(parsed.captured_tool_xml ==
                    "<vicuna_tool_call tool=\"exec\">\n"
                    "  <arg name=\"command\" type=\"string\">pwd</arg>\n"
                    "</vicuna_tool_call>",
                "expected tool XML capture to preserve only the canonical XML block")) {
        return 1;
    }
    if (!expect(parsed.captured_payload == valid_tool_xml, "expected captured payload to preserve thought plus XML")) {
        return 1;
    }

    std::vector<int32_t> write_only = { 2 };
    const std::string valid_memory_write_xml =
            "<think>This should be preserved durably.</think>\n"
            "<vicuna_tool_call tool=\"hard_memory_write\">\n"
            "  <arg name=\"memories\" type=\"json\">[{\"content\":\"The user prefers concise answers.\",\"kind\":\"user_model\",\"domain\":\"user_outcome\",\"tags\":[\"preference\"],\"isStatic\":true}]</arg>\n"
            "  <arg name=\"containerTag\" type=\"string\">vicuna-self-state</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(fabric.parse_tool_call_xml(valid_memory_write_xml, &parsed, &write_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls.size() == 1 &&
                parsed.message.tool_calls[0].name == "hard_memory_write",
                "expected hard-memory write tool XML to parse")) {
        return 1;
    }
    if (!expect(parsed.message.reasoning_content == "This should be preserved durably.",
                "expected hard-memory write XML to preserve hidden reasoning")) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls[0].arguments.find("\"containerTag\":\"vicuna-self-state\"") != std::string::npos &&
                parsed.message.tool_calls[0].arguments.find("\"isStatic\":true") != std::string::npos,
                "expected hard-memory write JSON arguments to preserve Supermemory fields")) {
        return 1;
    }

    std::vector<int32_t> relay_only = { 4 };
    const std::string relay_xml =
            "<think>The latest self-model revision says I should ask the user for clarification.</think>\n"
            "<vicuna_tool_call tool=\"telegram_relay\">\n"
            "  <arg name=\"text\" type=\"string\">Can you clarify which IBM division you meant?</arg>\n"
            "  <arg name=\"intent\" type=\"string\">question</arg>\n"
            "  <arg name=\"dedupeKey\" type=\"string\">dmn-relay-42</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(fabric.parse_tool_call_xml(relay_xml, &parsed, &relay_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls.size() == 1 &&
                parsed.message.tool_calls[0].name == "telegram_relay",
                "expected telegram relay tool XML to parse")) {
        return 1;
    }
    if (!expect(parsed.message.reasoning_content.find("latest self-model revision") != std::string::npos,
                "expected telegram relay XML to preserve hidden reasoning")) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls[0].arguments.find("\"intent\":\"question\"") != std::string::npos &&
                parsed.message.tool_calls[0].arguments.find("\"dedupeKey\":\"dmn-relay-42\"") != std::string::npos,
                "expected telegram relay JSON arguments to preserve intent and dedupe metadata")) {
        return 1;
    }

    const std::string invalid_tool_xml =
            "<vicuna_tool_call tool=\"exec\">"
            "<arg name=\"unknown\" type=\"string\">pwd</arg>"
            "</vicuna_tool_call>";
    if (!expect(!fabric.parse_tool_call_xml(invalid_tool_xml, &parsed, &exec_only, &error),
                "expected undeclared argument to be rejected")) {
        return 1;
    }

    const std::string stripped = fabric.strip_tool_call_xml_markup(valid_tool_xml);
    if (!expect(stripped == "I should inspect the filesystem first.",
                "expected XML markup stripping to preserve only visible prefix")) {
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
    if (!expect(reloadable_fabric.configure(true, true, true, &error), error.c_str())) {
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
    if (!expect(reloadable_fabric.maybe_reload(true, true, true, &reloaded, &error), error.c_str())) {
        return 1;
    }
    if (!expect(reloaded, "expected external catalog reload to occur")) {
        return 1;
    }

    specs.clear();
    if (!expect(reloadable_fabric.build_cognitive_specs(&specs), "failed to build reloaded cognitive specs")) {
        return 1;
    }
    if (!expect(specs.size() == 6, "expected built-ins plus one externally loaded capability")) {
        return 1;
    }

    llama_cognitive_command external_command = {};
    external_command.kind = LLAMA_COG_COMMAND_INVOKE_TOOL;
    external_command.tool_kind = specs.back().tool_kind;
    external_command.tool_spec_index = (int32_t) specs.size() - 1;
    std::snprintf(external_command.capability_id, sizeof(external_command.capability_id), "%s", specs.back().capability_id);
    if (!expect(reloadable_fabric.resolve_command(external_command, &error) != nullptr, error.c_str())) {
        return 1;
    }

    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH");
    std::remove(catalog_path.c_str());
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED");
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS");
    return 0;
}
