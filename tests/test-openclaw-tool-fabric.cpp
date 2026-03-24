#include "common/openclaw-session-binding.h"
#include "common/openclaw-tool-fabric-events.h"
#include "common/openclaw-tool-fabric.h"
#include "../tools/server/server-openclaw-fabric.h"
#include "../tools/server/server-task.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <limits>

static bool expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        return false;
    }
    return true;
}

int main() {
    openclaw_tool_capability_descriptor descriptor = {};
    descriptor.capability_id = "openclaw.test.query";
    descriptor.tool_surface_id = "vicuna.test.query";
    descriptor.capability_kind = "memory_adapter";
    descriptor.owner_plugin_id = "openclaw-test";
    descriptor.tool_name = "memory_query_test";
    descriptor.tool_family_id = "memory_test";
    descriptor.tool_family_name = "Memory Test";
    descriptor.tool_family_description = "Query one synthetic memory surface.";
    descriptor.method_name = "query";
    descriptor.method_description = "Run one synthetic memory query.";
    descriptor.description = "Query a synthetic memory surface";
    descriptor.input_schema_json = R"({"type":"object","properties":{"query":{"type":"string","description":"The retrieval query to run."}}})";
    descriptor.fixed_arguments_json = R"({"mode":"inspect"})";
    descriptor.output_contract = "completed_result";
    descriptor.side_effect_class = "memory_read";
    descriptor.approval_mode = "none";
    descriptor.execution_safety_class = "read_only";
    descriptor.execution_modes = {"sync"};
    descriptor.provenance_namespace = "openclaw/test/memory_query";
    descriptor.tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_QUERY;
    descriptor.tool_flags = LLAMA_COG_TOOL_ACTIVE_ELIGIBLE | LLAMA_COG_TOOL_DMN_ELIGIBLE;
    descriptor.latency_class = LLAMA_COG_TOOL_LATENCY_MEDIUM;
    descriptor.max_steps_reserved = 2;
    descriptor.dispatch_backend = "legacy_bash";

    std::string error;
    if (!expect(openclaw_tool_capability_descriptor_validate(descriptor, &error), error.c_str())) {
        return 1;
    }

    openclaw_tool_capability_descriptor missing_description_descriptor = descriptor;
    missing_description_descriptor.input_schema_json = R"({"type":"object","properties":{"query":{"type":"string"}}})";
    if (!expect(!openclaw_tool_capability_descriptor_validate(missing_description_descriptor, &error),
                "expected descriptor validation to reject missing parameter descriptions")) {
        return 1;
    }
    if (!expect(error.find("input_schema_json.properties.query") != std::string::npos,
                "expected missing-description error to identify the schema path")) {
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
    if (!expect(parsed_descriptor.fixed_arguments_json == descriptor.fixed_arguments_json,
                "descriptor round-trip lost fixed_arguments_json")) {
        return 1;
    }
    if (!expect(parsed_descriptor.execution_safety_class == descriptor.execution_safety_class,
                "descriptor round-trip lost execution_safety_class")) {
        return 1;
    }

    server_task active_react_task(SERVER_TASK_TYPE_COMPLETION);
    active_react_task.has_active_trace = true;
    active_react_task.foreground_text = "What is the weather in Chicago tomorrow?";
    if (!expect(server_task_should_prepare_authoritative_react(active_react_task),
                "expected active completion task with an active trace to be eligible for ReAct prompt preparation")) {
        return 1;
    }
    active_react_task.react_assistant_prefill = "<think>\nThought: ";
    if (!expect(server_task_has_authoritative_react_surface(active_react_task),
                "expected active completion task with a prepared ReAct prompt to be ReAct-ready")) {
        return 1;
    }
    if (!expect(!active_react_task.react_resuming_from_tool_result,
                "expected fresh active turn to remain outside resumed-from-tool-result state")) {
        return 1;
    }
    if (!expect(server_task_should_prepare_active_authoritative(active_react_task),
                "expected foreground active completion task to remain eligible for active authoritative preparation")) {
        return 1;
    }
    if (!expect(active_react_task.react_retry_limit == 12,
                "expected authoritative ReAct continuation budget to default to a bounded retry budget")) {
        return 1;
    }
    if (!expect(active_react_task.react_stage_retry_limit == 3,
                "expected authoritative ReAct stage retry budget to default to three attempts per stage")) {
        return 1;
    }
    if (!expect(active_react_task.react_retry_count == 0,
                "expected active ReAct retry counter to start at zero")) {
        return 1;
    }
    if (!expect(active_react_task.react_pending_terminal_action == LLAMA_AUTHORITATIVE_REACT_ACTION_NONE,
                "expected active ReAct pending terminal action to start empty")) {
        return 1;
    }
    active_react_task.foreground_consent_requested = true;
    active_react_task.foreground_consent_active = true;
    active_react_task.foreground_consent_episode_id = 44;
    active_react_task.foreground_consent_scope_id = "fgc_1";
    active_react_task.add_child(active_react_task.id, 777);
    if (!expect(active_react_task.child_tasks.size() == 1,
                "expected active task child copy to be created")) {
        return 1;
    }
    if (!expect(active_react_task.child_tasks.front().foreground_consent_requested &&
                active_react_task.child_tasks.front().foreground_consent_active &&
                active_react_task.child_tasks.front().foreground_consent_episode_id == 44 &&
                active_react_task.child_tasks.front().foreground_consent_scope_id == "fgc_1",
                "expected active task child copy to preserve foreground consent state")) {
        return 1;
    }

    server_task inactive_active_task(SERVER_TASK_TYPE_COMPLETION);
    if (!expect(!server_task_should_prepare_authoritative_react(inactive_active_task),
                "expected active task without a trace to be ineligible for ReAct prompt preparation")) {
        return 1;
    }
    inactive_active_task.foreground_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
    if (!expect(!server_task_should_prepare_active_authoritative(inactive_active_task),
                "expected system-origin completion tasks to bypass active authoritative preparation")) {
        return 1;
    }
    if (!expect(!server_task_has_authoritative_react_surface(inactive_active_task),
                "expected active task without a prepared ReAct prompt to be non-ReAct-ready")) {
        return 1;
    }

    server_task dmn_react_task(SERVER_TASK_TYPE_COMPLETION);
    dmn_react_task.has_dmn_trace = true;
    dmn_react_task.react_origin = SERVER_REACT_ORIGIN_DMN;
    if (!expect(server_task_should_prepare_authoritative_react(dmn_react_task),
                "expected DMN completion task with a DMN trace to be eligible for ReAct prompt preparation")) {
        return 1;
    }
    if (!expect(server_task_has_dmn_authoritative_identity(dmn_react_task),
                "expected DMN completion task to preserve DMN authoritative identity")) {
        return 1;
    }
    if (!expect(!server_task_should_prepare_active_authoritative(dmn_react_task),
                "expected DMN completion task to bypass generic active authoritative preparation")) {
        return 1;
    }
    dmn_react_task.react_assistant_prefill = "<think>\nThought: ";
    if (!expect(server_task_has_authoritative_react_surface(dmn_react_task),
                "expected DMN completion task with a prepared ReAct prompt to be ReAct-ready")) {
        return 1;
    }
    if (!expect(!dmn_react_task.react_resuming_from_tool_result,
                "expected fresh DMN turn to remain outside resumed-from-tool-result state")) {
        return 1;
    }

    server_task resumed_active_task(SERVER_TASK_TYPE_COMPLETION);
    resumed_active_task.has_active_trace = true;
    resumed_active_task.react_assistant_prefill = "<think>\nThought: ";
    resumed_active_task.react_origin = SERVER_REACT_ORIGIN_ACTIVE;
    resumed_active_task.react_resuming_from_tool_result = true;
    resumed_active_task.react_retry_count = 3;
    resumed_active_task.foreground_text = "Need the latest weather forecast.";
    if (!expect(resumed_active_task.react_resuming_from_tool_result,
                "expected resumed active turn to preserve tool-observation resume state")) {
        return 1;
    }
    if (!expect(resumed_active_task.react_retry_count == 3,
                "expected resumed active turn to preserve retry state across continuation")) {
        return 1;
    }
    server_task parent_task(SERVER_TASK_TYPE_COMPLETION);
    parent_task.foreground_text = "What is the stock price right now?";
    parent_task.react_retry_limit = 12;
    parent_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_ASK;
    parent_task.react_last_step_name = "tool_result";
    parent_task.react_last_step_output = "{\"tool_family_id\":\"weather\"}";
    parent_task.react_last_step_result = "{\"temperature_f\":72}";
    parent_task.react_last_tool_family_id = "weather";
    parent_task.react_last_tool_method_name = "current";
    parent_task.react_last_tool_capability_id = "openclaw.weather.current";
    parent_task.react_last_tool_observation = "{\"temperature_f\":72}";
    parent_task.react_history.push_back(server_react_history_entry{
            "select_tool_family",
            "reasoning",
            "<think>\nThought: use weather.\nAction: select_tool\n</think>"});
    parent_task.react_history.push_back(server_react_history_entry{
            "tool_result",
            "tool_result",
            "{\"temperature_f\":72}"});
    parent_task.add_child(100, 101);
    if (!expect(parent_task.child_tasks.size() == 1, "expected one child task to be created")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].foreground_text == parent_task.foreground_text,
                "expected child task to preserve foreground text for continuation policy")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_retry_limit == parent_task.react_retry_limit,
                "expected child task to preserve bounded continuation budget")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_pending_terminal_action == parent_task.react_pending_terminal_action,
                "expected child task to preserve pending terminal action state")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_last_step_name == parent_task.react_last_step_name,
                "expected child task to preserve pinned last-step name")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_last_step_result == parent_task.react_last_step_result,
                "expected child task to preserve pinned last-step result")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_last_tool_capability_id == parent_task.react_last_tool_capability_id,
                "expected child task to preserve pinned last tool capability")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_last_tool_observation == parent_task.react_last_tool_observation,
                "expected child task to preserve pinned last tool observation")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_history.size() == parent_task.react_history.size(),
                "expected child task to preserve staged react history entries")) {
        return 1;
    }
    if (!expect(parent_task.child_tasks[0].react_history[0].text == parent_task.react_history[0].text,
                "expected child task to preserve staged react history contents")) {
        return 1;
    }

    server_task wrong_type_task(SERVER_TASK_TYPE_EMBEDDING);
    wrong_type_task.has_active_trace = true;
    if (!expect(!server_task_should_prepare_authoritative_react(wrong_type_task),
                "expected non-completion task types to remain outside ReAct prompt preparation")) {
        return 1;
    }
    wrong_type_task.react_assistant_prefill = "<think>\nThought: ";
    if (!expect(!server_task_has_authoritative_react_surface(wrong_type_task),
                "expected non-completion task types to remain outside authoritative ReAct")) {
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
    if (!expect(specs.size() >= 5, "expected the builtin OpenClaw capability floor to remain present")) {
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
    if (!expect(fabric.capability_by_tool_name("ask_with_options") != nullptr, "expected ask_with_options tool lookup to succeed")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("hard_memory_query")->descriptor.execution_safety_class == "read_only",
                "expected hard_memory_query to remain read-only")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("hard_memory_write")->descriptor.execution_safety_class == "approval_required",
                "expected hard_memory_write to require approval")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("codex")->descriptor.execution_safety_class == "approval_required",
                "expected codex to require approval")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("telegram_relay")->descriptor.execution_safety_class == "approval_required",
                "expected telegram_relay to require approval")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("ask_with_options")->descriptor.execution_safety_class == "read_only",
                "expected ask_with_options transport to remain read-only")) {
        return 1;
    }
    if (!expect(fabric.capability_by_tool_name("missing") == nullptr, "expected unknown tool lookup to fail")) {
        return 1;
    }

    setenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS", "hard_memory_query", 1);
    server_openclaw_fabric legacy_filtered_fabric;
    if (!expect(legacy_filtered_fabric.configure(true, true, true, &error), error.c_str())) {
        return 1;
    }
    std::vector<llama_cognitive_tool_spec> legacy_specs;
    if (!expect(legacy_filtered_fabric.build_cognitive_specs(&legacy_specs), "failed to build legacy-filtered cognitive specs")) {
        return 1;
    }
    if (!expect(legacy_specs.size() >= 4, "expected legacy allowlist compatibility to preserve the builtin capability floor")) {
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
    if (!expect(legacy_filtered_fabric.capability_by_tool_name("ask_with_options") != nullptr,
                "expected legacy allowlist compatibility to retain ask_with_options")) {
        return 1;
    }
    if (!expect(legacy_filtered_fabric.capability_by_tool_name("codex") == nullptr,
                "expected codex to remain filtered when omitted from explicit allowlist")) {
        return 1;
    }
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS");

    std::vector<int32_t> hard_memory_query_only = { 0 };
    std::vector<server_openclaw_xml_tool_contract> contracts;
    if (!expect(fabric.build_xml_tool_contracts(&contracts, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(contracts.size() == 1, "expected one XML contract for the selected tool")) {
        return 1;
    }
    if (!expect(contracts[0].tool_family_id == "hard_memory", "expected hard-memory XML family")) {
        return 1;
    }
    if (!expect(contracts[0].method_name == "query", "expected hard-memory XML method")) {
        return 1;
    }
    if (!expect(!contracts[0].args.empty(), "expected schema-derived XML args")) {
        return 1;
    }
    if (!expect(!contracts[0].args[0].description.empty(), "expected XML arg requirements to retain parameter descriptions")) {
        return 1;
    }
    std::vector<server_openclaw_tool_family_contract> families;
    if (!expect(fabric.build_tool_family_contracts(&families, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(families.size() == 1 && families[0].tool_family_id == "hard_memory",
                "expected hard-memory tool family contract to build")) {
        return 1;
    }
    std::vector<server_openclaw_tool_method_contract> methods;
    if (!expect(fabric.build_tool_method_contracts("hard_memory", &methods, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(methods.size() == 1 && methods[0].method_name == "query",
                "expected hard-memory query method contract to build")) {
        return 1;
    }

    std::string guidance;
    if (!expect(fabric.render_tool_family_selection_guidance(&guidance, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(guidance.find("{\"tool_family_id\":\"hard_memory\"}") != std::string::npos,
                "expected staged guidance to expose the JSON tool-family selector")) {
        return 1;
    }
    if (!expect(guidance.find("durable memory primitives") != std::string::npos,
                "expected tool-family guidance to retain family descriptions")) {
        return 1;
    }
    if (!expect(fabric.render_tool_method_selection_guidance("hard_memory", &guidance, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(guidance.find("{\"method_name\":\"query\"}") != std::string::npos,
                "expected staged guidance to expose the JSON method selector")) {
        return 1;
    }
    if (!expect(guidance.find("Query Vicuña hard memory with a retrieval string.") != std::string::npos ||
                guidance.find("typed retrieval results") != std::string::npos,
                "expected tool-method guidance to retain method descriptions")) {
        return 1;
    }
    if (!expect(fabric.render_tool_call_xml_guidance(&guidance, nullptr, nullptr, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(guidance.find("Canonical example:") != std::string::npos &&
                guidance.find("tool=\"hard_memory\"") != std::string::npos &&
                guidance.find("method=\"query\"") != std::string::npos,
                "expected canonical XML guidance example")) {
        return 1;
    }
    if (!expect(guidance.find("retrieval query") != std::string::npos ||
                guidance.find("relevant durable memories") != std::string::npos,
                "expected hard-memory guidance to document the retrieval query contract")) {
        return 1;
    }
    server_openclaw_tool_argument_contract query_argument_contract = {};
    if (!expect(fabric.build_tool_argument_contract("hard_memory", "query", &query_argument_contract, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(query_argument_contract.input_schema.is_object() &&
                query_argument_contract.input_schema.value("type", "") == "object",
                "expected hard-memory query argument contract to expose the method JSON schema")) {
        return 1;
    }
    if (!expect(fabric.render_tool_argument_json_guidance("hard_memory", "query", &guidance, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(guidance.find("Emit exactly one JSON object") != std::string::npos,
                "expected argument guidance to require one JSON object")) {
        return 1;
    }

    const std::string enum_catalog_path = "/tmp/vicuna-openclaw-enum-catalog-test.json";
    const char * enum_catalog = R"({
        "catalog_version": 1,
        "capabilities": [
            {
                "capability_id": "openclaw.media.test",
                "tool_surface_id": "vicuna.media.test",
                "capability_kind": "tool",
                "owner_plugin_id": "openclaw-test",
                "tool_name": "media_test",
                "tool_family_id": "media_test",
                "tool_family_name": "Media Test",
                "tool_family_description": "Inspect a media catalog with explicit action enums",
                "method_name": "media_test",
                "method_description": "Inspect a media catalog with explicit action enums",
                "description": "Inspect a media catalog with explicit action enums",
                "input_schema_json": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The media action to perform.",
                            "enum": ["list_series", "lookup_series"]
                        },
                        "series_type": {
                            "type": "string",
                            "description": "The series type filter.",
                            "enum": ["standard", "daily", "anime"]
                        }
                    }
                },
                "output_contract": "completed_result",
                "side_effect_class": "service_api",
                "approval_mode": "policy_driven",
                "execution_modes": ["sync"],
                "provenance_namespace": "openclaw/openclaw-test/tool/media_test",
                "tool_kind": 4,
                "tool_flags": 27,
                "latency_class": 1,
                "max_steps_reserved": 2,
                "dispatch_backend": "legacy_bash"
            }
        ]
    })";
    {
        std::ofstream out(enum_catalog_path, std::ios::binary | std::ios::trunc);
        out << enum_catalog;
    }
    setenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH", enum_catalog_path.c_str(), 1);
    server_openclaw_fabric enum_fabric;
    if (!expect(enum_fabric.configure(true, true, true, &error), error.c_str())) {
        return 1;
    }
    int32_t enum_spec_index = -1;
    for (size_t i = 0; i < enum_fabric.capabilities().size(); ++i) {
        if (enum_fabric.capabilities()[i].descriptor.tool_name == "media_test") {
            enum_spec_index = (int32_t) i;
            break;
        }
    }
    if (!expect(enum_spec_index >= 0, "expected external enum test tool to load")) {
        return 1;
    }
    std::vector<int32_t> enum_only = { enum_spec_index };
    std::string enum_guidance;
    if (!expect(enum_fabric.render_tool_call_xml_guidance(&enum_guidance, nullptr, nullptr, &enum_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(enum_guidance.find("Allowed values: list_series, lookup_series.") != std::string::npos,
                "expected XML guidance to include enum-constrained action values")) {
        return 1;
    }
    if (!expect(enum_guidance.find("<arg name=\"action\" type=\"string\">list_series</arg>") != std::string::npos,
                "expected canonical XML example to use a valid enum value")) {
        return 1;
    }
    server_openclaw_parsed_tool_call enum_parsed = {};
    const std::string invalid_enum_xml =
            "<vicuna_tool_call tool=\"media_test\" method=\"media_test\">\n"
            "  <arg name=\"action\" type=\"string\">inspect</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(!enum_fabric.parse_tool_call_xml(invalid_enum_xml, &enum_parsed, nullptr, nullptr, &enum_only, &error),
                "expected invalid enum-constrained string value to be rejected")) {
        return 1;
    }
    if (!expect(error.find("unsupported value for argument \"action\": inspect") != std::string::npos,
                "expected enum rejection to explain the invalid argument value")) {
        return 1;
    }
    std::remove(enum_catalog_path.c_str());
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH");

    server_openclaw_parsed_tool_call parsed = {};
    server_openclaw_parsed_tool_selection parsed_tool_selection = {};
    server_openclaw_parsed_tool_arguments parsed_tool_arguments = {};
    const std::string valid_tool_selection_json =
            "<think>\n"
            "Thought: I should query hard memory first.\n"
            "</think>\n"
            "{\"action\":\"select_tool\",\"tool_family_id\":\"hard_memory\"}";
    if (!expect(fabric.parse_tool_selection_json(valid_tool_selection_json, &parsed_tool_selection, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed_tool_selection.tool_family_id == "hard_memory" &&
                parsed_tool_selection.xml_block == "{\"action\":\"select_tool\",\"tool_family_id\":\"hard_memory\"}",
                "expected staged tool selection JSON to resolve the hard-memory family")) {
        return 1;
    }
    const std::string valid_tool_selection_xml =
            "<think>\n"
            "Thought: I should query hard memory first.\n"
            "Action: select_tool\n"
            "</think>\n"
            "SelectedToolFamily: hard_memory";
    if (!expect(fabric.parse_tool_selection_xml(valid_tool_selection_xml, &parsed_tool_selection, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed_tool_selection.tool_family_id == "hard_memory",
                "expected staged tool selection to resolve the hard-memory family")) {
        return 1;
    }
    if (!expect(parsed_tool_selection.captured_planner_reasoning ==
                    "Thought: I should query hard memory first.\nAction: select_tool",
                "expected staged tool selection to preserve hidden reasoning")) {
        return 1;
    }
    if (!expect(parsed_tool_selection.xml_block == "SelectedToolFamily: hard_memory",
                "expected staged tool selection to retain the selector line")) {
        return 1;
    }
    server_openclaw_parsed_tool_method_selection parsed_tool_method = {};
    const std::string valid_tool_method_json =
            "<think>\n"
            "Thought: The query method is the only valid path.\n"
            "</think>\n"
            "{\"action\":\"select_method\",\"method_name\":\"query\"}";
    if (!expect(fabric.parse_tool_method_selection_json(valid_tool_method_json, "hard_memory", &parsed_tool_method, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed_tool_method.tool_family_id == "hard_memory" &&
                parsed_tool_method.method_name == "query" &&
                parsed_tool_method.xml_block == "{\"action\":\"select_method\",\"method_name\":\"query\"}",
                "expected staged method selection JSON to resolve the hard-memory query method")) {
        return 1;
    }
    const std::string valid_tool_method_xml =
            "<think>\n"
            "Thought: The query method is the only valid path.\n"
            "Action: select_method\n"
            "</think>\n"
            "SelectedMethod: query";
    if (!expect(fabric.parse_tool_method_selection_xml(valid_tool_method_xml, "hard_memory", &parsed_tool_method, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed_tool_method.tool_family_id == "hard_memory" &&
                parsed_tool_method.method_name == "query" &&
                parsed_tool_method.tool_name == "hard_memory_query",
                "expected staged method selection to resolve the hard-memory query method")) {
        return 1;
    }
    if (!expect(parsed_tool_method.captured_planner_reasoning ==
                    "Thought: The query method is the only valid path.\nAction: select_method",
                "expected staged method selection to preserve hidden reasoning")) {
        return 1;
    }
    if (!expect(parsed_tool_method.xml_block == "SelectedMethod: query",
                "expected staged method selection to retain the selector line")) {
        return 1;
    }
    const std::string valid_tool_arguments_json =
            "<think>\n"
            "Thought: The query should stay narrow and direct.\n"
            "</think>\n"
            "{\"action\":\"act\",\"query\":\"recent preferences\"}";
    if (!expect(fabric.parse_tool_arguments_json(valid_tool_arguments_json, "hard_memory", "query", &parsed_tool_arguments, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed_tool_arguments.arguments.value("query", std::string()) == "recent preferences" &&
                parsed_tool_arguments.message.tool_calls.size() == 1 &&
                parsed_tool_arguments.message.tool_calls.front().arguments == "{\"query\":\"recent preferences\"}",
                "expected method arguments JSON to produce one hard-memory query tool call")) {
        return 1;
    }

    const std::string invalid_tool_selection_json =
            "<think>\n"
            "Thought: I should query hard memory first.\n"
            "</think>\n"
            "{\"tool_family_id\":\"hard_memory\"}";
    if (!expect(!fabric.parse_tool_selection_json(invalid_tool_selection_json, &parsed_tool_selection, &hard_memory_query_only, &error),
                "expected staged tool selection JSON without an action field to be rejected")) {
        return 1;
    }
    if (!expect(error.find("action=\"select_tool\"") != std::string::npos,
                "expected staged tool selection rejection to mention the required action field")) {
        return 1;
    }

    const std::string invalid_tool_arguments_json =
            "<think>\n"
            "Thought: The query should stay narrow and direct.\n"
            "</think>\n"
            "{\"action\":\"select_method\",\"query\":\"recent preferences\"}";
    if (!expect(!fabric.parse_tool_arguments_json(invalid_tool_arguments_json, "hard_memory", "query", &parsed_tool_arguments, &hard_memory_query_only, &error),
                "expected staged method arguments JSON with the wrong action to be rejected")) {
        return 1;
    }
    if (!expect(error.find("action=\"act\"") != std::string::npos,
                "expected staged method arguments rejection to mention the required act action")) {
        return 1;
    }

    const std::string valid_tool_xml =
            "<think>\n"
            "Thought: I should query hard memory first.\n"
            "Action: act\n"
            "</think>\n"
            "<vicuna_tool_call tool=\"hard_memory\" method=\"query\">\n"
            "  <arg name=\"query\" type=\"string\">recent preferences</arg>\n"
            "</vicuna_tool_call>";
    const std::string selected_hard_memory_family = "hard_memory";
    const std::string selected_hard_memory_method = "query";
    if (!expect(fabric.parse_tool_call_xml(valid_tool_xml, &parsed, &selected_hard_memory_family, &selected_hard_memory_method, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls.size() == 1, "expected one parsed tool call")) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls[0].name == "hard_memory_query", "expected hard-memory parsed tool name")) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls[0].arguments == "{\"query\":\"recent preferences\"}",
                "expected typed JSON arguments from XML")) {
        return 1;
    }
    if (!expect(parsed.message.content.empty(),
                "expected hidden reasoning to be removed from visible assistant content")) {
        return 1;
    }
    if (!expect(parsed.message.reasoning_content ==
                    "Thought: I should query hard memory first.\nAction: act",
                "expected hidden reasoning to populate assistant reasoning_content")) {
        return 1;
    }
    if (!expect(parsed.captured_planner_reasoning ==
                    "Thought: I should query hard memory first.\nAction: act",
                "expected planner reasoning capture to preserve hidden reasoning text")) {
        return 1;
    }
    if (!expect(parsed.captured_tool_xml ==
                    "<vicuna_tool_call tool=\"hard_memory\" method=\"query\">\n"
                    "  <arg name=\"query\" type=\"string\">recent preferences</arg>\n"
                    "</vicuna_tool_call>",
                "expected tool XML capture to preserve only the canonical XML block")) {
        return 1;
    }
    if (!expect(parsed.captured_payload == valid_tool_xml, "expected captured payload to preserve thought plus XML")) {
        return 1;
    }

    const std::string mismatched_tool_xml =
            "<vicuna_tool_call tool=\"hard_memory\" method=\"write\">\n"
            "  <arg name=\"query\" type=\"string\">recent preferences</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(!fabric.parse_tool_call_xml(mismatched_tool_xml, &parsed, &selected_hard_memory_family, &selected_hard_memory_method, &hard_memory_query_only, &error),
                "expected staged tool-call parsing to reject a method outside the selected method contract")) {
        return 1;
    }

    std::vector<int32_t> write_only = { 1 };
    const std::string valid_memory_write_xml =
            "<think>This should be preserved durably.</think>\n"
            "<vicuna_tool_call tool=\"hard_memory\" method=\"write\">\n"
            "  <arg name=\"memories\" type=\"json\">[{\"content\":\"The user prefers concise answers.\",\"kind\":\"user_model\",\"domain\":\"user_outcome\",\"tags\":[\"preference\"],\"isStatic\":true}]</arg>\n"
            "  <arg name=\"containerTag\" type=\"string\">vicuna-self-state</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(fabric.parse_tool_call_xml(valid_memory_write_xml, &parsed, nullptr, nullptr, &write_only, &error), error.c_str())) {
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

    std::vector<int32_t> relay_only = { 3 };
    const std::string relay_xml =
            "<think>The latest self-model revision says I should ask the user for clarification.</think>\n"
            "<vicuna_tool_call tool=\"telegram\" method=\"relay\">\n"
            "  <arg name=\"text\" type=\"string\">Can you clarify which IBM division you meant?</arg>\n"
            "  <arg name=\"intent\" type=\"string\">question</arg>\n"
            "  <arg name=\"dedupeKey\" type=\"string\">dmn-relay-42</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(fabric.parse_tool_call_xml(relay_xml, &parsed, nullptr, nullptr, &relay_only, &error), error.c_str())) {
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

    std::vector<int32_t> ask_only = { 4 };
    const std::string ask_xml =
            "<think>I need the user to choose one concrete option before continuing.</think>\n"
            "<vicuna_tool_call tool=\"telegram\" method=\"ask_with_options\">\n"
            "  <arg name=\"question\" type=\"string\">Which deployment target should I use?</arg>\n"
            "  <arg name=\"options\" type=\"json\">[\"staging\",\"production\"]</arg>\n"
            "  <arg name=\"dedupeKey\" type=\"string\">active-ask-7</arg>\n"
            "</vicuna_tool_call>";
    if (!expect(fabric.parse_tool_call_xml(ask_xml, &parsed, nullptr, nullptr, &ask_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls.size() == 1 &&
                parsed.message.tool_calls[0].name == "ask_with_options",
                "expected ask_with_options tool XML to parse")) {
        return 1;
    }
    if (!expect(parsed.message.reasoning_content.find("choose one concrete option") != std::string::npos,
                "expected ask_with_options XML to preserve hidden reasoning")) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls[0].arguments.find("\"question\":\"Which deployment target should I use?\"") != std::string::npos &&
                parsed.message.tool_calls[0].arguments.find("\"options\":[\"staging\",\"production\"]") != std::string::npos &&
                parsed.message.tool_calls[0].arguments.find("\"dedupeKey\":\"active-ask-7\"") != std::string::npos,
                "expected ask_with_options JSON arguments to preserve question, options, and dedupe metadata")) {
        return 1;
    }

    const std::string invalid_tool_xml =
            "<vicuna_tool_call tool=\"hard_memory\" method=\"query\">"
            "<arg name=\"unknown\" type=\"string\">pwd</arg>"
            "</vicuna_tool_call>";
    if (!expect(!fabric.parse_tool_call_xml(invalid_tool_xml, &parsed, nullptr, nullptr, &hard_memory_query_only, &error),
                "expected undeclared argument to be rejected")) {
        return 1;
    }

    const std::string trailing_tool_xml =
            "<think>\n"
            "Thought: I should query hard memory first.\n"
            "</think>\n"
            "<vicuna_tool_call tool=\"hard_memory\" method=\"query\">\n"
            "  <arg name=\"query\" type=\"string\">recent preferences</arg>\n"
            "</vicuna_tool_call>\n"
            "Visible trailing content that should not remain authoritative.";
    if (!expect(fabric.recover_tool_call_xml(trailing_tool_xml, &parsed, nullptr, nullptr, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls.size() == 1 &&
                parsed.message.tool_calls[0].name == "hard_memory_query" &&
                parsed.message.tool_calls[0].arguments == "{\"query\":\"recent preferences\"}",
                "expected recovery to preserve a strict XML tool call when trailing content is emitted")) {
        return 1;
    }

    const std::string partial_tool_xml =
            "<think>\n"
            "Thought: I should query hard memory first.\n"
            "</think>\n"
            "<vicuna_tool_call tool=\"hard_memory\" method=\"query\">\n"
            "  <arg name=\"query\" type=\"string\">recent preferences</arg>\n";
    if (!expect(fabric.recover_tool_call_xml(partial_tool_xml, &parsed, nullptr, nullptr, &hard_memory_query_only, &error), error.c_str())) {
        return 1;
    }
    if (!expect(parsed.message.tool_calls.size() == 1 &&
                parsed.message.tool_calls[0].name == "hard_memory_query" &&
                parsed.message.tool_calls[0].arguments == "{\"query\":\"recent preferences\"}",
                "expected recovery to salvage the emitted tool call from a partial XML block")) {
        return 1;
    }
    if (!expect(parsed.message.reasoning_content == "Thought: I should query hard memory first.",
                "expected recovery to preserve hidden reasoning while salvaging partial XML")) {
        return 1;
    }

    const std::string stripped = fabric.strip_tool_call_xml_markup(valid_tool_xml);
    if (!expect(stripped == "Thought: I should query hard memory first.\nAction: act",
                "expected XML markup stripping to preserve only visible prefix")) {
        return 1;
    }
    const std::string stripped_tool_selection = fabric.strip_tool_call_xml_markup(valid_tool_selection_xml);
    if (!expect(stripped_tool_selection.find("I should query hard memory first.") != std::string::npos &&
                stripped_tool_selection.find("Action: select_tool") != std::string::npos,
                "expected XML markup stripping to remove staged tool-selection XML")) {
        return 1;
    }
    const std::string stripped_tool_method = fabric.strip_tool_call_xml_markup(valid_tool_method_xml);
    if (!expect(stripped_tool_method.find("The query method is the only valid path.") != std::string::npos &&
                stripped_tool_method.find("Action: select_method") != std::string::npos,
                "expected XML markup stripping to remove staged method-selection XML")) {
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
    std::snprintf(invalid_command.capability_id, sizeof(invalid_command.capability_id), "%s", "openclaw.test.fake");
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
                "capability_id": "openclaw.weather.initial",
                "tool_surface_id": "vicuna.weather.initial",
                "capability_kind": "tool",
                "owner_plugin_id": "openclaw-initial",
                "tool_name": "weather_initial",
                "tool_family_id": "weather",
                "tool_family_name": "Weather",
                "tool_family_description": "Read one synthetic weather surface.",
                "method_name": "lookup",
                "method_description": "Run one weather lookup.",
                "description": "Run an initial weather lookup",
                "input_schema_json": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "The weather query to run." }
                    }
                },
                "output_contract": "completed_result",
                "side_effect_class": "network_read",
                "approval_mode": "policy_driven",
                "execution_modes": ["sync"],
                "provenance_namespace": "openclaw/openclaw-initial/tool/weather_initial",
                "tool_kind": 4,
                "tool_flags": 27,
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
                "capability_id": "openclaw.weather.extra",
                "tool_surface_id": "vicuna.weather.extra",
                "capability_kind": "tool",
                "owner_plugin_id": "openclaw-extra",
                "tool_name": "weather_extra",
                "tool_family_id": "weather",
                "tool_family_name": "Weather",
                "tool_family_description": "Read one synthetic weather surface.",
                "method_name": "lookup",
                "method_description": "Run one weather lookup.",
                "description": "Run an extra weather lookup",
                "input_schema_json": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "The weather query to run." }
                    }
                },
                "output_contract": "completed_result",
                "side_effect_class": "network_read",
                "approval_mode": "policy_driven",
                "execution_modes": ["sync"],
                "provenance_namespace": "openclaw/openclaw-extra/tool/weather_extra",
                "tool_kind": 4,
                "tool_flags": 27,
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
    if (!expect(specs.size() >= 6, "expected built-ins plus one externally loaded capability")) {
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

    const char * filtered_exec_catalog = R"({
        "catalog_version": 1,
        "capabilities": [
            {
                "capability_id": "openclaw.exec.command",
                "tool_surface_id": "vicuna.exec.main",
                "capability_kind": "tool",
                "owner_plugin_id": "openclaw-stale",
                "tool_name": "exec",
                "tool_family_id": "exec",
                "tool_family_name": "Exec",
                "tool_family_description": "Run a raw shell command.",
                "method_name": "command",
                "method_description": "Run one raw shell command.",
                "description": "A stale raw exec entry that should be ignored.",
                "input_schema_json": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": { "type": "string", "description": "The shell command to run." }
                    }
                },
                "output_contract": "pending_then_result",
                "side_effect_class": "system_exec",
                "approval_mode": "policy_driven",
                "execution_modes": ["sync"],
                "provenance_namespace": "openclaw/openclaw-stale/tool/exec",
                "tool_kind": 4,
                "tool_flags": 27,
                "latency_class": 1,
                "max_steps_reserved": 2,
                "dispatch_backend": "legacy_bash"
            }
        ]
    })";
    {
        std::ofstream out(catalog_path, std::ios::binary | std::ios::trunc);
        out << filtered_exec_catalog;
    }
    server_openclaw_fabric filtered_exec_fabric;
    error.clear();
    if (!expect(filtered_exec_fabric.configure(true, true, true, &error), error.c_str())) {
        return 1;
    }
    specs.clear();
    if (!expect(filtered_exec_fabric.build_cognitive_specs(&specs), "failed to build specs for stale exec filter test")) {
        return 1;
    }
    if (!expect(specs.size() == 5,
                "expected stale exec catalog entries to be filtered out of the exposed capability set")) {
        return 1;
    }
    if (!expect(filtered_exec_fabric.capability_by_tool_name("exec") == nullptr,
                "expected stale exec catalog entries to remain unavailable by tool lookup")) {
        return 1;
    }

    const char * invalid_catalog = R"({
        "catalog_version": 1,
        "capabilities": [
            {
                "capability_id": "openclaw.weather.invalid",
                "tool_surface_id": "vicuna.weather.invalid",
                "capability_kind": "tool",
                "owner_plugin_id": "openclaw-invalid",
                "tool_name": "weather_invalid",
                "tool_family_id": "weather",
                "tool_family_name": "Weather",
                "tool_family_description": "Read one synthetic weather surface.",
                "method_name": "lookup",
                "method_description": "Run one weather lookup.",
                "description": "Run an invalid weather lookup",
                "input_schema_json": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "The weather query to run." }
                    }
                },
                "output_contract": "completed_result",
                "side_effect_class": "network_read",
                "approval_mode": "policy_driven",
                "execution_modes": ["sync"],
                "provenance_namespace": "openclaw/openclaw-invalid/tool/weather_invalid",
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
        out << invalid_catalog;
    }
    server_openclaw_fabric invalid_fabric;
    error.clear();
    if (!expect(!invalid_fabric.configure(true, true, true, &error),
                "expected external catalog without react eligibility flags to be rejected")) {
        return 1;
    }
    if (!expect(error.find("no active/DMN cognitive eligibility flags") != std::string::npos,
                "expected invalid external catalog error to mention missing react eligibility flags")) {
        return 1;
    }

    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH");
    std::remove(catalog_path.c_str());
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_ENABLED");
    unsetenv("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS");
    return 0;
}
