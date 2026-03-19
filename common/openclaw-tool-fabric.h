#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct openclaw_tool_capability_descriptor {
    std::string capability_id;
    std::string tool_surface_id;
    std::string capability_kind;
    std::string owner_plugin_id;
    std::string tool_name;
    std::string description;
    std::string input_schema_json;
    std::string output_contract;
    std::string side_effect_class;
    std::string approval_mode;
    std::vector<std::string> execution_modes;
    std::string provenance_namespace;
    int32_t tool_kind = 0;
    uint32_t tool_flags = 0;
    int32_t latency_class = 0;
    int32_t max_steps_reserved = 0;
    std::string dispatch_backend;
};

struct openclaw_tool_capability_catalog {
    int32_t catalog_version = 1;
    std::vector<openclaw_tool_capability_descriptor> capabilities;
};

bool openclaw_tool_capability_descriptor_validate(
        const openclaw_tool_capability_descriptor & descriptor,
        std::string * out_error = nullptr);

bool openclaw_tool_capability_catalog_validate(
        const openclaw_tool_capability_catalog & catalog,
        std::string * out_error = nullptr);

nlohmann::json openclaw_tool_capability_descriptor_to_json(
        const openclaw_tool_capability_descriptor & descriptor);

bool openclaw_tool_capability_descriptor_from_json(
        const nlohmann::json & data,
        openclaw_tool_capability_descriptor * out_descriptor,
        std::string * out_error = nullptr);

nlohmann::json openclaw_tool_capability_catalog_to_json(
        const openclaw_tool_capability_catalog & catalog);

bool openclaw_tool_capability_catalog_from_json(
        const nlohmann::json & data,
        openclaw_tool_capability_catalog * out_catalog,
        std::string * out_error = nullptr);
