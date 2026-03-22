#pragma once

#include "llama.h"
#include "chat.h"

#include "common/openclaw-tool-fabric.h"
#include "common/openclaw-tool-fabric-events.h"

#include <filesystem>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum server_openclaw_dispatch_backend {
    SERVER_OPENCLAW_DISPATCH_NONE = 0,
    SERVER_OPENCLAW_DISPATCH_LEGACY_BASH = 1,
    SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY = 2,
    SERVER_OPENCLAW_DISPATCH_LEGACY_CODEX = 3,
    SERVER_OPENCLAW_DISPATCH_LEGACY_TELEGRAM = 4,
};

struct server_openclaw_capability_runtime {
    openclaw_tool_capability_descriptor descriptor;
    llama_cognitive_tool_spec tool_spec = {};
    server_openclaw_dispatch_backend backend = SERVER_OPENCLAW_DISPATCH_NONE;
};

enum server_openclaw_xml_arg_type_mask : uint32_t {
    SERVER_OPENCLAW_XML_ARG_STRING  = 1u << 0,
    SERVER_OPENCLAW_XML_ARG_INTEGER = 1u << 1,
    SERVER_OPENCLAW_XML_ARG_NUMBER  = 1u << 2,
    SERVER_OPENCLAW_XML_ARG_BOOLEAN = 1u << 3,
    SERVER_OPENCLAW_XML_ARG_NULL    = 1u << 4,
    SERVER_OPENCLAW_XML_ARG_JSON    = 1u << 5,
};

struct server_openclaw_xml_arg_requirement {
    std::string name;
    std::string description;
    uint32_t allowed_types = 0;
    bool required = false;
};

struct server_openclaw_xml_tool_contract {
    std::string tool_name;
    std::string capability_id;
    std::string description;
    std::vector<server_openclaw_xml_arg_requirement> args;
};

struct server_openclaw_parsed_tool_call {
    common_chat_msg message;
    std::string visible_prefix;
    std::string xml_block;
    std::string captured_planner_reasoning;
    std::string captured_tool_xml;
    std::string captured_payload;
};

class server_openclaw_fabric {
public:
    bool configure(bool bash_enabled, bool hard_memory_enabled, bool codex_enabled, std::string * out_error = nullptr);
    bool maybe_reload(bool bash_enabled, bool hard_memory_enabled, bool codex_enabled, bool * out_reloaded = nullptr, std::string * out_error = nullptr);

    bool enabled() const;
    const openclaw_tool_capability_catalog & catalog() const;
    const std::vector<server_openclaw_capability_runtime> & capabilities() const;

    bool build_cognitive_specs(std::vector<llama_cognitive_tool_spec> * out_specs) const;
    bool build_chat_tools(
            std::vector<common_chat_tool> * out_tools,
            const std::vector<int32_t> * spec_indexes = nullptr) const;
    bool build_xml_tool_contracts(
            std::vector<server_openclaw_xml_tool_contract> * out_contracts,
            const std::vector<int32_t> * spec_indexes = nullptr,
            std::string * out_error = nullptr) const;
    bool render_tool_call_xml_guidance(
            std::string * out_guidance,
            const std::vector<int32_t> * spec_indexes = nullptr,
            std::string * out_error = nullptr) const;
    bool parse_tool_call_xml(
            const std::string & text,
            server_openclaw_parsed_tool_call * out_parsed,
            const std::vector<int32_t> * spec_indexes = nullptr,
            std::string * out_error = nullptr) const;
    bool recover_tool_call_xml(
            const std::string & text,
            server_openclaw_parsed_tool_call * out_parsed,
            const std::vector<int32_t> * spec_indexes = nullptr,
            std::string * out_error = nullptr) const;
    std::string strip_tool_call_xml_markup(const std::string & text) const;
    const server_openclaw_capability_runtime * capability_by_spec_index(int32_t spec_index) const;
    const server_openclaw_capability_runtime * capability_by_tool_name(const std::string & tool_name) const;

    const server_openclaw_capability_runtime * resolve_command(
            const llama_cognitive_command & command,
            std::string * out_error = nullptr) const;

private:
    bool rebuild_catalog(
            bool bash_enabled,
            bool hard_memory_enabled,
            bool codex_enabled,
            openclaw_tool_capability_catalog * out_catalog,
            std::vector<server_openclaw_capability_runtime> * out_capabilities,
            std::string * out_error = nullptr) const;
    bool load_external_catalog(
            const std::string & path,
            bool bash_enabled,
            bool hard_memory_enabled,
            bool codex_enabled,
            std::vector<server_openclaw_capability_runtime> * out_capabilities,
            std::string * out_error = nullptr) const;
    std::string current_external_catalog_signature() const;

    bool configured_enabled = false;
    std::string external_catalog_path;
    std::string catalog_source_signature;
    openclaw_tool_capability_catalog catalog_state = {};
    std::vector<server_openclaw_capability_runtime> capability_state;
};
