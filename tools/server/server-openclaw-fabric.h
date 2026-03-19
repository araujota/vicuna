#pragma once

#include "llama.h"
#include "chat.h"

#include "common/openclaw-tool-fabric.h"
#include "common/openclaw-tool-fabric-events.h"

#include <filesystem>
#include <string>
#include <vector>

enum server_openclaw_dispatch_backend {
    SERVER_OPENCLAW_DISPATCH_NONE = 0,
    SERVER_OPENCLAW_DISPATCH_LEGACY_BASH = 1,
    SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY = 2,
};

struct server_openclaw_capability_runtime {
    openclaw_tool_capability_descriptor descriptor;
    llama_cognitive_tool_spec tool_spec = {};
    server_openclaw_dispatch_backend backend = SERVER_OPENCLAW_DISPATCH_NONE;
};

class server_openclaw_fabric {
public:
    bool configure(bool bash_enabled, bool hard_memory_enabled, std::string * out_error = nullptr);
    bool maybe_reload(bool bash_enabled, bool hard_memory_enabled, bool * out_reloaded = nullptr, std::string * out_error = nullptr);

    bool enabled() const;
    const openclaw_tool_capability_catalog & catalog() const;
    const std::vector<server_openclaw_capability_runtime> & capabilities() const;

    bool build_cognitive_specs(std::vector<llama_cognitive_tool_spec> * out_specs) const;
    bool build_chat_tools(
            std::vector<common_chat_tool> * out_tools,
            const std::vector<int32_t> * spec_indexes = nullptr) const;
    const server_openclaw_capability_runtime * capability_by_spec_index(int32_t spec_index) const;
    const server_openclaw_capability_runtime * capability_by_tool_name(const std::string & tool_name) const;

    const server_openclaw_capability_runtime * resolve_command(
            const llama_cognitive_command & command,
            std::string * out_error = nullptr) const;

private:
    bool rebuild_catalog(
            bool bash_enabled,
            bool hard_memory_enabled,
            openclaw_tool_capability_catalog * out_catalog,
            std::vector<server_openclaw_capability_runtime> * out_capabilities,
            std::string * out_error = nullptr) const;
    bool load_external_catalog(
            const std::string & path,
            bool bash_enabled,
            bool hard_memory_enabled,
            std::vector<server_openclaw_capability_runtime> * out_capabilities,
            std::string * out_error = nullptr) const;
    std::string current_external_catalog_signature() const;

    bool configured_enabled = false;
    std::string external_catalog_path;
    std::string catalog_source_signature;
    openclaw_tool_capability_catalog catalog_state = {};
    std::vector<server_openclaw_capability_runtime> capability_state;
};
