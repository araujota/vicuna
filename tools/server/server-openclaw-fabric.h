#pragma once

#include "llama.h"

#include "common/openclaw-tool-fabric.h"
#include "common/openclaw-tool-fabric-events.h"

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

    bool enabled() const;
    const openclaw_tool_capability_catalog & catalog() const;
    const std::vector<server_openclaw_capability_runtime> & capabilities() const;

    bool build_cognitive_specs(std::vector<llama_cognitive_tool_spec> * out_specs) const;

    const server_openclaw_capability_runtime * resolve_command(
            const llama_cognitive_command & command,
            std::string * out_error = nullptr) const;

private:
    bool configured_enabled = false;
    openclaw_tool_capability_catalog catalog_state = {};
    std::vector<server_openclaw_capability_runtime> capability_state;
};
