#include "server-openclaw-fabric.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace {

constexpr const char * SERVER_OPENCLAW_XML_ROOT = "vicuna_tool_call";
constexpr const char * SERVER_OPENCLAW_XML_ARG = "arg";

struct builtin_capability_registration {
    const char * tool_id;
    server_openclaw_dispatch_backend backend;
    bool (*is_available)(bool bash_enabled, bool hard_memory_enabled, bool codex_enabled);
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

bool builtin_tool_enabled_for_env(const char * tool_id) {
    if (!tool_id) {
        return false;
    }
    if (env_list_contains("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS", tool_id, true)) {
        return true;
    }
    // Compatibility: older host-local env files often pin the builtin allowlist
    // to exec + hard_memory_query. Treat that legacy hard-memory entry as the
    // whole hard-memory tool layer so explicit durable writes are not silently
    // removed from the live OpenClaw surface.
    if (std::strcmp(tool_id, "hard_memory_write") == 0 &&
            env_list_contains("VICUNA_OPENCLAW_TOOL_FABRIC_TOOLS", "hard_memory_query", false)) {
        return true;
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

std::string trim_ascii_copy_local(const std::string & value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin]) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace((unsigned char) value[end - 1]) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

void extract_hidden_reasoning_and_visible_text(
        const std::string & text,
        std::string * out_reasoning,
        std::string * out_visible) {
    if (out_reasoning) {
        out_reasoning->clear();
    }
    if (out_visible) {
        out_visible->clear();
    }

    static const std::string think_open = "<think>";
    static const std::string think_close = "</think>";

    std::string visible;
    std::string reasoning;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t think_start = text.find(think_open, pos);
        if (think_start == std::string::npos) {
            visible += text.substr(pos);
            break;
        }

        visible += text.substr(pos, think_start - pos);
        const size_t body_start = think_start + think_open.size();
        const size_t think_end = text.find(think_close, body_start);
        if (think_end == std::string::npos) {
            visible += text.substr(think_start);
            break;
        }

        const std::string chunk = trim_ascii_copy_local(text.substr(body_start, think_end - body_start));
        if (!chunk.empty()) {
            if (!reasoning.empty()) {
                reasoning += "\n\n";
            }
            reasoning += chunk;
        }
        pos = think_end + think_close.size();
    }

    visible = trim_ascii_copy_local(visible);
    reasoning = trim_ascii_copy_local(reasoning);
    if (reasoning.empty() && !visible.empty()) {
        reasoning = visible;
        visible.clear();
    }

    if (out_reasoning) {
        *out_reasoning = std::move(reasoning);
    }
    if (out_visible) {
        *out_visible = std::move(visible);
    }
}

std::string strip_hidden_reasoning_markup_preserve_text(const std::string & text) {
    std::string reasoning;
    std::string visible;
    extract_hidden_reasoning_and_visible_text(text, &reasoning, &visible);
    if (reasoning.empty()) {
        return visible;
    }
    if (visible.empty()) {
        return reasoning;
    }
    return reasoning + "\n" + visible;
}

bool skip_ws(const std::string & text, size_t * pos) {
    if (!pos) {
        return false;
    }
    while (*pos < text.size() && std::isspace((unsigned char) text[*pos]) != 0) {
        ++(*pos);
    }
    return true;
}

bool parse_xml_name(const std::string & text, size_t * pos, std::string * out_name) {
    if (!pos || !out_name || *pos >= text.size()) {
        return false;
    }
    const size_t begin = *pos;
    while (*pos < text.size()) {
        const unsigned char ch = (unsigned char) text[*pos];
        if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == ':' || ch == '.') {
            ++(*pos);
            continue;
        }
        break;
    }
    if (*pos == begin) {
        return false;
    }
    *out_name = text.substr(begin, *pos - begin);
    return true;
}

bool parse_xml_quoted_value(const std::string & text, size_t * pos, std::string * out_value) {
    if (!pos || !out_value || *pos >= text.size()) {
        return false;
    }
    const char quote = text[*pos];
    if (quote != '"' && quote != '\'') {
        return false;
    }
    ++(*pos);
    const size_t begin = *pos;
    while (*pos < text.size() && text[*pos] != quote) {
        ++(*pos);
    }
    if (*pos >= text.size()) {
        return false;
    }
    *out_value = text.substr(begin, *pos - begin);
    ++(*pos);
    return true;
}

bool parse_xml_start_tag(
        const std::string & text,
        size_t * pos,
        std::string * out_tag_name,
        std::map<std::string, std::string> * out_attrs) {
    if (!pos || !out_tag_name || !out_attrs || *pos >= text.size() || text[*pos] != '<') {
        return false;
    }
    ++(*pos);
    if (!parse_xml_name(text, pos, out_tag_name)) {
        return false;
    }
    out_attrs->clear();
    while (*pos < text.size()) {
        skip_ws(text, pos);
        if (*pos >= text.size()) {
            return false;
        }
        if (text[*pos] == '>') {
            ++(*pos);
            return true;
        }
        if (text[*pos] == '/' && *pos + 1 < text.size() && text[*pos + 1] == '>') {
            return false;
        }
        std::string attr_name;
        if (!parse_xml_name(text, pos, &attr_name)) {
            return false;
        }
        skip_ws(text, pos);
        if (*pos >= text.size() || text[*pos] != '=') {
            return false;
        }
        ++(*pos);
        skip_ws(text, pos);
        std::string attr_value;
        if (!parse_xml_quoted_value(text, pos, &attr_value)) {
            return false;
        }
        (*out_attrs)[attr_name] = attr_value;
    }
    return false;
}

bool parse_xml_end_tag(const std::string & text, size_t * pos, const std::string & expected_name) {
    if (!pos || *pos >= text.size() || text[*pos] != '<' || *pos + 1 >= text.size() || text[*pos + 1] != '/') {
        return false;
    }
    *pos += 2;
    std::string tag_name;
    if (!parse_xml_name(text, pos, &tag_name)) {
        return false;
    }
    skip_ws(text, pos);
    if (*pos >= text.size() || text[*pos] != '>' || tag_name != expected_name) {
        return false;
    }
    ++(*pos);
    return true;
}

bool decode_xml_entities(const std::string & value, std::string * out_decoded) {
    if (!out_decoded) {
        return false;
    }
    out_decoded->clear();
    out_decoded->reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '&') {
            out_decoded->push_back(value[i]);
            continue;
        }
        if (value.compare(i, 4, "&lt;") == 0) {
            out_decoded->push_back('<');
            i += 3;
        } else if (value.compare(i, 4, "&gt;") == 0) {
            out_decoded->push_back('>');
            i += 3;
        } else if (value.compare(i, 5, "&amp;") == 0) {
            out_decoded->push_back('&');
            i += 4;
        } else if (value.compare(i, 6, "&quot;") == 0) {
            out_decoded->push_back('"');
            i += 5;
        } else if (value.compare(i, 6, "&apos;") == 0) {
            out_decoded->push_back('\'');
            i += 5;
        } else {
            return false;
        }
    }
    return true;
}

uint32_t xml_type_mask_from_schema_type(const nlohmann::json & type_json) {
    auto accumulate_type = [](const std::string & type_name, uint32_t * io_mask) {
        if (!io_mask) {
            return;
        }
        if (type_name == "string") {
            *io_mask |= SERVER_OPENCLAW_XML_ARG_STRING;
        } else if (type_name == "integer") {
            *io_mask |= SERVER_OPENCLAW_XML_ARG_INTEGER;
        } else if (type_name == "number") {
            *io_mask |= SERVER_OPENCLAW_XML_ARG_NUMBER;
        } else if (type_name == "boolean") {
            *io_mask |= SERVER_OPENCLAW_XML_ARG_BOOLEAN;
        } else if (type_name == "null") {
            *io_mask |= SERVER_OPENCLAW_XML_ARG_NULL;
        } else if (type_name == "object" || type_name == "array") {
            *io_mask |= SERVER_OPENCLAW_XML_ARG_JSON;
        }
    };

    uint32_t mask = 0;
    if (type_json.is_string()) {
        accumulate_type(type_json.get<std::string>(), &mask);
    } else if (type_json.is_array()) {
        for (const auto & entry : type_json) {
            if (entry.is_string()) {
                accumulate_type(entry.get<std::string>(), &mask);
            }
        }
    }
    return mask;
}

std::string xml_type_label(uint32_t mask) {
    std::vector<std::string> labels;
    if ((mask & SERVER_OPENCLAW_XML_ARG_STRING) != 0) {
        labels.push_back("string");
    }
    if ((mask & SERVER_OPENCLAW_XML_ARG_INTEGER) != 0) {
        labels.push_back("integer");
    }
    if ((mask & SERVER_OPENCLAW_XML_ARG_NUMBER) != 0) {
        labels.push_back("number");
    }
    if ((mask & SERVER_OPENCLAW_XML_ARG_BOOLEAN) != 0) {
        labels.push_back("boolean");
    }
    if ((mask & SERVER_OPENCLAW_XML_ARG_NULL) != 0) {
        labels.push_back("null");
    }
    if ((mask & SERVER_OPENCLAW_XML_ARG_JSON) != 0) {
        labels.push_back("json");
    }
    std::ostringstream out;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i > 0) {
            out << "|";
        }
        out << labels[i];
    }
    return out.str();
}

uint32_t xml_type_mask_from_label(const std::string & value) {
    if (value == "string") {
        return SERVER_OPENCLAW_XML_ARG_STRING;
    }
    if (value == "integer") {
        return SERVER_OPENCLAW_XML_ARG_INTEGER;
    }
    if (value == "number") {
        return SERVER_OPENCLAW_XML_ARG_NUMBER;
    }
    if (value == "boolean") {
        return SERVER_OPENCLAW_XML_ARG_BOOLEAN;
    }
    if (value == "null") {
        return SERVER_OPENCLAW_XML_ARG_NULL;
    }
    if (value == "json") {
        return SERVER_OPENCLAW_XML_ARG_JSON;
    }
    return 0;
}

bool build_xml_tool_contract_from_capability(
        const server_openclaw_capability_runtime & capability,
        server_openclaw_xml_tool_contract * out_contract,
        std::string * out_error) {
    if (!out_contract) {
        set_error(out_error, "xml contract output is required");
        return false;
    }

    nlohmann::json schema;
    try {
        schema = nlohmann::json::parse(capability.descriptor.input_schema_json);
    } catch (const std::exception & err) {
        set_error(out_error, std::string("failed to parse tool schema: ") + err.what());
        return false;
    }
    if (!schema.is_object() || schema.value("type", "") != "object") {
        set_error(out_error, "tool schema must be a top-level object");
        return false;
    }

    out_contract->tool_name = capability.descriptor.tool_name;
    out_contract->capability_id = capability.descriptor.capability_id;
    out_contract->description = capability.descriptor.description;
    out_contract->args.clear();

    std::set<std::string> required_args;
    if (schema.contains("required") && schema.at("required").is_array()) {
        for (const auto & entry : schema.at("required")) {
            if (entry.is_string()) {
                required_args.insert(entry.get<std::string>());
            }
        }
    }

    const nlohmann::json properties = schema.value("properties", nlohmann::json::object());
    if (!properties.is_object()) {
        set_error(out_error, "tool schema properties must be an object");
        return false;
    }

    for (auto it = properties.begin(); it != properties.end(); ++it) {
        server_openclaw_xml_arg_requirement arg;
        arg.name = it.key();
        arg.required = required_args.count(arg.name) > 0;
        arg.allowed_types = xml_type_mask_from_schema_type(it.value().value("type", nlohmann::json()));
        if (arg.allowed_types == 0) {
            set_error(out_error, "tool schema property \"" + arg.name + "\" has no supported XML type");
            return false;
        }
        out_contract->args.push_back(std::move(arg));
    }

    return true;
}

bool append_contracts_for_indexes(
        const std::vector<server_openclaw_capability_runtime> & capabilities,
        const std::vector<int32_t> * spec_indexes,
        std::vector<server_openclaw_xml_tool_contract> * out_contracts,
        std::string * out_error) {
    if (!out_contracts) {
        set_error(out_error, "xml contracts output is required");
        return false;
    }
    out_contracts->clear();

    auto append_contract = [&](const server_openclaw_capability_runtime & capability) -> bool {
        server_openclaw_xml_tool_contract contract;
        if (!build_xml_tool_contract_from_capability(capability, &contract, out_error)) {
            return false;
        }
        out_contracts->push_back(std::move(contract));
        return true;
    };

    if (!spec_indexes || spec_indexes->empty()) {
        out_contracts->reserve(capabilities.size());
        for (const auto & capability : capabilities) {
            if (!append_contract(capability)) {
                return false;
            }
        }
        return true;
    }

    out_contracts->reserve(spec_indexes->size());
    for (int32_t spec_index : *spec_indexes) {
        if (spec_index < 0 || spec_index >= (int32_t) capabilities.size()) {
            set_error(out_error, "tool spec index is out of range for XML contract");
            return false;
        }
        if (!append_contract(capabilities[(size_t) spec_index])) {
            return false;
        }
    }
    return true;
}

bool parse_integer_exact(const std::string & value, int64_t * out_value) {
    if (!out_value) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0') {
        return false;
    }
    *out_value = (int64_t) parsed;
    return true;
}

bool parse_number_exact(const std::string & value, double * out_value) {
    if (!out_value) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno != 0 || !end || *end != '\0') {
        return false;
    }
    *out_value = parsed;
    return true;
}

bool parse_arg_value_json(
        const std::string & raw_value,
        uint32_t declared_type,
        nlohmann::json * out_json,
        std::string * out_error) {
    if (!out_json) {
        set_error(out_error, "argument json output is required");
        return false;
    }

    if (declared_type == SERVER_OPENCLAW_XML_ARG_STRING) {
        std::string decoded;
        if (!decode_xml_entities(raw_value, &decoded)) {
            set_error(out_error, "failed to decode XML entities for string argument");
            return false;
        }
        *out_json = decoded;
        return true;
    }
    if (declared_type == SERVER_OPENCLAW_XML_ARG_INTEGER) {
        int64_t parsed = 0;
        if (!parse_integer_exact(trim_ascii_copy_local(raw_value), &parsed)) {
            set_error(out_error, "invalid integer argument");
            return false;
        }
        *out_json = parsed;
        return true;
    }
    if (declared_type == SERVER_OPENCLAW_XML_ARG_NUMBER) {
        double parsed = 0.0;
        if (!parse_number_exact(trim_ascii_copy_local(raw_value), &parsed)) {
            set_error(out_error, "invalid number argument");
            return false;
        }
        *out_json = parsed;
        return true;
    }
    if (declared_type == SERVER_OPENCLAW_XML_ARG_BOOLEAN) {
        const std::string normalized = trim_ascii_copy_local(raw_value);
        if (normalized == "true") {
            *out_json = true;
            return true;
        }
        if (normalized == "false") {
            *out_json = false;
            return true;
        }
        set_error(out_error, "invalid boolean argument");
        return false;
    }
    if (declared_type == SERVER_OPENCLAW_XML_ARG_NULL) {
        if (trim_ascii_copy_local(raw_value) != "null") {
            set_error(out_error, "null arguments must contain the literal null");
            return false;
        }
        *out_json = nullptr;
        return true;
    }
    if (declared_type == SERVER_OPENCLAW_XML_ARG_JSON) {
        try {
            *out_json = nlohmann::json::parse(raw_value);
        } catch (const std::exception & err) {
            set_error(out_error, std::string("invalid json argument: ") + err.what());
            return false;
        }
        if (!out_json->is_object() && !out_json->is_array()) {
            set_error(out_error, "json arguments must parse to an object or array");
            return false;
        }
        return true;
    }

    set_error(out_error, "unsupported declared XML argument type");
    return false;
}

std::string join_payload(const std::string & prefix, const std::string & xml_block) {
    const std::string trimmed_prefix = trim_ascii_copy_local(prefix);
    if (trimmed_prefix.empty()) {
        return xml_block;
    }
    return trimmed_prefix + "\n" + xml_block;
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
    if (value == "legacy_codex") {
        if (out_backend) {
            *out_backend = SERVER_OPENCLAW_DISPATCH_LEGACY_CODEX;
        }
        return true;
    }
    set_error(out_error, "unsupported dispatch_backend: " + value);
    return false;
}

bool exec_available(bool bash_enabled, bool /*hard_memory_enabled*/, bool /*codex_enabled*/) {
    return bash_enabled;
}

bool hard_memory_available(bool /*bash_enabled*/, bool hard_memory_enabled, bool /*codex_enabled*/) {
    return hard_memory_enabled;
}

bool codex_available(bool /*bash_enabled*/, bool /*hard_memory_enabled*/, bool codex_enabled) {
    return codex_enabled;
}

openclaw_tool_capability_descriptor build_exec_descriptor() {
    openclaw_tool_capability_descriptor exec = {};
    exec.capability_id = "openclaw.exec.command";
    exec.tool_surface_id = "vicuna.exec.main";
    exec.capability_kind = "tool";
    exec.owner_plugin_id = "openclaw-core";
    exec.tool_name = "exec";
    exec.description = "Run one bounded command invocation through the execution policy; do not use shell chaining or redirection";
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

openclaw_tool_capability_descriptor build_hard_memory_write_descriptor() {
    openclaw_tool_capability_descriptor memory = {};
    memory.capability_id = "openclaw.vicuna.hard_memory_write";
    memory.tool_surface_id = "vicuna.memory.hard_write";
    memory.capability_kind = "memory_adapter";
    memory.owner_plugin_id = "vicuna-memory";
    memory.tool_name = "hard_memory_write";
    memory.description = "Archive explicit durable memories to Vicuña hard memory and Supermemory";
    memory.input_schema_json = R"({
        "type":"object",
        "required":["memories"],
        "properties":{
            "memories":{
                "type":"array",
                "minItems":1,
                "items":{
                    "type":"object",
                    "required":["content"],
                    "properties":{
                        "content":{"type":"string"},
                        "title":{"type":"string"},
                        "key":{"type":"string"},
                        "kind":{"type":"string"},
                        "domain":{"type":"string"},
                        "tags":{"type":"array","items":{"type":"string"}},
                        "importance":{"type":"number"},
                        "confidence":{"type":"number"},
                        "gainBias":{"type":"number"},
                        "allostaticRelevance":{"type":"number"},
                        "isStatic":{"type":"boolean"}
                    }
                }
            },
            "containerTag":{"type":"string"}
        }
    })";
    memory.output_contract = "completed_result";
    memory.side_effect_class = "memory_write";
    memory.approval_mode = "none";
    memory.execution_modes = {"sync"};
    memory.provenance_namespace = "openclaw/vicuna-memory/memory_adapter/hard_memory_write";
    memory.tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_WRITE;
    memory.tool_flags =
            LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
            LLAMA_COG_TOOL_DMN_ELIGIBLE |
            LLAMA_COG_TOOL_REMEDIATION_SAFE |
            LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT;
    memory.latency_class = LLAMA_COG_TOOL_LATENCY_HIGH;
    memory.max_steps_reserved = 3;
    memory.dispatch_backend = "legacy_hard_memory";
    return memory;
}

openclaw_tool_capability_descriptor build_codex_descriptor() {
    openclaw_tool_capability_descriptor codex = {};
    codex.capability_id = "openclaw.vicuna.codex_cli";
    codex.tool_surface_id = "vicuna.codex.main";
    codex.capability_kind = "tool";
    codex.owner_plugin_id = "vicuna-runtime";
    codex.tool_name = "codex";
    codex.description = "Use the local Codex CLI to implement a repository change and rebuild the runtime";
    codex.input_schema_json = R"({"type":"object","required":["task"],"properties":{"task":{"type":"string"}}})";
    codex.output_contract = "pending_then_result";
    codex.side_effect_class = "self_modification";
    codex.approval_mode = "none";
    codex.execution_modes = {"background"};
    codex.provenance_namespace = "openclaw/vicuna-runtime/tool/codex";
    codex.tool_kind = LLAMA_TOOL_KIND_CODEX_CLI;
    codex.tool_flags =
            LLAMA_COG_TOOL_ACTIVE_ELIGIBLE |
            LLAMA_COG_TOOL_DMN_ELIGIBLE |
            LLAMA_COG_TOOL_REMEDIATION_SAFE |
            LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT;
    codex.latency_class = LLAMA_COG_TOOL_LATENCY_HIGH;
    codex.max_steps_reserved = 3;
    codex.dispatch_backend = "legacy_codex";
    return codex;
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
        {
            "hard_memory_write",
            SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY,
            hard_memory_available,
            build_hard_memory_write_descriptor,
        },
        {
            "codex",
            SERVER_OPENCLAW_DISPATCH_LEGACY_CODEX,
            codex_available,
            build_codex_descriptor,
        },
    };
    if (out_count) {
        *out_count = sizeof(registrations) / sizeof(registrations[0]);
    }
    return registrations;
}

}  // namespace

bool server_openclaw_fabric::configure(bool bash_enabled, bool hard_memory_enabled, bool codex_enabled, std::string * out_error) {
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
                codex_enabled,
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
        bool codex_enabled,
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
                codex_enabled,
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

bool server_openclaw_fabric::build_xml_tool_contracts(
        std::vector<server_openclaw_xml_tool_contract> * out_contracts,
        const std::vector<int32_t> * spec_indexes,
        std::string * out_error) const {
    return append_contracts_for_indexes(capability_state, spec_indexes, out_contracts, out_error);
}

bool server_openclaw_fabric::render_tool_call_xml_guidance(
        std::string * out_guidance,
        const std::vector<int32_t> * spec_indexes,
        std::string * out_error) const {
    if (!out_guidance) {
        set_error(out_error, "tool-call XML guidance output is required");
        return false;
    }

    std::vector<server_openclaw_xml_tool_contract> contracts;
    if (!build_xml_tool_contracts(&contracts, spec_indexes, out_error)) {
        return false;
    }
    if (contracts.empty()) {
        set_error(out_error, "tool-call XML guidance requires at least one tool");
        return false;
    }

    std::ostringstream out;
    out << "Tool-call output contract for this step:\n";
    out << "- Emit at most one short visible sentence before the XML block.\n";
    out << "- Then emit exactly one <" << SERVER_OPENCLAW_XML_ROOT << "> block and nothing after it.\n";
    out << "- Do not use JSON tool calls, markdown code fences, or alternate tool markup.\n";
    out << "- Each argument must be an <" << SERVER_OPENCLAW_XML_ARG << "> tag with quoted name and type attributes.\n";
    out << "- Allowed argument types are string, integer, number, boolean, null, and json.\n";
    out << "- String arguments must XML-escape special characters.\n";
    out << "- json arguments must contain valid JSON object or array text.\n";
    out << "Allowed tools for this step:\n";

    for (const auto & contract : contracts) {
        out << "- tool=\"" << contract.tool_name << "\"";
        if (!contract.capability_id.empty()) {
            out << " capability=\"" << contract.capability_id << "\"";
        }
        if (!contract.description.empty()) {
            out << " description=\"" << contract.description << "\"";
        }
        out << "\n";
        for (const auto & arg : contract.args) {
        out << "  - " << arg.name << ": " << xml_type_label(arg.allowed_types);
        if (arg.required) {
            out << " (required)";
        }
        if (contract.tool_name == "exec" && arg.name == "command") {
            out << " [single command only; no pipes, redirects, chaining, or substitution]";
        }
        out << "\n";
    }
    }

    const auto & example = contracts.front();
    out << "Canonical example:\n";
    out << "<" << SERVER_OPENCLAW_XML_ROOT << " tool=\"" << example.tool_name << "\">\n";
    for (const auto & arg : example.args) {
        out << "  <" << SERVER_OPENCLAW_XML_ARG << " name=\"" << arg.name << "\" type=\"";
        if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_STRING) != 0) {
            out << "string";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_INTEGER) != 0) {
            out << "integer";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_NUMBER) != 0) {
            out << "number";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_BOOLEAN) != 0) {
            out << "boolean";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_JSON) != 0) {
            out << "json";
        } else {
            out << "null";
        }
        out << "\">";
        if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_STRING) != 0) {
            out << "example";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_INTEGER) != 0) {
            out << "1";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_NUMBER) != 0) {
            out << "1.0";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_BOOLEAN) != 0) {
            out << "true";
        } else if ((arg.allowed_types & SERVER_OPENCLAW_XML_ARG_JSON) != 0) {
            out << "{\"key\":\"value\"}";
        } else {
            out << "null";
        }
        out << "</" << SERVER_OPENCLAW_XML_ARG << ">\n";
    }
    out << "</" << SERVER_OPENCLAW_XML_ROOT << ">";

    *out_guidance = out.str();
    return true;
}

bool server_openclaw_fabric::parse_tool_call_xml(
        const std::string & text,
        server_openclaw_parsed_tool_call * out_parsed,
        const std::vector<int32_t> * spec_indexes,
        std::string * out_error) const {
    if (!out_parsed) {
        set_error(out_error, "tool-call XML parse requires output storage");
        return false;
    }

    std::vector<server_openclaw_xml_tool_contract> contracts;
    if (!build_xml_tool_contracts(&contracts, spec_indexes, out_error)) {
        return false;
    }
    if (contracts.empty()) {
        set_error(out_error, "tool-call XML parse requires at least one available contract");
        return false;
    }

    const size_t root_start = text.find(std::string("<") + SERVER_OPENCLAW_XML_ROOT);
    if (root_start == std::string::npos) {
        set_error(out_error, "tool-call XML root was not found");
        return false;
    }
    const size_t root_end = text.rfind(std::string("</") + SERVER_OPENCLAW_XML_ROOT + ">");
    if (root_end == std::string::npos || root_end < root_start) {
        set_error(out_error, "tool-call XML closing tag was not found");
        return false;
    }
    const size_t root_close_end = root_end + std::strlen(SERVER_OPENCLAW_XML_ROOT) + 3;
    const std::string raw_prefix = trim_ascii_copy_local(text.substr(0, root_start));
    const std::string xml_block = text.substr(root_start, root_close_end - root_start);
    if (!trim_ascii_copy_local(text.substr(root_close_end)).empty()) {
        set_error(out_error, "tool-call XML must not have trailing content after the closing tag");
        return false;
    }

    size_t pos = 0;
    std::string tag_name;
    std::map<std::string, std::string> attrs;
    if (!parse_xml_start_tag(xml_block, &pos, &tag_name, &attrs) || tag_name != SERVER_OPENCLAW_XML_ROOT) {
        set_error(out_error, "invalid tool-call XML root tag");
        return false;
    }
    const auto root_tool_it = attrs.find("tool");
    if (root_tool_it == attrs.end() || trim_ascii_copy_local(root_tool_it->second).empty()) {
        set_error(out_error, "tool-call XML root is missing a quoted tool attribute");
        return false;
    }
    const std::string tool_name = trim_ascii_copy_local(root_tool_it->second);

    const server_openclaw_xml_tool_contract * contract = nullptr;
    for (const auto & candidate : contracts) {
        if (candidate.tool_name == tool_name) {
            contract = &candidate;
            break;
        }
    }
    if (!contract) {
        set_error(out_error, "tool-call XML selected an unknown or disallowed tool: " + tool_name);
        return false;
    }

    std::map<std::string, server_openclaw_xml_arg_requirement> arg_requirements;
    for (const auto & arg : contract->args) {
        arg_requirements[arg.name] = arg;
    }

    nlohmann::json arguments = nlohmann::json::object();
    std::set<std::string> seen_args;

    while (true) {
        skip_ws(xml_block, &pos);
        if (pos >= xml_block.size()) {
            set_error(out_error, "unterminated tool-call XML");
            return false;
        }
        if (xml_block.compare(pos, std::strlen("</") + std::strlen(SERVER_OPENCLAW_XML_ROOT), std::string("</") + SERVER_OPENCLAW_XML_ROOT) == 0) {
            break;
        }

        std::string arg_tag_name;
        std::map<std::string, std::string> arg_attrs;
        if (!parse_xml_start_tag(xml_block, &pos, &arg_tag_name, &arg_attrs) || arg_tag_name != SERVER_OPENCLAW_XML_ARG) {
            set_error(out_error, "tool-call XML may contain only arg children inside the root");
            return false;
        }

        const auto arg_name_it = arg_attrs.find("name");
        const auto arg_type_it = arg_attrs.find("type");
        if (arg_name_it == arg_attrs.end() || arg_type_it == arg_attrs.end()) {
            set_error(out_error, "each arg element requires quoted name and type attributes");
            return false;
        }
        const std::string arg_name = trim_ascii_copy_local(arg_name_it->second);
        const std::string arg_type_label = trim_ascii_copy_local(arg_type_it->second);
        if (arg_name.empty()) {
            set_error(out_error, "arg name must not be empty");
            return false;
        }
        if (seen_args.count(arg_name) > 0) {
            set_error(out_error, "duplicate argument in tool-call XML: " + arg_name);
            return false;
        }
        const auto requirement_it = arg_requirements.find(arg_name);
        if (requirement_it == arg_requirements.end()) {
            set_error(out_error, "tool-call XML used an undeclared argument: " + arg_name);
            return false;
        }

        const uint32_t declared_type = xml_type_mask_from_label(arg_type_label);
        if (declared_type == 0) {
            set_error(out_error, "tool-call XML used an unsupported arg type: " + arg_type_label);
            return false;
        }
        if ((requirement_it->second.allowed_types & declared_type) == 0) {
            set_error(out_error, "tool-call XML type mismatch for argument: " + arg_name);
            return false;
        }

        const size_t value_start = pos;
        const size_t value_end = xml_block.find(std::string("</") + SERVER_OPENCLAW_XML_ARG + ">", pos);
        if (value_end == std::string::npos) {
            set_error(out_error, "arg element is missing a closing tag");
            return false;
        }
        const std::string raw_value = xml_block.substr(value_start, value_end - value_start);
        if (declared_type != SERVER_OPENCLAW_XML_ARG_JSON && raw_value.find('<') != std::string::npos) {
            set_error(out_error, "arg payload contains nested markup");
            return false;
        }
        pos = value_end;
        if (!parse_xml_end_tag(xml_block, &pos, SERVER_OPENCLAW_XML_ARG)) {
            set_error(out_error, "invalid arg closing tag");
            return false;
        }

        nlohmann::json parsed_value;
        if (!parse_arg_value_json(raw_value, declared_type, &parsed_value, out_error)) {
            return false;
        }
        arguments[arg_name] = std::move(parsed_value);
        seen_args.insert(arg_name);
    }

    if (!parse_xml_end_tag(xml_block, &pos, SERVER_OPENCLAW_XML_ROOT)) {
        set_error(out_error, "invalid tool-call XML root closing tag");
        return false;
    }
    skip_ws(xml_block, &pos);
    if (pos != xml_block.size()) {
        set_error(out_error, "tool-call XML contains trailing bytes after the root close");
        return false;
    }

    for (const auto & arg : contract->args) {
        if (arg.required && seen_args.count(arg.name) == 0) {
            set_error(out_error, "tool-call XML is missing required argument: " + arg.name);
            return false;
        }
    }

    out_parsed->message = {};
    out_parsed->message.role = "assistant";
    std::string planner_reasoning;
    std::string visible_prefix;
    extract_hidden_reasoning_and_visible_text(raw_prefix, &planner_reasoning, &visible_prefix);
    out_parsed->message.content = visible_prefix;
    out_parsed->message.reasoning_content = planner_reasoning;
    out_parsed->message.tool_calls = { common_chat_tool_call { tool_name, arguments.dump(), "" } };
    out_parsed->visible_prefix = visible_prefix;
    out_parsed->xml_block = xml_block;
    out_parsed->captured_planner_reasoning = planner_reasoning;
    out_parsed->captured_tool_xml = xml_block;
    out_parsed->captured_payload = join_payload(raw_prefix, xml_block);
    return true;
}

std::string server_openclaw_fabric::strip_tool_call_xml_markup(const std::string & text) const {
    const std::string root_open = std::string("<") + SERVER_OPENCLAW_XML_ROOT;
    const std::string root_close = std::string("</") + SERVER_OPENCLAW_XML_ROOT + ">";
    size_t start = text.find(root_open);
    if (start == std::string::npos) {
        return strip_hidden_reasoning_markup_preserve_text(text);
    }
    size_t end = text.find(root_close, start);
    if (end == std::string::npos) {
        return strip_hidden_reasoning_markup_preserve_text(text.substr(0, start));
    }
    end += root_close.size();
    const std::string prefix = strip_hidden_reasoning_markup_preserve_text(text.substr(0, start));
    const std::string suffix = strip_hidden_reasoning_markup_preserve_text(text.substr(end));
    if (prefix.empty()) {
        return suffix;
    }
    if (suffix.empty()) {
        return prefix;
    }
    return prefix + "\n" + suffix;
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
        bool codex_enabled,
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
        if (!builtin_tool_enabled_for_env(registration.tool_id)) {
            continue;
        }
        if (!registration.is_available(bash_enabled, hard_memory_enabled, codex_enabled)) {
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
                    codex_enabled,
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
        bool codex_enabled,
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
            if (backend == SERVER_OPENCLAW_DISPATCH_LEGACY_CODEX && !codex_enabled) {
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
