#include "openclaw-tool-fabric.h"

#include <unordered_set>

using json = nlohmann::json;

static void set_error(std::string * out_error, const std::string & value) {
    if (out_error) {
        *out_error = value;
    }
}

static bool schema_node_requires_description(const json & node, bool is_root) {
    if (is_root || !node.is_object()) {
        return false;
    }
    if (node.contains("type")) {
        return true;
    }
    if (node.contains("properties") || node.contains("items")) {
        return true;
    }
    return false;
}

static bool validate_schema_descriptions(
        const json & node,
        const std::string & path,
        bool is_root,
        std::string * out_error) {
    if (!node.is_object()) {
        return true;
    }

    if (schema_node_requires_description(node, is_root)) {
        const std::string description = node.value("description", "");
        if (description.empty()) {
            set_error(out_error, "tool schema is missing a description at " + path);
            return false;
        }
    }

    if (node.contains("properties")) {
        const json & properties = node.at("properties");
        if (!properties.is_object()) {
            set_error(out_error, "tool schema properties must be an object at " + path);
            return false;
        }
        for (auto it = properties.begin(); it != properties.end(); ++it) {
            if (!validate_schema_descriptions(it.value(), path + ".properties." + it.key(), false, out_error)) {
                return false;
            }
        }
    }

    if (node.contains("items")) {
        const json & items = node.at("items");
        if (items.is_array()) {
            for (size_t i = 0; i < items.size(); ++i) {
                if (!validate_schema_descriptions(items.at(i), path + ".items[" + std::to_string(i) + "]", false, out_error)) {
                    return false;
                }
            }
        } else {
            if (!validate_schema_descriptions(items, path + ".items", false, out_error)) {
                return false;
            }
        }
    }

    return true;
}

bool openclaw_tool_capability_descriptor_validate(
        const openclaw_tool_capability_descriptor & descriptor,
        std::string * out_error) {
    if (descriptor.capability_id.empty()) {
        set_error(out_error, "capability_id is required");
        return false;
    }
    if (descriptor.tool_surface_id.empty()) {
        set_error(out_error, "tool_surface_id is required");
        return false;
    }
    if (descriptor.tool_name.empty()) {
        set_error(out_error, "tool_name is required");
        return false;
    }
    if (descriptor.provenance_namespace.empty()) {
        set_error(out_error, "provenance_namespace is required");
        return false;
    }
    if (descriptor.tool_kind <= 0) {
        set_error(out_error, "tool_kind must be positive");
        return false;
    }
    if (!descriptor.input_schema_json.empty()) {
        json schema;
        try {
            schema = json::parse(descriptor.input_schema_json);
        } catch (const std::exception & e) {
            set_error(out_error, std::string("input_schema_json must be valid JSON: ") + e.what());
            return false;
        }
        if (!validate_schema_descriptions(schema, "input_schema_json", true, out_error)) {
            return false;
        }
    }
    return true;
}

bool openclaw_tool_capability_catalog_validate(
        const openclaw_tool_capability_catalog & catalog,
        std::string * out_error) {
    if (catalog.catalog_version <= 0) {
        set_error(out_error, "catalog_version must be positive");
        return false;
    }

    std::unordered_set<std::string> ids;
    std::unordered_set<std::string> tool_surface_ids;
    std::unordered_set<std::string> namespaces;
    for (const auto & descriptor : catalog.capabilities) {
        std::string descriptor_error;
        if (!openclaw_tool_capability_descriptor_validate(descriptor, &descriptor_error)) {
            set_error(out_error, descriptor_error);
            return false;
        }
        if (!ids.insert(descriptor.capability_id).second) {
            set_error(out_error, "duplicate capability_id: " + descriptor.capability_id);
            return false;
        }
        if (!tool_surface_ids.insert(descriptor.tool_surface_id).second) {
            set_error(out_error, "duplicate tool_surface_id: " + descriptor.tool_surface_id);
            return false;
        }
        if (!namespaces.insert(descriptor.provenance_namespace).second) {
            set_error(out_error, "duplicate provenance_namespace: " + descriptor.provenance_namespace);
            return false;
        }
    }
    return true;
}

json openclaw_tool_capability_descriptor_to_json(
        const openclaw_tool_capability_descriptor & descriptor) {
    return {
        {"capability_id", descriptor.capability_id},
        {"tool_surface_id", descriptor.tool_surface_id},
        {"capability_kind", descriptor.capability_kind},
        {"owner_plugin_id", descriptor.owner_plugin_id},
        {"tool_name", descriptor.tool_name},
        {"description", descriptor.description},
        {"input_schema_json", descriptor.input_schema_json.empty() ? json::object() : json::parse(descriptor.input_schema_json)},
        {"output_contract", descriptor.output_contract},
        {"side_effect_class", descriptor.side_effect_class},
        {"approval_mode", descriptor.approval_mode},
        {"execution_modes", descriptor.execution_modes},
        {"provenance_namespace", descriptor.provenance_namespace},
        {"tool_kind", descriptor.tool_kind},
        {"tool_flags", descriptor.tool_flags},
        {"latency_class", descriptor.latency_class},
        {"max_steps_reserved", descriptor.max_steps_reserved},
        {"dispatch_backend", descriptor.dispatch_backend},
    };
}

bool openclaw_tool_capability_descriptor_from_json(
        const json & data,
        openclaw_tool_capability_descriptor * out_descriptor,
        std::string * out_error) {
    if (!out_descriptor || !data.is_object()) {
        set_error(out_error, "descriptor must be an object");
        return false;
    }

    openclaw_tool_capability_descriptor descriptor;
    descriptor.capability_id = data.value("capability_id", "");
    descriptor.tool_surface_id = data.value("tool_surface_id", "");
    descriptor.capability_kind = data.value("capability_kind", "tool");
    descriptor.owner_plugin_id = data.value("owner_plugin_id", "");
    descriptor.tool_name = data.value("tool_name", "");
    descriptor.description = data.value("description", "");
    if (data.contains("input_schema_json")) {
        descriptor.input_schema_json = data.at("input_schema_json").dump();
    }
    descriptor.output_contract = data.value("output_contract", "");
    descriptor.side_effect_class = data.value("side_effect_class", "");
    descriptor.approval_mode = data.value("approval_mode", "");
    descriptor.provenance_namespace = data.value("provenance_namespace", "");
    descriptor.tool_kind = data.value("tool_kind", 0);
    descriptor.tool_flags = data.value("tool_flags", 0u);
    descriptor.latency_class = data.value("latency_class", 0);
    descriptor.max_steps_reserved = data.value("max_steps_reserved", 0);
    descriptor.dispatch_backend = data.value("dispatch_backend", "");

    if (data.contains("execution_modes") && data.at("execution_modes").is_array()) {
        for (const auto & item : data.at("execution_modes")) {
            if (item.is_string()) {
                descriptor.execution_modes.push_back(item.get<std::string>());
            }
        }
    }

    std::string validation_error;
    if (!openclaw_tool_capability_descriptor_validate(descriptor, &validation_error)) {
        set_error(out_error, validation_error);
        return false;
    }
    *out_descriptor = std::move(descriptor);
    return true;
}

json openclaw_tool_capability_catalog_to_json(
        const openclaw_tool_capability_catalog & catalog) {
    json capabilities = json::array();
    for (const auto & descriptor : catalog.capabilities) {
        capabilities.push_back(openclaw_tool_capability_descriptor_to_json(descriptor));
    }
    return {
        {"catalog_version", catalog.catalog_version},
        {"capabilities", std::move(capabilities)},
    };
}

bool openclaw_tool_capability_catalog_from_json(
        const json & data,
        openclaw_tool_capability_catalog * out_catalog,
        std::string * out_error) {
    if (!out_catalog || !data.is_object()) {
        set_error(out_error, "catalog must be an object");
        return false;
    }

    openclaw_tool_capability_catalog catalog;
    catalog.catalog_version = data.value("catalog_version", 0);
    if (data.contains("capabilities") && data.at("capabilities").is_array()) {
        for (const auto & item : data.at("capabilities")) {
            openclaw_tool_capability_descriptor descriptor;
            if (!openclaw_tool_capability_descriptor_from_json(item, &descriptor, out_error)) {
                return false;
            }
            catalog.capabilities.push_back(std::move(descriptor));
        }
    }

    std::string validation_error;
    if (!openclaw_tool_capability_catalog_validate(catalog, &validation_error)) {
        set_error(out_error, validation_error);
        return false;
    }
    *out_catalog = std::move(catalog);
    return true;
}
