#include "openclaw-tool-fabric-events.h"

using json = nlohmann::json;

static void set_error(std::string * out_error, const std::string & value) {
    if (out_error) {
        *out_error = value;
    }
}

bool openclaw_tool_invocation_validate(
        const openclaw_tool_invocation & invocation,
        std::string * out_error) {
    if (invocation.invocation_id.empty()) {
        set_error(out_error, "invocation_id is required");
        return false;
    }
    if (invocation.capability_id.empty()) {
        set_error(out_error, "capability_id is required");
        return false;
    }
    if (invocation.origin_phase.empty()) {
        set_error(out_error, "origin_phase is required");
        return false;
    }
    return true;
}

bool openclaw_tool_observation_validate(
        const openclaw_tool_observation & observation,
        std::string * out_error) {
    if (observation.invocation_id.empty()) {
        set_error(out_error, "invocation_id is required");
        return false;
    }
    if (observation.status < OPENCLAW_TOOL_OBSERVATION_PENDING ||
            observation.status > OPENCLAW_TOOL_OBSERVATION_TIMED_OUT) {
        set_error(out_error, "status is out of range");
        return false;
    }
    return true;
}

json openclaw_tool_invocation_to_json(const openclaw_tool_invocation & invocation) {
    return {
        {"invocation_id", invocation.invocation_id},
        {"tool_surface_id", invocation.tool_surface_id},
        {"capability_id", invocation.capability_id},
        {"vicuna_session_id", invocation.vicuna_session_id},
        {"vicuna_run_id", invocation.vicuna_run_id},
        {"origin_phase", invocation.origin_phase},
        {"arguments_json", invocation.arguments_json.empty() ? json::object() : json::parse(invocation.arguments_json)},
        {"requested_mode", invocation.requested_mode},
        {"deadline_ms", invocation.deadline_ms},
        {"provenance_request_id", invocation.provenance_request_id},
    };
}

bool openclaw_tool_invocation_from_json(
        const json & data,
        openclaw_tool_invocation * out_invocation,
        std::string * out_error) {
    if (!out_invocation || !data.is_object()) {
        set_error(out_error, "invocation must be an object");
        return false;
    }
    openclaw_tool_invocation invocation;
    invocation.invocation_id = data.value("invocation_id", "");
    invocation.tool_surface_id = data.value("tool_surface_id", "");
    invocation.capability_id = data.value("capability_id", "");
    invocation.vicuna_session_id = data.value("vicuna_session_id", "");
    invocation.vicuna_run_id = data.value("vicuna_run_id", "");
    invocation.origin_phase = data.value("origin_phase", "");
    if (data.contains("arguments_json")) {
        invocation.arguments_json = data.at("arguments_json").dump();
    }
    invocation.requested_mode = data.value("requested_mode", "");
    invocation.deadline_ms = data.value("deadline_ms", 0);
    invocation.provenance_request_id = data.value("provenance_request_id", "");
    if (!openclaw_tool_invocation_validate(invocation, out_error)) {
        return false;
    }
    *out_invocation = std::move(invocation);
    return true;
}

json openclaw_tool_observation_to_json(const openclaw_tool_observation & observation) {
    return {
        {"invocation_id", observation.invocation_id},
        {"status", observation.status},
        {"summary_text", observation.summary_text},
        {"structured_payload_json", observation.structured_payload_json.empty() ? json::object() : json::parse(observation.structured_payload_json)},
        {"runtime_metrics_json", observation.runtime_metrics_json.empty() ? json::object() : json::parse(observation.runtime_metrics_json)},
        {"observed_at_ms", observation.observed_at_ms},
    };
}

bool openclaw_tool_observation_from_json(
        const json & data,
        openclaw_tool_observation * out_observation,
        std::string * out_error) {
    if (!out_observation || !data.is_object()) {
        set_error(out_error, "observation must be an object");
        return false;
    }
    openclaw_tool_observation observation;
    observation.invocation_id = data.value("invocation_id", "");
    observation.status = data.value("status", OPENCLAW_TOOL_OBSERVATION_PENDING);
    observation.summary_text = data.value("summary_text", "");
    if (data.contains("structured_payload_json")) {
        observation.structured_payload_json = data.at("structured_payload_json").dump();
    }
    if (data.contains("runtime_metrics_json")) {
        observation.runtime_metrics_json = data.at("runtime_metrics_json").dump();
    }
    observation.observed_at_ms = data.value("observed_at_ms", 0ll);
    if (!openclaw_tool_observation_validate(observation, out_error)) {
        return false;
    }
    *out_observation = std::move(observation);
    return true;
}
