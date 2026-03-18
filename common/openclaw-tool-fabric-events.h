#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

enum openclaw_tool_observation_status {
    OPENCLAW_TOOL_OBSERVATION_PENDING = 0,
    OPENCLAW_TOOL_OBSERVATION_AWAITING_APPROVAL = 1,
    OPENCLAW_TOOL_OBSERVATION_RUNNING = 2,
    OPENCLAW_TOOL_OBSERVATION_COMPLETED = 3,
    OPENCLAW_TOOL_OBSERVATION_FAILED = 4,
    OPENCLAW_TOOL_OBSERVATION_REJECTED = 5,
    OPENCLAW_TOOL_OBSERVATION_CANCELLED = 6,
    OPENCLAW_TOOL_OBSERVATION_TIMED_OUT = 7,
};

struct openclaw_tool_invocation {
    std::string invocation_id;
    std::string tool_surface_id;
    std::string capability_id;
    std::string vicuna_session_id;
    std::string vicuna_run_id;
    std::string origin_phase;
    std::string arguments_json;
    std::string requested_mode;
    int32_t deadline_ms = 0;
    std::string provenance_request_id;
};

struct openclaw_tool_observation {
    std::string invocation_id;
    int32_t status = OPENCLAW_TOOL_OBSERVATION_PENDING;
    std::string summary_text;
    std::string structured_payload_json;
    std::string runtime_metrics_json;
    int64_t observed_at_ms = 0;
};

bool openclaw_tool_invocation_validate(
        const openclaw_tool_invocation & invocation,
        std::string * out_error = nullptr);

bool openclaw_tool_observation_validate(
        const openclaw_tool_observation & observation,
        std::string * out_error = nullptr);

nlohmann::json openclaw_tool_invocation_to_json(const openclaw_tool_invocation & invocation);
bool openclaw_tool_invocation_from_json(
        const nlohmann::json & data,
        openclaw_tool_invocation * out_invocation,
        std::string * out_error = nullptr);

nlohmann::json openclaw_tool_observation_to_json(const openclaw_tool_observation & observation);
bool openclaw_tool_observation_from_json(
        const nlohmann::json & data,
        openclaw_tool_observation * out_observation,
        std::string * out_error = nullptr);
