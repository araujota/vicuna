#pragma once

#include <nlohmann/json.hpp>

#include <string>

struct openclaw_session_binding {
    std::string vicuna_session_id;
    std::string openclaw_binding_id;
    std::string channel_id;
    std::string account_id;
    std::string conversation_id;
};

bool openclaw_session_binding_validate(
        const openclaw_session_binding & binding,
        std::string * out_error = nullptr);

nlohmann::json openclaw_session_binding_to_json(const openclaw_session_binding & binding);

bool openclaw_session_binding_from_json(
        const nlohmann::json & data,
        openclaw_session_binding * out_binding,
        std::string * out_error = nullptr);
