#include "openclaw-session-binding.h"

using json = nlohmann::json;

static void set_error(std::string * out_error, const std::string & value) {
    if (out_error) {
        *out_error = value;
    }
}

bool openclaw_session_binding_validate(
        const openclaw_session_binding & binding,
        std::string * out_error) {
    if (binding.vicuna_session_id.empty()) {
        set_error(out_error, "vicuna_session_id is required");
        return false;
    }
    if (binding.openclaw_binding_id.empty()) {
        set_error(out_error, "openclaw_binding_id is required");
        return false;
    }
    return true;
}

json openclaw_session_binding_to_json(const openclaw_session_binding & binding) {
    return {
        {"vicuna_session_id", binding.vicuna_session_id},
        {"openclaw_binding_id", binding.openclaw_binding_id},
        {"channel_id", binding.channel_id},
        {"account_id", binding.account_id},
        {"conversation_id", binding.conversation_id},
    };
}

bool openclaw_session_binding_from_json(
        const json & data,
        openclaw_session_binding * out_binding,
        std::string * out_error) {
    if (!out_binding || !data.is_object()) {
        set_error(out_error, "binding must be an object");
        return false;
    }
    openclaw_session_binding binding;
    binding.vicuna_session_id = data.value("vicuna_session_id", "");
    binding.openclaw_binding_id = data.value("openclaw_binding_id", "");
    binding.channel_id = data.value("channel_id", "");
    binding.account_id = data.value("account_id", "");
    binding.conversation_id = data.value("conversation_id", "");
    if (!openclaw_session_binding_validate(binding, out_error)) {
        return false;
    }
    *out_binding = std::move(binding);
    return true;
}
