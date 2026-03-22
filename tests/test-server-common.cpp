#include "../tools/server/server-common.h"

#include <cstdio>

static bool expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        return false;
    }
    return true;
}

int main() {
    {
        const std::string raw_body =
                R"({"messages":[{"role":"assistant","content":"Old Tesla answer."},{"role":"user","content":"Ask which deployment target I want."}]})";
        const json parsed_body = {
            {"prompt", "User: What is Tesla stock price right now?\nAssistant: The current price of Tesla stock is $750.00."},
        };

        if (!expect(classify_foreground_role_for_request(raw_body, parsed_body) == LLAMA_SELF_STATE_EVENT_USER,
                    "expected raw request messages to determine the foreground role")) {
            return 1;
        }
        if (!expect(extract_foreground_message_text_for_request(raw_body, parsed_body) == "Ask which deployment target I want.",
                    "expected raw request messages to determine the foreground text")) {
            return 1;
        }
    }

    {
        const std::string raw_body = R"({"model":"vicuna-runtime"})";
        const json parsed_body = {
            {"prompt", "Summarize the latest runtime state."},
        };
        if (!expect(extract_foreground_message_text_for_request(raw_body, parsed_body) == "Summarize the latest runtime state.",
                    "expected parsed-body prompt fallback when raw messages are absent")) {
            return 1;
        }
    }

    return 0;
}
