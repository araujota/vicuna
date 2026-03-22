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

    {
        if (!expect(foreground_request_requires_fresh_tool_grounding(
                            "What’s the temperature supposed to be like in Chicago tomorrow?"),
                    "expected tomorrow weather request to require fresh tool grounding")) {
            return 1;
        }
        if (!expect(foreground_request_requires_fresh_tool_grounding(
                            "What is Tesla stock price right now?"),
                    "expected live stock-price request to require fresh tool grounding")) {
            return 1;
        }
        if (!expect(!foreground_request_requires_fresh_tool_grounding(
                             "Reply with exactly the single word orchid."),
                    "expected stable direct reply request to avoid forced fresh tool grounding")) {
            return 1;
        }
    }

    {
        if (!expect(authoritative_reply_is_procedural_non_answer(
                            "To provide an estimate of the temperature in Chicago tomorrow, I will use historical data from previous years to inform my response."),
                    "expected procedural weather deferral to be rejected as a non-answer")) {
            return 1;
        }
        if (!expect(authoritative_reply_is_procedural_non_answer(
                            "I don't have real-time access to current temperatures, so I can't provide live data."),
                    "expected lack-of-access disclaimer to be rejected as a non-answer")) {
            return 1;
        }
        if (!expect(!authoritative_reply_is_procedural_non_answer(
                             "Chicago will be around 49 degrees Fahrenheit tomorrow with a chance of rain."),
                    "expected substantive grounded weather answer to remain acceptable")) {
            return 1;
        }
    }

    {
        if (!expect(user_facing_text_violates_plain_prose_policy(
                            "As an AI language model, I cannot do that."),
                    "expected AI-disclaimer text to violate the plain-prose policy")) {
            return 1;
        }
        if (!expect(user_facing_text_violates_plain_prose_policy(
                            "1. First option\n2. Second option"),
                    "expected numbered lists to violate the plain-prose policy")) {
            return 1;
        }
        if (!expect(user_facing_text_violates_plain_prose_policy(
                            "- first item\n- second item"),
                    "expected dashed lists to violate the plain-prose policy")) {
            return 1;
        }
        if (!expect(user_facing_text_violates_plain_prose_policy(
                            "The answer is **here**."),
                    "expected markdown emphasis to violate the plain-prose policy")) {
            return 1;
        }
        if (!expect(!user_facing_text_violates_plain_prose_policy(
                             "The present working directory is slash workspace slash vicuna."),
                    "expected plain prose to remain allowed")) {
            return 1;
        }
    }

    {
        if (!expect(!authoritative_retry_requires_tool_escalation(
                             "What’s the temperature supposed to be like in Chicago tomorrow?",
                             1),
                    "expected mutable live-fact request to avoid escalation before repeated failures")) {
            return 1;
        }
        if (!expect(authoritative_retry_requires_tool_escalation(
                            "What’s the temperature supposed to be like in Chicago tomorrow?",
                            2),
                    "expected mutable live-fact request to escalate after repeated failures")) {
            return 1;
        }
        if (!expect(!authoritative_retry_requires_tool_escalation(
                             "Reply with exactly the single word orchid.",
                             4),
                    "expected stable request to avoid tool escalation even after retries")) {
            return 1;
        }
    }

    return 0;
}
