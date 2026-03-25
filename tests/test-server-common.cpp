#include "../tools/server/server-common.h"
#include "../tools/server/server-task.h"

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
                R"({"messages":[{"role":"system","content":"Write one short acknowledgement only."}]})";
        const json parsed_body = {
            {"prompt", "ignored"},
        };
        if (!expect(classify_foreground_role_for_request(raw_body, parsed_body) == LLAMA_SELF_STATE_EVENT_SYSTEM,
                    "expected system-only requests to classify as system-origin")) {
            return 1;
        }
    }

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
        if (!expect(foreground_request_requires_fresh_tool_grounding(
                            "What TV shows do I have in Sonarr?"),
                    "expected Sonarr library request to require fresh tool grounding")) {
            return 1;
        }
        if (!expect(foreground_request_requires_fresh_tool_grounding(
                            "Show me my Chaptarr queue status."),
                    "expected Chaptarr queue request to require fresh tool grounding")) {
            return 1;
        }
        if (!expect(!foreground_request_requires_fresh_tool_grounding(
                             "Reply with exactly the single word orchid."),
                    "expected stable direct reply request to avoid forced fresh tool grounding")) {
            return 1;
        }
    }

    {
        if (!expect(foreground_request_requires_external_action(
                            "Download One Battle After Another via Radarr."),
                    "expected explicit Radarr download request to require an external action")) {
            return 1;
        }
        if (!expect(foreground_request_requires_external_action(
                            "Please send the message to Telegram once the task is done."),
                    "expected outbound contact request to require an external action")) {
            return 1;
        }
        if (!expect(!foreground_request_requires_external_action(
                             "What TV shows do I have in Sonarr?"),
                    "expected read-only Sonarr library request to avoid external-action classification")) {
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
        if (!expect(authoritative_reply_is_procedural_non_answer(
                            "I cannot interact with external surfaces for this request."),
                    "expected false external-surface disclaimer to be rejected as a non-answer")) {
            return 1;
        }
        if (!expect(authoritative_reply_is_procedural_non_answer(
                            "To view the list of shows in Sonarr, you can access the web interface or use the command line."),
                    "expected procedural Sonarr web-interface guidance to be rejected as a non-answer")) {
            return 1;
        }
        if (!expect(!authoritative_reply_is_procedural_non_answer(
                             "Chicago will be around 49 degrees Fahrenheit tomorrow with a chance of rain."),
                    "expected substantive grounded weather answer to remain acceptable")) {
            return 1;
        }
    }

    {
        if (!expect(authoritative_reply_is_future_intent_status(
                            "I will delete the series now."),
                    "expected future-intent external-action reply to be detected")) {
            return 1;
        }
        if (!expect(authoritative_reply_is_future_intent_status(
                            "You can run the delete action from Sonarr."),
                    "expected delegated future-intent reply to be detected")) {
            return 1;
        }
        if (!expect(!authoritative_reply_is_future_intent_status(
                             "The delete request has already been sent to Sonarr."),
                    "expected grounded completion status to remain acceptable")) {
            return 1;
        }
    }

    {
        if (!expect(authoritative_visible_reply_looks_like_question(
                            "Which deployment target should I use?"),
                    "expected explicit question text to classify as a question-shaped reply")) {
            return 1;
        }
        if (!expect(!authoritative_visible_reply_looks_like_question(
                             "I removed Archer from Sonarr."),
                    "expected declarative status text to remain a non-question reply")) {
            return 1;
        }
    }

    {
        if (!expect(authoritative_visible_reply_looks_like_control_json(
                            "{\"tool_family_id\":\"web_search\"}"),
                    "expected bare tool-family JSON to classify as control-shaped")) {
            return 1;
        }
        if (!expect(authoritative_visible_reply_looks_like_control_json(
                            "{\"assistant_text\":\"hello\"}"),
                    "expected bare staged response JSON to classify as control-shaped")) {
            return 1;
        }
        if (!expect(authoritative_visible_reply_looks_like_control_json(
                            "{\"action\":\"select_tool\",\"tool_family_id\":\"web_search\"}"),
                    "expected staged action JSON to classify as control-shaped")) {
            return 1;
        }
        if (!expect(!authoritative_visible_reply_looks_like_control_json(
                             "{\"city\":\"Chicago\"}"),
                    "expected ordinary JSON content to remain outside control-shape detection")) {
            return 1;
        }
    }

    {
        json normalized_payload = json::object();
        std::string normalized_visible;
        bool recovered = false;
        if (!expect(authoritative_normalize_required_action_json(
                            "{\"tool_family_id\":\"hard_memory\"}",
                            "select_tool",
                            &normalized_payload,
                            &normalized_visible,
                            &recovered),
                    "expected required-action normalization to recover a missing action field")) {
            return 1;
        }
        if (!expect(recovered &&
                    normalized_payload.value("action", std::string()) == "select_tool" &&
                    normalized_payload.value("tool_family_id", std::string()) == "hard_memory" &&
                    normalized_visible == "{\"tool_family_id\":\"hard_memory\",\"action\":\"select_tool\"}",
                    "expected required-action normalization to inject the fixed action field")) {
            return 1;
        }

        std::string error;
        if (!expect(!authoritative_normalize_required_action_json(
                             "{\"action\":\"ask\",\"assistant_text\":\"hello\"}",
                             "answer",
                             &normalized_payload,
                             &normalized_visible,
                             nullptr,
                             &error),
                    "expected required-action normalization to reject conflicting action values")) {
            return 1;
        }
        if (!expect(error.find("action=\"answer\"") != std::string::npos,
                    "expected conflicting required-action normalization error to mention the required action")) {
            return 1;
        }
    }

    {
        if (!expect(infer_authoritative_action_from_visible_surface(
                            false,
                            "Which deployment target should I use?",
                            false,
                            LLAMA_SELF_STATE_EVENT_USER) == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK,
                    "expected visible question text to infer ask for active turns")) {
            return 1;
        }
        if (!expect(infer_authoritative_action_from_visible_surface(
                            false,
                            "I removed Archer from Sonarr.",
                            false,
                            LLAMA_SELF_STATE_EVENT_USER) == LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER,
                    "expected visible declarative text to infer answer for active turns")) {
            return 1;
        }
        if (!expect(infer_authoritative_action_from_visible_surface(
                            false,
                            "",
                            true,
                            LLAMA_SELF_STATE_EVENT_USER) == LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT,
                    "expected empty visible text after a tool result to infer wait")) {
            return 1;
        }
        if (!expect(infer_authoritative_action_from_visible_surface(
                            false,
                            "",
                            false,
                            LLAMA_SELF_STATE_EVENT_USER) == LLAMA_AUTHORITATIVE_REACT_ACTION_NONE,
                    "expected empty visible text without wait permission to remain uninferred")) {
            return 1;
        }
        if (!expect(infer_authoritative_action_from_visible_surface(
                            true,
                            "Write a private reflection about the outcome.",
                            false,
                            LLAMA_SELF_STATE_EVENT_SYSTEM) == LLAMA_AUTHORITATIVE_REACT_ACTION_INTERNAL_WRITE,
                    "expected visible DMN text to infer internal_write")) {
            return 1;
        }
        if (!expect(infer_authoritative_action_from_visible_surface(
                            true,
                            "",
                            false,
                            LLAMA_SELF_STATE_EVENT_SYSTEM) == LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT,
                    "expected empty visible DMN text to infer wait")) {
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
        if (!expect(bounded_runtime_log_excerpt("short reasoning", 64) == "short reasoning",
                    "expected bounded runtime log excerpt to preserve short text")) {
            return 1;
        }
        const std::string truncated = bounded_runtime_log_excerpt(
                "abcdefghijklmnopqrstuvwxyz",
                10);
        if (!expect(truncated.find("abcdefghij") == 0 &&
                            truncated.find("[truncated]") != std::string::npos,
                    "expected bounded runtime log excerpt to append an explicit truncation marker")) {
            return 1;
        }
    }

    {
        const std::string fallback = synthesize_deferred_terminal_failure_text("Delete The Simpsons off of Sonarr.");
        if (!expect(fallback.find("could not produce a complete final status message") != std::string::npos,
                    "expected deferred terminal fallback text to explain the final-status failure")) {
            return 1;
        }
        if (!expect(fallback.find("Please try the request again if needed.") != std::string::npos,
                    "expected deferred terminal fallback text to stay user-actionable")) {
            return 1;
        }
    }

    {
        std::string reason;
        if (!expect(runtime_telegram_assistant_message_is_carryable(
                            "Not much, just here to help! What's up with you?",
                            "telegram_relay_active",
                            &reason),
                    "expected ordinary Telegram assistant text to remain carryable")) {
            return 1;
        }
        if (!expect(!runtime_telegram_assistant_message_is_carryable(
                             "I ran into a problem while working on that request and could not complete it: request (22812 tokens) exceeds the available context size (16384 tokens), try increasing it",
                             "telegram_relay_active_error",
                             &reason),
                    "expected runtime context-window boilerplate to be non-carryable")) {
            return 1;
        }
        if (!expect(reason == "runtime_error_source" || reason == "terminal_failure_boilerplate",
                    "expected runtime error carryability rejection to explain why the message was dropped")) {
            return 1;
        }
        if (!expect(!runtime_telegram_assistant_message_is_carryable(
                             "The system's response to the user's greeting \"Hello!\" was to select the \"hard_memory\" tool, which is designed for managing memory storage tasks. However, since the user's greeting didn't specify a particular request, the system should have either prompted the user for their needs or provided a general assistance question.",
                             "telegram_relay_active",
                             &reason),
                    "expected controller meta-analysis to be non-carryable runtime Telegram dialogue")) {
            return 1;
        }
        if (!expect(reason == "controller_meta_analysis",
                    "expected controller meta-analysis carryability rejection reason")) {
            return 1;
        }
        if (!expect(!runtime_telegram_assistant_message_is_carryable(
                             "{\"action\":\"select_tool\",\"tool_family_id\":\"hard_memory\"}",
                             "telegram_relay_active",
                             &reason),
                    "expected staged control JSON to be non-carryable runtime Telegram dialogue")) {
            return 1;
        }
        if (!expect(reason == "controller_markup",
                    "expected control-json carryability rejection reason")) {
            return 1;
        }
    }

    {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.has_active_trace = true;
        task.disable_authoritative_react = true;
        if (!expect(!server_task_should_prepare_authoritative_react(task),
                    "expected explicit request-level bypass to disable authoritative ReAct preparation")) {
            return 1;
        }
    }

    {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.react_same_failure_count = 2;
        task.react_last_failure_class = "terminal_policy_reject";
        task.react_last_failure_detail = "example";
        task.react_history.push_back({"emit_response", "reasoning", "Thought: answer"});
        task.add_child(100, 101);
        if (!expect(task.child_tasks.size() == 1,
                    "expected add_child to create one child task")) {
            return 1;
        }
        if (!expect(task.child_tasks.front().react_same_failure_count == 2,
                    "expected add_child to preserve same-failure retry state")) {
            return 1;
        }
        if (!expect(task.child_tasks.front().react_last_failure_class == "terminal_policy_reject",
                    "expected add_child to preserve last failure class")) {
            return 1;
        }
        if (!expect(task.child_tasks.front().react_history.size() == 1,
                    "expected add_child to preserve staged history")) {
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

#ifdef TEST_SERVER_COMMON_VOCAB_MODEL
    {
        llama_model_params mparams = llama_model_default_params();
        mparams.vocab_only = true;
        llama_model * model = llama_model_load_from_file(TEST_SERVER_COMMON_VOCAB_MODEL, mparams);
        if (!expect(model != nullptr,
                    "expected vocab-only GGUF to load for think-block token budget coverage")) {
            return 1;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!expect(vocab != nullptr,
                    "expected loaded vocab-only model to expose a vocabulary")) {
            llama_model_free(model);
            return 1;
        }

        std::string oversized;
        for (int i = 0; i < 1400; ++i) {
            if (!oversized.empty()) {
                oversized += ' ';
            }
            oversized += "token";
        }

        size_t kept_tokens = 0;
        size_t original_tokens = 0;
        bool truncated = false;
        const std::string bounded = truncate_text_to_token_budget(
                vocab,
                oversized,
                SERVER_THINK_BLOCK_TOKEN_CAP,
                &kept_tokens,
                &original_tokens,
                &truncated);
        if (!expect(truncated,
                    "expected oversized think text to require token-budget truncation")) {
            llama_model_free(model);
            return 1;
        }
        if (!expect(original_tokens > SERVER_THINK_BLOCK_TOKEN_CAP,
                    "expected oversized think text to exceed the configured token cap")) {
            llama_model_free(model);
            return 1;
        }
        if (!expect(kept_tokens == SERVER_THINK_BLOCK_TOKEN_CAP,
                    "expected truncation to keep exactly the configured token cap")) {
            llama_model_free(model);
            return 1;
        }
        if (!expect(common_tokenize(vocab, bounded, true, true).size() <= SERVER_THINK_BLOCK_TOKEN_CAP,
                    "expected bounded think text to tokenize within the configured cap")) {
            llama_model_free(model);
            return 1;
        }

        size_t short_kept_tokens = 0;
        size_t short_original_tokens = 0;
        bool short_truncated = false;
        const std::string short_text = "short hidden thought";
        const std::string short_bounded = truncate_text_to_token_budget(
                vocab,
                short_text,
                SERVER_THINK_BLOCK_TOKEN_CAP,
                &short_kept_tokens,
                &short_original_tokens,
                &short_truncated);
        if (!expect(!short_truncated,
                    "expected short think text to remain untruncated")) {
            llama_model_free(model);
            return 1;
        }
        if (!expect(short_kept_tokens == short_original_tokens,
                    "expected short think text to preserve its token count")) {
            llama_model_free(model);
            return 1;
        }
        if (!expect(short_bounded == short_text,
                    "expected short think text to round-trip without modification")) {
            llama_model_free(model);
            return 1;
        }

        llama_model_free(model);
    }
#endif

    return 0;
}
