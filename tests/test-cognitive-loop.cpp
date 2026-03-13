#include "get-model.h"
#include "llama.h"

#include <array>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static float g_contradiction_score = 0.0f;
static float g_uncertainty_score = 0.0f;
static float g_broadcast_score = 0.0f;

static std::vector<llama_token> tokenize_or_die(const llama_vocab * vocab, const std::string & text) {
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize cognitive-loop prompt\n");
        std::exit(1);
    }
    return tokens;
}

static bool contradiction_head_callback(
        const llama_self_state_feature_vector * /*features*/,
        float * out_score,
        void * /*user_data*/) {
    *out_score = g_contradiction_score;
    return true;
}

static bool uncertainty_head_callback(
        const llama_self_state_feature_vector * /*features*/,
        float * out_score,
        void * /*user_data*/) {
    *out_score = g_uncertainty_score;
    return true;
}

static bool broadcast_head_callback(
        const llama_self_state_feature_vector * /*features*/,
        float * out_score,
        void * /*user_data*/) {
    *out_score = g_broadcast_score;
    return true;
}

static void configure_small_buckets(llama_past_lora_params & params) {
    params.enabled = true;

    const std::array<uint64_t, LLAMA_MEMORY_LORA_BUCKET_COUNT> periods = { 100, 200, 300, 400, 0 };
    const std::array<uint64_t, LLAMA_MEMORY_LORA_BUCKET_COUNT> half_lives = { 500, 600, 700, 800, 900 };
    const std::array<float, LLAMA_MEMORY_LORA_BUCKET_COUNT> base_scales = { 0.90f, 0.70f, 0.50f, 0.30f, 0.10f };

    for (int i = 0; i < LLAMA_MEMORY_LORA_BUCKET_COUNT; ++i) {
        params.host_memory_ratio[i] = 1.0f;
        params.device_memory_ratio[i] = 1.0f;
        params.min_rank[i] = 1;
        params.max_rank[i] = 2;
        params.condensation_period_us[i] = periods[i];
        params.decay_half_life_us[i] = half_lives[i];
        params.base_scale[i] = base_scales[i];
    }
}

static llama_context * create_context(llama_model * model) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        return nullptr;
    }

    llama_self_state_params params = llama_self_state_default_params();
    params.enable_learned_contradiction_head = true;
    params.enable_learned_uncertainty_head = true;
    params.enable_learned_broadcast_head = true;
    params.contradiction_head_callback = contradiction_head_callback;
    params.uncertainty_head_callback = uncertainty_head_callback;
    params.broadcast_head_callback = broadcast_head_callback;

    if (llama_self_state_configure(ctx, params) != 0) {
        llama_free(ctx);
        return nullptr;
    }

    const llama_self_state_time_point t0 = {
        /*.wall_clock_ms =*/ 1700000000000,
        /*.monotonic_ms =*/ 1000,
        /*.timezone_offset_minutes =*/ -360,
    };
    if (llama_self_state_set_time(ctx, t0) != 0) {
        llama_free(ctx);
        return nullptr;
    }

    llama_active_lora_params active = llama_active_lora_default_params();
    active.enabled = true;
    active.host_memory_ratio = 1.0f;
    active.device_memory_ratio = 1.0f;
    active.min_rank = 1;
    active.max_rank = 2;
    if (llama_active_lora_init(ctx, active) != 0) {
        llama_free(ctx);
        return nullptr;
    }

    llama_past_lora_params past = llama_past_lora_default_params();
    configure_small_buckets(past);
    if (llama_past_lora_init(ctx, past) != 0) {
        llama_free(ctx);
        return nullptr;
    }

    return ctx;
}

static bool expect_active_action(
        const char * label,
        llama_context * ctx,
        const llama_self_state_event & event,
        int32_t expected_action,
        llama_active_loop_trace * out_trace = nullptr) {
    llama_active_loop_trace trace = {};
    if (llama_active_loop_process(ctx, &event, &trace) != 0) {
        std::fprintf(stderr, "%s: failed to process active loop\n", label);
        return false;
    }
    if (trace.winner_action != expected_action) {
        std::fprintf(stderr, "%s: unexpected active action: got %d expected %d\n", label, trace.winner_action, expected_action);
        return false;
    }
    if (trace.candidate_count != LLAMA_ACTIVE_LOOP_MAX_CANDIDATES) {
        std::fprintf(stderr, "%s: unexpected active candidate count\n", label);
        return false;
    }
    if (out_trace) {
        *out_trace = trace;
    }
    return true;
}

static bool expect_dmn_action(
        const char * label,
        llama_context * ctx,
        uint64_t now_us,
        int32_t expected_action,
        int32_t min_burst_count = 0) {
    llama_dmn_tick_trace trace = {};
    if (llama_dmn_tick(ctx, now_us, &trace) != 0) {
        std::fprintf(stderr, "%s: failed to tick DMN\n", label);
        return false;
    }
    if (!trace.admitted) {
        std::fprintf(stderr, "%s: expected admitted DMN tick\n", label);
        return false;
    }
    if (trace.winner_action != expected_action) {
        std::fprintf(stderr, "%s: unexpected DMN action: got %d expected %d\n", label, trace.winner_action, expected_action);
        return false;
    }
    if (trace.candidate_count != LLAMA_DMN_MAX_CANDIDATES) {
        std::fprintf(stderr, "%s: unexpected DMN candidate count\n", label);
        return false;
    }
    if (trace.burst_count < min_burst_count) {
        std::fprintf(stderr, "%s: unexpected DMN burst count %d\n", label, trace.burst_count);
        return false;
    }
    if (trace.seed_source_mask == 0) {
        std::fprintf(stderr, "%s: missing DMN seed source mask\n", label);
        return false;
    }
    return true;
}

static bool expect_single_command(
        const char * label,
        llama_context * ctx,
        int32_t expected_origin,
        int32_t expected_kind,
        llama_cognitive_command * out_command = nullptr) {
    const int32_t count = llama_cognitive_command_count(ctx);
    if (count != 1) {
        std::fprintf(stderr, "%s: unexpected cognitive command count %d\n", label, count);
        return false;
    }

    llama_cognitive_command command = {};
    if (llama_cognitive_command_get(ctx, 0, &command) != 0) {
        std::fprintf(stderr, "%s: failed to retrieve cognitive command\n", label);
        return false;
    }
    if (command.origin != expected_origin || command.kind != expected_kind) {
        std::fprintf(stderr, "%s: unexpected command origin/kind (%d/%d)\n", label, command.origin, command.kind);
        return false;
    }
    if (out_command) {
        *out_command = command;
    }
    return true;
}

static bool expect_bash_request(
        const char * label,
        llama_context * ctx,
        const llama_cognitive_command & command,
        const char * expected_command_fragment,
        llama_bash_tool_request * out_request = nullptr) {
    llama_bash_tool_request request = {};
    if (llama_cognitive_bash_tool_get_request(ctx, command.command_id, &request) != 0) {
        std::fprintf(stderr, "%s: failed to retrieve bash request\n", label);
        return false;
    }
    if (request.command_id != command.command_id ||
        request.tool_job_id != command.tool_job_id ||
        request.origin != command.origin ||
        !request.command_ready ||
        request.command_text[0] == '\0') {
        std::fprintf(stderr, "%s: invalid bash request metadata\n", label);
        return false;
    }
    if (expected_command_fragment && std::strstr(request.command_text, expected_command_fragment) == nullptr) {
        std::fprintf(stderr, "%s: unexpected bash request command text: %s\n", label, request.command_text);
        return false;
    }
    if (out_request) {
        *out_request = request;
    }
    return true;
}

static bool expect_functional_family_state(
        const char * label,
        llama_context * ctx,
        int32_t family,
        bool expected_active,
        int32_t expected_microphase = -1,
        float min_gain = -1.0f) {
    llama_functional_lora_family_state state = {};
    if (llama_functional_lora_family_state_get(ctx, family, &state) != 0) {
        std::fprintf(stderr, "%s: failed to query functional family state\n", label);
        return false;
    }
    if (state.active_now != expected_active) {
        std::fprintf(stderr, "%s: unexpected active state for family %d\n", label, family);
        return false;
    }
    if (expected_microphase >= 0 && state.current_microphase != expected_microphase) {
        std::fprintf(stderr, "%s: unexpected microphase for family %d: got %d expected %d\n",
                label, family, state.current_microphase, expected_microphase);
        return false;
    }
    if (min_gain >= 0.0f && state.current_gain < min_gain) {
        std::fprintf(stderr, "%s: unexpected gain for family %d: got %f expected >= %f\n",
                label, family, state.current_gain, min_gain);
        return false;
    }
    return true;
}

static bool expect_functional_last_update(
        const char * label,
        llama_context * ctx,
        int32_t family,
        int32_t expected_settle_microphase) {
    llama_functional_lora_update_info update = {};
    if (llama_functional_lora_get_last_update(ctx, family, &update) != 0 || !update.valid) {
        std::fprintf(stderr, "%s: missing functional update for family %d\n", label, family);
        return false;
    }
    if (update.family != family || update.settle_microphase != expected_settle_microphase) {
        std::fprintf(stderr, "%s: unexpected functional update metadata for family %d\n", label, family);
        return false;
    }
    return true;
}

static bool settle_active_runner(const char * label, llama_context * ctx) {
    while (llama_cognitive_command_count(ctx) > 0) {
        llama_cognitive_command command = {};
        if (llama_cognitive_command_get(ctx, 0, &command) != 0) {
            std::fprintf(stderr, "%s: failed to retrieve pending active command\n", label);
            return false;
        }
        if (command.origin != LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
            std::fprintf(stderr, "%s: encountered non-active pending command while settling foreground state\n", label);
            return false;
        }

        if (command.kind == LLAMA_COG_COMMAND_EMIT_ANSWER || command.kind == LLAMA_COG_COMMAND_EMIT_ASK) {
            llama_cognitive_active_runner_status runner = {};
            if (llama_cognitive_active_runner_get(ctx, &runner) != 0 ||
                llama_active_loop_note_emit(ctx, runner.episode_id, 1) != 0) {
                std::fprintf(stderr, "%s: failed to settle foreground emit command\n", label);
                return false;
            }
        } else if (command.kind == LLAMA_COG_COMMAND_INVOKE_TOOL) {
            if (llama_cognitive_command_complete(ctx, command.command_id, true) != 0) {
                std::fprintf(stderr, "%s: failed to cancel foreground tool command\n", label);
                return false;
            }
        } else {
            std::fprintf(stderr, "%s: unexpected foreground command kind %d while settling\n", label, command.kind);
            return false;
        }
    }

    llama_cognitive_active_runner_status runner = {};
    if (llama_cognitive_active_runner_get(ctx, &runner) != 0 || runner.active || !runner.completed) {
        std::fprintf(stderr, "%s: foreground runner did not settle to terminal state\n", label);
        return false;
    }

    return true;
}

static bool ingest_event_without_runner(
        const char * label,
        llama_context * ctx,
        const llama_self_state_event & event) {
    if (llama_self_state_set_channel_state(ctx, LLAMA_SELF_STATE_CHANNEL_ACTIVE) != 0) {
        std::fprintf(stderr, "%s: failed to set self-state channel\n", label);
        return false;
    }
    if ((event.role == LLAMA_SELF_STATE_EVENT_TOOL &&
            llama_self_state_note_tool_event(ctx) != 0) ||
        (event.role != LLAMA_SELF_STATE_EVENT_TOOL &&
            llama_self_state_note_user_event(ctx) != 0)) {
        std::fprintf(stderr, "%s: failed to note self-state event\n", label);
        return false;
    }

    llama_self_state_event mutable_event = event;
    llama_self_state_feature_vector pre = {};
    llama_self_state_feature_vector post = {};
    if (llama_self_state_build_prewrite_features(ctx, &mutable_event, &pre) != 0 ||
        llama_self_state_apply_prewrite(ctx, &mutable_event, &pre) != 0) {
        std::fprintf(stderr, "%s: failed to apply prewrite state\n", label);
        return false;
    }

    const bool admitted =
            mutable_event.n_tokens > 0 ||
            pre.memory_write_pressure > 0.12f ||
            mutable_event.role == LLAMA_SELF_STATE_EVENT_TOOL;
    if (admitted) {
        mutable_event.flags |= LLAMA_SELF_STATE_EVENT_ADMITTED;
    }

    if (llama_self_state_build_postwrite_features(ctx, &mutable_event, &post) != 0 ||
        llama_self_state_apply_postwrite(ctx, &mutable_event, &post) != 0) {
        std::fprintf(stderr, "%s: failed to apply postwrite state\n", label);
        return false;
    }

    if ((mutable_event.flags & LLAMA_SELF_STATE_EVENT_ADMITTED) &&
        mutable_event.tokens &&
        mutable_event.n_tokens > 0 &&
        llama_active_lora_ingest(ctx, mutable_event.tokens, mutable_event.n_tokens) != 0) {
        std::fprintf(stderr, "%s: failed to ingest active LoRA span\n", label);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    {
        g_contradiction_score = 0.15f;
        g_uncertainty_score = 0.10f;
        g_broadcast_score = 0.30f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create active answer context\n");
            llama_model_free(model);
            return 1;
        }

        llama_cognitive_tool_spec spec = {};
        if (llama_cognitive_tool_spec_count(ctx) < 4 ||
            llama_cognitive_tool_spec_get(ctx, 0, &spec) != 0 ||
            spec.tool_kind != LLAMA_TOOL_KIND_GENERIC ||
            spec.name[0] == '\0') {
            std::fprintf(stderr, "default cognitive tool registry was not populated\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (llama_cognitive_tool_spec_get(ctx, 1, &spec) != 0 ||
            spec.tool_kind != LLAMA_TOOL_KIND_BASH_CLI ||
            spec.name[0] == '\0') {
            std::fprintf(stderr, "bash cognitive tool registry entry was not populated\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "The tool finished and the result is ready.");
        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };

        llama_active_loop_trace trace = {};
        if (!expect_active_action("active-answer", ctx, event, LLAMA_ACTIVE_LOOP_ACTION_ANSWER, &trace)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.loop_state.phase != LLAMA_COG_LOOP_PHASE_FINISH ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_ANSWER_READY ||
            trace.loop_state.tool_registry_count < 4 ||
            trace.tool_proposal.valid ||
            !trace.observation.valid ||
            trace.functional_activation.top_family != LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ||
            trace.functional_activation.microphase != LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION ||
            trace.observation.status != LLAMA_SELF_TOOL_JOB_COMPLETED) {
            std::fprintf(stderr, "active answer trace did not expose bounded loop scaffolding\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_cognitive_command command = {};
        llama_cognitive_active_runner_status runner = {};
        if (!expect_single_command("active-answer-command", ctx, LLAMA_COG_COMMAND_ORIGIN_ACTIVE, LLAMA_COG_COMMAND_EMIT_ANSWER, &command) ||
            llama_cognitive_active_runner_get(ctx, &runner) != 0 ||
            !runner.active ||
            runner.completed ||
            runner.pending_command_id != command.command_id ||
            runner.functional_microphase != LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION ||
            !expect_functional_family_state("active-answer-functional", ctx, LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, true, LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION, 0.10f)) {
            std::fprintf(stderr, "active answer runner did not retain pending emit command\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (llama_active_loop_note_emit(ctx, trace.episode_id, 32) != 0) {
            std::fprintf(stderr, "failed to note active emit\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (llama_cognitive_command_count(ctx) != 0 ||
            llama_cognitive_active_runner_get(ctx, &runner) != 0 ||
            runner.active ||
            !runner.completed) {
            std::fprintf(stderr, "active answer runner did not clear after emit\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!expect_functional_family_state("active-answer-functional-cleared", ctx, LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, false)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_active_loop_trace latest = {};
        if (llama_active_loop_get_last_trace(ctx, &latest) != 0 || !latest.emit_noted) {
            std::fprintf(stderr, "failed to retrieve active loop trace\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_self_model_state_info model_state = {};
        llama_self_register_info register_info = {};
        if (llama_self_state_get_model_state(ctx, &model_state) != 0 ||
            !model_state.forecast.valid ||
            model_state.horizons[LLAMA_SELF_HORIZON_INSTANT].epistemic.answerability <= 0.0f ||
            llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_ANSWERABILITY, &register_info) != 0 ||
            register_info.scalar_value <= 0.0f) {
            std::fprintf(stderr, "active loop did not preserve expanded self-model state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.85f;
        g_uncertainty_score = 0.85f;
        g_broadcast_score = 0.20f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create active ask context\n");
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "Which tool should I use to clarify this unknown error?");
        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 1.0f,
            /*.decoder_top_margin =*/ 0.0f,
        };

        if (!expect_active_action("active-ask", ctx, event, LLAMA_ACTIVE_LOOP_ACTION_ASK)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.10f;
        g_uncertainty_score = 0.10f;
        g_broadcast_score = 0.05f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create active act context\n");
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> goal_tokens = tokenize_or_die(vocab, "Inspect the repository and run the appropriate tools.");
        if (llama_self_state_upsert_goal(ctx, 1, goal_tokens.data(), goal_tokens.size(), 1.0f) != 0 ||
            llama_self_state_upsert_tool_job(ctx, 7, LLAMA_SELF_TOOL_JOB_PENDING, 1.0f) != 0) {
            std::fprintf(stderr, "failed to seed active act state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "Please search the repository and inspect the build logs.");
        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.1f,
            /*.decoder_top_margin =*/ 1.0f,
        };

        llama_active_loop_trace trace = {};
        if (!expect_active_action("active-act", ctx, event, LLAMA_ACTIVE_LOOP_ACTION_ACT, &trace)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.loop_state.phase != LLAMA_COG_LOOP_PHASE_PREPARE_TOOL ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_TOOL_REQUIRED ||
            !trace.loop_state.continuation_allowed ||
            !trace.loop_state.waiting_on_tool ||
            !trace.tool_proposal.valid ||
            trace.tool_proposal.tool_kind != LLAMA_TOOL_KIND_BASH_CLI ||
            trace.functional_activation.top_family != LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ||
            trace.functional_activation.microphase != LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP ||
            trace.tool_proposal.expected_steps <= 0) {
            std::fprintf(stderr, "active act trace did not expose tool preparation scaffolding\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_cognitive_command command = {};
        llama_cognitive_active_runner_status runner = {};
        if (!expect_single_command("active-act-command", ctx, LLAMA_COG_COMMAND_ORIGIN_ACTIVE, LLAMA_COG_COMMAND_INVOKE_TOOL, &command) ||
            command.tool_job_id <= 0 ||
            llama_cognitive_active_runner_get(ctx, &runner) != 0 ||
            !runner.active ||
            !runner.waiting_on_tool ||
            runner.pending_command_id != command.command_id ||
            runner.functional_microphase != LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP ||
            !expect_functional_family_state("active-act-functional", ctx, LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, true, LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP, 0.10f)) {
            std::fprintf(stderr, "active act runner did not retain waiting tool command\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_bash_tool_request request = {};
        if (!expect_bash_request("active-act-request", ctx, command, "find build .", &request)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_cognitive_host_state host = {};
        if (llama_cognitive_get_host_state(ctx, &host) != 0 || host.pending_tool_followup_count < 1) {
            std::fprintf(stderr, "active loop did not mark tool followup\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_bash_tool_result result = {};
        result.command_id = request.command_id;
        result.tool_job_id = request.tool_job_id;
        result.exit_code = 0;
        result.runtime_ms = 12;
        std::snprintf(result.stdout_text, sizeof(result.stdout_text), "%s", "build/logs/unit.log");

        const int32_t original_episode = trace.episode_id;
        if (llama_cognitive_bash_tool_submit_result(ctx, &result, &trace) != 0) {
            std::fprintf(stderr, "failed to submit active bash tool result\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.winner_action != LLAMA_ACTIVE_LOOP_ACTION_ANSWER ||
            !trace.observation.valid ||
            trace.observation.tool_kind != LLAMA_TOOL_KIND_BASH_CLI ||
            trace.observation.job_id != command.tool_job_id ||
            trace.observation.status != LLAMA_SELF_TOOL_JOB_COMPLETED ||
            trace.functional_activation.top_family != LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ||
            trace.functional_activation.microphase != LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_ANSWER_READY ||
            trace.episode_id != original_episode) {
            std::fprintf(stderr, "tool followup did not surface tool observation scaffolding\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!expect_functional_last_update(
                    "active-tool-selection-update",
                    ctx,
                    LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION,
                    LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_bash_tool_result last_result = {};
        if (llama_bash_tool_get_last_result(ctx, &last_result) != 0 ||
            last_result.command_id != command.command_id ||
            last_result.tool_job_id != command.tool_job_id ||
            last_result.exit_code != 0) {
            std::fprintf(stderr, "active bash result was not retained for observation\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!expect_single_command("active-tool-followup-command", ctx, LLAMA_COG_COMMAND_ORIGIN_ACTIVE, LLAMA_COG_COMMAND_EMIT_ANSWER, &command) ||
            llama_cognitive_active_runner_get(ctx, &runner) != 0 ||
            !runner.active ||
            runner.waiting_on_tool ||
            !expect_functional_family_state("active-tool-followup-functional", ctx, LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, true, LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION, 0.10f)) {
            std::fprintf(stderr, "active tool followup did not schedule final emit command\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (llama_cognitive_get_host_state(ctx, &host) != 0 || host.pending_tool_followup_count != 0) {
            std::fprintf(stderr, "tool followup count did not settle after bash result submission\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (llama_active_loop_note_emit(ctx, trace.episode_id, 24) != 0 ||
            llama_cognitive_command_count(ctx) != 0) {
            std::fprintf(stderr, "active tool followup emit did not clear pending command\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!expect_functional_family_state("active-tool-followup-functional-cleared", ctx, LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, false)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.0f;
        g_uncertainty_score = 0.0f;
        g_broadcast_score = 0.0f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create active wait context\n");
            llama_model_free(model);
            return 1;
        }

        const llama_self_state_event event = {
            /*.tokens =*/ nullptr,
            /*.n_tokens =*/ 0,
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };

        llama_active_loop_trace trace = {};
        if (!expect_active_action("active-wait", ctx, event, LLAMA_ACTIVE_LOOP_ACTION_WAIT, &trace)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.loop_state.phase != LLAMA_COG_LOOP_PHASE_FINISH ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_WAITING_ON_TOOL) {
            std::fprintf(stderr, "active wait trace did not expose bounded terminal state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.10f;
        g_uncertainty_score = 0.10f;
        g_broadcast_score = 0.05f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create functional ablation context\n");
            llama_model_free(model);
            return 1;
        }

        llama_functional_lora_ablation_config ablation = {};
        ablation.disabled_family_mask = (1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION);
        if (llama_functional_lora_set_ablation(ctx, ablation) != 0) {
            std::fprintf(stderr, "failed to configure functional family ablation\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> goal_tokens = tokenize_or_die(vocab, "Inspect the repository and run the appropriate tools.");
        if (llama_self_state_upsert_goal(ctx, 101, goal_tokens.data(), goal_tokens.size(), 1.0f) != 0 ||
            llama_self_state_upsert_tool_job(ctx, 107, LLAMA_SELF_TOOL_JOB_PENDING, 1.0f) != 0) {
            std::fprintf(stderr, "failed to seed functional ablation active state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "Please search the repository and inspect the build logs.");
        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.1f,
            /*.decoder_top_margin =*/ 1.0f,
        };

        llama_active_loop_trace trace = {};
        if (!expect_active_action("functional-ablation-active-act", ctx, event, LLAMA_ACTIVE_LOOP_ACTION_ACT, &trace) ||
            trace.functional_activation.top_family != LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ||
            !expect_functional_family_state("functional-ablation-family-state", ctx, LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, false)) {
            std::fprintf(stderr, "functional family ablation did not suppress live activation while preserving route intent\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (!settle_active_runner("functional-ablation-settle", ctx)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.0f;
        g_uncertainty_score = 0.0f;
        g_broadcast_score = 0.0f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create low-pressure DMN context\n");
            llama_model_free(model);
            return 1;
        }

        llama_dmn_tick_trace trace = {};
        if (llama_dmn_tick(ctx, 5000, &trace) != 0 || trace.admitted) {
            std::fprintf(stderr, "DMN admitted without pressure\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.winner_action != LLAMA_DMN_ACTION_SILENT) {
            std::fprintf(stderr, "unexpected low-pressure DMN action\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.loop_state.phase != LLAMA_COG_LOOP_PHASE_FINISH ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_PRESSURE_NOT_ADMITTED) {
            std::fprintf(stderr, "low-pressure DMN trace did not expose admission stop reason\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_cognitive_dmn_runner_status runner = {};
        if (llama_cognitive_command_count(ctx) != 0 ||
            llama_cognitive_dmn_runner_get(ctx, &runner) != 0 ||
            runner.active ||
            !runner.completed) {
            std::fprintf(stderr, "low-pressure DMN runner state was not terminal\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.functional_activation.microphase != LLAMA_FUNCTIONAL_MICROPHASE_NONE ||
            !expect_functional_family_state("low-pressure-dmn-functional", ctx, LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION, false)) {
            std::fprintf(stderr, "low-pressure DMN did not clear functional state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.90f;
        g_uncertainty_score = 0.90f;
        g_broadcast_score = 0.05f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create internal-write DMN context\n");
            llama_model_free(model);
            return 1;
        }

        llama_hard_memory_config hard_memory = llama_hard_memory_default_config();
        hard_memory.enabled = true;
        std::snprintf(hard_memory.base_url, sizeof(hard_memory.base_url), "%s", "http://127.0.0.1:1");
        std::snprintf(hard_memory.auth_token, sizeof(hard_memory.auth_token), "%s", "sm_test_token");
        std::snprintf(hard_memory.container_tag, sizeof(hard_memory.container_tag), "%s", "vicuna-cognitive");
        if (llama_hard_memory_configure(ctx, hard_memory) != 0) {
            std::fprintf(stderr, "failed to configure hard memory for counterfactual ladder test\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> handle_tokens = tokenize_or_die(vocab, "important contradiction memory cluster");
        if (llama_self_state_upsert_memory_handle(ctx, 5, LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER, handle_tokens.data(), handle_tokens.size(), 1.0f) != 0) {
            std::fprintf(stderr, "failed to seed DMN reactivation state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "I am not sure whether this contradiction is resolved.");
        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 1.0f,
            /*.decoder_top_margin =*/ 0.0f,
        };
        if (!ingest_event_without_runner("seed-internal-dmn", ctx, event)) {
            std::fprintf(stderr, "failed to seed internal-write pressure\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_dmn_tick_trace trace = {};
        if (llama_dmn_tick(ctx, 6000, &trace) != 0 ||
            !trace.admitted ||
            trace.winner_action != LLAMA_DMN_ACTION_INTERNAL_WRITE) {
            std::fprintf(stderr, "dmn-internal-write: failed to tick DMN (admitted=%d winner=%d terminal=%d deferred=%d)\n",
                    trace.admitted ? 1 : 0,
                    trace.winner_action,
                    trace.loop_state.terminal_reason,
                    trace.deferred_for_foreground ? 1 : 0);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.loop_state.phase != LLAMA_COG_LOOP_PHASE_FINISH ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_INTERNAL_WRITE_READY ||
            !trace.observation.valid ||
            trace.functional_activation.top_family != LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION ||
            trace.functional_activation.microphase != LLAMA_FUNCTIONAL_MICROPHASE_POST_ACTION_REFLECTION ||
            trace.observation.status != LLAMA_SELF_TOOL_JOB_COMPLETED) {
            std::fprintf(stderr, "internal-write DMN trace did not expose bounded loop observation state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_cognitive_dmn_runner_status runner = {};
        if (llama_cognitive_dmn_runner_get(ctx, &runner) != 0 ||
            runner.steps_taken < 2 ||
            runner.active) {
            std::fprintf(stderr, "internal-write DMN runner did not finish bounded local execution\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!expect_functional_last_update("dmn-self-observation-update", ctx, LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION, LLAMA_FUNCTIONAL_MICROPHASE_POST_ACTION_REFLECTION) ||
            !expect_functional_last_update("dmn-counterfactual-update", ctx, LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL, LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE) ||
            !expect_functional_family_state("dmn-internal-write-functional-cleared", ctx, LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION, false)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_favorable_state_profile favorable = {};
        if (llama_favorable_state_get(ctx, &favorable) != 0 ||
            favorable.dimension_count <= 0 ||
            favorable.priority_count != favorable.dimension_count ||
            favorable.aggregate_divergence <= 0.0f) {
            std::fprintf(stderr, "missing favorable-state profile after internal-write DMN tick\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        for (int32_t i = 1; i < favorable.priority_count; ++i) {
            const auto & prev = favorable.dimensions[favorable.priority_order[i - 1]];
            const auto & curr = favorable.dimensions[favorable.priority_order[i]];
            if (prev.weighted_divergence + 1.0e-6f < curr.weighted_divergence) {
                std::fprintf(stderr, "favorable-state priorities are not sorted by divergence\n");
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }

        llama_counterfactual_trace counterfactual = {};
        if (llama_counterfactual_get_last_trace(ctx, &counterfactual) != 0 ||
            counterfactual.candidate_count < 6 ||
            counterfactual.winner_index < 0 ||
            counterfactual.candidates[0].family != LLAMA_COUNTERFACTUAL_FAMILY_MESSAGE_VARIANT ||
            counterfactual.candidates[1].family != LLAMA_COUNTERFACTUAL_FAMILY_TOOL_ARGUMENTS ||
            counterfactual.candidates[2].family != LLAMA_COUNTERFACTUAL_FAMILY_HARD_MEMORY_QUERY ||
            counterfactual.candidates[3].family != LLAMA_COUNTERFACTUAL_FAMILY_TOOL_CHOICE ||
            counterfactual.candidates[4].family != LLAMA_COUNTERFACTUAL_FAMILY_TIMING_SHIFT) {
            std::fprintf(stderr, "counterfactual ladder was not generated in low-risk-first order\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        bool saw_lora_ablation = false;
        for (int32_t i = 0; i < counterfactual.candidate_count; ++i) {
            if (counterfactual.candidates[i].family == LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION) {
                saw_lora_ablation = true;
                break;
            }
        }
        if (!saw_lora_ablation) {
            std::fprintf(stderr, "counterfactual ladder did not include LoRA ablation\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_remediation_plan remediation = {};
        if (llama_remediation_get_last_plan(ctx, &remediation) != 0 ||
            remediation.source_family < 0 ||
            remediation.pre_divergence <= 0.0f ||
            !remediation.applied ||
            remediation.action != LLAMA_REMEDIATION_ACTION_ACTIVE_LORA_UPDATE) {
            std::fprintf(stderr, "DMN did not apply bounded Active LoRA remediation\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_active_lora_stats active_stats = {};
        if (llama_active_lora_get_stats(ctx, &active_stats) != 0 || active_stats.updates_applied < 2) {
            std::fprintf(stderr, "Active LoRA remediation did not advance runtime updates\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.10f;
        g_uncertainty_score = 0.10f;
        g_broadcast_score = 0.05f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create tool DMN context\n");
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> handle_tokens = tokenize_or_die(vocab, "repository inspection tool memory cluster");
        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "Please search the repository and inspect the build logs.");
        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        if (llama_self_state_upsert_memory_handle(ctx, 21, LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER, handle_tokens.data(), handle_tokens.size(), 1.0f) != 0 ||
            llama_active_loop_process(ctx, &event, nullptr) != 0 ||
            llama_self_state_upsert_tool_job(ctx, 9, LLAMA_SELF_TOOL_JOB_COMPLETED, 1.0f) != 0) {
            std::fprintf(stderr, "failed to seed tool DMN context\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_dmn_tick_trace trace = {};
        if (llama_dmn_tick(ctx, 7000, &trace) != 0 ||
            trace.admitted ||
            !trace.deferred_for_foreground ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_BACKGROUND_DEFERRED) {
            std::fprintf(stderr, "DMN did not defer while foreground runner work was still pending\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!settle_active_runner("settle-active-before-dmn", ctx)) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (llama_dmn_tick(ctx, 7001, &trace) != 0 ||
            !trace.admitted ||
            trace.winner_action != LLAMA_DMN_ACTION_INVOKE_TOOL) {
            std::fprintf(stderr, "dmn-invoke-tool: failed to tick DMN\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.loop_state.phase != LLAMA_COG_LOOP_PHASE_PREPARE_TOOL ||
            trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_TOOL_REQUIRED ||
            !trace.loop_state.waiting_on_tool ||
            !trace.tool_proposal.valid ||
            trace.tool_proposal.tool_kind != LLAMA_TOOL_KIND_BASH_CLI ||
            trace.functional_activation.top_family != LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ||
            trace.functional_activation.microphase != LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP ||
            trace.tool_proposal.job_id <= 0) {
            std::fprintf(stderr, "tool-focused DMN trace did not expose tool proposal scaffolding\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        llama_cognitive_command command = {};
        llama_cognitive_dmn_runner_status runner = {};
        if (!expect_single_command("dmn-invoke-tool-command", ctx, LLAMA_COG_COMMAND_ORIGIN_DMN, LLAMA_COG_COMMAND_INVOKE_TOOL, &command) ||
            command.tool_job_id != trace.tool_job_id ||
            llama_cognitive_dmn_runner_get(ctx, &runner) != 0 ||
            !runner.active ||
            !runner.waiting_on_tool ||
            runner.pending_command_id != command.command_id ||
            runner.functional_microphase != LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP ||
            !expect_functional_family_state("dmn-invoke-tool-functional", ctx, LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, true, LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP, 0.10f)) {
            std::fprintf(stderr, "tool-focused DMN runner did not retain pending tool command\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (!expect_bash_request("dmn-invoke-tool-request", ctx, command, "find . -maxdepth 3")) {
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_remediation_plan remediation = {};
        if (llama_remediation_get_last_plan(ctx, &remediation) != 0 ||
            remediation.action != LLAMA_REMEDIATION_ACTION_GATHER_INFO ||
            remediation.tool_job_id <= 0) {
            std::fprintf(stderr, "tool-focused DMN tick did not request bounded information gathering\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.88f;
        g_uncertainty_score = 0.72f;
        g_broadcast_score = 0.04f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create compression DMN context\n");
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "There are unresolved commitments, causal links, and social details to preserve.");
        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.9f,
            /*.decoder_top_margin =*/ 0.0f,
        };
        for (int32_t i = 0; i < 4; ++i) {
            if (!ingest_event_without_runner("seed-compression-dmn", ctx, event)) {
                std::fprintf(stderr, "failed to seed compression DMN pressure\n");
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }

        llama_dmn_tick_trace trace = {};
        if (llama_dmn_tick(ctx, 6500, &trace) != 0 || !trace.admitted) {
            std::fprintf(stderr, "compression DMN tick failed\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if ((trace.maintenance_mask & LLAMA_DMN_MAINTENANCE_COMPRESS_WORKING_MEMORY) == 0 ||
            !expect_functional_last_update("dmn-memory-compression-update", ctx, LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION, LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_AUDIT)) {
            std::fprintf(stderr, "compression DMN tick did not settle memory-compression update\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.10f;
        g_uncertainty_score = 0.10f;
        g_broadcast_score = 0.95f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create emit DMN context\n");
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> goal_tokens = tokenize_or_die(vocab, "follow up with the user on the next step");
        const std::vector<llama_token> handle_tokens = tokenize_or_die(vocab, "user follow up memory cluster");
        const std::vector<llama_token> tokens = tokenize_or_die(vocab, "Please continue the follow up again with more details.");
        const std::vector<llama_token> system_tokens = tokenize_or_die(vocab, "Following up with more detail for the user now.");

        if (llama_self_state_upsert_goal(ctx, 4, goal_tokens.data(), goal_tokens.size(), 1.0f) != 0 ||
            llama_self_state_upsert_memory_handle(ctx, 12, LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER, handle_tokens.data(), handle_tokens.size(), 1.0f) != 0) {
            std::fprintf(stderr, "failed to seed emit DMN goals\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const llama_self_state_event event = {
            /*.tokens =*/ tokens.data(),
            /*.n_tokens =*/ tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        if (!ingest_event_without_runner("seed-emit-dmn-first", ctx, event) ||
            !ingest_event_without_runner("seed-emit-dmn-second", ctx, event)) {
            std::fprintf(stderr, "failed to seed emit DMN context\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const llama_self_state_event system_event = {
            /*.tokens =*/ system_tokens.data(),
            /*.n_tokens =*/ system_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        llama_self_state_feature_vector pre = {};
        llama_self_state_feature_vector post = {};
        if (llama_self_state_build_prewrite_features(ctx, &system_event, &pre) != 0 ||
            llama_self_state_apply_prewrite(ctx, &system_event, &pre) != 0 ||
            llama_self_state_build_postwrite_features(ctx, &system_event, &post) != 0 ||
            llama_self_state_apply_postwrite(ctx, &system_event, &post) != 0) {
            std::fprintf(stderr, "failed to seed emit followup state\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_dmn_tick_trace trace = {};
        if (llama_dmn_tick(ctx, 8000, &trace) != 0 ||
            !trace.admitted ||
            (trace.winner_action != LLAMA_DMN_ACTION_EMIT && trace.winner_action != LLAMA_DMN_ACTION_INVOKE_TOOL) ||
            (trace.winner_action == LLAMA_DMN_ACTION_EMIT && trace.burst_count < 2)) {
            std::fprintf(stderr, "failed to sustain high-continuation DMN routing\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(ctx);
    }

    {
        g_contradiction_score = 0.0f;
        g_uncertainty_score = 0.0f;
        g_broadcast_score = 0.0f;

        const std::vector<llama_token> complaint_tokens = tokenize_or_die(vocab, "This feels frustrating, disappointing, and not helpful.");
        const llama_self_state_event complaint_event = {
            /*.tokens =*/ complaint_tokens.data(),
            /*.n_tokens =*/ complaint_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };

        llama_context * no_repair_ctx = create_context(model);
        llama_context * repair_ctx = create_context(model);
        if (!no_repair_ctx || !repair_ctx) {
            std::fprintf(stderr, "failed to create repair-pressure admission contexts\n");
            if (no_repair_ctx) {
                llama_free(no_repair_ctx);
            }
            if (repair_ctx) {
                llama_free(repair_ctx);
            }
            llama_model_free(model);
            return 1;
        }

        for (int i = 0; i < 3; ++i) {
            if (!ingest_event_without_runner("seed-no-repair-dmn", no_repair_ctx, complaint_event) ||
                !ingest_event_without_runner("seed-repair-dmn", repair_ctx, complaint_event)) {
                std::fprintf(stderr, "failed to seed repair-pressure dissatisfaction\n");
                llama_free(no_repair_ctx);
                llama_free(repair_ctx);
                llama_model_free(model);
                return 1;
            }
        }

        llama_self_updater_program no_repair_policy = {};
        llama_self_updater_program repair_policy = {};
        if (llama_self_state_get_updater_program(no_repair_ctx, &no_repair_policy) != 0 ||
            llama_self_state_get_updater_program(repair_ctx, &repair_policy) != 0) {
            std::fprintf(stderr, "failed to fetch repair-pressure admission policies\n");
            llama_free(no_repair_ctx);
            llama_free(repair_ctx);
            llama_model_free(model);
            return 1;
        }

        no_repair_policy.version += 1;
        no_repair_policy.repair_admission_floor = 0.20f;
        no_repair_policy.repair_admission_weight = 0.0f;
        if (llama_self_state_set_updater_program(no_repair_ctx, no_repair_policy) != 0) {
            std::fprintf(stderr, "failed to disable repair-pressure admission weight\n");
            llama_free(no_repair_ctx);
            llama_free(repair_ctx);
            llama_model_free(model);
            return 1;
        }

        repair_policy.version += 1;
        repair_policy.repair_admission_floor = 0.20f;
        repair_policy.repair_admission_weight = 0.65f;
        repair_policy.repair_emit_threshold = 1.0f;
        if (llama_self_state_set_updater_program(repair_ctx, repair_policy) != 0) {
            std::fprintf(stderr, "failed to raise repair-pressure admission weight\n");
            llama_free(no_repair_ctx);
            llama_free(repair_ctx);
            llama_model_free(model);
            return 1;
        }

        llama_dmn_tick_trace no_repair_trace = {};
        llama_dmn_tick_trace repair_trace = {};
        if (llama_dmn_tick(no_repair_ctx, 8500, &no_repair_trace) != 0 ||
            llama_dmn_tick(repair_ctx, 8500, &repair_trace) != 0) {
            std::fprintf(stderr, "failed to tick repair-pressure admission DMN contexts\n");
            llama_free(no_repair_ctx);
            llama_free(repair_ctx);
            llama_model_free(model);
            return 1;
        }

        if (no_repair_trace.pressure.repair <= 0.0f ||
            repair_trace.pressure.repair <= 0.0f ||
            no_repair_trace.admitted ||
            no_repair_trace.pressure.total >= 0.24f ||
            !repair_trace.admitted ||
            repair_trace.pressure.total < 0.24f) {
            std::fprintf(stderr,
                    "repair pressure did not control DMN admission through the single threshold "
                    "(no_repair admitted=%d total=%.3f repair=%.3f, repair admitted=%d total=%.3f repair=%.3f)\n",
                    no_repair_trace.admitted ? 1 : 0,
                    no_repair_trace.pressure.total,
                    no_repair_trace.pressure.repair,
                    repair_trace.admitted ? 1 : 0,
                    repair_trace.pressure.total,
                    repair_trace.pressure.repair);
            llama_free(no_repair_ctx);
            llama_free(repair_ctx);
            llama_model_free(model);
            return 1;
        }

        llama_governance_trace repair_governance = {};
        if (llama_governance_get_last_trace(repair_ctx, &repair_governance) != 0 ||
            repair_governance.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR) {
            std::fprintf(stderr, "repair-pressure admission bypassed separate repair emission governance\n");
            llama_free(no_repair_ctx);
            llama_free(repair_ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(no_repair_ctx);
        llama_free(repair_ctx);
    }

    {
        g_contradiction_score = 0.75f;
        g_uncertainty_score = 0.65f;
        g_broadcast_score = 1.00f;

        llama_context * ctx = create_context(model);
        if (!ctx) {
            std::fprintf(stderr, "failed to create repair-governance DMN context\n");
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> goal_tokens = tokenize_or_die(vocab, "repair trust with the user and continue carefully");
        const std::vector<llama_token> complaint_tokens = tokenize_or_die(vocab, "This was wrong, frustrating, awful, and useless. I hate this result.");
        const std::vector<llama_token> failed_tool_tokens = tokenize_or_die(vocab, "Tool failed and produced the wrong result again.");
        if (llama_self_state_upsert_goal(ctx, 13, goal_tokens.data(), goal_tokens.size(), 1.0f) != 0 ||
            llama_self_state_upsert_memory_handle(ctx, 33, LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER, complaint_tokens.data(), complaint_tokens.size(), 1.0f) != 0) {
            std::fprintf(stderr, "failed to seed repair-governance goals\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_self_updater_program repair_policy = {};
        if (llama_self_state_get_updater_program(ctx, &repair_policy) != 0) {
            std::fprintf(stderr, "failed to fetch repair-governance updater policy\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        repair_policy.version += 1;
        repair_policy.repair_emit_threshold = 0.45f;
        repair_policy.repair_dissatisfaction_floor = 0.20f;
        repair_policy.repair_recent_user_valence_floor = 0.12f;
        repair_policy.repair_inhibition_max = 0.82f;
        repair_policy.repair_admission_floor = 0.18f;
        repair_policy.repair_admission_weight = 0.55f;
        if (llama_self_state_set_updater_program(ctx, repair_policy) != 0) {
            std::fprintf(stderr, "failed to set repair-governance updater policy\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const llama_self_state_event complaint_event = {
            /*.tokens =*/ complaint_tokens.data(),
            /*.n_tokens =*/ complaint_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ 0,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        for (int i = 0; i < 5; ++i) {
            if (!ingest_event_without_runner("seed-repair-governance-complaint", ctx, complaint_event)) {
                std::fprintf(stderr, "failed to seed repair-governance dissatisfaction\n");
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }

        const llama_self_state_event failed_tool_event = {
            /*.tokens =*/ failed_tool_tokens.data(),
            /*.n_tokens =*/ failed_tool_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_TOOL_FAILED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 0.0f,
        };
        if (!ingest_event_without_runner("seed-repair-governance-tool-failure", ctx, failed_tool_event)) {
            std::fprintf(stderr, "failed to seed repair-governance tool failure\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (llama_self_state_upsert_tool_job(ctx, 41, LLAMA_SELF_TOOL_JOB_FAILED, 1.0f) != 0) {
            std::fprintf(stderr, "failed to seed repair-governance tool-state pressure\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_self_social_state_info social = {};
        if (llama_self_state_get_social_state(ctx, &social) != 0 ||
            social.dissatisfaction <= 0.15f ||
            social.recent_user_valence <= 0.10f) {
            std::fprintf(stderr, "repair-governance social state did not retain dissatisfaction signals\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_dmn_tick_trace trace = {};
        if (llama_dmn_tick(ctx, 9000, &trace) != 0) {
            std::fprintf(stderr, "repair-governance DMN tick failed\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (trace.admitted) {
            llama_governance_trace governance = {};
            if (llama_governance_get_last_trace(ctx, &governance) != 0 ||
                governance.proposal_family < -1) {
                std::fprintf(stderr, "DMN governance did not preserve repair-state signals under sustained dissatisfaction\n");
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            if (governance.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR &&
                (!governance.repair_rendered || governance.repair_message_length <= 0)) {
                std::fprintf(stderr, "repair-governance outcome did not include a repair message\n");
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }

        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
