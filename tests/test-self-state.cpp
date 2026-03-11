#include "get-model.h"
#include "llama.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize_or_die(const llama_vocab * vocab, const std::string & text) {
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize self-state test prompt\n");
        std::exit(1);
    }
    return tokens;
}

static bool contradiction_head_callback(
        const llama_self_state_feature_vector * features,
        float * out_score,
        void * /*user_data*/) {
    if (!features || !out_score) {
        return false;
    }

    *out_score = 0.85f;
    return true;
}

static bool uncertainty_head_callback(
        const llama_self_state_feature_vector * features,
        float * out_score,
        void * /*user_data*/) {
    if (!features || !out_score) {
        return false;
    }

    *out_score = 0.75f;
    return true;
}

static bool broadcast_head_callback(
        const llama_self_state_feature_vector * features,
        float * out_score,
        void * /*user_data*/) {
    if (!features || !out_score) {
        return false;
    }

    *out_score = 0.65f;
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

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (llama_self_state_register_count(ctx) != LLAMA_SELF_REGISTER_COUNT) {
        std::fprintf(stderr, "unexpected self-state register count\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_state_datetime datetime = {};
    if (llama_self_state_get_datetime(ctx, &datetime) != 0) {
        std::fprintf(stderr, "failed to query self-state datetime\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (datetime.wall_clock_ms <= 0 || datetime.monotonic_ms <= 0 ||
        datetime.local_hour < 0 || datetime.local_hour > 23 ||
        datetime.local_minute < 0 || datetime.local_minute > 59) {
        std::fprintf(stderr, "unexpected default self-state datetime\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_register_info register_info = {};
    if (llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_TIME_PHASE, &register_info) != 0 ||
        register_info.family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR ||
        register_info.scalar_value < 0.0f || register_info.scalar_value > 1.0f) {
        std::fprintf(stderr, "unexpected time-phase register\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!llama_self_state_register_name(LLAMA_SELF_REGISTER_CHANNEL_STATE)) {
        std::fprintf(stderr, "missing channel-state register name\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_self_state_time_point t0 = {
        /*.wall_clock_ms =*/ 1700000000000,
        /*.monotonic_ms =*/ 1000,
        /*.timezone_offset_minutes =*/ -360,
    };

    if (llama_self_state_set_time(ctx, t0) != 0) {
        std::fprintf(stderr, "failed to apply deterministic time point\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_note_user_event(ctx) != 0 ||
        llama_self_state_note_tool_event(ctx) != 0 ||
        llama_self_state_note_emit_event(ctx) != 0) {
        std::fprintf(stderr, "failed to apply self-state event anchors\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_self_state_time_point t1 = {
        /*.wall_clock_ms =*/ 1700000300000,
        /*.monotonic_ms =*/ 301000,
        /*.timezone_offset_minutes =*/ -360,
    };

    if (llama_self_state_set_time(ctx, t1) != 0 ||
        llama_self_state_get_datetime(ctx, &datetime) != 0) {
        std::fprintf(stderr, "failed to advance self-state time\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (datetime.session_age_ms != 300000 ||
        datetime.delta_since_last_user_ms != 300000 ||
        datetime.delta_since_last_tool_event_ms != 300000 ||
        datetime.delta_since_last_emit_ms != 300000) {
        std::fprintf(stderr, "unexpected elapsed self-state deltas\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_TOOL_SALIENCE, &register_info) != 0 ||
        register_info.scalar_value < 0.49f || register_info.scalar_value > 0.51f ||
        (register_info.source_mask & LLAMA_SELF_SOURCE_TOOL_EVENT) == 0) {
        std::fprintf(stderr, "unexpected tool-salience register\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_set_channel_state(ctx, LLAMA_SELF_STATE_CHANNEL_ACTIVE) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_CHANNEL_STATE, &register_info) != 0 ||
        register_info.family != LLAMA_SELF_REGISTER_FAMILY_CATEGORICAL ||
        register_info.categorical_value != LLAMA_SELF_STATE_CHANNEL_ACTIVE) {
        std::fprintf(stderr, "unexpected channel-state register\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_self_state_time_point invalid_time = {
        /*.wall_clock_ms =*/ 1700000299000,
        /*.monotonic_ms =*/ 300999,
        /*.timezone_offset_minutes =*/ -360,
    };

    if (llama_self_state_set_time(ctx, invalid_time) == 0) {
        std::fprintf(stderr, "accepted non-monotonic self-state time\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> identity_tokens = tokenize_or_die(vocab, "I am Vicuna and I maintain the strict memory runtime.");
    const std::vector<llama_token> goal_tokens = tokenize_or_die(vocab, "Finish the strict memory runtime implementation.");
    const std::vector<llama_token> commitment_tokens = tokenize_or_die(vocab, "I will finish the strict memory runtime implementation.");
    const std::vector<llama_token> memory_handle_tokens = tokenize_or_die(vocab, "strict memory runtime implementation working memory cluster");

    if (llama_self_state_set_identity(ctx, identity_tokens.data(), identity_tokens.size()) != 0 ||
        llama_self_state_upsert_goal(ctx, 7, goal_tokens.data(), goal_tokens.size(), 1.0f) != 0 ||
        llama_self_state_upsert_commitment(ctx, 11, commitment_tokens.data(), commitment_tokens.size(), 1.0f, true) != 0 ||
        llama_self_state_upsert_memory_handle(ctx, 17, LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER, memory_handle_tokens.data(), memory_handle_tokens.size(), 0.9f) != 0 ||
        llama_self_state_goal_count(ctx) != 1 ||
        llama_self_state_commitment_count(ctx) != 1 ||
        llama_self_state_memory_handle_count(ctx) != 1 ||
        llama_self_state_working_memory_count(ctx) != 0) {
        std::fprintf(stderr, "failed to configure self-state retrieval surfaces\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> commitment_event_tokens = tokenize_or_die(vocab, "I will finish the strict memory runtime implementation today.");
    llama_self_state_event commitment_event = {
        /*.tokens =*/ commitment_event_tokens.data(),
        /*.n_tokens =*/ commitment_event_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
        /*.decoder_entropy =*/ 0.4f,
        /*.decoder_top_margin =*/ 0.7f,
    };

    llama_self_state_feature_vector commitment_features = {};
    llama_self_state_feature_vector commitment_postwrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &commitment_event, &commitment_features) != 0 ||
        commitment_features.goal_top_similarity <= 0.05f ||
        commitment_features.commitment_top_similarity <= 0.05f ||
        commitment_features.identity_similarity <= 0.05f ||
        llama_self_state_apply_prewrite(ctx, &commitment_event, &commitment_features) != 0 ||
        llama_self_state_build_postwrite_features(ctx, &commitment_event, &commitment_postwrite) != 0 ||
        llama_self_state_apply_postwrite(ctx, &commitment_event, &commitment_postwrite) != 0 ||
        llama_self_state_working_memory_count(ctx) != 1 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_GOAL_RELEVANCE, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f) {
        std::fprintf(stderr, "self-state retrieval surfaces did not influence updates\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> related_event_tokens = tokenize_or_die(vocab, "strict memory runtime implementation");
    llama_self_state_event related_event = {
        /*.tokens =*/ related_event_tokens.data(),
        /*.n_tokens =*/ related_event_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ 0,
        /*.decoder_entropy =*/ 0.2f,
        /*.decoder_top_margin =*/ 0.8f,
    };

    llama_self_state_feature_vector retrieval_features = {};
    if (llama_self_state_build_prewrite_features(ctx, &related_event, &retrieval_features) != 0 ||
        retrieval_features.working_memory_top_similarity <= 0.05f ||
        retrieval_features.memory_handle_top_similarity <= 0.05f) {
        std::fprintf(stderr, "working-memory retrieval features were not populated\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params active = llama_active_lora_default_params();
    active.enabled = true;
    active.host_memory_ratio = 1.0f;
    active.device_memory_ratio = 1.0f;
    active.min_rank = 1;
    active.max_rank = 2;
    active.max_updates_before_rollover = 1;
    active.gain_max = 0.25f;

    llama_past_lora_params past = llama_past_lora_default_params();
    configure_small_buckets(past);
    const std::vector<llama_token> frozen_tokens = tokenize_or_die(vocab, "A frozen memory test span that should push the active adapter over its rollover boundary.");

    if (llama_active_lora_init(ctx, active) != 0 ||
        llama_past_lora_init(ctx, past) != 0 ||
        llama_active_lora_ingest(ctx, frozen_tokens.data(), frozen_tokens.size()) != 0 ||
        llama_past_lora_tick(ctx, 100) != 0 ||
        llama_self_state_memory_handle_count(ctx) < 2) {
        std::fprintf(stderr, "failed to bridge frozen LoRA buckets into self-state handles\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> user_tokens = tokenize_or_die(vocab, "I am not sure this answer is correct.");
    llama_self_state_event user_event = {
        /*.tokens =*/ user_tokens.data(),
        /*.n_tokens =*/ user_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
        /*.decoder_entropy =*/ 2.5f,
        /*.decoder_top_margin =*/ 0.20f,
    };

    llama_self_state_feature_vector prewrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &user_event, &prewrite) != 0 ||
        prewrite.novelty < 0.0f || prewrite.novelty > 1.0f ||
        prewrite.uncertainty_score <= 0.0f ||
        prewrite.contradiction_score <= 0.0f) {
        std::fprintf(stderr, "unexpected prewrite feature vector\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_apply_prewrite(ctx, &user_event, &prewrite) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_UNCERTAINTY, &register_info) != 0 ||
        register_info.scalar_value < 0.0f || register_info.scalar_value > 1.0f) {
        std::fprintf(stderr, "failed to apply prewrite features\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_state_feature_vector postwrite = {};
    if (llama_self_state_build_postwrite_features(ctx, &user_event, &postwrite) != 0 ||
        llama_self_state_apply_postwrite(ctx, &user_event, &postwrite) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_PRESSURE, &register_info) != 0 ||
        register_info.scalar_value < 0.0f || register_info.scalar_value > 1.0f) {
        std::fprintf(stderr, "failed to apply postwrite features\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_state_params params = llama_self_state_default_params();
    params.enable_learned_contradiction_head = true;
    params.enable_learned_uncertainty_head = true;
    params.enable_learned_broadcast_head = true;
    params.contradiction_head_callback = contradiction_head_callback;
    params.uncertainty_head_callback = uncertainty_head_callback;
    params.broadcast_head_callback = broadcast_head_callback;

    if (llama_self_state_configure(ctx, params) != 0) {
        std::fprintf(stderr, "failed to configure learned self-state heads\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> tool_tokens = tokenize_or_die(vocab, "Tool failed with error: cannot open file.");
    llama_self_state_event tool_event = {
        /*.tokens =*/ tool_tokens.data(),
        /*.n_tokens =*/ tool_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_TOOL_FAILED,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 0.0f,
    };

    llama_self_state_feature_vector learned_prewrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &tool_event, &learned_prewrite) != 0 ||
        learned_prewrite.contradiction_score < 0.84f ||
        learned_prewrite.uncertainty_score < 0.74f) {
        std::fprintf(stderr, "learned heads did not influence feature vector\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_apply_prewrite(ctx, &tool_event, &learned_prewrite) != 0 ||
        llama_self_state_build_postwrite_features(ctx, &tool_event, &postwrite) != 0 ||
        postwrite.broadcast_pressure_hint < 0.64f ||
        llama_self_state_apply_postwrite(ctx, &tool_event, &postwrite) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_CONTRADICTION, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f) {
        std::fprintf(stderr, "failed learned-head register update path\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_upsert_tool_job(ctx, 3, LLAMA_SELF_TOOL_JOB_PENDING, 0.6f) != 0 ||
        llama_self_state_upsert_tool_job(ctx, 4, LLAMA_SELF_TOOL_JOB_RUNNING, 0.8f) != 0) {
        std::fprintf(stderr, "failed to update tool jobs\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_tool_state_info tool_state = {};
    if (llama_self_state_get_tool_state(ctx, &tool_state) != 0 ||
        tool_state.pending_jobs != 1 ||
        tool_state.running_jobs != 1 ||
        tool_state.readiness >= 0.5f) {
        std::fprintf(stderr, "unexpected tool-state surface\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_build_postwrite_features(ctx, &tool_event, &postwrite) != 0 ||
        postwrite.tool_pending_pressure <= 0.0f ||
        postwrite.tool_readiness_score >= 0.5f) {
        std::fprintf(stderr, "tool lifecycle features were not populated\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_social_state_info social_state = {};
    if (llama_self_state_get_social_state(ctx, &social_state) != 0 ||
        social_state.familiarity <= 0.0f ||
        social_state.trust <= 0.0f || social_state.trust > 1.0f ||
        social_state.reciprocity <= 0.0f ||
        social_state.bond_strength <= 0.0f ||
        social_state.user_turn_count < 2 ||
        social_state.system_turn_count < 1) {
        std::fprintf(stderr, "social relationship state was not maintained as persistent scalars\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_reactivation_info reactivation = {};
    if (llama_self_state_reactivation_count(ctx) <= 0 ||
        llama_self_state_get_reactivation(ctx, 0, &reactivation) != 0 ||
        reactivation.handle_id != 17 ||
        reactivation.priority <= 0.0f ||
        reactivation.top_similarity <= 0.0f) {
        std::fprintf(stderr, "reactivation priorities were not updated\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_trace_count(ctx) != 3 ||
        llama_self_state_replay_trace(ctx, 1) != 0 ||
        llama_self_state_working_memory_count(ctx) != 1 ||
        llama_self_state_trace_count(ctx) != 3 ||
        llama_self_state_replay_trace(ctx, -1) != 0 ||
        llama_self_state_working_memory_count(ctx) != 3 ||
        llama_self_state_trace_count(ctx) != 3) {
        std::fprintf(stderr, "trace replay did not preserve deterministic state\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_updater_program program = llama_self_state_default_updater_program();
    program.version = 2;
    program.broadcast_social_weight = 0.05f;
    program.broadcast_contradiction_weight = 0.45f;
    program.broadcast_goal_weight = 0.25f;
    if (program.rule_count == 0) {
        std::fprintf(stderr, "default updater program is missing register rules\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    for (uint32_t i = 0; i < program.rule_count; ++i) {
        if (program.rules[i].register_id == LLAMA_SELF_REGISTER_BROADCAST_PRESSURE &&
            program.rules[i].phase_mask == LLAMA_SELF_UPDATER_PHASE_POSTWRITE) {
            program.rules[i].rise_gain = 0.75f;
            program.rules[i].feature_weights[0] = 0.82f;
        }
    }

    if (llama_self_state_set_updater_program(ctx, program) != 0) {
        std::fprintf(stderr, "failed to set updater program\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_updater_program invalid_program = program;
    invalid_program.version = 3;
    invalid_program.rules[0].register_id = LLAMA_SELF_REGISTER_CHANNEL_STATE;
    if (llama_self_state_set_updater_program(ctx, invalid_program) == 0) {
        std::fprintf(stderr, "accepted invalid updater program rule\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_updater_program program_roundtrip = {};
    llama_self_counterfactual_result counterfactual = {};
    if (llama_self_state_get_updater_program(ctx, &program_roundtrip) != 0 ||
        program_roundtrip.version != 2 ||
        program_roundtrip.rule_count == 0 ||
        program_roundtrip.rules[0].phase_mask == 0 ||
        llama_self_state_evaluate_counterfactual(ctx, program, -1, &counterfactual) != 0 ||
        counterfactual.updater_version != 2 ||
        counterfactual.replay_channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ||
        counterfactual.replayed_events != 3 ||
        counterfactual.working_memory_count != 3 ||
        counterfactual.broadcast_pressure <= 0.0f) {
        std::fprintf(stderr, "counterfactual updater evaluation failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const size_t trace_size = llama_self_state_trace_export_size(ctx);
    std::vector<uint8_t> trace_blob(trace_size);
    if (trace_size == 0 ||
        llama_self_state_trace_export(ctx, trace_blob.data(), trace_blob.size()) != 0) {
        std::fprintf(stderr, "failed to export self-state trace\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_context * counterfactual_ctx = llama_init_from_model(model, cparams);
    if (!counterfactual_ctx) {
        std::fprintf(stderr, "failed to create counterfactual context\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (llama_self_state_trace_import(counterfactual_ctx, trace_blob.data(), trace_blob.size(), true) != 0 ||
        llama_self_state_replay_trace_on_channel(counterfactual_ctx, -1, LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) != 0 ||
        llama_self_state_get_social_state(counterfactual_ctx, &social_state) != 0 ||
        social_state.user_turn_count != 0 ||
        social_state.system_turn_count != 0 ||
        llama_self_state_get_register(counterfactual_ctx, LLAMA_SELF_REGISTER_CHANNEL_STATE, &register_info) != 0 ||
        register_info.categorical_value != LLAMA_SELF_STATE_CHANNEL_WAITING) {
        std::fprintf(stderr, "counterfactual replay leaked into primary-channel state\n");
        llama_free(counterfactual_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    llama_free(counterfactual_ctx);

    llama_context * imported_ctx = llama_init_from_model(model, cparams);
    if (!imported_ctx) {
        std::fprintf(stderr, "failed to create imported context\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_trace_import(imported_ctx, trace_blob.data(), trace_blob.size(), true) != 0 ||
        llama_self_state_trace_count(imported_ctx) != 3 ||
        llama_self_state_replay_trace(imported_ctx, -1) != 0 ||
        llama_self_state_working_memory_count(imported_ctx) != 3) {
        std::fprintf(stderr, "trace import/replay failed\n");
        llama_free(imported_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(imported_ctx);

    if (llama_self_state_clear_trace(ctx) != 0 ||
        llama_self_state_trace_count(ctx) != 0) {
        std::fprintf(stderr, "trace clear failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
