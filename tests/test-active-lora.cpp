#include "get-model.h"
#include "llama.h"
#include "../src/llama-context.h"

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static bool test_callback_embedder(
        const llama_context * /*ctx*/,
        const llama_token * tokens,
        size_t n_tokens,
        float * out_embedding,
        size_t n_embedding,
        void * /*user_data*/) {
    if (!tokens || !out_embedding || n_embedding == 0) {
        return false;
    }

    for (size_t i = 0; i < n_embedding; ++i) {
        out_embedding[i] = 0.0f;
    }

    for (size_t i = 0; i < n_tokens; ++i) {
        out_embedding[i % n_embedding] += (tokens[i] & 1) ? -1.0f : 1.0f;
    }

    return true;
}

static std::vector<llama_token> tokenize_or_die(const llama_vocab * vocab, const std::string & text) {
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize test prompt\n");
        std::exit(1);
    }
    return tokens;
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create primary context\n");
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params params = llama_active_lora_default_params();
    params.enabled = true;
    params.host_memory_ratio = 1.0f;
    params.device_memory_ratio = 1.0f;
    params.min_rank = 1;
    params.max_rank = 2;
    params.train_context_tokens = 8;
    params.train_stride_tokens = 4;
    params.max_updates_before_rollover = 1;

    if (llama_active_lora_init(ctx, params) != 0) {
        fprintf(stderr, "failed to initialize Active LoRA\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_stats stats = {};
    if (llama_active_lora_get_stats(ctx, &stats) != 0) {
        fprintf(stderr, "failed to query Active LoRA stats\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_user_personality_lora_stats user_stats = {};
    if (llama_user_personality_lora_get_stats(ctx, &user_stats) != 0) {
        fprintf(stderr, "failed to query user personality LoRA stats\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!stats.enabled || stats.selected_rank < 1 ||
        stats.optimizer_step_count != 0 ||
        stats.embedding_type != LLAMA_ACTIVE_LORA_EMBEDDING_HIDDEN_STATE ||
        stats.embedding_is_custom || stats.embedding_dim != (uint32_t) llama_model_n_embd_out(model) ||
        stats.optimizer_last_update_norm != 0.0f) {
        fprintf(stderr, "unexpected Active LoRA init stats\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (!user_stats.enabled || user_stats.attached_for_simulation ||
        user_stats.selected_rank < 1 ||
        user_stats.updates_applied != 0 ||
        user_stats.optimizer_step_count != 0 ||
        user_stats.tokens_ingested != 0 ||
        user_stats.confidence != 0.0f) {
        fprintf(stderr, "unexpected user personality LoRA init stats\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_functional_lora_family_count(ctx) != LLAMA_FUNCTIONAL_LORA_COUNT) {
        fprintf(stderr, "unexpected functional LoRA family count\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        llama_functional_lora_family_config config = {};
        llama_functional_lora_family_state state = {};
        llama_functional_lora_snapshot_archive archive = {};
        if (llama_functional_lora_family_config_get(ctx, family, &config) != 0 ||
            llama_functional_lora_family_state_get(ctx, family, &state) != 0 ||
            llama_functional_lora_snapshot_archive_get(ctx, family, &archive) != 0 ||
            !config.enabled ||
            config.gain_clip_min != 0.0f ||
            config.gain_clip_max != 2.0f ||
            config.default_gain < 0.999f ||
            config.default_gain > 1.001f ||
            config.exploration_noise_initial_std <= config.exploration_noise_min_std ||
            config.bootstrap_perturbation_initial_std <= config.bootstrap_perturbation_min_std ||
            config.bootstrap_perturbation_min_std <= 0.0f ||
            config.bootstrap_weight_init_std <= 0.0f ||
            !state.compatible ||
            state.active_now ||
            state.current_gain != 0.0f ||
            state.predicted_gain < 0.999f ||
            state.predicted_gain > 1.001f ||
            state.last_noise != 0.0f ||
            state.current_bootstrap_std != config.bootstrap_perturbation_initial_std ||
            state.last_bootstrap_perturbation != 0.0f ||
            state.activation_count != 0 ||
            state.last_meta_loss != 0.0f ||
            archive.count != 0) {
            fprintf(stderr, "unexpected functional LoRA registry state for family %d\n", family);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }
    llama_functional_lora_trace functional_trace = {};
    if (llama_functional_lora_get_last_trace(ctx, &functional_trace) != 0 ||
        functional_trace.last_activation.microphase != 0) {
        fprintf(stderr, "unexpected default functional LoRA trace\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    llama_active_temporal_encoding_bias temporal_bias = {};
    if (llama_active_temporal_encoding_bias_get(ctx, &temporal_bias) != 0 ||
        temporal_bias.reward_bias != 0.0f ||
        temporal_bias.dampening_bias != 0.0f ||
        temporal_bias.effective_write_scale < 0.999f ||
        temporal_bias.effective_write_scale > 1.001f ||
        temporal_bias.last_update_norm != 0.0f ||
        temporal_bias.adam_step != 0 ||
        temporal_bias.applied_update_count != 0 ||
        temporal_bias.last_update_monotonic_ms != 0) {
        fprintf(stderr, "unexpected default temporal encoding bias\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt;
    for (int i = 0; i < 8; ++i) {
        prompt += "The quick brown fox jumps over the lazy dog. ";
    }
    std::vector<llama_token> tokens = tokenize_or_die(vocab, prompt);

    if (llama_active_lora_ingest(ctx, tokens.data(), tokens.size()) != 0) {
        fprintf(stderr, "failed to ingest evicted span\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_active_lora_get_stats(ctx, &stats) != 0) {
        fprintf(stderr, "failed to query Active LoRA stats after ingest\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (stats.updates_applied < 1 || stats.optimizer_step_count < 1 || stats.tokens_ingested < tokens.size() || !stats.rollover_ready ||
        stats.gain_mean <= 0.0f || stats.gain_max <= 0.0f || stats.gain_max > params.gain_max + 1.0e-6f ||
        stats.optimizer_last_update_norm <= 0.0f) {
        fprintf(stderr, "unexpected Active LoRA ingest stats\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (llama_user_personality_lora_get_stats(ctx, &user_stats) != 0 ||
        user_stats.updates_applied != 0 ||
        user_stats.tokens_ingested != 0 ||
        user_stats.optimizer_step_count != 0 ||
        user_stats.confidence != 0.0f) {
        fprintf(stderr, "evicted-span ingest unexpectedly changed user personality LoRA\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_context * ctx_pool = llama_init_from_model(model, cparams);
    if (!ctx_pool) {
        fprintf(stderr, "failed to create secondary context\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params pool_params = llama_active_lora_default_params();
    pool_params.enabled = true;
    pool_params.host_memory_ratio = 1.0f;
    pool_params.device_memory_ratio = 1.0f;
    pool_params.min_rank = 1;
    pool_params.max_rank = 2;
    pool_params.train_context_tokens = 8;
    pool_params.train_stride_tokens = 4;
    pool_params.embedding_type = LLAMA_ACTIVE_LORA_EMBEDDING_TOKEN_POOL;

    if (llama_active_lora_init(ctx_pool, pool_params) != 0) {
        fprintf(stderr, "failed to initialize token-pool Active LoRA\n");
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_stats pool_stats = {};
    if (llama_active_lora_get_stats(ctx_pool, &pool_stats) != 0 ||
        pool_stats.embedding_type != LLAMA_ACTIVE_LORA_EMBEDDING_TOKEN_POOL ||
        pool_stats.embedding_is_custom || pool_stats.embedding_dim != 64) {
        fprintf(stderr, "unexpected token-pool Active LoRA stats\n");
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_context * ctx_hash = llama_init_from_model(model, cparams);
    if (!ctx_hash) {
        fprintf(stderr, "failed to create explicit hash context\n");
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params hash_params = llama_active_lora_default_params();
    hash_params.enabled = true;
    hash_params.host_memory_ratio = 1.0f;
    hash_params.device_memory_ratio = 1.0f;
    hash_params.min_rank = 1;
    hash_params.max_rank = 2;
    hash_params.embedding_dim = 64;
    hash_params.embedding_type = LLAMA_ACTIVE_LORA_EMBEDDING_HASH;

    if (llama_active_lora_init(ctx_hash, hash_params) != 0) {
        fprintf(stderr, "failed to initialize explicit-hash Active LoRA\n");
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_stats hash_stats = {};
    if (llama_active_lora_get_stats(ctx_hash, &hash_stats) != 0 ||
        hash_stats.embedding_type != LLAMA_ACTIVE_LORA_EMBEDDING_HASH ||
        hash_stats.embedding_is_custom || hash_stats.embedding_dim != 64) {
        fprintf(stderr, "unexpected explicit-hash Active LoRA stats\n");
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_context * ctx_custom = llama_init_from_model(model, cparams);
    if (!ctx_custom) {
        fprintf(stderr, "failed to create callback context\n");
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params custom_params = llama_active_lora_default_params();
    custom_params.enabled = true;
    custom_params.host_memory_ratio = 1.0f;
    custom_params.device_memory_ratio = 1.0f;
    custom_params.min_rank = 1;
    custom_params.max_rank = 2;
    custom_params.embedding_dim = 16;
    custom_params.embedding_callback = test_callback_embedder;
    custom_params.embedding_type = LLAMA_ACTIVE_LORA_EMBEDDING_HASH;

    if (llama_active_lora_init(ctx_custom, custom_params) != 0) {
        fprintf(stderr, "failed to initialize callback Active LoRA\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_stats custom_stats = {};
    if (llama_active_lora_get_stats(ctx_custom, &custom_stats) != 0 ||
        !custom_stats.embedding_is_custom || custom_stats.embedding_dim != 16) {
        fprintf(stderr, "unexpected callback Active LoRA stats\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_active_lora_ingest(ctx_custom, tokens.data(), tokens.size()) != 0) {
        fprintf(stderr, "failed to ingest with callback Active LoRA\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_active_lora_get_stats(ctx_custom, &custom_stats) != 0 || custom_stats.updates_applied < 1) {
        fprintf(stderr, "unexpected callback Active LoRA ingest stats\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const uint32_t updates_before_redundant = stats.updates_applied;
    if (llama_active_lora_ingest(ctx, tokens.data(), tokens.size()) != 0) {
        fprintf(stderr, "failed to ingest redundant Active LoRA span\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_active_lora_get_stats(ctx, &stats) != 0 || stats.updates_applied != updates_before_redundant) {
        fprintf(stderr, "redundant Active LoRA span unexpectedly changed weights\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const uint64_t one_week_us = 7ull * 24ull * 60ull * 60ull * 1000000ull;
    if (llama_functional_lora_snapshot_maintain(ctx, 1) != 0 ||
        llama_functional_lora_snapshot_maintain(ctx, one_week_us + 1) != 0) {
        fprintf(stderr, "failed to run functional snapshot maintenance\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_functional_snapshot_maintenance_trace maintenance = {};
    if (llama_functional_lora_get_last_snapshot_maintenance(ctx, &maintenance) != 0 ||
        !maintenance.ran ||
        !maintenance.captured_any ||
        maintenance.captured_count != (uint32_t) LLAMA_FUNCTIONAL_LORA_COUNT) {
        fprintf(stderr, "unexpected functional snapshot maintenance trace\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        llama_functional_lora_snapshot_archive archive = {};
        if (llama_functional_lora_snapshot_archive_get(ctx, family, &archive) != 0 ||
            archive.count != 1 ||
            archive.family != family ||
            archive.last_capture_us != one_week_us + 1 ||
            archive.next_capture_due_us <= archive.last_capture_us ||
            !archive.items[0].valid) {
            fprintf(stderr, "unexpected functional snapshot archive contents for family %d\n", family);
            llama_free(ctx_custom);
            llama_free(ctx_hash);
            llama_free(ctx_pool);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    llama_functional_lora_snapshot_info snapshot_info = {};
    const int32_t snapshot_family = LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    if (llama_functional_lora_snapshot_info_get(ctx, snapshot_family, 0, &snapshot_info) != 0 ||
        !snapshot_info.valid) {
        fprintf(stderr, "failed to retrieve functional snapshot info\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const size_t snapshot_blob_size = llama_functional_lora_snapshot_blob_size(ctx, snapshot_family, 0);
    std::vector<uint8_t> snapshot_blob(snapshot_blob_size);
    if (snapshot_blob_size == 0 ||
        llama_functional_lora_snapshot_blob_export(ctx, snapshot_family, 0, snapshot_blob.data(), snapshot_blob.size()) != 0 ||
        llama_functional_lora_snapshot_blob_import(ctx_pool, snapshot_family, 0, snapshot_info, snapshot_blob.data(), snapshot_blob.size()) != 0) {
        fprintf(stderr, "failed to roundtrip functional snapshot blob\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_functional_lora_snapshot_archive imported_archive = {};
    llama_functional_lora_snapshot_info imported_info = {};
    if (llama_functional_lora_snapshot_archive_get(ctx_pool, snapshot_family, &imported_archive) != 0 ||
        llama_functional_lora_snapshot_info_get(ctx_pool, snapshot_family, 0, &imported_info) != 0 ||
        imported_archive.count != 1 ||
        !imported_info.valid ||
        imported_info.snapshot_id != snapshot_info.snapshot_id ||
        imported_info.captured_at_us != snapshot_info.captured_at_us) {
        fprintf(stderr, "functional snapshot blob import did not restore metadata\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_process_functional_signature process_signature = {};
    process_signature.valid = true;
    process_signature.signature_hash = 0x1234ULL;
    process_signature.scope_kind = LLAMA_PROCESS_FUNCTIONAL_SCOPE_PROCESS_STEP;
    process_signature.family = LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    process_signature.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    process_signature.microphase = LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE;
    process_signature.plan_mode = LLAMA_COG_PLAN_MODE_COMPOSITION;
    process_signature.plan_step_kind = LLAMA_COG_PLAN_STEP_OBSERVE_TOOL;
    process_signature.tool_kind = LLAMA_TOOL_KIND_BASH_CLI;
    process_signature.source_family = LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    process_signature.requires_tool_result = true;
    process_signature.transient_plan_id = 7;
    process_signature.transient_step_index = 1;
    process_signature.transient_source_id = 11;
    std::snprintf(process_signature.tool_name, sizeof(process_signature.tool_name), "%s", "bash-cli");
    std::snprintf(process_signature.capability_id, sizeof(process_signature.capability_id), "%s", "openclaw/local/exec");
    std::snprintf(process_signature.provenance_namespace, sizeof(process_signature.provenance_namespace), "%s", "openclaw/local/exec");
    std::snprintf(process_signature.semantic_key, sizeof(process_signature.semantic_key), "%s", "dmn/counterfactual_compare/step_kind:4/tool:bash-cli");
    if (!ctx->process_functional_set_execution(process_signature)) {
        fprintf(stderr, "failed to seed process-functional execution signature\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> process_tokens = tokenize_or_die(vocab, "process specialization weak outcome");
    const llama_self_state_event process_event = {
        /*.tokens =*/ process_tokens.data(),
        /*.n_tokens =*/ process_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
    };
    const llama_functional_outcome_snapshot process_before = {
        /*.favorable_divergence =*/ 0.8f,
        /*.user_satisfaction_risk =*/ 0.3f,
        /*.goal_progress_pressure =*/ 0.4f,
        /*.loop_inefficiency =*/ 0.5f,
        /*.recovery_urgency =*/ 0.2f,
        /*.answerability =*/ 0.5f,
        /*.preference_uncertainty =*/ 0.4f,
        /*.expected_steps_remaining =*/ 0.6f,
        /*.expected_inference_cost_remaining =*/ 0.5f,
    };
    const float process_metrics[] = { -0.2f, 0.0f, 0.0f };
    for (int i = 0; i < 6; ++i) {
        if (!ctx->functional_lora_apply_update(
                    LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL,
                    LLAMA_COG_COMMAND_ORIGIN_DMN,
                    LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_GENERATE,
                    LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE,
                    process_before,
                    process_before,
                    LLAMA_TOOL_KIND_BASH_CLI,
                    1,
                    process_metrics,
                    sizeof(process_metrics)/sizeof(process_metrics[0]),
                    -0.2f,
                    0.6f,
                    process_event,
                    nullptr)) {
            fprintf(stderr, "failed to accumulate process-functional evidence\n");
            llama_free(ctx_custom);
            llama_free(ctx_hash);
            llama_free(ctx_pool);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    llama_process_functional_entry_info process_entry = {};
    if (llama_process_functional_entry_count(ctx) < 1 ||
        llama_process_functional_entry_get(ctx, 0, &process_entry) != 0 ||
        !process_entry.valid ||
        process_entry.signature.signature_hash != process_signature.signature_hash) {
        fprintf(stderr, "process-functional entry was not created after repeated weak outcomes\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_process_functional_snapshot_maintain(ctx, 1) != 0 ||
        llama_process_functional_snapshot_maintain(ctx, one_week_us + 1) != 0) {
        fprintf(stderr, "failed to run process-functional snapshot maintenance\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_functional_snapshot_maintenance_trace process_maintenance = {};
    llama_functional_lora_snapshot_archive process_archive = {};
    llama_functional_lora_snapshot_info process_snapshot = {};
    if (llama_process_functional_get_last_snapshot_maintenance(ctx, &process_maintenance) != 0 ||
        !process_maintenance.ran ||
        !process_maintenance.captured_any ||
        process_maintenance.captured_count != 1 ||
        llama_process_functional_snapshot_archive_get(ctx, process_entry.slot, &process_archive) != 0 ||
        llama_process_functional_snapshot_info_get(ctx, process_entry.slot, 0, &process_snapshot) != 0 ||
        process_archive.count != 1 ||
        process_archive.family != LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL ||
        process_archive.last_capture_us != one_week_us + 1 ||
        process_archive.next_capture_due_us <= process_archive.last_capture_us ||
        !process_snapshot.valid) {
        fprintf(stderr, "unexpected process-functional snapshot archive contents\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const size_t process_entry_blob_size = llama_process_functional_entry_blob_size(ctx, process_entry.slot);
    const size_t process_snapshot_blob_size = llama_process_functional_snapshot_blob_size(ctx, process_entry.slot, 0);
    std::vector<uint8_t> process_entry_blob(process_entry_blob_size);
    std::vector<uint8_t> process_snapshot_blob(process_snapshot_blob_size);
    if (process_entry_blob_size == 0 ||
        process_snapshot_blob_size == 0 ||
        llama_process_functional_entry_blob_export(ctx, process_entry.slot, process_entry_blob.data(), process_entry_blob.size()) != 0 ||
        llama_process_functional_snapshot_blob_export(ctx, process_entry.slot, 0, process_snapshot_blob.data(), process_snapshot_blob.size()) != 0 ||
        llama_process_functional_entry_blob_import(ctx_pool, process_entry.slot, &process_entry, process_entry_blob.data(), process_entry_blob.size()) != 0 ||
        llama_process_functional_snapshot_blob_import(ctx_pool, process_entry.slot, 0, process_snapshot, process_snapshot_blob.data(), process_snapshot_blob.size()) != 0) {
        fprintf(stderr, "failed to roundtrip process-functional lifecycle blobs\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_functional_lora_snapshot_archive imported_process_archive = {};
    llama_functional_lora_snapshot_info imported_process_snapshot = {};
    if (llama_process_functional_snapshot_archive_get(ctx_pool, process_entry.slot, &imported_process_archive) != 0 ||
        llama_process_functional_snapshot_info_get(ctx_pool, process_entry.slot, 0, &imported_process_snapshot) != 0 ||
        imported_process_archive.count != 1 ||
        !imported_process_snapshot.valid ||
        imported_process_snapshot.snapshot_id != process_snapshot.snapshot_id ||
        imported_process_snapshot.captured_at_us != process_snapshot.captured_at_us) {
        fprintf(stderr, "process-functional snapshot import did not restore metadata\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_functional_activation_decision process_decision = {};
    process_decision.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    process_decision.microphase = LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE;
    process_decision.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
    process_decision.top_family = LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    process_decision.eligible_mask = (1ull << LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL);
    process_decision.activated_mask = (1ull << LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL);
    process_decision.gains[LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL] = 1.0f;
    process_decision.predicted_gains[LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL] = 1.0f;
    process_decision.hold_unit[LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL] = LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS;
    process_decision.hold_value[LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL] = 1;

    llama_functional_activation_decision inactive = {};
    inactive.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    inactive.microphase = LLAMA_FUNCTIONAL_MICROPHASE_NONE;
    inactive.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
    inactive.top_family = -1;

    const int32_t baseline_stack_count = llama_serving_lora_stack_count(ctx);
    llama_functional_lora_replay_override process_replay = {};
    process_replay.active = true;
    process_replay.family = LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    process_replay.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED;
    process_replay.snapshot_slot = 0;
    process_replay.replay_gain = 1.0f;
    process_replay.disable_bootstrap = true;
    if (llama_process_functional_replay_override_begin(ctx, process_entry.slot, process_replay) != 0 ||
        !ctx->functional_lora_activate(process_decision) ||
        llama_serving_lora_stack_count(ctx) != baseline_stack_count + 1) {
        fprintf(stderr, "process-functional archived replay did not activate through the shared serving stack\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_process_functional_trace process_trace = {};
    if (llama_process_functional_get_last_trace(ctx, &process_trace) != 0 ||
        !process_trace.valid ||
        !process_trace.activation_attached ||
        !process_trace.matched_existing_entry ||
        process_trace.matched_entry_slot != process_entry.slot ||
        llama_process_functional_replay_override_end(ctx, process_entry.slot) != 0 ||
        !ctx->functional_lora_activate(inactive) ||
        llama_serving_lora_stack_count(ctx) != baseline_stack_count) {
        fprintf(stderr, "process-functional archived replay did not cleanly restore runtime state\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    process_replay = {};
    process_replay.active = true;
    process_replay.family = LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL;
    process_replay.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL;
    process_replay.replay_gain = 1.0f;
    process_replay.perturbation_scale = 0.05f;
    process_replay.cosine_limit = 0.10f;
    process_replay.disable_bootstrap = true;
    if (llama_process_functional_replay_override_begin(ctx, process_entry.slot, process_replay) != 0 ||
        !ctx->functional_lora_activate(process_decision) ||
        llama_process_functional_replay_override_end(ctx, process_entry.slot) != 0 ||
        !ctx->functional_lora_activate(inactive) ||
        llama_serving_lora_stack_count(ctx) != baseline_stack_count) {
        fprintf(stderr, "process-functional orthogonal replay did not complete through the shared lifecycle APIs\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_functional_lora_differential_update process_diff = {};
    if (llama_process_functional_apply_differential_update(
                ctx,
                process_entry.slot,
                LLAMA_FUNCTIONAL_BIAS_PROPOSAL_HISTORICAL,
                LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL,
                -1,
                0.35f,
                0.45f,
                0.80f) != 0 ||
        llama_process_functional_get_last_differential_update(ctx, process_entry.slot, &process_diff) != 0 ||
        !process_diff.valid ||
        process_diff.family != LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL ||
        process_diff.proposal_family != LLAMA_FUNCTIONAL_BIAS_PROPOSAL_HISTORICAL ||
        process_diff.source_snapshot_slot != -1 ||
        std::fabs(process_diff.signed_score_delta - 0.35f) > 1.0e-6f ||
        process_diff.magnitude <= 0.0f ||
        process_diff.lora_difference_norm <= 0.0f ||
        process_diff.applied_update_norm <= 0.0f ||
        process_diff.robustness_score <= 0.0f) {
        fprintf(stderr, "process-functional differential update metadata was not retained\n");
        llama_free(ctx_custom);
        llama_free(ctx_hash);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx_custom);
    llama_free(ctx_hash);
    llama_free(ctx_pool);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
