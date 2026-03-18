#include "get-model.h"
#include "ggml.h"
#include "llama.h"
#include "llama-adapter.h"
#include "llama-context.h"
#include "llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize_or_die(const llama_vocab * vocab, const std::string & text) {
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize serving-lora prompt\n");
        std::exit(1);
    }
    return tokens;
}

static ggml_tensor * pick_request_target(const llama_model * model) {
    for (const auto & entry : model->tensors_by_name) {
        ggml_tensor * tensor = entry.second;
        if (tensor && tensor->ne[1] > 1) {
            return tensor;
        }
    }

    return nullptr;
}

static bool expect_layer(
        const llama_context * ctx,
        int32_t index,
        int32_t role,
        float min_scale = -1.0f) {
    llama_serving_lora_layer_info layer = {};
    if (llama_serving_lora_stack_layer(ctx, index, &layer) != 0) {
        fprintf(stderr, "failed to fetch serving layer %d\n", index);
        return false;
    }

    if (layer.role != role) {
        fprintf(stderr, "unexpected role at layer %d: got %d want %d\n", index, layer.role, role);
        return false;
    }

    if (min_scale >= 0.0f && layer.scale < min_scale) {
        fprintf(stderr, "unexpected scale at layer %d: got %f want >= %f\n", index, layer.scale, min_scale);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path, mparams),
            llama_model_free);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cparams),
            llama_free);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_import(
            llama_init_from_model(model.get(), cparams),
            llama_free);
    if (!ctx_import) {
        fprintf(stderr, "failed to create import context\n");
        return 1;
    }

    llama_active_lora_params active = llama_active_lora_default_params();
    active.enabled = true;
    active.host_memory_ratio = 1.0f;
    active.device_memory_ratio = 1.0f;
    active.min_rank = 1;
    active.max_rank = 2;

    if (llama_active_lora_init(ctx.get(), active) != 0) {
        fprintf(stderr, "failed to initialize Active LoRA\n");
        return 1;
    }
    if (llama_active_lora_init(ctx_import.get(), active) != 0) {
        fprintf(stderr, "failed to initialize Active LoRA for import context\n");
        return 1;
    }

    llama_past_lora_params past = llama_past_lora_default_params();
    past.enabled = true;
    for (int i = 0; i < LLAMA_MEMORY_LORA_BUCKET_COUNT; ++i) {
        past.host_memory_ratio[i] = 1.0f;
        past.device_memory_ratio[i] = 1.0f;
        past.min_rank[i] = 1;
        past.max_rank[i] = 2;
    }

    if (llama_past_lora_init(ctx.get(), past) != 0) {
        fprintf(stderr, "failed to initialize past LoRA stack\n");
        return 1;
    }
    if (llama_past_lora_init(ctx_import.get(), past) != 0) {
        fprintf(stderr, "failed to initialize past LoRA stack for import context\n");
        return 1;
    }

    if (llama_serving_lora_stack_count(ctx.get()) != 16 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 1, LLAMA_SERVING_LORA_LAYER_PAST_YEAR) ||
        !expect_layer(ctx.get(), 2, LLAMA_SERVING_LORA_LAYER_PAST_QUARTER) ||
        !expect_layer(ctx.get(), 3, LLAMA_SERVING_LORA_LAYER_PAST_MONTH) ||
        !expect_layer(ctx.get(), 4, LLAMA_SERVING_LORA_LAYER_PAST_WEEK) ||
        !expect_layer(ctx.get(), 5, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f) ||
        !expect_layer(ctx.get(), 6, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION, 0.0f) ||
        !expect_layer(ctx.get(), 7, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION, 0.0f) ||
        !expect_layer(ctx.get(), 8, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_PLANNING_COMPOSITION, 0.0f) ||
        !expect_layer(ctx.get(), 9, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_PLANNING_COMPOSITION, 0.0f) ||
        !expect_layer(ctx.get(), 10, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_COUNTERFACTUAL, 0.0f) ||
        !expect_layer(ctx.get(), 11, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_COUNTERFACTUAL, 0.0f) ||
        !expect_layer(ctx.get(), 12, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_MEMORY_COMPRESS, 0.0f) ||
        !expect_layer(ctx.get(), 13, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_MEMORY_COMPRESS, 0.0f) ||
        !expect_layer(ctx.get(), 14, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_SELF_OBSERVE, 0.0f) ||
        !expect_layer(ctx.get(), 15, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_SELF_OBSERVE, 0.0f)) {
        fprintf(stderr, "unexpected runtime-only serving stack\n");
        return 1;
    }
    const int32_t baseline_stack_count = llama_serving_lora_stack_count(ctx.get());

    const uint64_t one_week_us = 7ull * 24ull * 60ull * 60ull * 1000000ull;
    if (llama_functional_lora_snapshot_maintain(ctx.get(), 1) != 0 ||
        llama_functional_lora_snapshot_maintain(ctx.get(), one_week_us + 1) != 0) {
        fprintf(stderr, "failed to seed functional snapshot archive\n");
        return 1;
    }

    llama_functional_lora_snapshot_archive archive = {};
    if (llama_functional_lora_snapshot_archive_get(ctx.get(), LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION, &archive) != 0 ||
        archive.count != 1 ||
        !archive.items[0].valid) {
        fprintf(stderr, "failed to create archived functional replay input\n");
        return 1;
    }

    llama_functional_activation_decision decision = {};
    decision.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    decision.microphase = LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE;
    decision.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
    decision.top_family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    decision.eligible_mask = 1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    decision.activated_mask = 1ull << LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    decision.gains[LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION] = 1.0f;
    decision.predicted_gains[LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION] = 1.0f;
    decision.hold_unit[LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION] = LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS;
    decision.hold_value[LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION] = 1;
    llama_functional_activation_decision inactive = {};
    inactive.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    inactive.family_count = LLAMA_FUNCTIONAL_LORA_COUNT;
    inactive.top_family = -1;

    llama_functional_lora_replay_override replay = {};
    replay.active = true;
    replay.family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    replay.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED;
    replay.snapshot_slot = 0;
    replay.replay_gain = 1.0f;
    replay.disable_bootstrap = true;
    if (llama_functional_lora_replay_override_begin(ctx.get(), replay) != 0 ||
        !ctx->functional_lora_activate(decision) ||
        llama_serving_lora_stack_count(ctx.get()) > baseline_stack_count ||
        !expect_layer(ctx.get(), 6, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION, 0.0f) ||
        llama_functional_lora_replay_override_end(ctx.get(), LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION) != 0 ||
        !ctx->functional_lora_activate(inactive) ||
        llama_serving_lora_stack_count(ctx.get()) != baseline_stack_count) {
        fprintf(stderr, "archived functional replay changed serving stack cardinality\n");
        return 1;
    }

    replay = {};
    replay.active = true;
    replay.family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    replay.replay_mode = LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL;
    replay.snapshot_slot = -1;
    replay.replay_gain = 1.0f;
    replay.perturbation_scale = 0.08f;
    replay.cosine_limit = 0.10f;
    replay.disable_bootstrap = true;
    if (llama_functional_lora_replay_override_begin(ctx.get(), replay) != 0 ||
        !ctx->functional_lora_activate(decision) ||
        llama_serving_lora_stack_count(ctx.get()) > baseline_stack_count ||
        !expect_layer(ctx.get(), 6, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION, 0.0f) ||
        llama_functional_lora_replay_override_end(ctx.get(), LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION) != 0 ||
        !ctx->functional_lora_activate(inactive) ||
        llama_serving_lora_stack_count(ctx.get()) != baseline_stack_count) {
        fprintf(stderr, "orthogonal functional replay changed serving stack cardinality\n");
        return 1;
    }

    llama_process_functional_signature process_signature = {};
    process_signature.valid = true;
    process_signature.signature_hash = 0x1234ULL;
    process_signature.scope_kind = LLAMA_PROCESS_FUNCTIONAL_SCOPE_PROCESS_STEP;
    process_signature.family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    process_signature.loop_origin = LLAMA_COG_COMMAND_ORIGIN_DMN;
    process_signature.microphase = LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE;
    process_signature.plan_mode = LLAMA_COG_PLAN_MODE_COMPOSITION;
    process_signature.plan_step_kind = LLAMA_COG_PLAN_STEP_OBSERVE_TOOL;
    process_signature.tool_kind = LLAMA_TOOL_KIND_BASH_CLI;
    process_signature.source_family = LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION;
    process_signature.requires_tool_result = true;
    process_signature.transient_plan_id = 7;
    process_signature.transient_step_index = 1;
    process_signature.transient_source_id = 11;
    std::snprintf(process_signature.tool_name, sizeof(process_signature.tool_name), "%s", "bash-cli");
    std::snprintf(process_signature.semantic_key, sizeof(process_signature.semantic_key), "%s", "dmn/counterfactual_compare/step_kind:4/tool:bash-cli");
    if (!ctx->process_functional_set_execution(process_signature)) {
        fprintf(stderr, "failed to seed process-functional execution signature\n");
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
    const llama_functional_outcome_snapshot before = {
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
    const llama_functional_outcome_snapshot after = before;
    const float metrics[] = { -0.2f, 0.0f, 0.0f };
    for (int i = 0; i < 6; ++i) {
        if (!ctx->functional_lora_apply_update(
                    LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION,
                    LLAMA_COG_COMMAND_ORIGIN_DMN,
                    LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_GENERATE,
                    LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE,
                    before,
                    after,
                    LLAMA_TOOL_KIND_BASH_CLI,
                    1,
                    metrics,
                    sizeof(metrics)/sizeof(metrics[0]),
                    -0.2f,
                    0.6f,
                    process_event,
                    nullptr)) {
            fprintf(stderr, "failed to accumulate process-functional update evidence\n");
            return 1;
        }
    }

    llama_process_functional_entry_info process_entry = {};
    if (llama_process_functional_entry_count(ctx.get()) < 1 ||
        llama_process_functional_entry_get(ctx.get(), 0, &process_entry) != 0 ||
        !process_entry.valid ||
        process_entry.signature.signature_hash != process_signature.signature_hash) {
        fprintf(stderr, "process-functional entry was not created after repeated weak outcomes\n");
        return 1;
    }

    if (llama_process_functional_snapshot_maintain(ctx.get(), 1) != 0 ||
        llama_process_functional_snapshot_maintain(ctx.get(), one_week_us + 1) != 0) {
        fprintf(stderr, "failed to run process-functional snapshot maintenance\n");
        return 1;
    }

    llama_functional_snapshot_maintenance_trace process_maintenance = {};
    if (llama_process_functional_get_last_snapshot_maintenance(ctx.get(), &process_maintenance) != 0 ||
        !process_maintenance.ran ||
        !process_maintenance.captured_any ||
        process_maintenance.captured_count != 1) {
        fprintf(stderr, "unexpected process-functional snapshot maintenance trace\n");
        return 1;
    }

    llama_functional_lora_snapshot_archive process_archive = {};
    llama_functional_lora_snapshot_info process_snapshot = {};
    if (llama_process_functional_snapshot_archive_get(ctx.get(), process_entry.slot, &process_archive) != 0 ||
        llama_process_functional_snapshot_info_get(ctx.get(), process_entry.slot, 0, &process_snapshot) != 0 ||
        process_archive.count != 1 ||
        process_archive.family != LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION ||
        process_archive.last_capture_us != one_week_us + 1 ||
        process_archive.next_capture_due_us <= process_archive.last_capture_us ||
        !process_snapshot.valid) {
        fprintf(stderr, "unexpected process-functional snapshot archive contents\n");
        return 1;
    }

    const size_t process_entry_blob_size = llama_process_functional_entry_blob_size(ctx.get(), process_entry.slot);
    const size_t process_snapshot_blob_size = llama_process_functional_snapshot_blob_size(ctx.get(), process_entry.slot, 0);
    std::vector<uint8_t> process_entry_blob(process_entry_blob_size);
    std::vector<uint8_t> process_snapshot_blob(process_snapshot_blob_size);
    if (process_entry_blob_size == 0 ||
        process_snapshot_blob_size == 0 ||
        llama_process_functional_entry_blob_export(ctx.get(), process_entry.slot, process_entry_blob.data(), process_entry_blob.size()) != 0 ||
        llama_process_functional_snapshot_blob_export(ctx.get(), process_entry.slot, 0, process_snapshot_blob.data(), process_snapshot_blob.size()) != 0 ||
        llama_process_functional_entry_blob_import(ctx_import.get(), process_entry.slot, &process_entry, process_entry_blob.data(), process_entry_blob.size()) != 0 ||
        llama_process_functional_snapshot_blob_import(ctx_import.get(), process_entry.slot, 0, process_snapshot, process_snapshot_blob.data(), process_snapshot_blob.size()) != 0) {
        fprintf(stderr, "failed to roundtrip process-functional snapshot archive blobs\n");
        return 1;
    }

    llama_functional_lora_snapshot_archive imported_process_archive = {};
    llama_functional_lora_snapshot_info imported_process_snapshot = {};
    if (llama_process_functional_snapshot_archive_get(ctx_import.get(), process_entry.slot, &imported_process_archive) != 0 ||
        llama_process_functional_snapshot_info_get(ctx_import.get(), process_entry.slot, 0, &imported_process_snapshot) != 0 ||
        imported_process_archive.count != 1 ||
        !imported_process_snapshot.valid ||
        imported_process_snapshot.snapshot_id != process_snapshot.snapshot_id ||
        imported_process_snapshot.captured_at_us != process_snapshot.captured_at_us) {
        fprintf(stderr, "process-functional snapshot import did not restore metadata\n");
        return 1;
    }

    if (!ctx->process_functional_set_execution(process_signature) ||
        !ctx->functional_lora_activate(decision) ||
        llama_serving_lora_stack_count(ctx.get()) != baseline_stack_count + 2 ||
        !expect_layer(ctx.get(), baseline_stack_count, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_PROCESS_LEARNED, 0.0f) ||
        !expect_layer(ctx.get(), baseline_stack_count + 1, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_PROCESS_BOOTSTRAP)) {
        fprintf(stderr, "process-functional adapters were not attached for the matching process\n");
        return 1;
    }

    llama_process_functional_trace process_trace = {};
    if (llama_process_functional_get_last_trace(ctx.get(), &process_trace) != 0 ||
        !process_trace.valid ||
        !process_trace.activation_attached ||
        !process_trace.matched_existing_entry) {
        fprintf(stderr, "process-functional trace did not expose matching activation\n");
        return 1;
    }

    if (!ctx->process_functional_set_execution(llama_process_functional_signature {}) ||
        !ctx->functional_lora_activate(inactive) ||
        llama_serving_lora_stack_count(ctx.get()) != baseline_stack_count) {
        fprintf(stderr, "clearing process-functional execution did not remove process layers\n");
        return 1;
    }

    ggml_tensor * request_target = pick_request_target(model.get());
    if (!request_target) {
        fprintf(stderr, "failed to find a request LoRA target tensor\n");
        return 1;
    }

    auto request_adapter = std::make_unique<llama_adapter_lora>();
    if (!llama_adapter_lora_init_runtime(*model, *request_adapter, {{ request_target, 1 }}, "test.request")) {
        fprintf(stderr, "failed to initialize request runtime adapter\n");
        return 1;
    }

    llama_adapter_lora * request_adapters[] = { request_adapter.get() };
    float request_scales[] = { 0.5f };
    if (llama_set_adapters_lora(ctx.get(), request_adapters, 1, request_scales) != 0) {
        fprintf(stderr, "failed to set request adapters\n");
        model->loras.erase(request_adapter.get());
        return 1;
    }

    if (llama_serving_lora_stack_count(ctx.get()) != 17 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_REQUEST, 0.5f) ||
        !expect_layer(ctx.get(), 1, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 5, LLAMA_SERVING_LORA_LAYER_PAST_WEEK) ||
        !expect_layer(ctx.get(), 6, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f) ||
        !expect_layer(ctx.get(), 16, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_SELF_OBSERVE, 0.0f)) {
        fprintf(stderr, "request adapter replaced runtime memory layers\n");
        model->loras.erase(request_adapter.get());
        return 1;
    }

    request_scales[0] = 0.25f;
    if (llama_set_adapters_lora(ctx.get(), request_adapters, 1, request_scales) != 0 ||
        llama_serving_lora_stack_count(ctx.get()) != 17 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_REQUEST, 0.25f) ||
        !expect_layer(ctx.get(), 1, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 5, LLAMA_SERVING_LORA_LAYER_PAST_WEEK) ||
        !expect_layer(ctx.get(), 6, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f) ||
        !expect_layer(ctx.get(), 16, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_SELF_OBSERVE, 0.0f)) {
        fprintf(stderr, "request adapter update broke serving stack preservation\n");
        model->loras.erase(request_adapter.get());
        return 1;
    }

    request_scales[0] = 0.0f;
    if (llama_set_adapters_lora(ctx.get(), request_adapters, 1, request_scales) != 0 ||
        llama_serving_lora_stack_count(ctx.get()) != 16 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 5, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f) ||
        !expect_layer(ctx.get(), 15, LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_SELF_OBSERVE, 0.0f)) {
        fprintf(stderr, "clearing request adapter removed runtime memory layers\n");
        model->loras.erase(request_adapter.get());
        return 1;
    }

    model->loras.erase(request_adapter.get());
    return 0;
}
