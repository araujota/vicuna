#include "get-model.h"
#include "ggml.h"
#include "llama.h"
#include "llama-adapter.h"
#include "llama-context.h"
#include "llama-model.h"

#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

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
