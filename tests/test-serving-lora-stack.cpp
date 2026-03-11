#include "get-model.h"
#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"

#include <cstdio>
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

    if (llama_serving_lora_stack_count(ctx.get()) != 6 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 1, LLAMA_SERVING_LORA_LAYER_PAST_YEAR) ||
        !expect_layer(ctx.get(), 2, LLAMA_SERVING_LORA_LAYER_PAST_QUARTER) ||
        !expect_layer(ctx.get(), 3, LLAMA_SERVING_LORA_LAYER_PAST_MONTH) ||
        !expect_layer(ctx.get(), 4, LLAMA_SERVING_LORA_LAYER_PAST_WEEK) ||
        !expect_layer(ctx.get(), 5, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f)) {
        fprintf(stderr, "unexpected runtime-only serving stack\n");
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

    if (llama_serving_lora_stack_count(ctx.get()) != 7 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_REQUEST, 0.5f) ||
        !expect_layer(ctx.get(), 1, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 6, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f)) {
        fprintf(stderr, "request adapter replaced runtime memory layers\n");
        model->loras.erase(request_adapter.get());
        return 1;
    }

    request_scales[0] = 0.25f;
    if (llama_set_adapters_lora(ctx.get(), request_adapters, 1, request_scales) != 0 ||
        llama_serving_lora_stack_count(ctx.get()) != 7 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_REQUEST, 0.25f) ||
        !expect_layer(ctx.get(), 1, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 6, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f)) {
        fprintf(stderr, "request adapter update broke serving stack preservation\n");
        model->loras.erase(request_adapter.get());
        return 1;
    }

    request_scales[0] = 0.0f;
    if (llama_set_adapters_lora(ctx.get(), request_adapters, 1, request_scales) != 0 ||
        llama_serving_lora_stack_count(ctx.get()) != 6 ||
        !expect_layer(ctx.get(), 0, LLAMA_SERVING_LORA_LAYER_ALL_TIME) ||
        !expect_layer(ctx.get(), 5, LLAMA_SERVING_LORA_LAYER_ACTIVE, 0.0f)) {
        fprintf(stderr, "clearing request adapter removed runtime memory layers\n");
        model->loras.erase(request_adapter.get());
        return 1;
    }

    model->loras.erase(request_adapter.get());
    return 0;
}
