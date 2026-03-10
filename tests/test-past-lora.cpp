#include "get-model.h"
#include "llama.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize_or_die(const llama_vocab * vocab, const std::string & text) {
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize test prompt\n");
        std::exit(1);
    }
    return tokens;
}

static void configure_small_buckets(llama_past_lora_params & params) {
    params.enabled = true;

    const std::array<uint64_t, LLAMA_MEMORY_LORA_BUCKET_COUNT> periods = { 100, 200, 300, 400, 0 };
    const std::array<uint64_t, LLAMA_MEMORY_LORA_BUCKET_COUNT> half_lives = { 500, 600, 700, 800, 900 };
    const std::array<float, LLAMA_MEMORY_LORA_BUCKET_COUNT> base_scales = { 0.90f, 0.70f, 0.50f, 0.30f, 0.10f };

    for (int i = 0; i < LLAMA_MEMORY_LORA_BUCKET_COUNT; ++i) {
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
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params active = llama_active_lora_default_params();
    active.enabled = true;
    active.min_rank = 1;
    active.max_rank = 2;
    active.max_updates_before_rollover = 1;
    active.gain_max = 0.25f;

    if (llama_active_lora_init(ctx, active) != 0) {
        fprintf(stderr, "failed to initialize Active LoRA\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_past_lora_params past = llama_past_lora_default_params();
    configure_small_buckets(past);

    if (llama_past_lora_init(ctx, past) != 0) {
        fprintf(stderr, "failed to initialize past LoRA stack\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = tokenize_or_die(vocab, "A frozen memory test span that should push the active adapter over its rollover boundary.");

    if (llama_active_lora_ingest(ctx, tokens.data(), tokens.size()) != 0) {
        fprintf(stderr, "failed to ingest into Active LoRA\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_stats active_stats = {};
    if (llama_active_lora_get_stats(ctx, &active_stats) != 0 ||
        !active_stats.rollover_ready ||
        active_stats.gain_max <= 0.0f ||
        active_stats.gain_max > active.gain_max + 1.0e-6f) {
        fprintf(stderr, "unexpected Active LoRA state before past tick\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_past_lora_tick(ctx, 100) != 0) {
        fprintf(stderr, "failed to execute first past tick\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_past_lora_stats past_stats = {};
    if (llama_past_lora_get_stats(ctx, &past_stats) != 0 ||
        !past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK].populated ||
        past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK].version != 1 ||
        past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK].effective_scale <= 0.0f) {
        fprintf(stderr, "unexpected Past Week state after rollover tick\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_active_lora_get_stats(ctx, &active_stats) != 0 ||
        active_stats.rollover_ready ||
        active_stats.updates_applied != 0 ||
        active_stats.gain_max != 0.0f) {
        fprintf(stderr, "unexpected Active LoRA reset state after rollover tick\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const uint32_t week_version = past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK].version;
    if (llama_past_lora_tick(ctx, 150) != 0 || llama_past_lora_get_stats(ctx, &past_stats) != 0 ||
        past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK].version != week_version) {
        fprintf(stderr, "Past Week mutated without a due condensation job\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_past_lora_tick(ctx, 350) != 0 || llama_past_lora_get_stats(ctx, &past_stats) != 0 ||
        !past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH].populated) {
        fprintf(stderr, "Past Month did not populate on schedule\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_past_lora_tick(ctx, 700) != 0 || llama_past_lora_get_stats(ctx, &past_stats) != 0 ||
        !past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER].populated) {
        fprintf(stderr, "Past Quarter did not populate on schedule\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_past_lora_tick(ctx, 1100) != 0 || llama_past_lora_get_stats(ctx, &past_stats) != 0 ||
        !past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR].populated) {
        fprintf(stderr, "Past Year did not populate on schedule\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_past_lora_tick(ctx, 1600) != 0 || llama_past_lora_get_stats(ctx, &past_stats) != 0 ||
        !past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_ALL_TIME].populated) {
        fprintf(stderr, "All Time did not populate on schedule\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_ALL_TIME].effective_scale <= 0.0f ||
        past_stats.buckets[LLAMA_MEMORY_LORA_BUCKET_ALL_TIME].gain_max < 0.0f) {
        fprintf(stderr, "unexpected All Time decay or gain state\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
