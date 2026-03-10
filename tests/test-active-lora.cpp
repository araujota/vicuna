#include "get-model.h"
#include "llama.h"

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

    if (!stats.enabled || stats.selected_rank < 1 ||
        stats.embedding_type != LLAMA_ACTIVE_LORA_EMBEDDING_HASH ||
        stats.embedding_is_custom || stats.embedding_dim != 64) {
        fprintf(stderr, "unexpected Active LoRA init stats\n");
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

    if (stats.updates_applied < 1 || stats.tokens_ingested < tokens.size() || !stats.rollover_ready ||
        stats.gain_mean <= 0.0f || stats.gain_max <= 0.0f || stats.gain_max > params.gain_max + 1.0e-6f) {
        fprintf(stderr, "unexpected Active LoRA ingest stats\n");
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

    llama_context * ctx_custom = llama_init_from_model(model, cparams);
    if (!ctx_custom) {
        fprintf(stderr, "failed to create callback context\n");
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params custom_params = llama_active_lora_default_params();
    custom_params.enabled = true;
    custom_params.min_rank = 1;
    custom_params.max_rank = 2;
    custom_params.embedding_dim = 16;
    custom_params.embedding_callback = test_callback_embedder;
    custom_params.embedding_type = LLAMA_ACTIVE_LORA_EMBEDDING_HASH;

    if (llama_active_lora_init(ctx_custom, custom_params) != 0) {
        fprintf(stderr, "failed to initialize callback Active LoRA\n");
        llama_free(ctx_custom);
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
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_active_lora_ingest(ctx_custom, tokens.data(), tokens.size()) != 0) {
        fprintf(stderr, "failed to ingest with callback Active LoRA\n");
        llama_free(ctx_custom);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_active_lora_get_stats(ctx_custom, &custom_stats) != 0 || custom_stats.updates_applied < 1) {
        fprintf(stderr, "unexpected callback Active LoRA ingest stats\n");
        llama_free(ctx_custom);
        llama_free(ctx_pool);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx_custom);
    llama_free(ctx_pool);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
